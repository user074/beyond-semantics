from __future__ import annotations
import math, contextlib
from typing import Tuple, Optional, Dict
import torch

# -----------------------------
# Small numerics
# -----------------------------
def _softmax_lastdim(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    x = x - x.max(dim=-1, keepdim=True).values
    ex = torch.exp(x)
    return ex / ex.sum(dim=-1, keepdim=True).clamp_min(eps)

# -----------------------------
# RoPE rotations
# -----------------------------
def _rope_relative_rotate_keys(k: torch.Tensor, delta_steps: float, inv_freq: torch.Tensor):
    """
    Relative RoPE rotation (+delta_steps) to keys (uniform Δ for all tokens).
    k: (..., D) even D; inv_freq: (D/2,)
    """
    D = k.shape[-1]; assert D % 2 == 0
    half = D // 2
    inv = inv_freq.to(k.device, k.dtype)[:half]
    ang = delta_steps * inv
    cos, sin = torch.cos(ang), torch.sin(ang)
    while cos.dim() < k.dim():
        cos = cos.unsqueeze(0); sin = sin.unsqueeze(0)
    k0, k1 = k[..., :half], k[..., half:]
    k_rot0 = k0 * cos - k1 * sin
    k_rot1 = k0 * sin + k1 * cos
    return torch.cat([k_rot0, k_rot1], dim=-1)

# -----------------------------
# Capture q,k after RoPE
# -----------------------------
@contextlib.contextmanager
def _capture_rope_qk(model):
    """
    Patch HF LLaMA apply_rotary_pos_emb to capture (q,k) after RoPE.
    Returns cap['q'][layer], cap['k'][layer] as (B,H,T,D).
    """
    from transformers.models.llama import modeling_llama as llama_mod
    original_apply = llama_mod.apply_rotary_pos_emb
    cap = {"q": {}, "k": {}, "layer": None}

    # tag layer via hooks
    attn_layers = []
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        for li, layer in enumerate(model.model.layers):
            if hasattr(layer, "self_attn"):
                attn_layers.append((li, layer.self_attn))

    def _pre(li):
        def f(*args, **kwargs): cap["layer"] = li
        return f
    def _post(li):
        def f(*args, **kwargs): cap["layer"] = None
        return f

    pre_handles, post_handles = [], []
    for li, attn in attn_layers:
        pre_handles.append(attn.register_forward_pre_hook(_pre(li)))
        post_handles.append(attn.register_forward_hook(_post(li)))

    def patched_apply_rotary_pos_emb(q, k, cos, sin, position_ids):
        q2, k2 = original_apply(q, k, cos, sin, position_ids)
        li = cap["layer"]
        if li is not None:
            def to_BHTD(x):
                return x.transpose(1,2).contiguous() if (x.dim()==4 and x.shape[1] > x.shape[2]) else x
            cap["q"][li] = to_BHTD(q2.detach())
            cap["k"][li] = to_BHTD(k2.detach())
        return q2, k2

    llama_mod.apply_rotary_pos_emb = patched_apply_rotary_pos_emb
    try:
        yield cap
    finally:
        llama_mod.apply_rotary_pos_emb = original_apply
        for h in pre_handles + post_handles: h.remove()

def _get_inv_freq(model) -> torch.Tensor:
    # standard LLaMA path
    try:
        return model.model.layers[0].self_attn.rotary_emb.inv_freq.detach().clone()
    except Exception:
        for path in [
            "model.model.layers.0.self_attn.rope.inv_freq",
            "model.model.layers.0.self_attn.rotary_emb.inv_freq",
        ]:
            cur = model
            ok = True
            for name in path.split("."):
                if not hasattr(cur, name): ok=False; break
                cur = getattr(cur, name)
            if ok:
                return cur.detach().clone()
        raise AttributeError("Could not find RoPE inv_freq on this model.")

# -----------------------------
# Core diagnostics (LLM-only)
# -----------------------------
def _compute_uniform_delta_probe(
    cap: dict,
    vision_range: Tuple[int,int],
    query_row_idx: int,
    inv_freq: torch.Tensor,
    delta_steps: float = 1.0,
    restrict_to_past: bool = True,
    eps: float = 1e-12,
    text_ranges_masked=None,
):
    """
    Uniform Δ probe: rotate ONLY vision keys by Δ, measure
      alphaV(L,H), share_prod_notV(L,H)=αV(1-αV),
      delta_gv(L,H)  (intrinsic group-avg vision logit shift per Δ),
      delta_alphaV(L,H) (measured change of vision mass on FULL).
    """
    layers = sorted(cap["q"].keys()); assert layers, "No layers captured."
    L = max(layers) + 1
    H = cap["q"][layers[0]].shape[1]

    alphaV    = torch.full((L, H), float("nan"))
    share_prod_notV = torch.full((L, H), float("nan"))
    delta_gv  = torch.full((L, H), float("nan"))
    delta_alphaV = torch.full((L, H), float("nan"))

    v_s, v_e = vision_range

    for li in layers:
        q = cap["q"][li]; k = cap["k"][li]                 # (B,H,T,D) post-RoPE
        B,Hh,T,D = q.shape; assert B==1 and Hh==H
        t = int(query_row_idx); assert 0 <= t < T

        q_last = q[0, :, t, :].float()                     # (H,D)
        keys   = (k[0, :, :t, :].float() if restrict_to_past else k[0].float())  # (H,K,D)
        K = keys.shape[1]

        a = max(0, min(K, v_s)); b = max(0, min(K, v_e))
        vis_mask = torch.zeros(K, dtype=torch.bool, device=keys.device)
        txt_mask = torch.zeros(K, dtype=torch.bool, device=keys.device)
        if a < b: vis_mask[a:b] = True
        if text_ranges_masked:
            for ta, tb in text_ranges_masked:
                ta2, tb2 = max(0, min(K, ta)), max(0, min(K, tb))
                if ta2 < tb2: txt_mask[ta2:tb2] = True
        txt_mask[0] = False  # drop BOS if lands in text

        # baseline logits/attn on FULL
        logits0 = torch.einsum("hd,hkd->hk", q_last, keys) / math.sqrt(D)   # (H,K)
        attn0   = _softmax_lastdim(logits0, eps)                             # (H,K)

        # α_V and α_V(1−α_V) on FULL
        alphaV_h = attn0[:, vis_mask].sum(dim=-1)                            # (H,)
        alphaV[li] = alphaV_h.cpu()
        share_prod_notV[li] = (alphaV_h * (1.0 - alphaV_h)).cpu()

        # rotate ONLY vision keys by Δ
        keys_p = keys.clone()
        if a < b:
            keys_p[:, a:b, :] = _rope_relative_rotate_keys(keys[:, a:b, :], delta_steps, inv_freq)

        logits1 = torch.einsum("hd,hkd->hk", q_last, keys_p) / math.sqrt(D)
        attn1   = _softmax_lastdim(logits1, eps)

        # Δα_V on FULL
        delta_alphaV[li] = (attn1[:, vis_mask].sum(dim=-1) - attn0[:, vis_mask].sum(dim=-1)).cpu()

        # Δg_V per head (within-head α_v/α_V weights over V)
        if a < b:
            alpha_v = attn0[:, a:b]                                        # (H, |V|)
            denomV  = alpha_v.sum(dim=-1, keepdim=True).clamp_min(eps)     # (H,1)
            w_v     = alpha_v / denomV
            dlogits_v = (logits1[:, a:b] - logits0[:, a:b]) / max(delta_steps, eps)
            delta_gv[li] = (w_v * dlogits_v).sum(dim=-1).cpu()             # (H,)
        else:
            delta_gv[li] = torch.zeros(H).cpu()

    return (
        alphaV.numpy(),
        share_prod_notV.numpy(),
        delta_gv.numpy(),
        delta_alphaV.numpy(),
    )


# -----------------------------
# Public driver
# -----------------------------
def compute_positional_sensitivity(
    model,                                   # LlavaLlamaForCausalLM (or compatible)
    prepared,                                # your PreparedInputs from vlm_diag
    delta_steps: float = 1.0,
    restrict_to_past: bool = True,
    eps: float = 1e-12,
):
    """
    One-call positional diagnostics:
      - Uniform Δ probe: rotate ONLY vision keys by Δ
    Returns a dict of numpy arrays keyed by names below.
    """
    device = next(model.parameters()).device
    eos_id = getattr(model, "eos_token_id", None)
    if eos_id is None and hasattr(model, "get_input_embeddings"):
        eos_id = 2  # common for LLaMA; adjust if needed
    input_ids_cap = torch.cat([prepared.input_ids, torch.tensor([[eos_id]], device=device)], dim=1)
    query_row_idx = int(prepared.output_token_start)
    inv_freq = _get_inv_freq(model)

    # run a forward to capture post-RoPE q,k
    model.eval()
    with torch.inference_mode(), _capture_rope_qk(model) as cap:
        if prepared.image_tensor is None:
            _ = model(input_ids=input_ids_cap, use_cache=False, output_attentions=False, return_dict=True)
        else:
            _ = model(input_ids=input_ids_cap, images=prepared.image_tensor, image_sizes=[prepared.image_size],
                      use_cache=False, output_attentions=False, return_dict=True)

    # UNIFORM-Δ (vision only)
    alphaV, share_prod_notV, delta_gv, delta_alphaV = _compute_uniform_delta_probe(
        cap,
        vision_range=prepared.vision,
        query_row_idx=query_row_idx,
        inv_freq=inv_freq,
        delta_steps=delta_steps,
        restrict_to_past=restrict_to_past,
        eps=eps,
        text_ranges_masked=prepared.text_ranges_masked,
    )

    out = {
        "alphaV": alphaV,                               # (L,H)
        "share_prod_notV": share_prod_notV,             # (L,H) = α_V (1-α_V)
        "delta_gv": delta_gv,                           # (L,H)
        "delta_alphaV": delta_alphaV,                   # (L,H)
    }
    return out
