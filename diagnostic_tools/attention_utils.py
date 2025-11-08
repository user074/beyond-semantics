from __future__ import annotations
import math, contextlib
from typing import Tuple, Optional, Dict
import torch

import os
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import os
import sys
sys.path.append("../")
sys.path.append("../VLM-Visualizer")

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn.functional as F

# --- LLaVA / project imports ---
from LLaVA.llava.constants import (
    IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
)
from LLaVA.llava.conversation import conv_templates
from LLaVA.llava.model.builder import load_pretrained_model
from LLaVA.llava.utils import disable_torch_init
from LLaVA.llava.mm_utils import get_model_name_from_path
from LLaVA.llava.model.language_model.llava_llama import LlavaLlamaForCausalLM

#VLM-Visualizer local utils
from utils import (
    load_image,
    aggregate_llm_attention,
    heterogenous_stack,
    show_mask_on_image,  
)

# -----------------------------
# Small helpers & data classes
# -----------------------------
SYSTEM_PROMPT_LLaVA15 = (
    "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
)

def is_pure_language_model(model_name: str) -> bool:
    """Check if the model is a pure language model (like Vicuna) that doesn't use vision."""
    return "vicuna" in model_name.lower()

@dataclass
class ModelBundle:
    tokenizer: any
    model: LlavaLlamaForCausalLM
    image_processor: any
    context_len: int
    model_name: str

    @classmethod
    def load(
        cls,
        model_path: str,
        device: str = "cuda",
        load_8bit: bool = False,
        load_4bit: bool = False,
        model_base: Optional[str] = None,
    ) -> "ModelBundle":
        disable_torch_init()
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path,
            model_base,
            model_name,
            load_8bit,
            load_4bit,
            device=device,
        )
        return cls(tokenizer, model, image_processor, context_len, model_name)


@dataclass
class PreparedInputs:
    # Immutable inputs that can be reused for multiple calls to generation/diagnostics.
    prompt: str
    input_ids: torch.Tensor  # shape (1, T_in)
    image_tensor: torch.Tensor | List[torch.Tensor]
    image_size: Tuple[int, int]  # PIL size (W, H)
    # Ranges in the unified attention-matrix index space
    text_before: Tuple[int, int]
    vision: Tuple[int, int]
    text_after: Tuple[int, int]
    text_ranges_masked: List[Tuple[int, int]]  # system-prompt-masked text ranges
    output_token_start: int
    sys: Tuple[int, int]


@dataclass
class DiagnosticResult:
    response_text: str
    token_labels: List[str]
    attn_to_vis_share: List[float]
    attn_to_text_share: List[float]
    attn_to_sys_share: List[float]
    attn_to_vis_withsys_share: List[float]
    attn_to_text_withsys_share: List[float]
    pref_score: List[float]
    mean_vision_share: float
    mean_text_share: float
    mean_sys_share: float
    mean_vision_withsys_share: float
    mean_text_withsys_share: float
    mean_pref_score: float
    head_vision_share: Optional[np.ndarray] = None  # Shape: (layers, heads)
    plot_paths: Dict[str, str] = None  # Saved figure paths
    hidden_norms_layerwise: Optional[Dict[str, np.ndarray]] = None
    hidden_norms_paths: Optional[Dict[str, str]] = None


# -----------------------------
# Image helpers 
# -----------------------------
def expand2square(pil_img: Image.Image, background_color) -> Image.Image:
    w, h = pil_img.size
    if w == h:
        return pil_img
    if w > h:
        result = Image.new(pil_img.mode, (w, w), background_color)
        result.paste(pil_img, (0, (w - h) // 2))
        return result
    result = Image.new(pil_img.mode, (h, h), background_color)
    result.paste(pil_img, ((h - w) // 2, 0))
    return result


def prepare_images_for_llava(images: List[Image.Image], image_processor, model_cfg):
    """Pad to square when model_cfg.image_aspect_ratio == 'pad'. Returns (tensor, raw_images)."""
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_raw, new_pp = [], []
    if image_aspect_ratio == "pad":
        for im in images:
            im2 = expand2square(im, tuple(int(x * 255) for x in image_processor.image_mean))
            new_raw.append(im2)
            new_pp.append(image_processor.preprocess(im2, return_tensors="pt")["pixel_values"][0])
    elif image_aspect_ratio == "anyres":
        raise NotImplementedError("anyres not supported in this visualization util")
    else:
        raise NotImplementedError("Only 'pad' image_aspect_ratio is supported in this util")

    if all(x.shape == new_pp[0].shape for x in new_pp):
        stacked = torch.stack(new_pp, dim=0)
    else:
        stacked = new_pp  # fall back to a list if shapes differ
    return stacked, new_raw


# -----------------------------
# Prompt / tokenization helpers
# -----------------------------
def pick_conv_mode(model_name: str) -> str:
    ln = model_name.lower()
    if "llama-2" in ln:
        return "llava_llama_2"
    if "mistral" in ln:
        return "mistral_instruct"
    if "v1.6-34b" in ln:
        return "chatml_direct"
    if "v1" in ln:
        return "llava_v1"
    if "mpt" in ln:
        return "mpt"
    return "llava_v0"


def build_prompt(model_name: str, prompt_text: str, mm_use_im_start_end: bool, strip_system_prompt: bool) -> str:
    conv_mode = pick_conv_mode(model_name)
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles

    if mm_use_im_start_end:
        inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + prompt_text
    else:
        inp = DEFAULT_IMAGE_TOKEN + "\n" + prompt_text

    conv.append_message(roles[0], inp)
    conv.append_message(roles[1], None)
    prompt = conv.get_prompt()

    if strip_system_prompt:
        prompt = prompt.replace(SYSTEM_PROMPT_LLaVA15, "")
    return prompt


def build_prompt_text_only(model_name: str, prompt_text: str, strip_system_prompt: bool) -> str:
    """Build prompt for pure language models without image tokens."""
    conv_mode = pick_conv_mode(model_name)
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles

    # For text-only models, just use the prompt text directly
    conv.append_message(roles[0], prompt_text)
    conv.append_message(roles[1], None)
    prompt = conv.get_prompt()

    if strip_system_prompt:
        prompt = prompt.replace(SYSTEM_PROMPT_LLaVA15, "")
    return prompt


def tokenize_with_image_token(prompt: str, tokenizer, device: torch.device) -> torch.Tensor:
    ids = tokenizer(
        prompt, return_tensors="pt", add_special_tokens=False
    )["input_ids"]
    from LLaVA.llava.mm_utils import tokenizer_image_token
    return tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)


# -----------------------------
# Index layout helpers
# -----------------------------
def _compute_ranges(prompt: str, tokenizer, model, device: torch.device):
    parts = prompt.split(DEFAULT_IMAGE_TOKEN, 1)
    assert len(parts) == 2, "Prompt must contain the image placeholder token."
    text_before_raw, text_after_raw = parts[0], parts[1]

    before_ids = tokenizer(text_before_raw, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(device)
    after_ids = tokenizer(text_after_raw, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(device)

    n_before = int(before_ids.shape[0])
    n_after = int(after_ids.shape[0])
    P = int(model.get_vision_tower().num_patches)

    # Unified attention-matrix index layout:
    # [0]=BOS
    # [1 .. 1+n_before)                     = text_before
    # [1+n_before .. 1+n_before+P)          = vision
    # [1+n_before+P .. 1+n_before+P+n_after)= text_after
    text_before = (1, 1 + n_before)
    vision = (text_before[1], text_before[1] + P)
    text_after = (vision[1], vision[1] + n_after)

    return dict(
        text_before=text_before,
        vision=vision,
        text_after=text_after,
        text_before_raw=text_before_raw,
        text_after_raw=text_after_raw,
        P=P,
    )


def _compute_ranges_text_only(prompt: str, tokenizer, device: torch.device):
    """Compute ranges for text-only models (no vision component)."""
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(device)
    n_total = int(prompt_ids.shape[0])
    
    # For text-only models:
    # [0]=BOS
    # [1 .. 1+n_total) = all text (no vision component)
    text_before = (1, 1 + n_total)
    vision = (1 + n_total, 1 + n_total)  # empty vision range
    text_after = (1 + n_total, 1 + n_total)  # empty text_after range
    
    return dict(
        text_before=text_before,
        vision=vision,
        text_after=text_after,
        text_before_raw=prompt,  # entire prompt is "before" text
        text_after_raw="",  # no "after" text
        P=0,  # no vision patches
    )


def _find_subspan_ids(text_whole: str, sub_text: str, tokenizer, device: torch.device) -> Optional[Tuple[int, int]]:
    if not sub_text:
        return None
    whole = tokenizer(text_whole, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(device)
    sub = tokenizer(sub_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(device)
    if len(sub) == 0 or len(sub) > len(whole):
        return None
    for i in range(0, len(whole) - len(sub) + 1):
        if torch.all(whole[i : i + len(sub)] == sub):
            return (i, i + len(sub))
    return None


def _subtract_span(ranges_list: List[Tuple[int, int]], span: Optional[Tuple[int, int]]):
    if span is None:
        return ranges_list
    s, e = span
    out = []
    for a, b in ranges_list:
        if b <= s or a >= e:
            out.append((a, b))
        else:
            if a < s:
                out.append((a, s))
            if e < b:
                out.append((e, b))
    return out


# -----------------------------
# Attention aggregation helpers
# -----------------------------
def _aggregate_prompt_attention(outputs) -> torch.Tensor:
    """Match your previous construction for 'aggregated_prompt_attention'."""
    aggregated = []
    for layer in outputs["attentions"][0]:
        layer_attns = layer.squeeze(0)  # (H, Q, K)
        attns_per_head = layer_attns.mean(dim=0)  # (Q, K)
        cur = attns_per_head[:-1].clone().cpu()
        cur[1:, 0] = 0.0
        cur[1:] = cur[1:] / cur[1:].sum(-1, keepdim=True)
        aggregated.append(cur)
    return torch.stack(aggregated).mean(dim=0)  # (Q-1, K)


def _build_llm_attn_matrix(aggregated_prompt_attention: torch.Tensor, outputs) -> torch.Tensor:
    # Concat: [bos_row] + prompt_agg + decode-step attentions aggregated
    return heterogenous_stack(
        [torch.tensor([1], dtype=aggregated_prompt_attention.dtype)]
        + list(aggregated_prompt_attention)
        + list(map(aggregate_llm_attention, outputs["attentions"]))
    )


def _sum_over_ranges(vec: torch.Tensor, ranges: List[Tuple[int, int]], row_idx: Optional[int] = None) -> float:
    total = 0.0
    for a, b in ranges:
        end = min(b, row_idx) if row_idx is not None else b
        if a < end:
            total += float(vec[a:end].sum().item())
    return total

# -----------------------------
# Hidden-state probing helpers
# -----------------------------
def _pick_3d_tensors(container):
    import torch
    return [x for x in container if isinstance(x, torch.Tensor) and x.ndim == 3]

@torch.inference_mode()
def _norms_layers_by_positions_any(hidden_states, batch_idx: int = 0, kind: str = "l2", tstep: int = 0):
    """
    Returns a tensor of shape (L, S) with per-layer token norms for one batch item.
    Works with:
      A) generate(): hidden_states[time_step][layer] -> (B,S,D)
      B) forward():  hidden_states[layer] -> (B,S,D)
    For generate(), use tstep=0 to get the full prompt context (the largest S).
    """
    import torch

    # generate()-style: tuple over time-steps, each time-step is a tuple over layers
    if isinstance(hidden_states, (tuple, list)) and hidden_states and isinstance(hidden_states[0], (tuple, list)):
        layers = _pick_3d_tensors(hidden_states[tstep])
    else:
        # forward()-style: tuple over layers
        layers = _pick_3d_tensors(hidden_states)

    rows = []
    with torch.no_grad():
        for t in layers:
            x = t[batch_idx]  # (S, D)
            if x.dtype == torch.bfloat16:
                x = x.float()
            
            if kind == "rms":
                n = x.pow(2).mean(dim=-1).sqrt()     # (S,)
            else:
                n = torch.linalg.vector_norm(x, ord=2, dim=-1)  # (S,)
            rows.append(n)
    return torch.stack(rows)  # (L, S)

def _mask_from_ranges(ranges, length: int):
    import numpy as np
    mask = np.zeros(length, dtype=bool)
    for a, b in ranges:
        a = max(0, min(a, length))
        b = max(0, min(b, length))
        if b > a: mask[a:b] = True
    return mask

def _compute_hidden_state_norm_curves(prepared: PreparedInputs, hidden_states, *, kind: str = "l2", tstep: int = 0):
    """
    Returns:
      curves: dict[str, np.ndarray] with keys in {"sys","user_text","vision"(optional)}
              each array is shape (L,) = per-layer mean norm over that token group.
      grid:   np.ndarray of shape (L, S_prompt) with raw norms (prompt tokens only).
    """
    import numpy as np
    torch_grid = _norms_layers_by_positions_any(hidden_states, batch_idx=0, kind=kind, tstep=tstep)
    # Only look at the prompt side (before the first generated token)
    S_prompt = prepared.output_token_start
    torch_grid = torch_grid[:, :S_prompt]             # (L, S_prompt)
    grid = torch_grid.detach().cpu().numpy()

    L, S = grid.shape
    # Build masks for the three groups
    sys_mask   = _mask_from_ranges([prepared.sys], S) if hasattr(prepared, "sys") and prepared.sys else np.zeros(S, bool)
    user_mask  = _mask_from_ranges(prepared.text_ranges_masked, S)
    vis_range  = prepared.vision
    vis_mask   = _mask_from_ranges([vis_range], S) if (vis_range[1] - vis_range[0]) > 0 else np.zeros(S, bool)

    def safe_mean(mask):
        if mask.sum() == 0:
            return np.full((L,), np.nan, dtype=grid.dtype)
        return grid[:, mask].mean(axis=1)

    curves = {
        "sys":       safe_mean(sys_mask),
        "user_text": safe_mean(user_mask),
    }
    if vis_mask.any():
        curves["vision"] = safe_mean(vis_mask)

    return curves, grid

# -----------------------------
# Hidden-state plotters / savers
# -----------------------------
def _plot_hidden_norms_layerwise(curves: Dict[str, np.ndarray], out_path: str, kind: str = "L2"):
    import numpy as np, matplotlib.pyplot as plt
    plt.figure(figsize=(8, 3), dpi=140)
    for label, arr in curves.items():
        if arr is None: 
            continue
        # drop all-NaN curves (e.g., missing sys/vision)
        if np.all(np.isnan(arr)): 
            continue
        plt.plot(arr, linewidth=2, label=label.replace("_", " "))
    plt.xlabel("Layer")
    plt.ylabel(f"{kind} norm (mean over group)")
    plt.title(f"Hidden-state {kind} norms by layer")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def _plot_hidden_norms_heatmap(grid: np.ndarray, prepared: PreparedInputs, out_path: str):
    """
    grid: (L, S_prompt) per-layer per-position norms.
    Adds lightweight range markers for [text_before | vision | text_after].
    """
    import matplotlib.pyplot as plt
    import numpy as np
    L, S = grid.shape
    tb_s, tb_e = prepared.text_before
    v_s, v_e   = prepared.vision
    ta_s, ta_e = prepared.text_after
    S_prompt   = prepared.output_token_start

    plt.figure(figsize=(10, 4), dpi=150)
    im = plt.imshow(grid, aspect="auto", interpolation="nearest")
    plt.colorbar(im, label="Norm")
    plt.xlabel("Prompt position")
    plt.ylabel("Layer")

    # vertical separators (clipped to prompt)
    for x in [tb_e, v_e]:
        if 0 < x < S_prompt:
            plt.axvline(x=x-0.5, color="white", linestyle="--", linewidth=1)
    plt.title("Hidden-state norms heatmap (prompt tokens only)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def _save_hidden_norms(curves: Dict[str, np.ndarray], grid: np.ndarray, save_dir: str, kind: str = "l2") -> Dict[str, str]:
    import os, numpy as np
    paths = {}
    os.makedirs(save_dir, exist_ok=True)

    # npz with all arrays
    npz_path = os.path.join(save_dir, f"hidden_state_norms_{kind}.npz")
    np.savez(npz_path, grid=grid, **curves)
    paths["npz"] = npz_path

    # csv for the layerwise curves (layer, sys, user_text, vision?)
    layer = np.arange(grid.shape[0])[:, None]
    cols  = [curves.get("sys"), curves.get("user_text"), curves.get("vision")]
    names = ["sys", "user_text", "vision"]
    # assemble table w/ NaNs for missing columns
    stack_cols = [layer]
    header = ["layer"]
    for name, col in zip(names, cols):
        if col is not None:
            stack_cols.append(col.reshape(-1, 1))
            header.append(name)
    table = np.concatenate(stack_cols, axis=1)
    csv_path = os.path.join(save_dir, f"hidden_state_norms_{kind}.csv")
    np.savetxt(csv_path, table, delimiter=",", header=",".join(header), comments="", fmt="%.6f")
    paths["csv"] = csv_path
    return paths

# -----------------------------
# Public API
# -----------------------------
def prepare_inputs(
    mb: ModelBundle,
    image_path: Optional[str] = None,
    image_pil: Optional[Image.Image] = None,
    prompt_text: str = "",
    strip_system_prompt: bool = False,
) -> PreparedInputs:
    """Load/preprocess image, build prompt+ids, and precompute index ranges (system-prompt-masked)."""
    device = next(mb.model.parameters()).device
    
    is_text_only = is_pure_language_model(mb.model_name)
    
    if is_text_only:
        image_tensor = None
        image_size = (0, 0)
    else:
        assert (image_path is not None) ^ (image_pil is not None), "Provide exactly one of {image_path, image_pil}"
        pil = image_pil if image_pil is not None else load_image(image_path)
        image_tensor, raw_images = prepare_images_for_llava([pil], mb.image_processor, mb.model.config)
        image = raw_images[0]
        image_size = image.size
        if isinstance(image_tensor, list):
            image_tensor = [x.to(device, dtype=torch.float16) for x in image_tensor]
        else:
            image_tensor = image_tensor.to(device, dtype=torch.float16)

    if is_text_only:
        prompt = build_prompt_text_only(mb.model_name, prompt_text, strip_system_prompt)
        input_ids = mb.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    else:
        prompt = build_prompt(mb.model_name, prompt_text, mb.model.config.mm_use_im_start_end, strip_system_prompt)
        input_ids = tokenize_with_image_token(prompt, mb.tokenizer, device)

    if is_text_only:
        R = _compute_ranges_text_only(prompt, mb.tokenizer, device)
        n_total = R["text_before"][1] - R["text_before"][0]
    else:
        R = _compute_ranges(prompt, mb.tokenizer, mb.model, device)
        n_total = None
    
    tb_s, tb_e = R["text_before"]
    v_s, v_e = R["vision"]
    ta_s, ta_e = R["text_after"]

    text_ranges = [(tb_s, tb_e), (ta_s, ta_e)]
    sys_local = _find_subspan_ids(R["text_before_raw"], SYSTEM_PROMPT_LLaVA15, mb.tokenizer, device)
    if sys_local is not None:
        sys_a = tb_s + sys_local[0]
        sys_b = tb_s + sys_local[1]
        text_ranges = _subtract_span(text_ranges, (sys_a, sys_b))
    else:
        sys_a = tb_s
        sys_b = tb_s

    if is_text_only:
        output_token_start = 1 + n_total
    else:
        output_token_start = ta_e

    return PreparedInputs(
        prompt=prompt,
        input_ids=input_ids,
        image_tensor=image_tensor,
        image_size=image_size,
        text_before=(tb_s, tb_e),
        vision=(v_s, v_e),
        text_after=(ta_s, ta_e),
        text_ranges_masked=text_ranges,
        output_token_start=output_token_start,
        sys = (sys_a, sys_b),
    )


@torch.inference_mode()
def diagnose_once(
    mb: ModelBundle,
    prepared: PreparedInputs,
    max_new_tokens: int =32,
    do_sample: bool = False,
    save_dir: Optional[str] = None,
    make_plots: bool = True,
    probe_hidden_norms: bool = True,
    hidden_norm_kind: str = "l2",
) -> DiagnosticResult:
    """Run generation, build attention shares, optionally save plots."""
    device = next(mb.model.parameters()).device
    is_text_only = is_pure_language_model(mb.model_name)

    if is_text_only:
        outputs = mb.model.generate(
            prepared.input_ids,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            return_dict_in_generate=True,
            output_attentions=True,
            output_hidden_states=probe_hidden_norms,
        )
    else:
        outputs = mb.model.generate(
            prepared.input_ids,
            images=prepared.image_tensor,
            image_sizes=[prepared.image_size],
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            return_dict_in_generate=True,
            output_attentions=True,
            output_hidden_states=probe_hidden_norms,
        )

    text = mb.tokenizer.decode(outputs["sequences"][0]).strip()

    aggregated_prompt_attention = _aggregate_prompt_attention(outputs)
    llm_attn_matrix = _build_llm_attn_matrix(aggregated_prompt_attention, outputs)

    n_before = prepared.text_before[1] - prepared.text_before[0]
    n_after = prepared.text_after[1] - prepared.text_after[0]
    P = prepared.vision[1] - prepared.vision[0]
    n_out = len(outputs["sequences"][0])
    
    if is_text_only:
        n_out = len(outputs["sequences"][0]) - len(prepared.input_ids[0])
        expected_len = 1 + (n_before - 1) + n_out
    else:
        expected_len = n_before + P + n_after + n_out
    
    assert llm_attn_matrix.shape[0] == expected_len, f"{llm_attn_matrix.shape[0]} != {expected_len}"

    attn_to_vis_share, attn_to_text_share, attn_to_sys_share = [], [], []
    attn_to_vis_withsys_share, attn_to_text_withsys_share = [], []
    pref_score, token_labels = [], []
    
    for row_idx in range(prepared.output_token_start, llm_attn_matrix.shape[0]):
        row = llm_attn_matrix[row_idx]
        past = row[:row_idx]  # causal

        if is_text_only:
            vis_sum = 0.0
            txt_sum = _sum_over_ranges(past, prepared.text_ranges_masked, row_idx=row_idx)
        else:
            v_s, v_e = prepared.vision
            vis_sum = float(past[v_s : min(v_e, row_idx)].sum().item()) if row_idx > v_s else 0.0
            txt_sum = _sum_over_ranges(past, prepared.text_ranges_masked, row_idx=row_idx)
        
        sys_sum = 0.0
        if hasattr(prepared, 'sys') and prepared.sys is not None:
            sys_a, sys_b = prepared.sys
            sys_sum = float(past[sys_a : min(sys_b, row_idx)].sum().item()) if row_idx > sys_a else 0.0
        
        denom_masked = (vis_sum + txt_sum) if (vis_sum + txt_sum) > 0 else 1.0
        attn_to_vis_share.append(vis_sum / denom_masked)
        attn_to_text_share.append(txt_sum / denom_masked)
        
        denom_with_sys = (vis_sum + txt_sum + sys_sum) if (vis_sum + txt_sum + sys_sum) > 0 else 1.0
        attn_to_sys_share.append(sys_sum / denom_with_sys)
        attn_to_vis_withsys_share.append(vis_sum / denom_with_sys)
        attn_to_text_withsys_share.append(txt_sum / denom_with_sys)
        
        pref_score.append((vis_sum - txt_sum) / denom_masked)

        gen_tok = int(outputs["sequences"][0][row_idx - prepared.output_token_start].item())
        token_labels.append(mb.tokenizer.decode([gen_tok], add_special_tokens=False).strip() or "▯")

    mean_v = float(np.mean(attn_to_vis_share)) if attn_to_vis_share else 0.0
    mean_t = float(np.mean(attn_to_text_share)) if attn_to_text_share else 0.0
    mean_p = float(np.mean(pref_score)) if pref_score else 0.0
    mean_sys = float(np.mean(attn_to_sys_share)) if attn_to_sys_share else 0.0
    mean_v_withsys = float(np.mean(attn_to_vis_withsys_share)) if attn_to_vis_withsys_share else 0.0
    mean_t_withsys = float(np.mean(attn_to_text_withsys_share)) if attn_to_text_withsys_share else 0.0
    
    if is_text_only:
        step_attns = outputs["attentions"]
        num_layers = len(step_attns[0])
        H = step_attns[0][0].shape[1]
        head_np = np.zeros((num_layers, H))
        head_sys_np = np.zeros((num_layers, H))
    else:
        head_arr = _compute_mean_vision_share_per_head_masked(
            outputs,
            text_ranges=prepared.text_ranges_masked,
            vision_range=prepared.vision,
        )
        head_np = head_arr.numpy()
        head_sys_arr = _compute_mean_vision_share_per_head_masked(
            outputs,
            text_ranges=[prepared.sys],
            vision_range=prepared.vision,
        )
        head_sys_np = head_sys_arr.numpy()

    hidden_norms_layerwise = None
    hidden_norms_paths = None
    if probe_hidden_norms and ("hidden_states" in outputs):
        try:
            curves, grid = _compute_hidden_state_norm_curves(
                prepared, outputs["hidden_states"], kind=hidden_norm_kind, tstep=0
            )
            hidden_norms_layerwise = curves

            if make_plots and save_dir:
                os.makedirs(save_dir, exist_ok=True)
                curve_path = os.path.join(save_dir, f"hidden_state_norms_{hidden_norm_kind}_layerwise.png")
                _plot_hidden_norms_layerwise(curves, out_path=curve_path, kind=hidden_norm_kind.upper())
                heatmap_path = os.path.join(save_dir, f"hidden_state_norms_{hidden_norm_kind}_heatmap.png")
                _plot_hidden_norms_heatmap(grid, prepared, out_path=heatmap_path)
                file_paths = _save_hidden_norms(curves, grid, save_dir, kind=hidden_norm_kind)

                plot_paths = {} if "plot_paths" not in locals() else plot_paths
                plot_paths["hidden_norms_layerwise"] = curve_path
                plot_paths["hidden_norms_heatmap"] = heatmap_path
                hidden_norms_paths = file_paths
        except Exception as e:
            print(f"[warn] hidden-state probing skipped: {e}")

    plot_paths = {}
    if make_plots and save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        if is_text_only:
            shares_fig = os.path.join(save_dir, "decode_time_text_shares_system_masked.png")
            _plot_text_shares(attn_to_text_share, attn_to_sys_share, out_path=shares_fig)
            plot_paths["decode_shares"] = shares_fig
            
            sys_fig = os.path.join(save_dir, "decode_time_system_prompt_attention.png")
            _plot_system_attention(attn_to_sys_share, out_path=sys_fig)
            plot_paths["system_attention"] = sys_fig
        else:
            totals_fig = os.path.join(save_dir, "decode_time_attention_totals_system_masked.png")
            _plot_totals(llm_attn_matrix, prepared, out_path=totals_fig)
            plot_paths["decode_totals"] = totals_fig

            shares_fig = os.path.join(save_dir, "decode_time_modality_shares_system_masked.png")
            _plot_shares(attn_to_vis_share, attn_to_text_share, out_path=shares_fig)
            plot_paths["decode_shares"] = shares_fig

            heatmap_fig = os.path.join(save_dir, "mean_vision_share_per_head_masked_heatmap.png")
            _plot_head_heatmap(head_np, out_path=heatmap_fig)
            plot_paths["head_heatmap"] = heatmap_fig

            heatmap_fig = os.path.join(save_dir, "mean_sys_vision_share_per_head_masked_heatmap.png")
            _plot_head_heatmap(head_sys_np, out_path=heatmap_fig)
            plot_paths["head_sys_heatmap"] = heatmap_fig

            layer_fig = os.path.join(save_dir, "mean_vision_share_per_head_masked.png")
            _plot_layer_curve(head_np, out_path=layer_fig)
            plot_paths["layer_mean"] = layer_fig

    return DiagnosticResult(
        response_text=text,
        token_labels=token_labels,
        attn_to_vis_share=attn_to_vis_share,
        attn_to_text_share=attn_to_text_share,
        attn_to_sys_share=attn_to_sys_share,
        attn_to_vis_withsys_share=attn_to_vis_withsys_share,
        attn_to_text_withsys_share=attn_to_text_withsys_share,
        pref_score=pref_score,
        mean_vision_share=mean_v,
        mean_text_share=mean_t,
        mean_sys_share=mean_sys,
        mean_vision_withsys_share=mean_v_withsys,
        mean_text_withsys_share=mean_t_withsys,
        mean_pref_score=mean_p,
        head_vision_share=head_np,
        plot_paths=plot_paths,
        hidden_norms_layerwise=hidden_norms_layerwise,
        hidden_norms_paths=hidden_norms_paths,
    )


# -----------------------------
# Plotters (no plt.show, save only)
# -----------------------------
def _plot_totals(llm_attn_matrix: torch.Tensor, prepared: PreparedInputs, out_path: str):
    overall_vis, overall_txt = [], []
    v_s, v_e = prepared.vision

    for step, row in enumerate(llm_attn_matrix[prepared.output_token_start : llm_attn_matrix.shape[0]]):
        row_idx = prepared.output_token_start + step
        past = row[:row_idx]
        v = float(past[v_s : min(v_e, row_idx)].sum().item())
        t = _sum_over_ranges(past, prepared.text_ranges_masked, row_idx=row_idx)
        overall_vis.append(v)
        overall_txt.append(t)

    plt.figure(figsize=(14, 4))
    plt.plot(overall_vis, label="Vision (raw)")
    plt.plot(overall_txt, label="Text (user) (raw)")
    plt.title("Decode-time attention totals")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_shares(vis_shares: List[float], txt_shares: List[float], out_path: str):
    plt.figure(figsize=(14, 3))
    plt.plot(vis_shares, label="Vision share", linewidth=2)
    plt.plot(txt_shares, label="Text (user) share", linewidth=2)
    plt.ylim(0, 1)
    plt.title("Decode-time modality shares")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_head_heatmap(arr: np.ndarray, out_path: str):
    plt.figure(figsize=(10, 6), dpi=140)
    im = plt.imshow(arr, aspect="auto", interpolation="nearest", vmin=0.0, vmax=1.0)
    plt.xlabel("Head")
    plt.ylabel("Layer")
    plt.title("Mean Vision Attention Share per Head")
    cbar = plt.colorbar(im)
    cbar.set_label("Vision share (0 → text, 1 → vision)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_layer_curve(arr: np.ndarray, out_path: str, ylim: Optional[Tuple[float, float]] = (0, 1)):
    plt.figure(figsize=(8, 3), dpi=140)
    plt.plot(arr.mean(axis=1), linewidth=2)
    plt.ylim(ylim)
    plt.xlabel("Layer")
    plt.ylabel("Mean vision share")
    plt.title("Layerwise Mean Vision Share")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_text_shares(text_shares: List[float], sys_shares: List[float], out_path: str):
    """Plot text shares for pure language models (user text vs system prompt)."""
    plt.figure(figsize=(14, 3))
    plt.plot(text_shares, label="User text share", linewidth=2)
    plt.plot(sys_shares, label="System prompt share", linewidth=2)
    plt.ylim(0, 1)
    plt.title("Decode-time text attention shares")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_system_attention(sys_shares: List[float], out_path: str):
    """Plot system prompt attention over time for pure language models."""
    plt.figure(figsize=(14, 3))
    plt.plot(sys_shares, label="System prompt attention", linewidth=2, color='red')
    plt.ylim(0, 1)
    plt.title("System Prompt Attention Over Time")
    plt.xlabel("Generation Step")
    plt.ylabel("Attention Share")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# -----------------------------
# Per-head vision share helper
# -----------------------------
@torch.inference_mode()
def _compute_mean_vision_share_per_head_masked(
    outputs, text_ranges: List[Tuple[int, int]], vision_range: Tuple[int, int]
) -> torch.Tensor:
    """
    outputs['attentions']: tuple over decode steps.
    For each step & layer, take last query (new token), compute per-head:
      share = sum(vision)/[sum(vision)+sum(text_before)+sum(text_after)], with system span removed.
    Returns (L, H) tensor on CPU.
    """
    step_attns = outputs["attentions"]          # len = num_decode_steps
    num_layers = len(step_attns[0])
    H = step_attns[0][0].shape[1]
    device = step_attns[0][0].device

    head_vis_share = torch.zeros(num_layers, H, dtype=torch.float32, device=device)
    count = 0

    v_s, v_e = vision_range
    for step in step_attns:
        for l, layer_attn in enumerate(step):
            # (B, H, Q, K)
            attn = layer_attn.squeeze(0)     # (H, Q, K)
            last_q = attn[:, -1, :]          # (H, K)

            vis = last_q[:, v_s:v_e].sum(dim=-1)  # (H,)
            txt = torch.zeros_like(vis)
            for a, b in text_ranges:
                if a < b:
                    txt = txt + last_q[:, a:b].sum(dim=-1)

            denom = (vis + txt).clamp_min(1e-8)
            head_vis_share[l] += (vis / denom)
        count += 1

    head_vis_share /= max(count, 1)
    return head_vis_share.detach().cpu()
