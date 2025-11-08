"""
Visualize hidden-state norm comparisons across three models.

Two modes:
  1) --mode avg     : Dataset-average comparison (user_text, vision, vision/user_text ratio).
  2) --mode image   : Per-image comparison for a shared COCO id/file across models.

Expected directory layout under --runs_dir (examples):
  runs/coco_diagnostics/
    llava-v1.5-7b__sp-0/
      per_image/*.npz
    llava-v1.5-7b-normWmean__sp-0/
      per_image/*.npz
    llava-v1.5-7b-multilayerNorm__sp-0/
      per_image/*.npz

Each per-image .npz should contain (when probed with --probe_hidden_norms):
  - 'hidden_norms_user_text' : (L,)
  - 'hidden_norms_vision'    : (L,)
  (Optionally: 'hidden_norms_sys', 'hidden_norms_grid')

Usage examples
--------------
# Dataset-average (log-y for raw norms, log10 ratio on the right)
python viz_hidden_norms.py \
  --runs_dir runs/coco_diagnostics \
  --sp 0 \
  --mode avg \
  --logy \
  --early_k 6 \
  --out hidden_norms_avg_sp-0.png

# Per-image panel (1x3 subplots for three models)
python viz_hidden_norms.py \
  --runs_dir runs/coco_diagnostics \
  --sp 0 \
  --mode image \
  --image_key 000000012639 \
  --logy \
  --out hidden_norms_panel_sp-0_000000012639.png
"""

import os
import re
import glob
import argparse
from typing import Optional, List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------- Defaults (editable) -----------------------------
DEFAULT_MODEL_DIR_NAMES = [
    "llava-v1.5-7b",
    "llava-v1.5-7b-normWmean",
    "llava-v1.5-7b-multilayerNorm",
]

DEFAULT_TITLES_MAP = {
    "llava-v1.5-7b": "LLaVA-1.5-7B",
    "llava-v1.5-7b-normWmean": "+Normalize",
    "llava-v1.5-7b-multilayerNorm": "+Normalize +Multilayer",
}

HIDDEN_KEYS = {
    "user_text": "hidden_norms_user_text",
    "vision": "hidden_norms_vision",
    "sys": "hidden_norms_sys",
    "grid": "hidden_norms_grid",
}

# ----------------------------- Color scheme (matching analyze_coco_diagnostics.py) -----------------------------
def setup_plotting_style():
    """Setup consistent plotting style matching analyze_coco_diagnostics.py"""
    sns.set(style="whitegrid", font_scale=0.9)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

def get_model_color(model_name: str):
    """Get consistent color for model based on name (matching analyze_coco_diagnostics.py)"""
    palette = sns.color_palette("Set2")
    if "llava-v1.5-7b-multilayerNorm" in model_name:
        return palette[1]  # First Set2 color
    elif "llava-v1.5-7b-normWmean" in model_name:
        return palette[0]  # Second Set2 color
    elif "llava-v1.5-7b" in model_name:
        return palette[7]  # Third Set2 color
    else:
        return None  # Default matplotlib color

# ----------------------------- Filesystem helpers -----------------------------
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def list_npzs(per_img_dir: str) -> List[str]:
    if not os.path.isdir(per_img_dir):
        return []
    return sorted(
        os.path.join(per_img_dir, f)
        for f in os.listdir(per_img_dir)
        if f.endswith(".npz")
    )

def extract_coco_id_from_name(name: str) -> Optional[str]:
    m = re.search(r"(\d{12})", name)
    return m.group(1) if m else None

def find_common_npz(per_img_dirs: List[str],
                    image_key: Optional[str] = None,
                    npz_basename: Optional[str] = None) -> str:
    """
    Return a .npz basename that exists in all per_image dirs.
    Priority:
      1) exact npz_basename if provided
      2) any file whose name contains image_key
      3) the first intersection file
    """
    base_dir = per_img_dirs[0]
    base_npzs = list_npzs(base_dir)
    if not base_npzs:
        raise FileNotFoundError(f"No .npz files in {base_dir}")

    # Candidate list
    if npz_basename:
        candidates = [os.path.join(base_dir, npz_basename)] \
                     if os.path.isfile(os.path.join(base_dir, npz_basename)) else []
        if not candidates:
            raise FileNotFoundError(f"{npz_basename} not found in {base_dir}")
    elif image_key:
        candidates = [p for p in base_npzs if image_key in os.path.basename(p)]
        if not candidates:
            raise FileNotFoundError(f"No baseline .npz matched image_key='{image_key}' in {base_dir}")
    else:
        candidates = base_npzs

    # Find first candidate present in all
    for p in candidates:
        base = os.path.basename(p)
        if all(os.path.isfile(os.path.join(d, base)) for d in per_img_dirs[1:]):
            return base
    raise FileNotFoundError("No common per-image .npz basename found across all models.")

# ----------------------------- Loading hidden norms ---------------------------
def load_hidden_from_npz(npz_path: str) -> Dict[str, np.ndarray]:
    """Load hidden-norm arrays from a per-image npz. Returns dict with any found keys."""
    z = np.load(npz_path)
    out = {}
    for k, npz_k in HIDDEN_KEYS.items():
        if npz_k in z:
            out[k] = z[npz_k]
    return out

def gather_hidden_curves(per_img_dir: str) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Scan per_image/*.npz and collect:
        {'basename': {'user_text': (L,), 'vision': (L,), 'sys': (L,)?}}
    Only keeps entries that have BOTH user_text and vision.
    """
    curves: Dict[str, Dict[str, np.ndarray]] = {}
    for f in list_npzs(per_img_dir):
        base = os.path.basename(f)
        try:
            h = load_hidden_from_npz(f)
            if ("user_text" in h) and ("vision" in h):
                # Ensure same length crop if needed
                L = min(int(h["user_text"].shape[0]), int(h["vision"].shape[0]))
                rec = {
                    "user_text": np.asarray(h["user_text"][:L], dtype=np.float64),
                    "vision":    np.asarray(h["vision"][:L],    dtype=np.float64),
                }
                if "sys" in h and int(h["sys"].shape[0]) >= L:
                    rec["sys"] = np.asarray(h["sys"][:L], dtype=np.float64)
                curves[base] = rec
        except Exception:
            # Skip unreadable/malformed files
            continue
    return curves

def intersect_basenames(dicts: List[Dict[str, Dict]]) -> List[str]:
    """Intersection of basenames across multiple {basename: ...} dicts."""
    if not dicts:
        return []
    common = set(dicts[0].keys())
    for d in dicts[1:]:
        common &= set(d.keys())
    return sorted(common)

def _nanmean_1d_stack(vs: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stack to (N,L) and nanmean along axis 0.
    Returns (mean, stack). Crops each to min L first.
    """
    if not vs:
        return np.array([]), np.zeros((0, 0))
    L = min(int(x.shape[0]) for x in vs)
    stack = np.stack([x[:L] for x in vs], axis=0)  # (N,L)
    return np.nanmean(stack, axis=0), stack

# ----------------------------- Aggregation across dataset ---------------------
def collect_dataset_means(per_img_dirs: List[str]) -> Tuple[List[Dict], List[int]]:
    """
    For each model's per_image dir:
      - gather curves per image
      - intersect image set across all models
      - return list of dicts, each dict has:
         {'mean_user_text': (L,), 'mean_vision': (L,), 'mean_ratio': (L,), 'N': int, 'L': int}
    """
    per_model = [gather_hidden_curves(d) for d in per_img_dirs]
    common = intersect_basenames(per_model)
    if len(common) == 0:
        raise FileNotFoundError("No common images with hidden_norms across these models. "
                                "Re-run diagnostics with --probe_hidden_norms.")

    out: List[Dict] = []
    Ns: List[int] = []
    for d in per_model:
        ut_list, vi_list = [], []
        for base in common:
            rec = d.get(base)
            if rec is None:
                continue
            if ("user_text" in rec) and ("vision" in rec):
                ut_list.append(rec["user_text"])
                vi_list.append(rec["vision"])

        if len(ut_list) == 0:
            raise RuntimeError("A model has no usable 'hidden_norms_user_text'/'_vision' arrays.")

        mean_ut, ut_stack = _nanmean_1d_stack(ut_list)  # (L,), (N,L)
        mean_vi, vi_stack = _nanmean_1d_stack(vi_list)  # (L,), (N,L)
        L = min(mean_ut.shape[0], mean_vi.shape[0])
        mean_ut, mean_vi = mean_ut[:L], mean_vi[:L]

        # Average of per-image ratios is more robust than ratio of means
        eps = 1e-12
        ratio_per_img = np.divide(vi_stack[:, :L], ut_stack[:, :L] + eps,
                                  out=np.ones_like(vi_stack[:, :L]),
                                  where=(ut_stack[:, :L] != 0))
        mean_ratio = np.nanmean(ratio_per_img, axis=0)  # (L,)

        out.append({
            "mean_user_text": mean_ut,
            "mean_vision": mean_vi,
            "mean_ratio": mean_ratio,
            "N": ut_stack.shape[0],
            "L": L,
        })
        Ns.append(ut_stack.shape[0])
    return out, Ns

# ----------------------------- Plotting ---------------------------------------
def plot_hidden_dataset_average(
    summaries: List[Dict],
    titles: List[str],
    out_path: Optional[str] = None,
    logy: bool = True,
    early_k: int = 6,
    ratio_log10: bool = True,
    models: Optional[List[str]] = None,
):
    """
    Draws 3-panel figure:
      (a) mean user_text norms vs layer (3 models)
      (b) mean vision norms vs layer (3 models)
      (c) mean vision/user_text ratio vs layer (3 models), log10 if ratio_log10
    """
    n_models = len(summaries)
    assert len(titles) == n_models

    fig = plt.figure(figsize=(18, 4), dpi=150)
    gs = fig.add_gridspec(1, 3, wspace=0.18, hspace=0.2)

    # --- Left: user_text ---
    ax_ut = fig.add_subplot(gs[0, 0])
    for i, s in enumerate(summaries):
        y = s["mean_user_text"]
        color = get_model_color(models[i]) if models else None
        ax_ut.plot(y, label=titles[i], linewidth=2, color=color)
    ax_ut.set_xlabel("Layer"); ax_ut.set_ylabel("Mean norm (user_text)")
    if logy: ax_ut.set_yscale("log")
    ax_ut.set_title("User text norms (average)")
    ax_ut.grid(True, alpha=0.3)

    # --- Middle: vision ---
    ax_vi = fig.add_subplot(gs[0, 1])
    for i, s in enumerate(summaries):
        y = s["mean_vision"]
        color = get_model_color(models[i]) if models else None
        ax_vi.plot(y, label=titles[i], linewidth=2, color=color)
    ax_vi.set_xlabel("Layer"); ax_vi.set_ylabel("Mean norm (vision)")
    if logy: ax_vi.set_yscale("log")
    ax_vi.set_title("Vision norms (average)")
    ax_vi.grid(True, alpha=0.3)

    # --- Right: vision/user_text ratio ---
    ax_rt = fig.add_subplot(gs[0, 2])
    for i, s in enumerate(summaries):
        r = s["mean_ratio"].copy()
        color = get_model_color(models[i]) if models else None
        if ratio_log10:
            r = np.log10(np.maximum(r, 1e-12))
            ax_rt.plot(r, label=titles[i], linewidth=2, color=color)
            ax_rt.set_ylabel("log10(vision/text)")
            # Annotate early-layer average on the right panel
            if s["L"] >= early_k:
                m_early = np.nanmean(r[:early_k])
                # ax_rt.text(0.02, 0.95 - 0.08*i,
                #            f"{titles[i]}  early L<={early_k-1}: {m_early:+.2f}",
                #            transform=ax_rt.transAxes, va="top")
        else:
            ax_rt.plot(r, label=titles[i], linewidth=2, color=color)
            ax_rt.set_ylabel("vision/text")
            if s["L"] >= early_k:
                m_early = np.nanmean(r[:early_k])
                # ax_rt.text(0.02, 0.95 - 0.08*i,
                #            f"{titles[i]}  early L<={early_k-1}: {m_early:.2f}",
                #            transform=ax_rt.transAxes, va="top")

    ax_rt.set_xlabel("Layer")
    ax_rt.set_title("Vision/Text ratio")
    ax_rt.grid(True, alpha=0.3)

    # Add single shared legend using the rightmost axis
    ax_rt.legend(loc='upper right')

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
    else:
        plt.savefig("hidden_norms_dataset_average.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

def plot_hidden_per_image(
    per_model_records: List[Dict[str, np.ndarray]],
    titles: List[str],
    out_path: Optional[str] = None,
    logy: bool = True,
    show_ratio: bool = True,
    models: Optional[List[str]] = None,
):
    """
    Draws a 1x3 panel (one subplot per model) where each subplot has:
      - user_text and vision curves (optionally ratio on secondary axis)
    per_model_records[i] = {"user_text": (L,), "vision": (L,), "sys"?: (L,)}
    """
    n_models = len(per_model_records)
    fig = plt.figure(figsize=(6*n_models, 4), dpi=150)
    gs = fig.add_gridspec(1, n_models, wspace=0.22, hspace=0.22)

    for i, rec in enumerate(per_model_records):
        ax = fig.add_subplot(gs[0, i])
        ut = rec["user_text"].astype(np.float64)
        vi = rec["vision"].astype(np.float64)
        L = min(ut.shape[0], vi.shape[0])
        ut, vi = ut[:L], vi[:L]

        # Get consistent color for this model
        color = get_model_color(models[i]) if models else None
        
        ax.plot(ut, linewidth=2, label="user_text", color=color)
        ax.plot(vi, linewidth=2, label="vision", color=color, linestyle="--")
        ax.set_xlabel("Layer")
        ax.set_title(titles[i])
        ax.grid(True, alpha=0.3)
        if logy: ax.set_yscale("log")
        ax.set_ylabel("Norm")

        if show_ratio:
            eps = 1e-12
            ratio = np.divide(vi, ut + eps)
            ax2 = ax.twinx()
            ax2.plot(ratio, linestyle=":", linewidth=1, label="vision/text", color=color)
            ax2.set_ylabel("vision/text (right)")

            # Merge legends
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines + lines2, labels + labels2, loc="upper right")
        else:
            ax.legend(loc="upper right")

    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
    else:
        plt.savefig("hidden_norms_per_image.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

# ----------------------------- CLI / main -------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Hidden-state norm comparison across models.")
    ap.add_argument("--runs_dir", required=True, help="Root directory that contains <model>__sp-*/ subdirs.")
    ap.add_argument("--sp", type=int, choices=[0, 1], required=True, help="Use __sp-0 or __sp-1 run directories.")
    ap.add_argument("--mode", default="avg", choices=["avg", "image"],
                    help="'avg' for dataset mean; 'image' for per-image panel.")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODEL_DIR_NAMES,
                    help="Model directory name prefixes (subdirs under runs_dir). Default: 3 LLaVA variants.")
    ap.add_argument("--titles", nargs="+", default=None,
                    help="Plot titles for models (same order/length as --models). Default uses readable names.")
    ap.add_argument("--image_key", default=None,
                    help="(image mode) Substring to match per-image .npz filename (e.g., a 12-digit COCO id).")
    ap.add_argument("--npz_basename", default=None,
                    help="(image mode) Exact per-image .npz basename (overrides --image_key).")
    ap.add_argument("--out", default=None, help="Output figure path (PNG).")
    ap.add_argument("--logy", action="store_true",
                    help="Use log scale on Y for raw norms (recommended).")
    ap.add_argument("--no_ratio_log10", action="store_true",
                    help="Use linear ratio instead of log10 ratio (avg mode).")
    ap.add_argument("--early_k", type=int, default=6,
                    help="Early layers range for ratio annotation in avg mode (default: 6 -> layers [0..5]).")
    ap.add_argument("--save_csv", action="store_true",
                    help="Also save dataset-average curves to CSV in avg mode.")

    args = ap.parse_args()

    # Setup consistent plotting style
    setup_plotting_style()

    # Resolve run directories for chosen sp
    run_dirs = [os.path.join(args.runs_dir, f"{name}__sp-{args.sp}") for name in args.models]
    for d in run_dirs:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Missing model dir: {d}")
    per_img_dirs = [os.path.join(d, "per_image") for d in run_dirs]

    # Titles
    if args.titles is None:
        titles = [DEFAULT_TITLES_MAP.get(m, m) for m in args.models]
    else:
        if len(args.titles) != len(args.models):
            raise ValueError("--titles must match --models length.")
        titles = args.titles

    if args.mode == "avg":
        summaries, Ns = collect_dataset_means(per_img_dirs)
        print(f"[avg] Using {min(Ns)} common images across all models "
              f"(per-model Ns: {Ns}).")

        out_path = args.out or f"hidden_norms_avg_sp-{args.sp}.png"
        plot_hidden_dataset_average(
            summaries, titles, out_path=out_path,
            logy=args.logy,
            early_k=args.early_k,
            ratio_log10=(not args.no_ratio_log10),
            models=args.models,
        )
        print(f"[saved] {out_path}")

        if args.save_csv:
            # Save CSV with per-layer dataset means for each model
            # Columns: layer, <model0>_user_text, <model0>_vision, <model0>_ratio, <model1>_..., <model2>_...
            Ls = [s["L"] for s in summaries]
            L = min(Ls) if Ls else 0
            if L == 0:
                print("[warn] No layers available to save CSV.")
            else:
                layer = np.arange(L)[:, None]
                cols = [layer]
                header = ["layer"]
                for title, s in zip(titles, summaries):
                    ut = s["mean_user_text"][:L].reshape(-1, 1)
                    vi = s["mean_vision"][:L].reshape(-1, 1)
                    ra = s["mean_ratio"][:L].reshape(-1, 1)
                    safe_id = re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_")
                    cols += [ut, vi, ra]
                    header += [f"{safe_id}_user_text", f"{safe_id}_vision", f"{safe_id}_ratio"]
                table = np.concatenate(cols, axis=1)
                csv_path = (os.path.splitext(out_path)[0] + ".csv") if args.out else f"hidden_norms_avg_sp-{args.sp}.csv"
                np.savetxt(csv_path, table, delimiter=",", header=",".join(header), comments="", fmt="%.6f")
                print(f"[saved] {csv_path}")

        return

    # ----------------------- Per-image mode -----------------------
    npz_basename = find_common_npz(per_img_dirs, args.image_key, args.npz_basename)
    per_model_records = []
    for per_dir in per_img_dirs:
        p = os.path.join(per_dir, npz_basename)
        h = load_hidden_from_npz(p)
        if ("user_text" not in h) or ("vision" not in h):
            raise KeyError(f"{p} missing required hidden-norm arrays; "
                           f"re-run diagnostics with --probe_hidden_norms.")
        L = min(h["user_text"].shape[0], h["vision"].shape[0])
        per_model_records.append({
            "user_text": h["user_text"][:L],
            "vision":    h["vision"][:L],
        })

    out_path = args.out or f"hidden_norms_panel_sp-{args.sp}_{npz_basename.replace('.npz','')}.png"
    plot_hidden_per_image(per_model_records, titles, out_path=out_path, logy=args.logy, show_ratio=True, models=args.models)
    print(f"[saved] {out_path}")

if __name__ == "__main__":
    main()
