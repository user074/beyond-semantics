"""
Visualize per-image CMB heatmaps (vision share per head) for three models, plus the COCO image.

Expected directory layout under --runs_dir (example):
  llava-v1.5-7b__sp-0/
  llava-v1.5-7b-normWmean__sp-0/
  llava-v1.5-7b-multilayerNorm__sp-0/
(and the __sp-1 variants)

Within each model dir:
  per_image/*.npz with keys including either:
    - 'head_vision_share'  (preferred for CMB), or
    - 'cmb_baseline'       (fallback; also vision share per head)

Usage examples:
  # Visualize a specific image:
  python viz_cmb_panel.py \
    --runs_dir runs/coco_diagnostics \
    --sp 1 \
    --image_key 000000012639 \
    --coco_root ../Data/COCO/
  
  # Average over all results (omit --image_key):
  python viz_cmb_panel.py \
    --runs_dir runs/coco_diagnostics \
    --sp 1 \
    --coco_root ../Data/COCO/
"""

import os
import re
import glob
import argparse
from typing import Optional, List, Tuple
import numpy as np
import matplotlib.pyplot as plt

try:
    from PIL import Image
    PIL_OK = True
except Exception:
    PIL_OK = False

# The three model variants (order = baseline, +Normalize, +Normalize+Multilayer)
MODEL_DIR_NAMES = [
    "llava-v1.5-7b",
    "llava-v1.5-7b-normWmean",
    "llava-v1.5-7b-multilayerNorm",
]

TITLES = [
    "LLaVA-1.5-7B",
    "LLaVA-1.5-7B + Normalize",
    "LLaVA-1.5-7B + Normalize + Multilayer",
]

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

def find_coco_image(coco_root: str, npz_basename: str) -> Optional[str]:
    img_id = extract_coco_id_from_name(npz_basename)
    if not img_id:
        return None
    candidates = [
        os.path.join(coco_root, "val2017", f"{img_id}.jpg"),
        os.path.join(coco_root, "train2017", f"{img_id}.jpg"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    for sub in ("val2017", "train2017"):
        hits = glob.glob(os.path.join(coco_root, sub, f"*{img_id}*.jpg"))
        if hits:
            return hits[0]
    return None

def load_image(img_path: str) -> Optional[np.ndarray]:
    if not PIL_OK or not (img_path and os.path.isfile(img_path)):
        return None
    im = Image.open(img_path).convert("RGB")
    return np.array(im)

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
    # Gather baseline list from the first dir
    base_dir = per_img_dirs[0]
    base_npzs = list_npzs(base_dir)
    if not base_npzs:
        raise FileNotFoundError(f"No .npz files in {base_dir}")
    candidates: List[str]
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
        candidates = base_npzs  # try all

    # Pick the first candidate present in all per_image dirs
    for p in candidates:
        base = os.path.basename(p)
        if all(os.path.isfile(os.path.join(d, base)) for d in per_img_dirs[1:]):
            return base
    raise FileNotFoundError("No common per-image .npz basename found across all three models.")

def find_all_common_npzs(per_img_dirs: List[str]) -> List[str]:
    """
    Return a list of all .npz basenames that exist in all per_image dirs.
    """
    # Gather baseline list from the first dir
    base_dir = per_img_dirs[0]
    base_npzs = list_npzs(base_dir)
    if not base_npzs:
        raise FileNotFoundError(f"No .npz files in {base_dir}")
    
    # Find all files present in all per_image dirs
    common_npzs = []
    for p in base_npzs:
        base = os.path.basename(p)
        if all(os.path.isfile(os.path.join(d, base)) for d in per_img_dirs[1:]):
            common_npzs.append(base)
    
    if not common_npzs:
        raise FileNotFoundError("No common per-image .npz files found across all three models.")
    
    return common_npzs

def get_cmb_from_npz(npz_path: str,
                     key_preference: Tuple[str, str] = ("head_vision_share", "cmb_baseline")) -> np.ndarray:
    """
    Load vision-share-per-head (CMB) from a per-image npz.
    Tries 'head_vision_share' first, then 'cmb_baseline'.
    """
    z = np.load(npz_path)
    for k in key_preference:
        if k in z:
            return z[k]
    raise KeyError(f"Neither {key_preference[0]} nor {key_preference[1]} found in {npz_path}")

def robust_limits(arrs: List[np.ndarray], hi_q: float = 0.99, floor: float = 1e-12) -> Tuple[float, float]:
    valid = [a[np.isfinite(a)] for a in arrs if a is not None]
    if not valid:
        return 0.0, 1.0
    cat = np.concatenate(valid, axis=None)
    if cat.size == 0:
        return 0.0, 1.0
    vmax = float(np.quantile(cat, hi_q))
    if not np.isfinite(vmax) or vmax <= floor:
        vmax = float(np.nanmax(cat)) if np.isfinite(np.nanmax(cat)) else 1.0
    if not np.isfinite(vmax) or vmax <= floor:
        vmax = 1.0
    return 0.0, vmax

def main():
    ap = argparse.ArgumentParser(description="COCO image + per-image CMB heatmaps for three models.")
    ap.add_argument("--runs_dir", required=True, help="Path to coco_diagnostics directory containing model subdirs.")
    ap.add_argument("--sp", type=int, choices=[0, 1], required=True, help="Use __sp-0 or __sp-1 subdirs.")
    ap.add_argument("--image_key", default=None, help="Substring to match the per-image .npz filename (e.g., a 12-digit COCO id).")
    ap.add_argument("--npz_basename", default=None, help="Exact per-image .npz basename to use (overrides --image_key).")
    ap.add_argument("--img_path", default=None, help="Optional explicit path to the RGB image to show.")
    ap.add_argument("--coco_root", default=None, help="Optional COCO images root; script tries to locate the image automatically.")
    ap.add_argument("--use_key", default="head_vision_share",
                    choices=["head_vision_share", "cmb_baseline"],
                    help="Which npz key to read as CMB (default: head_vision_share). Fallback key is tried automatically.")
    ap.add_argument("--scale", default="fixed", choices=["fixed", "auto"],
                    help="Color scale: 'fixed' -> [0,1]; 'auto' -> robust [0, q99]. Default: fixed.")
    ap.add_argument("--out", default=None, help="Output figure path.")
    ap.add_argument("--no_show", action="store_true", help="If set, do not show the figure (just save).")
    args = ap.parse_args()

    # Build model run dirs for chosen sp
    run_dirs = [os.path.join(args.runs_dir, f"{name}__sp-{args.sp}") for name in MODEL_DIR_NAMES]
    for d in run_dirs:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Missing model dir: {d}")
    per_img_dirs = [os.path.join(d, "per_image") for d in run_dirs]

    # If image_key and npz_basename are both None, average over all results
    is_averaging = args.image_key is None and args.npz_basename is None
    if is_averaging:
        # Find all common npz files
        all_npz_basenames = find_all_common_npzs(per_img_dirs)
        print(f"Averaging over {len(all_npz_basenames)} common .npz files")
        
        # Load and average CMB arrays per model
        cmbs = []
        shapes = []
        for per_dir in per_img_dirs:
            cmb_list = []
            for npz_bn in all_npz_basenames:
                p = os.path.join(per_dir, npz_bn)
                cmb = get_cmb_from_npz(p, key_preference=(args.use_key, "cmb_baseline" if args.use_key == "head_vision_share" else "head_vision_share"))
                cmb_list.append(cmb)
            
            stacked = np.stack(cmb_list, axis=0)
            avg_cmb = np.mean(stacked, axis=0)
            cmbs.append(avg_cmb)
            shapes.append(avg_cmb.shape)
        
        # Use a placeholder for npz_basename in output filename
        npz_basename = "averaged"
    else:
        # Pick a common per-image file
        npz_basename = find_common_npz(per_img_dirs, args.image_key, args.npz_basename)

        # Load CMB arrays per model
        cmbs = []
        shapes = []
        for per_dir in per_img_dirs:
            p = os.path.join(per_dir, npz_basename)
            cmb = get_cmb_from_npz(p, key_preference=(args.use_key, "cmb_baseline" if args.use_key == "head_vision_share" else "head_vision_share"))
            cmbs.append(cmb)
            shapes.append(cmb.shape)

    if len(set(shapes)) != 1:
        raise ValueError(f"Inconsistent CMB shapes across models: {shapes}")

    # Color scaling
    if args.scale == "fixed":
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = robust_limits(cmbs, hi_q=0.99)

    # Resolve the RGB image (optional, only when not averaging)
    rgb = None
    if not is_averaging:
        if args.img_path and os.path.isfile(args.img_path):
            rgb = load_image(args.img_path)
        elif args.coco_root:
            maybe = find_coco_image(args.coco_root, npz_basename)
            if maybe:
                rgb = load_image(maybe)

    # Plot: 1x3 grid when averaging, 1x4 grid otherwise
    if is_averaging:
        fig = plt.figure(figsize=(18, 5), dpi=140)
        gs = fig.add_gridspec(1, 3, wspace=0.12, hspace=0.18)
        positions = [(0, 0), (0, 1), (0, 2)]
    else:
        fig = plt.figure(figsize=(24, 5), dpi=140)
        gs = fig.add_gridspec(1, 4, wspace=0.12, hspace=0.18)
        # (0,0): COCO image or placeholder
        ax_img = fig.add_subplot(gs[0, 0])
        if rgb is not None:
            ax_img.imshow(rgb)
            ax_img.set_title("COCO image")
        else:
            ax_img.text(0.5, 0.5, "Image not provided", ha="center", va="center", fontsize=12)
            ax_img.set_title("COCO image (missing)")
        ax_img.axis("off")
        positions = [(0, 1), (0, 2), (0, 3)]

    # Heatmaps
    for cmb, (r, c), title in zip(cmbs, positions, TITLES):
        ax = fig.add_subplot(gs[r, c])
        im = ax.imshow(cmb, aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax)
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        ax.set_title(f"{title}")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("CMB = vision share α_V")

    # fig.suptitle(f"CMB heatmaps (sp-{args.sp}) • file: {npz_basename}", y=0.995, fontsize=14)
    plt.tight_layout()
    if args.out:
        plt.savefig(args.out, dpi=200, bbox_inches="tight")
    else:
        nps_name = npz_basename.replace(".npz", "")
        plt.savefig(f"cmb_panel_sp-{args.sp}_{nps_name}.png", dpi=200, bbox_inches="tight")
    if not args.no_show:
        plt.show()
    plt.close(fig)

if __name__ == "__main__":
    main()