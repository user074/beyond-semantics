import os, argparse, json, math
from typing import List, Tuple, Dict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def list_npzs(per_img_dir: str) -> List[str]:
    return sorted([os.path.join(per_img_dir, f) for f in os.listdir(per_img_dir) if f.endswith(".npz")])

def nanmean_accumulate(sum_arr, cnt_arr, x):
    valid = np.isfinite(x)
    sum_arr[valid] += x[valid]
    cnt_arr[valid] += 1

def heatmap(arr: np.ndarray, title: str, out_path: str, vmin=0.0, vmax=1.0):
    plt.figure(figsize=(10, 6), dpi=140)
    im = plt.imshow(arr, aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax)
    plt.xlabel("Head"); plt.ylabel("Layer"); plt.title(title)
    cbar = plt.colorbar(im); cbar.set_label("Value")
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()

def layer_curve(arr: np.ndarray, title: str, ylabel: str, out_path: str, ylim=None, label=None):
    with np.errstate(invalid="ignore"):
        layer_vals = np.nanmean(arr, axis=1)
    plt.figure(figsize=(8,3), dpi=140)
    if label is None: plt.plot(layer_vals, linewidth=2)
    else: plt.plot(layer_vals, linewidth=2, label=label); plt.legend()
    if ylim is not None: plt.ylim(*ylim)
    plt.xlabel("Layer"); plt.ylabel(ylabel); plt.title(title)
    plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()
    return layer_vals

def load_meta(run_dir: str):
    mpath = os.path.join(run_dir, "meta.json")
    return json.load(open(mpath, "r", encoding="utf-8")) if os.path.isfile(mpath) else {}

def robust_limits(arrs: List[np.ndarray], hi_q: float = 0.99, floor: float = 1e-12) -> Tuple[float,float]:
    stack = np.stack(arrs, axis=0) if len(arrs) > 1 else arrs[0]
    if not np.isfinite(stack).any(): return 0.0, 1.0
    q = float(np.nanquantile(stack, hi_q))
    if not np.isfinite(q) or q <= floor:
        q = float(np.nanmax(stack)) if np.isfinite(stack).any() else 1.0
    if not np.isfinite(q) or q <= floor: q = 1.0
    return 0.0, q

def analyze_one_run(run_dir: str, delta_steps: float):
    per_img_dir = os.path.join(run_dir, "per_image")
    npzs = list_npzs(per_img_dir)
    if not npzs:
        print(f"[WARN] no .npz in {per_img_dir}")
        return None

    probe = np.load(npzs[0])
    L, H = probe["alphaV"].shape

    # accumulate means
    # UNIFORM-Δ
    sum_alphaV = np.zeros((L,H)); cnt_alphaV = np.zeros((L,H), dtype=np.int64)
    sum_delta_gv = np.zeros((L,H)); cnt_delta_gv = np.zeros((L,H), dtype=np.int64)
    sum_abs_delta_gv = np.zeros((L,H)); cnt_abs_delta_gv = np.zeros((L,H), dtype=np.int64)
    sum_delta_alphaV = np.zeros((L,H)); cnt_delta_alphaV = np.zeros((L,H), dtype=np.int64)

    used = 0
    for p in npzs:
        try:
            z = np.load(p)
            alphaV = z["alphaV"]
            delta_gv = z["delta_gv"]
            delta_alphaV = z["delta_alphaV"]
        except Exception as e:
            print(f"[SKIP] {p}: {e}"); continue

        nanmean_accumulate(sum_alphaV, cnt_alphaV, alphaV)
        nanmean_accumulate(sum_delta_gv, cnt_delta_gv, delta_gv)
        nanmean_accumulate(sum_abs_delta_gv, cnt_abs_delta_gv, np.abs(delta_gv))
        nanmean_accumulate(sum_delta_alphaV, cnt_delta_alphaV, delta_alphaV)
        used += 1

    avg_alphaV = np.divide(sum_alphaV, np.maximum(cnt_alphaV, 1), where=cnt_alphaV>0)
    avg_delta_gv = np.divide(sum_delta_gv, np.maximum(cnt_delta_gv, 1), where=cnt_delta_gv>0)
    avg_abs_delta_gv = np.divide(sum_abs_delta_gv, np.maximum(cnt_abs_delta_gv, 1), where=cnt_abs_delta_gv>0)
    avg_delta_alphaV = np.divide(sum_delta_alphaV, np.maximum(cnt_delta_alphaV, 1), where=cnt_delta_alphaV>0)

    out_dir = os.path.join(run_dir, "analysis"); os.makedirs(out_dir, exist_ok=True)

    # save numpy
    np.save(os.path.join(out_dir, "avg_alphaV.npy"), avg_alphaV)
    np.save(os.path.join(out_dir, "avg_delta_gv.npy"), avg_delta_gv)
    np.save(os.path.join(out_dir, "avg_abs_delta_gv.npy"), avg_abs_delta_gv)
    np.save(os.path.join(out_dir, "avg_delta_alphaV.npy"), avg_delta_alphaV)

    meta = load_meta(run_dir)
    summary = dict(
        images_used=int(used), L=int(L), H=int(H),
        run_dir=run_dir,
        model_name=meta.get("model_name"),
        strip_system_prompt=meta.get("strip_system_prompt"),
        delta_steps=float(delta_steps),
        mean_avg_abs_delta_gv=float(np.nanmean(avg_abs_delta_gv)),
        mean_avg_delta_alphaV=float(np.nanmean(avg_delta_alphaV)),
    )
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=2)
    print(summary)
    print(f"[OK] analyzed {used} images for {run_dir}")

    return dict(
        run_dir=run_dir,
        avg_abs_delta_gv=avg_abs_delta_gv,
        avg_delta_alphaV=avg_delta_alphaV,
        avg_delta_gv=avg_delta_gv,
        summary=summary
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", nargs="+", required=True)
    ap.add_argument("--delta_steps", type=float, default=1.0,
                    help="Must match the delta_steps used when running diagnostics")
    args = ap.parse_args()

    results = []
    for rd in args.run_dir:
        out = analyze_one_run(rd, delta_steps=args.delta_steps)
        if out is not None:
            results.append(out)

    # Final comparison plots (only keep the three that are useful)
    if len(results) >= 2:
        # Get base directory (parent of all run directories)
        base_dir = os.path.dirname(results[0]["run_dir"])
        
        # Use same Seaborn styling as the pie charts
        sns.set(style="whitegrid", font_scale=0.9)
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['ps.fonttype'] = 42

        # Use same color palette from Set2 for consistency
        palette = sns.color_palette("Set2")
        
        # Define colors based on model names using Seaborn palette
        def get_model_color(run_dir):
            model_name = os.path.basename(run_dir)
            if "llava-v1.5-7b-multilayerNorm" in model_name:
                return palette[1]  # First Set2 color
            elif "llava-v1.5-7b-normWmean" in model_name:
                return palette[0]  # Second Set2 color
            elif "llava-v1.5-7b" in model_name:
                return palette[7]  # Third Set2 color
            else:
                return None  # Default matplotlib color

        custom_labels = ["LLaVa 1.5", "+ Normalize", "+ Normalize + Multilayer"]

        # 1. compare_abs_delta_gv_layerwise_mean.png
        plt.figure(figsize=(6,3), dpi=140)
        for idx, r in enumerate(results):
            with np.errstate(invalid="ignore"):
                layer_vals = np.nanmean(r["avg_abs_delta_gv"], axis=1)
            label = custom_labels[idx] if idx < len(custom_labels) else os.path.basename(r["run_dir"])
            color = get_model_color(r["run_dir"])
            plt.plot(layer_vals, linewidth=2, label=label, color=color)
        
        plt.xlabel("Layer"); plt.ylabel("|Δg_V|")
        plt.title("|Δg_V| layerwise mean (uniform Δ)")
        plt.grid(True, alpha=0.3); plt.legend()
        comp_out1 = os.path.join(base_dir, "compare_abs_delta_gv_layerwise_mean.png")
        plt.tight_layout(); plt.savefig(comp_out1, dpi=150); plt.close()

        # 2. compare_delta_alphaV_and_abs_delta_gv_layerwise_mean.png
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4), dpi=140)
        
        # Left plot: Delta alphaV uniform
        for idx, r in enumerate(results):
            with np.errstate(invalid="ignore"):
                layer_vals = np.nanmean(r["avg_delta_alphaV"], axis=1)
            label = custom_labels[idx] if idx < len(custom_labels) else os.path.basename(r["run_dir"])
            color = get_model_color(r["run_dir"])
            ax1.plot(layer_vals, linewidth=2, label=label, color=color)
        
        ax1.set_xlabel("Layer"); ax1.set_ylabel("Δα_V")
        ax1.set_title("Δα_V layerwise")
        ax1.grid(True, alpha=0.3); ax1.legend()
        
        # Right plot: Delta gV uniform
        for idx, r in enumerate(results):
            with np.errstate(invalid="ignore"):
                layer_vals = np.nanmean(r["avg_abs_delta_gv"], axis=1)
            label = custom_labels[idx] if idx < len(custom_labels) else os.path.basename(r["run_dir"])
            color = get_model_color(r["run_dir"])
            ax2.plot(layer_vals, linewidth=2, label=label, color=color)
        
        ax2.set_xlabel("Layer"); ax2.set_ylabel("|Δg_V|")
        ax2.set_title("|Δg_V| layerwise")
        ax2.grid(True, alpha=0.3); ax2.legend()
        
        plt.tight_layout()
        comp_out_combined = os.path.join(base_dir, "compare_delta_alphaV_and_abs_delta_gv_layerwise_mean.png")
        plt.savefig(comp_out_combined, dpi=150); plt.close()

if __name__ == "__main__":
    main()
