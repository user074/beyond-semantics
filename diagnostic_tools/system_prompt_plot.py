import os, json, glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def plot_dataset_attention_pies(run_dir, save_dir=None, title_prefix="", side_by_side=False):
    """
    Aggregate mean shares from per-image JSONs under: <run_dir>/per_image/*.json
    and plot:
      (A) System vs Vision vs User
      (B) Vision vs User only
    """
    per_img_dir = os.path.join(run_dir, "per_image")
    files = sorted(glob.glob(os.path.join(per_img_dir, "*.json")))
    if not files:
        raise FileNotFoundError(f"No per-image JSONs found in {per_img_dir}")

    vis_masked_vals, txt_masked_vals, sys_vals, vis_wsys_vals, txt_wsys_vals = [], [], [], [], []

    for fp in files:
        try:
            s = json.load(open(fp, "r", encoding="utf-8"))
            # Masked (system removed) shares:
            vm = s.get("mean_vision_share", None)
            tm = s.get("mean_text_share", None)
            # With-system: you can reconstruct (Vision+User sums to 1-system)
            sm = s.get("mean_sys_share", None)

            if vm is not None and tm is not None:
                vis_masked_vals.append(float(vm))
                txt_masked_vals.append(float(tm))

            if sm is not None:
                sys_vals.append(float(sm))
                # If you want per-image with-system breakdown, map masked → with-system by
                # multiplying by (1 - sys) to get absolute masses, then renormalize.
                # Absolute masses:
                v_abs = (float(vm) if vm is not None else 0.0) * (1.0 - float(sm))
                t_abs = (float(tm) if tm is not None else 0.0) * (1.0 - float(sm))
                tot = v_abs + t_abs + float(sm)
                if tot > 0:
                    vis_wsys_vals.append(v_abs / tot)
                    txt_wsys_vals.append(t_abs / tot)
        except Exception:
            continue

    if not vis_masked_vals or not txt_masked_vals:
        raise RuntimeError("Missing mean_vision_share / mean_text_share in per-image JSONs.")

    # Aggregate means
    vis_masked_mean = float(np.mean(vis_masked_vals))
    txt_masked_mean = float(np.mean(txt_masked_vals))

    if sys_vals:
        sys_mean = float(np.mean(sys_vals))
        # If we built per-image with-system stats above:
        if vis_wsys_vals and txt_wsys_vals:
            vis_wsys_mean = float(np.mean(vis_wsys_vals))
            txt_wsys_mean = float(np.mean(txt_wsys_vals))
        else:
            # Fallback: map masked means into with-system using the dataset mean sys share
            v_abs = vis_masked_mean * (1.0 - sys_mean)
            t_abs = txt_masked_mean * (1.0 - sys_mean)
            tot = v_abs + t_abs + sys_mean
            vis_wsys_mean = v_abs / tot if tot > 0 else 0.0
            txt_wsys_mean = t_abs / tot if tot > 0 else 0.0
    else:
        # No system share recorded; treat as zero system
        sys_mean, vis_wsys_mean, txt_wsys_mean = 0.0, vis_masked_mean, txt_masked_mean

    # Normalize defensively
    def _norm(*vals):
        arr = np.array([max(0.0, float(v)) for v in vals], dtype=float)
        s = arr.sum()
        return arr / s if s > 0 else arr

    a_with_sys = _norm(sys_mean, vis_wsys_mean, txt_wsys_mean)
    a_masked   = _norm(vis_masked_mean, txt_masked_mean)

    model_name = json.load(open(os.path.join(run_dir, "meta.json")))\
        .get("model_name", os.path.basename(run_dir))
    prefix = f"{title_prefix or model_name}: "

    # Use the same Seaborn styling as the radar plot
    sns.set(style="whitegrid", font_scale=1.2)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42
    plt.rcParams['font.size'] = 12
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.labelweight'] = 'bold'

    # Use same color palette from Set2 for consistency
    palette = sns.color_palette("Set2")
    colors = [palette[0], palette[1], palette[2]]  # System, Vision, User text
    vision_color = palette[1]  # Vision color
    user_text_color = palette[2]  # User text color
    
    out_dir = save_dir or os.path.join(run_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)
    
    # Custom autopct function to show labels inside pie slices
    def make_autopct_with_labels(values, labels):
        def autopct(pct):
            # Find which slice this percentage corresponds to
            for i, val in enumerate(values):
                if abs(val * 100 - pct) < 0.1:  # Match within 0.1%
                    if pct > 5:  # Only show if slice is large enough
                        return f'{labels[i]}\n{pct:.1f}%'
                    break
            return ''
        return autopct
    
    if side_by_side:
        # Create side-by-side subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))
        
        # Pie A
        wedges, texts, autotexts = ax1.pie(a_with_sys, labels=None, 
                colors=colors, autopct=make_autopct_with_labels(a_with_sys, ["System prompt", "Vision", "User text"]), 
                startangle=0, textprops={'fontsize': 8, 'fontweight': 'bold'})
        ax1.set_title("Attention shares", fontsize=14, fontweight='bold')
        
        # Pie B
        wedges2, texts2, autotexts2 = ax2.pie(a_masked, labels=None, 
                colors=[vision_color, user_text_color], 
                autopct=make_autopct_with_labels(a_masked, ["Vision", "User text"]), 
                startangle=0, textprops={'fontsize': 8, 'fontweight': 'bold'})
        ax2.set_title("Attention shares (Vision vs User Text)", fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "dataset_pies_side_by_side.png"), dpi=150)
        plt.close()
    else:
        # Pie A
        plt.figure(figsize=(5.5, 5.5))
        wedges, texts, autotexts = plt.pie(a_with_sys, labels=None, 
                colors=colors, autopct=make_autopct_with_labels(a_with_sys, ["System prompt", "Vision", "User text"]), 
                startangle=90, textprops={'fontsize': 13, 'fontweight': 'bold'})
        plt.title("Average attention shares (System vs Vision vs Input Text)", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "dataset_pie_with_system.png"), dpi=150)
        plt.close()

        # Pie B
        plt.figure(figsize=(5.5, 5.5))
        wedges2, texts2, autotexts2 = plt.pie(a_masked, labels=None, 
                colors=[vision_color, user_text_color], 
                autopct=make_autopct_with_labels(a_masked, ["Vision", "User text"]), 
                startangle=90, textprops={'fontsize': 13, 'fontweight': 'bold'})
        plt.title("Average attention shares (Vision vs User Text)", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "dataset_pie_vision_vs_user_masked.png"), dpi=150)
        plt.close()

    print(f"[OK] wrote pies to {out_dir}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", nargs="+", required=True)
    ap.add_argument("--side_by_side", action="store_true", 
                    help="Plot pie charts side by side in a single figure")
    args = ap.parse_args()

    for run_dir in args.run_dir:
        plot_dataset_attention_pies(run_dir, side_by_side=args.side_by_side)

if __name__ == "__main__":
    main()