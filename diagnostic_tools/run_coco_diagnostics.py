
"""
Run VLM diagnostics over COCO val2017:
 - per image: save diagnose_once summary + arrays (head_vision_share, alphaV, delta_gv, delta_alphaV, etc.)
 - per run:   save meta + captions
 - with --pos_only: skip heavy diagnose_once, only run positional sensitivity diagnostics

python run_coco_diagnostics.py \
  --model_path ../checkpoints/llava-v1.5-7b \
  --image_dir ../Data/COCO/val2017 \
  --ann_file ../Data/COCO/annotations/captions_val2017.json \
  --out_root runs/coco_diagnostics_hidden_norms \
  --restrict_to_past \
  --max_new_tokens 32 \
  --probe_hidden_norms \
  --hidden_norm_kind l2
"""


import os, json, argparse
from typing import List, Optional
import numpy as np
import torch
from pycocotools.coco import COCO
from tqdm import tqdm

from attention_utils import ModelBundle, prepare_inputs, diagnose_once
from position_utils import compute_positional_sensitivity

def slugify(s: str) -> str:
    s = s.strip().replace("\\", "/").split("/")[-1]
    for ch in " :/\\|*?\"'<>":
        s = s.replace(ch, "-")
    return s

def ensure_dir(p: str): os.makedirs(p, exist_ok=True)
def save_json(path: str, obj): open(path, "w", encoding="utf-8").write(json.dumps(obj, indent=2))

def run_one_model_on_coco(
    model_path: str,
    image_dir: str,
    ann_file: str,
    out_root: str,
    strip_system_prompt: bool,
    prompt_text: str = "Describe the image.",
    max_new_tokens: int = 64,
    do_sample: bool = False,
    delta_steps: float = 1.0,
    restrict_to_past: bool = True,
    max_images: Optional[int] = None,
    resume: bool = True,
    pos_only: bool = False,  # Skip diagnose_once, only run positional sensitivity
    probe_hidden_norms: bool = False,
    hidden_norm_kind: str = "l2",
    save_hidden_grid: bool = False,
    save_hidden_plots: bool = False,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mb = ModelBundle.load(model_path, device=device)

    model_slug = slugify(mb.model_name or slugify(model_path))
    run_dir = os.path.join(out_root, f"{model_slug}__sp-{1 if strip_system_prompt else 0}")
    per_img_dir = os.path.join(run_dir, "per_image"); ensure_dir(per_img_dir)

    meta = dict(
        model_path=model_path,
        model_name=mb.model_name,
        strip_system_prompt=strip_system_prompt,
        prompt_text=prompt_text,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        delta_steps=delta_steps,
        restrict_to_past=restrict_to_past,
        pos_only=pos_only,
        ann_file=ann_file,
        image_dir=image_dir,
        device=device,
        probe_hidden_norms=probe_hidden_norms,
        hidden_norm_kind=hidden_norm_kind,
        save_hidden_grid=save_hidden_grid,
        save_hidden_plots=save_hidden_plots,
    )
    save_json(os.path.join(run_dir, "meta.json"), meta)

    coco_caps = COCO(ann_file)
    img_ids: List[int] = coco_caps.getImgIds()
    if max_images is not None:
        img_ids = img_ids[:max_images]

    preds = []
    for img_id in tqdm(img_ids, desc=f"Processing {model_slug}", unit="image"):
        img_info = coco_caps.loadImgs(img_id)[0]
        img_path = os.path.join(image_dir, img_info["file_name"])

        base = os.path.join(per_img_dir, f"{str(img_id).zfill(12)}")
        npz_path = base + ".npz"
        summ_json = base + ".json"
        artifacts_dir = base + "_artifacts" if save_hidden_plots else None

        if resume and os.path.isfile(npz_path) and os.path.isfile(summ_json):
            # Load existing caption for preds.json
            with open(summ_json, "r", encoding="utf-8") as f:
                s = json.load(f)
            preds.append({"image_id": img_id, "caption": s.get("caption", "")})
            continue

        try:
            prepared = prepare_inputs(
                mb,
                image_path=img_path,
                prompt_text=prompt_text,
                strip_system_prompt=strip_system_prompt,
            )
            
            if pos_only:
                caption_text = ""
                result = None
            else:
                result = diagnose_once(
                    mb, prepared,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    save_dir=artifacts_dir,
                    make_plots=save_hidden_plots,
                    probe_hidden_norms=probe_hidden_norms,
                    hidden_norm_kind=hidden_norm_kind,
                )
                caption_text = (result.response_text or "").replace("</s>", "").strip()

            pos = compute_positional_sensitivity(
                model=mb.model,
                prepared=prepared,
                delta_steps=delta_steps,
                restrict_to_past=restrict_to_past,
            )
            
            hidden_npz = {}
            if (not pos_only) and probe_hidden_norms and (result is not None):
                hn = result.hidden_norms_layerwise or {}
                # Per-layer norm curves: shape (num_layers,)
                if "sys" in hn and hn["sys"] is not None:
                    hidden_npz["hidden_norms_sys"] = hn["sys"]
                if "user_text" in hn and hn["user_text"] is not None:
                    hidden_npz["hidden_norms_user_text"] = hn["user_text"]
                if "vision" in hn and hn["vision"] is not None:
                    hidden_npz["hidden_norms_vision"] = hn["vision"]

                # Load the (layers × prompt_positions) grid if available
                # Requires save_hidden_plots=True so diagnose_once writes the .npz file
                if save_hidden_grid and save_hidden_plots and result.hidden_norms_paths and "npz" in result.hidden_norms_paths:
                    try:
                        grid_file = result.hidden_norms_paths["npz"]
                        with np.load(grid_file) as z:
                            if "grid" in z:
                                hidden_npz["hidden_norms_grid"] = z["grid"]  # Shape: (layers, prompt_positions)
                    except Exception as _e:
                        # Skip grid if file can't be loaded
                        pass


            if pos_only:
                np.savez_compressed(
                    npz_path,
                    alphaV=pos["alphaV"],
                    share_prod_notV=pos["share_prod_notV"],
                    delta_gv=pos["delta_gv"],
                    delta_alphaV=pos["delta_alphaV"],
                )
            else:
                np.savez_compressed(
                    npz_path,
                    head_vision_share=result.head_vision_share,  # Shape: (layers, heads)
                    alphaV=pos["alphaV"],
                    share_prod_notV=pos["share_prod_notV"],
                    delta_gv=pos["delta_gv"],
                    delta_alphaV=pos["delta_alphaV"],
                    **hidden_npz,
                )


            if pos_only:
                summary = dict(
                    image_id=img_id,
                    file_name=img_info["file_name"],
                    caption=caption_text,
                    pos_only=True,
                )
            else:
                summary = dict(
                    image_id=img_id,
                    file_name=img_info["file_name"],
                    caption=caption_text,
                    mean_vision_share=float(result.mean_vision_share),
                    mean_text_share=float(result.mean_text_share),
                    mean_sys_share=float(result.mean_sys_share),
                )
            if not pos_only and probe_hidden_norms and (result is not None):
                summary["hidden_norms"] = {
                    "kind": hidden_norm_kind,
                    "has_sys": bool("hidden_norms_sys" in hidden_npz),
                    "has_user_text": bool("hidden_norms_user_text" in hidden_npz),
                    "has_vision": bool("hidden_norms_vision" in hidden_npz),
                    "saved_grid": bool("hidden_norms_grid" in hidden_npz),
                    "artifacts_dir": artifacts_dir if save_hidden_plots else None,
                }

            save_json(summ_json, summary)
            preds.append({"image_id": img_id, "caption": caption_text})

        except Exception as e:
            save_json(summ_json, {"image_id": img_id, "file_name": img_info["file_name"], "error": repr(e)})

    save_json(os.path.join(run_dir, "preds.json"), preds)
    print(f"[OK] Finished run: {run_dir}")
    return run_dir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", action="append", required=True)
    ap.add_argument("--image_dir", type=str, required=True)
    ap.add_argument("--ann_file", type=str, required=True)
    ap.add_argument("--out_root", type=str, default="runs/coco_diagnostics")

    ap.add_argument("--strip_system_prompt", action="store_true")
    ap.add_argument("--prompt_text", type=str, default="Describe the image.")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--do_sample", action="store_true")

    ap.add_argument("--delta_steps", type=float, default=1.0)
    ap.add_argument("--restrict_to_past", action="store_true")

    ap.add_argument("--max_images", type=int, default=None)
    ap.add_argument("--no_resume", action="store_true")
    ap.add_argument("--pos_only", action="store_true", help="Only run positional sensitivity, skip heavy diagnose_once computation")
    
    ap.add_argument("--probe_hidden_norms", action="store_true",
                    help="Compute per-layer hidden-state norms for system/user/vision groups.")
    ap.add_argument("--hidden_norm_kind", type=str, default="l2", choices=["l2", "rms"],
                    help="Norm type for hidden-state probing.")
    ap.add_argument("--save_hidden_grid", action="store_true",
                    help="Also save the (layers × prompt-positions) norm grid into the per-image NPZ (can be large).")
    ap.add_argument("--save_hidden_plots", action="store_true",
                    help="Save hidden-state plots (layerwise curves + heatmap) to a per-image artifacts folder.")


    args = ap.parse_args()
    for mp in args.model_path:
        run_one_model_on_coco(
            model_path=mp,
            image_dir=args.image_dir,
            ann_file=args.ann_file,
            out_root=args.out_root,
            strip_system_prompt=args.strip_system_prompt,
            prompt_text=args.prompt_text,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            delta_steps=args.delta_steps,
            restrict_to_past=args.restrict_to_past,
            max_images=args.max_images,
            resume=not args.no_resume,
            pos_only=args.pos_only,
            probe_hidden_norms=args.probe_hidden_norms,
            hidden_norm_kind=args.hidden_norm_kind,
            save_hidden_grid=args.save_hidden_grid,
            save_hidden_plots=args.save_hidden_plots,
        )

if __name__ == "__main__":
    main()
