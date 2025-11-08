# Diagnostic Tools

This directory contains tools for diagnosing and analyzing the performance of the model.
You will need to download the checkpoints from the Hugging Face model zoo: https://huggingface.co/collections/user074/beyond-semantics

The checkpoints are stored in the `checkpoints` directory.
```
beyond-semantics
├── checkpoints
│   ├── llava-v1.5-7b
│   ├── llava-v1.5-7b-normWmean
│   └── llava-v1.5-7b-multilayerNorm
```

Otherwise you can directly use our colab notebook to go through all the steps: 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/user074/beyond-semantics/blob/main/diagnostic_tools/Diagnostic_Tools_Walkthrough.ipynb)


## Analyze the vision vs text token norms

`vlm_token_norms.py` is a script that computes and visualizes L2-norm distributions for vision and text tokens in LLaVA models.

```bash
python vlm_token_norms.py \
--model_path ../checkpoints/llava-v1.5-7b \
--vision_tower_path ../checkpoints/clip-vit-large-patch14-336 \
--coco_root ../Data/COCO \
--device cuda \
--weights_type bin \
--max_images 20 \
--save_vectors \
--out_dir outputs/token_norms
```

Compute and visualize L2-norm distributions for:
  - Vision tokens (Frozen CLIP -> projector -> LLM-input features)
  - Text tokens (LLM input embeddings)


## Diagnose the model through the COCO dataset
`run_coco_diagnostics.py` is a script that runs diagnostics over the COCO dataset.

```bash
python run_coco_diagnostics.py \
  --model_path ../checkpoints/llava-v1.5-7b \
  --image_dir ../Data/COCO/val2017 \
  --ann_file ../Data/COCO/annotations/captions_val2017.json \
  --out_root runs \
  --restrict_to_past \
  --max_new_tokens 32 \
  --probe_hidden_norms \
  --hidden_norm_kind l2 \
  --strip_system_prompt \
  --max_images 20
```

You can substitute the `--model_path` with the following models:
- llava-v1.5-7b
- llava-v1.5-7b-normWmean
- llava-v1.5-7b-multilayerNorm


The following parameters are measured:

- **Cross Modality Balance (CMB)**: Measures the vision and text token shares in the attention heads as a percentage. 1 means 100% vision shares, 0 means 100% text shares.
- **RoPE sensitivity probe**: Measures delta_gv (Δ_gv) and delta_alphaV (Δ_alphaV)
- **Hidden norms**: Measures the hidden norms of the system, user, and vision tokens in the attention heads.

**Key parameters to set:**
- `strip_system_prompt`: Whether to strip the system prompt from the prompt text. As our experiments show, there is a large discrepancy between the model performance with and without the system prompt since most of the textual attention is on the system prompt. For CMB analysis, we find it is better to strip the system prompt.
- `max_images`: The maximum number of images to process. You can set it to a smaller number for testing instead of processing the full COCO dataset.

This will create a directory with the following structure:
```
runs/
  llava-v1.5-7b__sp-0/
  llava-v1.5-7b-multilayerNorm__sp-0/
  llava-v1.5-7b-normWmean__sp-0/
```
Each directory contains per-image diagnosis results saved in the `per_image` subdirectory.

## Analyze the diagnosis results
### Analyze the RoPE sensitivity results

Analyze and visualize the diagnosis results using `analyze_coco_diagnostics.py`.
```bash
python analyze_coco_diagnostics.py --run_dir runs/llava-v1.5-7b__sp-0 runs/llava-v1.5-7b-multilayerNorm__sp-0 runs/llava-v1.5-7b-normWmean__sp-0
```
This will create comparison plots of the delta_gv and delta_alphaV across layers, showing the sensitivity of the model to positional encoding changes.

### Analyze the system prompt attention shares

As mentioned before, the system prompt might cause a large attention allocation to it. We can visualize the system prompt vs vision vs user text shares using `system_prompt_plot.py`. This will give us a clear visualization of each modality's shares. Results are saved in each model's analysis directory.  
```
python system_prompt_plot.py --run_dir runs/llava-v1.5-7b__sp-0 runs/llava-v1.5-7b-multilayerNorm__sp-0 runs/llava-v1.5-7b-normWmean__sp-0
```
This will create comparison plots of the system prompt vs vision vs user text shares across layers, showing the attention allocation to the system prompt.

### Analyze the Cross Modality Balance (CMB) results

As mentioned in the paper, we want to investigate the changing variables for the model inputs, which are the vision tokens and user input text tokens. Visualize the CMB results using `viz_cmb_panel.py`:

```bash
python viz_cmb_panel.py \
  --runs_dir runs \
  --sp 0 \
  --coco_root ../Data/COCO/ \
  --out cmb.png
```

**Note**: `sp 0` means with the system prompt, `sp 1` means without the system prompt. The command above will create average CMB results across all images.

In addition, you can visualize the CMB results for a specific image by specifying the image key:

```bash
python viz_cmb_panel.py \
  --runs_dir runs \
  --sp 0 \
  --image_key 000000006818 \
  --coco_root ../Data/COCO/ \
  --out cmb_panel_sp0_000000006818.png
```

### Analyze the hidden norms results

Analyze and visualize the hidden norms results using `viz_hidden_norms.py`. This script compares the hidden norms of the vision tokens and text tokens across layers. We can observe that the vision norms are orders of magnitude larger than the text norms in the early layers. The command below will create average hidden norms results across all images.

```bash
python viz_hidden_norms.py \
  --runs_dir runs \
  --sp 0 \
  --mode avg \
  --logy \
  --early_k 6 \
  --out hidden_norms_avg_sp0.png
```

## Visualize attention heatmaps over images

`viz_attention_image.py` is a tool for visualizing attention heatmaps over images, showing which image regions the model attends to when generating specific tokens. This is useful for understanding what parts of an image the model focuses on when producing certain words.

```bash
python viz_attention_image.py
```

The visualization shows:
- **Attention heatmap**: Color-coded overlay showing attention weights (red/yellow = high attention, blue = low attention)
- **Normalized entropy**: Measure of attention distribution (lower = more focused, higher = more diffuse)
- **Generated text**: The model's response for reference