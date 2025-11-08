# Beyond Semantics — LLaVA Training Add-ons

This repo extends the original [**LLaVA**](https://github.com/haotian-liu/LLaVA) training pipeline with a few flags and small architectural options used in our experiments for the intterventions on LLaVA.

## What’s included

* **Projector normalization** and **multi-layer visual features**
* **Vision token compression** (post-projection)
* Minimal changes to the original codebase (clearly scoped files & functions)

---

## Quick start

### 1) Environment

Install the environment.yaml file and activate it. 
<!-- Or follow the original [LLaVA](https://github.com/haotian-liu/LLaVA) setup (CUDA/PyTorch/etc.).  -->
These scripts assume the standard LLaVA training entrypoints are on your `$PATH`.

### 2) Data prep

Prepare data as in LLaVA, plus the notes under “Data gotchas” below.

### 3) Train

We provide two convenience scripts you can run directly to train the +Normalize + Multilayer features model:

```bash
# Stage 1: Pretraining (vision–language alignment)
bash multiNormpretrain.sh

# Stage 2: Instruction tuning / finetuning
bash multiNormfinetune.sh
```

Both scripts forward the flags described below to the LLaVA training CLI.

---

## New/modified flags

> All flags are **opt-in**, so leaving them out should reproduce vanilla LLaVA behavior.

### Spatial & norm options

* `--normalize_projector`
  Applies RMS/scale normalization to projector outputs (vision → LLM space) to better match text-token norm statistics.

* `--use_multilayer_features`
  Uses intermediate CLIP/Vision-tower features (e.g., layers 12/16/20/24) instead of only the final layer. Improves local geometry retention.

### Token compression (post-projection)

* `--compression_after_projection`
  Enables spatial downsampling **after** the MLP projector.

* `--final_spatial_size <N>`
  Sets the number of tokens **after** compression. Use a power-of-two square.
  Example: for **4×4** spatial tokens, set `--final_spatial_size 16`.

---

## Where the changes live (code map)

* `llava/model/multimodal_projector/builder.py`
  Projector variants and normalization plumbing.

* `llava/model/llava_arch.py`

  * `encode_images(...)`: switch among single-layer vs **multi-layer** vision features.

> If you want to change which visual layers are used for `--use_multilayer_features`, edit the layer picks inside `encode_images`.

---

## Data gotchas & fixes
Following are some common issues and fixes that you may encounter during training. We encoutnered these issues :'( unfortunately.
1. **OCR-VQA “file not found”**
   Use this mirror (the original URL frequently 404s):
   `https://huggingface.co/datasets/ej2/llava-ocr-vqa`

2. **Truncated image errors with PIL**
   Add this to `train.py` (or your main entrypoint) or there will be errors during training:

   ```python
   from PIL import ImageFile
   ImageFile.LOAD_TRUNCATED_IMAGES = True
   ```

3. **Finetune stage save error (can’t save model)**
   Related issues: [#1635](https://github.com/haotian-liu/LLaVA/issues/1635), [#1144](https://github.com/haotian-liu/LLaVA/issues/1144) on the LLaVA repo.
   **Fix**: locate the `generation_config.json` inside your **Vicuna** (or base LLM) folder used for training, set:

   ```json
   "do_sample": true
   ```

   Then re-run training.

---

## Notes & tips

* **`final_spatial_size` meaning**
  It’s the **token count** after compression (not the side length).
  Examples:

  * 1×1 → `final_spatial_size=1`
  * 4×4 → `final_spatial_size=16`
  * 8×8 → `final_spatial_size=64`

* **Throughput**
  Using multi-layer features primarily changes the projector MLP input size (e.g., concat multiple 1024-d CLIP layers to 4096-d). For most setups, the overhead is small compared to the LLM forward pass.

* **Reproducibility**
  Keep the **dataset list and stage-wise hyperparameters** aligned with vanilla LLaVA to make ablations comparable.


---

## Known limitations

* Some datasets/benchmarks are **semantic-heavy**; improvements may be modest unless the task stresses spatial relations.

---

## Upstream & citation

* Please see the original LLaVA repo for full training details, data URLs, and evaluation scripts.
* If you use these flags or scripts in a paper or report, cite LLaVA and our work accordingly.

---

### Footnote: Positional enhancements (exploratory)

For completeness, we include experimental positional options that were **not central to the paper** but may be useful for exploration:

* `--twoD_vision_embedding`: add 2D positional encodings to vision tokens.
* `--interleave` + `--interleave_sparsity <k>`: interleave explicit (x, y) coordinates as text tokens among vision tokens every *k* patches (typically power-of-two).
* `--add_spatial_coordinates`: directly inject numerical (x, y) coords (projected to embedding space) into vision embeddings.

These features are **research-only** and may require tuning; they are not guaranteed to work or improve standard benchmarks.
