# Instruction of Evaluation Process



## 1. Download Datasets

This repo evaluates on six datasets: CV-Bench, POPE, MMVP, GQA, VQAv2, and a synthetic dataset. Use the provided scripts to prepare data under the `Data/` directory. Run from anywhere.

Prerequisite: Install git-lfs before running any download scripts that fetch large files (images/parquet) from Hugging Face.


### Quick start: download everything that needs manual preparation

```bash
bash evaluation/data/download_data_scripts/download_all_datasets.sh
```

This installs:
- CV-Bench files into `Data/CV-Bench/`
- POPE COCO images into `Data/coco/`
- VQAv2 COCO test2015 images into `Data/test2015/`
- Synthetic parquet into `Data/simple-synthetic-dataset/data/`

MMVP and GQA are downloaded automatically by the evaluators at runtime (no action needed).

Note on experiment tracking: We also support Weights & Biases (wandb) for logging grader metrics. It's disabled by default. To enable it, set `use_wandb: true` under the `results:` section in the corresponding grader YAML in `conf/grader/`.

### CV-Bench
- Source: `https://huggingface.co/datasets/nyu-visionx/CV-Bench`
- No manual download is required. The evaluator loads CV-Bench via `datasets.load_dataset` at runtime.

### POPE
- Uses COCO 2014 validation images.
- Expected path root: `Data/coco/` (script unzips `val2014` into this directory)
- Script:

```bash
bash evaluation/data/download_data_scripts/download_pope_data.sh
```

### MMVP
- Images and questions are fetched automatically from Hugging Face inside the evaluator (see `evaluation/eval/eval_mmvp.py`). No manual download is required.

### GQA
- Images and questions are loaded via `datasets.load_dataset` in the evaluator (see `evaluation/eval/eval_gqa.py`). No manual download is required.

### VQAv2 (COCO test2015 images)
- Source images: COCO test2015
- Expected path: `Data/test2015/`
- Script:

```bash
bash evaluation/data/download_data_scripts/download_vqav2_test2015.sh
```

The evaluation questions are already included in this repo at `evaluation/data/llava_vqav2_mscoco_test2015.jsonl`.

### Synthetic dataset
- Source: Hugging Face dataset (default repo: `user074/2DSyntheticDataset`).
- Expected path: `Data/2DSyntheticDataset/data/train-00000-of-00001.parquet`
- Script:

```bash
bash evaluation/data/download_data_scripts/download_synthetic_data.sh
```

> **NOTE:** If you want to evaluate on a different dataset, you can follow the same structure and create a new yaml file in `conf/dataset/` and corresponding eval python script. Don't forget to create a new grader in `conf/grader/` and the corresponding grader python script.

## 2. Evaluation

Use the unified entrypoint to run evaluation and grading in one go.

```bash
python -m evaluation.eval_grade
```

Hydra lets you override configs on the command line. Examples:


- Run CV-Bench with default model and grading:
```bash
python -m evaluation.eval_grade dataset=cv_bench grader=cv_bench_grade
```


- Run MMVP (image) or MMVP (text-only):
```bash
python -m evaluation.eval_grade dataset=mmvp grader=mmvp_grade
python -m evaluation.eval_grade dataset=mmvp_text grader=mmvp_grade grader.params.results.directory=answers/mmvp_text
```


- Run GQA in simple mode (no LLM), or enable LLM grading and W&B:
```bash
python -m evaluation.eval_grade dataset=gqa grader=gqa_grade grader.params.grade_method=simple
python -m evaluation.eval_grade dataset=gqa grader=gqa_grade \
  grader.params.grade_method=llm \
  grader.params.llm_model.type=gpt grader.params.llm_model.model=gpt-4o-mini \
  grader.params.results.use_wandb=true
```


- Override model path/name for the dataset being evaluated:
```bash
python -m evaluation.eval_grade dataset=gqa \
  dataset.params.model.path=model/llava-v1.5-7b \
  dataset.params.model.name=llava-v1.5-7b
```


- Toggle permutation or generation settings (from `llava_config`):
```bash
python -m evaluation.eval_grade llava_config.permutation=false llava_config.temperature=0.2
```

Multirun sweeps (Hydra):

- Sweep over multiple models, you need to modify the config file:
```yaml
hydra:
  mode: MULTIRUN
  sweeper:
    # standard grid search
    # additional list sweeper
    list_params:
      dataset.params.model.path: model/llava-v1.5-7b-multilayerNorm,model/llava-v1.5-7b-normWmean
      dataset.params.model.name: llava-v1.5-7b-multilayerNorm,llava-v1.5-7b-normWmean
      # grader.params.model.name: llava-v1.5-7b,llava-v1.5-7b-multilayerNorm,llava-v1.5-7b-normWmean
```

Outputs are written under the `answers/` directory specified by each dataset/grader config, with a unique timestamped `unique_id` auto-assigned per run.
