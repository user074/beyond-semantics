#!/bin/bash

set -euo pipefail

# Wrapper to download all datasets that require manual preparation.
# Includes: CV-Bench, POPE (provided), VQAv2 test2015, Synthetic.
# MMVP and GQA are auto-downloaded during evaluation; we just print notes.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

echo "=== Downloading CV-Bench ==="
bash "${SCRIPT_DIR}/download_cv_bench_data.sh"

echo "=== Downloading POPE COCO val2014 ==="
bash "${SCRIPT_DIR}/download_pope_data.sh"

echo "=== Downloading VQAv2 COCO test2015 images ==="
bash "${SCRIPT_DIR}/download_vqav2_test2015.sh"

echo "=== Downloading Synthetic dataset parquet ==="
bash "${SCRIPT_DIR}/download_synthetic_data.sh"

echo "=== Notes ==="
echo "MMVP: images and questions are automatically pulled from Hugging Face by the evaluator."
echo "GQA: images and questions are automatically loaded via datasets.load_dataset."
echo "All datasets prepared."


