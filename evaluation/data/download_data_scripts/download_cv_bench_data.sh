#!/bin/bash

set -euo pipefail

# This script downloads the full CV-Bench dataset (including images)
# into the project's Data directory via git-lfs.
# Dataset: nyu-visionx/CV-Bench on Hugging Face

# Resolve directories
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ROOT_DIR=$(dirname "$(dirname "$(dirname "${SCRIPT_DIR}")")")
TARGET_DIR="${ROOT_DIR}/Data/CV-Bench"

REPO_URL="https://huggingface.co/datasets/nyu-visionx/CV-Bench"

# Require git and git-lfs to retrieve images
if ! command -v git &> /dev/null; then
  echo "Error: git is required. Please install git and re-run." >&2
  exit 1
fi
if ! command -v git-lfs &> /dev/null; then
  echo "Error: git-lfs is required to download images. Please install git-lfs and re-run." >&2
  exit 1
fi

git lfs install

if [ -d "${TARGET_DIR}/.git" ]; then
  echo "Existing CV-Bench repo found at ${TARGET_DIR}. Updating..."
  git -C "${TARGET_DIR}" remote set-url origin "${REPO_URL}" || true
  git -C "${TARGET_DIR}" fetch --all --prune
  git -C "${TARGET_DIR}" checkout main 2>/dev/null || true
  git -C "${TARGET_DIR}" pull --ff-only origin main || git -C "${TARGET_DIR}" pull --ff-only || true
  git -C "${TARGET_DIR}" lfs pull
else
  echo "Cloning full CV-Bench (with images) into ${TARGET_DIR}..."
  mkdir -p "${TARGET_DIR}"
  git clone "${REPO_URL}" "${TARGET_DIR}"
  git -C "${TARGET_DIR}" lfs pull
fi

echo "CV-Bench fully downloaded to ${TARGET_DIR}"

# If test.jsonl is missing but the split files exist, combine them
if [ ! -f "${TARGET_DIR}/test.jsonl" ]; then
  if [ -f "${TARGET_DIR}/test_2d.jsonl" ] && [ -f "${TARGET_DIR}/test_3d.jsonl" ]; then
    echo "Combining test_2d.jsonl and test_3d.jsonl into test.jsonl..."
    python3 "${SCRIPT_DIR}/combine_cv_bench.py" "${TARGET_DIR}"
  else
    echo "test.jsonl not found, and split files are missing. Skipping combine."
  fi
fi

# If images are not present, try to reconstruct from parquet files
if [ ! -d "${TARGET_DIR}/img" ] || ! find "${TARGET_DIR}/img" -type f -name "*.png" -print -quit | grep -q .; then
  if [ -f "${TARGET_DIR}/test_2d.parquet" ] || [ -f "${TARGET_DIR}/test_3d.parquet" ]; then
    echo "Reconstructing CV-Bench images from parquet files..."
    python3 "${TARGET_DIR}/build_img.py" --root "${TARGET_DIR}" --out "${TARGET_DIR}/img" --subset both
  else
    echo "Parquet files not found; cannot reconstruct images."
  fi
fi

