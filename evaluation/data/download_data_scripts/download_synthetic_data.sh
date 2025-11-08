#!/bin/bash

set -euo pipefail

# This script downloads the synthetic dataset into
# Data/2DSyntheticDataset/data/train-00000-of-00001.parquet
# using git-lfs to ensure large files are fetched. Override HF_REPO via env.


SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ROOT_DIR=$(dirname "$(dirname "$(dirname "${SCRIPT_DIR}")")")
REPO_DIR="${ROOT_DIR}/Data/2DSyntheticDataset"
TARGET_DIR="${REPO_DIR}/data"


HF_REPO="${HF_REPO:-user074/2DSyntheticDataset}"
PARQUET_PATH="train-00000-of-00001.parquet"

echo "Downloading synthetic dataset from ${HF_REPO}..."

# Require git and git-lfs to ensure large files are pulled
if ! command -v git &> /dev/null; then
  echo "Error: git is required. Please install git and re-run." >&2
  exit 1
fi
if ! command -v git-lfs &> /dev/null; then
  echo "Error: git-lfs is required to download large files. Please install git-lfs and re-run." >&2
  exit 1
fi

git lfs install

# Clone or update the dataset repo directly under Data/2DSyntheticDataset
if [ -d "${REPO_DIR}/.git" ]; then
  echo "Existing synthetic dataset repo found at ${REPO_DIR}. Updating..."
  git -C "${REPO_DIR}" remote set-url origin "https://huggingface.co/datasets/${HF_REPO}" || true
  git -C "${REPO_DIR}" fetch --all --prune
  git -C "${REPO_DIR}" checkout main 2>/dev/null || true
  git -C "${REPO_DIR}" pull --ff-only origin main || git -C "${REPO_DIR}" pull --ff-only || true
  git -C "${REPO_DIR}" lfs pull
else
  echo "Cloning dataset into ${REPO_DIR}..."
  mkdir -p "${ROOT_DIR}/Data"  # Only create parent, not REPO_DIR or TARGET_DIR
  git clone "https://huggingface.co/datasets/${HF_REPO}" "${REPO_DIR}"
  git -C "${REPO_DIR}" lfs pull
fi

# Verify expected parquet exists; do not copy elsewhere
EXPECTED_FILE="${TARGET_DIR}/${PARQUET_PATH}"
if [ -f "${EXPECTED_FILE}" ]; then
  echo "Synthetic dataset ready: ${EXPECTED_FILE}"
else
  echo "Error: ${EXPECTED_FILE} not found after clone. Ensure git-lfs pulled large files and the dataset layout matches. This parquet stores image data required by the evaluator." >&2
  exit 1
fi


