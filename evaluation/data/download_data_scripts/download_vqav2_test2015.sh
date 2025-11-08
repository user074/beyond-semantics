#!/bin/bash

set -euo pipefail

# This script downloads the COCO test2015 images required for VQAv2-style evaluation.
# Images are placed in Data/test2015 to match configs in vqa.yaml and vqa_text.yaml.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ROOT_DIR=$(dirname "$(dirname "$(dirname "${SCRIPT_DIR}")")")
TARGET_DIR="${ROOT_DIR}/Data/vqav2"

mkdir -p "${TARGET_DIR}"

ZIP_PATH="${TARGET_DIR}/test2015.zip"
URL="http://images.cocodataset.org/zips/test2015.zip"

echo "Downloading COCO test2015 images (VQAv2) to ${ZIP_PATH}..."
curl -fL "$URL" -o "$ZIP_PATH"

echo "Unzipping to ${TARGET_DIR}..."
unzip -q "$ZIP_PATH" -d "$TARGET_DIR"

echo "Done. Images should be in ${TARGET_DIR}/test2015"

# Step 3: remove test2015.zip
rm $TARGET_DIR/test2015.zip


