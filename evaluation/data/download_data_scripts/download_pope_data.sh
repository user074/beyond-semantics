#!/bin/bash


set -e


# This variable points to /evaluation/data/download, no matter where it is run from
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Go up three levels to reach the project root directory
ROOT_DIR=$(dirname "$(dirname "$(dirname "${SCRIPT_DIR}")")")

# The target path is: [root directory]/Data/
TARGET_DATA_DIR="${ROOT_DIR}/Data/coco"

mkdir -p $TARGET_DATA_DIR


# Step 1: Download val2014.zip
echo "Downloading val2014.zip..."
curl -o $TARGET_DATA_DIR/val2014.zip http://images.cocodataset.org/zips/val2014.zip


# Step 2: Unzip val2014.zip
echo "Unzipping val2014.zip..."
unzip -q $TARGET_DATA_DIR/val2014.zip -d $TARGET_DATA_DIR

# Step 3: remove val2014.zip
rm $TARGET_DATA_DIR/val2014.zip
