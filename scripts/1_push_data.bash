#!/usr/bin/env bash
set -euo pipefail

# Edit these values and run.

# Source video datasets
TRAIN_DATASET="happy8825/train_ecva_clean_no_tag"
TEST_DATASET="happy8825/valid_ecva_fixed"

# Media base where frames live
MEDIA_BASE="/hub_data3/seohyun"

# Destination HF frame datasets
OUT_TRAIN_REPO="happy8825/siglip_train"
OUT_TEST_REPO="happy8825/siglip_test"

TRAIN_SPLIT="train"
TEST_SPLIT="train"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "[push_data] train_dataset=$TRAIN_DATASET test_dataset=$TEST_DATASET media_base=$MEDIA_BASE"
python3 "$ROOT_DIR/build_hf_dataset.py" \
  --train_dataset "$TRAIN_DATASET" \
  --test_dataset "$TEST_DATASET" \
  --media_base "$MEDIA_BASE" \
  --out_train_repo "$OUT_TRAIN_REPO" \
  --out_test_repo "$OUT_TEST_REPO" \
  --train_split "$TRAIN_SPLIT" \
  --test_split "$TEST_SPLIT"

