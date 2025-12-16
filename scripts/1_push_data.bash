#!/usr/bin/env bash
set -euo pipefail

# Edit these values and run.

# Source video datasets
TRAIN_DATASET="happy8825/train_ecva_clean_no_tag"
TEST_DATASET="happy8825/valid_ecva_fixed"

# Media base where frames live (classification images root)
# NOTE: set to /hub_data4/seohyun to match 0_convert output_root parent
MEDIA_BASE="/hub_data4/seohyun"

# Destination HF frame datasets
OUT_TRAIN_REPO="happy8825/siglip_train"
OUT_TEST_REPO="happy8825/siglip_test"
# Optional: also push with actual image binary for debugging
OUT_TEST_REPO_WITH_IMAGE="happy8825/siglip_test_with_image"

TRAIN_SPLIT="train"
TEST_SPLIT="train"

# If your source videos are under a different base (e.g., /hub_data3), set here
VIDEO_MEDIA_BASE="/hub_data3/seohyun"

# Overwrite existing _MID.png when regenerating
OVERWRITE_EVAL_FRAMES=false
MIN_SEC_FILTER=2   # exclude frames with sec < this for TRAIN frame dataset

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
  --test_split "$TEST_SPLIT" \
  --video_media_base "$VIDEO_MEDIA_BASE" \
  $( [[ -n "$OUT_TEST_REPO_WITH_IMAGE" ]] && echo --out_test_repo_with_image "$OUT_TEST_REPO_WITH_IMAGE" ) \
  $( [[ "$OVERWRITE_EVAL_FRAMES" == true ]] && echo --overwrite_eval_frames ) \
  --min_sec_filter "$MIN_SEC_FILTER"
