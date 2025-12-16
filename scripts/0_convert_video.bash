#!/usr/bin/env bash
set -euo pipefail

# Edit the values below and run this script.

# Required
DATASET="happy8825/train_ecva_clean_no_tag"
MEDIA_BASE="/hub_data3/seohyun"

# Optional
FPS="0.5"           # target FPS for sampling
OVERWRITE=false      # set to true to re-generate existing frames
MAX_WORKERS=60       # parallel extraction workers

# Skip early seconds (frames with sec < MIN_SEC are ignored during extraction)
MIN_SEC=2

# Output root for classification frames (absolute)
# e.g., /hub_data4/seohyun/ecva
OUTPUT_ROOT="/hub_data4/seohyun/ecva"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "[convert_video] dataset=$DATASET media_base=$MEDIA_BASE fps=$FPS overwrite=$OVERWRITE"
python3 "$ROOT_DIR/convert_video.py" \
  --dataset "$DATASET" \
  --media_base "$MEDIA_BASE" \
  --fps "$FPS" \
  $( [[ "$OVERWRITE" == true ]] && echo "--overwrite" ) \
  --output_root "$OUTPUT_ROOT" \
  --max_workers "$MAX_WORKERS" \
  --min_sec "$MIN_SEC"
