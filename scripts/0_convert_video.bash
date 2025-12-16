#!/usr/bin/env bash
set -euo pipefail

# Edit the values below and run this script.

# Required
DATASET="happy8825/train_ecva_clean_no_tag"
MEDIA_BASE="/hub_data3/seohyun"

# Optional
FPS="1.0"           # target FPS for sampling
OVERWRITE=false      # set to true to re-generate existing frames

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "[convert_video] dataset=$DATASET media_base=$MEDIA_BASE fps=$FPS overwrite=$OVERWRITE"
python3 "$ROOT_DIR/convert_video.py" \
  --dataset "$DATASET" \
  --media_base "$MEDIA_BASE" \
  --fps "$FPS" \
  $( [[ "$OVERWRITE" == true ]] && echo "--overwrite" )

