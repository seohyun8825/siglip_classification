#!/usr/bin/env bash
set -euo pipefail

# Hardcoded settings (edit here)
REPO="happy8825/train_ecva_clean_no_tag"
SPLIT="train"
MEDIA_BASE="/hub_data3/seohyun"
OUT_ROOT="/hub_data4/seohyun/video_frame_cache"
CLIP_LEN=16
FRAME_SIZE=224
MODE="center"   # center|uniform|random
MAX_WORKERS=8
OVERWRITE=false

mkdir -p "${OUT_ROOT}"
echo "[cache] Caching ${REPO}:${SPLIT} → ${OUT_ROOT} (len=${CLIP_LEN}, size=${FRAME_SIZE})"

python -u "$(dirname "$0")/../internvideo_cache.py" \
  --repo "${REPO}" \
  --split "${SPLIT}" \
  --media_base "${MEDIA_BASE}" \
  --out_root "${OUT_ROOT}" \
  --clip_len ${CLIP_LEN} \
  --frame_size ${FRAME_SIZE} \
  --mode "${MODE}" \
  --max_workers ${MAX_WORKERS} \
  $( [[ "${OVERWRITE}" == "true" ]] && echo "--overwrite" )

echo "[cache] Done"

