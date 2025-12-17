#!/usr/bin/env bash
set -euo pipefail

# Test/valid cache settings
REPO="happy8825/valid_ecva_fixed"
SPLIT="train"
MEDIA_BASE="/hub_data3/seohyun"
OUT_ROOT="/hub_data4/seohyun/video_frame_cache/valid"
CLIP_LEN=16
FRAME_SIZE=224
MODE="center"   # center clip for eval consistency
MAX_WORKERS=80
OVERWRITE=false
FORMAT="npz"            # npz|jpg
BALANCED_TOTAL=0        # e.g. 100 -> 50 normal + 50 abnormal

mkdir -p "${OUT_ROOT}"
echo "[cache-valid] ${REPO}:${SPLIT} → ${OUT_ROOT} (len=${CLIP_LEN}, size=${FRAME_SIZE}, mode=${MODE})"
python -u "$(dirname "$0")/../internvideo_cache.py" \
  --repo "${REPO}" \
  --split "${SPLIT}" \
  --media_base "${MEDIA_BASE}" \
  --out_root "${OUT_ROOT}" \
  --clip_len ${CLIP_LEN} \
  --frame_size ${FRAME_SIZE} \
  --mode "${MODE}" \
  --max_workers ${MAX_WORKERS} \
  --format "${FORMAT}" \
  $( [[ ${BALANCED_TOTAL} -gt 0 ]] && echo --balanced_total ${BALANCED_TOTAL} ) \
  $( [[ "${OVERWRITE}" == "true" ]] && echo "--overwrite" )
echo "[cache-valid] Done"
