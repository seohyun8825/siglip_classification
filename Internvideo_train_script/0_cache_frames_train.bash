#!/usr/bin/env bash
set -euo pipefail

# Train cache settings
REPO="happy8825/train_ecva_clean_no_tag"
SPLIT="train"
MEDIA_BASE="/hub_data3/seohyun"
OUT_ROOT="/hub_data4/seohyun/video_frame_cache/train"
CLIP_LEN=16
FRAME_SIZE=224
MODE="random"   # random for train diversity
MAX_WORKERS=80
OVERWRITE=false
# storage format and balanced toy selection
FORMAT="npz"            # npz|jpg
BALANCED_TOTAL=0        # e.g. 100 -> 50 normal + 50 abnormal

mkdir -p "${OUT_ROOT}"
echo "[cache-train] ${REPO}:${SPLIT} → ${OUT_ROOT} (len=${CLIP_LEN}, size=${FRAME_SIZE}, mode=${MODE})"
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
echo "[cache-train] Done"
