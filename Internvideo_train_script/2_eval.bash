#!/usr/bin/env bash
set -euo pipefail

# Hardcoded settings (edit here)
MODEL_DIR="/hub_data4/seohyun/outputs/internvideo_ecva"  # contains best_head.pt
MEDIA_BASE="/hub_data3/seohyun"
TEST_REPO="happy8825/valid_ecva_fixed"
BATCH_SIZE=1
NUM_WORKERS=2
LIMIT=0
# optional cached frames root (speeds up eval)
CACHE_ROOT="/hub_data4/seohyun/video_frame_cache/valid"
# push settings (set empty to disable push)
PUSH_REPO="happy8825/internvideo_result"
OUT_DIR="${MODEL_DIR}"

python -u "$(dirname "$0")/../internvideo_eval.py" \
  --model_dir "${MODEL_DIR}" \
  --media_base "${MEDIA_BASE}" \
  --test_repo "${TEST_REPO}" \
  --batch_size ${BATCH_SIZE} \
  --num_workers ${NUM_WORKERS} \
  $( [[ ${LIMIT} -gt 0 ]] && echo --limit ${LIMIT} ) \
  --cache_root "${CACHE_ROOT}" \
  $( [[ -n "${PUSH_REPO}" ]] && echo --push_repo "${PUSH_REPO}" ) \
  --out_dir "${OUT_DIR}"

echo "[eval] Done"
