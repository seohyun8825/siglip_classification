#!/usr/bin/env bash
set -euo pipefail

# Hardcoded settings (edit here)
TRAIN_REPO="happy8825/train_ecva_clean_no_tag"   # HF dataset with videos column
EVAL_REPO="happy8825/valid_ecva_fixed"          # set empty to disable
EVAL_SPLIT="train"
MEDIA_BASE="/hub_data3/seohyun"

# Model / training hyperparams
OUTPUT_DIR="/hub_data4/seohyun/outputs/internvideo_ecva"
BASE_MODEL="revliter/internvideo_next_large_p14_res224_f16"
CLIP_LEN=16
FRAME_SIZE=224
BATCH_SIZE=1           # memory heavy; increase if fits
EPOCHS=1
LR=1e-3                # higher LR for head-only training
NUM_WORKERS=4
VAL_RATIO=0.1              # used only when EVAL_REPO empty
EVAL_INTERVAL=0            # 0=disabled; <1=fraction of epoch; >=1=fixed steps
HIDDEN="1024,512"     # classification head hidden layers
DROPOUT=0.1
FREEZE_BACKBONE=false   # train backbone+head
REPORT_TO=wandb         # none|wandb
DISABLE_FLASH_ATTN=false
PRECISION=bf16          # bf16 recommended on A6000
WEIGHT_DECAY=0.05
WARMUP_RATIO=0.05
GRAD_CLIP=1.0
DETECT_NAN=true
DEBUG_BATCHES=3
READER=decord            # auto|opencv|decord
MP_START=spawn           # default|spawn|fork|forkserver
PROGRESS_EVERY=50

# Cached frames (optional but recommended)
CACHE_ROOT_TRAIN="/hub_data4/seohyun/video_frame_cache/train"
CACHE_ROOT_EVAL="/hub_data4/seohyun/video_frame_cache/valid"

mkdir -p "${OUTPUT_DIR}"
echo "[train] Output: ${OUTPUT_DIR}"

python -u "$(dirname "$0")/../internvideo_train.py" \
  --train_repo "${TRAIN_REPO}" \
  --eval_repo "${EVAL_REPO}" \
  --eval_split "${EVAL_SPLIT}" \
  --media_base "${MEDIA_BASE}" \
  --output_dir "${OUTPUT_DIR}" \
  --base_model "${BASE_MODEL}" \
  --clip_len ${CLIP_LEN} \
  --frame_size ${FRAME_SIZE} \
  --batch_size ${BATCH_SIZE} \
  --epochs ${EPOCHS} \
  --lr ${LR} \
  --num_workers ${NUM_WORKERS} \
  --val_ratio ${VAL_RATIO} \
  --eval_interval ${EVAL_INTERVAL} \
  --hidden "${HIDDEN}" \
  --dropout ${DROPOUT} \
  $( [[ "${FREEZE_BACKBONE}" == "true" ]] && echo "--freeze_backbone" ) \
  $( [[ "${DISABLE_FLASH_ATTN}" == "true" ]] && echo "--disable_flash_attn" ) \
  --precision ${PRECISION} \
  --weight_decay ${WEIGHT_DECAY} \
  --warmup_ratio ${WARMUP_RATIO} \
  --grad_clip ${GRAD_CLIP} \
  $( [[ "${DETECT_NAN}" == "true" ]] && echo "--detect_nan" ) \
  --debug_batches ${DEBUG_BATCHES} \
  --reader ${READER} \
  --mp_start ${MP_START} \
  --progress_every ${PROGRESS_EVERY} \
  --cache_root_train "${CACHE_ROOT_TRAIN}" \
  --cache_root_eval "${CACHE_ROOT_EVAL}" \
  --report_to ${REPORT_TO}

echo "[train] Done"
