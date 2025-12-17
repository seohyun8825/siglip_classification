#!/usr/bin/env bash
set -euo pipefail

# Quick end-to-end demo with limits

# 0) Cache a small subset for train
echo "[demo] Caching small train subset"
REPO_TR="happy8825/train_ecva_clean_no_tag"
MEDIA_BASE="/hub_data3/seohyun"
OUT_TR="/hub_data4/seohyun/video_frame_cache/train"
# limits for quick demo
LIMIT_TR=2160
FORMAT=npz
BALANCED_TOTAL=2160   # balanced subset (half per class)
python -u "$(dirname "$0")/../internvideo_cache.py" \
  --repo "${REPO_TR}" --split train --media_base "${MEDIA_BASE}" \
  --out_root "${OUT_TR}" --clip_len 16 --frame_size 224 --mode random \
  --max_workers 20 --progress_every 1 --limit ${LIMIT_TR} --limit_mode head \
  --format ${FORMAT} --balanced_total ${BALANCED_TOTAL}

# 0b) Cache a small subset for valid
echo "[demo] Caching small valid subset"
REPO_VA="happy8825/valid_ecva_fixed"
OUT_VA="/hub_data4/seohyun/video_frame_cache/valid"
# For eval, use full valid set (no limit)
LIMIT_VA=0
python -u "$(dirname "$0")/../internvideo_cache.py" \
  --repo "${REPO_VA}" --split train --media_base "${MEDIA_BASE}" \
  --out_root "${OUT_VA}" --clip_len 16 --frame_size 224 --mode center \
  --max_workers 20 --progress_every 10 \
  --format ${FORMAT}

# 1) Train (backbone+head) on small limit by pointing to caches and overriding repo limits via precision/clip
echo "[demo] Training (short run)"
# Optionally freeze backbone for head-only training
FREEZE_BACKBONE=${FREEZE_BACKBONE:-false}   # set to false to train backbone+head
# auto-tune hyperparams by freeze state
if [[ "${FREEZE_BACKBONE}" == "true" ]]; then
  LR_DEFAULT=1e-3
  WD_DEFAULT=0.0
  HIDDEN_DEFAULT="512"
  DROPOUT_DEFAULT=0.0
  EPOCHS_DEFAULT=1
else
  LR_DEFAULT=5e-5
  WD_DEFAULT=0.05
  HIDDEN_DEFAULT="1024,512"
  DROPOUT_DEFAULT=0.1
  EPOCHS_DEFAULT=1
fi
# Verify a few samples (labels/paths + one forward) before training
VERIFY_INPUT=${VERIFY_INPUT:-64}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3} python -u "$(dirname "$0")/../internvideo_train.py" \
  --train_repo "${REPO_TR}" --eval_repo "${REPO_VA}" --eval_split train \
  --media_base "${MEDIA_BASE}" \
  --output_dir "/hub_data4/seohyun/outputs/internvideo_ecva_demo_all_layer_trainable" \
  --base_model "revliter/internvideo_next_large_p14_res224_f16" \
  --clip_len 16 --frame_size 224 --batch_size 8 --epochs ${EPOCHS_DEFAULT} \
  --lr ${LR_DEFAULT} --num_workers 2 --val_ratio 0.0 --eval_interval 20 \
  --report_to wandb --wandb_project "${WANDB_PROJECT:-internvideo_ecva}" \
  --hidden "${HIDDEN_DEFAULT}" --dropout ${DROPOUT_DEFAULT} \
  --precision bf16 --weight_decay ${WD_DEFAULT} --warmup_ratio 0.05 --grad_clip 1.0 \
  --reader decord --mp_start spawn --progress_every 20 \
  --cache_root_train "${OUT_TR}" --cache_root_eval "${OUT_VA}" \
  --limit_train ${LIMIT_TR} --limit_eval 0 \
  $( [[ "${FREEZE_BACKBONE}" == "true" ]] && echo "--freeze_backbone" ) \
  --train_manifest "${OUT_TR}/_manifest_${REPO_TR//\//_}_train.jsonl" \
  --verify_input ${VERIFY_INPUT}

# 2) Eval on limited set using cached frames, then push results to HF dataset
echo "[demo] Evaluating (limit) + pushing results"
PUSH_REPO="happy8825/internvideo_result"
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3} python -u "$(dirname "$0")/../internvideo_eval.py" \
  --model_dir "/hub_data4/seohyun/outputs/internvideo_ecva_demo" \
  --media_base "${MEDIA_BASE}" \
  --test_repo "${REPO_VA}" \
  --batch_size 1 --num_workers 2 \
  --cache_root "${OUT_VA}" --precision bf16 --log_each \
  --push_repo "${PUSH_REPO}" \
  --out_dir "/hub_data4/seohyun/outputs/internvideo_ecva_demo"

echo "[demo] Done"
