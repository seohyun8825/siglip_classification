#!/usr/bin/env bash
set -euo pipefail

# Edit these values and run.

TRAIN_REPO="happy8825/siglip_train"
MEDIA_BASE="/hub_data4/seohyun"
OUTPUT_DIR="/hub_data3/seohyun/outputs/siglip_ecva"

EPOCHS=1
BATCH_SIZE=32
LR=5e-5
MODEL_NAME="google/siglip2-base-patch16-224"
TRAIN_SPLIT="train"

# Optional eval during training
EVAL_REPO="happy8825/siglip_test"   # set empty to disable
EVAL_SPLIT="train"

# Validation from eval repo (fraction)
VAL_FROM_EVAL_PCT=1.0
SEED=42

# Logging/reporting: set REPORT_TO=wandb to enable W&B
REPORT_TO="wandb"           # none | wandb
WANDB_PROJECT="siglip_ecva" # used when REPORT_TO=wandb
WANDB_RUN_NAME="siglip2-base-ecva"

# Data sampling: all | balanced
DATA_SAMPLE="balanced"

# Validation schedule: <1.0 fraction of epoch (e.g., 0.2), >1.0 every N steps, 1.0 every epoch
EVAL_INTERVAL=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

python3 "$ROOT_DIR/train_siglip.py" \
  --train_repo "$TRAIN_REPO" \
  --media_base "$MEDIA_BASE" \
  --output_dir "$OUTPUT_DIR" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --lr "$LR" \
  --model_name "$MODEL_NAME" \
  --train_split "$TRAIN_SPLIT" \
  $( [[ -n "$EVAL_REPO" ]] && echo --eval_repo "$EVAL_REPO" ) \
  --eval_split "$EVAL_SPLIT" \
  --val_from_eval_pct "$VAL_FROM_EVAL_PCT" \
  --seed "$SEED" \
  --report_to "$REPORT_TO" \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_run_name "$WANDB_RUN_NAME" \
  --data_sample "$DATA_SAMPLE" \
  --eval_interval "$EVAL_INTERVAL"
