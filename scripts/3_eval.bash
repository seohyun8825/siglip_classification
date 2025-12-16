#!/usr/bin/env bash
set -euo pipefail

# Edit these values and run.

MODEL_DIR="happy8825/siglip-ecva-main"
TEST_REPO="happy8825/siglip_test"
MEDIA_BASE="/hub_data4/seohyun"
SPLIT="train"
OUT="/hub_data4/seohyun/outputs/siglip_ecva/eval_results.json"   # e.g., /hub_data4/seohyun/outputs/siglip_ecva/eval_results.json
CM_PNG="/hub_data4/seohyun/outputs/siglip_ecva/confusion_matrix.png"  # e.g., /hub_data4/seohyun/outputs/siglip_ecva/confusion_matrix.png

# Optional: push visualization samples to HF (0 to disable)
VISUALIZE_PUSH=50
VIS_REPO_NAMESPACE="happy8825"
VIS_REPO_PREFIX="siglip_classification_result"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

python3 "$ROOT_DIR/eval_siglip.py" \
  --model_dir "$MODEL_DIR" \
  --test_repo "$TEST_REPO" \
  --media_base "$MEDIA_BASE" \
  --split "$SPLIT" \
  $( [[ -n "$OUT" ]] && echo --out "$OUT" ) \
  $( [[ -n "$CM_PNG" ]] && echo --cm_png "$CM_PNG" ) \
  --visualize_push "$VISUALIZE_PUSH" \
  --vis_repo_namespace "$VIS_REPO_NAMESPACE" \
  --vis_repo_prefix "$VIS_REPO_PREFIX"
