#!/usr/bin/env bash
set -euo pipefail

# Edit these values and run.

MODEL_DIR="/hub_data3/seohyun/outputs/siglip_ecva"
TEST_REPO="happy8825/siglip_test"
MEDIA_BASE="/hub_data3/seohyun"
SPLIT="train"
OUT=""   # e.g., /hub_data3/seohyun/outputs/siglip_ecva/eval_results.json

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

python3 "$ROOT_DIR/eval_siglip.py" \
  --model_dir "$MODEL_DIR" \
  --test_repo "$TEST_REPO" \
  --media_base "$MEDIA_BASE" \
  --split "$SPLIT" \
  $( [[ -n "$OUT" ]] && echo --out "$OUT" )

