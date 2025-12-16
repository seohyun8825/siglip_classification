#!/usr/bin/env bash
set -euo pipefail

# Edit these and run to push a trained model to the HF Hub.

OUTPUT_DIR="/hub_data3/seohyun/outputs/siglip_ecva"   # parent; script will auto-pick best/last/checkpoint or use config.json in this dir
REPO_NAMESPACE="happy8825"
REPO_MAIN="siglip-ecva-main"      # optional; if empty script derives a name
REPO_BEST="siglip-ecva-best"      # optional; set empty to skip pushing best
PRIVATE=false                      # set true for private repos

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

python3 "$ROOT_DIR/push_model.py" \
  --output_dir "$OUTPUT_DIR" \
  --repo_namespace "$REPO_NAMESPACE" \
  $( [[ -n "$REPO_MAIN" ]] && echo --repo_main "$REPO_MAIN" ) \
  $( [[ -n "$REPO_BEST" ]] && echo --repo_best "$REPO_BEST" ) \
  $( [[ "$PRIVATE" == true ]] && echo --private )

