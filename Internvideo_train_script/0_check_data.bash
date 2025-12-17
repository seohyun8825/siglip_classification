#!/usr/bin/env bash
set -euo pipefail

# Hardcoded settings (edit here)
TRAIN_REPO="happy8825/train_ecva_clean_no_tag"
EVAL_REPO="happy8825/valid_ecva_fixed"   # or empty to split from train
MEDIA_BASE="/hub_data3/seohyun"

echo "[check] Listing unique video paths from $TRAIN_REPO ..."
python - << 'PY'
from datasets import load_dataset
import sys

train_repo = "happy8825/train_ecva_clean_no_tag"
ds = load_dataset(train_repo, split="train")
vids = set()
for r in ds:
    for v in r.get("videos", []) or []:
        if isinstance(v, str):
            vids.add(v)
print(f"[check] Train unique videos: {len(vids)}")
for i, v in enumerate(sorted(list(vids))[:10]):
    print(f"  {i+1:02d} {v}")
PY

echo "[check] OK"

