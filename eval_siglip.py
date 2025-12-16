import argparse
import json
import os
from typing import Dict, List
from datetime import datetime
import random
import time

import numpy as np
from datasets import load_dataset, Dataset, Image
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from transformers import AutoImageProcessor, AutoModelForImageClassification
import numpy as np
from PIL import Image
import torch
from huggingface_hub import HfApi


ID2LABEL = {0: "normal", 1: "abnormal"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained SigLIP2 classifier on a frame dataset.")
    p.add_argument("--model_dir", required=True)
    p.add_argument("--test_repo", required=True)
    p.add_argument("--media_base", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--out", default=None, help="Optional path to write eval_results.json (includes confusion matrix)")
    p.add_argument("--cm_png", default=None, help="Optional path to save confusion matrix heatmap PNG")
    p.add_argument("--visualize_push", type=int, default=0, help="If >0, randomly sample this many examples and push a visualization dataset to HF with images + predictions.")
    p.add_argument("--vis_repo_namespace", default="happy8825", help="HF namespace (user or org) for visualization repo")
    p.add_argument("--vis_repo_prefix", default="siglip_classification_result", help="Repo name prefix; timestamp is appended")
    p.add_argument("--vis_seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="Device to run eval on (default: auto)")
    return p.parse_args()


def _load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def main():
    args = parse_args()
    print("[eval] Loading processor...")
    # Load processor with fallback to base checkpoint if needed
    try:
        processor = AutoImageProcessor.from_pretrained(args.model_dir)
    except Exception as e:
        print(f"[WARN] Failed to load processor from {args.model_dir}: {e}")
        processor = AutoImageProcessor.from_pretrained("google/siglip2-base-patch16-224")

    print("[eval] Loading model...")
    # Load model using AutoModelForImageClassification so saved config determines class (siglip/siglip2/etc.)
    try:
        model = AutoModelForImageClassification.from_pretrained(args.model_dir)
    except Exception as e:
        print(f"[WARN] Failed to load model from {args.model_dir}: {e}")
        print("[WARN] Retrying with ignore_mismatched_sizes=True ...")
        model = AutoModelForImageClassification.from_pretrained(args.model_dir, ignore_mismatched_sizes=True)
    model.eval()

    # Device selection
    device = (
        "cuda" if (args.device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu"
    )
    model.to(device)
    print(f"[eval] Model moved to device: {device}")

    print(f"[eval] Loading dataset: {args.test_repo} split={args.split}")
    ds = load_dataset(args.test_repo, split=args.split)
    print(f"[eval] Dataset loaded: {len(ds)} samples")

    ys = []
    ps = []
    t0 = time.time()
    last = t0
    for i, row in enumerate(ds):
        rel = row["images"]
        gt = row["gt"]
        y = LABEL2ID.get(gt, int(gt) if isinstance(gt, (int, np.integer)) else 0)
        path = os.path.join(args.media_base, rel)
        try:
            img = _load_image(path)
        except Exception:
            img = Image.new("RGB", (224, 224), (0, 0, 0))
        inputs = processor(images=img, return_tensors="pt")
        with torch.no_grad():
            out = model(**{k: v.to(device) for k, v in inputs.items()})
            logits = out.logits.detach().cpu().numpy()[0]
            pred = int(np.argmax(logits))

        ys.append(y)
        ps.append(pred)

        if (i + 1) % 200 == 0 or (time.time() - last) > 5:
            pct = (i + 1) / max(1, len(ds)) * 100.0
            print(f"[eval] Progress: {i+1}/{len(ds)} ({pct:.1f}%)")
            last = time.time()

    acc = accuracy_score(ys, ps)
    f1 = f1_score(ys, ps, average="binary", pos_label=1)
    report = classification_report(ys, ps, target_names=["normal", "abnormal"], digits=4)
    cm = confusion_matrix(ys, ps, labels=[0, 1])

    results: Dict[str, float] = {"accuracy": acc, "f1": f1}
    print("Evaluation Results:")
    print(results)
    print("Classification Report:")
    print(report)
    print("Confusion Matrix (rows=true [normal, abnormal], cols=pred [normal, abnormal]):")
    print(cm)

    if args.out:
        try:
            out_dir = os.path.dirname(args.out)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
        except Exception:
            pass
        with open(args.out, "w") as f:
            payload = {**results, "report": report, "confusion_matrix": {"labels": ["normal", "abnormal"], "matrix": cm.tolist()}}
            json.dump(payload, f, indent=2)

    # Optional PNG heatmap
    cm_png_path = args.cm_png
    if args.cm_png or args.visualize_push > 0:
        if cm_png_path is None:
            cm_png_path = os.path.join(args.model_dir, "confusion_matrix.png")
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=150)
            im = ax.imshow(cm, cmap="Blues")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["normal", "abnormal"])
            ax.set_yticks([0, 1])
            ax.set_yticklabels(["normal", "abnormal"])
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            # Ensure parent directory exists
            try:
                cm_dir = os.path.dirname(cm_png_path)
                if cm_dir:
                    os.makedirs(cm_dir, exist_ok=True)
            except Exception:
                pass
            fig.savefig(cm_png_path)
            plt.close(fig)
            print(f"Saved confusion matrix PNG to: {cm_png_path}")
        except Exception as e:
            print(f"[WARN] Could not save confusion matrix PNG: {e}")

    # Optional visualization push to HF
    if args.visualize_push and args.visualize_push > 0:
        n = int(args.visualize_push)
        total = len(ys)
        idxs = list(range(total))
        random.Random(args.vis_seed).shuffle(idxs)
        idxs = idxs[: min(n, total)]

        # Build rows with image binary + predictions
        rows: List[Dict] = []
        for i in idxs:
            rel = ds[i]["images"]
            gt = ds[i]["gt"]
            pred_id = int(ps[i])
            pred = ID2LABEL.get(pred_id, str(pred_id))
            abs_img = os.path.join(args.media_base, rel)
            rows.append({
                "image": abs_img,
                "images": rel,
                "gt": gt,
                "pred": pred,
                "correct": bool(LABEL2ID.get(gt, gt) == pred_id),
            })

        vis_repo_id = f"{args.vis_repo_namespace}/{args.vis_repo_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        api = HfApi()
        api.create_repo(repo_id=vis_repo_id, repo_type="dataset", exist_ok=True)
        ds_vis = Dataset.from_list(rows)
        ds_vis = ds_vis.cast_column("image", Image())
        print(f"[vis] Pushing {len(ds_vis)} rows with images to {vis_repo_id} ...")
        ds_vis.push_to_hub(vis_repo_id)

        # Upload README with metrics and (optional) confusion matrix image
        readme = (
            f"# SigLIP Classification Eval\n\n"
            f"- Model: {args.model_dir}\n\n"
            f"- Test repo: {args.test_repo} (split: {args.split})\n\n"
            f"- Samples visualized: {len(ds_vis)} of {total}\n\n"
            f"## Metrics\n\n"
            f"- accuracy: {acc:.4f}\n\n"
            f"- f1: {f1:.4f}\n\n"
            f"- total: {total}, correct: {int(round(acc*total))}\n\n"
        )
        readme_path = os.path.join(args.model_dir, "VIS_README.md")
        with open(readme_path, "w") as f:
            f.write(readme)
            if cm_png_path and os.path.exists(cm_png_path):
                f.write("\n## Confusion Matrix\n\n")
                f.write("![](confusion_matrix.png)\n")

        # Upload files
        api.upload_file(path_or_fileobj=readme_path, path_in_repo="README.md", repo_id=vis_repo_id, repo_type="dataset")
        if cm_png_path and os.path.exists(cm_png_path):
            api.upload_file(path_or_fileobj=cm_png_path, path_in_repo="confusion_matrix.png", repo_id=vis_repo_id, repo_type="dataset")


if __name__ == "__main__":
    main()
