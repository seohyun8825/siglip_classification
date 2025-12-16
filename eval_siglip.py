import argparse
import json
import os
from typing import Dict

import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from transformers import AutoImageProcessor, Siglip2ForImageClassification
from PIL import Image
import torch


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
    return p.parse_args()


def _load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def main():
    args = parse_args()
    processor = AutoImageProcessor.from_pretrained(args.model_dir)
    model = Siglip2ForImageClassification.from_pretrained(args.model_dir)
    model.eval()

    ds = load_dataset(args.test_repo, split=args.split)

    ys = []
    ps = []
    for row in ds:
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
            out = model(**{k: v.to(model.device) for k, v in inputs.items()})
            logits = out.logits.detach().cpu().numpy()[0]
            pred = int(np.argmax(logits))

        ys.append(y)
        ps.append(pred)

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
        with open(args.out, "w") as f:
            payload = {**results, "report": report, "confusion_matrix": {"labels": ["normal", "abnormal"], "matrix": cm.tolist()}}
            json.dump(payload, f, indent=2)

    # Optional PNG heatmap
    if args.cm_png:
        try:
            import matplotlib.pyplot as plt
            import numpy as np

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
            fig.savefig(args.cm_png)
            plt.close(fig)
            print(f"Saved confusion matrix PNG to: {args.cm_png}")
        except Exception as e:
            print(f"[WARN] Could not save confusion matrix PNG: {e}")


if __name__ == "__main__":
    main()
