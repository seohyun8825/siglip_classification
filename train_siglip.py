import argparse
import os
import json
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Dict, List

import numpy as np
from datasets import load_dataset
from transformers import (
    AutoImageProcessor,
    Siglip2ForImageClassification,
    TrainingArguments,
    Trainer,
)
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score


ID2LABEL = {0: "normal", 1: "abnormal"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


def parse_args():
    p = argparse.ArgumentParser(description="Train SigLIP2 image classifier on frame dataset.")
    p.add_argument("--train_repo", required=True, help="HF dataset repo id for frames (e.g., happy8825/siglip_train)")
    p.add_argument("--media_base", required=True, help="Absolute base dir prefix to resolve images column")
    p.add_argument("--output_dir", required=True, help="Directory to save model outputs")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--model_name", default="google/siglip2-base-patch16-224")
    p.add_argument("--train_split", default="train")
    p.add_argument("--eval_repo", default=None, help="Optional eval repo id; if omitted, skip eval during training")
    p.add_argument("--eval_split", default="train")
    p.add_argument("--val_from_eval_pct", type=float, default=0.2, help="Use this fraction of eval_repo as validation set (default: 0.2). Set 0 to disable.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--report_to", choices=["none", "wandb"], default="none", help="Where to report logs (default: none).")
    p.add_argument("--wandb_project", default=None, help="W&B project name when report_to=wandb")
    p.add_argument("--wandb_run_name", default=None, help="W&B run name when report_to=wandb")
    return p.parse_args()


def _path(row, media_base: str) -> str:
    rel = row["images"]
    return os.path.join(media_base, rel)


def _label(row) -> int:
    gt = row["gt"]
    if isinstance(gt, str):
        return LABEL2ID.get(gt, 0)
    return int(gt)


def _load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Optional W&B setup
    if args.report_to == "wandb":
        if args.wandb_project:
            os.environ["WANDB_PROJECT"] = args.wandb_project
        if args.wandb_run_name:
            os.environ["WANDB_NAME"] = args.wandb_run_name

    processor = AutoImageProcessor.from_pretrained(args.model_name)
    model = Siglip2ForImageClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    train_ds = load_dataset(args.train_repo, split=args.train_split)
    eval_ds = load_dataset(args.eval_repo, split=args.eval_split) if args.eval_repo else None
    if eval_ds is not None and args.val_from_eval_pct and args.val_from_eval_pct > 0:
        try:
            split_dd = eval_ds.train_test_split(test_size=args.val_from_eval_pct, seed=args.seed)
            eval_ds = split_dd["test"]  # validation portion
            print(f"[train] Using {len(eval_ds)} samples for validation from eval_repo ({args.val_from_eval_pct*100:.1f}%).")
        except Exception as e:
            print(f"[WARN] Could not split eval_repo for validation: {e}")

    def transform(batch):
        images = []
        labels = []
        for rel, gt in zip(batch["images"], batch["gt"]):
            path = os.path.join(args.media_base, rel)
            try:
                img = _load_image(path)
            except Exception:
                # Replace with black image if missing; keep label to avoid shape mismatch
                img = Image.new("RGB", (224, 224), (0, 0, 0))
            images.append(img)
            labels.append(LABEL2ID.get(gt, int(gt) if isinstance(gt, (int, np.integer)) else 0))
        proc = processor(images=images, return_tensors="pt")
        proc["labels"] = labels
        return proc

    columns = ["pixel_values", "labels"]
    train_t = train_ds.with_transform(transform).remove_columns([c for c in train_ds.column_names if c not in ("images", "gt")])
    eval_t = None
    if eval_ds is not None:
        eval_t = eval_ds.with_transform(transform).remove_columns([c for c in eval_ds.column_names if c not in ("images", "gt")])

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="binary", pos_label=1)
        return {"accuracy": acc, "f1": f1}

    args_tr = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=50,
        evaluation_strategy="epoch" if eval_t is not None else "no",
        save_strategy="epoch" if eval_t is not None else "epoch",
        save_total_limit=2,
        load_best_model_at_end=True if eval_t is not None else False,
        metric_for_best_model="f1",
        report_to=([] if args.report_to == "none" else [args.report_to]),
        remove_unused_columns=False,
        dataloader_num_workers=4,
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=args_tr,
        train_dataset=train_t,
        eval_dataset=eval_t,
        tokenizer=processor,
        compute_metrics=compute_metrics if eval_t is not None else None,
    )

    trainer.train()

    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)

    if eval_t is not None:
        metrics = trainer.evaluate(eval_dataset=eval_t)
        print(metrics)
        with open(os.path.join(args.output_dir, "train_eval_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
