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
    AutoModelForImageClassification,
    AutoModel,
    TrainingArguments,
    Trainer,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    p.add_argument("--data_sample", choices=["all", "balanced"], default="all", help="Dataset sampling: 'all' uses all frames; 'balanced' downsamples the majority class evenly across source videos to match the minority count.")
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
    class VisionClassifier(nn.Module):
        def __init__(self, backbone_name: str, num_labels: int = 2):
            super().__init__()
            self.vision = AutoModel.from_pretrained(backbone_name)
            hidden = getattr(self.vision.config, "hidden_size", None) or getattr(self.vision.config, "vision_config", None).hidden_size
            self.classifier = nn.Linear(hidden, num_labels)
            self.num_labels = num_labels

        def forward(self, pixel_values=None, labels=None, **kwargs):
            out = self.vision(pixel_values=pixel_values, **kwargs)
            pooled = getattr(out, "pooler_output", None)
            if pooled is None:
                # Fallback to CLS token
                pooled = out.last_hidden_state[:, 0]
            logits = self.classifier(pooled)
            loss = None
            if labels is not None:
                loss = F.cross_entropy(logits, labels.long())
            return type("Output", (), {"loss": loss, "logits": logits})

    # Load model with robust fallbacks
    model = None
    try:
        model = AutoModelForImageClassification.from_pretrained(
            args.model_name,
            num_labels=2,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )
    except Exception as e:
        errmsg = str(e)
        print(f"[WARN] AutoModelForImageClassification failed: {errmsg}")
        # Try precise wrapper to preserve pre-trained patch embedding
        try:
            print("[train] Falling back to Vision backbone + custom classifier head.")
            model = VisionClassifier(args.model_name, num_labels=2)
        except Exception as e2:
            print(f"[WARN] Vision backbone fallback failed: {e2}")
            print("[train] Falling back to ignore_mismatched_sizes=True on AutoModelForImageClassification (less ideal).")
            model = AutoModelForImageClassification.from_pretrained(
                args.model_name,
                num_labels=2,
                id2label=ID2LABEL,
                label2id=LABEL2ID,
                ignore_mismatched_sizes=True,
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

    # Optional balanced sampling for train set
    if args.data_sample == "balanced":
        import random
        rng = random.Random(args.seed)

        images = list(train_ds["images"])  # type: ignore
        gts = list(train_ds["gt"])        # type: ignore

        def label_id(gt):
            return LABEL2ID.get(gt, int(gt) if isinstance(gt, (int, np.integer)) else 0)

        labels = [label_id(gt) for gt in gts]

        idx_norm = [i for i, y in enumerate(labels) if y == 0]
        idx_abn = [i for i, y in enumerate(labels) if y == 1]

        n_norm, n_abn = len(idx_norm), len(idx_abn)
        print(f"[train] Class counts before balance: normal={n_norm}, abnormal={n_abn}")

        if n_norm != n_abn and n_norm > 0 and n_abn > 0:
            # Determine majority/minority
            if n_norm > n_abn:
                majority_label, minority_label = 0, 1
                majority_indices, minority_indices = idx_norm, idx_abn
            else:
                majority_label, minority_label = 1, 0
                majority_indices, minority_indices = idx_abn, idx_norm

            target = len(minority_indices)

            # Group majority indices by (class_dir + video_stem)
            from pathlib import PurePosixPath

            def video_key(rel_path: str) -> str:
                p = PurePosixPath(rel_path)
                class_dir = p.parts[1] if len(p.parts) > 1 else ""
                base = p.name.rsplit(".", 1)[0]
                parts = base.rsplit("_", 2)
                stem = parts[0] if len(parts) == 3 else base
                return f"{class_dir}/{stem}"

            groups: Dict[str, List[int]] = {}
            for i in majority_indices:
                key = video_key(images[i])
                groups.setdefault(key, []).append(i)
            # Shuffle indices within each group
            for g in groups.values():
                rng.shuffle(g)
            # Shuffle group order for fairness
            group_keys = list(groups.keys())
            rng.shuffle(group_keys)

            selected_major: List[int] = []
            # Round-robin pick across groups until target reached
            while len(selected_major) < target and group_keys:
                progressed = False
                for k in list(group_keys):
                    if len(selected_major) >= target:
                        break
                    g = groups.get(k, [])
                    if g:
                        selected_major.append(g.pop())
                        progressed = True
                    # Drop empty groups to speed up
                    if not g:
                        groups.pop(k, None)
                        group_keys.remove(k)
                if not progressed:
                    break

            selected = minority_indices + selected_major
            print(f"[train] Balanced selection: {len(selected_major)} from majority, {len(minority_indices)} from minority → total {len(selected)}")
            # Apply selection to train_ds
            train_ds = train_ds.select(sorted(selected))
        else:
            print("[train] Dataset already balanced or empty; using all samples.")

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

    # Build TrainingArguments with version-robust kwargs
    ta_fields = getattr(TrainingArguments, "__dataclass_fields__", {}) or {}
    ta_kwargs = {"output_dir": args.output_dir}

    def set_if(field: str, value):
        if field in ta_fields:
            ta_kwargs[field] = value

    set_if("per_device_train_batch_size", args.batch_size)
    set_if("per_device_eval_batch_size", args.batch_size)
    set_if("num_train_epochs", args.epochs)
    set_if("learning_rate", args.lr)
    set_if("logging_steps", 50)
    set_if("save_total_limit", 2)
    set_if("remove_unused_columns", False)
    set_if("dataloader_num_workers", 4)
    set_if("fp16", True)

    # Evaluation/save strategy compatibility (handle older transformers versions)
    eval_enabled = False
    if eval_t is not None:
        if "evaluation_strategy" in ta_fields:
            ta_kwargs["evaluation_strategy"] = "epoch"
            eval_enabled = True
        elif "evaluate_during_training" in ta_fields:
            ta_kwargs["evaluate_during_training"] = True
            eval_enabled = True
        if "save_strategy" in ta_fields:
            ta_kwargs["save_strategy"] = "epoch"
        # Best model only if evaluation really enabled
        if eval_enabled:
            set_if("load_best_model_at_end", True)
            set_if("metric_for_best_model", "f1")
    else:
        if "evaluation_strategy" in ta_fields:
            ta_kwargs["evaluation_strategy"] = "no"
        elif "evaluate_during_training" in ta_fields:
            ta_kwargs["evaluate_during_training"] = False

    # Reporting backends (e.g., wandb)
    if args.report_to != "none" and "report_to" in ta_fields:
        ta_kwargs["report_to"] = [args.report_to]

    args_tr = TrainingArguments(**ta_kwargs)

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
