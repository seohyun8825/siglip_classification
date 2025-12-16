import argparse
import json
import os
from typing import Dict, List
from datetime import datetime
import random
import time
import tempfile


def _is_valid_local_model_dir(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    has_config = os.path.exists(os.path.join(path, "config.json"))
    has_weights = any(
        os.path.exists(os.path.join(path, f))
        for f in (
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
            "model.safetensors",
            "tf_model.h5",
            "model.ckpt.index",
            "flax_model.msgpack",
        )
    )
    return has_config and has_weights

import numpy as np
from datasets import load_dataset, Dataset, Image as HFImage
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from transformers import AutoImageProcessor, AutoModelForImageClassification
import numpy as np
from PIL import Image as PILImage
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


def _load_image(path: str) -> PILImage.Image:
    return PILImage.open(path).convert("RGB")


def main():
    args = parse_args()
    # Sanitize and auto-resolve model_dir (handle leading/trailing spaces, best/last subdirs)
    raw_model_dir = (args.model_dir or "").strip()
    candidate_abs = os.path.abspath(os.path.expanduser(raw_model_dir))
    # Consider it local only if it looks like a valid model dir
    is_local_dir = _is_valid_local_model_dir(candidate_abs)
    chdir_tmp = None  # if set, we'll temporarily chdir here to avoid local shadowing
    if is_local_dir:
        args.model_dir = candidate_abs
    else:
        # Try to recover if user passed parent dir; prefer best/, then last/, then latest checkpoint
        parent = os.path.abspath(os.path.expanduser(raw_model_dir))
        candidates = []
        for sub in ("best", "last"):
            p = os.path.join(parent, sub)
            if _is_valid_local_model_dir(p):
                candidates.append(p)
        # Fallback to newest checkpoint-*
        try:
            ckpts = []
            if os.path.isdir(parent):
                for name in os.listdir(parent):
                    if name.startswith("checkpoint-"):
                        p = os.path.join(parent, name)
                        if _is_valid_local_model_dir(p):
                            try:
                                step = int(name.split("-", 1)[1])
                            except Exception:
                                step = -1
                            ckpts.append((step, p))
            if ckpts:
                ckpts.sort(key=lambda x: x[0])
                candidates.append(ckpts[-1][1])
        except Exception:
            pass
        if candidates:
            print(f"[eval] Using auto-resolved model_dir: {candidates[0]}")
            args.model_dir = candidates[0]
            is_local_dir = True
        else:
            # As a last fallback, handle potential Hub repo IDs. If a local folder with the same
            # name exists but is NOT a valid model, force a Hub download to avoid being treated as local.
            looks_like_repo_id = "/" in raw_model_dir and not raw_model_dir.startswith("/")
            if os.path.isdir(raw_model_dir) and not _is_valid_local_model_dir(raw_model_dir) and looks_like_repo_id:
                # A local folder with the same name exists but is not a model. We want to load the
                # remote repo without downloading the entire snapshot. Workaround: temporarily chdir
                # to a directory where that relative path doesn't exist so Transformers treats it as a repo id.
                print(
                    f"[eval] Detected local folder shadowing repo id '{raw_model_dir}'. "
                    "Temporarily changing CWD to avoid local shadowing and force remote load."
                )
                args.model_dir = raw_model_dir
                is_local_dir = False
                chdir_tmp = "/"
            else:
                # Keep the raw string (may be a valid repo id); mark as non-local
                args.model_dir = raw_model_dir
                is_local_dir = False
    print("[eval] Loading processor...")
    # Load processor with fallback to base checkpoint if needed. If chdir_tmp is set, avoid local shadowing.
    old_cwd = None
    if chdir_tmp:
        try:
            old_cwd = os.getcwd()
            os.chdir(chdir_tmp)
        except Exception:
            old_cwd = None
    try:
        processor = AutoImageProcessor.from_pretrained(args.model_dir)
    except Exception as e:
        print(f"[WARN] Failed to load processor from {args.model_dir}: {e}")
        processor = AutoImageProcessor.from_pretrained("google/siglip2-base-patch16-224")
    finally:
        if old_cwd is not None:
            try:
                os.chdir(old_cwd)
            except Exception:
                pass

    print("[eval] Loading model...")
    # Load model using AutoModelForImageClassification so saved config determines class (siglip/siglip2/etc.)
    old_cwd = None
    if chdir_tmp:
        try:
            old_cwd = os.getcwd()
            os.chdir(chdir_tmp)
        except Exception:
            old_cwd = None
    try:
        model = AutoModelForImageClassification.from_pretrained(args.model_dir)
    except Exception as e:
        print(f"[WARN] Failed to load model from {args.model_dir}: {e}")
        print("[WARN] Retrying with ignore_mismatched_sizes=True ...")
        model = AutoModelForImageClassification.from_pretrained(args.model_dir, ignore_mismatched_sizes=True)
    finally:
        if old_cwd is not None:
            try:
                os.chdir(old_cwd)
            except Exception:
                pass
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
    # Timers and GPU memory tracking
    eval_wall_start = time.perf_counter()
    last = eval_wall_start
    infer_time_sum = 0.0
    mem_sum = 0
    mem_count = 0
    use_cuda = (device == "cuda")
    if use_cuda:
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass
    for i, row in enumerate(ds):
        rel = row["images"]
        gt = row["gt"]
        y = LABEL2ID.get(gt, int(gt) if isinstance(gt, (int, np.integer)) else 0)
        path = os.path.join(args.media_base, rel)
        try:
            img = _load_image(path)
        except Exception:
            img = PILImage.new("RGB", (224, 224), (0, 0, 0))
        inputs = processor(images=img, return_tensors="pt")
        with torch.no_grad():
            t_inf0 = time.perf_counter()
            out = model(**{k: v.to(device) for k, v in inputs.items()})
            t_inf1 = time.perf_counter()
            logits = out.logits.detach().cpu().numpy()[0]
            infer_time_sum += (t_inf1 - t_inf0)
            if use_cuda:
                try:
                    mem_sum += torch.cuda.memory_allocated()
                    mem_count += 1
                except Exception:
                    pass
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
    eval_wall_end = time.perf_counter()
    wall_seconds = eval_wall_end - eval_wall_start
    avg_infer_seconds = (infer_time_sum / max(1, len(ys)))
    gpu_peak_bytes = 0
    gpu_avg_bytes = 0
    if use_cuda:
        try:
            gpu_peak_bytes = torch.cuda.max_memory_allocated()
            if mem_count > 0:
                gpu_avg_bytes = int(mem_sum / mem_count)
        except Exception:
            pass

    results: Dict[str, float] = {"accuracy": acc, "f1": f1}
    print("Evaluation Results:")
    print(results)
    print("Classification Report:")
    print(report)
    print("Confusion Matrix (rows=true [normal, abnormal], cols=pred [normal, abnormal]):")
    print(cm)
    print(f"[eval] Samples: {len(ys)}  Total time: {wall_seconds:.2f}s  Avg infer: {avg_infer_seconds*1000:.2f} ms/sample")
    if use_cuda:
        print(f"[eval] GPU memory: avg {gpu_avg_bytes/1024/1024:.1f} MiB  peak {gpu_peak_bytes/1024/1024:.1f} MiB")

    if args.out:
        try:
            out_dir = os.path.dirname(args.out)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
        except Exception:
            pass
        with open(args.out, "w") as f:
            payload = {
                **results,
                "report": report,
                "confusion_matrix": {"labels": ["normal", "abnormal"], "matrix": cm.tolist()},
                "samples": len(ys),
                "eval_wall_seconds": wall_seconds,
                "avg_infer_seconds": avg_infer_seconds,
                "gpu_mem_avg_bytes": gpu_avg_bytes,
                "gpu_mem_peak_bytes": gpu_peak_bytes,
            }
            json.dump(payload, f, indent=2)

    # Optional PNG heatmap
    # Decide base directory to store local artifacts (README, cm image)
    artifacts_dir = args.model_dir if is_local_dir else tempfile.mkdtemp(prefix="siglip_eval_")
    if not is_local_dir:
        print(f"[eval] Using temporary artifacts dir: {artifacts_dir}")

    cm_png_path = args.cm_png
    if args.cm_png or args.visualize_push > 0:
        if cm_png_path is None:
            cm_png_path = os.path.join(artifacts_dir, "confusion_matrix.png")
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
        ds_vis = ds_vis.cast_column("image", HFImage())
        print(f"[vis] Pushing {len(ds_vis)} rows with images to {vis_repo_id} ...")
        ds_vis.push_to_hub(vis_repo_id)

        # Upload README with metrics/timings and (optional) confusion matrix image
        readme = (
            f"# SigLIP Classification Eval\n\n"
            f"- Model: {args.model_dir}\n\n"
            f"- Test repo: {args.test_repo} (split: {args.split})\n\n"
            f"- Samples visualized: {len(ds_vis)} of {total}\n\n"
            f"## Metrics\n\n"
            f"- accuracy: {acc:.4f}\n\n"
            f"- f1: {f1:.4f}\n\n"
            f"- total: {total}, correct: {int(round(acc*total))}\n\n"
            f"## Evaluation Runtime\n\n"
            f"- dataset size: {len(ys)}\n\n"
            f"- total time: {wall_seconds:.2f} sec\n\n"
            f"- avg infer time (per sample): {avg_infer_seconds*1000:.2f} ms\n\n"
        )
        if use_cuda:
            readme += (
                f"- GPU memory avg: {gpu_avg_bytes/1024/1024:.1f} MiB\n\n"
                f"- GPU memory peak: {gpu_peak_bytes/1024/1024:.1f} MiB\n\n"
            )
        else:
            readme += (f"- Device: CPU\n\n")
        readme_path = os.path.join(artifacts_dir, "VIS_README.md")
        # Ensure parent directory exists for README path (args.model_dir can be a repo-id like 'ns/name')
        try:
            rd_dir = os.path.dirname(readme_path)
            if rd_dir:
                os.makedirs(rd_dir, exist_ok=True)
        except Exception:
            pass
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
