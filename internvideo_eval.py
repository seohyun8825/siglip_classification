import argparse
import json
import os
import tempfile
from datetime import datetime
from typing import List, Dict

import numpy as np
import torch
from datasets import load_dataset, Dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import VideoMAEImageProcessor, AutoModel, AutoConfig

from internvideo_train import VideoDataset, collate_processor, ID2LABEL, LABEL2ID, pool_tokens

try:
    from icecream import ic  # optional pretty logger
except Exception:
    ic = None


def main():
    ap = argparse.ArgumentParser("Eval InternVideo-based classifier")
    ap.add_argument("--model_dir", required=True, help="Directory with best_head.pt (and train_config.json)")
    ap.add_argument("--media_base", required=True)
    ap.add_argument("--test_repo", required=True)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0, help="If >0, limit number of test videos")
    ap.add_argument("--cache_root", default=None, help="If set, load cached frames under this root")
    ap.add_argument("--push_repo", default=None, help="If set (namespace/repo), push per-sample results to HF dataset repo and write metrics to README.md")
    ap.add_argument("--out_dir", default=None, help="Where to write local eval files; defaults to model_dir")
    ap.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default="bf16")
    ap.add_argument("--disable_flash_attn", action="store_true", help="Try to disable flash attention / use SDPA if possible")
    ap.add_argument("--log_each", action="store_true", help="Print per-sample video/pred/gt during evaluation")
    args = ap.parse_args()

    cfg_path = os.path.join(args.model_dir, "train_config.json")
    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    base_model = cfg.get("base_model", "revliter/internvideo_next_large_p14_res224_f16")
    clip_len = int(cfg.get("clip_len", 16))
    frame_size = int(cfg.get("frame_size", 224))

    # Collect test videos
    ds = load_dataset(args.test_repo, split="train")
    vids: List[str] = []
    for r in ds:
        for v in r.get("videos", []) or []:
            if isinstance(v, str):
                vids.append(v)
    vids = sorted(set(vids))
    if args.limit and args.limit > 0:
        vids = vids[: args.limit]

    dataset = VideoDataset(vids, media_base=args.media_base, clip_len=clip_len, frame_size=frame_size, random_start=False, cache_root=args.cache_root)
    from torch.utils.data import DataLoader
    from internvideo_train import passthrough_collate
    dl = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, collate_fn=passthrough_collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Optionally disable flash attention via env (must be set before model init)
    if args.disable_flash_attn:
        os.environ["FLASH_ATTENTION_DISABLE"] = "1"
    config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
    # Try to force non-flash attention in config when requested
    if args.disable_flash_attn:
        for key in ("attn_implementation", "use_flash_attn", "flash_attn", "flash_attention"):
            if hasattr(config, key):
                try:
                    if key == "attn_implementation":
                        setattr(config, key, "sdpa")
                    else:
                        setattr(config, key, False)
                except Exception:
                    pass
    processor = VideoMAEImageProcessor.from_pretrained(base_model)
    backbone = AutoModel.from_pretrained(base_model, config=config, trust_remote_code=True)
    backbone.to(device)
    # Align backbone parameter dtype to requested precision (InternVideo disables autocast internally)
    if device.type == "cuda" and args.precision != "fp32":
        try:
            target = torch.float16 if args.precision == "fp16" else torch.bfloat16
            backbone.to(dtype=target)
            print(f"[eval] Backbone cast to {str(target).split('.')[-1]}")
        except Exception:
            pass

    # head
    import torch.nn as nn
    from internvideo_train import ClassificationHead, parse_hidden_dims
    in_dim = cfg.get("feature_dim") or cfg.get("hidden_size")
    if in_dim is None:
        # probe
        with torch.no_grad():
            import numpy as np
            dummy = [np.zeros((frame_size, frame_size, 3), dtype=np.uint8) for _ in range(clip_len)]
            pv = processor(dummy, return_tensors="pt")["pixel_values"].permute(0, 2, 1, 3, 4).to(device)
            feats = backbone.extract_features(pixel_values=pv)
            if feats.dim() == 3:
                _, d1, d2 = feats.shape
                in_dim = int(min(d1, d2))
            else:
                in_dim = int(feats.shape[-1])
    head = ClassificationHead(in_dim=in_dim, hidden_dims=parse_hidden_dims(
        ",".join(map(str, cfg.get("hidden", [1024])))
    ))
    state = torch.load(os.path.join(args.model_dir, "best_head.pt"), map_location="cpu")
    head.load_state_dict(state["head"])
    head.to(device)

    ys_true: List[int] = []
    ys_pred: List[int] = []
    records: List[Dict] = []
    backbone.eval(); head.eval()
    # dtype policy
    amp_dtype = torch.float16 if args.precision == "fp16" else (torch.bfloat16 if args.precision == "bf16" else None)
    seen = 0
    total = len(dataset)
    with torch.no_grad():
        for batch in dl:
            pv, ys, rels = collate_processor(batch, processor, device, dtype=amp_dtype)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda" and amp_dtype is not None)):
                feats = backbone.extract_features(pixel_values=pv)
            pooled = pool_tokens(feats, expected_feat_dim=in_dim)
            # keep head in fp32 for stability and to match training
            logits = head(pooled.float())
            pred = torch.argmax(logits, dim=-1)
            ys_true.extend(ys.tolist())
            ys_pred.extend(pred.tolist())
            # per-sample logs
            for i in range(len(rels)):
                gt_id = int(ys[i].item())
                pr_id = int(pred[i].item())
                rec = {
                    "video": rels[i],
                    "gt": ID2LABEL.get(gt_id, str(gt_id)),
                    "pred": ID2LABEL.get(pr_id, str(pr_id)),
                    "gt_id": gt_id,
                    "pred_id": pr_id,
                    "correct": bool(gt_id == pr_id),
                }
                records.append(rec)
                seen += 1
                if args.log_each:
                    msg = f"[eval] {seen}/{total} {rec['video']} | gt={rec['gt']} pred={rec['pred']} {'✓' if rec['correct'] else '✗'}"
                    if ic is not None:
                        ic(msg)
                    else:
                        print(msg)

    acc = accuracy_score(ys_true, ys_pred)
    f1 = f1_score(ys_true, ys_pred, average="binary", pos_label=1)
    metrics = {"accuracy": float(acc), "f1": float(f1), "samples": len(ys_true)}
    print(metrics)

    out_dir = args.out_dir or args.model_dir
    os.makedirs(out_dir, exist_ok=True)
    # save local files
    with open(os.path.join(out_dir, "eval_results.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(out_dir, "eval_samples.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Optional push to HF dataset repo
    if args.push_repo and str(args.push_repo).strip():
        repo_id = args.push_repo.strip()
        ds_out = Dataset.from_list(records)
        # push per-sample results
        ds_out.push_to_hub(repo_id)
        # write README with metrics via huggingface_hub
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            readme = f"""
            # InternVideo Eval Results

            - Model dir: `{args.model_dir}`
            - Test repo: `{args.test_repo}`
            - Samples: {metrics['samples']}
            - Accuracy: {metrics['accuracy']:.4f}
            - F1 (binary, abnormal=1): {metrics['f1']:.4f}
            - Pushed at: {datetime.utcnow().isoformat()}Z

            Columns:
            - `video` (relative path)
            - `gt` / `pred` (strings)
            - `gt_id` / `pred_id` (ints)
            - `correct` (bool)
            """.strip() + "\n"
            with tempfile.NamedTemporaryFile("w", delete=False) as tf:
                tf.write(readme)
                tmp_path = tf.name
            api.upload_file(
                path_or_fileobj=tmp_path,
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="dataset",
            )
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            print(f"[eval] Pushed dataset + README to: {repo_id}")
        except Exception as e:
            print(f"[eval] Push README failed: {e}")


if __name__ == "__main__":
    main()
