import argparse
import os
import sys
import json
import time
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
from transformers import AutoModel, AutoConfig, VideoMAEImageProcessor



try:
    from internvideo_train import pool_tokens, infer_label_from_path
except Exception:
    # Minimal local fallbacks
    def pool_tokens(feats: torch.Tensor, expected_feat_dim: Optional[int] = None) -> torch.Tensor:
        if feats.dim() != 3:
            return feats
        _, d1, d2 = feats.shape
        if expected_feat_dim is not None:
            if d1 == expected_feat_dim:
                return feats.mean(dim=2)
            if d2 == expected_feat_dim:
                return feats.mean(dim=1)
        return feats.mean(dim=2 if d1 <= d2 else 1)

    def infer_label_from_path(rel_path: str) -> int:
        p = rel_path.replace("\\", "/")
        return 1 if "/ecva/abnormal_video/" in p else 0


def list_npz(root: str, subdir_filters: Optional[List[str]] = None, limit: int = 0) -> List[str]:
    files: List[str] = []
    subdir_filters = [s.strip() for s in (subdir_filters or []) if s and s.strip()]
    for dirpath, _, filenames in os.walk(root):
        for n in filenames:
            if not n.lower().endswith('.npz'):
                continue
            p = os.path.join(dirpath, n)
            if subdir_filters:
                norm = p.replace('\\', '/').lower()
                if not any(('/' + s.strip('/').lower() + '/') in norm for s in subdir_filters):
                    continue
            files.append(p)
            if limit and len(files) >= limit:
                return files
    return files


def load_npz_frames(npz_path: str) -> List[np.ndarray]:
    with np.load(npz_path) as z:
        arr = z['frames']  # (T,H,W,C) uint8 RGB
        frames = [arr[i] for i in range(arr.shape[0])]
    return frames


@torch.no_grad()
def embed_videos(
    paths: List[str],
    base_model: str,
    precision: str = 'bf16',
    device: Optional[torch.device] = None,
    disable_flash_attn: bool = False,
    clip_len: int = 16,
    frame_size: int = 224,
) -> Tuple[np.ndarray, np.ndarray, int]:
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if disable_flash_attn:
        os.environ['FLASH_ATTENTION_DISABLE'] = '1'
    config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
    # Try to disable flash attention via config if requested
    if disable_flash_attn:
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
    # set precision
    dtype_map = {'fp16': torch.float16, 'bf16': torch.bfloat16, 'fp32': torch.float32}
    tgt_dtype = dtype_map.get(precision, torch.bfloat16)
    if device.type == 'cuda' and precision != 'fp32':
        try:
            backbone.to(dtype=tgt_dtype)
        except Exception:
            pass

    # Probe feature dimension
    def _probe_dim() -> int:
        dummy = [np.zeros((frame_size, frame_size, 3), dtype=np.uint8) for _ in range(clip_len)]
        pv = processor(dummy, return_tensors='pt')["pixel_values"].permute(0, 2, 1, 3, 4).to(device)
        feats = backbone.extract_features(pixel_values=pv)
        if feats.dim() == 3:
            _, d1, d2 = feats.shape
            return int(min(d1, d2))
        return int(feats.shape[-1])

    feat_dim = _probe_dim()
    amp_dtype = torch.float16 if precision == 'fp16' else (torch.bfloat16 if precision == 'bf16' else None)

    embs: List[np.ndarray] = []
    labels: List[int] = []
    t0 = time.perf_counter()
    for i, p in enumerate(paths, 1):
        try:
            frames = load_npz_frames(p)
        except Exception:
            continue
        if len(frames) == 0:
            continue
        # truncate/pad to clip_len
        if len(frames) < clip_len:
            frames = frames + [frames[-1]] * (clip_len - len(frames))
        frames = frames[:clip_len]
        # preprocess
        inputs = processor([frames], return_tensors='pt')
        pv = inputs['pixel_values'].permute(0, 2, 1, 3, 4).to(device, non_blocking=True)
        with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=(device.type == 'cuda' and amp_dtype is not None)):
            feats = backbone.extract_features(pixel_values=pv)
        pooled = pool_tokens(feats, expected_feat_dim=feat_dim)
        vec = pooled.float().detach().cpu().numpy()[0]
        embs.append(vec)
        labels.append(infer_label_from_path(p))
        if i % 50 == 0:
            elapsed = time.perf_counter() - t0
            print(f"[embed] {i}/{len(paths)} ({i/len(paths)*100:.1f}%) elapsed={elapsed:.1f}s")

    if not embs:
        return np.zeros((0, feat_dim), dtype=np.float32), np.zeros((0,), dtype=np.int64), feat_dim
    X = np.stack(embs, axis=0)
    y = np.array(labels, dtype=np.int64)
    return X, y, feat_dim


def summarize_embeddings(X: np.ndarray, y: np.ndarray) -> Dict:
    def stats_of(vals: np.ndarray) -> Dict:
        if vals.size == 0:
            return {"count": 0}
        q = np.quantile(vals, [0.0, 0.1, 0.5, 0.9, 1.0])
        return {
            "count": int(vals.size),
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=0)),
            "min": float(q[0]),
            "p10": float(q[1]),
            "p50": float(q[2]),
            "p90": float(q[3]),
            "max": float(q[4]),
        }

    norms = np.linalg.norm(X, axis=1)
    stats_all = stats_of(norms)
    out = {"total": stats_all, "by_class": {}}
    for c in [0, 1]:
        idx = (y == c)
        cls_norms = norms[idx]
        cls_stats = stats_of(cls_norms)
        out["by_class"][str(c)] = cls_stats
    # centers & separation
    mu_all = X.mean(axis=0)
    mu0 = X[y == 0].mean(axis=0) if np.any(y == 0) else np.zeros_like(mu_all)
    mu1 = X[y == 1].mean(axis=0) if np.any(y == 1) else np.zeros_like(mu_all)
    out["centers"] = {
        "mu_all_norm": float(np.linalg.norm(mu_all)),
        "mu0_norm": float(np.linalg.norm(mu0)),
        "mu1_norm": float(np.linalg.norm(mu1)),
        "delta_mu_norm": float(np.linalg.norm(mu1 - mu0)),
    }
    # top dims by abs delta
    delta = np.abs(mu1 - mu0)
    topk = min(10, delta.shape[0])
    top_idx = np.argsort(-delta)[:topk]
    out["top_dims_by_delta"] = [{"dim": int(i), "delta": float(delta[i])} for i in top_idx]
    return out


def main():
    ap = argparse.ArgumentParser("Extract InternVideo embeddings from cached NPZs and print distribution stats")
    ap.add_argument("--cache_root", default="/hub_data4/seohyun/video_frame_cache", help="Root dir containing NPZ caches (e.g., train/, valid/)")
    ap.add_argument("--subdir", nargs="*", default=["train", "valid"], help="Filter NPZ by subdir names (matches path segment)")
    ap.add_argument("--limit", type=int, default=500, help="Limit number of NPZ files (0 = all)")
    ap.add_argument("--base_model", default="revliter/internvideo_next_large_p14_res224_f16")
    ap.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default="bf16")
    ap.add_argument("--disable_flash_attn", action="store_true")
    ap.add_argument("--clip_len", type=int, default=16)
    ap.add_argument("--frame_size", type=int, default=224)
    ap.add_argument("--save_json", default=None, help="Optional path to save stats JSON")
    args = ap.parse_args()

    files = list_npz(args.cache_root, subdir_filters=args.subdir, limit=args.limit)
    if not files:
        print(f"[err] No NPZ files found under {args.cache_root} with filters {args.subdir}")
        sys.exit(1)
    print(f"[info] NPZ files: {len(files)}  (root={args.cache_root}, filters={args.subdir})")

    X, y, d = embed_videos(
        files,
        base_model=args.base_model,
        precision=args.precision,
        disable_flash_attn=args.disable_flash_attn,
        clip_len=args.clip_len,
        frame_size=args.frame_size,
    )
    print(f"[info] embeddings shape: {X.shape} (D={d})  labels: {y.shape}")
    stats = summarize_embeddings(X, y)
    #print("[stats]", json.dumps(stats, indent=2))
    if args.save_json:
        try:
            os.makedirs(os.path.dirname(args.save_json), exist_ok=True)
            with open(args.save_json, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"[info] wrote stats JSON: {args.save_json}")
        except Exception as e:
            print(f"[warn] write stats failed: {e}")


if __name__ == "__main__":
    main()

