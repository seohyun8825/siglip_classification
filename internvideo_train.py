import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import cv2
try:
    import os as _os
    _cv2_threads = int(_os.getenv("OPENCV_THREADS", "10"))
    cv2.setNumThreads(_cv2_threads)
except Exception:
    pass
try:
    import decord  # type: ignore
    _HAS_DECORD = True
except Exception:
    _HAS_DECORD = False
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import VideoMAEImageProcessor, AutoModel, AutoConfig
from sklearn.metrics import accuracy_score, f1_score


ID2LABEL = {0: "normal", 1: "abnormal"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


def infer_label_from_path(rel_path: str) -> int:
    p = rel_path.replace("\\", "/")
    if "ecva/abnormal_video/" in p:
        return 1
    return 0


def gather_video_list(hf_repo: str, split: str = "train") -> List[str]:
    ds = load_dataset(hf_repo, split=split)
    vids: List[str] = []
    for row in ds:
        vlist = row.get("videos")
        if isinstance(vlist, list):
            for v in vlist:
                if isinstance(v, str):
                    vids.append(v)
    # de-duplicate
    return sorted(set(vids))


def read_manifest_list(path: str) -> List[str]:
    """Read a cache manifest (.jsonl) or a plain text list of relative video paths.

    - JSONL: expects objects with key "video" and optional "cached" bool.
    - TXT: one relative path per line.
    Returns de-duplicated, sorted list of strings.
    """
    out: List[str] = []
    if not path or not os.path.isfile(path):
        return out
    try:
        if path.lower().endswith(".jsonl"):
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    rel = obj.get("video")
                    if not isinstance(rel, str):
                        continue
                    # If key present, keep only cached=True entries
                    if "cached" in obj and not bool(obj.get("cached")):
                        continue
                    out.append(rel)
        else:
            with open(path, "r") as f:
                for line in f:
                    rel = line.strip()
                    if rel:
                        out.append(rel)
    except Exception:
        pass
    return sorted(set(out))


def uniform_indices(num_frames: int, clip_len: int) -> List[int]:
    if num_frames <= 0:
        return [0] * clip_len
    if clip_len <= 1:
        return [min(num_frames - 1, 0)]
    # choose indices uniformly across video
    xs = np.linspace(0, max(0, num_frames - 1), clip_len)
    return [int(round(x)) for x in xs]


def read_frames_cv2(path: str, indices: List[int], resize: Optional[int] = None) -> List[np.ndarray]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out: List[np.ndarray] = []
    for idx in indices:
        j = max(0, min(total - 1, idx))
        cap.set(cv2.CAP_PROP_POS_FRAMES, j)
        ok, frame = cap.read()
        if not ok or frame is None:
            # fallback: try next
            ok2, frame2 = cap.read()
            if not ok2 or frame2 is None:
                continue
            frame = frame2
        if resize is not None and resize > 0:
            frame = cv2.resize(frame, (resize, resize), interpolation=cv2.INTER_LINEAR)
        # BGR -> RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out.append(frame)
    cap.release()
    return out


def read_frames_decord(path: str, indices: List[int], resize: Optional[int] = None) -> List[np.ndarray]:
    if not _HAS_DECORD:
        return []
    try:
        vr = decord.VideoReader(path)
    except Exception:
        return []
    total = len(vr)
    idxs = [max(0, min(total - 1, i)) for i in indices]
    try:
        batch = vr.get_batch(idxs).asnumpy()  # (N, H, W, C)
    except Exception:
        try:
            batch = np.stack([vr[i].asnumpy() for i in idxs], axis=0)
        except Exception:
            return []
    out: List[np.ndarray] = []
    for img in batch:
        if resize is not None and resize > 0:
            img = cv2.resize(img, (resize, resize), interpolation=cv2.INTER_LINEAR)
        out.append(img)
    return out


def read_clip_sequential_cv2(path: str, start_idx: int, clip_len: int, resize: Optional[int] = None) -> List[np.ndarray]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return []
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(start_idx)))
    out: List[np.ndarray] = []
    for _ in range(int(clip_len)):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if resize is not None and resize > 0:
            frame = cv2.resize(frame, (resize, resize), interpolation=cv2.INTER_LINEAR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out.append(frame)
    cap.release()
    return out


def read_clip_sequential_decord(path: str, start_idx: int, clip_len: int, resize: Optional[int] = None) -> List[np.ndarray]:
    if not _HAS_DECORD:
        return []
    try:
        vr = decord.VideoReader(path)
        total = len(vr)
    except Exception:
        return []
    s = max(0, min(int(start_idx), max(0, total - 1)))
    e = min(total, s + int(clip_len))
    try:
        batch = vr.get_batch(list(range(s, e))).asnumpy()
    except Exception:
        try:
            batch = np.stack([vr[i].asnumpy() for i in range(s, e)], axis=0)
        except Exception:
            return []
    out: List[np.ndarray] = []
    for img in batch:
        if resize is not None and resize > 0:
            img = cv2.resize(img, (resize, resize), interpolation=cv2.INTER_LINEAR)
        out.append(img)
    return out


def read_cached_images(cache_dir: str, clip_len: int, resize: Optional[int] = None) -> List[np.ndarray]:
    if not os.path.isdir(cache_dir):
        # maybe an .npz next to dir path (same stem under parent cache)
        pass
    # Prefer NPZ if exists at sibling path
    base_rel = os.path.relpath(cache_dir)
    parent = os.path.dirname(cache_dir)
    stem = os.path.basename(cache_dir)
    npz_path = os.path.join(parent, stem + ".npz")
    if os.path.isfile(npz_path):
        try:
            with np.load(npz_path) as z:
                arr = z["frames"]  # (T,H,W,C) uint8 RGB
                frames = [arr[i] for i in range(min(arr.shape[0], clip_len))]
        except Exception:
            frames = []
        out: List[np.ndarray] = []
        for img in frames:
            if resize is not None and resize > 0:
                h, w = img.shape[:2]
                if (h, w) != (resize, resize):
                    img = cv2.resize(img, (resize, resize), interpolation=cv2.INTER_LINEAR)
            out.append(img)
        return out
    # Else fallback to jpg/png directory
    if not os.path.isdir(cache_dir):
        return []
    names = sorted([n for n in os.listdir(cache_dir) if os.path.splitext(n)[1].lower() in (".jpg", ".png")])
    if not names:
        return []
    paths = [os.path.join(cache_dir, n) for n in names[: clip_len]]
    out: List[np.ndarray] = []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if resize is not None and resize > 0:
            h, w = img.shape[:2]
            if (h, w) != (resize, resize):
                img = cv2.resize(img, (resize, resize), interpolation=cv2.INTER_LINEAR)
        out.append(img)
    return out


@dataclass
class VideoSample:
    rel: str
    label: int


def passthrough_collate(batch):
    return batch


class VideoDataset(Dataset):
    def __init__(
        self,
        videos: List[str],
        media_base: str,
        clip_len: int = 16,
        frame_size: int = 224,
        missing_ok: bool = True,
        prefer_decord: bool = True,
        random_start: bool = True,
        cache_root: Optional[str] = None,
        include_after_incident: bool = True,
    ):
        self.samples: List[VideoSample] = []
        self.media_base = media_base
        self.clip_len = max(1, int(clip_len))
        self.frame_size = int(frame_size)
        self.prefer_decord = bool(prefer_decord and _HAS_DECORD)
        for rel in videos:
            p = rel.replace("\\", "/")
            if (not include_after_incident) and ("/ecva/after_incident/" in p):
                # Exclude ambiguous 'after incident' videos by default for binary classification
                continue
            y = infer_label_from_path(rel)
            self.samples.append(VideoSample(rel=rel, label=y))
        self.missing_ok = missing_ok
        self.random_start = bool(random_start)
        self.cache_root = cache_root

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        abs_path = os.path.join(self.media_base, s.rel)
        if not os.path.exists(abs_path):
            if not self.missing_ok:
                raise FileNotFoundError(abs_path)
            # return a black clip if missing
            black = np.zeros((self.frame_size, self.frame_size, 3), dtype=np.uint8)
            clip = [black for _ in range(self.clip_len)]
            return clip, s.label, s.rel
        # Use cached images if available
        if self.cache_root:
            cache_dir = os.path.join(self.cache_root, os.path.splitext(s.rel)[0])
            cached = read_cached_images(cache_dir, self.clip_len, resize=self.frame_size)
            if cached:
                if len(cached) < self.clip_len:
                    last = cached[-1]
                    cached.extend([last] * (self.clip_len - len(cached)))
                return cached[: self.clip_len], s.label, s.rel

        # Otherwise, read from video: count frames quickly (prefer decord if available)
        if self.prefer_decord:
            try:
                vr_tmp = decord.VideoReader(abs_path)
                total = len(vr_tmp)
            except Exception:
                cap = cv2.VideoCapture(abs_path)
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
        else:
            cap = cv2.VideoCapture(abs_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        # Choose a sequential clip start
        if total <= 0:
            start = 0
        elif total <= self.clip_len:
            start = 0
        else:
            if self.random_start:
                start = random.randint(0, max(0, total - self.clip_len))
            else:
                # centered clip for eval
                start = max(0, (total - self.clip_len) // 2)

        # Prefer decord for faster sequential access; fallback to OpenCV
        frames = read_clip_sequential_decord(abs_path, start, self.clip_len, resize=self.frame_size) if self.prefer_decord else []
        if not frames or len(frames) < 1:
            frames = read_clip_sequential_cv2(abs_path, start, self.clip_len, resize=self.frame_size)
        if len(frames) < self.clip_len:
            # pad last frame
            if len(frames) == 0:
                black = np.zeros((self.frame_size, self.frame_size, 3), dtype=np.uint8)
                frames = [black for _ in range(self.clip_len)]
            else:
                last = frames[-1]
                frames.extend([last] * (self.clip_len - len(frames)))
        return frames[: self.clip_len], s.label, s.rel


class ClassificationHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: List[int], num_labels: int = 2, dropout: float = 0.1):
        super().__init__()
        dims = [in_dim] + hidden_dims
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], num_labels))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def parse_hidden_dims(s: str) -> List[int]:
    s = (s or "").strip()
    if not s:
        return [1024]
    return [int(x) for x in s.split(",") if x.strip()]


def pool_tokens(feats: torch.Tensor, expected_feat_dim: Optional[int] = None) -> torch.Tensor:
    """Pool token dimension from extract_features output.

    Supports both (B, C, N) and (B, N, C) by consulting expected feature dim
    when provided. Falls back to a simple heuristic when unknown.
    """
    if feats.dim() != 3:
        return feats
    _, d1, d2 = feats.shape
    if expected_feat_dim is not None:
        if d1 == expected_feat_dim:
            return feats.mean(dim=2)
        if d2 == expected_feat_dim:
            return feats.mean(dim=1)
    # Heuristic: treat smaller inner dim as channel if ambiguous
    if d1 <= d2:
        return feats.mean(dim=2)
    else:
        return feats.mean(dim=1)


def pool_tokens(feats: torch.Tensor, expected_feat_dim: Optional[int] = None) -> torch.Tensor:
    """Pool token dimension from extract_features output.

    Supports both (B, C, N) and (B, N, C) by consulting expected feature dim
    when provided. Falls back to a simple heuristic when unknown.
    """
    if feats.dim() != 3:
        return feats
    _, d1, d2 = feats.shape
    if expected_feat_dim is not None:
        if d1 == expected_feat_dim:
            return feats.mean(dim=2)
        if d2 == expected_feat_dim:
            return feats.mean(dim=1)
    # Heuristic: treat smaller inner dim as channel if ambiguous
    if d1 <= d2:
        return feats.mean(dim=2)
    else:
        return feats.mean(dim=1)


def build_dataloaders(
    train_repo: str,
    eval_repo: Optional[str],
    eval_split: str,
    media_base: str,
    clip_len: int,
    frame_size: int,
    batch_size: int,
    num_workers: int,
    val_ratio: float,
    prefer_decord: bool,
    mp_context: Optional[str],
    cache_root_train: Optional[str],
    cache_root_eval: Optional[str],
    limit_train: int = 0,
    limit_eval: int = 0,
    train_videos_override: Optional[List[str]] = None,
    eval_videos_override: Optional[List[str]] = None,
    include_after_incident: bool = False,
):
    if train_videos_override is not None and len(train_videos_override) > 0:
        train_videos = list(train_videos_override)
    else:
        train_videos = gather_video_list(train_repo, split="train")
    if limit_train and limit_train > 0:
        train_videos = train_videos[: int(limit_train)]
    ds_va = None
    dl_va = None
    if eval_videos_override is not None and len(eval_videos_override) > 0:
        eval_videos = list(eval_videos_override)
    elif eval_repo and str(eval_repo).strip():
        eval_videos = gather_video_list(eval_repo, split=eval_split)
    else:
        eval_videos = None
    if eval_videos is not None:
        if limit_eval and limit_eval > 0:
            eval_videos = eval_videos[: int(limit_eval)]
        ds_va = VideoDataset(
            eval_videos,
            media_base=media_base,
            clip_len=clip_len,
            frame_size=frame_size,
            prefer_decord=prefer_decord,
            random_start=False,
            cache_root=cache_root_eval,
            include_after_incident=include_after_incident,
        )
        dl_va = DataLoader(
            ds_va,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=passthrough_collate,
            multiprocessing_context=(mp_context if (num_workers > 0 and mp_context) else None),
        )
    else:
        # no separate eval repo; if val_ratio>0, create small validation split from train; else disable eval
        if val_ratio and val_ratio > 0:
            random.shuffle(train_videos)
            n_val = max(1, int(len(train_videos) * val_ratio))
            eval_videos = train_videos[:n_val]
            train_videos = train_videos[n_val:]
            ds_va = VideoDataset(
                eval_videos,
                media_base=media_base,
                clip_len=clip_len,
                frame_size=frame_size,
                prefer_decord=prefer_decord,
                random_start=False,
                cache_root=cache_root_eval,
                include_after_incident=include_after_incident,
            )
            dl_va = DataLoader(
                ds_va,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=passthrough_collate,
                multiprocessing_context=(mp_context if (num_workers > 0 and mp_context) else None),
            )

    ds_tr = VideoDataset(
        train_videos,
        media_base=media_base,
        clip_len=clip_len,
        frame_size=frame_size,
        prefer_decord=prefer_decord,
        random_start=True,
        cache_root=cache_root_train,
        include_after_incident=include_after_incident,
    )
    dl_tr = DataLoader(
        ds_tr,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        # Safer defaults to avoid worker deadlocks with video backends
        persistent_workers=False,
        # prefetch_factor can trigger deadlocks on some setups; leave None
        # multiprocessing context
        multiprocessing_context=(mp_context if (num_workers > 0 and mp_context) else None),
        collate_fn=passthrough_collate,
    )
    return ds_tr, ds_va, dl_tr, dl_va


def collate_processor(
    batch,
    processor: VideoMAEImageProcessor,
    device: torch.device,
    dtype: Optional[torch.dtype] = None,
):
    # batch: List[(frames:list[np.ndarray], y:int, rel:str)]
    frames_list = [item[0] for item in batch]
    ys = torch.tensor([item[1] for item in batch], dtype=torch.long, device=device)
    rels = [item[2] for item in batch]
    inputs = processor(frames_list, return_tensors="pt")
    # pixel_values: (B, T, C, H, W) -> (B, C, T, H, W)
    pv = inputs["pixel_values"].permute(0, 2, 1, 3, 4)
    # non_blocking transfer; let autocast control dtype by default
    pv = pv.to(device, non_blocking=True)
    if device.type == "cuda":
        # If a target dtype is provided (e.g., fp16/bf16), align inputs; else keep as-is
        if dtype is not None:
            pv = pv.to(dtype)
        pv = pv.contiguous(memory_format=torch.channels_last_3d)
    return pv, ys, rels


def evaluate(
    model,
    head,
    dataloader,
    processor,
    device: torch.device,
    prec: str = "bf16",
    expected_feat_dim: Optional[int] = None,
) -> Dict[str, float]:
    model.eval()
    head.eval()
    ys_true: List[int] = []
    ys_pred: List[int] = []
    amp_dtype = torch.float16 if prec == "fp16" else (torch.bfloat16 if prec == "bf16" else None)
    with torch.no_grad():
        for batch in dataloader:
            pv, ys, _ = collate_processor(batch, processor, device, dtype=amp_dtype)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda" and amp_dtype is not None)):
                feats = model.extract_features(pixel_values=pv)
                pooled = pool_tokens(feats, expected_feat_dim=expected_feat_dim)
            logits = head(pooled.float())
            preds = torch.argmax(logits, dim=-1)
            ys_true.extend(ys.tolist())
            ys_pred.extend(preds.tolist())
    acc = accuracy_score(ys_true, ys_pred)
    f1 = f1_score(ys_true, ys_pred, average="binary", pos_label=1)
    return {"accuracy": float(acc), "f1": float(f1)}


def main():
    ap = argparse.ArgumentParser("Train InternVideo-based video classifier (normal vs abnormal)")
    ap.add_argument("--train_repo", required=True)
    ap.add_argument("--eval_repo", default=None, help="Optional separate eval repo; set empty to disable eval")
    ap.add_argument("--eval_split", default="train")
    ap.add_argument("--media_base", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--base_model", default="revliter/internvideo_next_large_p14_res224_f16")
    ap.add_argument("--clip_len", type=int, default=16)
    ap.add_argument("--frame_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--eval_interval", type=float, default=0.0, help="Eval interval: 0 disables; <1 = fraction of epoch; >=1 = fixed steps")
    ap.add_argument("--hidden", type=str, default="1024", help="Comma-separated hidden dims for classification head, e.g. '1024,512'")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--freeze_backbone", action="store_true")
    ap.add_argument("--report_to", default="none", choices=["none", "wandb"]) 
    ap.add_argument("--wandb_project", default=None, help="W&B project name when --report_to wandb")
    ap.add_argument("--wandb_run_name", default=None, help="W&B run name when --report_to wandb")
    ap.add_argument("--disable_flash_attn", action="store_true", help="Disable FlashAttention/flash SDPA kernels if possible")
    ap.add_argument("--grad_clip", type=float, default=1.0, help="Max grad-norm for clipping; <=0 disables")
    ap.add_argument("--detect_nan", action="store_true", help="Stop if loss becomes NaN and print a hint")
    ap.add_argument("--debug_batches", type=int, default=3, help="Print detailed timing/memory for the first N batches")
    ap.add_argument("--precision", choices=["fp16", "bf16", "fp32"], default="bf16", help="Numeric precision for backbone/inputs")
    ap.add_argument("--weight_decay", type=float, default=0.05)
    ap.add_argument("--warmup_ratio", type=float, default=0.05, help="Linear warmup ratio of total steps (0~1)")
    ap.add_argument("--reader", choices=["auto", "opencv", "decord"], default="auto")
    ap.add_argument("--mp_start", choices=["default", "spawn", "fork", "forkserver"], default="spawn", help="Multiprocessing start method for DataLoader workers")
    ap.add_argument("--progress_every", type=int, default=20, help="Print progress every N steps")
    ap.add_argument("--cache_root_train", default=None, help="If set, load training clips from cached images under this root")
    ap.add_argument("--cache_root_eval", default=None, help="If set, load eval clips from cached images under this root")
    ap.add_argument("--limit_train", type=int, default=0, help="If >0, limit number of training videos (head slice)")
    ap.add_argument("--limit_eval", type=int, default=0, help="If >0, limit number of eval videos (head slice)")
    ap.add_argument("--train_manifest", default=None, help="Path to cache manifest (.jsonl) or text list for training videos")
    ap.add_argument("--eval_manifest", default=None, help="Path to cache manifest (.jsonl) or text list for eval videos")
    ap.add_argument("--verify_input", type=int, default=0, help="If >0, verify labels/paths and run a tiny forward on up to N samples before training")
    ap.add_argument("--exclude_after_incident", action="store_true", help="Exclude 'after_incident' videos from train/eval sets")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")

    # Optional Weights & Biases logging
    use_wandb = (args.report_to == "wandb")
    if use_wandb:
        try:
            import wandb  # type: ignore
            wb_project = os.getenv("WANDB_PROJECT") or (args.wandb_project or "internvideo_ecva")
            wb_run_name = os.getenv("WANDB_RUN_NAME") or (args.wandb_run_name or os.path.basename(args.output_dir))
            wandb.init(project=wb_project, name=wb_run_name)
            wandb.config.update({
                "base_model": args.base_model,
                "clip_len": args.clip_len,
                "frame_size": args.frame_size,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "lr": args.lr,
                "precision": args.precision,
                "freeze_backbone": bool(args.freeze_backbone),
                "weight_decay": args.weight_decay,
                "warmup_ratio": args.warmup_ratio,
                "grad_clip": args.grad_clip,
                "eval_interval": args.eval_interval,
                "reader": args.reader,
            }, allow_val_change=True)
        except Exception as e:
            print(f"[WARN] wandb init failed: {e}")
            use_wandb = False

    # Optionally disable FlashAttention / PyTorch flash SDPA kernels
    if args.disable_flash_attn and torch.cuda.is_available():
        try:
            # Avoid local import that would shadow 'torch' symbol in this scope
            torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=True)
            os.environ["FLASH_ATTENTION_DISABLE"] = "1"
            print("[train] Disabled FlashAttention/flash SDPA kernels")
        except Exception as e:
            print(f"[WARN] Could not disable flash attention kernels: {e}")

    # Configure reader + multiprocessing context
    prefer_decord = (_HAS_DECORD and (args.reader in ("auto", "decord")))
    mp_ctx = None if args.mp_start == "default" else args.mp_start

    # Data
    # Optional: override train/eval video lists from manifest files
    vids_tr_override = read_manifest_list(args.train_manifest) if args.train_manifest else None
    vids_va_override = read_manifest_list(args.eval_manifest) if args.eval_manifest else None

    ds_tr, ds_va, dl_tr, dl_va = build_dataloaders(
        train_repo=args.train_repo,
        eval_repo=args.eval_repo,
        eval_split=args.eval_split,
        media_base=args.media_base,
        clip_len=args.clip_len,
        frame_size=args.frame_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        prefer_decord=prefer_decord,
        mp_context=mp_ctx,
        cache_root_train=args.cache_root_train,
        cache_root_eval=args.cache_root_eval,
        limit_train=args.limit_train,
        limit_eval=args.limit_eval,
        train_videos_override=vids_tr_override,
        eval_videos_override=vids_va_override,
        include_after_incident=(not args.exclude_after_incident),
    )
    print(f"[train] Train videos: {len(ds_tr)}  Eval videos: {0 if ds_va is None else len(ds_va)}")
    print(f"[train] Reader: {'decord' if prefer_decord else 'opencv'}  workers: {args.num_workers}  clip_len: {args.clip_len}  frame_size: {args.frame_size}  mp_start: {args.mp_start}")

    # Optional quick verification on training set
    if args.verify_input and len(ds_tr) > 0:
        from collections import Counter
        n_chk = max(1, min(int(args.verify_input), len(ds_tr)))
        first_rels = [ds_tr.samples[i].rel for i in range(n_chk)]
        label_list = [infer_label_from_path(r) for r in first_rels]
        dist = Counter(label_list)
        miss = 0
        for r in first_rels:
            ok = False
            if args.cache_root_train:
                cache_dir = os.path.join(args.cache_root_train, os.path.splitext(r)[0])
                npz_path = cache_dir + ".npz"
                if os.path.isfile(npz_path) or os.path.isdir(cache_dir):
                    ok = True
            if not ok and os.path.exists(os.path.join(args.media_base, r)):
                ok = True
            if not ok:
                miss += 1
        print(f"[verify] first {n_chk} label_dist={dict(dist)} abnormal_ratio={dist.get(1,0)/float(n_chk):.3f} missing={miss}")

    # Model (InternVideo only; no fallback)
    try:
        config = AutoConfig.from_pretrained(args.base_model, trust_remote_code=True)
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
        processor = VideoMAEImageProcessor.from_pretrained(args.base_model)
        backbone = AutoModel.from_pretrained(args.base_model, config=config, trust_remote_code=True)
    except ImportError as e:
        msg = str(e)
        if "flash_attn" in msg or "flash-attn" in msg:
            raise SystemExit(
                "[ERROR] The selected InternVideo model requires 'flash_attn'.\n"
                "Install it in your current env, e.g.:\n"
                "  pip install flash-attn --no-build-isolation\n"
                "or follow the official FlashAttention install docs."
            )
        raise
    backbone.to(device)
    # InternVideo Next often expects reduced precision for FlashAttention kernels
    did_reduce_backbone = False
    prec = args.precision
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    use_dtype = dtype_map.get(prec, torch.bfloat16)
    if device.type == "cuda" and prec != "fp32":
        try:
            backbone.to(dtype=use_dtype)
            did_reduce_backbone = True
            print(f"[train] Backbone cast to {str(use_dtype).split('.')[-1]}")
        except Exception:
            pass
    in_dim = getattr(config, "embed_dims", None)
    if in_dim is None:
        # fallbacks: try common attributes
        in_dim = getattr(config, "hidden_size", None)
    if in_dim is None:
        # last resort: query one forward pass
        with torch.no_grad():
            dummy = [np.zeros((args.frame_size, args.frame_size, 3), dtype=np.uint8) for _ in range(args.clip_len)]
            pv = processor(dummy, return_tensors="pt")["pixel_values"].permute(0, 2, 1, 3, 4).to(device)
            feats = backbone.extract_features(pixel_values=pv)
            if feats.dim() == 3:
                _, d1, d2 = feats.shape
                in_dim = int(min(d1, d2))
            else:
                in_dim = int(feats.shape[-1])
    print(f"[train] Feature dim: {in_dim}")
    head = ClassificationHead(in_dim=in_dim, hidden_dims=parse_hidden_dims(args.hidden), num_labels=2, dropout=args.dropout)
    head.to(device)

    if args.freeze_backbone:
        for p in backbone.parameters():
            p.requires_grad = False
        print("[train] Backbone frozen")

    trainable_params = [p for p in list(head.parameters()) + ([] if args.freeze_backbone else list(backbone.parameters())) if p.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    # If backbone params are reduced precision, avoid GradScaler unscale errors by disabling AMP scaler
    use_scaler = (device.type == "cuda" and not did_reduce_backbone and prec == "fp16")
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    best_acc = -1.0
    best_path = os.path.join(args.output_dir, "best_head.pt")
    last_path = os.path.join(args.output_dir, "last_head.pt")

    # resolve eval schedule
    steps_per_epoch = max(1, math.ceil(len(ds_tr) / max(1, args.batch_size)))
    if args.eval_interval and args.eval_interval > 0:
        if args.eval_interval < 1:
            eval_every = max(1, math.ceil(steps_per_epoch * float(args.eval_interval)))
            schedule_desc = f"steps@{eval_every} (fraction {args.eval_interval})"
        else:
            eval_every = int(args.eval_interval)
            schedule_desc = f"steps@{eval_every}"
    else:
        eval_every = None
        schedule_desc = "disabled"
    print(f"[train] Eval schedule: {schedule_desc}")
    print(f"[train] Total steps: {steps_per_epoch * max(1, args.epochs)}  (per-epoch: {steps_per_epoch})")

    # Forward shape/dtype verification on one mini-batch
    if args.verify_input and len(ds_tr) > 0:
        try:
            it = iter(dl_tr)
            batch = next(it)
            amp_dtype = torch.float16 if prec == "fp16" else (torch.bfloat16 if prec == "bf16" else None)
            pv, ys, rels = collate_processor(batch, processor, device, dtype=amp_dtype)
            pv_min = float(pv.min().item()) if pv.numel() else 0.0
            pv_max = float(pv.max().item()) if pv.numel() else 0.0
            print(f"[verify] pv shape={tuple(pv.shape)} dtype={pv.dtype} range=({pv_min:.1f},{pv_max:.1f})")
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda" and amp_dtype is not None)):
                    feats = backbone.extract_features(pixel_values=pv)
                print(f"[verify] feats shape={tuple(feats.shape)} dtype={feats.dtype}")
                pooled = pool_tokens(feats, expected_feat_dim=in_dim)
                logits = head(pooled.float())
                print(f"[verify] logits shape={tuple(logits.shape)} example={logits[0].detach().cpu().tolist() if logits.shape[0]>0 else []}")
        except Exception as e:
            print(f"[verify] forward check failed: {e}")

    eval_log_path = os.path.join(args.output_dir, "eval_log.jsonl")
    def log_eval_jsonl(metrics: Dict[str, float], step: int, epoch: int):
        try:
            with open(eval_log_path, "a") as f:
                rec = {"step": step, "epoch": epoch, **metrics}
                f.write(json.dumps(rec) + "\n")
        except Exception:
            pass

    # LR scheduler with linear warmup
    total_steps = steps_per_epoch * max(1, args.epochs)
    warmup_steps = int(total_steps * max(0.0, min(1.0, args.warmup_ratio)))
    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

    global_step = 0
    debug_batches = max(0, int(getattr(args, "debug_batches", 0)))
    last_iter_end = time.perf_counter()
    train_start = last_iter_end
    for epoch in range(args.epochs):
        backbone.train()
        head.train()
        running = 0.0
        steps = 0
        for batch in dl_tr:
            t_loop_start = time.perf_counter()
            load_sec = t_loop_start - last_iter_end
            amp_dtype = torch.float16 if prec == "fp16" else (torch.bfloat16 if prec == "bf16" else None)
            pv, ys, _ = collate_processor(batch, processor, device, dtype=amp_dtype)
            t_coll_end = time.perf_counter()
            if device.type == "cuda" and steps < debug_batches:
                try:
                    torch.cuda.reset_peak_memory_stats()
                except Exception:
                    pass
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=(device.type == "cuda" and amp_dtype is not None)):
                t_fwd_start = time.perf_counter()
                feats = backbone.extract_features(pixel_values=pv)
                pooled = pool_tokens(feats, expected_feat_dim=in_dim)
                # keep head in fp32 for stability
                logits = head(pooled.float())
                loss = F.cross_entropy(logits, ys)
                t_fwd_end = time.perf_counter()
            # Optional finiteness checks to pinpoint NaNs
            if steps < max(1, debug_batches) and device.type == "cuda":
                try:
                    if not torch.isfinite(feats).all():
                        print("[debug] NaN/Inf in feats; pv", pv.dtype, pv.min().item(), pv.max().item())
                        raise SystemExit(1)
                    if not torch.isfinite(logits).all():
                        print("[debug] NaN/Inf in logits; feats", feats.dtype, feats.min().item(), feats.max().item())
                        raise SystemExit(1)
                    if not torch.isfinite(loss):
                        print("[debug] NaN/Inf in loss; logits", logits.min().item(), logits.max().item())
                        raise SystemExit(1)
                except Exception:
                    pass
            if args.detect_nan and (torch.isnan(loss).item() or torch.isinf(loss).item()):
                print("[ERR] Loss is NaN/Inf. Tips: lower --lr (e.g., 1e-5), enable --grad_clip, freeze backbone or check FP16 kernels.")
                raise SystemExit(1)
            opt.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                # correct AMP pattern: scale -> backward -> step -> update
                t_bwd_start = time.perf_counter()
                scaler.scale(loss).backward()
                if args.grad_clip and args.grad_clip > 0:
                    try:
                        scaler.unscale_(opt)
                        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=args.grad_clip)
                    except Exception:
                        pass
                scaler.step(opt)
                scaler.update()
                t_bwd_end = time.perf_counter()
            else:
                t_bwd_start = time.perf_counter()
                loss.backward()
                if args.grad_clip and args.grad_clip > 0:
                    try:
                        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=args.grad_clip)
                    except Exception:
                        pass
                opt.step()
                t_bwd_end = time.perf_counter()
            try:
                scheduler.step()
            except Exception:
                pass
            running += float(loss.detach().item())
            steps += 1
            global_step += 1
            if steps % max(1, args.progress_every) == 0:
                elapsed = time.perf_counter() - train_start
                done = (epoch * steps_per_epoch) + steps
                total = steps_per_epoch * max(1, args.epochs)
                rate = done / max(1e-6, elapsed)
                remain = max(0, total - ((epoch * steps_per_epoch) + steps))
                eta_sec = remain / max(1e-6, rate)
                def _fmt(s):
                    m, s = divmod(int(s), 60)
                    h, m = divmod(m, 60)
                    return f"{h:02d}:{m:02d}:{s:02d}"
                pct = 100.0 * done / max(1, total)
                print(
                    f"[train] epoch {epoch+1} step {steps} loss {running/steps:.4f} | "
                    f"global {done}/{total} ({pct:.1f}%) | speed {rate:.2f} it/s | eta {_fmt(eta_sec)}"
                )
                if use_wandb:
                    try:
                        import wandb  # type: ignore
                        wandb.log({
                            "train/loss": float(running/steps),
                            "train/step": int(global_step),
                            "train/epoch_float": float(epoch + steps/steps_per_epoch),
                            "train/lr": float(opt.param_groups[0].get("lr", 0.0)),
                        })
                    except Exception:
                        pass

            # Detailed debug for first N batches
            if steps <= debug_batches:
                total_sec = time.perf_counter() - t_loop_start
                if device.type == "cuda":
                    try:
                        mem_alloc = torch.cuda.memory_allocated() / (1024**2)
                        mem_peak = torch.cuda.max_memory_allocated() / (1024**2)
                    except Exception:
                        mem_alloc = mem_peak = -1
                else:
                    mem_alloc = mem_peak = -1
                print(
                    "[debug] step",
                    steps,
                    f"load={load_sec*1000:.1f}ms",
                    f"collate={ (t_fwd_start - t_coll_end)*1000:.1f}ms",
                    f"forward={ (t_fwd_end - t_fwd_start)*1000:.1f}ms",
                    f"backward+step={ (t_bwd_end - t_bwd_start)*1000:.1f}ms",
                    f"total={total_sec*1000:.1f}ms",
                    f"gpu_alloc={mem_alloc:.1f}MiB",
                    f"gpu_peak={mem_peak:.1f}MiB",
                )

            # step-level eval
            if eval_every is not None and dl_va is not None and (global_step % eval_every == 0):
                metrics = evaluate(backbone, head, dl_va, processor, device, expected_feat_dim=in_dim)
                print(f"[eval] step {global_step} metrics: {metrics}")
                log_eval_jsonl(metrics, step=global_step, epoch=epoch + 1)
                if use_wandb:
                    try:
                        import wandb  # type: ignore
                        wandb.log({
                            "eval/accuracy": metrics.get("accuracy", 0.0),
                            "eval/f1": metrics.get("f1", 0.0),
                            "eval/step": int(global_step),
                            "eval/epoch": int(epoch + 1),
                        })
                    except Exception:
                        pass
                # save best by accuracy
                if metrics["accuracy"] > best_acc:
                    best_acc = metrics["accuracy"]
                    torch.save({"head": head.state_dict()}, best_path)
                    with open(os.path.join(args.output_dir, "best_metrics.json"), "w") as f:
                        json.dump({"best_accuracy": best_acc, "epoch": epoch + 1, "step": global_step}, f, indent=2)
                # also update last
                torch.save({"head": head.state_dict()}, last_path)

        # epoch eval
        if dl_va is not None:
            metrics = evaluate(backbone, head, dl_va, processor, device, expected_feat_dim=in_dim)
            print(f"[eval] epoch {epoch+1} metrics: {metrics}")
            log_eval_jsonl(metrics, step=global_step, epoch=epoch + 1)
            if use_wandb:
                try:
                    import wandb  # type: ignore
                    wandb.log({
                        "eval/accuracy": metrics.get("accuracy", 0.0),
                        "eval/f1": metrics.get("f1", 0.0),
                        "eval/step": int(global_step),
                        "eval/epoch": int(epoch + 1),
                    })
                except Exception:
                    pass
            # save last
            torch.save({"head": head.state_dict()}, last_path)
            # save best by accuracy
            if metrics["accuracy"] > best_acc:
                best_acc = metrics["accuracy"]
                torch.save({"head": head.state_dict()}, best_path)
                with open(os.path.join(args.output_dir, "best_metrics.json"), "w") as f:
                    json.dump({"best_accuracy": best_acc, "epoch": epoch + 1, "step": global_step}, f, indent=2)
        else:
            # even without eval, save last periodically per epoch
            torch.save({"head": head.state_dict()}, last_path)

        # mark end of last iteration for next epoch's load timing
        last_iter_end = time.perf_counter()

    # Save config
    summary = {
        "base_model": args.base_model,
        "clip_len": args.clip_len,
        "frame_size": args.frame_size,
        "hidden": parse_hidden_dims(args.hidden),
        "dropout": args.dropout,
        "freeze_backbone": bool(args.freeze_backbone),
        "feature_dim": int(in_dim),
        "train_repo": args.train_repo,
        "eval_repo": args.eval_repo,
        "eval_split": args.eval_split,
        "eval_interval": args.eval_interval,
        "media_base": args.media_base,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
    }
    with open(os.path.join(args.output_dir, "train_config.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Finish W&B run
    if use_wandb:
        try:
            import wandb  # type: ignore
            wandb.summary.update({"best_accuracy": float(best_acc)})
            wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
