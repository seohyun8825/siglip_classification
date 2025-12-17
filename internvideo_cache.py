import argparse
import os
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple

import cv2
import numpy as np
from datasets import load_dataset

try:
    import decord  # type: ignore
    _HAS_DECORD = True
except Exception:
    _HAS_DECORD = False


def list_unique_videos(repo: str, split: str = "train") -> List[str]:
    ds = load_dataset(repo, split=split)
    vids = []
    for r in ds:
        for v in r.get("videos", []) or []:
            if isinstance(v, str):
                vids.append(v)
    return sorted(set(vids))


def infer_label_from_path(rel: str) -> int:
    p = rel.replace("\\", "/")
    if "ecva/abnormal_video/" in p:
        return 1
    return 0


def _count_frames(path: str, prefer_decord: bool = True) -> int:
    if prefer_decord and _HAS_DECORD:
        try:
            vr = decord.VideoReader(path)
            return len(vr)
        except Exception:
            pass
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def _read_clip(path: str, start: int, clip_len: int, resize: Optional[int], prefer_decord: bool = True) -> List[np.ndarray]:
    # prefer decord sequential read
    if prefer_decord and _HAS_DECORD:
        try:
            vr = decord.VideoReader(path)
            s = max(0, min(start, max(0, len(vr) - 1)))
            e = min(len(vr), s + clip_len)
            try:
                batch = vr.get_batch(list(range(s, e))).asnumpy()
            except Exception:
                batch = np.stack([vr[i].asnumpy() for i in range(s, e)], axis=0)
            out = []
            for img in batch:
                if resize and resize > 0:
                    img = cv2.resize(img, (resize, resize), interpolation=cv2.INTER_LINEAR)
                out.append(img)
            return out
        except Exception:
            pass
    # fallback cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return []
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(start)))
    out = []
    for _ in range(int(clip_len)):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if resize and resize > 0:
            frame = cv2.resize(frame, (resize, resize), interpolation=cv2.INTER_LINEAR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out.append(frame)
    cap.release()
    return out


def _save_frames(frames: List[np.ndarray], out_dir: str) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i, img in enumerate(frames):
        # convert back to BGR for cv2.imwrite
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        p = os.path.join(out_dir, f"{i:06d}.jpg")
        cv2.imwrite(p, bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        paths.append(p)
    return paths


def _save_npz(frames: List[np.ndarray], out_root: str, rel: str) -> str:
    # Save a single compressed npz per video: T,H,W,C in RGB uint8
    arr = np.stack(frames, axis=0)
    out_path = os.path.join(out_root, os.path.splitext(rel)[0] + ".npz")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, frames=arr)
    return out_path


def process_one(rel: str, media_base: str, out_root: str, clip_len: int, frame_size: int, mode: str, overwrite: bool = False, prefer_decord: bool = True, fmt: str = "npz") -> dict:
    src = os.path.join(media_base, rel)
    dst_dir = os.path.join(out_root, os.path.splitext(rel)[0])
    # skip if already cached
    if not overwrite:
        if fmt == "jpg" and os.path.isdir(dst_dir):
            exts = (".jpg", ".png")
            cnt = sum(1 for n in os.listdir(dst_dir) if os.path.splitext(n)[1].lower() in exts)
            if cnt >= clip_len:
                return {"video": rel, "cached": True, "frames": cnt, "format": fmt}
        if fmt == "npz":
            npz_path = os.path.join(out_root, os.path.splitext(rel)[0] + ".npz")
            if os.path.isfile(npz_path):
                try:
                    with np.load(npz_path) as z:
                        t = int(z["frames"].shape[0])
                except Exception:
                    t = clip_len
                if t >= 1:
                    return {"video": rel, "cached": True, "frames": t, "format": fmt}
    total = _count_frames(src, prefer_decord=prefer_decord)
    if total <= 0:
        return {"video": rel, "cached": False, "error": "no_frames"}
    if total <= clip_len:
        start = 0
    else:
        if mode == "center":
            start = max(0, (total - clip_len) // 2)
        elif mode == "uniform":
            # spread roughly uniformly, but still read sequential from nearest start
            start = 0
        elif mode == "random":
            import random as _r
            start = _r.randint(0, max(0, total - clip_len))
        else:
            start = max(0, (total - clip_len) // 2)
    frames = _read_clip(src, start, clip_len, frame_size, prefer_decord=prefer_decord)
    if not frames:
        return {"video": rel, "cached": False, "error": "read_fail"}
    if fmt == "jpg":
        paths = _save_frames(frames, dst_dir)
        return {"video": rel, "cached": True, "frames": len(paths), "format": fmt}
    else:
        npz_path = _save_npz(frames, out_root, rel)
        return {"video": rel, "cached": True, "frames": len(frames), "format": fmt}


def _fmt_seconds(s: float) -> str:
    if s == float('inf') or s != s:
        return "--:--:--"
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def main():
    ap = argparse.ArgumentParser("Cache fixed clips (frames) per video for fast training")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--media_base", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--clip_len", type=int, default=16)
    ap.add_argument("--frame_size", type=int, default=224)
    ap.add_argument("--mode", choices=["center", "uniform", "random"], default="center")
    ap.add_argument("--max_workers", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--progress_every", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0, help="If >0, cache only this many videos")
    ap.add_argument("--limit_mode", choices=["head", "random"], default="head")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--backend", choices=["auto", "opencv", "decord"], default="auto")
    ap.add_argument("--format", choices=["jpg", "npz"], default="npz")
    ap.add_argument("--balanced_total", type=int, default=0, help="If >0, pick a balanced subset of this total videos (half per class) before caching")
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    vids = list_unique_videos(args.repo, split=args.split)
    # Build balanced subset first if requested
    if args.balanced_total and args.balanced_total > 0:
        import random as _r
        rs = _r.Random(args.seed)
        cls0 = [v for v in vids if infer_label_from_path(v) == 0]
        cls1 = [v for v in vids if infer_label_from_path(v) == 1]
        rs.shuffle(cls0); rs.shuffle(cls1)
        half = max(1, args.balanced_total // 2)
        vids = cls0[:half] + cls1[:half]
        rs.shuffle(vids)
    # limit subset if requested (applied after balancing)
    if args.limit and args.limit > 0:
        if args.limit_mode == "random":
            import random as _r
            _r.Random(args.seed).shuffle(vids)
        vids = vids[: args.limit]
    print(f"[cache] Repo {args.repo} split={args.split} unique videos: {len(vids)}")

    results = []
    ok = 0
    errs = 0
    total = len(vids)
    start = time.perf_counter()
    prefer_decord = (_HAS_DECORD and (args.backend in ("auto", "decord")))
    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as ex:
        futs = {
            ex.submit(process_one, rel, args.media_base, args.out_root, args.clip_len, args.frame_size, args.mode, args.overwrite, prefer_decord, args.format): rel
            for rel in vids
        }
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            results.append(r)
            if r.get("cached"):
                ok += 1
            else:
                errs += 1
            if i % max(1, args.progress_every) == 0 or i == total:
                elapsed = time.perf_counter() - start
                rate = (i / elapsed) if elapsed > 0 else 0.0
                remain = max(0, total - i)
                eta = (remain / rate) if rate > 0 else float('inf')
                pct = 100.0 * i / max(1, total)
                print(
                    f"[cache] {i}/{total} ({pct:.1f}%) cached={ok} err={errs} "
                    f"speed={rate:.2f} it/s eta={_fmt_seconds(eta)}"
                )
    # final tallies
    ok = sum(1 for r in results if r.get("cached"))
    errs = sum(1 for r in results if not r.get("cached"))
    print(f"[cache] done: cached={ok} errors={errs}")

    # write manifest jsonl
    mani = os.path.join(args.out_root, f"_manifest_{args.repo.replace('/', '_')}_{args.split}.jsonl")
    with open(mani, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"[cache] wrote manifest: {mani}")


if __name__ == "__main__":
    main()
