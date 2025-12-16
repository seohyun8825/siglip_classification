import argparse
import os
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Set, Tuple, Optional

import cv2
from datasets import Dataset, load_dataset, Image
from huggingface_hub import HfApi


def parse_args():
    p = argparse.ArgumentParser(description="Build and push frame-level HF datasets from extracted images.")
    p.add_argument("--train_dataset", required=True, help="Source HF dataset repo with videos for train")
    p.add_argument("--test_dataset", required=True, help="Source HF dataset repo with videos for test")
    p.add_argument("--media_base", required=True, help="Absolute base directory containing extracted images under ecva/*_classification/")
    p.add_argument("--video_media_base", default=None, help="Absolute base directory of source videos for test mid-frame extraction. Defaults to --media_base if omitted.")
    p.add_argument("--out_train_repo", required=True, help="Destination HF dataset repo for frames (e.g., happy8825/siglip_train)")
    p.add_argument("--out_test_repo", required=True, help="Destination HF dataset repo for frames (e.g., happy8825/siglip_test)")
    p.add_argument("--out_test_repo_with_image", default=None, help="Optional: also push a debug repo that includes an 'image' column with the PNG content")
    p.add_argument("--train_split", default="train", help="Split to read from source train dataset (default: train)")
    p.add_argument("--test_split", default="train", help="Split to read from source test dataset (default: train)")
    p.add_argument("--overwrite_eval_frames", action="store_true", help="Overwrite existing mid-frame PNGs for test set")
    p.add_argument("--min_sec_filter", type=int, default=-1, help="If >=0, exclude frames with sec < this value when building the TRAIN frame dataset")
    return p.parse_args()


def _collect_unique_videos(ds) -> Set[str]:
    uniq: Set[str] = set()
    if "videos" not in ds.column_names:
        raise ValueError("Dataset must contain a 'videos' column with list[str] of relative paths.")
    for row in ds:
        vids = row.get("videos", []) or []
        if isinstance(vids, str):
            vids = [vids]
        for v in vids:
            if isinstance(v, str):
                uniq.add(v)
    return uniq


def _classification_dir_for(rel_video: str) -> Tuple[PurePosixPath, str]:
    p = PurePosixPath(rel_video)
    if len(p.parts) < 3 or p.parts[0] != "ecva":
        raise ValueError(f"Unexpected relative path (needs to start with ecva/...): {rel_video}")
    folder = p.parts[1]
    rel_dir = PurePosixPath("ecva") / f"{folder}_classification"
    return rel_dir, p.stem


def _label_from_rel_path(rel_path: str) -> str:
    if "abnormal_video_classification" in rel_path:
        return "abnormal"
    return "normal"


def _parse_sec_from_name(name: str) -> Optional[int]:
    base, ext = os.path.splitext(name)
    parts = base.rsplit("_", 2)
    if len(parts) < 3:
        return None
    sec_str = parts[1]
    if not sec_str.isdigit():
        return None
    try:
        return int(sec_str)
    except Exception:
        return None


def _gather_images_for_videos(
    videos: Iterable[str],
    media_base: str,
    exclude_suffix: Optional[str] = None,
    min_sec_filter: Optional[int] = None,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    seen_rel: Set[str] = set()
    for rel_video in sorted(videos):
        try:
            rel_dir, stem = _classification_dir_for(rel_video)
        except Exception as e:
            print(f"[WARN] Skip invalid path {rel_video}: {e}")
            continue

        abs_dir = os.path.join(media_base, rel_dir.as_posix())
        if not os.path.isdir(abs_dir):
            print(f"[WARN] Missing classification dir (skip): {abs_dir}")
            continue

        # Gather files that belong to this video by stem prefix
        try:
            for name in os.listdir(abs_dir):
                if not name.lower().endswith(".png"):
                    continue
                if not name.startswith(stem + "_"):
                    continue
                if exclude_suffix and name.endswith(exclude_suffix):
                    continue
                if min_sec_filter is not None and min_sec_filter >= 0:
                    sec = _parse_sec_from_name(name)
                    if sec is not None and sec < min_sec_filter:
                        continue
                abs_path = os.path.join(abs_dir, name)
                rel_path = os.path.relpath(abs_path, media_base)
                rel_path_posix = PurePosixPath(rel_path).as_posix()
                if rel_path_posix in seen_rel:
                    continue
                seen_rel.add(rel_path_posix)
                rows.append({
                    "images": rel_path_posix,
                    "gt": _label_from_rel_path(rel_path_posix),
                })
        except Exception as e:
            print(f"[WARN] Error listing {abs_dir}: {e}")
            continue

    return rows


def _ensure_mid_frame(rel_video: str, video_media_base: str, media_base: str, overwrite: bool) -> Optional[str]:
    """Ensure a single mid-frame PNG exists for the given video.

    Returns the relative images path (from media_base) if created or already exists,
    else None on failure.
    """
    try:
        rel_dir, stem = _classification_dir_for(rel_video)
    except Exception as e:
        print(f"[WARN] Skip invalid path {rel_video}: {e}")
        return None

    abs_video = os.path.join(video_media_base, rel_video)
    abs_out_dir = os.path.join(media_base, rel_dir.as_posix())
    os.makedirs(abs_out_dir, exist_ok=True)
    out_name = f"{stem}_MID.png"
    abs_out = os.path.join(abs_out_dir, out_name)
    rel_out = PurePosixPath(os.path.relpath(abs_out, media_base)).as_posix()

    if os.path.exists(abs_out) and not overwrite:
        return rel_out

    cap = cv2.VideoCapture(abs_video)
    if not cap.isOpened():
        print(f"[WARN] Failed to open video: {abs_video}")
        return None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    mid_idx = max(0, frame_count // 2)
    if frame_count <= 0:
        mid_idx = 0

    if mid_idx > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid_idx)
    ok, frame = cap.read()
    if not ok or frame is None:
        # fallback: try to read first frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        print(f"[WARN] Could not decode mid frame for: {abs_video}")
        return None

    try:
        ok_write = cv2.imwrite(abs_out, frame)
        if not ok_write:
            print(f"[WARN] Failed to write: {abs_out}")
            return None
    except Exception as e:
        print(f"[WARN] Exception writing {abs_out}: {e}")
        return None

    return rel_out


def _push_rows(rows: List[Dict[str, str]], repo_id: str):
    if not rows:
        print(f"[push] No rows to push for {repo_id}")
        return
    api = HfApi()
    # Create repo if it doesn't exist.
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

    ds = Dataset.from_list(rows)
    print(f"[push] Pushing {len(ds)} rows to {repo_id} ...")
    ds.push_to_hub(repo_id)
    print(f"[push] Done: {repo_id}")


def _push_rows_with_image(rows: List[Dict[str, str]], repo_id: str, media_base: str):
    if not rows:
        print(f"[push] No rows to push for {repo_id}")
        return
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

    # Build rows with an 'image' path resolved to abs path for upload
    rows2: List[Dict[str, str]] = []
    for r in rows:
        abs_img = os.path.join(media_base, r["images"])
        rows2.append({**r, "image": abs_img})

    ds = Dataset.from_list(rows2)
    ds = ds.cast_column("image", Image())
    print(f"[push] Pushing {len(ds)} rows with image column to {repo_id} ...")
    ds.push_to_hub(repo_id)
    print(f"[push] Done: {repo_id}")


def main():
    args = parse_args()

    print(f"[build] Loading source train dataset: {args.train_dataset} split={args.train_split}")
    train_src = load_dataset(args.train_dataset, split=args.train_split)
    print(f"[build] Loading source test dataset: {args.test_dataset} split={args.test_split}")
    test_src = load_dataset(args.test_dataset, split=args.test_split)

    train_vids = _collect_unique_videos(train_src)
    test_vids = _collect_unique_videos(test_src)
    print(f"[build] Unique videos - train: {len(train_vids)}, test: {len(test_vids)}")

    # Train: gather all frames (excluding _MID)
    minsec = args.min_sec_filter if args.min_sec_filter is not None and args.min_sec_filter >= 0 else None
    train_rows = _gather_images_for_videos(train_vids, args.media_base, exclude_suffix="_MID.png", min_sec_filter=minsec)

    # Test: ensure a single mid-frame per video and build rows
    video_media_base = args.video_media_base or args.media_base
    test_rows: List[Dict[str, str]] = []
    for rel_video in sorted(test_vids):
        rel_img = _ensure_mid_frame(rel_video, video_media_base, args.media_base, args.overwrite_eval_frames)
        if not rel_img:
            continue
        test_rows.append({
            "images": rel_img,
            "gt": _label_from_rel_path(rel_img),
        })

    print("[build] Train rows by label:")
    t_abn = sum(1 for r in train_rows if r["gt"] == "abnormal")
    t_nor = len(train_rows) - t_abn
    print(f"  normal={t_nor} abnormal={t_abn} total={len(train_rows)}")

    print("[build] Test rows by label:")
    v_abn = sum(1 for r in test_rows if r["gt"] == "abnormal")
    v_nor = len(test_rows) - v_abn
    print(f"  normal={v_nor} abnormal={v_abn} total={len(test_rows)}")

    _push_rows(train_rows, args.out_train_repo)
    _push_rows(test_rows, args.out_test_repo)
    if args.out_test_repo_with_image:
        _push_rows_with_image(test_rows, args.out_test_repo_with_image, args.media_base)


if __name__ == "__main__":
    main()
