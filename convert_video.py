import argparse
import os
import sys
import math
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Set, Tuple

import cv2
from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Extract frames from videos listed in an HF dataset.")
    parser.add_argument("--dataset", required=True, help="HF dataset repo id (e.g., happy8825/train_ecva_clean_no_tag)")
    parser.add_argument("--media_base", required=True, help="Absolute base directory that contains media (e.g., /hub_data3/seohyun)")
    parser.add_argument("--fps", type=float, default=2.0, help="Target sampling FPS (default: 2.0)")
    parser.add_argument("--split", default="train", help="Split name to read from HF dataset (default: train)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing images if present")
    return parser.parse_args()


def _collect_unique_videos(ds) -> Set[str]:
    uniq: Set[str] = set()
    if "videos" not in ds.column_names:
        raise ValueError("Dataset must contain a 'videos' column with list[str] of relative paths.")
    for row in ds:
        vids = row.get("videos", []) or []
        if isinstance(vids, str):
            vids = [vids]
        for v in vids:
            if not isinstance(v, str):
                continue
            uniq.add(v)
    return uniq


def _classification_dir_for(rel_video: str, media_base: str) -> Tuple[str, str]:
    """Return (abs_dir, rel_dir) for the classification output directory.

    Expects paths like: ecva/abnormal_video/xxx.mp4 or ecva/normal_video/xxx.mp4 or ecva/after_incident/xxx.mp4
    Output: ecva/<folder>_classification/
    """
    p = PurePosixPath(rel_video)
    if len(p.parts) < 3 or p.parts[0] != "ecva":
        raise ValueError(f"Unexpected relative path (needs to start with ecva/...): {rel_video}")
    folder = p.parts[1]  # abnormal_video | after_incident | normal_video
    rel_dir = PurePosixPath("ecva") / f"{folder}_classification"
    abs_dir = os.path.join(media_base, rel_dir.as_posix())
    return abs_dir, rel_dir.as_posix()


def _label_from_rel_dir(rel_dir: str) -> str:
    # Not used in convert step, but keep here for reference.
    if "abnormal_video_classification" in rel_dir:
        return "abnormal"
    return "normal"


def _extract_frames(
    video_path: str,
    out_dir: str,
    video_stem: str,
    target_fps: float,
    overwrite: bool,
) -> Tuple[int, int]:
    """Extract frames sequentially using native fps stepping and emit at target_fps.

    Returns: (num_written, num_skipped_existing)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Failed to open video: {video_path}")
        return 0, 0

    # Native fps may be 0 or NaN depending on container; default to 30.
    native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if not (native_fps and math.isfinite(native_fps) and native_fps > 0):
        native_fps = 30.0

    frame_time = 1.0 / native_fps
    step = 1.0 / target_fps
    next_sample_t = 0.0

    k_per_second: Dict[int, int] = {}
    written = 0
    skipped_exist = 0

    frame_idx = 0
    current_t = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # time of this frame
        if frame_idx == 0:
            current_t = 0.0
        else:
            current_t += frame_time

        # Emit frames at t >= next_sample_t
        while current_t + 1e-9 >= next_sample_t:
            sec = int(math.floor(next_sample_t))
            k = k_per_second.get(sec, 0)
            k_per_second[sec] = k + 1

            filename = f"{video_stem}_{sec:06d}_{k}.png"
            out_path = os.path.join(out_dir, filename)

            if os.path.exists(out_path) and not overwrite:
                skipped_exist += 1
            else:
                # Save PNG
                try:
                    ok_write = cv2.imwrite(out_path, frame)
                    if ok_write:
                        written += 1
                except Exception as e:
                    print(f"[WARN] Failed to write {out_path}: {e}")

            next_sample_t += step
            # Protect against infinite loop if step is tiny
            if step <= 0:
                break

        frame_idx += 1

    cap.release()
    return written, skipped_exist


def main():
    args = parse_args()

    os.makedirs(args.media_base, exist_ok=True)

    print(f"[convert_video] Loading dataset: {args.dataset} split={args.split}")
    ds = load_dataset(args.dataset, split=args.split)
    unique_videos = _collect_unique_videos(ds)
    print(f"[convert_video] Found {len(unique_videos)} unique video paths in dataset")

    processed = 0
    skipped_missing = 0
    total_written = 0
    total_skipped_exist = 0
    per_out_dir_counts: Dict[str, int] = {}

    for rel_video in sorted(unique_videos):
        try:
            abs_video = os.path.join(args.media_base, rel_video)
            abs_out_dir, rel_out_dir = _classification_dir_for(rel_video, args.media_base)
        except Exception as e:
            print(f"[WARN] Skip invalid path {rel_video}: {e}")
            continue

        if not os.path.isfile(abs_video):
            print(f"[WARN] Missing video file: {abs_video}")
            skipped_missing += 1
            continue

        os.makedirs(abs_out_dir, exist_ok=True)

        stem = Path(rel_video).stem

        written, skipped_exist = _extract_frames(
            abs_video,
            abs_out_dir,
            stem,
            target_fps=args.fps,
            overwrite=args.overwrite,
        )

        processed += 1
        total_written += written
        total_skipped_exist += skipped_exist
        per_out_dir_counts[rel_out_dir] = per_out_dir_counts.get(rel_out_dir, 0) + written

    print("[convert_video] Summary")
    print(f"  Processed videos: {processed}")
    print(f"  Missing videos: {skipped_missing}")
    print(f"  Total images written: {total_written}")
    print(f"  Existing images skipped: {total_skipped_exist}")
    print("  By output folder:")
    for k in sorted(per_out_dir_counts.keys()):
        print(f"    {k}: {per_out_dir_counts[k]}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.")
        sys.exit(1)

