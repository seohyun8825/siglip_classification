import argparse
import os
import sys
import math
from pathlib import Path, PurePosixPath
import time
from typing import Dict, Iterable, List, Set, Tuple

import cv2
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed


def parse_args():
    parser = argparse.ArgumentParser(description="Extract frames from videos listed in an HF dataset.")
    parser.add_argument("--dataset", required=True, help="HF dataset repo id (e.g., happy8825/train_ecva_clean_no_tag)")
    parser.add_argument("--media_base", required=True, help="Absolute base directory that contains media (e.g., /hub_data3/seohyun)")
    parser.add_argument("--fps", type=float, default=2.0, help="Target sampling FPS (default: 2.0)")
    parser.add_argument("--split", default="train", help="Split name to read from HF dataset (default: train)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing images if present")
    parser.add_argument("--output_root", default=None, help="Absolute directory where classification folders live (e.g., /hub_data4/seohyun/ecva). If omitted, defaults to <media_base>/ecva")
    parser.add_argument("--max_workers", type=int, default=8, help="Max parallel workers for extraction (default: 8)")
    parser.add_argument("--min_sec", type=int, default=0, help="Skip frames with sec < min_sec (default: 0)")
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


def _classification_dir_for(rel_video: str, output_root: str) -> Tuple[str, str]:
    """Return (abs_dir, rel_dir) for the classification output directory.

    Expects input relative videos like ecva/<folder>/<name>.mp4.
    We write frames under: <output_root>/<folder>_classification/
    and logical rel_dir is: ecva/<folder>_classification
    """
    p = PurePosixPath(rel_video)
    if len(p.parts) < 3 or p.parts[0] != "ecva":
        raise ValueError(f"Unexpected relative path (needs to start with ecva/...): {rel_video}")
    folder = p.parts[1]  # abnormal_video | after_incident | normal_video
    rel_dir = PurePosixPath("ecva") / f"{folder}_classification"
    abs_dir = os.path.join(output_root, f"{folder}_classification")
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
    min_sec: int,
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

            if sec >= min_sec:
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
    if args.output_root is None:
        args.output_root = os.path.join(args.media_base, "ecva")
    os.makedirs(args.output_root, exist_ok=True)

    print(f"[convert_video] Loading dataset: {args.dataset} split={args.split}")
    ds = load_dataset(args.dataset, split=args.split)
    unique_videos = _collect_unique_videos(ds)
    total_tasks = len(unique_videos)
    print(f"[convert_video] Found {total_tasks} unique video paths in dataset")
    print(f"[convert_video] Using max_workers={args.max_workers} output_root={args.output_root}")

    processed = 0
    skipped_missing = 0
    total_written = 0
    total_skipped_exist = 0
    per_out_dir_counts: Dict[str, int] = {}

    def worker(rel_video: str):
        try:
            abs_video = os.path.join(args.media_base, rel_video)
            abs_out_dir, rel_out_dir = _classification_dir_for(rel_video, args.output_root)
        except Exception as e:
            return {"processed": 0, "missing": 0, "written": 0, "skipped": 0, "by_dir": {}, "warn": f"invalid path {rel_video}: {e}"}

        if not os.path.isfile(abs_video):
            return {"processed": 0, "missing": 1, "written": 0, "skipped": 0, "by_dir": {}, "warn": f"Missing video file: {abs_video}"}

        os.makedirs(abs_out_dir, exist_ok=True)
        stem = Path(rel_video).stem

        written, skipped_exist = _extract_frames(
            abs_video,
            abs_out_dir,
            stem,
            target_fps=args.fps,
            overwrite=args.overwrite,
            min_sec=args.min_sec,
        )
        return {"processed": 1, "missing": 0, "written": written, "skipped": skipped_exist, "by_dir": {rel_out_dir: written}, "video": rel_video, "out_dir": rel_out_dir}

    # Parallel execution
    futures = []
    last_report = time.time()
    report_every = max(1, total_tasks // 50)  # ~2% increments
    with ThreadPoolExecutor(max_workers=max(1, args.max_workers)) as ex:
        for rel in sorted(unique_videos):
            futures.append(ex.submit(worker, rel))
        for fut in as_completed(futures):
            res = fut.result()
            if warn := res.get("warn"):
                print(f"[WARN] {warn}")
            processed += res.get("processed", 0)
            skipped_missing += res.get("missing", 0)
            total_written += res.get("written", 0)
            total_skipped_exist += res.get("skipped", 0)
            for k, v in res.get("by_dir", {}).items():
                per_out_dir_counts[k] = per_out_dir_counts.get(k, 0) + v

            # Optional per-video line with generated image count
            if res.get("processed", 0) == 1:
                vname = res.get("video", "")
                odir = res.get("out_dir", "")
                w = res.get("written", 0)
                s = res.get("skipped", 0)
                if vname:
                    print(f"[convert_video] {vname} -> {odir} +{w} imgs (exist_skip {s}) total_generated={total_written}")

            now = time.time()
            if (
                processed % report_every == 0
                or (now - last_report) >= 2.0
                or processed == total_tasks
            ):
                pct = (processed / total_tasks * 100.0) if total_tasks else 100.0
                print(
                    f"[convert_video] Progress {processed}/{total_tasks} ({pct:.1f}%) "
                    f"written={total_written} missing={skipped_missing} exist_skip={total_skipped_exist}"
                )
                last_report = now

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
