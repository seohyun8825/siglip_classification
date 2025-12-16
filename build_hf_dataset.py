import argparse
import os
from pathlib import Path, PurePosixPath
from typing import Dict, Iterable, List, Set, Tuple

from datasets import Dataset, load_dataset
from huggingface_hub import HfApi


def parse_args():
    p = argparse.ArgumentParser(description="Build and push frame-level HF datasets from extracted images.")
    p.add_argument("--train_dataset", required=True, help="Source HF dataset repo with videos for train")
    p.add_argument("--test_dataset", required=True, help="Source HF dataset repo with videos for test")
    p.add_argument("--media_base", required=True, help="Absolute base directory containing extracted images under ecva/*_classification/")
    p.add_argument("--out_train_repo", required=True, help="Destination HF dataset repo for frames (e.g., happy8825/siglip_train)")
    p.add_argument("--out_test_repo", required=True, help="Destination HF dataset repo for frames (e.g., happy8825/siglip_test)")
    p.add_argument("--train_split", default="train", help="Split to read from source train dataset (default: train)")
    p.add_argument("--test_split", default="train", help="Split to read from source test dataset (default: train)")
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


def _gather_images_for_videos(videos: Iterable[str], media_base: str) -> List[Dict[str, str]]:
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


def main():
    args = parse_args()

    print(f"[build] Loading source train dataset: {args.train_dataset} split={args.train_split}")
    train_src = load_dataset(args.train_dataset, split=args.train_split)
    print(f"[build] Loading source test dataset: {args.test_dataset} split={args.test_split}")
    test_src = load_dataset(args.test_dataset, split=args.test_split)

    train_vids = _collect_unique_videos(train_src)
    test_vids = _collect_unique_videos(test_src)
    print(f"[build] Unique videos - train: {len(train_vids)}, test: {len(test_vids)}")

    train_rows = _gather_images_for_videos(train_vids, args.media_base)
    test_rows = _gather_images_for_videos(test_vids, args.media_base)

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


if __name__ == "__main__":
    main()

