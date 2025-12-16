# SigLIP2 ECVA Classification Pipeline

This folder provides a simple 3–4 bash step pipeline to go from videos → frame extraction → HF dataset push → SigLIP2 train/eval.

## Prereqs
- Python 3.10+
- Install deps: `pip install -r requirements.txt`
- HF auth: set `HF_TOKEN` or run `huggingface-cli login`

## Data assumptions
- Source HF datasets contain a column `videos` with relative paths like `ecva/normal_video/353_normal.mp4`.
- Actual media root is given via `--media_base` (e.g., `/hub_data3/seohyun`). The real file becomes `/hub_data3/seohyun/ecva/normal_video/353_normal.mp4`.
- Class mapping by folder:
  - `ecva/abnormal_video/...` → abnormal
  - `ecva/after_incident/...` → normal
  - `ecva/normal_video/...` → normal

## Scripts (simple, edit-and-run)
- `scripts/0_convert_video.bash` — edit DATASET/MEDIA_BASE/FPS and run
- `scripts/1_push_data.bash` — edit TRAIN/TEST dataset ids, MEDIA_BASE, output repos
- `scripts/2_train.bash` — edit TRAIN_REPO, MEDIA_BASE, OUTPUT_DIR, hyperparams
- `scripts/3_eval.bash` — edit MODEL_DIR, TEST_REPO, MEDIA_BASE

## 1) Convert Video → Frames
Edit variables inside `scripts/0_convert_video.bash` then run:
```
bash scripts/0_convert_video.bash
```
This creates:
- `ecva/abnormal_video_classification/`
- `ecva/after_incident_classification/`
- `ecva/normal_video_classification/`
under the given `media_base` and writes PNG frames named `{stem}_{sec:06d}_{k}.png`.

## 2) Build + Push HF datasets
Edit variables inside `scripts/1_push_data.bash` then run:
```
bash scripts/1_push_data.bash
```
This expands videos → images and pushes rows schema:
- `images`: relative path like `ecva/abnormal_video_classification/353_normal_000002_0.png`
- `gt`: `normal` or `abnormal`

## 3) Train
Edit variables inside `scripts/2_train.bash` then run:
```
bash scripts/2_train.bash
```
Uses `google/siglip2-base-patch16-224` with `num_labels=2`. Label ids: normal=0, abnormal=1.

Optional eval during training:
```
--eval_repo happy8825/siglip_test
```

## 4) Evaluate
Edit variables inside `scripts/3_eval.bash` then run:
```
bash scripts/3_eval.bash
```
Outputs accuracy and F1; also saves `eval_results.json` if `--out` is given.

## Notes
- Missing videos are skipped with warnings.
- If frames exist, default is skip. Use `--overwrite` in conversion to regenerate.
- Frame sampling is time-based at target fps; filenames avoid collisions using `sec` + `k`.
