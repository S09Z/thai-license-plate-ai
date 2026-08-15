# v0.1 — YOLOv8n fine-tuned on ThailLand Plates

First real `models/detector/best.pt`. Design: `docs/superpowers/specs/2026-08-05-detector-v0.1-training-design.md`. Script: `scripts/train_detector.py`.

## Dataset

- ThailLand Plates (single `license-plate` class, whole-plate bounding boxes)
- https://universe.roboflow.com/ittichai-boonyarakthunya/thailand-license-plates-cu3sw
- CC BY 4.0
- 200 images total, `nc: 1`, `names: ['license-plate']`
- Split: 141 train / 52 valid / 7 test (Roboflow-provided `train/valid/test` + `data.yaml`)

**Caveat:** 200 images is thin, and the validation split is small (52 images). The mAP below is
noisy and should not be read as a production-quality accuracy number — it is a first honest
baseline, not a claim of real-world robustness.

## Hyperparameters

- Base checkpoint: `yolov8n.pt` (COCO-pretrained)
- Epochs requested: 100
- Epochs run: 74 (early stopping triggered by `patience`; best checkpoint observed at epoch 54)
- Batch: 16
- Image size: 640
- Patience: 20
- Optimizer: auto (resolved: SGD-family, `lr0: 0.01`, `lrf: 0.01`, `momentum: 0.937`, `weight_decay: 0.0005`)
- Device: cpu

## Metrics (validation on `best.pt`, 52 images / 52 instances)

- mAP50: 0.995
- mAP50-95: 0.841
- Precision: 1.000
- Recall: 0.997

## Provenance

- Training script commit: `9a42bfb`
- Machine: macOS-26.5.2-arm64-arm-64bit / arm (Apple M3, CPU-only)
- Weights location: `models/detector/best.pt` (gitignored — not in commit; `.gitignore`'s "store
  release models externally, not in git")
- Training ran in two sessions (interrupted, then resumed via Ultralytics' native
  `model.train(resume=True)` from `runs/detect/train/weights/last.pt`) rather than a single
  uninterrupted `scripts/train_detector.py` invocation. `best.pt` was copied to
  `models/detector/best.pt` manually after resume completed. Metrics above are Ultralytics'
  own final validation pass on `best.pt`, printed at the end of training.

## Note: a prior run used the wrong dataset

An earlier attempt at this task downloaded a different Roboflow project
(`thailand-license-plate-recognition`, 11 classes for individual digit/dash characters) instead of
the single-class whole-plate dataset this spec calls for. That mismatch was caught before it was
mistaken for this v0.1 result. The mistrained character-detection model was kept (not discarded)
at `models/char-detector-v0/`, since a future character-level extraction phase (see this spec's
"Deferred: OCR fine-tuning" section) can use it as a starting point rather than training from
scratch again.

## Next

- `docs/benchmark/bench_detect.py` measures real detection latency against these weights.
- If mAP is judged insufficient, the next lever is Approach (C) from the design spec: merging
  additional public datasets before retraining.
