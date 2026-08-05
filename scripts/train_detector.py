"""Fine-tune a pretrained YOLOv8n detector on the ThailLand Plates dataset.

Produces the project's first real ``models/detector/best.pt``. Every
``/detect`` and ``/recognize`` call returns 503 until this file exists
(``app/core/config.py``'s ``detector_model_path`` default, checked by
``detector.detector.PlateDetector``).

Fine-tunes Ultralytics' COCO-pretrained ``yolov8n.pt`` rather than training
from scratch -- the ~200-image dataset is far too small to train a detector
from random initialization.

Requires the ThailLand Plates dataset already extracted to
``datasets/raw/thailand-license-plates-v1/`` (manual download, "YOLOv8"
export format, from
https://universe.roboflow.com/thailland-plates/thailand-license-plates --
Roboflow requires a login, so this step is not scripted).

Run from the repository root::

    poetry run python scripts/train_detector.py
    poetry run python scripts/train_detector.py --epochs 50 --batch 8

After it finishes, copy the printed summary block into
``docs/experiments/detector-v0.1.md``.
"""

import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = REPO_ROOT / "datasets" / "raw" / "thailand-license-plates-v1" / "data.yaml"
OUTPUT_WEIGHTS = REPO_ROOT / "models" / "detector" / "best.pt"

DEFAULT_EPOCHS = 100
DEFAULT_BATCH = 16
DEFAULT_IMGSZ = 640
DEFAULT_PATIENCE = 20


def parse_args() -> argparse.Namespace:
    """Parse command-line overrides for the training run.

    Returns:
        The parsed arguments (epochs, batch, imgsz, patience).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    return parser.parse_args()


def git_commit_hash() -> str:
    """Return the short hash of the current commit.

    Returns:
        The 7-character short SHA of ``HEAD``.
    """
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def main() -> None:
    """Fine-tune the detector and print a training/validation summary."""
    args = parse_args()

    if not DATA_YAML.exists():
        sys.exit(
            f"Missing {DATA_YAML}.\n\n"
            "Download the 'YOLOv8' export from "
            "https://universe.roboflow.com/thailland-plates/thailand-license-plates "
            f"and extract it to {DATA_YAML.parent}/ before running this script."
        )

    print(f"machine     : {platform.platform()} / {platform.processor()}")
    print(f"data        : {DATA_YAML.relative_to(REPO_ROOT)}")
    print(f"epochs      : {args.epochs}  batch: {args.batch}  imgsz: {args.imgsz}")
    print()

    model = YOLO("yolov8n.pt")
    model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device="cpu",
        patience=args.patience,
    )

    trainer = model.trainer
    OUTPUT_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(trainer.best, OUTPUT_WEIGHTS)

    metrics = model.val(data=str(DATA_YAML))

    print()
    print("=== training complete ===")
    print(f"run dir     : {trainer.save_dir.relative_to(REPO_ROOT)}")
    weights_path = OUTPUT_WEIGHTS.relative_to(REPO_ROOT)
    print(f"weights     : {weights_path} (gitignored, local only)")
    print(f"epochs run  : {trainer.epoch + 1} / {args.epochs} requested")
    print(f"mAP50       : {metrics.box.map50:.4f}")
    print(f"mAP50-95    : {metrics.box.map:.4f}")
    print(f"precision   : {metrics.box.mp:.4f}")
    print(f"recall      : {metrics.box.mr:.4f}")
    print(f"git commit  : {git_commit_hash()}")
    print()
    print("Copy the block above into docs/experiments/detector-v0.1.md")


if __name__ == "__main__":
    main()
