"""Measure per-stage ``/recognize`` latency on the real-plate eval set.

Unlike ``bench_recognize.py`` (a synthetic scene with a fixed box, so detection
is never exercised), this runs the *real* detector + perspective + OCR +
post/RAG over the hand-verified photographs in ``eval/plates.jsonl`` and reports
median latency per stage. It is the Phase 15b companion to
``bench_recognize_accuracy.py``: same images, timing instead of correctness.

A second pass isolates the OCR stage into text *detection* vs *recognition* by
running the recognition head alone (``TextRecognition``) on the same crops. That
pass runs only after every detector call is done, because interleaving paddle's
recognition-head init with torch's NMS corrupts tensors in this environment.

Run from the repository root::

    poetry run python docs/benchmark/bench_recognize_latency.py
    poetry run python docs/benchmark/bench_recognize_latency.py --repeats 3
"""

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.core.config import Settings  # noqa: E402
from app.services.detection_service import get_detector  # noqa: E402
from app.services.ocr_service import get_recognizer  # noqa: E402
from detector.pipelines.perspective import correct_perspective  # noqa: E402
from ocr.reading import split_reading  # noqa: E402
from postprocess.plate import normalize_plate_text  # noqa: E402
from rag.validator import resolve_province  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_FILE = REPO_ROOT / "eval" / "plates.jsonl"
IMAGE_DIRS = [
    REPO_ROOT / "datasets/raw/thailand-license-plates-v1/valid/images",
    REPO_ROOT / "datasets/raw/thailand-license-plates-v1/test/images",
]
TOTAL_BUDGET_MS = 100.0


def _find_image(name: str) -> Path | None:
    """Locate an eval image by file name across the dataset splits."""
    for directory in IMAGE_DIRS:
        candidate = directory / name
        if candidate.is_file():
            return candidate
    return None


def _load_images() -> list[np.ndarray]:
    """Decode every eval image that can be found on disk."""
    images = []
    for line in EVAL_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        name = json.loads(line)["img"]
        path = _find_image(name)
        if path is None:
            continue
        arr = np.frombuffer(path.read_bytes(), np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is not None:
            images.append(image)
    return images


def _elapsed_ms(since: float) -> float:
    """Milliseconds since a ``perf_counter`` reading."""
    return (time.perf_counter() - since) * 1000


def main() -> None:
    """Time each pipeline stage on the real eval set and print a report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repeats", type=int, default=3, help="Warm passes over the set to time."
    )
    args = parser.parse_args()

    settings = Settings()
    crop_size = (settings.plate_crop_width, settings.plate_crop_height)
    images = _load_images()
    if not images:
        print("no eval images found; check the dataset is present")
        return

    detector = get_detector()
    recognizer = get_recognizer()

    stages: dict[str, list[float]] = {
        "detect": [],
        "perspective": [],
        "ocr": [],
        "postprocess+rag": [],
    }
    crops: list[np.ndarray] = []

    # Warm both engines once (weight load / graph build), then time.
    for image in images:
        dets = detector.detect(image)
        if dets:
            crop = correct_perspective(image, dets[0], output_size=crop_size)
            recognizer.recognize(crop)
        break

    for _ in range(args.repeats):
        for image in images:
            t = time.perf_counter()
            dets = detector.detect(image)
            stages["detect"].append(_elapsed_ms(t))
            if not dets:
                continue
            box = max(dets, key=lambda d: (d.x2 - d.x1) * (d.y2 - d.y1))

            t = time.perf_counter()
            crop = correct_perspective(image, box, output_size=crop_size)
            stages["perspective"].append(_elapsed_ms(t))
            crops.append(crop)

            t = time.perf_counter()
            lines = recognizer.recognize(crop)
            stages["ocr"].append(_elapsed_ms(t))

            t = time.perf_counter()
            reading = split_reading(lines)
            normalize_plate_text(reading.plate_text)
            resolve_province(reading.province_candidates)
            stages["postprocess+rag"].append(_elapsed_ms(t))

    def med(name: str) -> float:
        return statistics.median(stages[name]) if stages[name] else 0.0

    total = sum(med(s) for s in stages)

    print(f"machine : {platform.platform()} / {platform.processor()}")
    print(f"images  : {len(images)}  crop {crop_size}  repeats {args.repeats}")
    print(
        f"det cfg : text_det_limit={settings.ocr_det_limit_type}/"
        f"{settings.ocr_det_limit_side_len}"
    )
    print("\nper-stage median (ms), full shipped pipeline:")
    for name in stages:
        print(f"  {name:<16}: {med(name):7.1f}")
    print(f"  {'TOTAL':<16}: {total:7.1f}")
    if total <= TOTAL_BUDGET_MS:
        print(f"\nmeets <{TOTAL_BUDGET_MS:.0f}ms budget")
    else:
        print(
            f"\n{total / TOTAL_BUDGET_MS:.1f}x over the <{TOTAL_BUDGET_MS:.0f}ms budget"
        )

    # OCR decomposition: recognition head alone, on the crops just gathered.
    # Runs only now that every torch detector call is finished.
    from paddleocr import TextRecognition

    rec = TextRecognition(model_name="th_PP-OCRv5_mobile_rec")
    rec.predict(crops[0])  # warm
    rec_only: list[float] = []
    for crop in crops:
        t = time.perf_counter()
        list(rec.predict(crop))
        rec_only.append(_elapsed_ms(t))
    rec_med = statistics.median(rec_only)
    ocr_med = med("ocr")
    print("\nOCR decomposition (why OCR dominates):")
    print(f"  recognition head only : {rec_med:7.1f} ms")
    print(f"  text detection (rest) : {ocr_med - rec_med:7.1f} ms")
    print(f"  => detection is ~{(ocr_med - rec_med) / ocr_med:.0%} of OCR latency")


if __name__ == "__main__":
    main()
