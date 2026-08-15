"""Benchmark the shipped detector stage on a synthetic scene.

Measures :class:`detector.detector.PlateDetector` against the ``<25ms``
detection budget in ``CLAUDE.md``, using the same synthetic scene
``bench_recognize.py`` renders (see its ``render_scene()`` and
``PLATE_BOX``). Requires real weights at ``models/detector/best.pt`` --
run ``scripts/train_detector.py`` first.

The input is *rendered text on a flat background, not a photograph*. Results
bound the model's speed, not its real-world accuracy -- accuracy is reported
separately in ``docs/experiments/detector-v0.1.md`` from Ultralytics' own
validation mAP.

Run from the repository root::

    poetry run python docs/benchmark/bench_detect.py
"""

import argparse
import platform
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bench_recognize import PLATE_BOX, render_scene  # noqa: E402

from app.core.config import Settings  # noqa: E402
from detector.detector import PlateDetector  # noqa: E402

DETECTION_BUDGET_MS = 25.0


def main() -> None:
    """Benchmark the detector stage and print a latency report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repeats", type=int, default=15, help="warm iterations to time"
    )
    args = parser.parse_args()

    settings = Settings()
    if not Path(settings.detector_model_path).is_file():
        sys.exit(
            f"Missing {settings.detector_model_path}.\n\n"
            "Run scripts/train_detector.py first to produce real weights."
        )

    scene = render_scene()
    detector = PlateDetector(
        model_path=settings.detector_model_path,
        conf_threshold=settings.detector_conf_threshold,
    )

    started = time.perf_counter()
    detections = detector.detect(scene)
    cold_ms = (time.perf_counter() - started) * 1000

    timings: list[float] = []
    for _ in range(args.repeats):
        stage = time.perf_counter()
        detections = detector.detect(scene)
        timings.append((time.perf_counter() - stage) * 1000)

    median = statistics.median(timings)

    print(f"machine     : {platform.platform()} / {platform.processor()}")
    print(f"weights     : {settings.detector_model_path}")
    print(f"scene       : {scene.shape[1]}x{scene.shape[0]}  known box {PLATE_BOX}")
    print(f"found       : {len(detections)} box(es)")
    for detection in detections:
        print(
            f"  ({detection.x1}, {detection.y1}, {detection.x2}, {detection.y2})"
            f"  conf={detection.confidence:.3f}"
        )
    print(f"cold start  : {cold_ms:.0f} ms  (loads detector weights)")
    print(
        f"warm median : {median:.0f} ms  min {min(timings):.0f}  max {max(timings):.0f}"
        f"  n={args.repeats}"
    )
    if median > DETECTION_BUDGET_MS:
        print(
            f"budget      : <{DETECTION_BUDGET_MS:.0f} ms"
            f"  -> {median / DETECTION_BUDGET_MS:.1f}x OVER"
        )
    else:
        print(f"budget      : <{DETECTION_BUDGET_MS:.0f} ms  -> within budget")


if __name__ == "__main__":
    main()
