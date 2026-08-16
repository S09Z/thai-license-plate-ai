"""Benchmark the YuNet face detection stage on synthetic 720p frames.

Measures :class:`face.detector.FaceDetector` against the ``<25ms`` detection
budget in ``CLAUDE.md``. Requires the ONNX model -- run ``make
fetch-face-model`` first.

Two frames are timed, flat and textured, because the Haar cascade this model
was chosen over swung 9x between them (6ms vs 56ms, see
``docs/benchmark/face-phase10.md``). YuNet is a fixed-size CNN and should
*not* show that spread; the second frame exists to check that claim rather
than assume it.

The frames contain **no actual faces**. This bounds the model's speed, not its
accuracy -- no face has been put in front of this pipeline, and accuracy stays
unmeasured until real photographs are available.

Run from the repository root::

    poetry run python docs/benchmark/bench_face.py
"""

import argparse
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.core.config import Settings  # noqa: E402
from detector.detector import PlateDetector  # noqa: E402
from face.detector import FaceDetector  # noqa: E402

DETECTION_BUDGET_MS = 25.0
FRAME_HEIGHT = 720
FRAME_WIDTH = 1280


def flat_frame() -> np.ndarray:
    """Return a uniform 720p BGR frame."""
    return np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), 96, dtype=np.uint8)


def textured_frame() -> np.ndarray:
    """Return a 720p BGR frame of deterministic noise."""
    generator = np.random.default_rng(seed=0)
    return generator.integers(
        0, 256, size=(FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8
    )


def _time(detect: object, frame: np.ndarray, repeats: int) -> list[float]:
    """Time ``repeats`` warm calls of ``detect`` on ``frame``, in ms."""
    timings: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        detect(frame)  # type: ignore[operator]
        timings.append((time.perf_counter() - started) * 1000)
    return timings


def _report(label: str, timings: list[float]) -> float:
    """Print a median/min/max line and return the median."""
    median = statistics.median(timings)
    print(
        f"{label:<22}: {median:6.1f} ms  min {min(timings):5.1f}"
        f"  max {max(timings):5.1f}  n={len(timings)}"
    )
    return median


def main() -> None:
    """Benchmark the face stage and print a latency report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repeats", type=int, default=15, help="warm iterations to time"
    )
    args = parser.parse_args()

    settings = Settings()
    if not Path(settings.face_model_path).is_file():
        sys.exit(
            f"Missing {settings.face_model_path}.\n\n"
            "Run `make fetch-face-model` first to download the YuNet model."
        )

    detector = FaceDetector(
        model_path=settings.face_model_path,
        conf_threshold=settings.face_conf_threshold,
    )

    flat = flat_frame()
    textured = textured_frame()

    started = time.perf_counter()
    faces = detector.detect(flat)
    cold_ms = (time.perf_counter() - started) * 1000

    print(f"machine     : {platform.platform()} / {platform.processor()}")
    print(f"model       : {settings.face_model_path}")
    print(f"frame       : {FRAME_WIDTH}x{FRAME_HEIGHT}  (no faces present)")
    print(f"found       : {len(faces)} face(s) on the flat frame")
    print(f"cold start  : {cold_ms:.0f} ms  (loads the ONNX model)")
    print()

    flat_median = _report(
        "face, flat frame", _time(detector.detect, flat, args.repeats)
    )
    textured_median = _report(
        "face, textured frame", _time(detector.detect, textured, args.repeats)
    )
    worst = max(flat_median, textured_median)

    if worst > DETECTION_BUDGET_MS:
        print(
            f"budget      : <{DETECTION_BUDGET_MS:.0f} ms"
            f"  -> {worst / DETECTION_BUDGET_MS:.1f}x OVER"
        )
    else:
        print(f"budget      : <{DETECTION_BUDGET_MS:.0f} ms  -> within budget")

    # The camera tick issues both requests concurrently, so its floor is the
    # slower stage, not the sum. Both numbers are printed to make that visible.
    if Path(settings.detector_model_path).is_file():
        plate_detector = PlateDetector(
            model_path=settings.detector_model_path,
            conf_threshold=settings.detector_conf_threshold,
        )
        plate_detector.detect(textured)
        print()
        plate_median = _report(
            "plate, textured frame",
            _time(plate_detector.detect, textured, args.repeats),
        )
        concurrent = max(plate_median, textured_median)
        print(f"{'tick, concurrent':<22}: {concurrent:6.1f} ms")
        print(f"{'tick, if serial':<22}: {plate_median + textured_median:6.1f} ms")
    else:
        print()
        print(f"plate stage skipped: no weights at {settings.detector_model_path}")


if __name__ == "__main__":
    main()
