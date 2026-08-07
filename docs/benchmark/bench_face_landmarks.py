"""Benchmark the LBF 68-point landmark fit that sits on top of face detection.

Measures :class:`face.landmarks.FaceLandmarker` against the ``<25ms`` detection
budget in ``CLAUDE.md``. Requires the LBF model -- run ``make
fetch-face-landmark-model`` first.

The fit is per-face work, unlike detection, so one face and three faces are
timed separately: the budget question is not "how fast is a fit" but "how many
faces can a 200ms camera tick carry".

Synthetic boxes on a noise frame bound the *cost* of a fit, not its accuracy --
the regression runs the same amount of work wherever it is pointed. Pass
``--image`` with a photograph containing a face to time the real path, detection
included.

Run from the repository root::

    poetry run python docs/benchmark/bench_face_landmarks.py
    poetry run python docs/benchmark/bench_face_landmarks.py --image face.jpg
"""

import argparse
import platform
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.core.config import Settings  # noqa: E402
from detector.detector import Detection  # noqa: E402
from face.detector import FaceDetector  # noqa: E402
from face.landmarks import FaceLandmarker  # noqa: E402

DETECTION_BUDGET_MS = 25.0
FRAME_HEIGHT = 720
FRAME_WIDTH = 1280
# Roughly a head at conversational distance in a 720p frame.
FACE_WIDTH = 140
FACE_HEIGHT = 180


def textured_frame() -> np.ndarray:
    """Return a 720p BGR frame of deterministic noise."""
    generator = np.random.default_rng(seed=0)
    return generator.integers(
        0, 256, size=(FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8
    )


def synthetic_boxes(count: int) -> list[Detection]:
    """Return ``count`` face-sized boxes spread across the frame."""
    return [
        Detection(
            x1=100 + index * 300,
            y1=200,
            x2=100 + index * 300 + FACE_WIDTH,
            y2=200 + FACE_HEIGHT,
            confidence=0.9,
        )
        for index in range(count)
    ]


def _time_fit(
    landmarker: FaceLandmarker,
    frame: np.ndarray,
    boxes: list[Detection],
    repeats: int,
) -> list[float]:
    """Time ``repeats`` warm fits over ``boxes``, in ms."""
    timings: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        landmarker.fit(frame, boxes)
        timings.append((time.perf_counter() - started) * 1000)
    return timings


def _report(label: str, timings: list[float]) -> float:
    """Print a median/min/max line and return the median."""
    median = statistics.median(timings)
    print(
        f"{label:<26}: {median:6.1f} ms  min {min(timings):5.1f}"
        f"  max {max(timings):5.1f}  n={len(timings)}"
    )
    return median


def main() -> None:
    """Benchmark the landmark stage and print a latency report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repeats", type=int, default=15, help="warm iterations to time"
    )
    parser.add_argument(
        "--image", type=Path, help="photograph with a face, for the real path"
    )
    args = parser.parse_args()

    settings = Settings()
    if not Path(settings.face_landmark_model_path).is_file():
        sys.exit(
            f"Missing {settings.face_landmark_model_path}.\n\n"
            "Run `make fetch-face-landmark-model` first to download the LBF model."
        )

    landmarker = FaceLandmarker(model_path=settings.face_landmark_model_path)
    frame = textured_frame()
    one_box = synthetic_boxes(1)

    started = time.perf_counter()
    landmarker.fit(frame, one_box)
    cold_ms = (time.perf_counter() - started) * 1000

    print(f"machine     : {platform.platform()} / {platform.processor()}")
    print(f"model       : {settings.face_landmark_model_path}")
    print(f"frame       : {FRAME_WIDTH}x{FRAME_HEIGHT}  (synthetic boxes)")
    print(f"cold start  : {cold_ms:.0f} ms  (loads the 54MB LBF model)")
    print()

    one = _report("fit, 1 face", _time_fit(landmarker, frame, one_box, args.repeats))
    three = _report(
        "fit, 3 faces",
        _time_fit(landmarker, frame, synthetic_boxes(3), args.repeats),
    )
    print(f"{'per extra face':<26}: {(three - one) / 2:6.1f} ms")
    print()

    if one > DETECTION_BUDGET_MS:
        print(
            f"budget      : <{DETECTION_BUDGET_MS:.0f} ms per face"
            f"  -> {one / DETECTION_BUDGET_MS:.1f}x OVER"
        )
    else:
        print(f"budget      : <{DETECTION_BUDGET_MS:.0f} ms per face -> within budget")

    if args.image is None:
        return

    # The real path: detect first, then fit the boxes detection actually found.
    image = cv2.imread(str(args.image))
    if image is None:
        sys.exit(f"Could not read {args.image}")

    detector = FaceDetector(
        model_path=settings.face_model_path,
        conf_threshold=settings.face_conf_threshold,
    )
    boxes = detector.detect(image)
    print()
    print(f"photograph  : {args.image} {image.shape[1]}x{image.shape[0]}")
    print(f"found       : {len(boxes)} face(s)")

    if not boxes:
        return

    detect_timings = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        detector.detect(image)
        detect_timings.append((time.perf_counter() - started) * 1000)

    detect_median = _report("detect, photograph", detect_timings)
    fit_median = _report(
        "fit, photograph", _time_fit(landmarker, image, boxes, args.repeats)
    )
    print(f"{'detect + fit, serial':<26}: {detect_median + fit_median:6.1f} ms")


if __name__ == "__main__":
    main()
