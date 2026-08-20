"""Benchmark the fast face-detection path against the full-resolution one.

The Phase 13 realtime loop trades a little precision for a much higher frame
rate by downscaling the frame before YuNet runs (``?fast=true``). This measures
how much that buys: an ``n``-pixel longest edge cost, then the corresponding
per-tick budget at a nominal 60 fps camera loop.

Run from the repository root::

    poetry run python docs/benchmark/bench_face_fast.py
    poetry run python docs/benchmark/bench_face_fast.py --repeats 40

Requires ``make fetch-face-model`` first. Mirrors the texture conditioning used
in ``bench_face.py``: YuNet is input-size-bound, not content-bound, so the
textured frame bounds the *cost* of a detection, and the flat/720p split from
Phase 10 is carried over so the numbers stay comparable.
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
from app.services.face_service import _prepare  # noqa: E402
from face.detector import FaceDetector  # noqa: E402

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FRAME_BUDGET_MS = 1000.0 / 60.0  # the 60 fps target, one frame of budget.


def textured_frame() -> np.ndarray:
    """Return a 720p BGR frame of deterministic noise."""
    generator = np.random.default_rng(seed=0)
    return generator.integers(
        0, 256, size=(FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8
    )


def _time_detect(
    detector: FaceDetector, image: np.ndarray, repeats: int
) -> list[float]:
    """Time ``repeats`` warm detections, in ms."""
    timings: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        detector.detect(image)
        timings.append((time.perf_counter() - started) * 1000)
    return timings


def _report(label: str, timings: list[float]) -> float:
    """Print a median/min/max line and return the median."""
    median = statistics.median(timings)
    print(
        f"{label:<34}: {median:6.2f} ms  min {min(timings):5.2f}"
        f"  max {max(timings):5.2f}  n={len(timings)}"
    )
    return median


def main() -> None:
    """Benchmark fast versus full-resolution face detection."""
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
    full = textured_frame()
    fast, sx, sy = _prepare(full, settings.face_fast_max_size)

    detector.detect(fast)  # warm on the smallest input first (cheapest load).

    print(f"machine     : {platform.platform()} / {platform.processor()}")
    print(
        f"frame       : {FRAME_WIDTH}x{FRAME_HEIGHT} full  ->  "
        f"{fast.shape[1]}x{fast.shape[0]} fast"
    )
    print(
        f"fast scale  : x{sx:.2f}, y{sy:.2f} "
        f"(max edge {settings.face_fast_max_size})"
    )
    print()

    full_ms = _report("detect, 720p full", _time_detect(detector, full, args.repeats))
    fast_ms = _report("detect, fast", _time_detect(detector, fast, args.repeats))
    print(f"{'speedup':<29}: {full_ms / fast_ms:5.1f}x")
    print()

    # The loop can only keep the nominal 60fps cadence if the whole tick fits
    # the frame budget; self-scheduling stretches the cadence rather than
    # stacking, so this is the honest ceiling for a loopback round trip.
    fast_fps = 1000.0 / fast_ms
    print(f"{'fast tick ceiling':<26}: {fast_fps:5.1f} fps")
    print(f"{'60fps budget':<30}: 60 fps")
    if fast_ms > FRAME_BUDGET_MS:
        print(
            f"budget      : 60fps ({FRAME_BUDGET_MS:.1f} ms/frame) -> "
            f"fast detection alone is {fast_ms / FRAME_BUDGET_MS:.1f}x OVER"
        )
    else:
        print(
            f"budget      : 60fps ({FRAME_BUDGET_MS:.1f} ms/frame) -> "
            "fast detection is within a single-frame budget, not counting "
            "network/encode"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
