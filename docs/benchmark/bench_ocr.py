"""Benchmark the shipped OCR stage on a synthetic Thai plate.

Measures what the application actually runs — :class:`ocr.recognizer.PlateOCR`
followed by :func:`ocr.reading.split_reading` — and reports recognition
accuracy and latency against the ``<40ms`` OCR budget in ``CLAUDE.md``.

The input is *rendered text, not a photograph*. Results bound the engine's
speed, not real-world accuracy; see ``docs/benchmark/ocr-phase3.md``.

Run from the repository root::

    poetry run python docs/benchmark/bench_ocr.py
"""

import argparse
import platform
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ocr.reading import split_reading  # noqa: E402
from ocr.recognizer import PlateOCR  # noqa: E402

# Thonburi genuinely contains Thai glyphs. Not every macOS font that looks
# Thai does: Ayuthaya renders shapes the recognizer reads as Latin, and an
# earlier run silently benchmarked Bhaiksuki tofu boxes.
FONT_PATH = "/System/Library/Fonts/Supplemental/Thonburi.ttc"

EXPECTED_PLATE = "กข 1234"
EXPECTED_PROVINCE = "ชลบุรี"

# Matches the perspective stage's output_size default (itself a placeholder).
CROP_SIZE = (256, 128)
OCR_BUDGET_MS = 40.0


def render_plate() -> np.ndarray:
    """Render a synthetic two-line Thai plate crop.

    Returns:
        A BGR image of shape ``(128, 256, 3)``: plate number above, province
        below, mirroring the layout of a real Thai plate.

    Raises:
        FileNotFoundError: If the Thai font is missing.
        AssertionError: If the render produced no ink, which would make every
            downstream number meaningless.
    """
    if not Path(FONT_PATH).exists():
        raise FileNotFoundError(f"Thai font not found: {FONT_PATH}")

    width, height = CROP_SIZE
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for text, size, top in ((EXPECTED_PLATE, 52, 8), (EXPECTED_PROVINCE, 28, 78)):
        font = ImageFont.truetype(FONT_PATH, size)
        text_width = draw.textbbox((0, 0), text, font=font)[2]
        draw.text(((width - text_width) // 2, top), text, font=font, fill=(0, 0, 0))

    image = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)
    assert image.min() < 128, "render produced no ink; font cannot draw the text"
    return image


def main() -> None:
    """Benchmark the OCR stage and print an accuracy and latency report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repeats", type=int, default=15, help="warm iterations to time"
    )
    args = parser.parse_args()

    crop = render_plate()
    recognizer = PlateOCR(lang="th", min_confidence=0.5)

    started = time.perf_counter()
    lines = recognizer.recognize(crop)
    cold_ms = (time.perf_counter() - started) * 1000

    timings = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        recognizer.recognize(crop)
        timings.append((time.perf_counter() - started) * 1000)

    reading = split_reading(lines)
    median = statistics.median(timings)

    print(f"machine     : {platform.platform()} / {platform.processor()}")
    print(f"crop        : {crop.shape[1]}x{crop.shape[0]}")
    print(f"fragments   : {[(line.text, round(line.confidence, 3)) for line in lines]}")
    print(
        f"plate_text  : {reading.plate_text!r} @ {reading.plate_confidence:.3f}"
        f"   {'MATCH' if reading.plate_text == EXPECTED_PLATE else 'MISMATCH'}"
        f" (want {EXPECTED_PLATE!r})"
    )
    province_ok = EXPECTED_PROVINCE in reading.province_candidates
    print(
        f"provinces   : {list(reading.province_candidates)}"
        f"   {'MATCH' if province_ok else 'MISMATCH'}"
        f" (want {EXPECTED_PROVINCE!r})"
    )
    print(f"cold start  : {cold_ms:.0f} ms  (loads weights)")
    print(
        f"warm median : {median:.0f} ms   min {min(timings):.0f}  "
        f"max {max(timings):.0f}  n={args.repeats}"
    )
    print(
        f"budget      : <{OCR_BUDGET_MS:.0f} ms"
        f"  -> {median / OCR_BUDGET_MS:.0f}x OVER"
    )


if __name__ == "__main__":
    main()
