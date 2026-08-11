"""Benchmark the assembled recognition pipeline on a synthetic scene.

Measures what ``POST /recognize`` actually runs per plate — perspective
correction, OCR, post-processing and RAG validation — against the ``<100ms``
total budget in ``CLAUDE.md``.

**Detection is not measured.** No plate weights have ever been trained, so
``models/detector/best.pt`` does not exist. This script substitutes a fixed
known box for the detector, which makes every total printed here a *lower
bound* on the real end-to-end figure, not the figure itself.

The input is *rendered text, not a photograph*. Results bound the pipeline's
speed, not real-world accuracy; see ``docs/benchmark/recognize-phase6.md``.

Run from the repository root::

    poetry run python docs/benchmark/bench_recognize.py
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

from app.core.config import Settings  # noqa: E402
from detector.detector import Detection  # noqa: E402
from detector.pipelines.perspective import correct_perspective  # noqa: E402
from ocr.reading import PlateReading, split_reading  # noqa: E402
from ocr.recognizer import PlateOCR  # noqa: E402
from postprocess.plate import NormalizedPlate, normalize_plate_text  # noqa: E402
from rag.validator import ProvinceMatch, resolve_province  # noqa: E402

# Thonburi genuinely contains Thai glyphs; several macOS fonts that look Thai
# do not. See the note in bench_ocr.py.
FONT_PATH = "/System/Library/Fonts/Supplemental/Thonburi.ttc"

EXPECTED_PLATE = "กข 1234"
EXPECTED_PROVINCE = "ชลบุรี"

SCENE_SIZE = (640, 480)
# Where the plate is drawn in the scene, and therefore the box the substitute
# detector reports.
PLATE_BOX = Detection(x1=180, y1=160, x2=460, y2=300, confidence=0.91)

TOTAL_BUDGET_MS = 100.0


def render_scene() -> np.ndarray:
    """Render a synthetic scene with one two-line Thai plate in it.

    Returns:
        A BGR image of shape ``(480, 640, 3)``: a grey background with a
        bordered white plate at :data:`PLATE_BOX`, number above province.

    Raises:
        FileNotFoundError: If the Thai font is missing.
        AssertionError: If the render produced no ink, which would make every
            downstream number meaningless.
    """
    if not Path(FONT_PATH).exists():
        raise FileNotFoundError(f"Thai font not found: {FONT_PATH}")

    canvas = Image.new("RGB", SCENE_SIZE, (110, 115, 120))
    draw = ImageDraw.Draw(canvas)

    box = (PLATE_BOX.x1, PLATE_BOX.y1, PLATE_BOX.x2, PLATE_BOX.y2)
    # A dark border gives the perspective stage a quadrilateral to find, as a
    # real plate's edge does.
    draw.rectangle(box, fill=(255, 255, 255), outline=(20, 20, 20), width=3)

    plate_width = PLATE_BOX.x2 - PLATE_BOX.x1
    for text, size, offset in ((EXPECTED_PLATE, 58, 12), (EXPECTED_PROVINCE, 30, 88)):
        font = ImageFont.truetype(FONT_PATH, size)
        text_width = draw.textbbox((0, 0), text, font=font)[2]
        draw.text(
            (PLATE_BOX.x1 + (plate_width - text_width) // 2, PLATE_BOX.y1 + offset),
            text,
            font=font,
            fill=(0, 0, 0),
        )

    image = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)
    assert image.min() < 128, "render produced no ink; font cannot draw the text"
    return image


def main() -> None:
    """Benchmark the assembled pipeline and print an accuracy and latency report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repeats", type=int, default=15, help="warm iterations to time"
    )
    args = parser.parse_args()

    settings = Settings()
    scene = render_scene()
    recognizer = PlateOCR(
        lang=settings.ocr_lang,
        min_confidence=settings.ocr_min_confidence,
        det_limit_side_len=settings.ocr_det_limit_side_len,
        det_limit_type=settings.ocr_det_limit_type,
    )
    crop_size = (settings.plate_crop_width, settings.plate_crop_height)

    def run_once() -> (
        tuple[dict[str, float], PlateReading, NormalizedPlate, ProvinceMatch | None]
    ):
        """Run one plate through the pipeline, returning per-stage timings."""
        stage = time.perf_counter()
        crop = correct_perspective(
            scene,
            PLATE_BOX,
            output_size=crop_size,
            padding=settings.plate_crop_padding,
        )
        perspective_ms = (time.perf_counter() - stage) * 1000

        stage = time.perf_counter()
        lines = recognizer.recognize(crop)
        ocr_ms = (time.perf_counter() - stage) * 1000

        stage = time.perf_counter()
        reading = split_reading(lines)
        plate = normalize_plate_text(reading.plate_text)
        province = resolve_province(reading.province_candidates)
        post_ms = (time.perf_counter() - stage) * 1000

        return (
            {
                "perspective": perspective_ms,
                "ocr": ocr_ms,
                "postprocess+rag": post_ms,
                "total": perspective_ms + ocr_ms + post_ms,
            },
            reading,
            plate,
            province,
        )

    started = time.perf_counter()
    run_once()
    cold_ms = (time.perf_counter() - started) * 1000

    runs = [run_once() for _ in range(args.repeats)]
    timings = [run[0] for run in runs]
    _, reading, plate, province = runs[-1]

    def median_of(stage: str) -> float:
        return statistics.median(timing[stage] for timing in timings)

    total = median_of("total")

    print(f"machine     : {platform.platform()} / {platform.processor()}")
    print(f"scene       : {scene.shape[1]}x{scene.shape[0]}  crop {crop_size}")
    print(f"raw reading : {reading.plate_text!r} {list(reading.province_candidates)}")
    print(
        f"plate_text  : {plate.text!r}  well_formed={plate.is_well_formed}"
        f"   {'MATCH' if plate.text == EXPECTED_PLATE else 'MISMATCH'}"
        f" (want {EXPECTED_PLATE!r})"
    )
    if province is None:
        print(f"province    : None (abstained)   MISMATCH (want {EXPECTED_PROVINCE!r})")
    else:
        matched = province.province == EXPECTED_PROVINCE
        print(
            f"province    : {province.province!r} @ {province.score:.3f}"
            f"   {'MATCH' if matched else 'MISMATCH'} (want {EXPECTED_PROVINCE!r})"
        )
    print(f"cold start  : {cold_ms:.0f} ms  (loads OCR weights)")
    print("detection   : unmeasured  (no trained weights exist)")
    for stage in ("perspective", "ocr", "postprocess+rag"):
        print(f"  {stage:<15}: {median_of(stage):7.1f} ms")
    print(f"  {'TOTAL (lower bound)':<15}: {total:7.1f} ms   n={args.repeats}")
    print(
        f"budget      : <{TOTAL_BUDGET_MS:.0f} ms"
        f"  -> {total / TOTAL_BUDGET_MS:.1f}x OVER"
        if total > TOTAL_BUDGET_MS
        else f"budget      : <{TOTAL_BUDGET_MS:.0f} ms  -> within budget"
    )


if __name__ == "__main__":
    main()
