"""Full recognition use case: detect, rectify, read, normalize and validate.

This is the first caller of the perspective, post-processing and RAG stages,
which shipped as libraries in Phases 2, 4 and 5. It composes them; it adds no
logic of its own beyond deciding what a failed stage means for the response.
"""

import base64
import logging
import time
from collections.abc import Sequence

import cv2
import numpy as np

from app.core.config import get_settings
from app.schemas.detection import BoundingBox
from app.schemas.recognize import PlateResult, RecognizeResponse
from app.services.detection_service import get_detector
from app.services.ocr_service import get_recognizer
from app.utils.image import load_image
from detector.detector import Detection
from detector.pipelines.perspective import correct_perspective
from ocr.reading import TextLine, split_reading
from postprocess.plate import normalize_plate_text
from rag.validator import resolve_province

logger = logging.getLogger(__name__)


def _elapsed_ms(since: float) -> float:
    """Return milliseconds elapsed since a ``perf_counter`` reading.

    Args:
        since: The earlier :func:`time.perf_counter` value.

    Returns:
        The elapsed time in milliseconds.
    """
    return (time.perf_counter() - since) * 1000


def _encode_crop_png(crop: np.ndarray) -> str | None:
    """Encode a rectified crop as a ``data:image/png`` base64 URI.

    Args:
        crop: The perspective-corrected plate crop (BGR).

    Returns:
        The crop as a data URI, or ``None`` if PNG encoding failed.
    """
    ok, buffer = cv2.imencode(".png", crop)
    if not ok:
        return None
    encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _build_result(
    box: Detection, lines: Sequence[TextLine], crop: np.ndarray, include_crop: bool
) -> PlateResult:
    """Turn one plate's recognized fragments into a finished result.

    Args:
        box: The detection the fragments were read from.
        lines: Fragments recognized in the rectified crop, in any order.
        crop: The perspective-corrected crop the fragments were read from.
        include_crop: Whether to attach the crop as a base64 PNG.

    Returns:
        The plate's number and province, each reported as unresolved rather
        than guessed when its stage could not vouch for it.
    """
    reading = split_reading(lines)
    plate = normalize_plate_text(reading.plate_text)
    province = resolve_province(reading.province_candidates)

    return PlateResult(
        box=BoundingBox(
            x1=box.x1, y1=box.y1, x2=box.x2, y2=box.y2, confidence=box.confidence
        ),
        plate_text=plate.text,
        plate_confidence=reading.plate_confidence,
        is_well_formed=plate.is_well_formed,
        province=province.province if province else None,
        province_confidence=province.score if province else None,
        province_candidates=list(reading.province_candidates),
        crop_png=_encode_crop_png(crop) if include_crop else None,
    )


def recognize_plates(
    data: bytes, content_type: str, include_crops: bool = False
) -> RecognizeResponse:
    """Run the full pipeline over an uploaded scene.

    Every detected plate is recognized, so a scene holding several plates
    reports several results. Per-stage latency is logged, not returned: the
    benchmark under ``docs/benchmark/`` is the authority on those numbers.

    Args:
        data: Raw bytes of the uploaded file.
        content_type: Content type declared by the client.
        include_crops: When true, attach each plate's rectified crop as a
            base64 PNG so a client can show what was read. Off by default so
            the live camera path keeps its per-frame payloads small.

    Returns:
        One :class:`~app.schemas.recognize.PlateResult` per plate, empty when
        the detector found none.

    Raises:
        ImageValidationError: If the upload fails validation or decoding.
        FileNotFoundError: If the configured detector weights are missing.
    """
    settings = get_settings()
    started = time.perf_counter()

    image = load_image(
        data,
        content_type,
        max_bytes=settings.max_upload_bytes,
        allowed_types=settings.allowed_image_types,
    )

    detect_started = time.perf_counter()
    detections = get_detector().detect(image)
    detect_ms = _elapsed_ms(detect_started)

    plates: list[PlateResult] = []
    perspective_ms = 0.0
    ocr_ms = 0.0
    postprocess_ms = 0.0

    for detection in detections:
        stage_started = time.perf_counter()
        try:
            crop = correct_perspective(
                image,
                detection,
                output_size=(settings.plate_crop_width, settings.plate_crop_height),
                padding=settings.plate_crop_padding,
            )
        except ValueError:
            # One unusable box must not cost the caller the plates around it.
            logger.warning(
                "skipped detection with no area in the image",
                extra={"box": str(detection)},
            )
            continue
        perspective_ms += _elapsed_ms(stage_started)

        stage_started = time.perf_counter()
        lines = get_recognizer().recognize(crop)
        ocr_ms += _elapsed_ms(stage_started)

        stage_started = time.perf_counter()
        plates.append(_build_result(detection, lines, crop, include_crops))
        postprocess_ms += _elapsed_ms(stage_started)

    logger.info(
        "recognized plates",
        extra={
            "plates": len(plates),
            "detect_ms": round(detect_ms, 1),
            "perspective_ms": round(perspective_ms, 1),
            "ocr_ms": round(ocr_ms, 1),
            "postprocess_ms": round(postprocess_ms, 1),
            "total_ms": round(_elapsed_ms(started), 1),
        },
    )
    return RecognizeResponse(count=len(plates), plates=plates)
