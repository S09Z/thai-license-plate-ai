"""OCR use case: validate a plate crop, recognize its text, shape the result."""

from functools import lru_cache

from app.core.config import get_settings
from app.schemas.ocr import OcrResponse, TextLineOut
from app.utils.image import load_image
from ocr.reading import split_reading
from ocr.recognizer import PlateOCR


@lru_cache
def get_recognizer() -> PlateOCR:
    """Return the process-wide OCR recognizer, built from settings on first use.

    Returns:
        A memoized :class:`PlateOCR`; its engine loads on first recognition.
    """
    settings = get_settings()
    return PlateOCR(
        lang=settings.ocr_lang,
        min_confidence=settings.ocr_min_confidence,
        det_limit_side_len=settings.ocr_det_limit_side_len,
        det_limit_type=settings.ocr_det_limit_type,
    )


def read_plate(data: bytes, content_type: str) -> OcrResponse:
    """Recognize plate text in an uploaded crop.

    Args:
        data: Raw bytes of the uploaded file.
        content_type: Content type declared by the client.

    Returns:
        The plate reading, with an empty ``plate_text`` when nothing was read.

    Raises:
        ImageValidationError: If the upload fails validation or decoding.
    """
    settings = get_settings()
    crop = load_image(
        data,
        content_type,
        max_bytes=settings.max_upload_bytes,
        allowed_types=settings.allowed_image_types,
    )

    lines = get_recognizer().recognize(crop)
    reading = split_reading(lines)
    return OcrResponse(
        plate_text=reading.plate_text,
        plate_confidence=reading.plate_confidence,
        province_candidates=list(reading.province_candidates),
        lines=[
            TextLineOut(text=line.text, confidence=line.confidence) for line in lines
        ],
    )
