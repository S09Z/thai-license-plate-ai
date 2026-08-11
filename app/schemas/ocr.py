"""Schemas for the OCR endpoint."""

from pydantic import BaseModel


class TextLineOut(BaseModel):
    """A single recognized line of text.

    Attributes:
        text: The recognized characters.
        confidence: Recognition confidence, in ``[0, 1]``.
    """

    text: str
    confidence: float


class OcrResponse(BaseModel):
    """Response body for ``POST /ocr``.

    Attributes:
        plate_text: Raw text of the plate-number line, empty when nothing was
            recognized. Never interpreted or executed, only reported.
        plate_confidence: Confidence of the plate-number line.
        province_candidates: Raw text of each line below the plate number,
            left unresolved for later province matching.
        lines: Every recognized line, ordered top to bottom.
    """

    plate_text: str
    plate_confidence: float
    province_candidates: list[str]
    lines: list[TextLineOut]
