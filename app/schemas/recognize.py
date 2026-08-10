"""Schemas for the full recognition endpoint."""

from pydantic import BaseModel

from app.schemas.detection import BoundingBox


class PlateResult(BaseModel):
    """One plate carried through the whole pipeline.

    Attributes:
        box: Where the plate was detected in the source image.
        plate_text: The plate number in canonical ``"กข 1234"`` form when it
            matched the Thai plate pattern, otherwise the reading exactly as
            recognized. Never interpreted or executed, only reported.
        plate_confidence: Confidence of the plate-number row.
        is_well_formed: Whether the number matched the Thai plate pattern. A
            misread is reported as read rather than coerced into a plausible
            plate, because a confidently wrong number is worse than an
            admitted failure.
        province: The canonical province name, or ``None`` when the knowledge
            base could not vouch for one. ``None`` means unknown, not failure.
        province_confidence: Similarity behind ``province``, ``None`` when no
            province was resolved.
        province_candidates: Raw text of every row below the plate number,
            kept verbatim so a client can show what was actually read.
        crop_png: The perspective-corrected plate crop as a ``data:image/png``
            base64 URI, so a client can show what the OCR actually read.
            ``None`` unless the request asked for crops; the live camera path
            leaves it off to keep per-frame payloads small.
    """

    box: BoundingBox
    plate_text: str
    plate_confidence: float
    is_well_formed: bool
    province: str | None
    province_confidence: float | None
    province_candidates: list[str]
    crop_png: str | None = None


class RecognizeResponse(BaseModel):
    """Response body for ``POST /recognize``.

    Attributes:
        count: Number of plates reported.
        plates: One result per recognized plate, in detection order.
    """

    count: int
    plates: list[PlateResult]
