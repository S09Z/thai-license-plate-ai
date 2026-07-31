"""Schemas for the detection endpoint."""

from pydantic import BaseModel


class BoundingBox(BaseModel):
    """A detected plate region in pixel coordinates.

    Attributes:
        x1: Left edge of the box.
        y1: Top edge of the box.
        x2: Right edge of the box.
        y2: Bottom edge of the box.
        confidence: Model confidence for the detection, in ``[0, 1]``.
    """

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float


class DetectionResponse(BaseModel):
    """Response body for ``POST /detect``.

    Attributes:
        count: Number of plates detected.
        boxes: The detected plate regions.
    """

    count: int
    boxes: list[BoundingBox]
