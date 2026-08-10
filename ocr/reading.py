"""Plate reading structures and the pure logic that shapes them.

Deliberately free of any OCR engine dependency, so this layer stays trivially
testable and reusable by post-processing (Phase 4) and RAG validation (Phase 5).
"""

from collections.abc import Sequence
from dataclasses import dataclass

# A fragment joins the row above it when their vertical spans overlap by at
# least this fraction of the shorter fragment. Thai plates separate the number
# and the province into clear bands, so a half-height overlap keeps them apart
# while still absorbing baseline jitter between fragments of one row.
_ROW_OVERLAP_RATIO = 0.5


@dataclass(frozen=True)
class TextLine:
    """A single text fragment recognized in a plate crop.

    An OCR engine reports one box per fragment, not per visual line: ``กข 1234``
    typically arrives as two fragments. Geometry is carried here so
    :func:`group_into_rows` can reassemble the lines.

    Attributes:
        text: The recognized characters.
        confidence: Recognition confidence, in ``[0, 1]``.
        top: Y coordinate of the fragment's top edge within the crop.
        bottom: Y coordinate of the fragment's bottom edge within the crop.
        left: X coordinate of the fragment's left edge within the crop.
    """

    text: str
    confidence: float
    top: float
    bottom: float
    left: float


@dataclass(frozen=True)
class PlateReading:
    """The interpretation of a plate crop's recognized text.

    Attributes:
        plate_text: Raw text of the plate-number row, empty when nothing
            was recognized.
        plate_confidence: Confidence of the plate-number row, ``0.0`` when
            nothing was recognized.
        province_candidates: Raw text of every row below the plate number.
            Left unresolved here; Phase 5 matches them against the province
            knowledge base.
    """

    plate_text: str
    plate_confidence: float
    province_candidates: tuple[str, ...]


def _shares_row(fragment: TextLine, anchor: TextLine) -> bool:
    """Report whether a fragment belongs to a row anchored by another.

    The comparison is against the row's *anchor* — its topmost fragment —
    rather than the row's accumulated top-to-bottom span. A row's span grows
    as fragments join it, and on a skewed crop one tall fragment can drag that
    span down far enough to overlap the band below, merging the province into
    the number. Anchoring to a fixed fragment keeps a row from growing into
    the row beneath it.

    Args:
        fragment: The fragment being placed.
        anchor: The row's topmost fragment, fixed when the row was opened.

    Returns:
        ``True`` when the vertical spans overlap enough to be one visual row.
    """
    overlap = min(fragment.bottom, anchor.bottom) - max(fragment.top, anchor.top)
    shortest = min(fragment.bottom - fragment.top, anchor.bottom - anchor.top)
    if shortest <= 0:
        return False
    return overlap / shortest >= _ROW_OVERLAP_RATIO


def group_into_rows(fragments: Sequence[TextLine]) -> list[list[TextLine]]:
    """Group recognized fragments into visual rows.

    Fragments whose boxes overlap the row's anchor vertically form one row,
    ordered left to right within it, which is reading order for a Thai plate.

    Args:
        fragments: Recognized fragments, in any order.

    Returns:
        Rows ordered top to bottom, each ordered left to right.
    """
    rows: list[list[TextLine]] = []
    anchors: list[TextLine] = []

    for fragment in sorted(fragments, key=lambda item: item.top):
        if rows and _shares_row(fragment, anchors[-1]):
            rows[-1].append(fragment)
        else:
            rows.append([fragment])
            anchors.append(fragment)

    return [sorted(row, key=lambda item: item.left) for row in rows]


def split_reading(lines: Sequence[TextLine]) -> PlateReading:
    """Split recognized fragments into a plate number and province candidates.

    Thai plates carry the registration number above the province name, so the
    topmost row is taken as the number and every row below it is kept as a
    province candidate.

    Args:
        lines: Recognized text fragments, in any order.

    Returns:
        The resulting :class:`PlateReading`; empty when ``lines`` is empty.
    """
    rows = group_into_rows(lines)
    if not rows:
        return PlateReading(plate_text="", plate_confidence=0.0, province_candidates=())

    plate, *below = rows
    return PlateReading(
        plate_text=" ".join(fragment.text for fragment in plate),
        # A joined row is only as trustworthy as its least certain fragment.
        plate_confidence=min(fragment.confidence for fragment in plate),
        province_candidates=tuple(
            " ".join(fragment.text for fragment in row) for row in below
        ),
    )
