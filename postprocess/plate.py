"""Normalization of a raw plate-number reading into a canonical form."""

import re
from dataclasses import dataclass

from postprocess.thai import strip_thai_marks

# A Thai registration number is an optional leading digit (used since 2012),
# one to three consonants, then one to four digits. Anything else is a misread.
_PLATE_PATTERN = re.compile(r"^(\d?[ก-ฮ]{1,3})(\d{1,4})$")

# Everything a plate number cannot contain: separators, Latin letters, and the
# punctuation an engine emits around glyphs it is unsure of.
_NOISE_PATTERN = re.compile(r"[^ก-ฮ0-9]")

_WHITESPACE_PATTERN = re.compile(r"\s+")


@dataclass(frozen=True)
class NormalizedPlate:
    """A plate number after cleanup, with the parts a caller needs.

    Attributes:
        text: Canonical ``"กข 1234"`` form when well formed; otherwise the
            input with whitespace collapsed, so callers can still show what
            was read.
        letters: The consonant group including any leading digit, empty when
            the reading is not well formed.
        digits: The trailing digit group, empty when not well formed.
        is_well_formed: Whether the reading matches the Thai plate pattern.
    """

    text: str
    letters: str
    digits: str
    is_well_formed: bool


def normalize_plate_text(raw: str) -> NormalizedPlate:
    """Normalize raw recognized text into a canonical plate number.

    Separators, stray marks and non-plate characters are removed, then the
    result is checked against the Thai plate pattern. Text that does not match
    is reported as-is with ``is_well_formed`` false rather than being coerced —
    a wrong plate number is worse than an admitted failure.

    Args:
        raw: Text of the plate-number row, exactly as recognized.

    Returns:
        The :class:`NormalizedPlate` for ``raw``.
    """
    collapsed = _WHITESPACE_PATTERN.sub(" ", raw).strip()
    cleaned = _NOISE_PATTERN.sub("", strip_thai_marks(collapsed))
    match = _PLATE_PATTERN.match(cleaned)

    if match is None:
        return NormalizedPlate(
            text=collapsed, letters="", digits="", is_well_formed=False
        )

    letters, digits = match.groups()
    return NormalizedPlate(
        text=f"{letters} {digits}",
        letters=letters,
        digits=digits,
        is_well_formed=True,
    )
