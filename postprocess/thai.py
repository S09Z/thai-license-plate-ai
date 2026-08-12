"""Primitives for comparing Thai text that an OCR engine has degraded."""

# Thai combining vowel and tone marks: sara am's mark, the above/below vowels,
# and the tone/diacritic block. Base consonants and spacing vowels (เ แ โ ใ ไ ำ)
# are deliberately absent — those survive recognition and carry real identity.
_COMBINING_MARKS = frozenset(
    "ั"  # mai han akat
    "ิีึืฺุู"  # above/below vowels, phinthu
    "็่้๊๋์ํ๎"  # tones, thanthakhat
)


def strip_thai_marks(text: str) -> str:
    """Remove Thai combining vowel and tone marks, leaving the skeleton.

    The recognizer reliably reads Thai consonants but drops the small marks
    above and below them, so comparisons must ignore those marks to be useful.

    Args:
        text: Any Thai text.

    Returns:
        ``text`` with every combining mark removed.
    """
    return "".join(character for character in text if character not in _COMBINING_MARKS)
