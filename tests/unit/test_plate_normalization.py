"""Tests for normalizing raw OCR plate text into a canonical form."""

from postprocess.plate import NormalizedPlate, normalize_plate_text


def test_normalize_inserts_canonical_spacing() -> None:
    """Letters and digits are separated by exactly one space."""
    assert normalize_plate_text("กข1234") == NormalizedPlate(
        text="กข 1234", letters="กข", digits="1234", is_well_formed=True
    )


def test_normalize_drops_separators_and_surrounding_whitespace() -> None:
    """OCR punctuation and padding are noise, not plate content."""
    assert normalize_plate_text("  กข - 1234 ").text == "กข 1234"


def test_normalize_keeps_the_modern_leading_digit_prefix() -> None:
    """Plates issued since 2012 carry a digit ahead of the consonants."""
    assert normalize_plate_text("1กข2345") == NormalizedPlate(
        text="1กข 2345", letters="1กข", digits="2345", is_well_formed=True
    )


def test_normalize_strips_stray_combining_marks_from_letters() -> None:
    """A plate consonant carries no vowel mark; OCR sometimes hallucinates one."""
    assert normalize_plate_text("กุข1234").text == "กข 1234"


def test_normalize_flags_text_that_is_not_a_plate() -> None:
    """A misread that is not Thai at all must not be presented as a plate."""
    result = normalize_plate_text("VEZT")

    assert result.is_well_formed is False
    assert result.letters == ""
    assert result.digits == ""


def test_normalize_flags_too_many_digits() -> None:
    """Thai plates carry at most four digits, so five is a misread."""
    assert normalize_plate_text("กข12345").is_well_formed is False


def test_normalize_handles_empty_input() -> None:
    """An unread plate normalizes to empty rather than raising."""
    assert normalize_plate_text("") == NormalizedPlate(
        text="", letters="", digits="", is_well_formed=False
    )
