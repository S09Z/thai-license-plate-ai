"""Tests for mapping an OCR province candidate onto a canonical province."""

from postprocess.provinces import THAI_PROVINCES, match_province
from postprocess.thai import strip_thai_marks


def test_province_list_is_complete() -> None:
    """Thailand has 77 provinces, counting Bangkok."""
    assert len(THAI_PROVINCES) == 77
    assert len(set(THAI_PROVINCES)) == 77


def test_match_province_accepts_an_exact_name() -> None:
    """A cleanly recognized province maps to itself."""
    assert match_province("ชลบุรี") == "ชลบุรี"


def test_match_province_ignores_surrounding_whitespace() -> None:
    """Row joining can leave padding around the candidate."""
    assert match_province("  ภูเก็ต ") == "ภูเก็ต"


def test_match_province_recovers_dropped_vowel_marks() -> None:
    """The recognizer drops Thai vowel and tone marks; consonants survive."""
    assert match_province("ชลบร") == "ชลบุรี"


def test_match_province_returns_none_for_an_unknown_candidate() -> None:
    """Deterministic matching must not guess; Phase 5 handles the hard cases."""
    assert match_province("ชลบรดี") is None


def test_match_province_returns_none_for_empty_input() -> None:
    """No candidate means no province."""
    assert match_province("") is None


def test_stripping_marks_keeps_every_province_distinct() -> None:
    """Mark-insensitive matching is only safe while no two provinces collide."""
    skeletons = [strip_thai_marks(province) for province in THAI_PROVINCES]

    assert len(set(skeletons)) == len(THAI_PROVINCES)
