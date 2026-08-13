"""Tests for the character-level similarity metric used to retrieve provinces."""

from rag.similarity import levenshtein_distance, similarity_ratio


def test_distance_between_identical_strings_is_zero() -> None:
    """Identical text needs no edits."""
    assert levenshtein_distance("ชลบุรี", "ชลบุรี") == 0


def test_distance_counts_a_single_inserted_character() -> None:
    """The recognizer's trailing hallucination costs exactly one edit."""
    assert levenshtein_distance("ชลบร", "ชลบรด") == 1


def test_distance_counts_a_single_substitution() -> None:
    """A swapped consonant costs one edit, not two."""
    assert levenshtein_distance("ระนอง", "ระยอง") == 1


def test_distance_from_empty_string_is_the_other_length() -> None:
    """Building text from nothing costs one insert per character."""
    assert levenshtein_distance("", "ตาก") == 3
    assert levenshtein_distance("ตาก", "") == 3


def test_distance_is_symmetric() -> None:
    """Edit distance does not depend on argument order."""
    assert levenshtein_distance("เพชรบร", "เพชรบรณ") == levenshtein_distance(
        "เพชรบรณ", "เพชรบร"
    )


def test_ratio_of_identical_strings_is_one() -> None:
    """A perfect match scores 1.0."""
    assert similarity_ratio("ภูเก็ต", "ภูเก็ต") == 1.0


def test_ratio_of_two_empty_strings_is_one() -> None:
    """Two empty strings are trivially equal, and must not divide by zero."""
    assert similarity_ratio("", "") == 1.0


def test_ratio_of_wholly_different_strings_is_zero() -> None:
    """No shared characters scores 0.0 rather than a small positive value."""
    assert similarity_ratio("ตาก", "ชลบ") == 0.0


def test_ratio_is_normalized_by_the_longer_string() -> None:
    """One edit against a five-character name costs one fifth of the score."""
    assert similarity_ratio("ชลบร", "ชลบรด") == 0.8
