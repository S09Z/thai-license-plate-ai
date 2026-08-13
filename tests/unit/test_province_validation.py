"""Tests for RAG-style correction of a degraded province candidate."""

from postprocess.provinces import THAI_PROVINCES
from postprocess.thai import strip_thai_marks
from rag.validator import correct_province, resolve_province


def test_corrects_the_real_recognizer_misread() -> None:
    """The Phase 3 gap: the engine reads ชลบุรี as ชลบรดี.

    Phase 4's deterministic matcher returns ``None`` here by design. Closing
    this case is the reason Phase 5 exists.
    """
    match = correct_province("ชลบรดี")

    assert match is not None
    assert match.province == "ชลบุรี"
    assert not match.is_exact


def test_corrects_a_candidate_with_bleeding_plate_digits() -> None:
    """The other observed Phase 3 output for the same plate."""
    match = correct_province("ชลบร 9")

    assert match is not None
    assert match.province == "ชลบุรี"


def test_exact_candidate_is_reported_as_exact_at_full_score() -> None:
    """A clean read must not be routed through fuzzy scoring."""
    match = correct_province("ชลบุรี")

    assert match is not None
    assert match.province == "ชลบุรี"
    assert match.is_exact
    assert match.score == 1.0


def test_mark_stripped_candidate_is_still_exact() -> None:
    """Dropped vowel marks alone are handled deterministically, not fuzzily."""
    match = correct_province("ชลบร")

    assert match is not None
    assert match.province == "ชลบุรี"
    assert match.is_exact


def test_abstains_when_two_provinces_are_equally_close() -> None:
    """เพชรบุรี and เพชรบูรณ์ differ only in marks the recognizer drops.

    Returning either one would be a confident guess. Abstaining is correct.
    """
    assert correct_province("เพชรบรดี") is None


def test_plate_text_does_not_resolve_to_a_province() -> None:
    """A plate number must never be mistaken for a province name."""
    assert correct_province("กข 1234") is None


def test_latin_misread_does_not_resolve_to_a_province() -> None:
    """'VEZL' is a real Phase 3 misread; it names no province."""
    assert correct_province("VEZL") is None


def test_empty_candidate_returns_none() -> None:
    """No candidate means no province."""
    assert correct_province("") is None
    assert correct_province("   ") is None


def test_every_province_survives_mark_stripping() -> None:
    """The degradation the recognizer always applies must always be recoverable."""
    for province in THAI_PROVINCES:
        match = correct_province(strip_thai_marks(province))

        assert match is not None, province
        assert match.province == province


def test_degraded_province_never_resolves_to_a_different_province() -> None:
    """The core safety invariant: abstain, never mis-attribute.

    Covers the three degradations the recognizer was actually observed to
    apply. Each may return ``None`` — but never a *different* province, because
    a confidently wrong province is worse than an admitted failure.
    """
    degradations = (
        lambda name: strip_thai_marks(name),
        lambda name: strip_thai_marks(name) + "ดี",
        lambda name: strip_thai_marks(name) + " 9",
    )

    for province in THAI_PROVINCES:
        for degrade in degradations:
            match = correct_province(degrade(province))

            assert match is None or match.province == province, (
                f"{province!r} degraded to {degrade(province)!r} "
                f"resolved to {match.province!r}"  # type: ignore[union-attr]
            )


def test_truncating_phetchabun_aliases_onto_phetchaburi() -> None:
    """One province truncates exactly onto another, and it is unrecoverable.

    Losing the last character of เพชรบูรณ์ leaves เพชรบร, which *is* the
    mark-free skeleton of เพชรบุรี. No matcher can tell them apart — the
    distinguishing character is gone — so this is recorded rather than fixed.
    It is the only such pair in the 77, asserted below.
    """
    match = correct_province(strip_thai_marks("เพชรบูรณ์")[:-1])

    assert match is not None
    assert match.province == "เพชรบุรี"
    assert match.is_exact

    skeletons = {strip_thai_marks(province): province for province in THAI_PROVINCES}
    aliases = [
        province
        for province in THAI_PROVINCES
        if skeletons.get(strip_thai_marks(province)[:-1], province) != province
    ]

    assert aliases == ["เพชรบูรณ์"]


def test_resolve_picks_the_best_candidate_from_a_reading() -> None:
    """`PlateReading` carries several province candidates; one must win."""
    match = resolve_province(["1234", "ชลบรดี", "กข"])

    assert match is not None
    assert match.province == "ชลบุรี"


def test_resolve_prefers_an_exact_candidate_over_a_fuzzy_one() -> None:
    """A clean candidate outranks a degraded one regardless of order."""
    match = resolve_province(["เชียงราย", "ชลบรดี"])

    assert match is not None
    assert match.province == "เชียงราย"
    assert match.is_exact


def test_resolve_returns_none_when_no_candidate_is_a_province() -> None:
    """Rows that hold no province at all resolve to nothing."""
    assert resolve_province(["กข 1234", "VEZL"]) is None


def test_resolve_returns_none_for_no_candidates() -> None:
    """An empty reading resolves to nothing."""
    assert resolve_province([]) is None
