"""Unit tests for the pure accuracy-scoring functions."""

from eval.scoring import (
    Aggregate,
    aggregate,
    char_error_rate,
    edit_distance,
    normalize_plate,
    plate_exact_match,
    score_one,
)


def test_normalize_plate_removes_all_whitespace() -> None:
    assert normalize_plate("กข 1234") == "กข1234"
    assert normalize_plate("  กข\t12 34 ") == "กข1234"


def test_edit_distance_basic_cases() -> None:
    assert edit_distance("abc", "abc") == 0
    assert edit_distance("", "abc") == 3
    assert edit_distance("abc", "") == 3
    assert edit_distance("abc", "abd") == 1  # substitution
    assert edit_distance("abc", "aXbc") == 1  # insertion


def test_plate_exact_match_ignores_spacing() -> None:
    assert plate_exact_match("กข 1234", "กข1234")
    assert not plate_exact_match("กข 1234", "กค 1234")


def test_char_error_rate_normalizes_before_comparing() -> None:
    assert char_error_rate("กข 1234", "กข1234") == 0.0
    # one wrong char out of six normalized chars
    assert char_error_rate("กค1234", "กข1234") == 1 / 6


def test_char_error_rate_edge_cases() -> None:
    assert char_error_rate("", "") == 0.0
    assert char_error_rate("abc", "") == 1.0  # produced text against empty truth


def test_score_one_grades_every_field() -> None:
    outcome = score_one(
        img="a.jpg",
        plate_pred="1กฒ 1753",
        plate_truth="1กฒ1753",
        province_pred="กรุงเทพมหานคร",
        province_truth="กรุงเทพมหานคร",
    )
    assert outcome.plate_correct
    assert outcome.plate_cer == 0.0
    assert outcome.province_correct


def test_score_one_flags_unresolved_province() -> None:
    outcome = score_one(
        img="a.jpg",
        plate_pred="กข 1",
        plate_truth="กข 1",
        province_pred=None,
        province_truth="ภูเก็ต",
    )
    assert outcome.plate_correct
    assert not outcome.province_correct


def test_aggregate_reduces_to_corpus_metrics() -> None:
    outcomes = [
        score_one("a.jpg", "กข 1", "กข 1", "ภูเก็ต", "ภูเก็ต"),  # perfect
        score_one("b.jpg", "กค 1", "กข 1", None, "ภูเก็ต"),  # 1/3 CER, both wrong
    ]
    agg = aggregate(outcomes)
    assert agg.count == 2
    assert agg.plate_exact_rate == 0.5
    assert agg.province_accuracy == 0.5
    assert abs(agg.mean_cer - (0.0 + 1 / 3) / 2) < 1e-9


def test_aggregate_empty_is_all_zero() -> None:
    assert aggregate([]) == Aggregate(0, 0.0, 0.0, 0.0)
