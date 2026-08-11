"""Tests for grouping recognized text fragments into a plate reading."""

from ocr.reading import PlateReading, TextLine, group_into_rows, split_reading


def _line(text: str, confidence: float, top: float, left: float) -> TextLine:
    """Build a TextLine with a conventional 40px-tall box."""
    return TextLine(
        text=text, confidence=confidence, top=top, bottom=top + 40.0, left=left
    )


def test_group_into_rows_joins_fragments_sharing_a_line() -> None:
    """Boxes whose vertical spans overlap belong to the same visual row."""
    fragments = [
        _line("1234", 0.99, top=6.0, left=90.0),
        _line("กข", 0.93, top=8.0, left=20.0),
    ]

    rows = group_into_rows(fragments)

    assert len(rows) == 1
    assert [fragment.text for fragment in rows[0]] == ["กข", "1234"]


def test_group_into_rows_separates_vertically_distinct_lines() -> None:
    """Boxes on different visual lines stay in separate rows, top-to-bottom."""
    fragments = [
        _line("ชลบุรี", 0.80, top=72.0, left=60.0),
        _line("กข 1234", 0.95, top=5.0, left=20.0),
    ]

    rows = group_into_rows(fragments)

    assert [[fragment.text for fragment in row] for row in rows] == [
        ["กข 1234"],
        ["ชลบุรี"],
    ]


def test_group_into_rows_of_nothing_is_empty() -> None:
    """No fragments yields no rows."""
    assert group_into_rows([]) == []


def test_split_reading_joins_fragments_of_the_plate_number() -> None:
    """A plate number split across boxes is rejoined in reading order.

    This is the real PaddleOCR behaviour: it returned 'กข' and '1234' as two
    separate detections of one visual line.
    """
    fragments = [
        _line("1234", 0.99, top=6.0, left=90.0),
        _line("กข", 0.93, top=8.0, left=20.0),
        _line("ชลบุรี", 0.80, top=76.0, left=60.0),
    ]

    reading = split_reading(fragments)

    assert reading.plate_text == "กข 1234"
    assert reading.province_candidates == ("ชลบุรี",)


def test_split_reading_confidence_is_the_weakest_fragment_in_the_row() -> None:
    """A joined row is only as trustworthy as its least certain fragment."""
    fragments = [
        _line("1234", 0.99, top=6.0, left=90.0),
        _line("กข", 0.62, top=8.0, left=20.0),
    ]

    assert split_reading(fragments).plate_confidence == 0.62


def test_split_reading_takes_topmost_row_as_plate_number() -> None:
    """Thai plates carry the number above the province, so the top row wins."""
    fragments = [
        _line("ชลบุรี", 0.88, top=70.0, left=60.0),
        _line("กข 1234", 0.95, top=4.0, left=20.0),
    ]

    reading = split_reading(fragments)

    assert reading == PlateReading(
        plate_text="กข 1234",
        plate_confidence=0.95,
        province_candidates=("ชลบุรี",),
    )


def test_split_reading_keeps_every_row_below_the_first_as_a_candidate() -> None:
    """Extra rows stay as province candidates for Phase 5 to disambiguate."""
    fragments = [
        _line("กข 1234", 0.9, top=2.0, left=20.0),
        _line("ชลบุรี", 0.7, top=60.0, left=60.0),
        _line("CHONBURI", 0.6, top=110.0, left=55.0),
    ]

    reading = split_reading(fragments)

    assert reading.province_candidates == ("ชลบุรี", "CHONBURI")


def test_split_reading_of_single_row_has_no_province() -> None:
    """A one-row result is a plate number with no province to match."""
    reading = split_reading([_line("กข 1234", 0.9, top=3.0, left=20.0)])

    assert reading.plate_text == "กข 1234"
    assert reading.province_candidates == ()


def test_split_reading_of_nothing_is_empty() -> None:
    """No recognized text yields an empty reading rather than an error."""
    assert split_reading([]) == PlateReading(
        plate_text="", plate_confidence=0.0, province_candidates=()
    )
