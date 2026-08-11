"""Tests for the PaddleOCR plate recognizer wrapper."""

import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from ocr import recognizer as recognizer_module
from ocr.recognizer import PlateOCR


def _poly(top: float, left: float = 10.0) -> list[list[float]]:
    """Return a rectangular polygon anchored at ``left`` with its top at ``top``."""
    right = left + 190.0
    return [[left, top], [right, top], [right, top + 30.0], [left, top + 30.0]]


class _FakeEngine:
    """Callable stand-in for a PaddleOCR instance, recording its calls."""

    def __init__(self, results: list[dict[str, Any]]) -> None:
        self._results = results
        self.calls: list[tuple[int, ...]] = []

    def predict(self, image: np.ndarray) -> list[dict[str, Any]]:
        self.calls.append(image.shape)
        return self._results


@pytest.fixture
def crop() -> np.ndarray:
    """Return a blank plate-sized crop."""
    return np.zeros((128, 256, 3), dtype=np.uint8)


def test_recognize_maps_engine_output_to_text_lines(
    monkeypatch: pytest.MonkeyPatch, crop: np.ndarray
) -> None:
    """Texts, scores and polygons become TextLines ordered top-to-bottom."""
    engine = _FakeEngine(
        [
            {
                "rec_texts": ["ชลบุรี", "กข 1234"],
                "rec_scores": [0.81, 0.96],
                "rec_polys": [_poly(70.0), _poly(6.0)],
            }
        ]
    )
    monkeypatch.setattr(
        recognizer_module, "_load_paddleocr", lambda *args, **kwargs: engine
    )

    lines = PlateOCR(lang="th", min_confidence=0.3).recognize(crop)

    assert [line.text for line in lines] == ["กข 1234", "ชลบุรี"]
    assert [line.confidence for line in lines] == [0.96, 0.81]
    assert [line.top for line in lines] == [6.0, 70.0]
    assert engine.calls == [(128, 256, 3)]


def test_recognize_carries_box_geometry_from_polygons(
    monkeypatch: pytest.MonkeyPatch, crop: np.ndarray
) -> None:
    """Each fragment keeps its box extents so rows can be reassembled later."""
    engine = _FakeEngine(
        [
            {
                "rec_texts": ["กข"],
                "rec_scores": [0.9],
                "rec_polys": [_poly(6.0, left=24.0)],
            }
        ]
    )
    monkeypatch.setattr(
        recognizer_module, "_load_paddleocr", lambda *args, **kwargs: engine
    )

    (line,) = PlateOCR(lang="th", min_confidence=0.3).recognize(crop)

    assert (line.top, line.bottom, line.left) == (6.0, 36.0, 24.0)


def test_recognize_drops_lines_below_min_confidence(
    monkeypatch: pytest.MonkeyPatch, crop: np.ndarray
) -> None:
    """Low-confidence noise is discarded rather than passed downstream."""
    engine = _FakeEngine(
        [
            {
                "rec_texts": ["กข 1234", "~junk~"],
                "rec_scores": [0.95, 0.12],
                "rec_polys": [_poly(5.0), _poly(60.0)],
            }
        ]
    )
    monkeypatch.setattr(
        recognizer_module, "_load_paddleocr", lambda *args, **kwargs: engine
    )

    lines = PlateOCR(lang="th", min_confidence=0.5).recognize(crop)

    assert [line.text for line in lines] == ["กข 1234"]


def test_recognize_returns_nothing_when_engine_finds_no_text(
    monkeypatch: pytest.MonkeyPatch, crop: np.ndarray
) -> None:
    """An empty recognition result is not an error."""
    engine = _FakeEngine([{"rec_texts": [], "rec_scores": [], "rec_polys": []}])
    monkeypatch.setattr(
        recognizer_module, "_load_paddleocr", lambda *args, **kwargs: engine
    )

    assert PlateOCR(lang="th", min_confidence=0.3).recognize(crop) == []


def test_recognize_tolerates_result_without_recognition_keys(
    monkeypatch: pytest.MonkeyPatch, crop: np.ndarray
) -> None:
    """A result carrying no recognition fields yields no lines."""
    engine = _FakeEngine([{}])
    monkeypatch.setattr(
        recognizer_module, "_load_paddleocr", lambda *args, **kwargs: engine
    )

    assert PlateOCR(lang="th", min_confidence=0.3).recognize(crop) == []


def test_load_paddleocr_skips_submodels_a_plate_crop_does_not_need(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The engine is built without document orientation/unwarping stages.

    A crop arriving here is already detected and rectified, so those stages
    only cost latency; measured, they also corrupt results on upscaled crops.
    """
    captured: dict[str, Any] = {}

    class _FakePaddleOCR:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setitem(
        sys.modules, "paddleocr", SimpleNamespace(PaddleOCR=_FakePaddleOCR)
    )

    recognizer_module._load_paddleocr("th", 192, "max")

    assert captured == {
        "lang": "th",
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
        "text_det_limit_side_len": 192,
        "text_det_limit_type": "max",
    }


def test_engine_is_loaded_once_with_configured_language(
    monkeypatch: pytest.MonkeyPatch, crop: np.ndarray
) -> None:
    """Weights load lazily on first use, with the configured language."""
    langs: list[str] = []

    def _fake_load(
        lang: str, det_limit_side_len: int, det_limit_type: str
    ) -> _FakeEngine:
        langs.append(lang)
        return _FakeEngine([{"rec_texts": [], "rec_scores": [], "rec_polys": []}])

    monkeypatch.setattr(recognizer_module, "_load_paddleocr", _fake_load)

    plate_ocr = PlateOCR(lang="th", min_confidence=0.3)
    assert langs == []

    plate_ocr.recognize(crop)
    plate_ocr.recognize(crop)

    assert langs == ["th"]


def test_detection_limit_is_forwarded_to_the_engine(
    monkeypatch: pytest.MonkeyPatch, crop: np.ndarray
) -> None:
    """The configured detection-input cap reaches ``_load_paddleocr``."""
    calls: list[tuple[str, int, str]] = []

    def _fake_load(
        lang: str, det_limit_side_len: int, det_limit_type: str
    ) -> _FakeEngine:
        calls.append((lang, det_limit_side_len, det_limit_type))
        return _FakeEngine([{"rec_texts": [], "rec_scores": [], "rec_polys": []}])

    monkeypatch.setattr(recognizer_module, "_load_paddleocr", _fake_load)

    PlateOCR(
        lang="th",
        min_confidence=0.3,
        det_limit_side_len=160,
        det_limit_type="max",
    ).recognize(crop)

    assert calls == [("th", 160, "max")]
