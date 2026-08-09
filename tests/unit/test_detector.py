"""Tests for the YOLO plate detector wrapper."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from detector import detector as detector_module
from detector.detector import Detection, PlateDetector


class _FakeBoxes:
    """Stands in for an Ultralytics ``Boxes`` object."""

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray) -> None:
        self.xyxy = xyxy
        self.conf = conf


class _FakeResult:
    """Stands in for a single Ultralytics ``Results`` entry."""

    def __init__(self, boxes: _FakeBoxes | None) -> None:
        self.boxes = boxes


class _FakeModel:
    """Callable stand-in for ``ultralytics.YOLO`` recording its invocations."""

    def __init__(self, results: list[_FakeResult]) -> None:
        self._results = results
        self.calls: list[dict[str, Any]] = []

    def __call__(self, image: np.ndarray, **kwargs: Any) -> list[_FakeResult]:
        self.calls.append(kwargs)
        return self._results


@pytest.fixture
def image() -> np.ndarray:
    """Return a blank BGR image."""
    return np.zeros((32, 32, 3), dtype=np.uint8)


def test_detect_maps_model_output_to_detections(
    monkeypatch: pytest.MonkeyPatch, image: np.ndarray
) -> None:
    """Model boxes are converted into integer-bounded Detection objects."""
    model = _FakeModel(
        [
            _FakeResult(
                _FakeBoxes(
                    xyxy=np.array([[10.6, 20.2, 110.9, 60.4]]),
                    conf=np.array([0.87]),
                )
            )
        ]
    )
    monkeypatch.setattr(detector_module, "_load_yolo", lambda _path: model)

    detections = PlateDetector("models/detector/best.pt", 0.25).detect(image)

    assert detections == [Detection(x1=10, y1=20, x2=110, y2=60, confidence=0.87)]
    assert model.calls == [{"conf": 0.25, "verbose": False}]


def test_detect_returns_empty_list_when_no_boxes(
    monkeypatch: pytest.MonkeyPatch, image: np.ndarray
) -> None:
    """A result carrying no boxes yields no detections."""
    monkeypatch.setattr(
        detector_module, "_load_yolo", lambda _path: _FakeModel([_FakeResult(None)])
    )

    assert PlateDetector("models/detector/best.pt", 0.25).detect(image) == []


def test_model_is_loaded_once_across_calls(
    monkeypatch: pytest.MonkeyPatch, image: np.ndarray
) -> None:
    """Weights are loaded lazily on first use and reused afterwards."""
    loads: list[str] = []

    def _fake_load(path: str) -> _FakeModel:
        loads.append(path)
        return _FakeModel([_FakeResult(None)])

    monkeypatch.setattr(detector_module, "_load_yolo", _fake_load)

    plate_detector = PlateDetector("models/detector/best.pt", 0.25)
    assert loads == []

    plate_detector.detect(image)
    plate_detector.detect(image)

    assert loads == ["models/detector/best.pt"]


def test_detect_raises_when_weights_file_missing(
    tmp_path: Path, image: np.ndarray
) -> None:
    """A missing weights file fails with a clear error, only when used."""
    missing = tmp_path / "absent.pt"
    plate_detector = PlateDetector(str(missing), 0.25)

    with pytest.raises(FileNotFoundError, match=str(missing)):
        plate_detector.detect(image)
