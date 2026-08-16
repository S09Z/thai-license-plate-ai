"""Tests for the YuNet face detector wrapper.

No ONNX file is loaded: ``_load_yunet`` is patched throughout, so the suite
stays offline and weight-free like the plate detector's tests.
"""

from pathlib import Path

import numpy as np
import pytest

from face import detector as face_module
from face.detector import FaceDetector


class _FakeYuNet:
    """Stands in for ``cv2.FaceDetectorYN``, recording its input sizes."""

    def __init__(self, faces: np.ndarray | None) -> None:
        self._faces = faces
        self.input_sizes: list[tuple[int, int]] = []

    def setInputSize(self, size: tuple[int, int]) -> None:  # noqa: N802
        """Record the frame size YuNet was told to expect."""
        self.input_sizes.append(size)

    def detect(self, image: np.ndarray) -> tuple[int, np.ndarray | None]:
        """Return the canned ``(retval, faces)`` pair."""
        return 1, self._faces


def _face_row(x: float, y: float, w: float, h: float, score: float) -> list[float]:
    """Build one YuNet output row: box, ten landmark values, then the score."""
    return [x, y, w, h, *([0.0] * 10), score]


@pytest.fixture
def image() -> np.ndarray:
    """Return a blank BGR frame that is wider than it is tall."""
    return np.zeros((48, 64, 3), dtype=np.uint8)


def test_detect_converts_width_height_to_corner_coordinates(
    monkeypatch: pytest.MonkeyPatch, image: np.ndarray
) -> None:
    """YuNet reports ``x, y, w, h``; detections carry ``x1, y1, x2, y2``."""
    faces = np.array([_face_row(10.6, 20.2, 30.9, 40.4, 0.93)], dtype=np.float32)
    monkeypatch.setattr(
        face_module, "_load_yunet", lambda _path, _conf: _FakeYuNet(faces)
    )

    (detection,) = FaceDetector("models/face/yunet.onnx", 0.6).detect(image)

    assert (detection.x1, detection.y1, detection.x2, detection.y2) == (10, 20, 40, 60)
    assert detection.confidence == pytest.approx(0.93)


def test_detect_discards_landmark_columns(
    monkeypatch: pytest.MonkeyPatch, image: np.ndarray
) -> None:
    """Columns 4-13 are landmarks; this phase reports geometry only."""
    row = _face_row(1.0, 2.0, 3.0, 4.0, 0.8)
    row[4:14] = [float(index) for index in range(10)]
    monkeypatch.setattr(
        face_module,
        "_load_yunet",
        lambda _path, _conf: _FakeYuNet(np.array([row], dtype=np.float32)),
    )

    (detection,) = FaceDetector("models/face/yunet.onnx", 0.6).detect(image)

    assert (detection.x1, detection.y1, detection.x2, detection.y2) == (1, 2, 4, 6)
    assert detection.confidence == pytest.approx(0.8)


def test_detect_returns_empty_list_when_no_faces(
    monkeypatch: pytest.MonkeyPatch, image: np.ndarray
) -> None:
    """A ``None`` face array means no faces, not an error."""
    monkeypatch.setattr(
        face_module, "_load_yunet", lambda _path, _conf: _FakeYuNet(None)
    )

    assert FaceDetector("models/face/yunet.onnx", 0.6).detect(image) == []


def test_detect_sets_input_size_to_frame_dimensions(
    monkeypatch: pytest.MonkeyPatch, image: np.ndarray
) -> None:
    """YuNet needs ``(width, height)`` up front, in that order."""
    model = _FakeYuNet(None)
    monkeypatch.setattr(face_module, "_load_yunet", lambda _path, _conf: model)

    FaceDetector("models/face/yunet.onnx", 0.6).detect(image)

    assert model.input_sizes == [(64, 48)]


def test_model_is_loaded_once_across_calls(
    monkeypatch: pytest.MonkeyPatch, image: np.ndarray
) -> None:
    """The ONNX model is loaded lazily on first use and reused afterwards."""
    loads: list[tuple[str, float]] = []

    def _fake_load(path: str, conf: float) -> _FakeYuNet:
        loads.append((path, conf))
        return _FakeYuNet(None)

    monkeypatch.setattr(face_module, "_load_yunet", _fake_load)

    face_detector = FaceDetector("models/face/yunet.onnx", 0.6)
    assert loads == []

    face_detector.detect(image)
    face_detector.detect(image)

    assert loads == [("models/face/yunet.onnx", 0.6)]


def test_detect_raises_when_model_file_missing(
    tmp_path: Path, image: np.ndarray
) -> None:
    """A missing ONNX file fails with a clear error, only when used."""
    missing = tmp_path / "absent.onnx"
    face_detector = FaceDetector(str(missing), 0.6)

    with pytest.raises(FileNotFoundError, match=str(missing)):
        face_detector.detect(image)


def test_detect_reuses_input_size_across_frames_of_the_same_shape(
    monkeypatch: pytest.MonkeyPatch, image: np.ndarray
) -> None:
    """Every frame sets its own size, so a resized stream stays correct."""
    model = _FakeYuNet(None)
    monkeypatch.setattr(face_module, "_load_yunet", lambda _path, _conf: model)

    face_detector = FaceDetector("models/face/yunet.onnx", 0.6)
    face_detector.detect(image)
    face_detector.detect(np.zeros((10, 20, 3), dtype=np.uint8))

    assert model.input_sizes == [(64, 48), (20, 10)]
