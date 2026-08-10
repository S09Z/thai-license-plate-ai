"""Tests for apparent-gender and expression attribute inference.

Both ``cv2.dnn`` nets are faked throughout, returning canned score vectors, so
the suite stays offline and weight-free like the detector and landmark tests
beside it.
"""

from pathlib import Path

import numpy as np
import pytest

from detector.detector import Detection
from face import attributes as attributes_module
from face.attributes import EyeCenters, FaceAttributeReader


class _FakeNet:
    """Stands in for a ``cv2.dnn`` net, returning a canned forward() output."""

    def __init__(self, output: np.ndarray) -> None:
        self._output = output
        self.blobs: list[np.ndarray] = []

    def setInput(self, blob: np.ndarray) -> None:
        self.blobs.append(blob)

    def forward(self) -> np.ndarray:
        return self._output


@pytest.fixture
def image() -> np.ndarray:
    """Return a blank 200x200 BGR frame, large enough to crop and align."""
    return np.zeros((200, 200, 3), dtype=np.uint8)


@pytest.fixture
def box() -> Detection:
    """Return one face box in corner coordinates."""
    return Detection(x1=40, y1=40, x2=140, y2=160, confidence=0.9)


@pytest.fixture
def centered_eyes() -> EyeCenters:
    """Return plausible eye centers inside ``box``, for alignment to succeed."""
    return (70.0, 90.0), (110.0, 90.0)


def _reader(
    monkeypatch: pytest.MonkeyPatch,
    *,
    gender_output: np.ndarray,
    expression_output: np.ndarray,
    min_confidence: float = 0.5,
) -> FaceAttributeReader:
    """Build a reader whose two nets are stubbed with canned outputs."""
    monkeypatch.setattr(
        attributes_module, "_load_gender_net", lambda *_a: _FakeNet(gender_output)
    )
    monkeypatch.setattr(
        attributes_module,
        "_load_expression_net",
        lambda *_a: _FakeNet(expression_output),
    )
    return FaceAttributeReader(
        gender_model_path="models/face/gender_net.caffemodel",
        gender_proto_path="models/face/gender_deploy.prototxt",
        expression_model_path="models/face/expression.onnx",
        min_confidence=min_confidence,
    )


def test_read_without_boxes_never_loads_either_model(
    monkeypatch: pytest.MonkeyPatch, image: np.ndarray
) -> None:
    """No face means no inference, so an empty frame costs nothing."""

    def _fail(*_a: object) -> None:
        raise AssertionError("a model must not load when there are no faces")

    monkeypatch.setattr(attributes_module, "_load_gender_net", _fail)
    monkeypatch.setattr(attributes_module, "_load_expression_net", _fail)

    reader = FaceAttributeReader(
        gender_model_path="g.caffemodel",
        gender_proto_path="g.prototxt",
        expression_model_path="e.onnx",
        min_confidence=0.5,
    )

    assert reader.read(image, [], []) == []


def test_read_maps_gender_argmax_to_female(
    monkeypatch: pytest.MonkeyPatch,
    image: np.ndarray,
    box: Detection,
    centered_eyes: EyeCenters,
) -> None:
    """Softmax index 1 is ``female`` in the Levi-Hassner label order."""
    reader = _reader(
        monkeypatch,
        gender_output=np.array([[0.2, 0.8]], dtype=np.float32),
        expression_output=np.zeros((1, 7), dtype=np.float32),
    )

    (result,) = reader.read(image, [box], [centered_eyes])

    assert result.apparent_gender == "female"
    assert result.apparent_gender_confidence == pytest.approx(0.8)


def test_read_maps_gender_argmax_to_male(
    monkeypatch: pytest.MonkeyPatch,
    image: np.ndarray,
    box: Detection,
    centered_eyes: EyeCenters,
) -> None:
    """Softmax index 0 is ``male``."""
    reader = _reader(
        monkeypatch,
        gender_output=np.array([[0.9, 0.1]], dtype=np.float32),
        expression_output=np.zeros((1, 7), dtype=np.float32),
    )

    (result,) = reader.read(image, [box], [centered_eyes])

    assert result.apparent_gender == "male"
    assert result.apparent_gender_confidence == pytest.approx(0.9)


def test_read_maps_expression_argmax_to_its_label(
    monkeypatch: pytest.MonkeyPatch,
    image: np.ndarray,
    box: Detection,
    centered_eyes: EyeCenters,
) -> None:
    """Index 3 of the seven FER labels is ``happy``."""
    logits = np.full((1, 7), -10.0, dtype=np.float32)
    logits[0, 3] = 10.0
    reader = _reader(
        monkeypatch,
        gender_output=np.array([[0.5, 0.5]], dtype=np.float32),
        expression_output=logits,
    )

    (result,) = reader.read(image, [box], [centered_eyes])

    assert result.expression == "happy"
    assert result.expression_confidence > 0.9


def test_low_confidence_gender_abstains_but_still_reports_the_number(
    monkeypatch: pytest.MonkeyPatch,
    image: np.ndarray,
    box: Detection,
    centered_eyes: EyeCenters,
) -> None:
    """Below threshold the label is withheld; the confidence is not."""
    reader = _reader(
        monkeypatch,
        gender_output=np.array([[0.51, 0.49]], dtype=np.float32),
        expression_output=np.zeros((1, 7), dtype=np.float32),
        min_confidence=0.9,
    )

    (result,) = reader.read(image, [box], [centered_eyes])

    assert result.apparent_gender is None
    assert result.apparent_gender_confidence == pytest.approx(0.51)


def test_missing_eye_centers_abstains_expression_but_not_gender(
    monkeypatch: pytest.MonkeyPatch, image: np.ndarray, box: Detection
) -> None:
    """A landmark fit that did not converge should not block gender.

    Gender needs only the face box; expression needs the aligned crop, so a
    fit failure narrows the abstention to the field that actually depends on
    it rather than discarding both.
    """
    reader = _reader(
        monkeypatch,
        gender_output=np.array([[0.9, 0.1]], dtype=np.float32),
        expression_output=np.zeros((1, 7), dtype=np.float32),
    )

    (result,) = reader.read(image, [box], [None])

    assert result.apparent_gender == "male"
    assert result.expression is None
    assert result.expression_confidence == 0.0


def test_read_returns_one_result_per_box(
    monkeypatch: pytest.MonkeyPatch,
    image: np.ndarray,
    box: Detection,
    centered_eyes: EyeCenters,
) -> None:
    """Two boxes in, two attribute results out."""
    reader = _reader(
        monkeypatch,
        gender_output=np.array([[0.9, 0.1]], dtype=np.float32),
        expression_output=np.zeros((1, 7), dtype=np.float32),
    )
    second = Detection(x1=20, y1=20, x2=60, y2=80, confidence=0.7)

    results = reader.read(image, [box, second], [centered_eyes, centered_eyes])

    assert len(results) == 2


def test_gender_blob_is_227_square_with_the_reference_mean(
    monkeypatch: pytest.MonkeyPatch,
    image: np.ndarray,
    box: Detection,
    centered_eyes: EyeCenters,
) -> None:
    """The Caffe net was trained on 227x227 crops with a fixed BGR mean.

    A mismatched blob shape or mean silently degrades every prediction rather
    than raising, so the preprocessing itself needs its own assertion.
    """
    net = _FakeNet(np.array([[0.9, 0.1]], dtype=np.float32))
    monkeypatch.setattr(attributes_module, "_load_gender_net", lambda *_a: net)
    monkeypatch.setattr(
        attributes_module,
        "_load_expression_net",
        lambda *_a: _FakeNet(np.zeros((1, 7), dtype=np.float32)),
    )
    reader = FaceAttributeReader(
        gender_model_path="g.caffemodel",
        gender_proto_path="g.prototxt",
        expression_model_path="e.onnx",
        min_confidence=0.5,
    )

    reader.read(image, [box], [centered_eyes])

    (blob,) = net.blobs
    assert blob.shape == (1, 3, 227, 227)


def test_gender_net_raises_file_not_found_when_the_model_is_missing(
    tmp_path: Path, image: np.ndarray, box: Detection, centered_eyes: EyeCenters
) -> None:
    """A missing model must not surface as OpenCV's own ``cv2.error``."""
    missing = tmp_path / "absent.caffemodel"
    reader = FaceAttributeReader(
        gender_model_path=str(missing),
        gender_proto_path=str(tmp_path / "absent.prototxt"),
        expression_model_path="models/face/expression.onnx",
        min_confidence=0.5,
    )

    with pytest.raises(FileNotFoundError, match=str(missing)):
        reader.read(image, [box], [centered_eyes])


def test_expression_net_raises_file_not_found_when_the_model_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    image: np.ndarray,
    box: Detection,
    centered_eyes: EyeCenters,
) -> None:
    """Same 503-not-500 contract for the second model."""
    monkeypatch.setattr(
        attributes_module,
        "_load_gender_net",
        lambda *_a: _FakeNet(np.array([[0.9, 0.1]], dtype=np.float32)),
    )
    missing = tmp_path / "absent.onnx"
    reader = FaceAttributeReader(
        gender_model_path="models/face/gender_net.caffemodel",
        gender_proto_path="models/face/gender_deploy.prototxt",
        expression_model_path=str(missing),
        min_confidence=0.5,
    )

    with pytest.raises(FileNotFoundError, match=str(missing)):
        reader.read(image, [box], [centered_eyes])


def test_models_load_once_across_calls(
    monkeypatch: pytest.MonkeyPatch,
    image: np.ndarray,
    box: Detection,
    centered_eyes: EyeCenters,
) -> None:
    """Both nets load lazily on first use and are reused afterwards."""
    loads: list[str] = []

    def _fake_gender(model_path: str, _proto: str) -> _FakeNet:
        loads.append(model_path)
        return _FakeNet(np.array([[0.9, 0.1]], dtype=np.float32))

    def _fake_expression(model_path: str) -> _FakeNet:
        loads.append(model_path)
        return _FakeNet(np.zeros((1, 7), dtype=np.float32))

    monkeypatch.setattr(attributes_module, "_load_gender_net", _fake_gender)
    monkeypatch.setattr(attributes_module, "_load_expression_net", _fake_expression)

    reader = FaceAttributeReader(
        gender_model_path="g.caffemodel",
        gender_proto_path="g.prototxt",
        expression_model_path="e.onnx",
        min_confidence=0.5,
    )
    assert loads == []

    reader.read(image, [box], [centered_eyes])
    reader.read(image, [box], [centered_eyes])

    assert loads == ["g.caffemodel", "e.onnx"]
