"""Tests for the ``POST /detect/faces`` endpoint.

Inference is mocked throughout so the suite stays fast and never loads the
ONNX model.
"""

import io
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.core.config import get_settings
from app.services.face_service import get_face_detector
from detector.detector import Detection
from face.detector import FaceDetector


def _png_bytes(width: int = 16, height: int = 8) -> bytes:
    """Return a tiny in-memory PNG suitable for upload tests."""
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color=(0, 128, 255)).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture(autouse=True)
def reset_singletons() -> Iterator[None]:
    """Clear cached settings and the detector so env patches take effect."""
    get_settings.cache_clear()
    get_face_detector.cache_clear()
    yield
    get_settings.cache_clear()
    get_face_detector.cache_clear()


@pytest.fixture
def stub_faces(monkeypatch: pytest.MonkeyPatch) -> list[Detection]:
    """Patch FaceDetector.detect to return a result without the model."""
    detections = [Detection(x1=3, y1=4, x2=13, y2=16, confidence=0.88)]

    def _fake_detect(self: FaceDetector, image: np.ndarray) -> list[Detection]:
        return detections

    monkeypatch.setattr(FaceDetector, "detect", _fake_detect)
    return detections


def test_detect_faces_returns_boxes_for_valid_png(
    client: TestClient, stub_faces: list[Detection]
) -> None:
    """A valid upload returns 200 with the detected face boxes."""
    response = client.post(
        "/detect/faces", files={"file": ("scene.png", _png_bytes(), "image/png")}
    )

    assert response.status_code == 200
    assert response.json() == {
        "count": 1,
        "boxes": [{"x1": 3, "y1": 4, "x2": 13, "y2": 16, "confidence": 0.88}],
    }


def test_detect_faces_returns_empty_when_no_faces(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A frame with no faces is a successful, empty result."""
    monkeypatch.setattr(FaceDetector, "detect", lambda self, image: [])

    response = client.post(
        "/detect/faces", files={"file": ("scene.png", _png_bytes(), "image/png")}
    )

    assert response.status_code == 200
    assert response.json() == {"count": 0, "boxes": []}


def test_detect_faces_rejects_unsupported_content_type(
    client: TestClient, stub_faces: list[Detection]
) -> None:
    """A disallowed content type returns 415."""
    response = client.post(
        "/detect/faces", files={"file": ("scene.pdf", _png_bytes(), "application/pdf")}
    )

    assert response.status_code == 415


def test_detect_faces_rejects_oversize_upload(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, stub_faces: list[Detection]
) -> None:
    """An upload past the configured cap returns 413."""
    monkeypatch.setenv("APP_MAX_UPLOAD_BYTES", "10")
    get_settings.cache_clear()
    get_face_detector.cache_clear()

    response = client.post(
        "/detect/faces", files={"file": ("scene.png", _png_bytes(), "image/png")}
    )

    assert response.status_code == 413


def test_detect_faces_rejects_undecodable_payload(
    client: TestClient, stub_faces: list[Detection]
) -> None:
    """Bytes that do not decode to an image return 400."""
    response = client.post(
        "/detect/faces", files={"file": ("scene.png", b"not an image", "image/png")}
    )

    assert response.status_code == 400


def test_detect_faces_reports_unavailable_when_model_missing(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With no ONNX model installed, a valid upload returns 503, not 500."""
    monkeypatch.setenv("APP_FACE_MODEL_PATH", str(tmp_path / "absent.onnx"))
    get_settings.cache_clear()
    get_face_detector.cache_clear()

    response = client.post(
        "/detect/faces", files={"file": ("scene.png", _png_bytes(), "image/png")}
    )

    assert response.status_code == 503


def test_plate_detection_still_works(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Registering the face router leaves POST /detect untouched."""
    from detector.detector import PlateDetector

    monkeypatch.setattr(
        PlateDetector,
        "detect",
        lambda self, image: [Detection(x1=1, y1=2, x2=3, y2=4, confidence=0.5)],
    )

    response = client.post(
        "/detect", files={"file": ("plate.png", _png_bytes(), "image/png")}
    )

    assert response.status_code == 200
    assert response.json()["count"] == 1
