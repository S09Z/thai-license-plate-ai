"""Tests for the POST /detect endpoint.

Inference is mocked throughout: no trained plate model exists yet, and the
suite must stay fast and weight-free.
"""

import io
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.core.config import get_settings
from app.services.detection_service import get_detector
from detector.detector import Detection, PlateDetector


def _png_bytes(width: int = 16, height: int = 8) -> bytes:
    """Return a tiny in-memory PNG suitable for upload tests."""
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color=(0, 128, 255)).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def stub_detections(monkeypatch: pytest.MonkeyPatch) -> list[Detection]:
    """Patch PlateDetector.detect to return a fixed result without weights."""
    detections = [Detection(x1=1, y1=2, x2=30, y2=14, confidence=0.91)]

    def _fake_detect(self: PlateDetector, image: np.ndarray) -> list[Detection]:
        return detections

    monkeypatch.setattr(PlateDetector, "detect", _fake_detect)
    return detections


@pytest.fixture
def tiny_upload_cap(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Shrink the upload size cap so oversize uploads are cheap to test."""
    monkeypatch.setenv("APP_MAX_UPLOAD_BYTES", "10")
    get_settings.cache_clear()
    get_detector.cache_clear()
    yield
    get_settings.cache_clear()
    get_detector.cache_clear()


def test_detect_returns_boxes_for_valid_png(
    client: TestClient, stub_detections: list[Detection]
) -> None:
    """A valid upload returns 200 with the detected boxes."""
    response = client.post(
        "/detect", files={"file": ("plate.png", _png_bytes(), "image/png")}
    )

    assert response.status_code == 200
    assert response.json() == {
        "count": 1,
        "boxes": [{"x1": 1, "y1": 2, "x2": 30, "y2": 14, "confidence": 0.91}],
    }


def test_detect_rejects_unsupported_content_type(
    client: TestClient, stub_detections: list[Detection]
) -> None:
    """An upload outside the content-type allowlist returns 415."""
    response = client.post(
        "/detect", files={"file": ("plate.pdf", _png_bytes(), "application/pdf")}
    )

    assert response.status_code == 415


def test_detect_rejects_oversize_upload(
    client: TestClient, stub_detections: list[Detection], tiny_upload_cap: None
) -> None:
    """An upload beyond the configured size cap returns 413."""
    response = client.post(
        "/detect", files={"file": ("plate.png", _png_bytes(), "image/png")}
    )

    assert response.status_code == 413


def test_detect_reports_unavailable_when_weights_missing(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With no weights installed, a valid upload returns 503 rather than 500."""
    monkeypatch.setenv("APP_DETECTOR_MODEL_PATH", str(tmp_path / "absent.pt"))
    get_settings.cache_clear()
    get_detector.cache_clear()

    response = client.post(
        "/detect", files={"file": ("plate.png", _png_bytes(), "image/png")}
    )

    get_settings.cache_clear()
    get_detector.cache_clear()

    assert response.status_code == 503


def test_detect_rejects_undecodable_payload(
    client: TestClient, stub_detections: list[Detection]
) -> None:
    """Bytes that do not decode to an image return 400."""
    response = client.post(
        "/detect", files={"file": ("plate.png", b"not an image", "image/png")}
    )

    assert response.status_code == 400
