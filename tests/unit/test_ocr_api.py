"""Tests for the POST /ocr endpoint.

Recognition is mocked: the real engine downloads models on first use, which
would make the suite slow and network-dependent.
"""

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from ocr.reading import TextLine
from ocr.recognizer import PlateOCR


def _png_bytes(width: int = 32, height: int = 16) -> bytes:
    """Return a tiny in-memory PNG suitable for upload tests."""
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color=(240, 240, 240)).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def stub_recognizer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch PlateOCR.recognize to return fragments as the real engine does.

    PaddleOCR splits ``กข 1234`` into two boxes, so the stub does too; the
    endpoint is expected to rejoin them.
    """

    def _fake_recognize(self: PlateOCR, crop: np.ndarray) -> list[TextLine]:
        return [
            TextLine(text="กข", confidence=0.93, top=5.0, bottom=45.0, left=20.0),
            TextLine(text="1234", confidence=0.95, top=6.0, bottom=46.0, left=90.0),
            TextLine(text="ชลบุรี", confidence=0.82, top=70.0, bottom=105.0, left=60.0),
        ]

    monkeypatch.setattr(PlateOCR, "recognize", _fake_recognize)


@pytest.fixture
def stub_empty_recognizer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch PlateOCR.recognize to find no text at all."""
    monkeypatch.setattr(PlateOCR, "recognize", lambda self, crop: [])


def test_ocr_returns_plate_text_and_province_candidates(
    client: TestClient, stub_recognizer: None
) -> None:
    """A valid crop returns the plate number plus province candidates."""
    response = client.post(
        "/ocr", files={"file": ("crop.png", _png_bytes(), "image/png")}
    )

    assert response.status_code == 200
    assert response.json() == {
        "plate_text": "กข 1234",
        # The weakest fragment in the joined row.
        "plate_confidence": 0.93,
        "province_candidates": ["ชลบุรี"],
        # ``lines`` stays at fragment granularity, so a client can inspect
        # exactly what the engine saw before it was joined into rows.
        "lines": [
            {"text": "กข", "confidence": 0.93},
            {"text": "1234", "confidence": 0.95},
            {"text": "ชลบุรี", "confidence": 0.82},
        ],
    }


def test_ocr_returns_empty_reading_when_no_text_found(
    client: TestClient, stub_empty_recognizer: None
) -> None:
    """An unreadable crop is a 200 with an empty reading, not an error."""
    response = client.post(
        "/ocr", files={"file": ("crop.png", _png_bytes(), "image/png")}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["plate_text"] == ""
    assert body["plate_confidence"] == 0.0
    assert body["province_candidates"] == []
    assert body["lines"] == []


def test_ocr_rejects_unsupported_content_type(
    client: TestClient, stub_recognizer: None
) -> None:
    """An upload outside the content-type allowlist returns 415."""
    response = client.post(
        "/ocr", files={"file": ("crop.pdf", _png_bytes(), "application/pdf")}
    )

    assert response.status_code == 415


def test_ocr_rejects_undecodable_payload(
    client: TestClient, stub_recognizer: None
) -> None:
    """Bytes that do not decode to an image return 400."""
    response = client.post(
        "/ocr", files={"file": ("crop.png", b"not an image", "image/png")}
    )

    assert response.status_code == 400
