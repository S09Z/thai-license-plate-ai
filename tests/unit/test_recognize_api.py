"""Tests for the POST /recognize endpoint.

Both engines are stubbed: no plate weights exist, and the real OCR engine
downloads models on first use. The suite stays fast, weight-free and offline.
"""

import io
from collections.abc import Callable, Iterator, Sequence

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.core.config import get_settings
from app.services.detection_service import get_detector
from app.services.ocr_service import get_recognizer
from detector.detector import Detection, PlateDetector
from ocr.reading import TextLine
from ocr.recognizer import PlateOCR

# Boxes below sit inside this frame; a plate crop needs real pixels for the
# perspective stage to operate on.
IMAGE_SIZE = (320, 240)

FIRST_PLATE = Detection(x1=10, y1=20, x2=210, y2=120, confidence=0.91)
SECOND_PLATE = Detection(x1=40, y1=140, x2=240, y2=230, confidence=0.77)


def _png_bytes(size: tuple[int, int] = IMAGE_SIZE) -> bytes:
    """Return an in-memory PNG scene suitable for upload tests."""
    buffer = io.BytesIO()
    Image.new("RGB", size, color=(240, 240, 240)).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture(autouse=True)
def reset_singletons() -> Iterator[None]:
    """Drop the memoized detector and recognizer around every test.

    Both are ``lru_cache``d per process, so a stubbed instance would otherwise
    leak into later tests.
    """
    get_settings.cache_clear()
    get_detector.cache_clear()
    get_recognizer.cache_clear()
    yield
    get_settings.cache_clear()
    get_detector.cache_clear()
    get_recognizer.cache_clear()


@pytest.fixture
def stub_detector(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[Sequence[Detection]], None]:
    """Return an installer that patches PlateDetector.detect with fixed boxes."""

    def _install(detections: Sequence[Detection]) -> None:
        def _fake_detect(self: PlateDetector, image: np.ndarray) -> list[Detection]:
            return list(detections)

        monkeypatch.setattr(PlateDetector, "detect", _fake_detect)

    return _install


@pytest.fixture
def stub_ocr(monkeypatch: pytest.MonkeyPatch) -> Callable[[str, str], None]:
    """Return an installer that patches PlateOCR.recognize with fixed text.

    Fragments are placed as the real engine reports them: the number band
    above, the province band below.
    """

    def _install(plate_text: str, province_text: str) -> None:
        lines = [
            TextLine(text=plate_text, confidence=0.93, top=5.0, bottom=45.0, left=20.0)
        ]
        if province_text:
            lines.append(
                TextLine(
                    text=province_text,
                    confidence=0.82,
                    top=70.0,
                    bottom=105.0,
                    left=60.0,
                )
            )

        def _fake_recognize(self: PlateOCR, crop: np.ndarray) -> list[TextLine]:
            return list(lines)

        monkeypatch.setattr(PlateOCR, "recognize", _fake_recognize)

    return _install


def _post(
    client: TestClient, data: bytes = b"", content_type: str = "image/png"
) -> httpx.Response:
    """POST an upload to /recognize."""
    return client.post(
        "/recognize",
        files={"file": ("scene.png", data or _png_bytes(), content_type)},
    )


def test_recognize_returns_one_result_per_detected_plate(
    client: TestClient,
    stub_detector: Callable[[Sequence[Detection]], None],
    stub_ocr: Callable[[str, str], None],
) -> None:
    """Every detected plate is recognized and reported, not just the best one."""
    stub_detector([FIRST_PLATE, SECOND_PLATE])
    stub_ocr("กข 1234", "ชลบุรี")

    response = _post(client)

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    assert body["plates"][0] == {
        "box": {"x1": 10, "y1": 20, "x2": 210, "y2": 120, "confidence": 0.91},
        "plate_text": "กข 1234",
        "plate_confidence": 0.93,
        "is_well_formed": True,
        "province": "ชลบุรี",
        "province_confidence": 1.0,
        "province_candidates": ["ชลบุรี"],
    }
    assert body["plates"][1]["box"]["confidence"] == 0.77


def test_recognize_recovers_a_damaged_province(
    client: TestClient,
    stub_detector: Callable[[Sequence[Detection]], None],
    stub_ocr: Callable[[str, str], None],
) -> None:
    """The real Phase 3 misread is corrected by the RAG stage."""
    stub_detector([FIRST_PLATE])
    stub_ocr("กข 1234", "ชลบรดี")

    plate = _post(client).json()["plates"][0]

    assert plate["province"] == "ชลบุรี"
    assert plate["province_confidence"] == pytest.approx(0.8)
    # The raw reading survives, so a client can show what was actually seen.
    assert plate["province_candidates"] == ["ชลบรดี"]


def test_recognize_reports_unknown_province_when_rag_abstains(
    client: TestClient,
    stub_detector: Callable[[Sequence[Detection]], None],
    stub_ocr: Callable[[str, str], None],
) -> None:
    """An ambiguous candidate yields no province rather than a guess."""
    stub_detector([FIRST_PLATE])
    stub_ocr("กข 1234", "เพชรบรดี")

    plate = _post(client).json()["plates"][0]

    assert plate["province"] is None
    assert plate["province_confidence"] is None
    assert plate["province_candidates"] == ["เพชรบรดี"]
    # Abstention is not failure: the plate number is still reported.
    assert plate["plate_text"] == "กข 1234"


def test_recognize_reports_a_misread_number_verbatim(
    client: TestClient,
    stub_detector: Callable[[Sequence[Detection]], None],
    stub_ocr: Callable[[str, str], None],
) -> None:
    """Text failing the plate pattern is returned as read, never coerced."""
    stub_detector([FIRST_PLATE])
    stub_ocr("VEZL", "")

    plate = _post(client).json()["plates"][0]

    assert plate["plate_text"] == "VEZL"
    assert plate["is_well_formed"] is False
    assert plate["province"] is None
    assert plate["province_candidates"] == []


def test_recognize_returns_no_plates_when_none_are_detected(
    client: TestClient,
    stub_detector: Callable[[Sequence[Detection]], None],
    stub_ocr: Callable[[str, str], None],
) -> None:
    """A scene without plates is a 200 with an empty list, not an error."""
    stub_detector([])
    stub_ocr("กข 1234", "ชลบุรี")

    response = _post(client)

    assert response.status_code == 200
    assert response.json() == {"count": 0, "plates": []}


def test_recognize_skips_a_box_with_no_area_in_the_image(
    client: TestClient,
    stub_detector: Callable[[Sequence[Detection]], None],
    stub_ocr: Callable[[str, str], None],
) -> None:
    """One degenerate box is dropped; the plates around it still come back."""
    outside = Detection(x1=400, y1=400, x2=500, y2=500, confidence=0.5)
    stub_detector([FIRST_PLATE, outside])
    stub_ocr("กข 1234", "ชลบุรี")

    response = _post(client)

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    assert body["plates"][0]["box"]["confidence"] == 0.91


def test_recognize_rejects_unsupported_content_type(
    client: TestClient,
    stub_detector: Callable[[Sequence[Detection]], None],
    stub_ocr: Callable[[str, str], None],
) -> None:
    """An upload outside the content-type allowlist returns 415."""
    stub_detector([FIRST_PLATE])
    stub_ocr("กข 1234", "ชลบุรี")

    assert _post(client, content_type="application/pdf").status_code == 415


def test_recognize_rejects_undecodable_payload(
    client: TestClient,
    stub_detector: Callable[[Sequence[Detection]], None],
    stub_ocr: Callable[[str, str], None],
) -> None:
    """Bytes that do not decode to an image return 400."""
    stub_detector([FIRST_PLATE])
    stub_ocr("กข 1234", "ชลบุรี")

    assert _post(client, data=b"not an image").status_code == 400


def test_recognize_rejects_oversize_upload(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    stub_detector: Callable[[Sequence[Detection]], None],
    stub_ocr: Callable[[str, str], None],
) -> None:
    """An upload beyond the configured size cap returns 413."""
    stub_detector([FIRST_PLATE])
    stub_ocr("กข 1234", "ชลบุรี")
    monkeypatch.setenv("APP_MAX_UPLOAD_BYTES", "10")
    get_settings.cache_clear()

    assert _post(client).status_code == 413


def test_recognize_reports_unavailable_when_weights_missing(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    stub_ocr: Callable[[str, str], None],
) -> None:
    """With no weights installed, a valid upload returns 503 rather than 500."""
    stub_ocr("กข 1234", "ชลบุรี")
    monkeypatch.setenv("APP_DETECTOR_MODEL_PATH", "models/detector/does-not-exist.pt")
    get_settings.cache_clear()
    get_detector.cache_clear()

    assert _post(client).status_code == 503
