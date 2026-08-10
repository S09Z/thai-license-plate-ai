"""Tests for the ``POST /detect/faces`` endpoint.

Inference is mocked throughout so the suite stays fast and never loads the
ONNX model.
"""

import io
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.core.config import get_settings
from app.services.face_service import get_face_detector, get_face_landmarker
from detector.detector import Detection
from face.detector import FaceDetector
from face.landmarks import FaceLandmarker, FacialLandmarks


def _png_bytes(width: int = 16, height: int = 8) -> bytes:
    """Return a tiny in-memory PNG suitable for upload tests."""
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color=(0, 128, 255)).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture(autouse=True)
def reset_singletons() -> Iterator[None]:
    """Clear cached settings and the models so env patches take effect."""
    get_settings.cache_clear()
    get_face_detector.cache_clear()
    get_face_landmarker.cache_clear()
    yield
    get_settings.cache_clear()
    get_face_detector.cache_clear()
    get_face_landmarker.cache_clear()


@pytest.fixture
def stub_faces(monkeypatch: pytest.MonkeyPatch) -> list[Detection]:
    """Patch FaceDetector.detect to return a result without the model."""
    detections = [Detection(x1=3, y1=4, x2=13, y2=16, confidence=0.88)]

    def _fake_detect(self: FaceDetector, image: np.ndarray) -> list[Detection]:
        return detections

    monkeypatch.setattr(FaceDetector, "detect", _fake_detect)
    return detections


@pytest.fixture
def stub_landmarks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch FaceLandmarker.fit to return a result without the model.

    The stub honours ``mesh`` rather than ignoring it, so the tests exercise
    the flag the route actually forwards.
    """
    features = FacialLandmarks(
        right_eyebrow=[(4, 6)],
        left_eyebrow=[(10, 6)],
        nose=[(7, 9)],
        right_eye=[(5, 8)],
        left_eye=[(9, 8)],
        mouth=[(7, 13)],
    )
    meshed = replace(
        features,
        jaw=[(3, 10)],
        points=[(4, 6), (10, 6), (7, 9), (5, 8), (9, 8), (7, 13), (3, 10)],
        triangles=[(0, 1, 2), (2, 3, 4)],
    )

    def _fake_fit(
        self: FaceLandmarker,
        image: np.ndarray,
        boxes: list[Detection],
        *,
        mesh: bool = False,
    ) -> list[FacialLandmarks]:
        return [meshed if mesh else features] * len(boxes)

    monkeypatch.setattr(FaceLandmarker, "fit", _fake_fit)


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
        "faces": [
            {
                "box": {"x1": 3, "y1": 4, "x2": 13, "y2": 16, "confidence": 0.88},
                "landmarks": None,
            }
        ],
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
    assert response.json() == {"count": 0, "faces": []}


def test_detect_faces_reports_landmarks_when_asked(
    client: TestClient, stub_faces: list[Detection], stub_landmarks: None
) -> None:
    """``?landmarks=true`` adds the six named feature groups to each face.

    The exact comparison is deliberate: the three null mesh keys are what
    proves the jaw stays behind its own flag rather than riding along here.
    """
    response = client.post(
        "/detect/faces?landmarks=true",
        files={"file": ("scene.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 200
    (face,) = response.json()["faces"]
    assert face["box"]["x1"] == 3
    assert face["landmarks"] == {
        "right_eyebrow": [[4, 6]],
        "left_eyebrow": [[10, 6]],
        "nose": [[7, 9]],
        "right_eye": [[5, 8]],
        "left_eye": [[9, 8]],
        "mouth": [[7, 13]],
        "jaw": None,
        "points": None,
        "triangles": None,
    }


def test_detect_faces_reports_the_mesh_when_asked(
    client: TestClient, stub_faces: list[Detection], stub_landmarks: None
) -> None:
    """``?mesh=true`` adds the jaw, the flat point array and its triangles."""
    response = client.post(
        "/detect/faces?mesh=true",
        files={"file": ("scene.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 200
    (face,) = response.json()["faces"]
    assert face["landmarks"]["jaw"] == [[3, 10]]
    assert face["landmarks"]["triangles"] == [[0, 1, 2], [2, 3, 4]]
    assert len(face["landmarks"]["points"]) == 7
    # The six feature groups are still reported, so mesh is a superset rather
    # than a different response the client has to special-case.
    assert face["landmarks"]["mouth"] == [[7, 13]]


def test_detect_faces_mesh_needs_the_landmark_model(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_faces: list[Detection],
) -> None:
    """Mesh implies fitting, so a missing model is still a 503 not a 500."""
    monkeypatch.setenv("APP_FACE_LANDMARK_MODEL_PATH", str(tmp_path / "absent.yaml"))
    get_settings.cache_clear()

    response = client.post(
        "/detect/faces?mesh=true",
        files={"file": ("scene.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 503


def test_detect_faces_fast_rescales_boxes_back_to_source_pixels(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, stub_faces: list[Detection]
) -> None:
    """Downscaling the frame must not leak into the coordinates the client sees.

    The 16x8 upload is capped to an 8px longest edge, so detector coordinates
    arrive in an 8x4 frame and are scaled back by (2, 2) to source pixels.
    """
    monkeypatch.setenv("APP_FACE_FAST_MAX_SIZE", "8")
    get_settings.cache_clear()
    get_face_detector.cache_clear()

    response = client.post(
        "/detect/faces?fast=true",
        files={"file": ("scene.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 200
    (face,) = response.json()["faces"]
    assert face["box"] == {"x1": 6, "y1": 8, "x2": 26, "y2": 32, "confidence": 0.88}


def test_detect_faces_fast_rescales_landmark_points_too(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    stub_faces: list[Detection],
    stub_landmarks: None,
) -> None:
    """Feature points found on the small frame come back in source pixels."""
    monkeypatch.setenv("APP_FACE_FAST_MAX_SIZE", "8")
    get_settings.cache_clear()
    get_face_detector.cache_clear()
    get_face_landmarker.cache_clear()

    response = client.post(
        "/detect/faces?fast=true&landmarks=true",
        files={"file": ("scene.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 200
    (face,) = response.json()["faces"]
    assert face["landmarks"]["mouth"] == [[14, 26]]
    assert face["landmarks"]["right_eyebrow"] == [[8, 12]]


def test_detect_faces_fast_with_no_cap_leaves_coordinates_alone(
    client: TestClient, stub_faces: list[Detection]
) -> None:
    """Below the cap the fast path passes the frame through unchanged.

    The default cap (480) is larger than a 16px upload, so nothing rescales and
    the result must be byte-identical to the non-fast request.
    """
    base = client.post(
        "/detect/faces", files={"file": ("scene.png", _png_bytes(), "image/png")}
    )
    fast = client.post(
        "/detect/faces?fast=true",
        files={"file": ("scene.png", _png_bytes(), "image/png")},
    )

    assert fast.status_code == 200
    assert fast.json() == base.json()


def test_detect_faces_fast_still_rejects_bad_uploads(
    client: TestClient, stub_faces: list[Detection]
) -> None:
    """The fast path does not skip the upload-validation contract."""
    response = client.post(
        "/detect/faces?fast=true",
        files={"file": ("scene.png", b"not an image", "image/png")},
    )

    assert response.status_code == 400


def test_openapi_exposes_the_fast_parameter(client: TestClient) -> None:
    """The schema advertises ``fast`` so clients can discover the fast path."""
    schema = client.get("/openapi.json").json()
    params = {
        param["name"]
        for param in schema["paths"]["/detect/faces"]["post"]["parameters"]
    }

    assert "fast" in params


def test_detect_faces_ignores_the_landmark_model_by_default(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_faces: list[Detection],
) -> None:
    """Without the opt-in, a missing landmark model changes nothing."""
    monkeypatch.setenv("APP_FACE_LANDMARK_MODEL_PATH", str(tmp_path / "absent.yaml"))
    get_settings.cache_clear()
    get_face_landmarker.cache_clear()

    response = client.post(
        "/detect/faces", files={"file": ("scene.png", _png_bytes(), "image/png")}
    )

    assert response.status_code == 200
    assert response.json()["faces"][0]["landmarks"] is None


def test_detect_faces_reports_unavailable_when_landmark_model_missing(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stub_faces: list[Detection],
) -> None:
    """An opted-in request with no LBF model returns 503, not 500.

    OpenCV raises its own ``cv2.error`` here, which would escape as a 500; this
    is the check that keeps the 503 contract honest.
    """
    monkeypatch.setenv("APP_FACE_LANDMARK_MODEL_PATH", str(tmp_path / "absent.yaml"))
    get_settings.cache_clear()
    get_face_landmarker.cache_clear()

    response = client.post(
        "/detect/faces?landmarks=true",
        files={"file": ("scene.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 503


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
