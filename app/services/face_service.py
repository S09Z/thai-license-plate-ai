"""Face detection use case: validate an upload, run the detector, shape it."""

from functools import lru_cache

from app.core.config import get_settings
from app.schemas.detection import BoundingBox, DetectionResponse
from app.utils.image import load_image
from face.detector import FaceDetector


@lru_cache
def get_face_detector() -> FaceDetector:
    """Return the process-wide face detector, built from settings on first use.

    Returns:
        A memoized :class:`FaceDetector`; the model loads on its first
        detection.
    """
    settings = get_settings()
    return FaceDetector(
        model_path=settings.face_model_path,
        conf_threshold=settings.face_conf_threshold,
    )


def detect_faces(data: bytes, content_type: str) -> DetectionResponse:
    """Detect human faces in an uploaded image.

    Reports face locations only; nothing here identifies a person.

    Args:
        data: Raw bytes of the uploaded file.
        content_type: Content type declared by the client.

    Returns:
        The detected face boxes and their count.

    Raises:
        ImageValidationError: If the upload fails validation or decoding.
        FileNotFoundError: If the configured face model is missing.
    """
    settings = get_settings()
    image = load_image(
        data,
        content_type,
        max_bytes=settings.max_upload_bytes,
        allowed_types=settings.allowed_image_types,
    )

    detections = get_face_detector().detect(image)
    boxes = [
        BoundingBox(
            x1=detection.x1,
            y1=detection.y1,
            x2=detection.x2,
            y2=detection.y2,
            confidence=detection.confidence,
        )
        for detection in detections
    ]
    return DetectionResponse(count=len(boxes), boxes=boxes)
