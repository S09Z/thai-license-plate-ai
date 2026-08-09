"""Detection use case: validate an upload, run the detector, shape the result."""

from functools import lru_cache

from app.core.config import get_settings
from app.schemas.detection import BoundingBox, DetectionResponse
from app.utils.image import load_image
from detector.detector import PlateDetector


@lru_cache
def get_detector() -> PlateDetector:
    """Return the process-wide detector, built from settings on first use.

    Returns:
        A memoized :class:`PlateDetector`; weights load on its first detection.
    """
    settings = get_settings()
    return PlateDetector(
        model_path=settings.detector_model_path,
        conf_threshold=settings.detector_conf_threshold,
    )


def detect_plates(data: bytes, content_type: str) -> DetectionResponse:
    """Detect license plates in an uploaded image.

    Args:
        data: Raw bytes of the uploaded file.
        content_type: Content type declared by the client.

    Returns:
        The detected plate boxes and their count.

    Raises:
        ImageValidationError: If the upload fails validation or decoding.
        FileNotFoundError: If the configured detector weights are missing.
    """
    settings = get_settings()
    image = load_image(
        data,
        content_type,
        max_bytes=settings.max_upload_bytes,
        allowed_types=settings.allowed_image_types,
    )

    detections = get_detector().detect(image)
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
