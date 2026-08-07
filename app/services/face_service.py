"""Face detection use case: validate an upload, run the detector, shape it."""

from functools import lru_cache

from app.core.config import get_settings
from app.schemas.detection import BoundingBox
from app.schemas.face import Face, FaceResponse, FacialLandmarksModel
from app.utils.image import load_image
from face.detector import FaceDetector
from face.landmarks import FaceLandmarker, FacialLandmarks


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


@lru_cache
def get_face_landmarker() -> FaceLandmarker:
    """Return the process-wide landmark fitter, built from settings on first use.

    Returns:
        A memoized :class:`FaceLandmarker`; the 54 MB model loads on its first
        fit, so an installation that never asks for landmarks never pays for it.
    """
    return FaceLandmarker(model_path=get_settings().face_landmark_model_path)


def _to_model(fitted: FacialLandmarks) -> FacialLandmarksModel:
    """Convert fitted landmark groups into their response shape."""
    return FacialLandmarksModel(
        right_eyebrow=fitted.right_eyebrow,
        left_eyebrow=fitted.left_eyebrow,
        nose=fitted.nose,
        right_eye=fitted.right_eye,
        left_eye=fitted.left_eye,
        mouth=fitted.mouth,
        jaw=fitted.jaw,
        points=fitted.points,
        triangles=fitted.triangles,
    )


def detect_faces(
    data: bytes, content_type: str, landmarks: bool = False, mesh: bool = False
) -> FaceResponse:
    """Detect human faces in an uploaded image.

    Reports geometry only; nothing here identifies a person.

    Args:
        data: Raw bytes of the uploaded file.
        content_type: Content type declared by the client.
        landmarks: Whether to fit feature points inside each detected face.
        mesh: Whether to also report the jaw contour, the flat 68-point array
            and its triangulation. Implies fitting, so it works on its own.

    Returns:
        The detected faces and their count, with landmarks only when asked for.

    Raises:
        ImageValidationError: If the upload fails validation or decoding.
        FileNotFoundError: If a model needed for this request is missing.
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

    # Mesh implies fitting: the triangulation is built from the fitted points,
    # so asking for it alone is enough.
    if not (landmarks or mesh):
        return FaceResponse(count=len(boxes), faces=[Face(box=box) for box in boxes])

    # A fit that does not converge returns nothing rather than a partial list,
    # so pad back to one entry per box instead of zipping the boxes away.
    fitted = get_face_landmarker().fit(image, detections, mesh=mesh)
    groups: list[FacialLandmarksModel | None] = [_to_model(face) for face in fitted] + [
        None
    ] * (len(boxes) - len(fitted))

    faces = [
        Face(box=box, landmarks=group) for box, group in zip(boxes, groups, strict=True)
    ]
    return FaceResponse(count=len(faces), faces=faces)
