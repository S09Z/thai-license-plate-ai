"""Face detection use case: validate an upload, run the detector, shape it."""

from functools import lru_cache

import cv2
import numpy as np

from app.core.config import get_settings
from app.schemas.detection import BoundingBox
from app.schemas.face import (
    Face,
    FaceAttributesModel,
    FaceResponse,
    FacialLandmarksModel,
)
from app.utils.image import load_image
from detector.detector import Detection
from face.attributes import EyeCenters, FaceAttributeReader, FaceAttributes
from face.detector import FaceDetector
from face.landmarks import FaceLandmarker, FacialLandmarks


def _prepare(
    image: np.ndarray, max_size: int | None
) -> tuple[np.ndarray, float, float]:
    """Return an image to infer on and the scale back to source pixels.

    The fast path downsizes frames so YuNet runs on a fraction of the pixels;
    the returned ``sx``/``sy`` map one work-image pixel onto ``sx``×``sy``
    source pixels, which is what puts the answer back in the coordinates a
    client drew with.

    Args:
        image: The decoded ``(H, W, 3)`` BGR frame.
        max_size: Longest-edge cap, or ``None`` to infer at full size.

    Returns:
        A ``(work_image, sx, sy)`` tuple where ``work_image`` is at most
        ``max_size`` on its longest edge.
    """
    height, width = image.shape[:2]
    if max_size is None or max(width, height) <= max_size:
        return image, 1.0, 1.0

    scale = max_size / max(width, height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    small = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return small, width / new_width, height / new_height


def _to_source(box: Detection, sx: float, sy: float) -> Detection:
    """Map a detection from work-image pixels back into source pixels."""
    return Detection(
        x1=int(box.x1 * sx),
        y1=int(box.y1 * sy),
        x2=int(box.x2 * sx),
        y2=int(box.y2 * sy),
        confidence=box.confidence,
    )


def _to_work(box: Detection, sx: float, sy: float) -> Detection:
    """Map a source-pixel detection down into the work image's pixels."""
    return Detection(
        x1=int(box.x1 / sx),
        y1=int(box.y1 / sy),
        x2=int(box.x2 / sx),
        y2=int(box.y2 / sy),
        confidence=box.confidence,
    )


def _scale_points(
    points: list[tuple[int, int]], sx: float, sy: float
) -> list[tuple[int, int]]:
    """Rescale every landmark in a point group back into source pixels."""
    return [(int(x * sx), int(y * sy)) for x, y in points]


def _scale_landmarks(fitted: FacialLandmarks, sx: float, sy: float) -> FacialLandmarks:
    """Rescale a fitted landmark set back into source-pixel coordinates."""
    return FacialLandmarks(
        right_eyebrow=_scale_points(fitted.right_eyebrow, sx, sy),
        left_eyebrow=_scale_points(fitted.left_eyebrow, sx, sy),
        nose=_scale_points(fitted.nose, sx, sy),
        right_eye=_scale_points(fitted.right_eye, sx, sy),
        left_eye=_scale_points(fitted.left_eye, sx, sy),
        mouth=_scale_points(fitted.mouth, sx, sy),
        jaw=(None if fitted.jaw is None else _scale_points(fitted.jaw, sx, sy)),
        points=(
            None if fitted.points is None else _scale_points(fitted.points, sx, sy)
        ),
        triangles=fitted.triangles,
    )


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


@lru_cache
def get_face_attribute_reader() -> FaceAttributeReader:
    """Return the process-wide attribute reader, built from settings on first use.

    Returns:
        A memoized :class:`FaceAttributeReader`; each of its two nets loads
        lazily on first inference, so an installation that never asks for
        attributes never loads either model.
    """
    settings = get_settings()
    return FaceAttributeReader(
        gender_model_path=settings.face_gender_model_path,
        gender_proto_path=settings.face_gender_proto_path,
        expression_model_path=settings.face_expression_model_path,
        min_confidence=settings.face_attribute_min_confidence,
    )


def _eye_centers(fitted: FacialLandmarks) -> EyeCenters:
    """Average each eye's points into the two centers alignment needs.

    Returned in the subject's-right-then-left order the ArcFace template in
    :mod:`face.attributes` expects, which is left-to-right across the image.
    """

    def _center(points: list[tuple[int, int]]) -> tuple[float, float]:
        xs = [x for x, _ in points]
        ys = [y for _, y in points]
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    return _center(fitted.right_eye), _center(fitted.left_eye)


def _to_attributes_model(attributes: FaceAttributes) -> FaceAttributesModel:
    """Convert an inferred attribute record into its response shape."""
    return FaceAttributesModel(
        expression=attributes.expression,
        expression_confidence=attributes.expression_confidence,
        apparent_gender=attributes.apparent_gender,
        apparent_gender_confidence=attributes.apparent_gender_confidence,
    )


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
    data: bytes,
    content_type: str,
    landmarks: bool = False,
    mesh: bool = False,
    fast: bool = False,
    attributes: bool = False,
) -> FaceResponse:
    """Detect human faces in an uploaded image.

    Reports geometry and, when asked, a per-face expression and apparent
    gender inferred and discarded on the spot; nothing here identifies a
    person or persists a frame.

    Args:
        data: Raw bytes of the uploaded file.
        content_type: Content type declared by the client.
        landmarks: Whether to fit feature points inside each detected face.
        mesh: Whether to also report the jaw contour, the flat 68-point array
            and its triangulation. Implies fitting, so it works on its own.
        fast: Downscale the frame server-side before inference and rescale the
            results back, trading a little precision for a much higher frame
            rate. Response coordinates stay in source-frame pixels either way.
        attributes: Whether to infer each face's expression and apparent
            gender. Implies fitting, since the expression crop is aligned to
            the eye centers the landmarker reports.

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

    work, sx, sy = _prepare(image, settings.face_fast_max_size if fast else None)
    raw_detections = get_face_detector().detect(work)
    detections = [_to_source(box, sx, sy) for box in raw_detections]

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

    # Mesh and attributes both imply fitting: the triangulation is built from
    # the fitted points, and the expression crop is aligned to the fitted eye
    # centers, so asking for either alone is enough.
    if not (landmarks or mesh or attributes):
        return FaceResponse(count=len(boxes), faces=[Face(box=box) for box in boxes])

    # The fit runs on the same (possibly downscaled) frame detection saw, so
    # its boxes must be mapped down into that frame's pixels first.
    work_boxes = [_to_work(box, sx, sy) for box in detections]
    fitted = get_face_landmarker().fit(work, work_boxes, mesh=mesh)

    # A fit that does not converge returns nothing rather than a partial list,
    # so everything derived from it is padded back to one entry per box instead
    # of zipping the boxes away. Landmark groups are reported only when the
    # geometry itself was asked for; attributes reuse the fit purely for eyes.
    pad = len(boxes) - len(fitted)
    if landmarks or mesh:
        groups: list[FacialLandmarksModel | None] = [
            _to_model(_scale_landmarks(face, sx, sy)) for face in fitted
        ] + [None] * pad
    else:
        groups = [None] * len(boxes)

    if attributes:
        # Eye centers come from the fit in work-image pixels, so gender and
        # expression run on that same frame and its boxes — never mixing a
        # source-pixel box with a work-pixel crop.
        eye_centers: list[EyeCenters | None] = [
            _eye_centers(face) for face in fitted
        ] + [None] * pad
        inferred = get_face_attribute_reader().read(work, work_boxes, eye_centers)
        attrs: list[FaceAttributesModel | None] = [
            _to_attributes_model(record) for record in inferred
        ]
    else:
        attrs = [None] * len(boxes)

    faces = [
        Face(box=box, landmarks=group, attributes=attr)
        for box, group, attr in zip(boxes, groups, attrs, strict=True)
    ]
    return FaceResponse(count=len(faces), faces=faces)
