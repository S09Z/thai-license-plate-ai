"""Schemas for the face detection endpoint.

Face boxes alone had the same shape and meaning as plate boxes, so Phase 10
reused ``DetectionResponse``. Landmarks end that: they are face-specific, and
bolting them onto the shared ``BoundingBox`` would leak face concepts into
``/detect`` and ``/recognize``.
"""

from pydantic import BaseModel

from app.schemas.detection import BoundingBox


class FacialLandmarksModel(BaseModel):
    """Feature points of one face, as ``[x, y]`` pixel pairs.

    ``right`` and ``left`` are the *subject's*, following the iBUG-68
    convention, so the right-hand groups appear on the left of the image.

    The last three are the whole-face mesh and are ``None`` unless ``mesh``
    was requested; the jaw traces the face boundary, which is opt-in.

    Attributes:
        right_eyebrow: Points along the subject's right eyebrow.
        left_eyebrow: Points along the subject's left eyebrow.
        nose: Points down the bridge and across the base.
        right_eye: Points around the subject's right eye.
        left_eye: Points around the subject's left eye.
        mouth: Points around the outer and inner lips.
        jaw: Points along the face boundary, mesh only.
        points: All 68 points in iBUG order, the array ``triangles`` indexes.
        triangles: Delaunay triangles as index triples into ``points``.
    """

    right_eyebrow: list[tuple[int, int]]
    left_eyebrow: list[tuple[int, int]]
    nose: list[tuple[int, int]]
    right_eye: list[tuple[int, int]]
    left_eye: list[tuple[int, int]]
    mouth: list[tuple[int, int]]
    jaw: list[tuple[int, int]] | None = None
    points: list[tuple[int, int]] | None = None
    triangles: list[tuple[int, int, int]] | None = None


class FaceAttributesModel(BaseModel):
    """Inferred apparent gender and expression for one face.

    Both fields are gated independently: a label falls back to ``None`` when
    its winning score is below the configured threshold, but the score itself
    is always reported so a caller can see how close a call it was. Named to
    describe the face, not the person: ``expression`` (not "emotion", which a
    face does not reliably reveal) and ``apparent_gender`` (a visual-presentation
    classifier, not a determination of sex).

    Attributes:
        expression: One of the seven FER labels, or ``None`` when abstained.
        expression_confidence: The winning expression score, always reported.
        apparent_gender: ``"male"`` or ``"female"``, or ``None`` when abstained.
        apparent_gender_confidence: The winning gender score, always reported.
    """

    expression: str | None = None
    expression_confidence: float
    apparent_gender: str | None = None
    apparent_gender_confidence: float


class Face(BaseModel):
    """One detected face and, when requested, its feature points.

    Attributes:
        box: Where the face sits in the frame.
        landmarks: Feature points, or ``None`` when they were not requested.
        attributes: Inferred expression and apparent gender, or ``None`` when
            ``attributes`` was not requested.
    """

    box: BoundingBox
    landmarks: FacialLandmarksModel | None = None
    attributes: FaceAttributesModel | None = None


class FaceResponse(BaseModel):
    """Response body for ``POST /detect/faces``.

    Attributes:
        count: Number of faces detected.
        faces: The detected faces.
    """

    count: int
    faces: list[Face]
