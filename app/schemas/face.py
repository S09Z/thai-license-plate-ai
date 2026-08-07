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


class Face(BaseModel):
    """One detected face and, when requested, its feature points.

    Attributes:
        box: Where the face sits in the frame.
        landmarks: Feature points, or ``None`` when they were not requested.
    """

    box: BoundingBox
    landmarks: FacialLandmarksModel | None = None


class FaceResponse(BaseModel):
    """Response body for ``POST /detect/faces``.

    Attributes:
        count: Number of faces detected.
        faces: The detected faces.
    """

    count: int
    faces: list[Face]
