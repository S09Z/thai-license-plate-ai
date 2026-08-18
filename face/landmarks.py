"""68-point facial landmark fitting on top of already-detected face boxes.

Geometry only: this reports where a face's features sit within a frame. It does
not identify anyone, build a template, match against a gallery, or persist
anything. See :mod:`face` for the scope boundary this shares with the detector.
"""

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from detector.detector import Detection

logger = logging.getLogger(__name__)

# iBUG-68 index ranges. "right" and "left" are the *subject's*, so the right-hand
# groups sit on the left of the image — getting this backwards is the easiest
# mistake here and nothing downstream would catch it.
_JAW = slice(0, 17)
_RIGHT_EYEBROW = slice(17, 22)
_LEFT_EYEBROW = slice(22, 27)
_NOSE = slice(27, 36)
_RIGHT_EYE = slice(36, 42)
_LEFT_EYE = slice(42, 48)
_MOUTH = slice(48, 68)


@dataclass(frozen=True)
class FacialLandmarks:
    """The features of one face, in pixel coordinates.

    Points 0-16 of the iBUG-68 set trace the jaw contour. They stay out of the
    default result: face shape is the most identity-bearing part of the 68, so
    reporting it is opt-in through ``mesh`` rather than something every caller
    receives. The mesh fields are all ``None`` unless it was asked for.

    Attributes:
        right_eyebrow: Five points along the subject's right eyebrow.
        left_eyebrow: Five points along the subject's left eyebrow.
        nose: Nine points down the bridge and across the base.
        right_eye: Six points around the subject's right eye.
        left_eye: Six points around the subject's left eye.
        mouth: Twenty points around the outer and inner lips.
        jaw: Seventeen points along the face boundary, mesh only.
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


def _load_facemark(model_path: str) -> Any:
    """Load the LBF landmark model, failing loudly when it is absent.

    OpenCV constructs a Facemark happily without a model and only fails inside
    ``fit()``, with a ``cv2.error`` that the route would surface as a 500. The
    explicit existence check restores the documented 503 contract.

    Args:
        model_path: Filesystem path to ``lbfmodel.yaml``.

    Returns:
        A loaded ``cv2.face.Facemark`` instance.

    Raises:
        FileNotFoundError: If no file exists at ``model_path``.
    """
    if not Path(model_path).is_file():
        raise FileNotFoundError(f"Face landmark model not found: {model_path}")

    facemark = cv2.face.createFacemarkLBF()
    facemark.loadModel(model_path)
    return facemark


def _to_points(rows: np.ndarray) -> list[tuple[int, int]]:
    """Convert a block of float landmark rows to integer pixel pairs."""
    return [(int(x), int(y)) for x, y in rows]


def _triangulate(points: list[tuple[int, int]]) -> list[tuple[int, int, int]]:
    """Delaunay-triangulate landmark points into index triples.

    Indices rather than coordinates: they are a quarter of the payload and let
    a client check the topology against the points it already has.

    The bounding rectangle is derived from the points themselves, not from the
    face box that seeded them. Fitted points routinely land outside that box —
    one of 68 did on the reference portrait — and ``Subdiv2D`` raises on any
    insert outside its rectangle.

    Args:
        points: Landmark pixel pairs, in the order clients will index them.

    Returns:
        Triangles as ``(i, j, k)`` triples, or an empty list when the points
        are collinear or coincident and no triangle exists.
    """
    if not points:
        return []

    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    # +3 rather than +2: Subdiv2D treats the rectangle as half-open, so a point
    # exactly on the far edge is outside it.
    subdiv = cv2.Subdiv2D(
        (
            min(xs) - 1,
            min(ys) - 1,
            max(xs) - min(xs) + 3,
            max(ys) - min(ys) + 3,
        )
    )

    index_of: dict[tuple[float, float], int] = {}
    for index, (x, y) in enumerate(points):
        vertex = (float(x), float(y))
        # First index wins: a duplicated coordinate is one vertex to Subdiv2D,
        # and picking one owner keeps the triangle list self-consistent.
        index_of.setdefault(vertex, index)
        subdiv.insert(vertex)

    # Two OpenCV quirks in one line. getTriangleList() returns a bare ``()``
    # rather than an empty array when nothing triangulates, so it is normalised
    # through asarray/reshape; and .tolist() is load-bearing rather than tidy,
    # because iterating the array directly yields numpy float32 scalars whose
    # hashing costs 7x more than native floats below (1.75 ms vs 0.23 ms).
    raw = np.asarray(subdiv.getTriangleList(), dtype=np.float64).reshape(-1, 6)

    triangles: list[tuple[int, int, int]] = []
    for x1, y1, x2, y2, x3, y3 in raw.tolist():
        corners = [(x1, y1), (x2, y2), (x3, y3)]
        # Triangles touching Subdiv2D's virtual super-rectangle have corners
        # that are not landmarks at all, so they index nothing.
        if any(corner not in index_of for corner in corners):
            continue
        triple = tuple(index_of[corner] for corner in corners)
        if len(set(triple)) == 3:
            triangles.append(triple)  # type: ignore[arg-type]

    return triangles


class FaceLandmarker:
    """Fits 68-point landmarks inside face boxes using OpenCV's LBF model."""

    def __init__(self, model_path: str) -> None:
        """Configure the landmarker without loading its model.

        Args:
            model_path: Filesystem path to ``lbfmodel.yaml``.
        """
        self._model_path = model_path
        self._model: Any | None = None

    def fit(
        self, image: np.ndarray, boxes: list[Detection], *, mesh: bool = False
    ) -> list[FacialLandmarks]:
        """Locate facial features inside each detected face box.

        Args:
            image: BGR frame the boxes were detected in.
            boxes: Face boxes in corner coordinates, from the face detector.
            mesh: Also report the jaw contour, the flat 68-point array and its
                Delaunay triangulation. Off by default; see
                :class:`FacialLandmarks`.

        Returns:
            One :class:`FacialLandmarks` per box, or an empty list when there
            are no boxes or the fit does not converge.

        Raises:
            FileNotFoundError: If the configured model file is missing.
        """
        if not boxes:
            return []

        if self._model is None:
            logger.info("loading face landmark model", extra={"path": self._model_path})
            self._model = _load_facemark(self._model_path)

        # OpenCV wants x, y, w, h, which is the convention the face detector
        # already converted away from; convert back at this boundary only.
        rects = np.array(
            [[box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1] for box in boxes],
            dtype=np.int32,
        )
        ok, fitted = self._model.fit(image, rects)

        if not ok or fitted is None:
            return []

        return [
            self._group(np.asarray(face).reshape(-1, 2), mesh=mesh) for face in fitted
        ]

    @staticmethod
    def _group(points: np.ndarray, *, mesh: bool = False) -> FacialLandmarks:
        """Slice one 68-point array into the named feature groups."""
        groups = FacialLandmarks(
            right_eyebrow=_to_points(points[_RIGHT_EYEBROW]),
            left_eyebrow=_to_points(points[_LEFT_EYEBROW]),
            nose=_to_points(points[_NOSE]),
            right_eye=_to_points(points[_RIGHT_EYE]),
            left_eye=_to_points(points[_LEFT_EYE]),
            mouth=_to_points(points[_MOUTH]),
        )
        if not mesh:
            return groups

        flat = _to_points(points)
        return replace(
            groups,
            jaw=_to_points(points[_JAW]),
            points=flat,
            triangles=_triangulate(flat),
        )
