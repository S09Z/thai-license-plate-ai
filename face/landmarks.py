"""68-point facial landmark fitting on top of already-detected face boxes.

Geometry only: this reports where a face's features sit within a frame. It does
not identify anyone, build a template, match against a gallery, or persist
anything. See :mod:`face` for the scope boundary this shares with the detector.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from detector.detector import Detection

logger = logging.getLogger(__name__)

# iBUG-68 index ranges. "right" and "left" are the *subject's*, so the right-hand
# groups sit on the left of the image — getting this backwards is the easiest
# mistake here and nothing downstream would catch it.
_RIGHT_EYEBROW = slice(17, 22)
_LEFT_EYEBROW = slice(22, 27)
_NOSE = slice(27, 36)
_RIGHT_EYE = slice(36, 42)
_LEFT_EYE = slice(42, 48)
_MOUTH = slice(48, 68)


@dataclass(frozen=True)
class FacialLandmarks:
    """The four requested features of one face, in pixel coordinates.

    Points 0-16 of the iBUG-68 set trace the jaw contour and are dropped: they
    are not one of the requested features, and carrying them would mean
    reporting a fuller face description than the request asked for.

    Attributes:
        right_eyebrow: Five points along the subject's right eyebrow.
        left_eyebrow: Five points along the subject's left eyebrow.
        nose: Nine points down the bridge and across the base.
        right_eye: Six points around the subject's right eye.
        left_eye: Six points around the subject's left eye.
        mouth: Twenty points around the outer and inner lips.
    """

    right_eyebrow: list[tuple[int, int]]
    left_eyebrow: list[tuple[int, int]]
    nose: list[tuple[int, int]]
    right_eye: list[tuple[int, int]]
    left_eye: list[tuple[int, int]]
    mouth: list[tuple[int, int]]


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

    import cv2

    facemark = cv2.face.createFacemarkLBF()
    facemark.loadModel(model_path)
    return facemark


def _to_points(rows: np.ndarray) -> list[tuple[int, int]]:
    """Convert a block of float landmark rows to integer pixel pairs."""
    return [(int(x), int(y)) for x, y in rows]


class FaceLandmarker:
    """Fits 68-point landmarks inside face boxes using OpenCV's LBF model."""

    def __init__(self, model_path: str) -> None:
        """Configure the landmarker without loading its model.

        Args:
            model_path: Filesystem path to ``lbfmodel.yaml``.
        """
        self._model_path = model_path
        self._model: Any | None = None

    def fit(self, image: np.ndarray, boxes: list[Detection]) -> list[FacialLandmarks]:
        """Locate facial features inside each detected face box.

        Args:
            image: BGR frame the boxes were detected in.
            boxes: Face boxes in corner coordinates, from the face detector.

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

        return [self._group(np.asarray(face).reshape(-1, 2)) for face in fitted]

    @staticmethod
    def _group(points: np.ndarray) -> FacialLandmarks:
        """Slice one 68-point array into the named feature groups."""
        return FacialLandmarks(
            right_eyebrow=_to_points(points[_RIGHT_EYEBROW]),
            left_eyebrow=_to_points(points[_LEFT_EYEBROW]),
            nose=_to_points(points[_NOSE]),
            right_eye=_to_points(points[_RIGHT_EYE]),
            left_eye=_to_points(points[_LEFT_EYE]),
            mouth=_to_points(points[_MOUTH]),
        )
