"""YuNet-based face detection.

OpenCV's ONNX face detector stays confined to this module; callers work with
the same plain :class:`~detector.detector.Detection` values the plate
detector returns.

Scope: this locates faces, it does not recognize them. YuNet also emits five
facial landmarks per face; they are discarded here on purpose.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np

from detector.detector import Detection

logger = logging.getLogger(__name__)

# YuNet returns one row per face: columns 0-3 are the box as x, y, w, h,
# columns 4-13 are five landmark x/y pairs, and column 14 is the score.
_SCORE_COLUMN = 14


def _load_yunet(model_path: str, conf_threshold: float) -> Any:
    """Load the YuNet ONNX face detector from disk.

    OpenCV is imported lazily so that merely importing this module stays
    cheap. Patched out in unit tests.

    Args:
        model_path: Filesystem path to the ``.onnx`` model.
        conf_threshold: Minimum score for a face to be reported.

    Returns:
        The configured ``cv2.FaceDetectorYN``.

    Raises:
        FileNotFoundError: If ``model_path`` does not exist.
    """
    if not Path(model_path).is_file():
        raise FileNotFoundError(f"Face detector model not found: {model_path}")

    import cv2

    return cv2.FaceDetectorYN.create(model_path, "", (0, 0), conf_threshold)


class FaceDetector:
    """Detects human faces in an image using the YuNet ONNX model."""

    def __init__(self, model_path: str, conf_threshold: float) -> None:
        """Configure the detector without loading the model.

        Args:
            model_path: Filesystem path to the ``.onnx`` model.
            conf_threshold: Minimum score for a face to be reported.
        """
        self._model_path = model_path
        self._conf_threshold = conf_threshold
        self._model: Any | None = None

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Detect face regions in an image.

        The model is loaded on first call, so an absent file surfaces only
        when detection is actually attempted.

        Args:
            image: Image to run detection on, as an ``(H, W, 3)`` BGR array.

        Returns:
            The detected face boxes, empty if the model found none.

        Raises:
            FileNotFoundError: If the configured model file does not exist.
        """
        if self._model is None:
            logger.info("loading face detector model", extra={"path": self._model_path})
            self._model = _load_yunet(self._model_path, self._conf_threshold)

        height, width = image.shape[:2]
        self._model.setInputSize((width, height))
        _retval, faces = self._model.detect(image)

        if faces is None:
            return []

        return [
            Detection(
                x1=int(row[0]),
                y1=int(row[1]),
                x2=int(row[0]) + int(row[2]),
                y2=int(row[1]) + int(row[3]),
                confidence=float(row[_SCORE_COLUMN]),
            )
            for row in faces
        ]
