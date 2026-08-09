"""YOLO-based license plate detection.

Ultralytics, torch and OpenCV stay confined to this module; callers work with
plain :class:`Detection` values.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detection:
    """A single detected plate region in pixel coordinates.

    Attributes:
        x1: Left edge of the box.
        y1: Top edge of the box.
        x2: Right edge of the box.
        y2: Bottom edge of the box.
        confidence: Model confidence for the detection, in ``[0, 1]``.
    """

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float


def _load_yolo(model_path: str) -> Any:
    """Load Ultralytics YOLO weights from disk.

    Imported lazily so that merely importing this module does not pull in
    torch. Patched out in unit tests.

    Args:
        model_path: Filesystem path to the ``.pt`` weights.

    Returns:
        The loaded Ultralytics model.

    Raises:
        FileNotFoundError: If ``model_path`` does not exist.
    """
    if not Path(model_path).is_file():
        raise FileNotFoundError(f"Detector weights not found: {model_path}")

    from ultralytics import YOLO

    return YOLO(model_path)


class PlateDetector:
    """Detects license plates in an image using a YOLO model."""

    def __init__(self, model_path: str, conf_threshold: float) -> None:
        """Configure the detector without loading weights.

        Args:
            model_path: Filesystem path to the ``.pt`` weights.
            conf_threshold: Minimum confidence for a detection to be kept.
        """
        self._model_path = model_path
        self._conf_threshold = conf_threshold
        self._model: Any | None = None

    def detect(self, image: np.ndarray) -> list[Detection]:
        """Detect plate regions in an image.

        Weights are loaded on first call, so an absent model file surfaces
        only when detection is actually attempted.

        Args:
            image: Image to run detection on, as an ``(H, W, 3)`` BGR array.

        Returns:
            The detected plate boxes, empty if the model found none.

        Raises:
            FileNotFoundError: If the configured weights file does not exist.
        """
        if self._model is None:
            logger.info("loading detector weights", extra={"path": self._model_path})
            self._model = _load_yolo(self._model_path)

        results = self._model(image, conf=self._conf_threshold, verbose=False)

        detections: list[Detection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for (x1, y1, x2, y2), confidence in zip(
                boxes.xyxy.tolist(), boxes.conf.tolist(), strict=True
            ):
                detections.append(
                    Detection(
                        x1=int(x1),
                        y1=int(y1),
                        x2=int(x2),
                        y2=int(y2),
                        confidence=float(confidence),
                    )
                )
        return detections
