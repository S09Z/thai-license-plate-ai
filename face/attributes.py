"""Apparent gender and facial expression inference for already-detected faces.

This is a deliberate widening of the face pipeline's scope. Phases 10-12
(detection, landmarks, mesh) report only where a face's features sit in a
frame; this module infers something about the person. That crossing is the
entire reason for the pattern below: **infer, render, discard**. Nothing here
is stored, no frame is persisted, and nothing produced by this module is ever
linked to a plate number — the two pipelines share a frame but not an output.

Naming is deliberately not the colloquial word for either field:

- ``expression``, not ``emotion``: a facial configuration does not reliably
  indicate an internal emotional state (Barrett et al., 2019). This reports
  what the face looks like, not what the person feels.
- ``apparent_gender``, not ``sex``: this is a binary classifier over visual
  presentation, trained on two labels. It is not a determination of sex, and
  there is no third output for cases it cannot place.

Both nets run through ``cv2.dnn``, already a dependency of the landmark and
mesh stages, so this adds no new package.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from detector.detector import Detection

logger = logging.getLogger(__name__)

EyeCenters = tuple[tuple[float, float], tuple[float, float]]

# Levi & Hassner (2015) gender net: 227x227 BGR crops, this fixed per-channel
# mean, no RGB swap. The prototxt's own final layer is already softmax, so
# forward() returns a probability distribution, not raw logits.
_GENDER_INPUT_SIZE = (227, 227)
_GENDER_MEAN = (78.4263377603, 87.7689143744, 114.895847746)
_GENDER_LABELS = ("male", "female")
_GENDER_CROP_PAD = 20

# OpenCV Zoo's MobileFaceNet expression model: 112x112, BGR->RGB, normalized
# to [-1, 1]. Softmax is applied here defensively — whether the exported ONNX
# graph already includes one is unconfirmed, but softmax is monotonic in its
# input, so the predicted label is unaffected either way; only a reported
# confidence would be distorted if the model already normalizes its output.
_EXPRESSION_INPUT_SIZE = (112, 112)
_EXPRESSION_LABELS = (
    "angry",
    "disgust",
    "fearful",
    "happy",
    "neutral",
    "sad",
    "surprised",
)

# The ArcFace 112x112 reference template's two eye positions. Only two of its
# five points, because this codebase's landmarker (face/landmarks.py) exposes
# eye centers but not the nose/mouth points the full template also uses.
# estimateAffinePartial2D solves a 4-DOF similarity transform (scale,
# rotation, translation), which two correspondences already determine.
_TEMPLATE_EYES = np.array([[38.2946, 51.6963], [73.5318, 51.5014]], dtype=np.float32)


@dataclass(frozen=True)
class FaceAttributes:
    """Inferred attributes for one face, each independently gated.

    Attributes:
        expression: One of the seven FER labels, or ``None`` if
            ``expression_confidence`` fell below the configured threshold.
        expression_confidence: The winning class's softmax score, reported
            even when abstained so the number is never hidden.
        apparent_gender: ``"male"`` or ``"female"``, or ``None`` if
            ``apparent_gender_confidence`` fell below the threshold.
        apparent_gender_confidence: The winning class's score, likewise
            always reported.
    """

    expression: str | None
    expression_confidence: float
    apparent_gender: str | None
    apparent_gender_confidence: float


def _load_gender_net(model_path: str, proto_path: str) -> Any:
    """Load the Caffe gender net, failing loudly when either file is absent.

    Args:
        model_path: Filesystem path to ``gender_net.caffemodel``.
        proto_path: Filesystem path to ``gender_deploy.prototxt``.

    Returns:
        A loaded ``cv2.dnn.Net``.

    Raises:
        FileNotFoundError: If the weights or the prototxt is missing.
    """
    if not Path(model_path).is_file():
        raise FileNotFoundError(f"Face gender model not found: {model_path}")
    if not Path(proto_path).is_file():
        raise FileNotFoundError(f"Face gender prototxt not found: {proto_path}")
    return cv2.dnn.readNetFromCaffe(proto_path, model_path)


def _load_expression_net(model_path: str) -> Any:
    """Load the ONNX expression net, failing loudly when it is absent.

    Args:
        model_path: Filesystem path to the expression recognition ONNX file.

    Returns:
        A loaded ``cv2.dnn.Net``.

    Raises:
        FileNotFoundError: If no file exists at ``model_path``.
    """
    if not Path(model_path).is_file():
        raise FileNotFoundError(f"Face expression model not found: {model_path}")
    return cv2.dnn.readNetFromONNX(model_path)


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Convert raw scores to a probability distribution, shift-stabilized."""
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    normalized: np.ndarray = exp / exp.sum()
    return normalized


def _crop_padded(image: np.ndarray, box: Detection, pad: int) -> np.ndarray:
    """Crop a face box with a pixel margin, clamped to the frame's bounds."""
    height, width = image.shape[:2]
    x1 = max(box.x1 - pad, 0)
    y1 = max(box.y1 - pad, 0)
    x2 = min(box.x2 + pad, width)
    y2 = min(box.y2 + pad, height)
    return image[y1:y2, x1:x2]


def _align_face(image: np.ndarray, eyes: EyeCenters) -> np.ndarray | None:
    """Warp a frame so its two eye centers land on the ArcFace template.

    Args:
        image: The full BGR frame the eye centers were measured in.
        eyes: Two eye centers, in the same order as ``_TEMPLATE_EYES``.

    Returns:
        A 112x112 aligned BGR crop, or ``None`` if the two points coincide
        and no similarity transform exists.
    """
    src = np.array(eyes, dtype=np.float32)
    matrix, _ = cv2.estimateAffinePartial2D(src, _TEMPLATE_EYES)
    if matrix is None:
        return None
    size = _EXPRESSION_INPUT_SIZE
    return cv2.warpAffine(image, matrix, size)


class FaceAttributeReader:
    """Infers apparent gender and expression for already-detected faces."""

    def __init__(
        self,
        gender_model_path: str,
        gender_proto_path: str,
        expression_model_path: str,
        min_confidence: float,
    ) -> None:
        """Configure the reader without loading either model.

        Args:
            gender_model_path: Filesystem path to the gender Caffe weights.
            gender_proto_path: Filesystem path to the gender Caffe prototxt.
            expression_model_path: Filesystem path to the expression ONNX
                model.
            min_confidence: Minimum winning-class score for a field to be
                reported; below it the field is ``None`` but its confidence
                is still returned.
        """
        self._gender_model_path = gender_model_path
        self._gender_proto_path = gender_proto_path
        self._expression_model_path = expression_model_path
        self._min_confidence = min_confidence
        self._gender_net: Any | None = None
        self._expression_net: Any | None = None

    def read(
        self,
        image: np.ndarray,
        boxes: list[Detection],
        eye_centers: list[EyeCenters | None],
    ) -> list[FaceAttributes]:
        """Infer attributes for each detected face.

        Args:
            image: BGR frame the boxes were detected in.
            boxes: Face boxes in corner coordinates.
            eye_centers: One entry per box: two eye centers to align the
                expression crop with, or ``None`` when the landmark fit did
                not converge for that face. Expression abstains on ``None``
                without affecting gender, which only needs the box.

        Returns:
            One :class:`FaceAttributes` per box, in the same order.

        Raises:
            FileNotFoundError: If a needed model file is missing.
        """
        if not boxes:
            return []

        if self._gender_net is None:
            logger.info(
                "loading face gender model", extra={"path": self._gender_model_path}
            )
            self._gender_net = _load_gender_net(
                self._gender_model_path, self._gender_proto_path
            )

        results = []
        for box, eyes in zip(boxes, eye_centers, strict=True):
            gender, gender_confidence = self._read_gender(image, box)
            if eyes is None:
                expression, expression_confidence = None, 0.0
            else:
                expression, expression_confidence = self._read_expression(image, eyes)
            results.append(
                FaceAttributes(
                    expression=expression,
                    expression_confidence=expression_confidence,
                    apparent_gender=gender,
                    apparent_gender_confidence=gender_confidence,
                )
            )
        return results

    def _read_gender(
        self, image: np.ndarray, box: Detection
    ) -> tuple[str | None, float]:
        # read() loads the net before ever calling this, so it is not None here;
        # the assert states that to the type checker, which cannot see across
        # the method boundary.
        assert self._gender_net is not None
        face = _crop_padded(image, box, _GENDER_CROP_PAD)
        blob = cv2.dnn.blobFromImage(
            face, 1.0, _GENDER_INPUT_SIZE, _GENDER_MEAN, swapRB=False
        )
        self._gender_net.setInput(blob)
        scores = self._gender_net.forward()[0]
        index = int(np.argmax(scores))
        confidence = float(scores[index])
        label = _GENDER_LABELS[index] if confidence >= self._min_confidence else None
        return label, confidence

    def _read_expression(
        self, image: np.ndarray, eyes: EyeCenters
    ) -> tuple[str | None, float]:
        aligned = _align_face(image, eyes)
        if aligned is None:
            return None, 0.0

        if self._expression_net is None:
            logger.info(
                "loading face expression model",
                extra={"path": self._expression_model_path},
            )
            self._expression_net = _load_expression_net(self._expression_model_path)

        # scalefactor=1/127.5, mean=127.5 on all three channels: (x - 127.5) /
        # 127.5, equivalent to x/127.5 - 1. swapRB=True gives BGR->RGB.
        blob = cv2.dnn.blobFromImage(
            aligned,
            1 / 127.5,
            _EXPRESSION_INPUT_SIZE,
            (127.5, 127.5, 127.5),
            swapRB=True,
        )
        self._expression_net.setInput(blob)
        probabilities = _softmax(self._expression_net.forward()[0])
        index = int(np.argmax(probabilities))
        confidence = float(probabilities[index])
        label = (
            _EXPRESSION_LABELS[index] if confidence >= self._min_confidence else None
        )
        return label, confidence
