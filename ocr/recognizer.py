"""PaddleOCR-based text recognition for plate crops.

PaddleOCR and its paddle backend stay confined to this module; callers work
with plain :class:`~ocr.reading.TextLine` values.
"""

import logging
from typing import Any

import numpy as np

from ocr.reading import TextLine

logger = logging.getLogger(__name__)


def _load_paddleocr(lang: str) -> Any:
    """Build a PaddleOCR engine for a language.

    Imported lazily so that merely importing this module does not pull in the
    paddle backend. Patched out in unit tests.

    Document orientation, unwarping and textline-orientation stages are
    disabled: crops reaching this point are already detected and rectified by
    Phases 1-2. Measured on a synthetic Thai plate, dropping them roughly
    halved latency (915 ms -> 468 ms) and *improved* accuracy, since the
    orientation classifier corrupted upscaled crops.

    No ``*_model_name`` is passed on purpose: supplying one makes PaddleOCR
    ignore ``lang`` and silently fall back to a non-Thai recognizer.

    Args:
        lang: PaddleOCR language code (``"th"`` for Thai).

    Returns:
        The constructed PaddleOCR pipeline. Weights are downloaded to
        ``~/.paddlex/official_models`` on first construction.
    """
    from paddleocr import PaddleOCR

    return PaddleOCR(
        lang=lang,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )


class PlateOCR:
    """Recognizes text lines in a rectified plate crop."""

    def __init__(self, lang: str, min_confidence: float) -> None:
        """Configure the recognizer without loading the engine.

        Args:
            lang: PaddleOCR language code.
            min_confidence: Lines scoring below this are discarded.
        """
        self._lang = lang
        self._min_confidence = min_confidence
        self._engine: Any | None = None

    def recognize(self, crop: np.ndarray) -> list[TextLine]:
        """Recognize text lines in a plate crop.

        The engine is built on first call, so its model download happens only
        when recognition is actually attempted.

        Args:
            crop: Rectified plate image, as an ``(H, W, 3)`` BGR array.

        Returns:
            Recognized lines above the confidence floor, ordered top to bottom.
        """
        if self._engine is None:
            logger.info("loading OCR engine", extra={"lang": self._lang})
            self._engine = _load_paddleocr(self._lang)

        lines: list[TextLine] = []
        for result in self._engine.predict(crop):
            texts = result["rec_texts"] if "rec_texts" in result else []
            scores = result["rec_scores"] if "rec_scores" in result else []
            polygons = result["rec_polys"] if "rec_polys" in result else []

            for text, score, polygon in zip(texts, scores, polygons, strict=True):
                if float(score) < self._min_confidence:
                    continue
                lines.append(
                    TextLine(
                        text=text,
                        confidence=float(score),
                        top=float(min(point[1] for point in polygon)),
                        bottom=float(max(point[1] for point in polygon)),
                        left=float(min(point[0] for point in polygon)),
                    )
                )

        lines.sort(key=lambda line: line.top)
        return lines
