"""Validation and decoding of uploaded image payloads."""

import cv2
import numpy as np


class ImageValidationError(Exception):
    """Base error raised when an uploaded image fails validation."""


class UnsupportedImageTypeError(ImageValidationError):
    """Raised when the declared content type is not in the allowlist."""


class ImageTooLargeError(ImageValidationError):
    """Raised when the payload exceeds the configured size cap."""


class InvalidImageDataError(ImageValidationError):
    """Raised when the payload does not decode to an image."""


def load_image(
    data: bytes,
    content_type: str,
    *,
    max_bytes: int,
    allowed_types: tuple[str, ...],
) -> np.ndarray:
    """Validate an uploaded payload and decode it into an image array.

    Checks are ordered cheapest-first: the declared content type, then the
    payload size, then an actual decode (a declared type is never trusted on
    its own).

    Args:
        data: Raw bytes of the uploaded file.
        content_type: Content type declared by the client.
        max_bytes: Maximum accepted payload size, in bytes.
        allowed_types: Content types accepted by the endpoint.

    Returns:
        The decoded image as an ``(H, W, 3)`` BGR array.

    Raises:
        UnsupportedImageTypeError: If ``content_type`` is not allowed.
        ImageTooLargeError: If ``data`` exceeds ``max_bytes``.
        InvalidImageDataError: If ``data`` does not decode to an image.
    """
    if content_type not in allowed_types:
        raise UnsupportedImageTypeError(f"Unsupported content type: {content_type}")

    if len(data) > max_bytes:
        raise ImageTooLargeError(f"Upload exceeds {max_bytes} bytes")

    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise InvalidImageDataError("Payload does not decode to an image")

    return image
