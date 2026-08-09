"""Tests for upload image validation and decoding."""

import io

import numpy as np
import pytest
from PIL import Image

from app.utils.image import (
    ImageTooLargeError,
    InvalidImageDataError,
    UnsupportedImageTypeError,
    load_image,
)

ALLOWED = ("image/jpeg", "image/png")


def _png_bytes(width: int = 8, height: int = 4) -> bytes:
    """Return a tiny in-memory PNG suitable for upload tests."""
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color=(255, 0, 0)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_load_image_decodes_png_to_bgr_array() -> None:
    """A valid PNG decodes to an (H, W, 3) BGR array."""
    image = load_image(
        _png_bytes(), "image/png", max_bytes=10_000, allowed_types=ALLOWED
    )

    assert isinstance(image, np.ndarray)
    assert image.shape == (4, 8, 3)


def test_load_image_rejects_unsupported_content_type() -> None:
    """A content type outside the allowlist is rejected before decoding."""
    with pytest.raises(UnsupportedImageTypeError):
        load_image(
            _png_bytes(), "application/pdf", max_bytes=10_000, allowed_types=ALLOWED
        )


def test_load_image_rejects_payload_over_max_bytes() -> None:
    """A payload larger than the configured cap is rejected."""
    data = _png_bytes()

    with pytest.raises(ImageTooLargeError):
        load_image(data, "image/png", max_bytes=len(data) - 1, allowed_types=ALLOWED)


def test_load_image_rejects_undecodable_bytes() -> None:
    """Bytes that do not decode to an image are rejected."""
    with pytest.raises(InvalidImageDataError):
        load_image(
            b"definitely not an image",
            "image/png",
            max_bytes=10_000,
            allowed_types=ALLOWED,
        )
