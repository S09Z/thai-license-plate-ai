"""Tests for plate perspective correction."""

import cv2
import numpy as np
import pytest

from detector.detector import Detection
from detector.pipelines.perspective import (
    correct_perspective,
    find_plate_quad,
    order_corners,
)

# A tilted plate quadrilateral, in TL, TR, BR, BL order.
PLATE_CORNERS = np.array(
    [[40.0, 30.0], [250.0, 55.0], [240.0, 150.0], [35.0, 130.0]], dtype=np.float32
)
SCENE_SIZE = (180, 300)  # (height, width)


def _plate_scene() -> np.ndarray:
    """Return a dark scene containing one bright, tilted plate quadrilateral."""
    scene = np.full((*SCENE_SIZE, 3), 20, dtype=np.uint8)
    cv2.fillConvexPoly(scene, PLATE_CORNERS.astype(np.int32), (235, 235, 235))
    return scene


def _plate_box() -> Detection:
    """Return a detection box loosely enclosing the plate in `_plate_scene`."""
    return Detection(x1=33, y1=27, x2=253, y2=153, confidence=0.9)


def test_order_corners_sorts_shuffled_points() -> None:
    """Points in arbitrary order are returned as TL, TR, BR, BL."""
    shuffled = PLATE_CORNERS[[2, 0, 3, 1]]

    assert np.allclose(order_corners(shuffled), PLATE_CORNERS)


def test_order_corners_is_idempotent() -> None:
    """Ordering already-ordered corners leaves them unchanged."""
    assert np.allclose(order_corners(PLATE_CORNERS), PLATE_CORNERS)


def test_find_plate_quad_locates_tilted_plate() -> None:
    """The plate quadrilateral is recovered close to its true corners."""
    quad = find_plate_quad(_plate_scene())

    assert quad is not None
    assert np.allclose(quad, PLATE_CORNERS, atol=6.0)


def test_find_plate_quad_returns_none_without_a_quad() -> None:
    """A featureless image yields no quadrilateral rather than a bogus one."""
    blank = np.full((*SCENE_SIZE, 3), 90, dtype=np.uint8)

    assert find_plate_quad(blank) is None


def test_correct_perspective_rectifies_plate_to_output_size() -> None:
    """The tilted plate is warped to fill the requested output rectangle."""
    output = correct_perspective(_plate_scene(), _plate_box(), output_size=(256, 128))

    assert output.shape == (128, 256, 3)
    # A true rectification maps the plate onto the whole output, so every
    # corner of the result is plate-coloured rather than background.
    corners = [output[0, 0], output[0, -1], output[-1, 0], output[-1, -1]]
    assert all(int(pixel.min()) > 150 for pixel in corners)


def test_correct_perspective_falls_back_to_resized_crop() -> None:
    """With no quadrilateral found, the padded box crop is returned resized."""
    blank = np.full((*SCENE_SIZE, 3), 90, dtype=np.uint8)

    output = correct_perspective(blank, _plate_box(), output_size=(256, 128))

    assert output.shape == (128, 256, 3)
    assert int(output.min()) == 90


def test_correct_perspective_clamps_box_to_image_bounds() -> None:
    """Padding that runs past the image edge is clamped, not an error."""
    scene = _plate_scene()
    edge_box = Detection(x1=0, y1=0, x2=SCENE_SIZE[1], y2=SCENE_SIZE[0], confidence=0.5)

    output = correct_perspective(scene, edge_box, output_size=(64, 32))

    assert output.shape == (32, 64, 3)


def test_correct_perspective_rejects_degenerate_box() -> None:
    """A zero-area box is a caller error, not a silent empty crop."""
    with pytest.raises(ValueError):
        correct_perspective(
            _plate_scene(), Detection(x1=50, y1=50, x2=50, y2=90, confidence=0.5)
        )
