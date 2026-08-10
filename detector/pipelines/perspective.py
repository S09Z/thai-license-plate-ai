"""Perspective correction for detected plate regions.

A detection box is axis-aligned, so a plate photographed at an angle stays
skewed inside it. This module recovers the plate's quadrilateral and warps it
onto a flat rectangle, which is what the OCR stage expects.
"""

import logging

import cv2
import numpy as np

from detector.detector import Detection

logger = logging.getLogger(__name__)

# Fraction of the crop area a candidate contour must cover to be a plate.
_MIN_QUAD_AREA_RATIO = 0.2
# Contours to consider, largest first, before giving up.
_MAX_CANDIDATES = 5
# approxPolyDP tolerance, as a fraction of contour perimeter.
_APPROX_EPSILON_RATIO = 0.02


def order_corners(points: np.ndarray) -> np.ndarray:
    """Order four corner points as top-left, top-right, bottom-right, bottom-left.

    Uses coordinate sums and differences, which is orientation-stable for the
    moderate tilts plates are photographed at.

    Args:
        points: A ``(4, 2)`` array of ``(x, y)`` corners in any order.

    Returns:
        A ``(4, 2)`` float32 array ordered TL, TR, BR, BL.
    """
    ordered = np.zeros((4, 2), dtype=np.float32)
    coordinate_sum = points.sum(axis=1)
    coordinate_diff = np.diff(points, axis=1).ravel()

    ordered[0] = points[np.argmin(coordinate_sum)]
    ordered[2] = points[np.argmax(coordinate_sum)]
    ordered[1] = points[np.argmin(coordinate_diff)]
    ordered[3] = points[np.argmax(coordinate_diff)]
    return ordered


def find_plate_quad(crop: np.ndarray) -> np.ndarray | None:
    """Locate the plate's quadrilateral outline within a crop.

    Args:
        crop: Image region expected to contain a plate, as a BGR array.

    Returns:
        The four corners ordered TL, TR, BR, BL, or ``None`` when no
        sufficiently large four-sided contour is present.
    """
    grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grey, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    minimum_area = _MIN_QUAD_AREA_RATIO * crop.shape[0] * crop.shape[1]
    largest = sorted(contours, key=cv2.contourArea, reverse=True)[:_MAX_CANDIDATES]

    for contour in largest:
        perimeter = cv2.arcLength(contour, True)
        approximation = cv2.approxPolyDP(
            contour, _APPROX_EPSILON_RATIO * perimeter, True
        )
        if len(approximation) == 4 and cv2.contourArea(approximation) >= minimum_area:
            return order_corners(approximation.reshape(4, 2).astype(np.float32))

    return None


def warp_plate(
    image: np.ndarray, quad: np.ndarray, output_size: tuple[int, int]
) -> np.ndarray:
    """Warp a quadrilateral region onto a flat rectangle.

    Args:
        image: Source image containing the quadrilateral, as a BGR array.
        quad: Four corners ordered TL, TR, BR, BL.
        output_size: Target ``(width, height)`` in pixels.

    Returns:
        The rectified region at exactly ``output_size``.
    """
    width, height = output_size
    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(quad, destination)
    return cv2.warpPerspective(image, transform, (width, height))


def correct_perspective(
    image: np.ndarray,
    box: Detection,
    *,
    output_size: tuple[int, int] = (256, 128),
    padding: float = 0.04,
) -> np.ndarray:
    """Turn a detection box into a deskewed, fixed-size plate crop.

    The box is padded slightly (plate edges often sit right on the boundary)
    and clamped to the image. When the plate's quadrilateral cannot be found,
    the padded crop is returned resized rather than failing, so a skewed crop
    still reaches OCR.

    Args:
        image: Full source image, as an ``(H, W, 3)`` BGR array.
        box: Detection box locating the plate.
        output_size: Target ``(width, height)`` of the rectified crop.
        padding: Fraction of box size to expand by on each side.

    Returns:
        The rectified plate crop at exactly ``output_size``.

    Raises:
        ValueError: If ``box`` has zero width or height after clamping.
    """
    image_height, image_width = image.shape[:2]
    pad_x = int(round((box.x2 - box.x1) * padding))
    pad_y = int(round((box.y2 - box.y1) * padding))

    x1 = max(0, box.x1 - pad_x)
    y1 = max(0, box.y1 - pad_y)
    x2 = min(image_width, box.x2 + pad_x)
    y2 = min(image_height, box.y2 + pad_y)

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Detection box has no area within the image: {box}")

    crop = image[y1:y2, x1:x2]

    quad = find_plate_quad(crop)
    if quad is None:
        logger.debug("no plate quadrilateral found; falling back to resized crop")
        return cv2.resize(crop, output_size, interpolation=cv2.INTER_LINEAR)

    return warp_plate(crop, quad, output_size)
