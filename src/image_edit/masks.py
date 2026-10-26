"""Mask construction.

A mask is a single-channel uint8 image: 255 where the edit may land,
0 everywhere else. Coordinates for boxes are normalised 0-1000
(x0, y0, x1, y1), matching visual-grounding output, so a located object
maps straight onto any canvas size.

Three builders, in increasing sophistication:

- mask_from_boxes: rectangles. Only as good as the boxes.
- mask_from_dark_region: inside a box, keep the dark pixels. For a
  black garment on a light figure this traces the garment's real
  outline instead of pasting a rectangle.
- mask_from_diff: inside region boxes, keep the pixels that ACTUALLY
  changed between original and edit. Self-aligning -- the tool for
  move operations (remove from A, appear at B, limbs re-pose between),
  where no single colour or box describes the change.
"""

from __future__ import annotations

import numpy as np
import cv2

Box = tuple[float, float, float, float]  # normalised 0-1000: x0, y0, x1, y1


def _box_pixels(shape: tuple[int, int], box: Box, pad: float = 0.0) -> tuple[int, int, int, int]:
    h, w = shape[:2]
    x0, y0, x1, y1 = box
    px0 = max(0, int((x0 - pad) / 1000 * w))
    py0 = max(0, int((y0 - pad) / 1000 * h))
    px1 = min(w, int((x1 + pad) / 1000 * w))
    py1 = min(h, int((y1 + pad) / 1000 * h))
    return px0, py0, px1, py1


def mask_from_boxes(shape: tuple[int, int], boxes: list[Box]) -> np.ndarray:
    """Union of rectangular regions."""
    mask = np.zeros(shape[:2], np.uint8)
    for box in boxes:
        x0, y0, x1, y1 = _box_pixels(shape, box)
        mask[y0:y1, x0:x1] = 255
    return mask


def _clean(mask: np.ndarray, close_size: int = 7, open_size: int = 5) -> np.ndarray:
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((close_size, close_size), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((open_size, open_size), np.uint8))
    return mask


def mask_from_dark_region(
    image: np.ndarray,
    box: Box,
    luminance_threshold: int = 95,
    pad: float = 5.0,
) -> np.ndarray:
    """Dark pixels inside `box`, morphologically cleaned.

    Traces the outline of a dark object (e.g. a black coat) rather than
    the box around it.
    """
    x0, y0, x1, y1 = _box_pixels(image.shape, box, pad)
    roi = np.zeros(image.shape[:2], np.uint8)
    roi[y0:y1, x0:x1] = 255
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dark = (gray < luminance_threshold).astype(np.uint8) * 255
    return _clean(cv2.bitwise_and(dark, roi))


def mask_from_diff(
    original: np.ndarray,
    edit: np.ndarray,
    boxes: list[Box],
    diff_threshold: int = 25,
    close_size: int = 11,
    dilate_size: int = 13,
    region_margin: int = 15,
) -> np.ndarray:
    """Pixels that actually changed, restricted to `boxes` (+ margin).

    The diff threshold rejects the model's low-level re-render noise on
    near-identical background; closing + dilation add a safety margin
    around every changed pixel; the region cap means drift elsewhere in
    the model's output can never leak into the composite.
    """
    regions = mask_from_boxes(original.shape, boxes)
    diff = np.abs(original.astype(int) - edit.astype(int)).max(axis=2)
    changed = (diff > diff_threshold).astype(np.uint8) * 255
    mask = cv2.bitwise_and(changed, regions)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((close_size, close_size), np.uint8))
    mask = cv2.dilate(mask, np.ones((dilate_size, dilate_size), np.uint8))
    allowed = cv2.dilate(regions, np.ones((region_margin, region_margin), np.uint8))
    return cv2.bitwise_and(mask, allowed)


def guard_mask(mask: np.ndarray, dilation: int = 41) -> np.ndarray:
    """The mask dilated well beyond any feather spread.

    Verification is judged OUTSIDE the guard: if a pixel changed out
    there, no feather or blur can be blamed -- the guarantee failed.
    """
    return cv2.dilate(mask, np.ones((dilation, dilation), np.uint8))
