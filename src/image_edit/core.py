"""Core compositing: load, align, paste back.

Everything here is deterministic NumPy/OpenCV. The single equation the
whole package stands on:

    out = original * (1 - alpha) + edit * alpha

where alpha is the feathered mask. Where alpha == 0 the output IS the
original, arithmetically -- that is the guarantee.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def load_image(path: str) -> np.ndarray:
    """Load an image as a BGR uint8 array (OpenCV convention)."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return img


def save_png(image: np.ndarray, path: str) -> None:
    """Save losslessly as PNG.

    The byte-identity guarantee only survives lossless encoding: a JPEG
    round-trip recompresses the whole frame and dirties pixels far
    outside any mask (measured: ~35k changed pixels on a 2000x1600
    artwork from JPEG quality-92 alone). Masters are PNG, always.
    """
    if not path.lower().endswith(".png"):
        raise ValueError("surgical masters must be saved as .png (lossless)")
    if not cv2.imwrite(path, image):
        raise IOError(f"could not write: {path}")


def align_edit(edit_path: str, target_shape: tuple[int, int]) -> np.ndarray:
    """Load a model edit and resize it to the original's exact (H, W).

    Generative edit endpoints return their own canvas size; the paste-back
    arithmetic is pixel-to-pixel, so the edit must be aligned first.
    Resampling only ever touches the edit, never the original.
    """
    h, w = target_shape[:2]
    pil = Image.open(edit_path).convert("RGB")
    resized = pil.resize((w, h), Image.LANCZOS)
    return cv2.cvtColor(np.asarray(resized), cv2.COLOR_RGB2BGR)


def feather_alpha(mask: np.ndarray, sigma: float = 5.0) -> np.ndarray:
    """Turn a binary mask (0/255) into a soft alpha in [0, 1]."""
    return cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (0, 0), sigma)


def paste_back(
    original: np.ndarray,
    edit: np.ndarray,
    mask: np.ndarray,
    feather_sigma: float = 5.0,
) -> np.ndarray:
    """Composite `edit` onto `original` through the feathered mask.

    Pixels where the feathered alpha is 0 are copied from the original
    bit-for-bit; the feather only softens the seam.
    """
    if original.shape != edit.shape:
        raise ValueError(f"shape mismatch: original {original.shape} vs edit {edit.shape}")
    alpha = feather_alpha(mask, feather_sigma)
    out = original * (1.0 - alpha[..., None]) + edit * alpha[..., None]
    return np.round(out).astype(np.uint8)
