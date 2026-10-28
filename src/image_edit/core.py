"""Core compositing: load, align, paste back -- and stack.

Everything here is deterministic NumPy/OpenCV/PIL. The single equation
the paste-back half of the package stands on:

    out = original * (1 - alpha) + edit * alpha

where alpha is the feathered mask. Where alpha == 0 the output IS the
original, arithmetically -- that is the guarantee.

The stack half is the dual operation: instead of rescuing a flat image
from a model's re-render, build the image from its layers in the first
place -- base first, then each RGBA layer composited over it in order
(Porter-Duff "over"). Same arithmetic family, opposite direction:
nothing generative is involved, so there is nothing to verify against;
the layer files are the source of truth and the build is reproducible.
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


def load_layer(path: str) -> Image.Image:
    """Load a layer as RGBA. Any source format is converted; the alpha
    channel is the layer's shape -- where it is 0, layers below show
    through unchanged."""
    try:
        return Image.open(path).convert("RGBA")
    except FileNotFoundError:
        raise
    except Exception as exc:
        raise FileNotFoundError(f"could not read layer: {path}") from exc


def stack_layers(
    canvas_size: tuple[int, int],
    background: tuple[int, int, int, int],
    layers: list[tuple[Image.Image, int, int]],
) -> Image.Image:
    """Composite RGBA layers onto a fresh canvas, in order.

    `layers` is a list of (image, x, y): each image is alpha-composited
    over everything below it at offset (x, y), clipped at the canvas
    edges. Order is the whole semantics -- a later opaque layer hides
    earlier ones exactly where it covers them (that is how an occluder
    layer removes a line from under a subject without editing either).
    """
    canvas = Image.new("RGBA", canvas_size, background)
    for image, x, y in layers:
        # Clip the layer to the canvas rectangle before compositing;
        # offsets may be negative or push the layer partly off-canvas.
        src_x, src_y = max(0, -x), max(0, -y)
        dst_x, dst_y = max(0, x), max(0, y)
        w = min(image.width - src_x, canvas.width - dst_x)
        h = min(image.height - src_y, canvas.height - dst_y)
        if w <= 0 or h <= 0:
            continue
        region = image.crop((src_x, src_y, src_x + w, src_y + h))
        canvas.alpha_composite(region, (dst_x, dst_y))
    return canvas


def save_png_pil(image: Image.Image, path: str) -> None:
    """Save a PIL image losslessly as PNG (same masters rule as save_png)."""
    if not path.lower().endswith(".png"):
        raise ValueError("masters must be saved as .png (lossless)")
    image.save(path, format="PNG")
