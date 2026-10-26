"""The guarantee check.

verify_outside(original, final, mask) answers one question with pixel
arithmetic, not vibes: did anything outside the edit region change?

The verdict that ships an edit is `changed_outside_guard == 0` at
tolerance 0 -- byte-identical, not "looks the same".
"""

from __future__ import annotations

import numpy as np

from .masks import guard_mask


def max_channel_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-pixel maximum absolute channel difference."""
    return np.abs(a.astype(int) - b.astype(int)).max(axis=2)


def verify_outside(
    original: np.ndarray,
    final: np.ndarray,
    mask: np.ndarray,
    guard_dilation: int = 41,
) -> dict:
    """Compare `final` against `original` outside the guard-dilated mask.

    Returns a report dict. `guarantee_holds` is True only when exactly
    zero pixels outside the guard changed at all (tolerance 0).
    `raw_edit_*` fields, when the caller also passes the raw model edit
    via verify_report(), quantify how much the un-pasted model output
    would have drifted.
    """
    guard = guard_mask(mask, guard_dilation)
    outside = guard == 0
    d = max_channel_diff(final, original)
    changed_outside = int(((d > 0) & outside).sum())
    return {
        "image_pixels": int(original.shape[0] * original.shape[1]),
        "mask_pixels": int((mask > 0).sum()),
        "guard_pixels": int((guard > 0).sum()),
        "changed_inside_mask": int(((d > 2) & (mask > 0)).sum()),
        "changed_outside_guard": changed_outside,
        "max_diff_outside_guard": int(d[outside].max()) if outside.any() else 0,
        "guarantee_holds": changed_outside == 0,
    }


def count_changed_outside(a: np.ndarray, b: np.ndarray, mask: np.ndarray, guard_dilation: int = 41) -> int:
    """Changed pixels outside the guard, for any pair (e.g. the raw edit)."""
    guard = guard_mask(mask, guard_dilation)
    d = max_channel_diff(a, b)
    return int(((d > 0) & (guard == 0)).sum())
