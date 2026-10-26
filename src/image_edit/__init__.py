"""@pontistudios/image-edit — surgical image edits with a proof.

The pipeline in one line: a generative model re-renders the WHOLE image
(and drifts everything), code pastes back only the masked region, and a
pixel diff verifies that everything outside the mask is byte-identical
to the original.

Modules:
    core    -- image IO, alignment, feathered paste-back composite
    masks   -- mask construction (explicit, dark-region, change-diff)
    verify  -- the outside-the-mask guarantee check
    cli     -- `image-edit composite` / `image-edit verify`
"""

__version__ = "0.1.0"

from .core import load_image, save_png, align_edit, paste_back
from .masks import mask_from_dark_region, mask_from_diff, mask_from_boxes, guard_mask
from .verify import verify_outside

__all__ = [
    "load_image",
    "save_png",
    "align_edit",
    "paste_back",
    "mask_from_dark_region",
    "mask_from_diff",
    "mask_from_boxes",
    "guard_mask",
    "verify_outside",
]
