"""@pontistudios/image-edit — an image-editing suite with a proof habit.

Two halves, one arithmetic family:

* Surgical edits of flat images. A generative model re-renders the
  WHOLE image (and drifts everything), code pastes back only the masked
  region, and a pixel diff verifies that everything outside the mask is
  byte-identical to the original.
* Layered builds (`stack`). When an image is assembled from layers,
  keep the layers: composite them in order, deterministically, and the
  build is reproducible from its sources -- occlusion by layer order
  replaces editing altogether.

Modules:
    core    -- image IO, alignment, feathered paste-back, layer stacking
    masks   -- mask construction (explicit, dark-region, change-diff)
    verify  -- the outside-the-mask guarantee check
    cli     -- `image-edit composite` / `image-edit verify` / `image-edit stack`
"""

__version__ = "0.2.0"

from .core import load_image, load_layer, save_png, save_png_pil, align_edit, paste_back, stack_layers
from .masks import mask_from_dark_region, mask_from_diff, mask_from_boxes, guard_mask
from .verify import verify_outside

__all__ = [
    "load_image",
    "load_layer",
    "save_png",
    "save_png_pil",
    "align_edit",
    "paste_back",
    "stack_layers",
    "mask_from_dark_region",
    "mask_from_diff",
    "mask_from_boxes",
    "guard_mask",
    "verify_outside",
]
