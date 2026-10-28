"""Command line: `image-edit composite`, `image-edit verify`, `image-edit stack`.

Examples
--------
Colour/garment change traced by luminance inside a located box
(boxes are normalised 0-1000: x0 y0 x1 y1)::

    image-edit composite --orig art.png --edit model-edit.webp \
        --mask-mode dark --dark-box 684,204,929,809 \
        --out art-surgical.png --report report.json

Move operation, mask built from the actual pixel diff inside regions
(one box for where the object was, one for where it lands, one for any
limb that re-poses)::

    image-edit composite --orig art.png --edit model-edit.webp \
        --mask-mode diff \
        --regions 620,435,765,725 840,185,980,735 675,145,845,415 \
        --out art-surgical.png --report report.json

Explicit mask file (255 = edit may land)::

    image-edit composite --orig art.png --edit model-edit.webp \
        --mask-mode file --mask-file mask.png --out art-surgical.png

Build an image from layers instead of editing a flat one (layers are
composited in the order given; each spec is path[@x,y][:WxH])::

    image-edit stack --out banner.png --size 2560x1440 --bg 000000 \
        bridge.png sil-left.png@200,430:620x620 sil-right.png@1740,430:620x620 \
        lines-left.png@200,430:620x620 lines-right.png@1740,430:620x620
"""

from __future__ import annotations

import argparse
import json
import re
import sys

import cv2
from PIL import Image

from .core import (
    align_edit,
    load_image,
    load_layer,
    paste_back,
    save_png,
    save_png_pil,
    stack_layers,
)
from .masks import mask_from_dark_region, mask_from_diff
from .verify import count_changed_outside, verify_outside


def _parse_box(text: str) -> tuple[float, float, float, float]:
    parts = [float(p) for p in text.replace(",", " ").split()]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(f"box needs 4 numbers (x0 y0 x1 y1), got: {text!r}")
    return tuple(parts)  # type: ignore[return-value]


def build_mask(args, original, edit):
    if args.mask_mode == "dark":
        return mask_from_dark_region(
            original, args.dark_box, luminance_threshold=args.dark_threshold
        )
    if args.mask_mode == "diff":
        return mask_from_diff(
            original, edit, args.regions, diff_threshold=args.diff_threshold
        )
    if args.mask_mode == "file":
        mask = cv2.imread(args.mask_file, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"could not read mask: {args.mask_file}")
        if mask.shape != original.shape[:2]:
            mask = cv2.resize(mask, (original.shape[1], original.shape[0]))
        return mask
    raise ValueError(f"unknown mask mode: {args.mask_mode}")


def cmd_composite(args) -> int:
    original = load_image(args.orig)
    edit = align_edit(args.edit, original.shape)
    mask = build_mask(args, original, edit)
    final = paste_back(original, edit, mask, feather_sigma=args.feather)

    report = verify_outside(original, final, mask, guard_dilation=args.guard_dilate)
    report["raw_edit_changed_outside_guard"] = count_changed_outside(
        edit, original, mask, guard_dilation=args.guard_dilate
    )
    report["files"] = {"original": args.orig, "edit": args.edit, "out": args.out}

    save_png(final, args.out)
    # Re-read the saved master: the guarantee is claimed about the FILE.
    reread = load_image(args.out)
    report["saved_file_matches_verified_array"] = bool((reread == final).all())

    print(json.dumps(report, indent=2))
    if args.report:
        with open(args.report, "w") as fh:
            json.dump(report, fh, indent=2)
    return 0 if report["guarantee_holds"] and report["saved_file_matches_verified_array"] else 1


_SIZE_SUFFIX = re.compile(r":(\d+)x(\d+)$")
_OFFSET_SUFFIX = re.compile(r"@(-?\d+),(-?\d+)$")


def _parse_layer_spec(text: str):
    """Parse 'path[@x,y][:WxH]' -> (path, (x, y), (w, h) | None)."""
    size = None
    m = _SIZE_SUFFIX.search(text)
    if m:
        size = (int(m.group(1)), int(m.group(2)))
        text = text[: m.start()]
    offset = (0, 0)
    m = _OFFSET_SUFFIX.search(text)
    if m:
        offset = (int(m.group(1)), int(m.group(2)))
        text = text[: m.start()]
    if not text:
        raise argparse.ArgumentTypeError(f"layer spec has no path: {text!r}")
    return text, offset, size


def _parse_size(text: str) -> tuple[int, int]:
    m = re.fullmatch(r"(\d+)x(\d+)", text)
    if not m:
        raise argparse.ArgumentTypeError(f"size needs WxH, got: {text!r}")
    return int(m.group(1)), int(m.group(2))


def _parse_bg(text: str) -> tuple[int, int, int, int]:
    if not re.fullmatch(r"[0-9a-fA-F]{6}([0-9a-fA-F]{2})?", text):
        raise argparse.ArgumentTypeError(f"bg needs RRGGBB or RRGGBBAA hex, got: {text!r}")
    vals = tuple(int(text[i : i + 2], 16) for i in range(0, len(text), 2))
    return vals if len(vals) == 4 else (*vals, 255)  # type: ignore[return-value]


def cmd_stack(args) -> int:
    specs = [_parse_layer_spec(s) for s in args.layers]
    loaded = []
    for path, offset, size in specs:
        image = load_layer(path)
        if size is not None:
            image = image.resize(size, Image.LANCZOS)
        loaded.append((image, offset, {"file": path, "at": list(offset), "size": list(image.size)}))

    canvas_size = args.size or loaded[0][0].size
    background = args.bg
    final = stack_layers(canvas_size, background, [(im, off[0], off[1]) for im, off, _ in loaded])
    save_png_pil(final, args.out)

    reread = Image.open(args.out).convert("RGBA")
    report = {
        "canvas": list(canvas_size),
        "background": "".join(f"{c:02x}" for c in background),
        "layers": [meta for _, _, meta in loaded],
        "out": args.out,
        "saved_file_matches_verified_array": bool(
            reread.size == final.size and list(reread.getdata()) == list(final.getdata())
        ),
    }
    print(json.dumps(report, indent=2))
    if args.report:
        with open(args.report, "w") as fh:
            json.dump(report, fh, indent=2)
    return 0 if report["saved_file_matches_verified_array"] else 1


def cmd_verify(args) -> int:
    original = load_image(args.orig)
    final = load_image(args.final)
    mask = cv2.imread(args.mask_file, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"could not read mask: {args.mask_file}")
    report = verify_outside(original, final, mask, guard_dilation=args.guard_dilate)
    print(json.dumps(report, indent=2))
    return 0 if report["guarantee_holds"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="image-edit", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    comp = sub.add_parser("composite", help="paste a model edit back through a mask and verify")
    comp.add_argument("--orig", required=True)
    comp.add_argument("--edit", required=True, help="the model's full re-render")
    comp.add_argument("--out", required=True, help="output PNG (lossless, enforced)")
    comp.add_argument("--report", help="write the verification report JSON here")
    comp.add_argument("--mask-mode", choices=["dark", "diff", "file"], required=True)
    comp.add_argument("--dark-box", type=_parse_box, help="'x0,y0,x1,y1' (0-1000) for --mask-mode dark")
    comp.add_argument("--dark-threshold", type=int, default=95)
    comp.add_argument("--regions", type=_parse_box, nargs="+", help="boxes for --mask-mode diff")
    comp.add_argument("--diff-threshold", type=int, default=25)
    comp.add_argument("--mask-file", help="mask PNG for --mask-mode file")
    comp.add_argument("--feather", type=float, default=5.0, help="feather sigma in px")
    comp.add_argument("--guard-dilate", type=int, default=41, help="guard dilation kernel")
    comp.set_defaults(fn=cmd_composite)

    ver = sub.add_parser("verify", help="check an existing composite against its original + mask")
    ver.add_argument("--orig", required=True)
    ver.add_argument("--final", required=True)
    ver.add_argument("--mask-file", required=True)
    ver.add_argument("--guard-dilate", type=int, default=41)
    ver.set_defaults(fn=cmd_verify)

    stk = sub.add_parser("stack", help="build an image by compositing RGBA layers in order")
    stk.add_argument("--out", required=True, help="output PNG (lossless, enforced)")
    stk.add_argument("--report", help="write the build report JSON here")
    stk.add_argument("--size", type=_parse_size, help="canvas WxH; default: first layer's size")
    stk.add_argument("--bg", type=_parse_bg, default=(0, 0, 0, 0),
                     help="canvas background RRGGBB[AA] hex (default transparent)")
    stk.add_argument("layers", nargs="+",
                     help="layer specs in stacking order: path[@x,y][:WxH]")
    stk.set_defaults(fn=cmd_stack)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
