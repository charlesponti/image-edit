# @pontistudios/image-edit

**An image-editing suite with a proof habit: surgical edits of flat images, and reproducible builds from layers.**

## Surgical edits (`composite` / `verify`)

Every generative editor — ChatGPT's image edits, Gemini, the rest — re-renders the *whole* image when you ask for a small change. The vendor docs admit it: OpenAI's mask is "entirely prompt-based" guidance, Gemini's API has no mask parameter at all ("semantic masking" is a sentence in your prompt). Ask for a different coat colour and the face, the background and the typography all quietly drift. Measured on a real edit: **2,674,842 of 2,741,191 pixels outside the target region changed — 97.5% of everything you didn't ask about.**

This package is the fix, and it's mostly arithmetic:

1. **Let the model re-render.** Ask for the edit in plain words and take its full-frame output as untrusted raw material.
2. **Build a mask** of where the edit is allowed to exist — traced from the original (colour segmentation inside a located box) or from the pixel diff between original and edit (for moves).
3. **Paste back.** `out = original × (1 − α) + edit × α`, with α the feathered mask. Where α is 0, the output *is* the original, bit-for-bit.
4. **Verify.** Dilate the mask into a guard zone and diff the final file against the original outside it. The edit ships only at **0 changed pixels, max difference 0**.
5. **Save lossless.** PNG masters only — a JPEG round-trip recompresses the whole frame and voids the guarantee (measured: ~35k noise-changed pixels at quality 92).

## Install

```bash
pip install -e .
```

## Use

Garment/colour change — mask traced by luminance inside a located box (boxes are normalised 0–1000 coordinates, straight from a visual-grounding pass):

```bash
image-edit composite --orig art.png --edit model-edit.webp \
    --mask-mode dark --dark-box 684,204,929,809 \
    --out art-surgical.png --report report.json
```

Move an object — mask built from the actual pixel difference inside region boxes (where it was, where it lands, any limb that re-poses):

```bash
image-edit composite --orig art.png --edit model-edit.webp \
    --mask-mode diff \
    --regions 620,435,765,725 840,185,980,735 675,145,845,415 \
    --out art-surgical.png --report report.json
```

A JSON verification report is printed (and saved with `--report`):

```json
{
  "mask_pixels": 354504,
  "changed_inside_mask": 316811,
  "changed_outside_guard": 0,
  "max_diff_outside_guard": 0,
  "raw_edit_changed_outside_guard": 2675462,
  "guarantee_holds": true,
  "saved_file_matches_verified_array": true
}
```

The CLI exits non-zero when the guarantee fails.

## Build from layers (`stack`)

The dual of the surgical edit: when an image is *assembled* — artwork over a base, an occluder over a line, linework over both — don't flatten it and edit the flat file later. Keep the layers and rebuild:

```bash
image-edit stack --out banner.png --size 2560x1440 --bg 000000 \
    bridge.png sil-left.png@200,430:620x620 sil-right.png@1740,430:620x620 \
    lines-left.png@200,430:620x620 lines-right.png@1740,430:620x620
```

Layers composite in the order given (Porter-Duff "over"). Each layer spec is `path[@x,y][:WxH]`: an optional canvas offset and an optional resize. Order is the whole semantics — an opaque layer hides exactly what it covers, which is how the banner above removes the bridge's deck line from under the two hosts: a black silhouette layer sits between the bridge and their wireframe linework, so the line stops at each silhouette and resumes after it. No model, no mask, no redraw — and the build is reproducible from the layer files byte-for-byte (verified: a CLI rebuild of a script-built banner differed in 0 of 3,686,400 pixels).

The command prints a build report (canvas, background, each layer's file/offset/size, and a re-read check that the saved file matches the verified array). Output is PNG-only, same masters rule as `composite`.

Rule of thumb: if you own the layers, `stack`; if all you have is the flat image, `composite`.

## Case studies

Two real productions, run before this code was packaged (same algorithms, now the CLI): [`examples/say-less.md`](examples/say-less.md) — a coat recolour via a luminance-traced mask, and a handbag moved to the other arm via a three-region diff mask. Both verified at 0 changed pixels outside the guard.

## What the guarantee does and doesn't cover

It guarantees everything *outside* the mask is the original image, provably. Inside the mask you get the model's redraw, for better or worse — mask craft (tight, self-aligning, diff-based where possible) is how you keep that region small and honest. The package also can't fix a model edit whose geometry doesn't align with the original; the diff mask absorbs small shifts, large ones are a re-roll.

## Tests

```bash
python -m unittest discover -s tests
```

Synthetic suite: a hostile edit that randomises every pixel still cannot leak outside the mask; drift outside the diff regions is rejected; an intended change inside them lands.

## Layout

```
src/image_edit/
    core.py    load / align / feathered paste-back / layer stacking / lossless save
    masks.py   box union, dark-region tracing, change-diff masks, guard
    verify.py  the outside-the-mask guarantee check
    cli.py     image-edit composite | verify | stack
```
