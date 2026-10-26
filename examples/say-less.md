# Case study: "Say Less" (Yua) — two surgical edits, Mon 26 Oct 2026

Artwork: wordless landscape 5:4 illustration, Yua full-body in a long black tailored coat, one hand over her mouth, a giant empty speech bubble filling the left of frame. Canvas 2000×1600.

## 1. Coat recolour: black → deep crimson

- Model instruction: *"Change only the colour of her long black coat to deep crimson red. Everything else stays exactly the same."*
- Mask: visual grounding located the coat (normalised box 684,204 → 929,809); inside it, luminance < 95 traced the coat's actual outline → 354,865 mask pixels.
- Raw model edit, changed pixels **outside** the guard-dilated mask: **2,674,842** (97.5% of the outside area — face, bubble, background all re-rendered).
- After paste-back: **0 changed pixels outside the guard, max channel difference 0**. Inside the mask, 317,168 pixels took the new colour.
- First pass was saved as JPEG and failed verification (~35k outside changes) — pure recompression noise. That failure is why masters are PNG-only in this package.

## 2. Move the handbag to her left arm

A move is the hard case: three regions change — where the bag was (plus the arm that held it), where it lands (left arm/forearm), and the mouth-hand corridor (the model re-poses that arm).

- Regions (normalised): `620,435,765,725` (old bag + arm), `840,185,980,735` (destination arm + bag), `675,145,845,415` (mouth/right-arm corridor).
- Mask: change-diff inside the regions (threshold 25, closed, dilated) → 329,114 pixels, ~10% of canvas. Rectangles were *not* pasted: only genuinely changed pixels came across.
- Raw model edit changed 2,560,425 pixels outside the guard. After paste-back: **0 changed outside, max diff 0**, PNG re-read identical to the verified array.

## Lesson the numbers teach

The guarantee protects everything outside the mask; it cannot improve what's inside it. Both edits were only as good as the model's redraw within ~10–11% of the frame — and everything else, including a deliberately fragile fragmented-cubist face a few dozen pixels wide, survived untouched because the code never let the model near it.
