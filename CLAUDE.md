# Inspecting hand-soldered joints with bench-vision

Two views: **scope** = microscope straight down (coverage, bridges, texture);
**side** = low oblique from the left, ~40° (fillet profile, height, lifted pads; may show iron/hand).
Coordinates for `capture_region` are full-res pixels of the rotated frame; every reply gives the scale.

## Workflow
1. `list_cameras`, then `grid(cam="scope", rows=4, cols=6)` for an overview the user can point at ("B3").
2. Zoom with `capture_cell` (add `margin=0.25` if a joint sits on a grid line) or `capture_region`.
3. Check profiles on `side` with `capture_region` around the same joints.
4. Before rework: `save_reference(cam, "u1-before")`; after: `compare` and zoom into the numbered boxes.
5. If the image is soft or dark, say so and use `set_control` before judging: `focus_absolute`,
   `exposure_time_absolute` or `gain` (turn off `focus_automatic_continuous` / set `auto_exposure=1` first;
   older kernels: `focus_auto`, `exposure_auto`, `exposure_absolute`).

## What to look for (and which view decides)
| Condition | Looks like | Decide with |
|---|---|---|
| Good fillet | Smooth, concave meniscus from pad up the lead; shiny (leaded) or satin (lead-free); pad fully wetted to its edge; lead outline visible | scope for wetting, side for concave profile (contact angle well under 90°) |
| Cold joint | Dull, grainy or crystalline; lumpy or balled (convex); poor wetting with a dark ring or crack around the lead | scope for texture; side for the balled, convex shape |
| Insufficient solder | Copper of the pad or ring still visible; hole not filled; thin flat fillet | scope (exposed copper); side (low/flat profile) |
| Bridge | Solder spanning between adjacent pads, pins or traces | scope; confirm on side that it is solder, not glare or flux |
| Lifted pad | Pad tilted or raised off the board, gap or shadow under it, cracked trace at its edge | side (gap under the pad); scope for a torn trace |
| Unsoldered pin | Bare shiny lead in a clean hole or on a dry pad, no fillet at all | scope |
| Solder ball | Small round bead on the solder mask, not attached to a pad | scope to find it; side shows a 3D highlight and shadow (not a stain) |

Traps: specular glare looks like a bridge or a void; flux residue looks dull like a cold joint; lead-free
solder is naturally satin, so judge cold joints by shape and wetting, not shine alone. The side camera
foreshortens: never call a bridge from side alone.

## How to answer
Always give a per-joint table, then next steps:

| Joint | Verdict | Confidence | Evidence (view) |
|---|---|---|---|
| J3 | Bridge to J4 | high | solder spans the gap on scope; visible as one mass on side |

- Verdict is one of: good, cold, insufficient, bridge, lifted pad, unsoldered, solder ball, unsure.
- Confidence is high / medium / low. Use low (or unsure) when the image is blurry, glary, occluded or
  the joint is only partly in view; never guess past what the pixels show.
- For anything below high, end with the exact next call, e.g.
  `capture_region(cam="side", x=1180, y=420, w=300, h=200)` and what it should settle.
- Say what you could not see (other side of the board, under a component, inside a through-hole).
