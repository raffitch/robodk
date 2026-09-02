---
name: extrusion-take-figures
description: "Per-take paper/app figures for the extrusion module — the set, and the correctness traps real captures exposed (colour band, fitted centre, mesh pitch, honest exaggeration)."
metadata: 
  node_type: memory
  type: project
  originSessionId: 56e2244c-bc6f-45a2-8693-b217e31e6c2e
  modified: 2026-08-29T11:48:04.753Z
---

`tasni/modules/extrusion/figures.py` (merged to `main` `ba8d2b3` 2026-08-28, extended
`e0b78d7` 2026-08-29) renders `plan` / `heightmap` / `mesh` / `iso` / `profile` /
`pipeline` per take plus a per-trial `stack` and `tube`, 300 dpi PNG + vector PDF,
**from the archive alone** — no robot, RoboDK or camera. Serving is render-if-missing,
so takes archived earlier produce figures with zero cell time. Needs
`pip install -e .[figures]` (matplotlib); without it measurements are unchanged.

Traps found by rendering REAL captures, each of which silently corrupts a figure while
every synthetic test stays green:

1. **Colour range must come from the deposit band.** D435i dropouts hundreds of mm below
   the work plane own the scale otherwise, and the ring — the whole subject — flattens to
   one colour (a −45…+5 mm scale instead of −1…10 mm).
2. **The nominal centre must be FITTED, not averaged.** The archive writes a CLOSED ring,
   so its first point repeats and the arithmetic mean is biased by radius/N — 0.33 mm on
   the cell's 181-point 40 mm ring — which made the plotted RMS (11.45) disagree with the
   manifest (11.31) a reader checks it against.
3. **A mesh pitch finer than the cloud's own spacing yields specks, not a surface.**
   Isolated cells share no edges: 26 triangles from a 1517-point bead. `_auto_cell`
   coarsens until cells average a few returns. The deposit lands at ~3 mm because the
   chain voxel-downsamples at 2 mm — that IS the measurement's resolution.
4. **`set_box_aspect` stretches Z regardless of what the axis label claims.** A flat
   `(1, 1, .55)` box exaggerated the bead ~6× while the title said ×2. The box must carry
   the data's own proportions for a stated ×N to be true.
5. **The archived `height-or-pointcloud.npy` is the CREST, not the bead** (578 points on
   the first cell ring, flanks discarded by design). For anything that needs the bead's
   width, read `take_stages()`'s `radial_trimmed` / `deposit_cluster` instead.

`mesh` (the surfaced view: work surface + deposit, each from above and rotated) exists
because the OLD paper's mesh pictures came from `macros/3DScan.py:354-358` — a Poisson
mesh shown in an interactive `o3d.visualization.draw_geometries` window, rotated and
screenshotted by hand, writing no file. The ring-stack chain never meshed at all. It is
deliberately 2.5-D (one top-down RGB-D frame measures a height field; a solid would
invent the underside) and leaves unmeasured cells open, so the ring's hole stays a hole.

Cost per take: `pipeline` ~34 s, `mesh` ~9 s, the rest ~1 s each, all drawn eagerly by
`RingMeasureJob` outside the camera hold. Both chain-replaying figures share a
one-entry `take_stages` cache keyed on the manifest's mtime.

Lesson worth keeping: rendering real archived data found every one of these; the
synthetic tests were green throughout. Look at the picture, not only the assertions.

Lineage: successor to the original `PostExtrusionToolpath` (on the user's Desktop) —
alpha shape → skeletonize → `overlay.png`. Related:
[[pfh-paper-ring-stack-experiment]], [[restart-tasni-backend-after-code-edits]].
