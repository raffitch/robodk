---
name: extrusion-debug-images-mirrored
description: "Extrusion debug rasters were left-right mirrored against color.png; fixed at write time 2026-08-31, but older archives are still un-flipped"
metadata:
  node_type: memory
  type: project
---

`segmentation.png` / `skeleton.png` / `comparison.png` are rasterised in WORK XY
(+X -> +column), but the cell's camera looks down with its +X axis along work **-X**, so
the saved PNGs came out **mirrored left-to-right against the `color.png` beside them**.
Vertical always agreed.

Measured on `runs/extrusion/20260831-190027-dd013e33`: +40 mm work X -> raster **col +40**,
photo **u -174**; +40 mm work Y -> both DOWN.

Fixed by `archive._write_debug_raster` (flip at WRITE time). In-memory arrays stay in work
coordinates — `_rasterize`'s `lo` origin and every pixel->mm conversion depend on that — and
the web UI serves these FILES (`module.py` SERVED_FILES), so one flip covers archive + UI.

**How to apply:**
- **Archives written BEFORE 2026-08-31 ~19:00 are NOT flipped**, including the whole
  2026-08-30 golden archive. Mirror one side yourself when comparing a historical take's
  segmentation to its own photo.
- Only ever a display convention: `depth.npy` and every number in `report.json` unaffected.
- The operator found it by eye. It had already misled the assistant into naming the wrong
  side of a ring in the same session — when reasoning about WHERE on a ring something is,
  state the frame (work angle vs clock-position-in-the-photo) explicitly.

Related: [[extrusion-take-figures]], [[first-live-take-board-halo]], [[crest-height-shortfall]].
