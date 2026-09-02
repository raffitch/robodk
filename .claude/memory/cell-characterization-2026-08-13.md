---
name: cell-characterization-2026-08-13
description: "The KUKA cell's measured depth-quality envelope - incidence costs ~4x what distance costs; numbers that Milestone B (Tasks 6-9) must be built from"
metadata: 
  node_type: memory
  type: project
  originSessionId: 963be792-72bf-4fcf-ae9a-77dd06dcf629
  modified: 2026-08-13T10:35:13.997Z
---

The cell was characterized on 2026-08-13 (`characterization/characterization-20260813.json`,
`discovery: false`, camera `cam-858b95e78d20`). This is the first characterization
that ever existed, so before it every surface lock warned "calibration verification
missing or expired". That warning is now silent.

Fronto-parallel, 100% board coverage, measured standoff -> plane RMS:
310 mm -> 0.934 | 400 -> 0.982 | 498 -> 1.115 | 599 -> 1.512 | 795 -> 2.049 mm.
Incidence at ~310 mm: 1.0 deg -> 0.650 | 9.1 -> 2.006 | 19.6 -> 4.969 | 29.4 -> 7.430 mm.
`length_err` <= 0.43 mm and `length_spread` <= 0.032 mm everywhere.

**Why:** Milestone B's planner (Tasks 6-9) must choose standoff, cone and pose count
from real data rather than invented defaults. Two facts decide the design:
incidence costs about **4x what distance costs** (distance triples RMS across the
whole band; 29 deg tilt multiplies it by 11), and distance is a **real trade, not
free** - an earlier claim that it was free came from contaminated measurements and
is retracted. For a 2x1 m platform: 310 mm = 0.65 mm/42 tiles, 498 = 1.13/16,
795 = 2.05/9.

**How to apply:** protect squareness first, standoff second. `choose_dstar` returns
the CLOSEST passing trial (310 mm) but Task 6 wants the FARTHEST (795 mm) - both are
derivable from the stored `trials`. Do NOT gate on `plane_max` (pinned against the
outlier-rejection band at close range) or `height_repeat` (30 s at one pose, not
tour-scale drift). The budget on file is a regression detector, not a process
requirement - the downstream tolerance is still undefined and only the user can set
it. See [[characterize-tool-cell-defects]] for the five measurement bugs the cell
runs exposed, and `docs/adaptive-scan-frame-implementation-plan.md` for the full
table plus caveats.
