---
name: path-completeness-not-stable
description: "The extrusion pipeline's path_completeness swings 0.253-0.891 on identical scenes; it tracks where the circle fit lands, not what the camera saw. Judge levers on ROI coverage instead."
metadata: 
  node_type: memory
  type: project
  originSessionId: 29056919-c652-4e99-b9ca-e8ca5b163359
  modified: 2026-09-01T11:50:27.802Z
---

2026-09-01, trial `20260901-153855-8fe68421` (16 takes, one pose, arm never
moved, stack untouched): reported `path_completeness` ranged **0.253 to 0.891**
while the underlying sector coverage varied by only a few percent (ROI n
7553–8172).

**Why:** completeness is scored against the NOMINAL circle, and the measured
circle fit is under-constrained on a ring with a 100°+ gap. The fitted radius
wandered 37.7–44.2 mm against a nominal 42.0, and a few mm of radius error
swings completeness by a factor of three. Gap and fitted radius move with it.

**Consequence — this is the trap:** take 07 read 0.891 / gap 39.2° (the best
ring-2 measurement ever recorded) on the spatial-filter-OFF arm. Six presses
later, same arm, take 10 read 0.263. Quoting take 07 alone would have "proved"
spatial-off a large win; the coverage statistic showed it was actually 4% WORSE.

**What to use instead** for judging any lever: total ROI points in the ring band
and the along/across-stereo-baseline ratio, on a centre PINNED to the applied
nominal (never a per-take fit). Both reproduce to ~±0.05 across sessions.

**Corollary:** layer-2 invalidity is TWO compounding faults — a real sensor
anisotropy (ratio ~2.3) AND a fit that cannot stabilise on a partial ring.
Fixing either alone will not yield a valid layer-2 measurement.

Related: [[layer2-stack-segmentation]], [[ring2-dropout-spatial-filter-cleared]].
