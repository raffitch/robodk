---
name: roll-verdict-vertical-flattens-bead
description: "2026-09-01 17:43 paired run ANSWERS the roll probe: the vertical view flattens the bead into the table where the bead runs ACROSS the stereo baseline. Sensor artifact, not geometry. 8/8 vertical invalid, 8/8 rolled valid, strict alternation."
metadata:
  node_type: memory
  type: project
---

`runs/extrusion/20260901-174339-36ba48cf`, layer 1, 16 takes alternating roll 0 / 90.
**The pair is clean** (T_work_camera: identical xyz 203.03/137.89/308.20, optical axis
(0,0,-1) both, camera +X work -X -> work +Y = exactly 90 deg; un-rolled colour crops
indistinguishable). So this IS the roll probe [[roll-probe-camera-vs-scene]] and the
verdict is unambiguous: **the dropout moves with the camera.**

**What the vertical view does:** it does not lose depth pixels (~500 per 10 deg bin in
the ring annulus, same as rolled) -- it reads the bead as FLAT. Work azimuth 150-180 deg:
vertical max height 0.6-2.4 mm, rolled 5.0-9.7 mm. Same at 320-340 (1.0-2.6 vs 5.3-6.2).

**Why that voids the take:** flattened below the ~1.5 mm deposit floor -> cloud breaks
in two -> `_rasterize` (processing.py:190) keeps ONLY the largest connected component,
silently discarding the other arc -> open 155 deg skeleton (endpoints 2 vs 0) ->
completeness 0.40-0.46 against the 0.90 gate. Not a marginal threshold; a discarded arc.

**The size mismatch is arc coverage, NOT scale.** Refit the ROLLED view's own complete
ring on just the 155 deg window the vertical view survives on: 41.6->37.4, 41.7->38.7,
41.6->37.8, 41.9->38.5. A +/-3-4 mm swing -- bigger than the 1.2-1.6 mm the two views
differ by. See [[path-completeness-not-stable]].

**The rule (pooled 8 pairs):** crest deficit rises 0.9 -> 1.8 mm from bead-ALONG-baseline
to bead-ACROSS-baseline, and the big loss needs BOTH across AND narrow (2.53 mm mean in
narrow x across; 0.31/0.45/1.36 in the other three cells). Roll 0 mean deficit 2.02 mm
(83/288 bins >3 mm); roll 90 0.30 mm (7/288). **The rolled view is not right, it is LUCKY
on this ring** -- its across-baseline axis (work ~79/259) lands on the ring's two widest
arcs (12-18 mm). Both views under-report their own across-baseline arcs.

**This is why "sometimes valid at neutral".** The 13:13 run (`20260901-131341-2b12355c`)
passed 8/8 at roll 0 -- and its weakest bins are the SAME two: az 165 only 104 points
clear the floor (crest 3.13 mm), az 330 only 139 (2.11), vs 400-600 elsewhere. The defect
is ALWAYS present in the vertical view; pass/fail is whether those arcs stay above floor.

Everything above came out of the archive. **Reprocess, do not re-run the robot.**
Related: [[ring2-dropout-spatial-filter-cleared]], [[crest-height-shortfall]],
[[depth-pimples-census-and-preset-landmine]], [[apply-characterization-ignores-validity]].
