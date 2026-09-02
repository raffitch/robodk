---
name: layer2-stack-segmentation
description: "Layer >=2 never measured validly (0/6 takes) because arc assembly was off above layer 1 and the largest 110-deg arc became \"the ring\"; fixed by a deposit floor under layer N + assembly everywhere."
metadata: 
  node_type: memory
  type: project
  originSessionId: 730828c5-1611-45cf-887d-7fccf7f3c54b
  modified: 2026-08-31T17:44:33.426Z
---

Layer >= 2 of a ring stack had **never once** been measured validly on the cell —
0 of 6 takes across 2026-08-30 (board, completeness 0.56-0.62) and 2026-08-31
(MDF, 0.27-0.36). Not a capture regression and **not caused by the MDF swap**.

Cause: the ring reaches DBSCAN whole (36/36 angular bins in the work ROI) and
leaves it as 5-7 arcs — a hand-placed ring 2's crest swings ~10 mm around the
circumference, so the 3D neighbourhood breaks at eps 5 mm. Layer 1 fragments the
same way (2 arcs) and was rescued by arc assembly; above layer 1 assembly was
deliberately off, so the **largest arc alone** became the measurement.

**Why:** the ban was protecting a real thing — assembly judges candidates on
circle-fit shape alone, so it cannot tell an arc of ring 2 from ring 1's crescent,
and fusing them erases the displacement the PFH experiment measures. But the
alternative it left in place ("keep the largest arc") was worse.

**How to apply:** the fix is a **deposit floor under layer N at the top of layer
N-1**, applied to the deposit population *after* the compactness filter — NOT to
the ROI band (measured: that starves compactness and makes the take worse,
0.294 -> 0.232). With the layer beneath gone, assembly is safe everywhere.

Traps that cost time here:
- **A height cut alone is a no-op.** Sweeping it 0->11 mm moves coverage by at
  most one bin. Assembly is what recovers the ring; the floor is what makes
  assembly *safe*. Read the floor as a safety device, not a coverage device.
- **The two rings do NOT separate by height in the deposit cloud.** The layer-2
  height histogram runs continuously 1->19 mm — ring 2 sits ON ring 1 and the
  camera sees one unbroken flank. The apparent separation is only in the *crest*
  cloud (post upward-normal). I recommended a fix on that false premise; the
  measurement killed it.
- **The two archives move in OPPOSITE directions and both are right.**
  2026-08-31: 0.294 -> 0.515 (was one arc). 2026-08-30: 0.62 -> 0.50 (was padded
  with ring 1's crest). The tell is the fitted radius: 3 repeats of one physical
  ring collapse from 7.24 mm spread to 0.29 mm.
- **Layer 2 must STAY invalid.** `tests/test_extrusion_golden.py` holds
  `LAYER2_MAX_COMPLETENESS = 0.75` for that. Do not raise it.

Still open and the real blocker: a contiguous ~50 deg sector the chain cannot use.
ROI points per 10 deg sector in the bead annulus fall to 22-121 in the 140-190 deg
band against 250-466 elsewhere in the SAME frame (layer 1: 60-377). Do NOT quote
the golden test's "200-530 valid depth pixels" beside these — that is a different
measurement on the 2026-08-30 frame. A 19 mm stack seen from one top-down pose
shadows itself; that is [[multiview-inspection-spec]] territory, not segmentation.

Related: [[deposit-segmentation-spec-plan]], [[pfh-paper-ring-stack-experiment]],
[[tasni-backend-native-crash]].
