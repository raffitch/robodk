---
name: raster-free-centreline-beats-nksr
description: "2026-08-31 offline result — a raster-free polar-crest centreline beats the shipped skeleton (radius sd 0.074→0.047 mm, 11x on the partial ring), so NKSR/NoKSR is not needed for measurement."
metadata: 
  node_type: memory
  type: project
  originSessionId: 5da34dff-5914-4cd0-ab61-9769baaa5d06
  modified: 2026-08-31T12:01:24.893Z
---

Run in `D:\DesktopStuff\nksr-verification\` (deliberately OUTSIDE the repo; nothing
in the main repo was modified). The question was whether NKSR would measure the ring
better than the shipped chain. Answered by running the cheap control first instead:
measure the centreline on the ORIGINAL float coordinates — polar unwrap → per-angle
parabola vertex of h(r) → the same `fit_circle_xy` — with the shipped segmentation
untouched in front of it.

Across the 8 archived layer-1 takes of trial `20260830-202416-293b208d`:
radius sd **0.0737 → 0.0469 mm**, centre-offset sd **0.1573 → 0.0565 mm**, and on the
63%-complete layer-2 ring the radius spread is **0.880 mm shipped vs 0.076 mm**
(11.5x). Shape RMS is unchanged (1.60 → 1.65 mm), which is the control proving the
gain is not bought by smoothing roundness away.

**Z axis (asked separately):** the gain extends to height but is much smaller —
take-to-take sd of mean ring height 0.0543 → 0.0394 mm (1.4x). The lattice reaches Z
only indirectly: the shipped chain takes the centreline's Z from `points[nearest, 2]`,
a nearest-neighbour lookup at the QUANTISED xy, and `ring_geometry` reads every height
statistic off that spline. Around-the-ring profile spread is preserved (2.29 vs
2.08 mm) so it is not flattening, and the two methods agree on absolute height to
0.014 mm. **Absolute Z is dominated by the substrate plane fit (the datum) and depth
noise, which this work does not touch** — if Z is the priority the leverage is there,
not in the centreline.

**Per-sector profile (second pass).** The centreline is now a PROFILE — 180 angles ×
(crest height, crest radius, bead width, footprint asymmetry) from one shared per-angle
fit. Measured noise floor per sector: height 0.159 mm (3σ detect 0.48), width 0.346
(1.04), radius 0.153 (0.46), asymmetry 0.331 (0.99), against 9–11 mm of real variation
around these lumpy hand-placed rings. **Ring averaging buys less than √N**: ring-mean
height sd 0.039 mm where independent sectors predict ~0.012, so the 180 sectors act like
**~16 independent ones** — the rest is common-mode (substrate datum, depth bias) and is
the floor under every ring-level number. **Parked vs re-approach**: the 8 takes are 5 in
excursion 1 + 3 separate excursions sharing one archived `T_work_camera`, so pose error
is uncorrected and lands in the data — yet the moved takes scatter LESS (0.0267 vs
0.0876 shipped), so a re-approach penalty is **not resolvable at n=5 vs 3**. Never quote
one from this archive. **Gates**: radial trim dormant (0–19 of ~2900 points); the 23°
upward-normal cut keeps only 30–33% of the bead body, so the SHIPPED centreline sees the
top third only — but the profile metric does NOT inherit that cut (it fits the full
deposit body). **No buckled specimen exists**, so collapse detection is a computed
threshold (~0.5 mm height loss / ~1.0 mm width gain per sector), not a capability.

**Peer-reviewed by robodkclaude-13 (the session that built the segmentation swap).**
Verdict: implement, but as an ADDITIONAL reported metric, never a replacement — on the
2026-08-29 contamination fixture the BRANCH GUARD is what caught it and a crest fit has
no topology instrument behind it. Its design: run both estimators and treat their
disagreement as the alarm (a medial-axis and a ridge estimator diverge differently under
contamination than under honest geometry). It confirmed my 0.0737 baseline is
bit-identical to its Task 7 golden run, so the 0.0502 figure is the PRE-swap chain —
that old discrepancy is closed. Sequencing: branch from its `deposit-segmentation` push,
never rebase before it; and do NOT bundle this with the known `_rasterize` bead-dilation
fix (implemented, measured, reverted, pinned `xfail(strict=True)` because it trades one
real frame for another). Rule it stated that generalises: **rasterize to decide topology,
never to measure** — it hit the same class in `compactness_filter` (7 decision flips in
72 rotations, spread ≈ √2 mm).

**Datum sized (verified myself, and it corrects the peer's framing).** Its 0.205 mm
intercept spread reproduces exactly, but the intercept is evaluated at the work-frame
ORIGIN where tilt noise is levered in. UNDER THE RING the datum varies sd **0.0338 mm**
= **74%** of the 0.0394 mm ring-height variance, leaving 0.0202 if perfect. So the datum
IS the dominant Z term but is worth ~0.019 mm, **not an order of magnitude** — quoting
the intercept overstates it ~2x. Independently matches the common-mode floor from the
effective-N argument. The plane fit carries a ~+0.19 mm Fisher-consistency correction
(`bias_correction_mm`) that any datum audit must account for.

**REFUTED, by joint test with robodkclaude-13: disagreement-as-alarm.** Running both
estimators and watching their disagreement (even as an angular departure-from-constant-
offset, calibrated within each frame) does NOT separate contaminated frames from honest
ones. Decisive: `ring1_low_relief_20260829.npz` is a CLEAN 1 mm-worded fixture and reads
z −4.53 / −2.48 against the 8-take honest population — more extreme than two of three
contaminated fixtures. The statistics detect "not one of the eight archived takes".
Structural reason: of 1478 contamination points reaching the work ROI, **7 survive to the
crest**, so the chain cleans it upstream of both estimators. **Rule: metric cleaning and
topological perturbation are different events** — a few surviving points barely move a
fitted radius but can still spawn a skeleton branch, so a contamination detector must
read topology or raw membership. That is why the branch guard works and this cannot.

**That control also invalidated one of my own Established claims.** If a CLEAN frame of a
different ring is a 4.5σ outlier against the 8-take population, the per-sector floors
(and the 0.48/1.04 mm buckling thresholds from them) are THIS RING's repeatability at one
pose, not instrument specs. Never quote them without that qualifier. Report rescoped.

**Two captures that must be CREATED, not searched for:** (1) a deliberate contamination
frame — the 2026-08-29 takes that passed VALID with +0.6–0.7 mm bias were never archived
(only the one that crashed was fixtured), and without such a frame NO contamination
detector is testable; (2) a second clean ring on protocol-2 depth, to separate ring
identity from instrument resolution. **Both have a day-one consumer:**
`tests/test_extrusion_golden.py` reprocesses archived takes read-only against frozen
per-take baselines and skips without `runs/`, so each capture needs only a take entry
plus a frozen expectation — no new harness. Ask for captures that way; it decides whether
they happen.

**Why this matters beyond the one result:** the existing report had already measured
σ 0.044 at the crest stage and 0.108 at the final spline — the information was in the
cloud and the raster was spending it. The control recovers it. A reconstructor is not
needed for MEASUREMENT; NoKSR specifically is the wrong tool (its edge is scale on
large sparse scenes, this is a dense single-view 9k-point ring, and a stronger prior
bridges gaps more confidently — the stated failure condition). Reconstruction survives
only as a visualisation option.

**What is NOT established: trueness.** The two methods disagree by 0.34 mm on layer 1
and this archive cannot referee it — they measure different things (medial axis of the
crest footprint vs ridge of maximum height). Repeatability is blind to constant bias;
the introduced-offset protocol is what settles it. See
[[pfh-paper-ring-stack-experiment]].

**Why:** it kept a session from being spent on a CUDA toolchain for a gain the control
already delivers, and it is the "own piece of work, with its own validation" the
segmentation report filed the crest fit as. See [[deposit-segmentation-spec-plan]].

**How to apply:**
- Probing the raster lattice by TRANSLATING the cloud is a null test — `_rasterize`
  anchors the grid at `xy.min() - margin`, so the grid rides along and the radius
  moves exactly 0.0000 mm. Move the grid ORIGIN under a fixed cloud.
- Any new estimator must be swept over its own arbitrary knobs, not just scored on
  repeatability. The first version of this one binned points and fitted per bin, which
  made bin COUNT a confound (0.17–0.26 mm of movement, worse than the lattice);
  separating fit support from output sampling fixed it. Distinguish bias knobs from
  noise knobs — the support window moves the mean 0.186 mm but leaves the sd at
  0.036–0.056, so it is a calibration constant, not an error source.
- Completeness must come from raw angular occupancy, never from estimator output, or a
  method can erase a placement fault like layer 2's.
- The shipped baseline measured here is 0.0737 mm at working-tree HEAD, NOT the
  0.0502 mm recorded on 2026-08-30; three commits landed in between and the difference
  was never reconciled. Do not quote the two interchangeably.

Report: https://claude.ai/code/artifact/592700bf-c59a-4b9b-8041-59be652bb35a
