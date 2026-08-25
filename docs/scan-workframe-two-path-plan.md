# Scan Workframe Survey - Two-Path Engineering Plan

**Status:** IMPLEMENTED 2026-08-13 — all 17 tasks of
[scan-workframe-implementation-plan.md](scan-workframe-implementation-plan.md) merged to
branch `calibration-improvements`, 393 tests green (`py -3.10 -m pytest -q`) plus a clean
frontend build. See the "Hardware validation TODO" section (§18) below for what is
deliberately still unverified on the real cell.  
**Purpose:** agreed design for the two-path workframe survey  
**Scope:** flat work surfaces used to create a RoboDK workframe and visible work rectangle  
**Out of scope:** general 3D object scanning and the Cylinder Test implementation

## 1. Goal

Tasni must create a trustworthy flat workframe while keeping the RealSense camera at
the closest **validated** measurement distance. A small platform may be measured in one
view. A platform too large for that view must be measured from multiple close views
rather than by moving far away and sacrificing depth quality.

The finished RoboDK result must contain:

- a workframe whose origin, axes, and Z plane have known provenance;
- the complete intended rectangular boundary;
- four named corner points for edge-to-edge programming;
- measurement quality and acceptance evidence;
- an explicit distinction between measured geometry and user-specified geometry.

The operator should not need to know whether the surface is "small" or "large" before
starting. Tasni determines the path after the first center acquisition.

## 2. Agreed product decisions

There are only two workframe survey paths:

1. **Compact workframe:** the complete physical rectangle fits at the validated optimal
   camera distance with enough margin for the planned scan views.
2. **Large workframe:** it does not fit at that distance, so Tasni performs a guided
   center-plus-four-corner survey while maintaining the same close-range measurement
   quality.

Both paths produce the same workframe/rectangle result shape. They differ only in how
the boundary evidence is collected.

In addition to the two survey paths, the first release keeps one **non-survey fast
path**: the **user-specified region** — a rectangle of operator-entered dimensions
projected onto the measured plane, centred on the reticle. It produces the same result
shape but its boundary provenance is `user specified - plane measured, boundary
declared`, it is drawn visually distinct in review, and it can never be presented or
accepted as a measured boundary. It replaces (and relabels) today's fixed
1,000 × 1,000 mm crop, so a quick extrusion trial does not require a five-position
pendant survey.

Default frame convention (accepted 2026-08-12):

- origin: calculated rectangle center;
- +Z: fitted surface normal, oriented consistently with the robot cell;
- +X: physical long edge of the rectangle;
- +Y: completes a right-handed frame;
- corners: named `C1..C4`, clockwise **viewed looking along −Z** (from above the
  surface), with `C1` the corner nearest the robot base; ordering must be
  deterministic across repeated surveys of the same surface.

The centered origin supports the most common center-work use case. The named corners
and full rectangle support edge-to-edge jobs without requiring a second workframe.

## 3. Important definitions

### Neutral pose

The convenient, safe robot starting pose in the lab. It is not a measurement target and
does not define the camera distance. The operator starts there and then follows Tasni's
guidance to the measurement pose.

### Validated optimal distance (`d*`)

The closest camera-to-surface distance at which the complete cell produces repeatable,
sufficiently complete, and sufficiently accurate measurements. It is not simply the
camera's advertised minimum range.

`d*` must be established experimentally using the existing A3 ChArUco board and/or
another dimensionally known flat artifact. Validation includes the RealSense, camera
settings, hand-eye calibration, robot pose registration, and surface processing as one
measurement chain.

### Compact versus large

These terms are relative to `d*`, not to the neutral pose and not to the absolute table
size:

- **Compact:** all four boundaries fit confidently at `d*`, including safety/scan margin.
- **Large:** one or more boundaries cannot be observed confidently at `d*`.

### Survey versus production scan

The **workframe survey** establishes the plane, rectangle, and coordinate system. A
future **object scan** measures objects placed in that frame and is a separate workflow.
For a known-flat workframe, creating a dense point cloud of every interior millimeter is
not required, but the plane and boundary still require quantified evidence.

## 4. Top-level UX

```text
Start at neutral pose
        |
Place reticle near approximate surface center
        |
Guide to d* and approximately level
        |
Capture authoritative center measurement
        |
Can all four physical boundaries be observed confidently at d*?
        |
   +----+----+
   |         |
  Yes        No
   |         |
Compact      Guided center + four-corner survey
survey       at the same d*
   |         |
   +----+----+
        |
Review exact measured rectangle, axes, and quality
        |
Create scan targets -> mandatory dry run -> scan/validate -> insert
```

Tasni may suggest the path automatically, but the operator must see and confirm the
classification. An expert override may exist for diagnostics; it must not allow weak
geometry to be labeled as measured and accepted.

## 5. Establishing the optimal distance

Before production use, run a one-time cell characterization at several candidate
distances. Repeat it after changes to the camera mount, camera settings, calibration,
robot, or relevant processing.

At each distance, collect repeated measurements of the A3 ChArUco/known artifact and
evaluate at minimum:

- valid-depth coverage and spatial distribution;
- plane-fit RMS and maximum residual;
- repeatability of plane height and normal;
- recovered known length/width error;
- workframe origin and angle repeatability;
- sensitivity to surface reflectivity/texture and illumination;
- hand-eye/robot-pose registration error across several robot poses;
- oblique-incidence captures at (at least) the worst planned scan-pose tilt, so the
  tolerances are not validated only at normal incidence.

Choose the closest distance that passes the downstream job's error budget reliably,
not the distance producing the smallest voxel in one favorable capture.

Characterization must be built as an **in-app tool** (a mode of the calibration module,
whose ChArUco capture loop and metrics already exist), not a manual procedure in a
document — otherwise re-characterization after camera/calibration/robot changes will
silently not happen. Each characterization run stores a dated `calibration_id` that the
§10 gates consume; "calibration expired" is evaluated against it.

The user requested "minimal" error, but that is not an acceptance specification. Before
implementation, define the maximum permissible job error. A proposed engineering rule is
to allocate no more than one third of that error to workframe measurement, leaving margin
for robot positioning and the process itself.

### Intrinsic fit calculation

At perpendicular distance `d`, the pinhole footprint is approximately:

```text
view_width_mm  = d * image_width  / fx
view_height_mm = d * image_height / fy
```

A rectangle of dimensions `Sx, Sy` needs at least:

```text
d_fit = max(margin * Sx * fx / image_width,
            margin * Sy * fy / image_height)
```

This is only a first eligibility calculation. The final compact test must project the
measured rectangle through every planned scan pose and prove required coverage. A
rectangle that barely fits in the center view can leave the image during an oblique
view.

Using the current saved 1280x720 intrinsics, the raw view is approximately:

| Distance | Raw view width | Raw view height |
|---:|---:|---:|
| 300 mm | 432 mm | 243 mm |
| 500 mm | 719 mm | 404 mm |
| 800 mm | 1,151 mm | 647 mm |

These are not safe rectangle limits. Margins, oblique views, depth validity, and
calibration uncertainty reduce the usable area. The production compact limit must be
derived after `d*` and the scan-pose policy are validated.

## 6. Path A - Compact workframe

### Entry conditions

All conditions must pass at `d*`:

- a surface is detected under the reticle;
- all four physical boundaries are detected with confidence;
- raw (not cosmetically trimmed) boundaries remain inside a configured guard region;
- the rectangle is sufficiently centered;
- the plane is within the survey tilt tolerance;
- expected planned-view coverage passes the hard threshold;
- rectangle identity is consistent across a short multi-frame acquisition.

The coverage condition means classification runs the scan-pose planner in
**predict-only mode** (no robot motion) against the candidate rectangle — it is part of
compact eligibility, not a post-lock check.

Seeing camera points beyond the platform can corroborate an edge, but is not itself proof
that the platform fits. A boundary requires color/segmentation and/or depth-discontinuity
evidence.

### Acquisition

1. Guide the camera to `d*` and near-normal incidence.
2. Freeze the recommended pose; do not recompute it while the operator follows it.
3. Stop the robot and explicitly refresh its actual position from the driver/RoboDK.
4. Capture a synchronized multi-frame RGB-D burst and the associated camera pose.
5. Measure the plane and all four boundaries from that authoritative acquisition.
6. Generate scan targets whose predicted coverage includes the complete rectangle.
7. Hard-fail if reachability, collision, or coverage requirements cannot be satisfied.

### Output

The rectangle is labeled `camera measured - complete boundary`. The review view must draw
the authoritative locked polygon that will be inserted, not a previous live/SAM outline.

## 7. Path B - Large workframe: center plus four corners

The camera remains at `d*`. Tasni guides the operator around the rectangle instead of
backing away.

### Capture sequence

1. **Center:** initial plane, approximate datum, and automatic compact/large decision.
2. **Corner 1:** capture the corner and both adjacent edges.
3. **Corner 2:** clockwise from Corner 1.
4. **Corner 3:** clockwise from Corner 2.
5. **Corner 4:** clockwise from Corner 3.

The UI must make ordering unmistakable and show the accumulating polygon in both the
camera view and a simple top-down diagram.

At each position:

1. The user jogs the robot using the pendant.
2. The camera supplies range, level, corner visibility, and freshness guidance.
3. The user stops the robot; Tasni shows `Measuring...` rather than pretending RoboDK is
   following the pendant continuously.
4. Tasni explicitly requests the current robot joints/pose.
5. Tasni captures multiple fresh RGB-D frames.
6. The capture is accepted only if standoff, tilt, depth support, edge support, timestamp
   freshness, and robot-pose freshness pass.

Each accepted acquisition is an atomic record:

```text
RGB frames + depth frames + robot joints + base->camera pose
+ camera calibration identity + timestamps + quality metrics
```

### Geometry estimation

Do not estimate the workframe from only four selected 3D pixels. Use all supported
evidence:

1. Transform plane points from all five acquisitions into the robot base frame.
2. Fit one robust global plane and calculate per-position residuals.
3. Project each corner's boundary evidence onto the global plane.
4. Robustly fit the four physical edge lines.
5. Solve a global constrained rectangle:
   - opposite edges parallel;
   - adjacent edges perpendicular within tolerance;
   - corners are the fitted line intersections;
   - detected local corners agree with those intersections.
6. Calculate dimensions, center, axes, closure error, and uncertainty/confidence.

The global fit must not silently force poor data into a perfect rectangle. Report both
the unconstrained evidence residuals and the constrained result. Reject the survey if
the discrepancy is too large.

**Conditioning note (review outcome, 2026-08-12; corrected 2026-08-13 — see below):**
fitting each edge line from two corner-local segments is well-conditioned in
*direction* — the corner-to-corner baseline is the full edge length, a large lever arm.
The real accuracy floor is **cross-capture registration**: each of the five captures
carries hand-eye plus robot absolute-pose error, so edge *positions* inherit a
systematic ~1–2 mm uncertainty that the constrained fit cannot remove and corner-closure
error only partially reveals. The Phase 0 error budget must assume this floor for the
five-position path.

**Correction (Task 9 implementation finding, 2026-08-13): the unconstrained-vs-
constrained discrepancy is NOT the primary diagnostic for cross-capture registration
error — it is structurally blind to the dominant failure mode.** The constrained
rectangle model has 5 degrees of freedom (one shared orientation `theta` plus 4
independent edge offsets); the unconstrained model has 8 (4 independent lines × 2 DOF
each). The 3 degrees of freedom the constraint removes are **all angular** — the four
edge offsets are free parameters, fitted identically to the evidence in both models. So
`discrepancy_mm` can only ever detect **angular** cross-capture inconsistency (one
capture's edge tilted relative to the others); it cannot see a pure **translational**
registration error at all — which is exactly the dominant hand-eye/robot-pose error mode
across five separately-registered capture positions.

This was measured, not just derived: injecting a 150 mm rigid translation into one
capture's evidence produced a confident, badly-mis-sized rectangle while
`discrepancy_mm` went *down* (0.084 → 0.070 mm) rather than up. The field that actually
caught it is `corner_agreement_mm` (the fitted rectangle's corners vs. the surveyed
corner points), which reported 150.02 mm on the same injected fault.

The implementation (`five_position.py::finish`) therefore gates on **both**
`survey_rect_discrepancy_mm` (angular consistency) and `survey_corner_agreement_mm`
(translational consistency — see `rect_fit.py`'s module docstring for the full
derivation), and treats `corner_agreement_mm is None` as a failed check, never a pass.
Anyone extending the five-position geometry should treat `corner_agreement_mm` — not
`discrepancy_mm` — as the primary registration-error diagnostic.

### Failure and recovery

- If an edge/corner is weak: ask the operator to reposition and recapture that corner.
- If a rounded or obstructed corner cannot define two edges: request one or more
  additional edge-midpoint captures.
- If captures are not coplanar within a warn tolerance: label the surface `non-flat`
  with per-region plane residuals reported, and require explicit operator acceptance.
  Reject only above a second, hard threshold. A large fabrication table with
  millimetre-scale bow must be neither silently forced flat nor uselessly rejected.
- If a boundary is visually and geometrically invisible: camera-only automation cannot
  recover it; offer a known-dimensions or physical-datum workflow in a later scope.
- Never invent a fixed-size rectangle and label it as fully measured.

### Output

The final complete rectangle is visible in RoboDK even though it never fit in one camera
image. It is labeled `camera measured - five-position boundary survey` with its quality
report and acquisition provenance.

A later enhancement may drive the robot to suggested corner viewpoints under operator
supervision (the calibration module already performs collision-screened automatic
tours); the first release is pendant-jog only, which is the safer choice near unknown
table edges but should not remain the only option forever.

## 8. Role of the A3 ChArUco board

The board is not required on every workframe if the cell has passed validation and the
physical edges are detectable. It remains important for:

- hand-eye calibration;
- determining and periodically verifying `d*`;
- known-scale dimension and frame-orientation checks;
- detecting calibration or camera-mount drift;
- independent acceptance testing of the complete measurement chain.

The camera cannot validate its own systematic calibration bias using an unknown flat
surface. A featureless plane also contains no information about yaw. For production
confidence, schedule board-based verification at startup, after relevant changes, or at
a validated interval.

An optional future workflow may use the board as a temporary workframe datum, but it is
not part of the two-path minimum design.

## 9. Live guidance model

**Corrected premise (review outcome, 2026-08-12).** When the KUKA driver is connected
and monitoring, RoboDK *does* mirror pendant jogging continuously — this is
live-verified in the current cell (the backend pose-hold releases on a real axis move,
tracks it, and refreezes; the vision-escape thresholds exist precisely for the
driver-not-monitoring case). Guidance therefore has two modes rather than one
pessimistic assumption:

- **Driver monitoring active** (normal mode): live pose-tracked guidance (range, level,
  X/Y, segmentation, measurement age) may be shown as live, with staleness detection on
  the mirrored pose itself.
- **Driver not monitoring**: live X/Y based on a frozen robot pose must not be
  presented as real-time truth; guidance degrades to camera-only readouts plus explicit
  measure steps.

In both modes:

- each authoritative capture explicitly refreshes the real robot position from the
  driver/RoboDK and passes freshness gates — live mirroring never substitutes for this;
- authoritative captures use a step-and-measure interaction:
  `Jog -> Stop -> Measure -> Accept`;
- a stale measurement is visibly marked and can never satisfy readiness;
- the recommended target pose is frozen after acquisition until `Remeasure` is chosen.

Once an axis enters its tolerance, the UI shows `IN POSITION`, not a noisy residual that
encourages the operator to chase zero. Show one highest-priority correction at a time.

## 10. Required quality report and hard gates

Before target creation/insertion, report at minimum:

- acquisition mode: compact or five-position;
- boundary provenance;
- camera calibration identity/date;
- standoff per acquisition;
- plane RMS/max residual and per-position residuals;
- normal and normal-repeatability estimate;
- width, height, and their uncertainty/repeatability estimate;
- edge-line residuals;
- parallelism, perpendicularity, and corner-closure errors;
- predicted scan coverage;
- actual post-scan coverage, including weakest edge;
- robot pose/timestamp freshness;
- pass/warn/fail decision and reasons.

Hard failures must include:

- stale or missing robot pose;
- stale or incomplete RGB-D acquisition;
- missing required physical edge;
- plane/corner inconsistency above tolerance;
- insufficient predicted or actual coverage;
- too few reachable/collision-free targets;
- calibration missing, expired, or failing verification;
- robot movement after lock;
- dry-run failure.

Collision filtering and the RoboDK dry run should be mandatory for real motion. A soft
collision bypass is not appropriate for a production target set.

## 11. Source-of-truth contract

Maintain one immutable `LockedWorkframeSurvey` used by review, planning, scan ROI, and
RoboDK insertion. It should contain:

```text
mode
captures[]
plane_base
corners_base
center_base
frame_base
size_mm
boundary_provenance
quality
calibration_id
locked_robot_state
```

Live color/SAM segmentation may propose boundaries and improve operator feedback. Depth
and calibrated transforms provide metric geometry. The locked review must display the
exact `corners_base` that downstream code will use. No frontend display latch may replace
the locked polygon with older live geometry.

## 12. Relationship to the current implementation

The existing code already provides useful components:

- camera transport and multi-frame depth capture;
- stored intrinsics/distortion and hand-eye transform;
- surface plane/rectangle fitting;
- color and SAM boundary proposals;
- RoboDK camera-pose transforms;
- scan pose generation, reachability, collision checks, TSDF fusion, and coverage;
- workframe/rectangle/mesh insertion.

Important current behaviors that this plan supersedes:

- fixed 1,000 x 1,000 mm large-surface crop — retained only as the relabeled
  **user-specified region** fast path (§2): operator-entered dimensions, provenance
  `user specified`, never presented as measured;
- one center aim for a workzone larger than the camera footprint;
- automatic full/crop classification based partly on near-complete depth fill;
- continuously recomputed live distance target;
- backend and frontend geometry freezes that can hide a lateral jog;
- preserving the old live outline after the authoritative lock;
- advisory UI values presented as if all were mandatory;
- soft coverage/collision behavior for crop target sets.

## 13. Proposed implementation phases

No implementation should begin until this plan is reviewed and the acceptance tolerances
are selected.

### Phase 0 - Measurement characterization

- Define downstream job tolerance.
- Build the repeated ChArUco/known-artifact distance test **as an in-app
  characterization tool** (calibration-module mode) producing a stored, dated
  `calibration_id` — not a manual procedure.
- Select `d*`, compact guard margin, and calibration verification interval.
- Record a baseline uncertainty/error budget (assume the ~1–2 mm registration floor
  for the five-position path, §7).

### Phase 1 - Immutable capture and quality contracts

- Define typed center/corner capture records.
- Add explicit robot-position refresh and timestamp checks.
- Add multi-frame measurement quality metrics.
- Remove dependence on continuously mirrored RoboDK pose for readiness.

### Phase 2 - Compact classifier and locked review

- Implement four-boundary confidence and guard-band eligibility at `d*`.
- Freeze the correction target.
- Make locked geometry the sole review/planner/insertion source.
- Validate predicted coverage across planned scan poses.

### Phase 3 - Five-position survey

- Implement the guided capture state machine.
- Add global plane, edge-line, and constrained-rectangle fitting.
- Add recapture and optional edge-midpoint recovery.
- Generate the same locked result contract as the compact path.

### Phase 4 - Safety and execution gates

- Make collision-free target availability and dry run mandatory.
- Hard-gate predicted/actual boundary coverage.
- Reject target use after the locked robot/surface state changes.

### Phase 5 - Workframe validation

- Validate inserted origin, normal, dimensions, and corners against a known artifact or
  independent touch/check procedure.
- Store repeatability results across repeated complete surveys.
- Permit production insertion only when configured acceptance criteria pass.

## 14. Verification matrix

Unit/synthetic tests:

- compact rectangle fits/overruns each image boundary;
- oblique planned views preserve/lose required coverage;
- five perfect captures recover the expected rectangle;
- noisy corner/edge evidence produces bounded error;
- one bad corner requests recapture rather than biasing the rectangle;
- non-coplanar samples are rejected;
- parallelism/perpendicularity/closure thresholds behave correctly;
- stale pose/frame pairs are rejected;
- frame axes and corner ordering remain deterministic.

Mock integration tests:

- explicit robot-position refresh occurs for every authoritative capture;
- movement after capture invalidates the lock;
- compact and five-position paths produce the same result schema;
- the displayed locked polygon equals the planner and insertion polygon;
- target generation hard-fails low coverage/collision/dry-run cases.

Cell acceptance tests:

- repeat a compact known rectangle in several positions/orientations;
- repeat a large known rectangle with the five-position survey;
- compare recovered dimensions, origin, and normal with independent ground truth;
- exercise different surface colors/textures and edge backgrounds;
- test one deliberately weak/occluded corner and verify recapture behavior;
- confirm the robot can use all four edges without exceeding the physical surface;
- demonstrate repeatability over multiple days and after reconnecting RoboDK/camera.

## 15. Decisions still required before implementation

1. Maximum permissible workframe origin, plane-height, dimension, and angular errors.
2. ~~Frame convention~~ — **resolved 2026-08-12**: center origin plus named corners is
   accepted as the single convention; configurable origins are deferred.
3. Required confidence/behavior when a corner is rounded or physically inaccessible.
4. ~~Known-dimensions path~~ — **resolved 2026-08-12**: the user-specified region ships
   in the first release as a labeled non-survey fast path (§2); the ChArUco-datum
   workflow remains a future fallback.
5. Calibration verification interval and the operational response to a failed check.
6. Whether the current camera resolution/settings are fixed during characterization.

## 16. Review questions for the independent agent

The reviewing agent should challenge specifically:

- Is five-position evidence sufficient for the stated known-flat assumption?
- Is the proposed `d*` characterization adequate and traceable?
- What error budget and acceptance decision rule should be used?
- Is fitting four global edge lines from corner-local captures well-conditioned?
- Should the origin be center, a physical corner datum, or configurable?
- How should robot pose and camera timestamps be synchronized given the current transport?
- Are additional edge-midpoint captures needed by default rather than only on failure?
- What safety gates must be mandatory before moving the real KUKA?
- Which parts of the existing scan code can be retained without preserving its current
  full/crop ambiguity and double-freeze behavior?

## 17. Review outcome (2026-08-12)

An independent review agreed with the overall architecture — in particular the two-path
split with a single result shape (§2), the immutable `LockedWorkframeSurvey` as the sole
review/planner/insertion source (§11, which directly addresses the recorded
live-outline-vs-locked-polygon and double-freeze bug class), the experimental `d*`
characterization (§5), and mandatory collision/dry-run gates (§10). Four corrections
were incorporated into this revision:

1. **§9 premise corrected.** RoboDK does mirror pendant jogging when the KUKA driver is
   monitoring (live-verified in this cell). Live pose-tracked guidance is kept as the
   normal mode; step-and-measure remains the authoritative-capture protocol in both
   modes.
2. **User-specified region promoted** (§2, §12, §15.4): the known-dimensions rectangle
   ships in the first release as a labeled non-survey fast path replacing the fixed
   1 m² crop, so quick trials never require a five-position survey.
3. **Edge-fit conditioning answered** (§7): direction is well-conditioned via the
   corner-to-corner baseline; the accuracy floor is cross-capture registration
   (~1–2 mm systematic), which the Phase 0 error budget must absorb.
4. **Phase 0 as an in-app tool** (§5, §13): characterization that isn't a button will
   not be re-run; it stores a dated `calibration_id` consumed by the §10 gates.

Answers to the remaining §16 questions: five-position evidence is sufficient *given*
the flatness warn/reject tiers and the registration-error budget; the centered origin
with named corners is accepted as the single convention; classification runs the
planner predict-only (§6); and the §12 retained-component list was verified accurate
against the current code (`work_crop_mm`, the live distance target, and both freeze
layers all exist and are superseded as described).

## 18. Hardware validation TODO (2026-08-13)

Implementation (all 17 tasks) is complete and unit/synthetic/mock-integration tested
(393 tests green), but several items were deliberately deferred to the real KUKA cell
during implementation — recorded here rather than fixed blind, per each task's review.
None of them block merging the software; all of them block calling the feature
production-validated. Cross-reference: `docs/scan-workframe-implementation-plan.md`'s
`progress.md` ledger (Task 13, Task 14, Task 15 entries) has the original findings this
section summarizes.

1. **Corner-aiming plane selection.** With the reticle centred exactly on a 90° table
   corner, the camera sees at most ~25% table — a geometric ceiling (the two near edges,
   each a straight line through image centre, always quarter a rectangular frame),
   confirmed by ray-tracing all four corners of the test fixture, independent of table
   size or standoff. `survey_surface`'s plane RANSAC is a pure inlier-count majority
   vote with no expected-plane hint, so at that ceiling it can select the BACKGROUND
   plane as "the surface" instead of the table. This fails safe today whenever the
   background is meaningfully nearer or farther than the table — the standoff gate
   catches it (a floor-lock case was measured moving the reported standoff from 420 mm
   to 1170 mm, well outside the accepted window). The residual risk is a background
   surface at a SIMILAR standoff to the table (e.g. an adjacent panel or fixture near
   table height) being silently accepted as the table plane. The corner-capture metrics
   (`purity` / `coverage`, `survey_corner_min_plane_coverage_frac`) are only as
   trustworthy as the plane RANSAC selected — they cannot detect a wrong-but-plausible
   plane, only a too-small one. Needs a real-cell check with an adjacent surface at
   table-like height in frame.
2. **Survey diagram chirality.** `SurveyPanel.tsx` maps world X/Y straight onto SVG
   x/y, and SVG's Y axis points down, so whether the accumulating top-down polygon
   reads clockwise to an operator standing at the robot depends on the handedness of
   the KUKA base frame in this cell. The diagram exists specifically so the operator
   can confirm the C1→C2→C3→C4 clockwise order called for in §2 — this needs a
   physical-cell check (stand at the robot, walk the four corners, confirm the on-screen
   polygon direction matches).
3. **`d*` has never been measured.** `tools/characterize_distance.py` exists, has unit
   tests for its pure-geometry helpers, and is headless-importable — but its
   interactive, operator-jogged capture path has never run end-to-end against the real
   D435i/KUKA. Until it does, `accurate_min_mm` (and the rest of the standoff/tilt
   bands in `ScanConfig`) are placeholders for the experimentally-validated `d*` §5
   calls for, and the whole error budget in §5 is unvalidated against real hardware.
4. **The five-position registration floor is an estimate.** The ~1–2 mm systematic
   cross-capture registration uncertainty assumed in §7's Phase 0 error budget is an
   engineering estimate (hand-eye + robot absolute-pose error composition), not a
   measured number. It should be measured once `d*` characterization (item 3) and a
   real five-position survey of a known rectangle are both available.
5. **`height_repeat_mm` caveat.** With two capture normals that are exactly opposite
   (e.g. two views 180° apart in yaw), `height_repeat_mm` reports a misleadingly
   confident `0.0` (the mean-normal vector's magnitude collapses to zero, and the
   norm-guard silently avoids a crash rather than surfacing the degeneracy) while
   `normal_repeat_deg` correctly reports `90`. `choose_dstar`'s selection logic is safe
   because it gates on both fields together, but any future consumer — a UI, a report,
   a human — must never read `height_repeat_mm` alone as a health/repeatability number.

