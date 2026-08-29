# Cylinder Test: Current Implementation and Live-Test Handoff

Last updated: 2026-08-12. Active branch: `calibration-improvements`.
Current implementation commits: `e36b4d5`, `a0670f1`, and the scan-surface placement
change below.

This is the authoritative current-state handoff for Tasni's extrusion/cylinder
module. `docs/HANDOFF_EXTRUSION_CYLINDER.md` is the original requirements document;
its final "not implemented" status is historical and no longer accurate.

## Current outcome

The Cylinder Test is implemented end to end in the Tasni app:

- Browser-only live analytic preview responds immediately to sliders. It does not
  create executable robot coordinates or approve a path.
- `Generate coordinates & fingerprint` freezes recipe, setup, and dense XYZ points
  into an exact backend plan.
- RoboDK receives one native XYZ+IJK curve per layer, not hundreds of targets.
- A native Curve Follow/Robot Machining Project owns interpolation,
  approach/retract, process/rapid speed, blending, and path start/finish events.
- The linked generated program is pinned to the selected fixed XYZRPW orientation
  and then validated with collision checking.
- Complete dry run uses mock `AirOn`/`AirOff` programs, never physical outputs.
- Live print remains locked until the exact current fingerprint passes dry run and
  the operator confirms the live run.
- Each printed layer captures one RGB-D observation, processes/archives it, and can
  produce an opt-in bounded radial correction with a new fingerprint.

Primary code:

- `tasni/modules/extrusion/module.py`: API, plan/preflight/dry/live interlocks.
- `tasni/modules/extrusion/service.py`: dry/live jobs and cleanup behavior.
- `tasni/modules/extrusion/toolpath.py`: deterministic layered circle coordinates.
- `tasni/modules/extrusion/surface.py`: scan → extrusion placement handoff and checks.
- `tasni/modules/extrusion/processing.py`: single-frame measurement pipeline.
- `tasni/core/rdk_io.py`: native curves, machining projects, IK, generated programs.
- `tasni/webui/src/pages/Extrusion.tsx`: controls, oblique bird's-eye preview, workflow.
- `tests/test_extrusion.py` and `tests/test_extrusion_job.py`: main regression coverage.

## Why RoboDK returned status -5

The operator reported:

```text
RoboDK could not generate the curve-follow program (status -5.0)
```

The brief curve appearance was real: `AddCurve` succeeded, but RoboDK could not
find a feasible start/path pose for the Curve Follow Project. It was not a RoboDK
license failure and was not related to the hardware-I/O approval.

The placement UI previously exposed center X/Y but implicitly fixed the build plane
at selected-frame Z=0. Selecting `World` therefore placed the first layer near world
Z = half the bead diameter. During the live diagnosis, the current spindle TCP was:

```text
Frame: World
XYZ:   [-5.084, -1505.704, 377.227] mm
RPW:   [-177.072, 58.385, 89.722] deg
```

A circle near World origin was consequently far from the current reachable work
area. Commit `a0670f1` fixed the placement workflow:

- `build_plane_z_mm` is explicit and fingerprinted.
- `POST /api/modules/extrusion/current-tcp` reads the selected TCP in the selected
  frame without robot motion.
- `Seed path start from current TCP` makes circle angle zero equal to the current
  TCP, derives center X/Y and build-plane Z, and captures the exact current RPW.
- Preflight samples fixed-orientation IK across the layers before enabling dry run.
- Selecting `World` shows a warning that all values are station-world coordinates.
- A negative Curve Follow setup error now reports tool, frame, XYZ bounds, and RPW.

## Placement on the scanned work surface (preferred workflow)

The intended flow is: **scan the surface → insert it → build on that frame**. The Scan
module's insert creates `Tasni Work Frame`, the `Tasni Work Surface` rectangle, and the
fused mesh, and records the applied run in `runs/scan/active.json`. The Cylinder Test
now reads that pointer.

- `GET /api/modules/extrusion/scan-surface` reports the applied surface (frame, size,
  run id, applied time) or that none is applied.
- **Center on scanned surface** sets the work frame to the scan frame, centers the
  cylinder on the measured rectangle, and sets build-plane Z = 0 (the scan frame's
  origin lies on the surface). It records the originating run in `setup.scan_run_id`.
- That run id is part of the plan fingerprint, so **re-scanning the table invalidates a
  surface-placed plan** exactly like editing the recipe does.
- Preflight then enforces the placement: same run, same frame, and the wall
  (`radius + bead/2`) must fit inside the measured rectangle, reporting signed per-edge
  margins.
- Manual placement is still legal. **Seed path start from current TCP** clears
  `scan_run_id`, and preflight reports `placement: "manual"` with an advisory when a
  scanned surface exists on another frame.

Why the centre cannot be derived from the recorded extents: the scan puts its frame
origin on the rectangle **corner** nearest the robot base, and frame +Y is `Z × X`,
which on this cell points *off* the rectangle (Y spans about −295..0 mm). So (0, 0) is a
corner and (size/2, size/2) has the wrong Y sign. The insert therefore publishes
`rectangle_corners_frame_mm` and `rectangle_center_frame_mm` in **frame** coordinates,
and extrusion centres on the corner mean. A surface applied before those fields existed
is recovered from its `report.json`; if neither source is available the module refuses
to centre rather than guess (`available: false`).

Primary code: `tasni/modules/extrusion/surface.py`,
`tasni/modules/scan/plane.py:rectangle_in_frame`, and the payload written by
`tasni/modules/scan/service.py:insert_scan`.

## Automatic inspection pose (derived, not taught)

The inspection viewpoint used to be a hand-taught RoboDK target picked from a
dropdown, so the distance was whatever that pose happened to be and the cylinder
landed wherever it landed in the frame. With `setup.inspection_auto` (the default
for new plans) the pose is derived from the same placement the cylinder is built
on, and a **joint** target is created per layer.

- **Centred by construction.** The aim point is the cylinder axis at the top of the
  layer just deposited (`build_plane_z + bead + (i-1)*layer_height`). Every
  candidate pose puts that point on the camera's +Z axis at exactly the standoff,
  so it projects onto the principal point whatever roll/tilt is chosen.
- **Distance from the optics, not a constant.** Standoff is the pinhole
  fit-to-frame distance — the same rule as `scan/planner.py` — clamped into the
  accurate depth band (`extrusion.inspection_min_mm/_max_mm`, default 300–800 mm).
  Anchor: the operator measured an A3 sheet filling this camera's frame at ~380–400
  mm, and 297 mm × fy / H = 375 mm reproduces that from the intrinsics alone (the
  short side binds). At cylinder scale the near limit binds instead: an 86 mm wall
  would frame at 138 mm, inside the D435i's blind zone, so the standoff is held at
  300 mm and the UI says so (`framing.clamped_to = "near"`, ~40% of frame height).
  An object too large to frame *within* the band is refused, never answered by
  backing the camera out past `inspection_max_mm`.
- **There is no close-range headroom on this cell.** The D435i server already requests
  its maximum native depth profile, 1280×720 at 30 fps, whose MinZ is ~280 mm — so the
  300 mm `inspection_min_mm` floor is already the sensor limit, not a conservative
  choice. For the current 95 mm outside ring diameter the pinhole fit with 15% margin is
  135 mm, but that is an optics number the depth sensor cannot honour. The measure-only
  clamp (`measure_close_range_min_mm`) is therefore 300 mm as well, and the operator
  "tool detached/clear" checkbox has been removed. MinZ scales with depth width, so
  getting genuinely closer means dropping the server's depth resolution first.
- **Fronto-parallel first.** The 2026-08-13 characterization measured incidence
  costing ~4× what distance costs, so candidates are ordered straight-down → roll
  (free: still square to the surface, different wrist config) → 10° tilt.
- **Roll is measured from the ROBOT, not the work frame.** `pose_from_aim` takes a
  `reference_x`, and the service passes the camera's own +X at the job's start
  joints (read back with `RdkIO.camera_axes_in_frame`, no motion), so "roll 0"
  means the orientation the operator parked the camera in. It used to be
  hard-coded to the work frame's +X, and on this cell those are 180° apart: the
  Realsense TCP at the parked joints reads X=[-1,0,0] Y=[0,1,0] Z=[0,0,-1] in
  `Tasni Work Frame`, so frame-referenced "roll 0" was 179.7° from the robot's
  natural camera orientation and RoboDK could only reach it by flipping the wrist.
  The station-less `/inspection-pose` preview has no robot to ask, so it keeps the
  frame axis and LABELS which convention it used (`roll_reference` is `"frame_x"`
  or `"camera_at_start"`, with the vector) — a roll number without that label can
  be read exactly backwards.
- **The wrist branch is gated, not assumed.** The target is solved with
  `solve_joints_on_neutral_branch` (same JointsConfig as start, A4/A6 bounded by
  `setup.maximum_tool_axis_spin_deg`), not a seeded `SolveIK` — a seeded solve
  returns whichever branch is nearest and will hand back a flip. Measured: the old
  frame-referenced viewpoint had four IK branches, ALL flipped, and the stored one
  sat 178° from parked on axis 4 while passing collision validation, because a
  flipped wrist is not a collision. No qualifying branch is a candidate REJECTION,
  not a run failure. The accepted joints and their dA4/dA5/dA6 vs start are
  recorded in the pose block, and the inspection program gets the same
  interpolated wrist check the layer path uses.
- **Same authoritative gate for printing.** Each candidate is created, given an
  inspection program, and put through `update_program(collisions=...)`; the first to
  pass is used. If none does, the run fails with every rejection listed. On the
  live-print paths that check stays ON: nothing backs off, tilts past the configured
  cone, or drops collision checking to obtain a pass — straight down at 300 mm over a
  fresh print is the tightest clearance, and the spindle shares the flange with the
  camera.
- **Measure-only is the one exception (collisions OFF).** The hand-placed ring stack
  is not in the station model, so the check can only speak about cell furniture, and it
  was rejecting otherwise good camera-only poses. `MeasureLayerBody` and
  `CharacterizeBody` therefore default `collision_check_enabled=False`; candidates are
  still IK-, reachability- and wrist-screened, and the pose block records
  `"reachable, feasible"` rather than `"reachable, collision-free"`. Pass true in the
  request body to opt back in.
- Targets are named `<program>_Inspect_Target`, i.e. inside the existing
  `TasniCylinder_` namespace, so **Reset** and the normal artifact lifecycle already
  clean them. The chosen pose (and every rejected candidate) is logged in the dry-run
  report and archived in each layer's `provenance.inspection_pose`.

`POST /api/modules/extrusion/inspection-pose` previews the geometry with no station
and no motion; `preflight` returns the same block. Manual mode is unchanged — clear
the checkbox and the dropdown is required again.

Primary code: `tasni/modules/extrusion/inspection.py` (pure numpy),
`service.py:_build_inspection_move`, `rdk_io.py:create_inspection_target`.

## Ring-stack measure-only experiment (paper evidence, 2026-08-27)

> **Doing the paper cell run? Read [pfh-paper-handoff.md](pfh-paper-handoff.md).**
> It is the task page: what is still missing, the operator order (including the
> *Apply* step that was missed), and what must not be claimed. This section is the
> module reference behind it.

Design: `docs/superpowers/specs/2026-08-27-ring-stack-measure-only-design.md`;
plan: `docs/superpowers/plans/2026-08-27-ring-stack-measure-only.md`.

`MEASURE_ONLY` (`tasni/modules/extrusion/measure.py`) measures hand-placed dried
rings: each press moves only the camera to the derived inspection pose, takes one
RGB-D frame, runs `process_observation` (with the previous ring's measured top as
the floor) and archives a take under a `MEASURE_ONLY` trial that `/trials` never
counts as a print. `characterize_ring` derives radius / bead / height / centre from
ring 1 so the recipe comes from the physical ring, not a caliper.
`GET /trials/{id}/paper-summary` groups takes by the operator's introduced offset
and reports deviation, timing (`acquisition_to_path_ms`), height and bead numbers
with a ready-to-paste Markdown block.

First real-ring evidence (2026-08-28): `20260828-171615-f088cf48/characterize-01`
proved item-Start motion, capture, archive and return, but the original
largest-DBSCAN-cluster rule selected a broad ChArUco-board residual (4,134 points)
instead of the visible ring (1,609 points). Its 52.77 mm radius / 51.12 mm bead
result is invalid. Characterization now selects a complete, radially compact ring
cluster. Offline replay of the exact frame returns radius 39.17 mm, centre
(217.94, 150.44) mm, bead footprint 13.26 mm and top Z 6.14 mm; the real depth
fixture is in `tests/fixtures/extrusion/ring1/`. A fresh cell characterization is
still required before Apply or paper measurements.

A second cell attempt at 300 mm (after the ring-selector fix) was safely rejected:
the ring was physically present, but the depth frame yielded one broad 5,748-point
board-like cluster and no independent ring-like cluster. That attempt exposed two
operational need now implemented: raw color/depth/pose archiving even when
characterization fails. (It also produced a 175 mm close-range option built on a wrong
MinZ reading; that clamp is back at 300 mm — see the MinZ note above.) The rejected attempt predated that archiving and therefore has no raw frame.

**2026-08-28 evening: the protocol ran end to end on the cell** — see the session
`runs/extrusion/20260828-192115-47fb78ea` (characterize-01 + layer-001, both valid; numbers
in AGENTS.md). The one operator slip was skipping *Apply to recipe & placement* between
Characterize and Measure, so layer-001's deviation is dominated by a 15 mm centre offset
against the stale plan.

### Scoring against the operator's ground truth (2026-08-28)

`paper_summary` grades every take against the offset the operator typed, not just
against nominal:

- `detection_error_mm` per condition = `|measured center_offset_mm − annotation.introduced_offset_mm|`.
  A take whose manifest predates the measured offset *vector* returns `None` and is
  excluded — missing is not zero, and averaging it in would read as a perfect measurement.
- `shift_consistency` machine-checks the relation a pure translation must satisfy
  (`max = d`, `mean = 2d/pi`, `RMS = d/sqrt(2)`) against the condition's means, inside a
  band of `max(1.5 mm, 15% of d)` — 1.5 mm being this cell's own floor (board consistency
  1.26 mm, work-plane RMS 1.39 mm). Every disagreeing statistic is named, and the
  Markdown block prints a `WARNING` line rather than letting a bad condition average in.
- The zero-offset group is scored against `(0, 0)`, which makes it the baseline and makes
  a skipped *Apply* impossible to miss: replaying the 2026-08-28 take reports a 15.38 mm
  detection error against its "no offset introduced" annotation. Re-labelled with the
  displacement that was actually there, the same frame scores **0.002 mm**.

`figures.expected_ring(take)` returns the nominal ring translated by the introduced
offset — the ground truth — and `plan` and `stack` draw it (teal dash-dot) whenever a
non-zero offset was recorded. A take with no introduced offset gets no such line: a
duplicate of the nominal circle would read as evidence of something.

Adding that sixth legend entry exposed two layout defects that were already there and
are now regression-tested: the in-axes legend **covered the measured centreline** (it is
now below the axes, two columns, clear of the x-axis label), and the honesty caption
**was clipped at both ends** once an introduced offset made it long (it now wraps at
`CAPTION_WRAP`, without breaking "hand-placed" across lines).

### Board depth noise fused to the ring — the radial trim (2026-08-28 evening)

The first paper-protocol take after *Apply* (`20260828-204846-5b455377/layer-001`)
failed on the cell with `branch guard exhausted after 3 attempt(s)` while the colour
frame showed one clean ring. Reproduced offline from the archived frame, exactly
(mask 5906 / skeleton 302 / 1 branch pixel). Root cause, measured:

- The bare ChArUco board, unfiltered, 80–150 mm from the ring: z p50 **0.8 mm**, p99
  **+4.8 mm** — so the work frame's Z=0 is right, but **22.7 % of the board clears the
  2.5 mm deposit floor** (`max(deposit_min_height_mm, plane_distance_threshold_m·1000)`).
- Board patches touching the ring join its DBSCAN cluster (largest cluster 1680 of
  1743 points), pass the upward-normal test (a flat board faces straight up), and of
  the 639 points reaching the raster **136 (21 %) were board** at r 55–72 mm. Dilated by
  the 12.8 mm bead kernel they form a lobe fused to the ring: a skeleton T-junction at
  r 48.5 mm with a **37 mm arm**, longer than the 20 mm spur limit, on all three attempts.
- Characterize passed on the same ring minutes earlier by luck, not robustness: its
  coarse bead 13.48 vs 12.79 mm meant a 7 px vs 6 px dilation, which closed the gap
  instead of forming a T.

**Raising the floor is the wrong fix and is dangerous.** The bead's own top is z p25
1.8 / p50 3.8 mm, so it overlaps the board noise: a 3.0 mm floor read **r 36.7** for the
42.6 mm ring and called it valid; 5.0 mm read r 24.0, valid. Confidently wrong numbers
that pass every gate.

**The fix separates bead from board by shape.** `processing._radial_trim` runs after
the largest cluster is chosen and before the crest is picked: fit a circle, keep points
within a band of the *fitted* radius, refit, tightening through
`extrusion.radial_trim_schedule_mm = [15, 12, 10]` (first band wide enough to hold the
whole bead under a ~5 mm contamination-biased fit; last band ≈ bead half-width + fit
slack; at 8 the characterization fixture's coarse pass loses its ring). It follows a
displaced ring because it is about the fitted circle, not the nominal. Sweep evidence
(scratch, 2026-08-28): `[15,12,10]` was the only schedule that fixed the failed frame
**and** both synthetic board-lobe scenes without moving the frames that already worked:

| case | before | after |
|---|---|---|
| 20:48 real frame (failed on the cell) | branch guard exhausted | r 42.31, offset 1.28 mm, valid |
| 19:21 real frame (passed on the cell) | r 39.91 | r 40.10 (+0.19; centre +0.35 mm) |
| synthetic ring + board lobe | r 61.04 (+1.04 bias) | r 60.91 |
| synthetic clean ring | r 60.77 | r 60.77 (no-op) |
| ring1 characterization fixture | r 39.17 | r 39.36, centre +0.15 mm — inside its test tolerances |

The failed frame is now `tests/fixtures/extrusion/ring2/` (depth + K + T + applied
recipe, no colour), and `tests/test_extrusion_measure.py` renders synthetic board
patches (`_board_bias_patch`) so the failure mode is covered without a cell.

**Offline reprocessing now measures against the take's own plan.** `reprocess_saved_layer`
rebuilt the plan from `trial.json`, but a measure-only session is created *before*
Characterize → Apply, so `trial.json` carries the pre-Apply recipe and centre (r 40 at
(212.1, 149.7) for a take measured at r 42.6 about (214.6, 146.7)) — reprocessing from
it reproduced the stale-plan artifact. It now uses `manifest.recipe`, the centre fitted
from the archived `nominal_path.json`, and the take's own provenance (intrinsics and
processing config live on the manifest for measure-only takes; the trial-level copy is
the live-print fallback), and it carries `geometry` into the rewritten manifest.
The 20:48 take reprocessed to **r 42.31, centre offset 1.28 mm, mean/RMS/max
1.51/1.85/3.91 mm, shape RMS 1.58, completeness 0.992** — the first zero-offset
baseline take, recovered with no robot time.

### Figures (2026-08-28)

`tasni/modules/extrusion/figures.py` renders six figures per take from the archive
alone — no robot, no RoboDK, no camera — at 300 dpi PNG plus vector PDF:

| figure | what it shows |
|---|---|
| `plan` | deposit cloud, extracted centreline, nominal ring, mm axes, scale bar |
| `heightmap` | bird's-eye relief of the re-projected depth frame with a z colourbar |
| `mesh` | the frame SURFACED: work surface and deposit, each from above and rotated |
| `iso` | oblique cloud + centreline, vertical exaggeration stated on the axis |
| `profile` | unrolled height z(θ) and radial deviation Δr(θ) over 360° |
| `pipeline` | the method figure: the six arrays the chain held, in the order it held them |

plus two per trial: `figures/stack.{png,pdf}` (every layer's latest take, plan +
oblique) and `figures/tube.{png,pdf}` (the commanded bead against the measured
footprint). This is the successor to the original `PostExtrusionToolpath`
`overlay.png` (alpha shape → skeletonize → matplotlib) whose 3-D companion was only
ever `plt.show()`n.

`RingMeasureJob` draws them after the manifest is written and OUTSIDE the camera
hold; a drawing failure is logged and the measurement stands. Serving is
render-if-missing, so takes archived before this existed still produce figures:

```
GET /api/modules/extrusion/trials/{id}/layers/{dir}/figures/{plan|heightmap|mesh|iso|profile|pipeline}.{png|pdf}
GET /api/modules/extrusion/trials/{id}/layers/{dir}/files/{color|comparison|segmentation|skeleton|side}.png
GET /api/modules/extrusion/trials/{id}/figures/{stack|tube}.{png|pdf}
```

In the app, click a take in the measurement table to open its gallery; the
bird's-eye now draws the measured centrelines (red) over the commanded ones (teal).
Needs `pip install -e .[figures]` (matplotlib); without it measurements run
unchanged and only the figures are skipped.

Two things the first real capture forced, both of which would silently corrupt a
paper figure:

- **The colour range comes from the deposit band**, not the whole frame. D435i
  dropouts hundreds of mm below the work plane otherwise own the scale and the
  ring flattens to one colour (measured: a −45…+5 mm scale instead of −1…10 mm).
- **The nominal centre is FITTED, not averaged.** The archive writes a closed ring,
  so its first point repeats and the mean is biased by radius/N — 0.33 mm on the
  cell's 181-point 40 mm ring, which made the plotted RMS (11.45) disagree with the
  manifest (11.31) a reader checks it against. They now match to 1e-6.

### The surfaced view (`mesh`, 2026-08-29)

The old paper's mesh pictures — the frame as a surface, seen from the top and then
rotated — came from `macros/3DScan.py`, which builds a Poisson mesh and hands it to
`o3d.visualization.draw_geometries` (`macros/3DScan.py:354-358`). That window is
interactive: it was rotated and screenshotted by hand, and the code wrote no image;
the mesh itself went to a `TemporaryDirectory` only to be `RDK.AddFile`d into the
station. The ring-stack chain never meshed at all — it goes depth → cloud → ROI →
cluster → crest → raster → skeleton → spline — so there was nothing to render.

`mesh` restores it as a file, from the archive, with no display: one row per
surface (work surface, then deposit only), each drawn from above and rotated.

- **2.5-D, not a solid.** One top-down RGB-D frame measures a height field; closing
  it into a solid would invent the underside the camera never saw.
- **Gaps stay gaps.** The mesh is gridded and triangulated in place, so a cell with
  no return has no vertex and no triangle can cover it. A convex/Delaunay hull would
  roof over the ring's hole. Triangles spanning more than 25 mm of height are dropped
  so a dropout cliff is not bridged into a wall that was never there.
- **The pitch comes from the cloud.** A grid finer than the cloud's own spacing gives
  isolated cells and a scatter of specks (26 triangles from the 1517-point bead);
  `_auto_cell` coarsens until cells average a few returns. The deposit lands at
  ~3 mm because the chain voxel-downsamples at 2 mm — that IS the measurement's
  resolution.
- **The deposit panel reads `radial_trimmed`, not the archived cloud.** The archive
  stores the CREST the centreline was thinned from (578 points on the first cell
  ring, flanks already discarded); the chain's own deposit cluster is the same bead
  with its sides on, which is what closes into a surface with a width.
- **The stated exaggeration is the real one.** `set_box_aspect` stretches Z whatever
  the data says — a flat `(1, 1, .55)` box exaggerated this bead ~6× while the axis
  claimed ×2. The box now carries the data's own proportions, so the ×N on the axis
  is what the picture does.

Cost: ~9 s per take at 300 dpi (against `pipeline`'s ~34 s), drawn eagerly with the
rest. Both figures re-run the chain, so `take_stages` memoises the LAST take, keyed
on the manifest's mtime — a reprocessed take is never served from the cache.

Cell protocol: scan surface applied → Center on scanned surface → Generate → place
ring 1 → Characterize → Apply → Generate → Measure L1 ×5 (noise floor) → re-place
×3 → ring 2 true → Measure L2 ×3 → ring 3 true → Measure L3 ×3 → mark where ring 3
sits, shift it 10 mm (type it in) → Measure ×3 → 15 mm from the SAME marks → Measure
×3 → Paper summary. Keep offsets ≤ 25 mm (radial ROI ±30 mm). Expected for a pure
shift d: offset ≈ d, max ≈ d, mean ≈ 0.64 d, RMS ≈ 0.71 d. The shift is scored
**paired** against the layer's last undisplaced take (so the L3 "true" takes must
precede it) — `docs/pfh-paper-handoff.md` §3.

Proven on synthetic RGB-D before any robot motion (`tests/extrusion_synthetic.py`
renders rings through the cell's real 720p intrinsics from the pose the job
derives): a true 60 mm ring reads mean 0.80 / RMS 0.92 mm; the same ring shifted
10 mm in +X reads offset 9.92, max 10.96, mean 6.39, RMS 7.12 mm; a stacked ring
shifted 10 mm over a true ring 1 reads 10.17 mm WITH the per-layer floor and
exhausts the branch guard without it.

The first real depth capture and its transforms are stored (npz-compressed) in
`tests/fixtures/extrusion/ring1/`; the regression test reprocesses it to the
corrected metrics above. Colour is intentionally omitted because this selector is
geometric and must not depend on material colour.

## Exact operator retry sequence

1. Refresh the Cylinder Test page and connect/refresh the RoboDK station.
2. Select the actual print tool and inspection tool. Leave **Derive the inspection
   pose from the cylinder** checked (an inspection target is only needed if you
   clear it).
3. Place the path, preferring the scanned surface:
   - **Preferred:** if the Scan surface row is not green, run the Scan module and insert
     its result, then click **Center on scanned surface**.
   - **Manual alternative:** jog the selected print TCP to the intended first point on
     the circle and click **Seed path start from current TCP**.
4. Review center X/Y, build-plane Z, and RPW. If using `World`, confirm these are
   deliberately world coordinates.
5. Generate coordinates and fingerprint.
6. Run geometry/station preflight. It must show all sampled IK poses reachable, and the
   placement section must not report an overhang or a stale scan.
7. Run the complete RoboDK dry run. Do not proceed live until collision validation,
   simulation, inspection motion, and return-to-start all pass.

Changing any recipe/setup value invalidates the fingerprint and prior checks, as does
re-scanning the surface a plan was centred on.

## Most recent live verification

A read-only current-TCP seed was tested with:

- print tool: `spindle`
- frame: `World`
- radius: 40 mm
- one layer
- inspection tool: `Realsense`
- inspection target: `NEUTRAL`

Results:

1. Geometry preflight passed.
2. All 9 sampled fixed-orientation path poses had IK solutions.
3. Native curve and Curve Follow program generation succeeded; status -5 did not
   recur.
4. Collision validation then stopped at 2.6%:

```text
Collision detected
Program: TasniCylinder_DRY_00b6aef849_L001
Instruction 5: MoveJ 1
```

This is the current next problem if the operator uses that same spindle/World seed:
inspect the generated approach move and the station collision pair. Do not bypass or
disable collision checking merely to obtain a pass. Likely adjustment points are the
chosen start placement, exact print-tool orientation, approach clearance, or an
incorrect/stale collision model. Confirm the physical cell before changing any
collision-map configuration.

The above audit ran immediately before the final failed-artifact retention change was
loaded, so that specific test artifact was cleaned. On the next failed dry run, the
native path artifacts remain for inspection as described below.

## Failed-artifact lifecycle

RoboDK items owned by this module use the `TasniCylinder_` prefix:

- `<program>_Curve`: dense native curve object.
- `<program>_Settings`: Curve Follow/Robot Machining Project.
- `<program>`: linked generated robot program.
- `<program>_Inspect`: inspection program, when created.

Behavior after `a0670f1`:

- Successful dry runs clean temporary native path artifacts.
- Failed curve generation retains curve/settings (and any linked program).
- Failed program/collision validation retains curve/settings/program for inspection.
- Dry-run mock I/O programs are always deleted, including on failure.
- **Reset / clean RoboDK path** removes stale `TasniCylinder_` artifacts.
- Existing user items outside that namespace are not removed.
- A retry with the same generated item names removes/replaces those exact stale items.

The curve may initially appear beneath `World` during `RDK.AddCurve`, then it is
parented to the selected frame. If the selected frame is `World`, remaining beneath
World is expected. The dense path is deliberately a curve, not a target list.

## Valve, approval, and license facts

The verified legacy mapping from `231006_RoboArchPaper.rdk` is:

```text
AirOn:  Set IO_508=1; Set IO_601=1
AirOff: Set IO_508=0; Set IO_601=0
```

The local ignored `tasni.config.json` currently has the operator's hardware-I/O
approval active. Approval is a local safety interlock and is distinct from mapping
discovery, RoboDK licensing, geometry feasibility, IK, and collision validation.

An earlier misleading free-license message came from a private RoboDK process launched
with `-SKIPINI`; it did not load the user's normal license settings. The setup tool was
fixed to use a licensed isolated instance without `-SKIPINI`. Do not diagnose status
-5 as a license issue.

Dry runs call generated mock programs whose instructions are comments only. Physical
outputs remain blocked. Live jobs additionally force `AirOff` on startup, around layer
boundaries/faults, and before inspection/return behavior.

## RoboDK API design decisions

Commit `e36b4d5` replaced target-per-point generation with the native manufacturing
workflow:

1. `RDK.AddCurve` receives Nx6 `[X,Y,Z,I,J,K]` vertices with no projection.
2. The curve is parented to the selected work frame.
3. `RDK.AddMachiningProject` creates a Curve Follow Project.
4. `setPoseFrame`, `setPoseTool`, project pose/joints, and
   `setMachiningParameters(part=curve)` define generation context.
5. Machining parameters configure process/rapid speeds, blending, approach/retract,
   and `CallPathStart`/`CallPathFinish`.
6. `UpdatePath` and project `Update(COLLISION_OFF)` generate the linked program.
7. The path-to-tool seed is COMPUTED (`curve_follow_seed_T`) so RoboDK generates
   the commanded rotation: a Curve Follow Project mirrors the roll it is seeded
   with, and inverting that mirror is what removed the axis-4 wrist flip. The
   generated program is then kept as emitted — **zero station targets** — and
   verified: per-instruction pose error, the interpolated wrist-flip sample check,
   valve-call placement, and an unchanged station target count.
8. Program `Update(COLLISION_ON)` is the authoritative complete validation.

Official references used during implementation:

- <https://robodk.com/doc/en/Robot-Machining-Curve-Follow-Project.html>
- <https://robodk.com/doc/en/PythonAPI/examples.html#points-to-curve>
- <https://robodk.com/doc/en/PythonAPI/examples.html#robot-machining-settings>
- <https://robodk.com/doc/en/Robot-Machining.html>

## Verification and runtime state

After the scan-surface placement change:

- Full Python suite: 244 passed.
- Extrusion-focused suite: 19 passed.
- Frontend TypeScript check and production build: passed (existing chunk warning only).

At commit `a0670f1`:

- Full Python suite: 234 passed.
- Extrusion-focused suite: 14 passed.
- Frontend TypeScript check: passed.
- Frontend production build: passed (only the existing chunk-size warning).
- Backend was restarted after the final changes and reported idle with no plan.
- Vite dev server remained running and picked up frontend changes.
- No `server/` files changed, so no Jetson deploy/restart was required.
- Commit `a0670f1` was pushed to `origin/calibration-improvements`.

Useful verification commands:

```powershell
py -3.10 -m pytest tests/test_extrusion.py tests/test_extrusion_job.py -q
py -3.10 -m pytest -q
cd tasni/webui
npm run typecheck
npm run build
```

## Safe next-agent priorities

1. Ask the operator to retry with the path placed on a freshly scanned surface
   (**Center on scanned surface**) rather than the `spindle`/`World` seed that produced
   the collision below. A circle centred on the measured table sits in the reachable
   work area by construction; world zero did not.
2. If preflight rejects a sampled coordinate, use the returned frame/XYZ and inspect
   the exact setup; do not start dry run.
3. If Curve Follow generation or collision validation fails, inspect the retained
   `TasniCylinder_*` curve/settings/program in RoboDK before Reset.
4. For the observed `MoveJ 1` collision, identify the collision pair and distinguish a
   real approach hazard from stale/oversized station geometry. Preserve fail-closed
   behavior.
5. Only after a complete dry-run pass should the operator enable the already-approved
   live workflow and explicitly confirm the run.

