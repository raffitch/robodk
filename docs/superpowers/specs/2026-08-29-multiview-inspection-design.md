# Multi-view inspection + side photo — design

**Status:** design approved in principle by the operator on 2026-08-29 ("I still want to
push"), written for implementation by a follow-up session via `superpowers:writing-plans`.
**Scope:** the ring-stack **measure-only** experiment and its paper export first; the live
print reuses the same capture function later (§10). **Both features are optional toggles
and default OFF**, so the validated single-view chain and every number already archived
stay exactly as they are.

Background you need before touching this: `docs/pfh-paper-handoff.md` (the experiment,
its traps, the paired detection error), `docs/superpowers/specs/2026-08-27-ring-stack-measure-only-design.md`
(the chain this extends), and the memory/notes on the 2026-08-13 cell characterization
(incidence vs distance costs).

## 1. Why this exists

The mock extruded rings are **thin** (crest 2–4 mm above the board over much of the ring;
the cell's bare-board depth noise reaches +4.8 mm p99 at 300 mm). One straight-down RGB-D
frame sees the crest at grazing signal-to-noise and the flanks hardly at all, so the
deposit cluster is sparse and the centreline comes from a handful of points per degree.
Repeating frames from the same pose only averages the same missing flanks.

The operator wants two additions, both as optional toggles:

1. **Multi-view capture** — the top view as today plus **three tilted views at 120°
   azimuth spacing** (a "Mercedes star", i.e. one ring of the scan module's dome at a
   single cone angle), **merged into one work-frame point cloud** that feeds the existing
   reconstruction. Goal: more of the bead (flanks, both sides of the crest), denser
   sampling, a more accurate ring.
2. **Side photo** — one RGB frame per ring taken with the camera pitched to look at the
   bead **from the side**, near-horizontally, so the paper can show what the layer looks
   like in profile. Documentation, not measurement.

### What multi-view will and will not buy (say this in the paper too)

- It **does** add the flanks: a view tilted by θ sees the near flank at incidence reduced
  by θ, and three azimuths put every flank point within 60° of some camera. For a thin
  bead the flanks ARE most of the bead, which is why the top view struggles.
- It does **not** raise lateral resolution (fixed by the sensor at ≈0.4 mm/px at 300 mm,
  and 300 mm is the D435i depth floor at 1280×720) and it does not average depth noise
  unless the views are registered to better than that noise (≈0.65–0.9 mm RMS at 310 mm).
  Hand-eye board consistency is 1.26 mm and the scan-chain audit found a ≈2 % lateral
  scale mismatch (Jetson aligns depth with factory intrinsics, host back-projects with
  calibrated ones). **Left uncorrected, views from opposite sides disagree by 1–2 mm at
  the ring and a naive merge blurs it.** §4.3 is the registration step that makes the
  merge safe; §8 is the on-cell A/B that decides whether merged takes go in the paper.
- Tilt costs board quality: at ≈310 mm the 2026-08-13 characterization measured plane RMS
  0.65 mm at 1°, 2.0 mm at 9°, 5.0 mm at 20°. Part of that is systematic (a scale/tilt
  warp that grows with tilt) and the per-view **levelling** in §4.3 removes exactly that
  part; the rest is why the tilt stays modest (default 20°, cap 30°).

## 2. What exists today (verified in code, `main` @ `f4f06e6`)

- `inspection.py`: `pose_from_aim(aim, standoff, tilt_deg, azimuth_deg, roll_deg,
  reference_x)` already builds a camera pose on a cone about the surface normal with the
  aim point on the optical axis at the standoff. `pose_candidates` orders roll → tilt →
  azimuth fallbacks; `inspection_plan` publishes descriptors for preview.
- `service._build_inspection_move(rdk, plan, layer, inspection_name, config, camera,
  start_joints, seed_pose, collisions, near_mm)` walks the candidates: creates a target,
  a one-move program, validates it (`update_program(collisions=…)`), checks the wrist
  branch, returns `{artifacts, validation, target, pose}`. **One program = one pose.**
- `measure._inspect_and_capture` = build move → run → wait → settle → re-select
  tool/frame → `rdk.camera_pose_T()` → `camera.grab(with_depth=True)` → one frame.
  `RingMeasureJob` calls it once, processes, archives, returns home; timings:
  `move_to_pose_ms, settle_ms, capture_ms, total_ms, return_ms → inspection_cycle_ms`.
- `processing.process_observation(color, depth, T_work_camera, K, plan, layer, config,
  floor_profile, stages)`: back-project → height+radial ROI → deposit cluster → radial
  trim → top surface (upward normals) → raster/thin/prune → spline → `compare_circle`.
  Everything after the first line operates on **work-frame points**; that is the seam.
- `archive.write_layer` stores `color.png`, `depth.npy`, paths, metrics; the manifest's
  `provenance.T_work_camera` + `camera_intrinsics` + `processing_config` make
  `service.reprocess_saved_layer` reproducible offline.
- `figures.py` renders plan/heightmap/iso/profile/pipeline per take and stack/tube per
  trial from the archive; `paper_docx.py` assembles the Word draft from `paper_summary`.
- `tests/extrusion_synthetic.py` renders a depth image of analytic rings from **any**
  `T_work_camera` (`render_scene`), so multi-view scenes need no new renderer.

## 3. Operator-facing behaviour

Ring-stack card gains two checkboxes (persisted in `localStorage`, echoed in the
"This press records: …" line and in the run log):

- **Multi-view (top + 3 tilted)** — off by default. On: every Measure / Characterize press
  captures four views and measures the merged cloud.
- **Side photo** — off by default. On: after the measurement views the arm returns home,
  then makes a **second, separately timed excursion** to the side pose, takes one RGB
  frame, and returns. The photo never affects the numbers; a failure of the side excursion
  never invalidates the take.

Preflight (the API's plan preview) lists the extra poses per layer with the same
reachability descriptors as today's candidates, so an unreachable side pose is known
before the arm moves.

## 4. Design

### 4.1 Poses (`inspection.py`)

**Multi-view star.** For a layer with aim point `aim` and framing standoff `s`
(unchanged), the view set is

| view | tilt | azimuth | notes |
|---|---|---|---|
| `top` | 0 | – | exactly today's candidate walk (roll → tilt → azimuth fallbacks) |
| `star-0`, `star-120`, `star-240` | `multiview_tilt_deg` | `multiview_azimuth_offset_deg + k·120` | one candidate walk **per view**: roll candidates first, then azimuth ±`multiview_azimuth_slack_deg` (default 20°) in 10° steps, then tilt −5° steps down to `multiview_tilt_min_deg` (10°). Never 0° (that is the top view again). |

Same standoff for every view, so the aim point stays on the optical axis at the same
distance and the ring fills the frame identically. Azimuth is measured in the work
frame from +X (the paired-detection axis), so the star's orientation is reproducible
across takes; the roll reference is the camera-as-parked axis, exactly as today.

`star_view_candidates(aim, standoff, config, reference_x) -> list[{name, candidates}]`
returns the ordered candidate list per view; `inspection_plan` gains
`layers[i].views` with the descriptors (no 4×4s) when multi-view is requested.

**Side pose.** Target the **near crest** of the ring, not its centre:

```
outward = (cos az, sin az, 0)          az = side_view_azimuth_deg, in the work frame
crest   = centre + R·outward + (0, 0, layer_top_z)
elev    = max(side_view_elevation_deg,
              asin((side_view_min_camera_z_mm − layer_top_z) / side_view_standoff_mm))
camera  = crest + standoff·(cos elev · outward + sin elev · ẑ)
optical axis = crest − camera;  image "up" = ẑ projected ⊥ axis  (upright photo)
```

Defaults: azimuth 0° (frame +X, fallbacks 90/180/270 in that order), elevation 15°,
standoff 250 mm, `side_view_min_camera_z_mm` 80 mm. The elevation floor exists because
the camera housing (≈25 mm tall about the optical centre) and the tool flange behind it
would otherwise sit at bead height, i.e. on the table; the derived `elev` is reported in
the pose descriptor so a photo that is not "perpendicular" says by how much. Roll
fallback 180° (image upside down; the figure step flips it back using the recorded roll).
A pure side-on photo (`side_view_elevation_deg = 0`) is allowed only when the floor
permits it — the formula, not the operator, decides. The derived elevation is capped at
45°: above that it is no longer a side view, and preflight refuses the pose with the
number rather than photographing the crest from above.

`side_view_pose(recipe, setup, layer_index, config) -> dict` and
`side_view_candidates(...)` mirror the existing API. The RGB lens focuses fine at 250 mm;
depth is irrelevant here (the D435i's 280 mm MinZ only limits depth), so this standoff
is **not** clamped by `inspection_min_mm`. `side_view_standoff_mm` ≥ 150.

### 4.2 Capture (`measure.py`)

`_inspect_and_capture` becomes the single-view primitive it already is, and a new

```
capture_views(services, ctx, plan, layer, *, program_stem, start_joints, seed_pose,
              collisions, artifacts, views: list[str], frames_per_view: int) -> list[View]
```

runs it once per requested view, **in the order top → star-0 → star-120 → star-240**,
seeding each view's candidate walk with the previous pose (fewest wrist changes). Each
view is its own RoboDK target + program (`<stem>_Inspect_<view>`), created, validated and
deleted exactly as today. `View` = `{name, descriptor, T_work_camera, color, depth,
frames, move_ms, settle_ms, capture_ms, error}`.

- `frames_per_view` (default 1). When > 1 the depth frames are combined **per pixel by
  the median of valid (non-zero) samples** and the last colour frame is kept; the manifest
  records the count. This is the cheap lever for depth noise at a fixed pose and is
  independent of multi-view; exposed in config, not in the UI.
- A star view whose candidate walk finds no reachable pose, or whose capture fails, is
  recorded with `error` and skipped; the take continues. The **top view failing fails the
  take** (as today), because it is the reference every other view is registered to.
- The camera lease (`_camera_hold`) spans the whole set; the RoboDK artifacts list
  accumulates every view's target/program for the `finally` cleanup.

**Side excursion** (`capture_side_view(...)`): runs **after** the measurement views AND
after `rdk.move_j_joints(start_joints)` has returned the arm home, so the measurement
excursion's `return_ms` is the same trip it is today. It builds its own program
(`<stem>_Side`), validates with `collisions = side_view_collision_check` (config, default
**True** — this pose is low and near the table; the measure-only default of OFF for the
top view exists because RoboDK's check rejected good *overhead* poses against furniture,
which does not apply here), moves, settles, grabs one frame (colour + depth, depth
archived for completeness only), returns home. Timed as `side_view_excursion_ms` (out +
settle + capture + back). Any failure is logged and stored as `side_view.error`; the
take's validity is untouched.

### 4.3 Merge and registration (`processing.py`, new `multiview.py`)

Refactor first, behaviour-preserving:

```
observation_points(depth, K, T_work_camera) -> points            # the first line of today
process_points(points, *, plan, layer, config, floor_profile, stages, counts) -> ProcessingResult
process_observation(...)  = process_points(observation_points(...), ...)   # unchanged API
```

Every existing test keeps passing through `process_observation`. Then:

```
merge_views(views: list[ViewPoints], *, plan, config, diagnostics) -> np.ndarray
```

with `ViewPoints = {name, points (work frame), T_work_camera}`. Steps, per non-top view:

1. **Levelling** (removes the tilt-dependent systematic). Take the view's points in the
   board annulus `R + radial_roi_margin + 10 … R + multiview_level_annulus_mm (default 90)`
   with `|z| < multiview_level_max_abs_z_mm (15)`; fit a plane by least squares with one
   round of 2.5 mm outlier rejection; apply the rigid transform that maps the fitted plane
   to `z = 0` about the annulus centroid (rotation about the in-plane axis + z shift).
   Record `levelling = {tilt_deg, dz_mm, rms_mm, points}`. Skip (and warn) when fewer than
   `multiview_level_min_points (2000)` survive. Apply to the top view too — it is the
   reference, but a levelled reference makes the diagnostic comparable across views and
   the single-view chain already assumes the build plane is `z = 0`.
2. **XY alignment** (removes the hand-eye/scale disagreement between views). Run the
   view's OWN ring extraction as far as the radial trim (`_filter_deposit` + `_radial_trim`
   on the view alone, floor profile included) and fit a circle to the trimmed cluster;
   translate the view in XY so that centre coincides with the top view's cluster centre
   found the same way. Record `xy_shift_mm = [dx, dy]`, `ring_points`. A view whose ring
   cannot be found is merged **unshifted** (hand-eye only) with a warning, unless
   `multiview_registration = "strict"`, in which case it is dropped.
   `multiview_registration = "none"` skips this step (pure hand-eye merge) — that is the
   A/B control in §8.
3. **Merge**: concatenate the top view and every used view; optional voxel thinning
   `multiview_voxel_mm` (default 0 = off — the chain's raster is 1 mm/px and thins
   itself). The merged cloud goes through `process_points` unchanged; the report gains
   `merge = {registration, views_used, views_dropped, merged_points, merge_ms}` and
   `views = [{name, backprojected, ring_points, levelling, xy_shift_mm, used, warning}]`.

Why centre alignment rather than ICP: the ring is a torus, so ICP has an unconstrained
yaw and needs a good initial guess; a circle fit is exact for the shape we know we have,
costs nothing, and its correction vector **is the inter-view registration error** we want
measured. Why the top view is the reference: it keeps the merged centre continuous with
every single-view number already archived, so paired detection errors across capture
modes remain comparable. Known residual: a per-view scale error inflates that view's
radius, so merged bead width may read up to ≈2 % of R (≈0.8 mm) wide until the intrinsics
alignment on the Jetson is fixed; report, do not hide.

`characterize_ring` gets the same treatment: `characterize_points(points, …)` with
`characterize_ring` as the single-view wrapper; multi-view Characterize merges with
registration `"none"` first (no recipe centre yet), then re-registers about the fitted
centre and refits — one extra pass.

### 4.4 Archive and manifest (`archive.py`, `models.py`)

Layer directory layout, additive:

```
layer-00N[-takeMM]/
  color.png depth.npy               ← the TOP view, exactly where they are today
  views/star-0/{color.png,depth.npy,pose.json}
  views/star-120/… views/star-240/…
  merged_points.npy                 ← the cloud the chain measured (multi-view only)
  side/{color.png,depth.npy,pose.json}
  figures/…                         ← + views.png/pdf, side.png/pdf
```

`LayerManifest` gains

```
capture: {"mode": "single" | "multi",
          "frames_per_view": 1,
          "views": [{"name": "top", "descriptor": {...}, "T_work_camera": [[..]],
                     "color_file": "color.png", "depth_file": "depth.npy",
                     "move_ms": .., "settle_ms": .., "capture_ms": .., "used": true,
                     "error": null}, ...]}          # absent/None on old takes = single
side_view: {"color_file": "side/color.png", "depth_file": "side/depth.npy",
            "descriptor": {...}, "T_work_camera": [[..]],
            "excursion_ms": .., "error": null} | None
```

`processing.timings_ms` gains `merge_ms` and `views_capture_ms` (sum over views);
`acquisition_to_path_ms = views_capture_ms + total_ms` (capture of *all* measurement
views to path — that is what the paper's requirement 3 means when four views are taken);
`move_to_pose_ms` becomes the sum of the view-to-view moves. `inspection_cycle_ms` is
unchanged in meaning: departure → last measurement view → processing → home. The side
excursion is **not** in it; it is `side_view_excursion_ms`.

`write_layer(..., views=..., side_view=..., merged_points=...)` writes the extra files;
`provenance.T_work_camera` keeps the top view's pose so old readers stay correct.

### 4.5 Reprocess (`service.reprocess_saved_layer`)

Detects `capture.mode == "multi"`, reloads every `used` view, rebuilds the merge with the
take's own processing config (registration setting included) and re-runs
`process_points`. New parameter `views: "as_captured" | "top_only"` so any multi-view take
can be re-measured from its top view alone — the offline half of the A/B in §8. The
top-only result is written as a sibling report (`report-top-only.json`), never over the
take's numbers.

### 4.6 Figures (`figures.py`)

- `views.png/pdf` (multi-view takes): plan view of the four back-projected, registered
  clouds in four colours over the nominal ring, with each view's `xy_shift_mm` and
  levelling tilt in the legend. This is the method figure for the merge and the picture
  that shows whether the views agree.
- `side.png/pdf` (side photo): the RGB cropped to the bead. The crop window is the
  projection, through `K` and the recorded pose, of the near-crest segment
  `±side_view_crop_deg (30°)` of the ring at heights `0 … layer_top_z + bead`, with a
  margin; a scale bar from `fx` and the aim distance (valid because the bead is at that
  distance); flipped upright when the roll fallback was used. Caption states the elevation.
- `pipeline.png` keeps working for multi-view takes (stages come from `process_points`).

### 4.7 Paper summary and Word draft (`measure.paper_summary`, `paper_docx.py`)

- Every timing statistic is grouped by **capture mode** (`single`, `multi (n views)`), so
  a 4-view excursion is never pooled with a 1-view one; `side_view_excursion_ms` is its own
  row when present. The condition table is unchanged (its numbers are per take regardless
  of how the cloud was captured); the per-take table gains a "Views" column.
- Method paragraph: when any valid take is multi-view, the capture sentence becomes
  "…captured one RGB-D frame from each of four viewpoints — one normal to the work plane
  and three tilted by {tilt}° at 120° azimuth spacing — levelled each view to the work
  plane, aligned them on the ring, merged them in the work frame, and reconstructed…".
  Numbers come from the takes, never from constants.
- Figures: `views.png` for the first valid multi-view take; `side.png` for the latest
  valid take of each condition that has one, captioned "Layer k, side view at {elev}°".
- Registration diagnostic sentence (results, working record): "Across n multi-view takes
  the tilted views were displaced from the top view by {mean ± sd} mm before alignment" —
  the honest statement of inter-view consistency.

### 4.8 API and UI (`module.py`, `Extrusion.tsx`)

`MeasureLayerBody` and the characterize body gain `views: Literal["single","multi"] =
"single"` and `side_view: bool = False`. The job records both in the manifest and the run
log's first line. Preflight returns the star and side descriptors. The card's two
checkboxes map straight onto them; the annotation echo reads e.g.
"layer 3 · stacked true · introduced offset (0, 0) mm · 4 views · side photo".

### 4.9 Configuration (`ExtrusionConfig`, defaults)

```
multiview_tilt_deg: 20.0            (10–30)      multiview_tilt_min_deg: 10.0
multiview_azimuth_offset_deg: 0.0   multiview_azimuth_slack_deg: 20.0
multiview_frames_per_view: 1        (1–7)
multiview_registration: "centre"    ("centre" | "strict" | "none")
multiview_level_annulus_mm: 90.0    multiview_level_max_abs_z_mm: 15.0
multiview_level_min_points: 2000    multiview_voxel_mm: 0.0
side_view_azimuth_deg: 0.0          side_view_azimuth_fallbacks_deg: [90, 180, 270]
side_view_elevation_deg: 15.0       side_view_min_camera_z_mm: 80.0
side_view_standoff_mm: 250.0        (≥150)       side_view_collision_check: True
side_view_crop_deg: 30.0
```

## 5. Error handling, summarised

| failure | effect |
|---|---|
| star view unreachable / capture error | view skipped with `error`; take continues; `views_used` < 4 recorded and shown |
| top view fails | take fails, raw archived (as today) |
| levelling has too few board points | view merged un-levelled, warning |
| view's own ring not found | merged unshifted (`centre`), dropped (`strict`), warning |
| merged cloud fails processing | every view's raw frame archived; Reprocess offers `top_only` |
| side excursion fails (pose, capture, motion) | `side_view.error`; take valid; log says so |
| side pose needs elevation above 45° to clear the floor | side view refused at preflight with the number |

## 6. Testing (proof before the cell)

Synthetic scenes come from `tests/extrusion_synthetic.py` rendered from each pose.

- **Poses**: star views keep the aim on axis at the standoff; azimuths 120° apart from the
  configured offset; fallback walk never yields tilt 0; side pose has the near crest on
  axis, upright image, camera z ≥ floor, elevation raised when needed, refusal above 45°.
- **Levelling**: inject a 1.0° tilt + 1.5 mm dz into one view's `T_work_camera`; after
  levelling the residual plane is < 0.1 mm / 0.05°.
- **XY alignment**: inject a 1.5 mm lateral error into a view; `xy_shift_mm` recovers it
  to < 0.2 mm; `registration = "none"` leaves it and the merged shape RMS grows.
- **Gain**: a thin wavy ring (crest 2.5 mm) rendered from four views yields more trimmed
  ring points than the top view alone and a radius within 0.3 mm of truth; the top view
  alone on the same scene is allowed to fail — that is the case we are building for.
- **Degradation**: one star view with no ring → warning, take still valid; two views
  missing → still valid, `views_used = 2`.
- **Job** (FakeRdk/FakeCamera): 4 targets + 4 programs created and deleted; manifest
  `capture.views` complete; timings sum as specified; toggles off → identical to today
  (the existing measure tests are the regression).
- **Side view**: separate excursion after the return home; `side/color.png` archived;
  a raised capture error leaves the take valid; collision flag as configured.
- **Reprocess**: multi-view take rebuilds to the archived numbers; `top_only` writes the
  sibling report and leaves the manifest alone.
- **Summary/docx/figures**: timing grouped by capture mode; method sentence switches;
  `views.png` and `side.png` rendered and listed; per-take "Views" column.

## 7. Non-goals

TSDF/mesh reconstruction (the chain measures points; a mesh adds nothing to the numbers —
a rendered mesh figure can come later from `merged_points.npy`); ICP; changing the
Jetson stream or depth profile; fixing the intrinsics alignment (separate, prerequisite
for sub-millimetre inter-view agreement); wiring multi-view into the live print (§10).

## 8. Cell protocol: the A/B that decides whether merged takes go in the paper

On ring 1 in place (the thin one), same placement, no offset, phase `noise floor`:
3 takes single → 3 takes multi (`centre`) → 3 takes multi reprocessed `top_only`. Compare
per take: trimmed ring points, radius, centre scatter, shape RMS, completeness, and the
per-view `xy_shift_mm` (inter-view error before alignment). Decide:

- merged centre scatter ≤ single AND ring points ≥ 2× → use multi for the paper takes;
- `xy_shift_mm` > 1.5 mm on average → the merge is carrying hand-eye error; keep
  `centre` registration and say so in the paper, or fix the intrinsics alignment first;
- no gain on points → the bead is not the limit; leave multi off and take the side photos
  only.

Timing takes for requirement 3 are reported per capture mode either way.

## 9. Implementation tasks (for `writing-plans`)

1. Config keys + `LayerManifest.capture/side_view` models (+ validation tests).
2. `inspection.py`: `star_view_candidates`, `side_view_pose/candidates`, `inspection_plan`
   `views` (+ pose tests).
3. `processing.py` seam refactor (`observation_points`/`process_points`,
   `characterize_points`) — behaviour-preserving, existing tests green.
4. `multiview.py`: levelling, centre alignment, merge, diagnostics (+ synthetic tests).
5. `measure.py`: `capture_views`, job wiring, timings, archive layout (+ job tests).
6. `capture_side_view` + archive + `side.png` figure (+ tests).
7. `reprocess_saved_layer` multi-view + `top_only`; `tools/multiview_ab.py` printing the
   §8 table from a session.
8. `paper_summary`/`paper_docx`/`figures`: capture-mode grouping, method sentence,
   `views.png`, side figures, per-take column (+ tests).
9. API bodies + preflight descriptors + UI toggles + Run guide text.
10. Docs: `pfh-paper-handoff.md` §3 (A/B step, side photo step), `AGENTS.md`, memory.

Estimate: ~2 agent-days. The paper's cell run on the single-view chain is **not** blocked
by this; the A/B in §8 is the first thing to run once tasks 1–5 land.

## 10. Later: the live print

`CylinderPrintJob` builds its inspection move at `service.py:636/944` with the same
`_build_inspection_move`; swapping in `capture_views` + `merge_views` there is mechanical
once the measure-only path is cell-validated. Not in this spec's scope because the print
loop is cell-validated and the paper does not need it.
