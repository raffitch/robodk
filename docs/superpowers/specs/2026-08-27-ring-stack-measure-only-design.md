# Ring-stack measure-only experiment — design (paper evidence)

Date: 2026-08-27. Status: approved by the operator; implementation plan in
`docs/superpowers/plans/2026-08-27-ring-stack-measure-only.md`.

## 1. Why this exists

The extrusion module was built for the **Prototypes for Humanity 2026 short paper**
("Real-time Error Mapping, Correction, and Archival …", Tchakerian / Hayek /
Daneluzzo). **Deadline 1 September 2026.** The co-author package
(`C:\Users\User\Desktop\desktop\RoboArch_to_PFH_Coauthor_Package\`) says the paper is
written and compliant (1,887 / 2,000 words) and is waiting for three numbers,
under the rule *"do not invent or estimate values that were not measured"*:

| Priority | Number the paper needs | Where it comes from after this work |
|---|---|---|
| 2 (highest value) | Geometric deviation nominal ↔ extracted centreline: mean abs / RMS / max, mm | `DeviationMetrics` per measured ring, plus the new **centre offset** and **shape RMS** |
| 3 | Scan-to-feedback time over ~10 runs, mean ± sd | new `capture_ms` + existing `total_ms` → `acquisition_to_path_ms` |
| optional | path-extraction success X/N; layer-height / bead-width variation | `valid` count; new **height profile** and **bead width** stats |

Priority 1 (count of recipes / trials in the legacy Firebase archive) is **not** in
scope — it does not come from this repo.

**The experiment is NOT a print.** The operator has dried, already-extruded rings in
hand (≈5–6 mm tall on average, wavy up to ~10 mm — "a snake around the
circumference"). They will place them on the scanned work surface **one on top of the
other by hand**, and deliberately misplace some to introduce a known deviation. No
material is extruded, no valve is switched, no material recipe is recorded.

Consequently what the paper can claim from this is a **controlled validation of the
sensing-and-comparison chain with a known introduced offset** — not the deposition
deviation of a printed cylinder. The handoff to the paper must keep that wording.

## 2. What exists today (verified in code, `main` @ `9ed159c`)

- `tasni/modules/extrusion/service.py:CylinderPrintJob` (`:460-690`) is one loop per
  layer: print program → valve OFF → `_build_inspection_move` → move → settle →
  `camera_pose_T` → `grab(with_depth=True)` → `process_observation` →
  `archive.write_layer` → next layer. **There is no pause to place a ring, and the
  print program would sweep the tool through a hand-placed ring.** It also requires
  `hardware_io_test_approved`, the quick-sim approval and calls `AirOn/AirOff`.
- `_build_inspection_move` (`service.py:167-260`) + `inspection.py` derive a
  collision-validated, wrist-gated viewpoint per layer (300 mm straight down, aimed at
  the cylinder axis at that layer's top). **Cell-validated 2026-08-27.** Reuse as-is.
- `processing.py:process_observation` (`:222-360`) does depth → work-frame points →
  ROI (`z ∈ [2.5 mm, nominal_top + 15 mm]`, radius ± 30 mm) → voxel → outliers →
  DBSCAN largest → upward normals → raster → thin → branch guard → ordered skeleton →
  3-D spline → `compare_circle`. The **ROI floor is layer-independent (2.5 mm)**, so
  layer N's cloud contains every ring below it; a displaced ring exposes a crescent
  of the ring beneath that can contaminate the skeleton.
- `comparison.py:compare_circle` measures **radial deviation from the NOMINAL centre**
  (`processing.py:328`), so a bodily shift of the ring counts as error — correct for
  the paper. A ring translated by `d` gives `mean = 2d/π`, `RMS = d/√2`, `max = d`.
- Archive (`archive.py`): `runs/extrusion/<trial>/trial.json` +
  `layer-NNN/{manifest.json, color.png, depth.npy, measured_path.json, …}`.
  `LayerManifest.mode` already exists (default `"LIVE_PRINT"`).
  `reprocess_saved_layer` re-runs processing from the archived raw frame offline.
- Tests: `tests/test_extrusion.py` (API/geometry), `tests/test_extrusion_job.py`
  (`FakeRdk`, `FakeCamera`, `fake_processing`), `tests/test_extrusion_processing.py`
  (helper-level only). **There is no end-to-end test of `process_observation` and no
  synthetic RGB-D renderer.** No layer has ever been archived from the cell
  (`runs/extrusion/*` hold `trial.json` only).
- Camera intrinsics at 1280×720: fx ≈ 889.9, fy ≈ 890.8 (`tasni.config.json`);
  at 300 mm one pixel ≈ 0.34 mm. Active surface: `runs/scan/active.json`
  (`Tasni Work Frame`, 424.8 × 297.4 mm, centre (212.4, 148.7), plane RMS 1.39 mm).
  Hand-eye calibration verdict `borderline`, board consistency 1.26 mm — this is the
  honest error floor; do not report sub-millimetre claims on top of it.

## 3. Operator protocol (what the cell run looks like)

1. Scan surface applied (already: `20260825-154713`). **Center on scanned surface**.
2. Place **ring 1** within ~50 mm of the table centre. Press **Characterize ring**
   (robot moves) → measured radius, bead width, height mean/min/max, centre. Press
   **Apply to recipe & placement** → recipe radius / bead / layer height and the
   cylinder centre are set from the physical ring; plan regenerated. No calipers.
3. **Noise floor**: Measure layer 1 five times without touching the ring.
   **Placement repeatability**: lift and re-place ring 1 three times, measure each.
4. Stack: ring 2 placed true → Measure L2; ring 3 true → Measure L3.
5. **Displacements** (type the value into *introduced offset* before pressing
   Measure so ground truth is archived next to the result): shift a ring **10 mm**
   along frame +X, then **15 mm**. Keep every offset ≤ 25 mm (the radial ROI is
   ±30 mm). Optional: prop one side of a ring to produce a height/tilt case.
6. Every measurement records its timing; ≥ 12 measurements give the paper's
   mean ± sd. Press **Paper summary** → per-condition table + timing.

Expected readouts for a pure shift `d` (the built-in sanity check): centre offset
≈ (d, 0); max ≈ d; mean ≈ 0.64 d; RMS ≈ 0.71 d. A 10 mm shift → 10 / 6.4 / 7.1 mm.

## 4. Design

### 4.1 Mode `MEASURE_ONLY`

A **measurement session** is one trial directory with `trial.json.mode =
"MEASURE_ONLY"` (plus `experiment.note`). Each **Measure** press is one *take* of one
layer: `layer-002` (take 1), `layer-002-take02`, … Each take archives exactly what a
live layer does (raw colour + depth, derived images, paths, manifest) so
`reprocess_saved_layer` works unchanged.

Per press, `RingMeasureJob` (new file `tasni/modules/extrusion/measure.py`) does:

```
measure_station_requirements (work frame, inspection tool, collision proxy; NO print tool, NO AirOn/AirOff)
camera readiness grab under the lease (same as live)
apply_run_mode("run_robot"); ensure_real_robot_link; start_joints = current_joints()
with _camera_hold("extrusion-measure"):
    inspect = _build_inspection_move(rdk, plan, layer, ..., start_joints, seed_pose=session.last_pose, collisions)
    start_program(inspect) real → _wait_program → sleep(settle_s)
    use_named_tool_frame(inspection_tool, work_frame); T_work_camera = camera_pose_T()
    t0 → frame = grab(with_depth=True) → capture_ms
    processed = process_observation(..., floor_profile=session.measured_top(layer_index - 1))
    archive.write_layer(manifest(mode="MEASURE_ONLY", take, annotation, geometry, timings incl. capture_ms))
finally: stop program if running; move_j_joints(start_joints); delete_items(artifacts)
```

**Never** calls `run_station_program`, never creates a layer/curve program, never
reads `hardware_io_test_approved`. `CylinderPrintJob` is **left untouched** (it is
cell-validated); the ~25 lines of inspect-and-capture are duplicated in `measure.py`
rather than refactored out of the live loop.

Gates for `/measure/layer`: a generated plan whose fingerprint matches; station
connected; `confirm_robot_motion: true`. **No** dry-run / quick-sim / preflight
gate — the only motion is the inspection move, which is collision-validated and
wrist-gated at execution exactly as in the live print.

Session state (`ExtrusionModule._measure_session`): `trial_id`, `takes` per layer,
`last_pose`, `tops` (measured spline per layer, latest take), `characterization`.
Persisted as `<trial>/session.json` after every take; `GET /measure/session` rebuilds
from disk so a backend restart does not lose the stack.

### 4.2 Processing additions (`processing.py`, `comparison.py`)

All are pure functions with unit tests; `process_observation`'s default behaviour is
unchanged when the new arguments are omitted.

- **Per-layer floor** — `process_observation(..., floor_profile: np.ndarray | None)`.
  When given (the previous layer's `measured_xyz`, N×3), a point is kept only if
  `z > z_floor(nearest floor sample by XY) + config.layer_floor_margin_mm` (default
  2.0). Nearest by XY via `cKDTree`, so a displaced ring still maps to the right
  local height. Layer 1 / no previous take → current behaviour
  (`deposit_min_height_mm`). Record `floor_source ∈ {"build_plane",
  "previous_layer_measured"}` and `floor_z_mean_mm` in the report.
- **Centre offset** — in `compare_circle`: `center_offset_mm = fitted − nominal`
  (dx, dy) and `center_offset_norm_mm`. **Shape error** — radii about the *fitted*
  centre minus the fitted radius: `shape_rms_mm`, `shape_max_mm` ("ring is not round"
  separated from "ring placed wrong"). Added to `DeviationMetrics` with defaults so
  old manifests still validate.
- **Height profile** — new `RingGeometry` model on `LayerManifest.geometry`:
  `top_z_{mean,min,max,std}_mm` from the measured spline z, and
  `height_{mean,min,max}_mm = top_z − reference`, reference = previous layer's
  `top_z` at the nearest sample (or the build plane for layer 1);
  `height_reference ∈ {"build_plane", "previous_layer_measured"}`.
- **Bead width** — `bead_width_profile(cluster_xyz, center, bins=config.bead_width_bins)`:
  per angular bin, radial extent = p97.5 − p2.5 of point radii on the largest cluster
  **before** the upward-normal filter (so the bead's flanks count); report
  mean/min/max and the per-bin array. Documented as the *XY footprint width* of
  the ring.
- **Timing** — the job injects `capture_ms` into `report.timings_ms` and sets
  `acquisition_to_path_ms = capture_ms + total_ms`.
- **Characterize** — `characterize_ring(color, depth, T_work_camera, K, *,
  search_center_mm, search_radius_mm, config) -> CharacterizationResult`:
  pass 1 (coarse): ROI = cylinder `search_radius_mm` (config default 150) around
  `search_center_mm`, `z ∈ [deposit_min_height, config.characterize_max_height_mm
  (40)]`; shared filter chain; `fit_circle_xy` on the largest cluster → coarse centre
  and radius; `bead_width_profile` → coarse bead. Pass 2: build a throwaway recipe /
  setup (radius, bead, centre from pass 1, `layer_count=1`, `layer_height` = coarse
  height) and run `process_observation` → refined centreline, radius, height
  profile. Returns `radius_mm, center_mm, bead_width_{mean,min,max}_mm,
  top_z_{mean,min,max}_mm, measured_xyz, images, report`.
  To share the filter chain, extract `_filter_deposit(points, config) -> (points,
  counts)` and `_top_surface(points, config)` from `process_observation`
  behaviour-preservingly.

### 4.3 API (`module.py`, prefix `/api/modules/extrusion`)

| Method/path | Body | Effect |
|---|---|---|
| `GET /measure/session` | – | current MEASURE_ONLY session (from memory or newest on disk) |
| `POST /measure/session/new` | `{note}` | new trial dir, `mode: MEASURE_ONLY` |
| `POST /measure/characterize` | `{confirm_robot_motion, collision_check_enabled}` | job `extrusion-characterize`; result stored in session |
| `POST /measure/apply-characterization` | – | recipe `radius/bead/layer_height` and setup `center_x/y` (`build_plane_z = 0`, keep `scan_run_id`) from the characterization; regenerate plan; invalidate preflight/sim/dry-run |
| `POST /measure/layer` | `{fingerprint, layer_index, annotation: {introduced_offset_mm: [dx,dy] \| null, note}, confirm_robot_motion, collision_check_enabled}` | job `extrusion-measure` |
| `GET /trials/{id}/paper-summary` | – | per-condition table (grouped by `annotation.introduced_offset_mm`), timing mean ± sd, height/bead stats, valid X/N, plus a ready-to-paste Markdown block |

`GET /trials`: `summary.total_trials` / `total_layers` count **only** `LIVE_PRINT`;
add `measure_only_trials` and `measure_only_takes`. Characterization archives under
`<trial>/characterize-NN/`.

Config additions (`ExtrusionConfig`): `layer_floor_margin_mm: 2.0`,
`characterize_search_radius_mm: 150.0`, `characterize_max_height_mm: 40.0`,
`bead_width_bins: 36`.

### 4.4 UI (`tasni/webui/src/pages/Extrusion.tsx`)

One new card, **"Ring stack — measure only (no extrusion)"**, under *Safety workflow*:

1. *Characterize ring 1* button (robot moves; confirm checkbox) → shows radius / bead /
   height / centre → *Apply to recipe & placement*.
2. Layer selector (1…N), *introduced offset X / Y mm* inputs, note, *Measure layer N —
   ROBOT MOVES* button, cancel.
3. Results table per take: layer, take, offset dx/dy/|d|, mean |dev| / RMS / max,
   shape RMS, radius, height mean/min/max, bead, `acquisition_to_path_ms`, VALID.
4. *Paper summary* button → renders the Markdown block from the endpoint.

### 4.5 Synthetic RGB-D fixture and tests (proof before the cell)

`tests/extrusion_synthetic.py` (a helper module like `test_calibration_synthetic.py`):
`render_ring_depth(K, size_px, T_work_camera, *, plane_z=0.0, rings: list[RingSpec],
noise_mm=0.5, seed=0) -> np.ndarray[uint16 mm]`. `RingSpec(radius_mm, bead_mm,
center_xy_mm, z_base_mm, height_fn: θ → mm)`. Cross-section = semi-ellipse of width
`bead` and height `height_fn(θ)`; sample θ×φ at ≤ 0.25 mm, plane at 1 mm over
±200 mm, transform to camera with `inv(T_work_camera)`, project with `K`, z-buffer
minimum depth per pixel, add Gaussian noise. Camera pose from
`inspection.pose_from_aim(aim, 300, reference_x=[-1,0,0])` with the real 720p `K`.

Tests (numbers are the acceptance criteria; tolerances allow voxel 2 mm + raster
1 mm/px + 0.5 mm noise):

| test | asserts |
|---|---|
| true ring at nominal | mean abs < 1.0, RMS < 1.0, radius ± 1.0, completeness ≥ 0.95, offset norm < 1.0, valid |
| shifted 10 mm in +X | offset ≈ (10, 0) ± 1; max 10 ± 1.5; mean 6.4 ± 1; RMS 7.1 ± 1 |
| wavy height 7.5 + 2.5 sin 2θ | top_z min ≈ 5, max ≈ 10 (± 1.5); std > 1 |
| stack of two, top ring shifted 10 mm, floor from ring 1's measured spline | offset ≈ 10 ± 1.5 (without the floor the blended answer is well under 10 — assert the floor is what fixes it) |
| characterize 60 mm / 8 mm ring when the recipe says 75 / 6 | radius 60 ± 1, bead 8 ± 2, height ± 1.5, centre ± 1 |
| bead width profile on a rendered ring | mean within ± 25 % of bead |
| `RingMeasureJob` with `FakeRdk`/`FakeCamera` | never `run_station_program`, never `create_extrusion_layer_program`; inspection created, started, waited; returns to `START`; artifacts deleted; manifest `mode == "MEASURE_ONLY"`, `take` increments, annotation stored, `capture_ms` present |
| API | `/measure/layer` refused without matching fingerprint / confirm; works without `hardware_io_test_approved` and without dry-run; `/trials` excludes MEASURE_ONLY from printed counts; paper-summary groups by introduced offset |
| reprocess | an archived take reprocesses to the same metrics |

After the first real capture, copy its `color.png` + `depth.npy` (+ `manifest.json`)
into `tests/fixtures/extrusion/ring1/` (npz-compressed) as a regression fixture.

## 5. Non-goals

No correction execution or claim; no material recipe UI; no multi-view / TSDF; no
change to `CylinderPrintJob` behaviour; no new Tasni module; no Firebase; no
Tailscale work (already shipped: `health.connection_route`, default camera IP
`100.123.63.127`).

## 6. Working rules for the executing session

- Work in a **worktree on branch `extrusion-ring-stack`**, never on `main`; merge
  `--no-ff` when green; commit + push every task.
- `py -3.10` (no `python` on PATH). Never round-trip sources through PowerShell
  `Get-Content`/`Set-Content` (mojibakes UTF-8).
- Don't run the full pytest suite. Targeted:
  `py -3.10 -m pytest tests/test_extrusion.py tests/test_extrusion_job.py tests/test_extrusion_processing.py tests/test_extrusion_measure.py -q`
  and `cd tasni/webui && npm run typecheck && npm run build`.
- The backend caches modules: **restart Tasni** before any cell test and check
  `/api/health` → `build.stale`.
- `ExtrusionConfig` is `extra="forbid"`: new fields need defaults; never remove one
  the operator's `tasni.config.json` may still carry.
