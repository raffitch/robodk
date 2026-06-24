# Handoff — surface-aware scan planning (survey → planner → execute)

**Status:** design locked, not started. Implement next session.
**Branch to work on:** `calibration-improvements` (where the scan module already lives).
**Author of design:** worked out interactively 2026-06-23; this file is the spec.

---

## 1. The problem we're solving

Today the scan picks a **fixed** standoff and a **fixed** cone/count regardless of the
surface. The live gate forces the operator to `scan.ideal_distance_mm` (500 mm,
`tasni/core/config.py:265`) and pose generation always uses `pose_count=12`,
`cone_half_angle_deg=40`, `roll_max_deg=30` (`config.py:300-303`). A 100 mm puck and an
800 mm tabletop get the same scan. That's the "arbitrary distance, not fit for any
surface" the user flagged.

**Goal:** derive standoff, view angles, view count, and voxel size **from the measured
surface** (its size, position, tilt, and shape), so every surface is captured at the best
quality the camera can give.

Note: the cone is *already* centered on the measured distance — `generate_scan_targets`
passes `look = reading.distance_mm` (`tasni/modules/scan/service.py:134`). So this is not
a rebuild; it's replacing the **fixed knobs** (target standoff, cone, count, voxel) with
**functions of the measured surface**, plus a live visual aiming aid.

---

## 2. Design decisions (locked — do not relitigate)

1. **Coverage vs accuracy is the central tension.** Depth error grows ~quadratically with
   distance (D435i good ~0.3–1.0 m; that's why `depth_max_m=1.5`). So "frame the whole
   surface in one shot" is the *low-quality* answer for big surfaces. The planner biases
   to the **closest accurate standoff**, not the farthest framing shot.
2. **Aim at the measured surface centroid, not the camera's optical axis.** This makes
   "centering" automatic — the operator only has to get the surface roughly in frame and
   squared up; the math recenters the cone on the real surface centroid.
3. **Survey tilt tolerance is tighter than the scan cone** (~5–8° vs 35°). An oblique
   survey foreshortens the extent and biases the centroid, so the *measurement* frame must
   be squared up even though the *scan* then samples a wider cone.
4. **Two modes, chosen by whether the surface fits the accurate band:**
   - **Quality mode** (fits): closest accurate standoff, aim at centroid, small cone,
     voxel scaled to standoff, full TSDF tour → mesh + frame + rectangle.
   - **Reference-rectangle mode** (doesn't fit, ≳ ~1 m): **no robot tour, no fusion,
     no mesh.** A single backed-off, fully-framed survey frame → plane → frame +
     min-area rectangle only. The rectangle/frame is what downstream programming needs;
     the mesh is a quality nicety we honestly drop for oversized surfaces.
   - Threshold is **derived** from `K` + the accurate band (≈1 m at the noisy edge,
     less at good quality), exposed as a human-readable config cap.
5. **Live depth-driven overlay = the measurement, rendered.** Per-frame plane fit draws a
   **1‑2‑5 adaptive metric grid** on the surface + the **detected outline** + a
   **fully-framed indicator** (red when the outline touches the image border, green when
   fully inside + squared + in range). What the operator sees *is* the SurveyMeasurement —
   no gap between HUD and captured frame.
6. **Overlay is client-drawn from vector coords in the gate payload** (normalized 0–1),
   not baked into the JPEG — same decision the code already made for HUD text
   (`module.py` `live_start` comment, AimHud.tsx).
7. **Grid spacing = 1‑2‑5 nice-numbers** (…10/20/50/100/200/500 mm…), chosen so the
   projected cell stays ~50–80 px on screen. Adaptive, always round metric values.
8. **Reference-rectangle path is strictly frame + rectangle** (no coarse mesh — decided).
9. **Tiling is deferred** but the planner returns `aims: list[AimPoint]`, so tiling later =
   "emit N aims," not a structural change (see §9).

---

## 3. Architecture — three pure stages + a stable contract

```
Survey (measure one squared-up frame)
        │  SurveyMeasurement
        ▼
plan_scan (decide standoff / mode / cone / count / voxel)
        │  ScanPlan{ mode, aims:[AimPoint], voxel_size_m, warnings }
        ▼
execute  (per aim → synthesize seed pose → existing generate_calibration_poses)
```

### Data contracts (new)

```python
@dataclass
class SurveyMeasurement:           # camera-frame measurement of ONE depth frame
    detected: bool
    standoff_mm: float | None      # plane distance (median along normal)
    tilt_deg: float | None         # plane normal vs optical axis (0 = fronto-parallel)
    tilt_b_deg: float | None       # KUKA B/C tool-rotation fix (reuse existing math)
    tilt_c_deg: float | None
    centroid_cam_mm: np.ndarray | None   # inlier centroid in CAMERA frame (mm)
    extent_mm: tuple[float, float] | None  # surface bounding-rect size (real-world mm)
    shape: str                     # "rect" | "circle" | "unknown" (from hull; advisory)
    fully_framed: bool             # inliers do NOT touch the image border (+ margin)
    fov_deg: tuple[float, float]   # (hfov, vfov) from K + size — recorded for the planner
    # --- overlay vectors, normalized 0..1 image coords (client draws these) ---
    outline_uv: list[tuple[float, float]] | None   # detected surface polygon/rect
    grid_uv: list[tuple[tuple[float,float], tuple[float,float]]] | None  # line segments
    grid_spacing_mm: float | None
    ok: bool                       # survey-gate pass (detected + in-range + squared + framed)

@dataclass
class AimPoint:
    point_base_mm: np.ndarray      # look-at target in robot BASE frame
    view_dir_base: np.ndarray      # desired camera forward (= -surface_normal in base)
    standoff_mm: float
    cone_half_angle_deg: float
    roll_max_deg: float
    n_views: int

@dataclass
class ScanPlan:
    mode: str                      # "quality" | "reference"
    aims: list[AimPoint]           # 1 today (quality); N later (tiling). [] for reference.
    voxel_size_m: float
    warnings: list[str]
```

**Execution wiring:** for each `AimPoint`, build a synthetic **seed camera pose**
(`position = point_base + standoff·normal`, `forward = -normal`, up = operator's current
camera up projected) and call the **existing** `generate_calibration_poses(seed_T,
count=aim.n_views, look_distance_mm=aim.standoff_mm, cone_half_angle_deg=...,
roll_max_deg=...)` (`tasni/modules/calibration/poses.py:38`). Because that function
computes `center = seed_pos + look·fwd`, a correctly-built seed makes `center` land
exactly on `point_base` — **so poses.py needs no change.** Reachability/collision/
`select_diverse`/capture/fuse downstream are all untouched. Concatenate poses across aims.

---

## 4. The math (put in the planner, pure + unit-tested)

**FOV from intrinsics** (no magic constants — read `cfg.camera.K`, `cfg.camera.size`):
```
hfov = 2·atan(W / (2·fx)),  vfov = 2·atan(H / (2·fy))
footprint at distance d (perpendicular):  w = d·W/fx,  h = d·H/fy
```

**Standoff to frame a surface (Sx, Sy) with margin m (~1.3):**
```
d_fit = max(m·Sx·fx/W,  m·Sy·fy/H)
standoff* = clamp(d_fit, accurate_min_mm, accurate_max_mm)
```
- Small surface → `d_fit < accurate_min` → clamp up to `accurate_min` (closest accurate
  distance; surface fills part of the frame — fine, resolution is set by voxel not framing).
- `d_fit ≤ accurate_max` → **quality mode** at `standoff*`.
- `d_fit  > accurate_max` → **reference mode** (can't frame at quality → rectangle only).

**Voxel scaled to standoff** (4 mm is coarse for a 100 mm part):
```
voxel = clamp(standoff_mm · k, voxel_min_m, voxel_max_m)   # k ≈ 0.008 → 4 mm @ 500 mm
```

**Cone + count by how 3D the surface is** (v1: a UI/config choice, not auto-detected):
- **flat top** (default): small cone (~15–20°), few views (~6–8). View count saturates
  fast on a plane — it mostly averages depth noise. Don't over-promise count.
- **raised object / pedestal sides matter**: wider cone (~35–40°), more views (~12–14) so
  oblique views catch side walls the nadir view occludes.

---

## 5. Mode selection & reference-rectangle short-circuit

`survey_surface` requires `fully_framed` to produce a trustworthy extent. Then:

- **quality:** proceed as today but with derived standoff/cone/count/voxel and centroid
  aim. One `AimPoint`. Full tour → fuse → `work_plane_from_points` → mesh + frame + rect.
- **reference:** the survey frame was captured fully-framed (operator backed off). Skip the
  tour and fusion entirely: back-project the **single survey depth frame** to base-frame
  points (camera pose × back-projected pixels), run the existing
  `work_plane_from_points` (`tasni/modules/scan/plane.py:200`) on those points, emit
  frame + rectangle. Accuracy on a ~1 m rectangle to ±cm is fine from one coarse frame.
  `ScanResult.mesh_obj_path = None`; UI labels it "reference surface (no fine mesh)".

This means `ScanCaptureJob` gains a fast path that never moves the robot for reference
mode — or, cleaner, reference mode is handled in `generate_scan_targets`/a new
`survey_locate(services)` that returns a `ScanResult` directly (no targets, no Run).
Decide during implementation; the simplest is a dedicated `reference_locate()` that
produces the same `ScanResult` shape so insert/review are unchanged.

---

## 6. Implementation phases

### Phase 1 — `tasni/modules/scan/survey.py` (pure, unit-tested) ⭐ start here
Full-frame surface segmentation from one depth frame (the center-patch `depth_gate.py` is
too local for extent). Reuse `plane.fit_plane`, `plane._oriented_rectangle`,
`plane._min_area_rectangle`.
- `survey_surface(depth, K, thresholds, *, depth_scale) -> SurveyMeasurement`:
  back-project the (downsampled) full depth frame to camera 3D → RANSAC plane → inliers →
  min-area rectangle → extent (real-world mm), centroid (camera mm), normal, tilt
  (reuse the B/C tilt-fix math from `depth_gate.evaluate_depth_gate:121-127`).
- **Border test** for `fully_framed`: do inlier pixels touch the image border (within an N-px
  margin)? If yes → not fully framed.
- **Overlay builder:** project the rectangle corners + a 1‑2‑5 metric grid (aligned to the
  rectangle axes) through `K` to image pixels, normalize to 0–1 → `outline_uv`, `grid_uv`,
  `grid_spacing_mm`. Pick spacing so the projected cell ≈ 50–80 px.
- Tests: synthetic planar depth (flat, tilted, off-center, partially-out-of-frame); assert
  extent, centroid, tilt, fully_framed, and that grid spacing snaps to 1‑2‑5.

### Phase 2 — `tasni/modules/scan/planner.py` (pure, unit-tested)
- `plan_scan(survey, K, size, scan_cfg) -> ScanPlan` implementing §4–§5.
- Tests: small/medium/large surfaces → expected mode, standoff clamp, voxel; large →
  reference mode with empty `aims`; verify the FOV/standoff formulas against hand calcs.

### Phase 3 — wire into the scan service
- `generate_scan_targets` (`tasni/modules/scan/service.py:73`): replace the fixed-config
  pose call (`:140-143`) with `survey_surface` → `plan_scan` → per-aim seed synthesis →
  `generate_calibration_poses` (concatenate). Aim at `centroid` (transform
  `centroid_cam_mm` to base via `rdk.camera_pose_T()`). Keep all the reachability/
  collision/`select_diverse` machinery (`:144-197`).
- Add reference-mode path (`reference_locate`) returning a `ScanResult` (frame + rectangle,
  `mesh_obj_path=None`) with no targets and no Run.
- Surface the plan in the return dict + logs (mode, standoff, voxel, cone, count, extent).

### Phase 4 — live overlay
- **Backend:** the scan survey gate must run **live** to drive the overlay. Use the existing
  interleave path (`livepreview.py` `depth_probe`, gated by `scan.live_depth_gate`). Point
  `depth_probe` at `survey_surface(...).to_dict()` so the `gate` event carries
  `outline_uv`/`grid_uv`/`fully_framed`. ⚠ **Honest caveat:** a depth grab over Wi-Fi is
  ~6–11 s, so the overlay refreshes only every interleave; the color video stays smooth in
  between and the overlay "sticks" (last sample) until the next. Near-live only on a
  **wired** Jetson link (already the project's #1 recommended fix — see
  `docs/jetson-scanner.md`). The authoritative check still runs at Create-targets.
- **Frontend** (`tasni/webui/src/pages/AimHud.tsx`): add a vector layer that draws
  `outline_uv` (color by `fully_framed`: green inside / red touching border) and `grid_uv`
  as polylines on the existing 1280×720 `viewBox` (multiply normalized coords by W/H). Add
  a "FRAMED ✓ / OVERFLOW ✗" lamp. Extend `GateReading` with the new optional fields.
  Add a 4th lamp ("FRAMED") in `Scan.tsx` `lamps` (`:252-256`).

### Phase 5 — config (`tasni/core/config.py`, `class ScanConfig` ~:250)
Add (names/defaults — tune on hardware):
| field | default | meaning |
|---|---|---|
| `accurate_min_mm` | 300 | near edge of the camera's accurate depth band |
| `accurate_max_mm` | 800 | far edge for *quality*; beyond → reference mode |
| `frame_margin` | 1.3 | surface must fit FOV with this margin |
| `reference_max_mm` | 1000 | human-readable cap; > this size ⇒ reference mode |
| `survey_max_tilt_deg` | 6.0 | tighter survey squareness (vs `max_tilt_deg=35`) |
| `voxel_k` | 0.008 | voxel = standoff_mm·k, clamped |
| `voxel_min_m` | 0.002 | finest voxel (small/close surfaces) |
| `voxel_max_m` | 0.006 | coarsest voxel |
| `surface_type` | "flat" | "flat" \| "raised" → cone/count preset |
| `flat_cone_deg` / `flat_views` | 18 / 8 | flat-top preset |
| `raised_cone_deg` / `raised_views` | 38 / 13 | raised-object preset |
| `grid_target_px` | 64 | desired on-screen grid cell size (1‑2‑5 picker target) |
Keep existing knobs as fallbacks. `survey_max_tilt_deg` replaces `max_tilt_deg` **only**
for the survey/Create-targets gate (the scan cone still uses the wide value).

### Phase 6 — UI copy + docs
- Rename "Aim at the table" flow to make the survey explicit ("Survey the surface");
  show measured extent + chosen mode + standoff before Run.
- Update `CLAUDE.md` roadmap + `tasni/README.md` scan notes when done.

---

## 7. Test / validation plan
- **Unit (pure, no hardware):** Phases 1–2 tests above. These run in CI/pytest like the
  existing `plane.py`/`poses.py` tests. This is the bulk of the correctness surface.
- **Integration (mock RoboDK/camera):** `generate_scan_targets` produces aims and seeds
  for a synthetic survey; reference mode returns a `ScanResult` with no targets.
- **Hardware (operator, on the KUKA):** small surface (fits) → quality mesh at a closer,
  surface-appropriate standoff; large surface (>1 m) → reference rectangle, no tour; verify
  the live grid skews with tilt and the outline turns red on overflow. **Best on a wired
  Jetson link** (Wi-Fi depth latency makes the live grid laggy).

---

## 8. Deferred: tiling (how the seam already supports it)
When large surfaces need full quality (not just a rectangle), `plan_scan` returns **N**
`AimPoint`s (a grid for rectangles, radial for circles) at the close standoff with overlap,
instead of one. The executor loop and TSDF fusion already handle many aims/views — TSDF is
built to fuse overlapping tiles. **No rewrite**: tiling is "the planner emits more aims."
Until then, large = reference rectangle (honest, labeled).

---

## 9. Subagent guidance for implementation (per user)
The user OK'd subagents to go faster **as long as they run on the correct model
(Opus 4.8 / `claude-opus-4-8`)** — do **not** let an implementation/review subagent fall
back to a weaker model.
- When spawning via the Agent tool, pass `model: "opus"` for any agent writing or reviewing
  code (the core work). Reserve cheaper models only for trivial mechanical file search, if at all.
- Good parallel split: one agent on Phase 1 (`survey.py` + tests), one on Phase 2
  (`planner.py` + tests) — they're independent pure modules with clear contracts (§3). Then
  do Phase 3 wiring yourself (it touches the shared service) and Phase 4 frontend.
- Keep each subagent's scope to a single file + its test; hand it the contracts from §3 and
  the math from §4 verbatim.

---

## 10. Key file map
| file | role | what changes |
|---|---|---|
| `tasni/modules/scan/survey.py` | **NEW** full-frame surface measurement + overlay vectors | create |
| `tasni/modules/scan/planner.py` | **NEW** `plan_scan` (standoff/mode/cone/count/voxel) | create |
| `tasni/modules/scan/depth_gate.py` | center-patch gate (tilt-fix math to reuse) | keep; reuse `:121-127` |
| `tasni/modules/scan/plane.py` | RANSAC + min-area rectangle | reuse (no change) |
| `tasni/modules/scan/poses.py`→`calibration/poses.py:38` | cone pose generator | reuse (no change) |
| `tasni/modules/scan/service.py` | orchestration | `generate_scan_targets` rewire + `reference_locate` |
| `tasni/modules/scan/module.py` | REST + `live_start` | point survey gate at `survey_surface`; maybe a `/survey` route |
| `tasni/core/config.py` | `ScanConfig` | add §5 knobs |
| `tasni/core/livepreview.py` | interleave depth probe | reuse `depth_probe` seam (no change) |
| `tasni/webui/src/pages/AimHud.tsx` | HUD | add outline+grid vector layer, `GateReading` fields |
| `tasni/webui/src/pages/Scan.tsx` | scan page | 4th "FRAMED" lamp; show mode/extent/standoff |

## 11. Open risks
- **Wi-Fi depth latency** caps the live overlay refresh (~6–11 s/sample). Recommend wired
  Ethernet for the survey; overlay sticks between samples. Authoritative check is at
  Create-targets regardless.
- **Single-view extent** is only trustworthy when `fully_framed`. That's enforced (survey
  refuses otherwise → reference mode or "back off").
- **`surface_type` is manual in v1** (flat vs raised). Auto-detecting "how 3D" from the
  survey (e.g. inlier_frac + height above surroundings) is a later refinement.
