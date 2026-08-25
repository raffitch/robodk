# Scan module audit — 2026-08-14

End-to-end review of the scan workflow: the operator journey, what each click triggers
on the host and on the Jetson, where the measurement chain is inconsistent, and what to
change. Branch `calibration-improvements` @ `8067ee6`.

**Goal being audited against:** place highly accurate work frames of any dimension in the
robot world, as efficiently as possible, within what a D435i + Jetson Nano can actually do.

**State of the branch at audit time**
- `pytest tests/test_scan_job.py tests/test_scan_planner.py tests/test_five_position.py`
  → **116 passed, 1 FAILED** (`test_generate_refuses_when_too_far`, see A2).
- `npm run typecheck` → clean.
- Jetson `realsense-camera` active, listening on 1024, auto-pull timer running.

---

## 1. The walk — what actually happens

### Page load
`Scan.tsx` mounts → `GET /api/rdk/status`, `/config`, `/status`, `/survey/state` → auto
`POST /connect` (polls through the 117 MB station load, `link_real_robot`) → auto
`POST /live/start`.

### Live aiming
`POST /live/start` opens **two** TCP connections to `10.12.171.70:1024`:

| Connection | Handshake | Jetson handler |
|---|---|---|
| video | `MODE COLOR H264 B4000 SCAN` | `stream_h264` → `gst-launch-1.0 … nvv4l2h264enc` + a feeder thread |
| telemetry | `MODE TELEMETRY` | `stream_telemetry` — blocks on a condition var, sends whatever the feeder last published |

The feeder thread pulls `pipeline.wait_for_frames()`, and **every ≥1 s also computes
`scan_plane_telemetry` inline on that same thread**: centre-patch depth → `fit_nearest_plane`
(nearest coherent plane, *not* largest) → expand across a strided full-frame sample →
`center_connected_mask` (pure-Python 8-connected BFS) → min-area rectangle → density-cliff
trim + colour edge veto → 180-cell coverage-dot grid → publish.

Host `analyze()` runs per **video** frame: reticle overlay, 960 px JPEG re-encode, hand the
frame to `SamBoundaryWorker` (EdgeSAM ONNX, background thread → `boundary` event), then
`live_scan_telemetry_payload` re-derives all gates from the *last* telemetry payload,
`stabilize_live_scan_payload` applies EMA + a hold gated on the RoboDK camera pose, and
`annotate_pose_liveness` marks staleness. `AimHud` renders DETECT / DISTANCE / ANGLE /
CENTER / EDGE A / FRAMED plus jog bars; Lock unlocks after `gate.ok` holds 1 s.

### Lock
`POST /surface/lock` → `lock_scan_surface` → `_authoritative_acquisition`:
stop live → mount camera tool → `_camera_hold` → **five separate `grab(with_depth=True)`
calls, each its own TCP connection**, full aligned+filtered depth (lz4 `.npy`) + q100 JPEG →
median fuse → `evaluate_depth_gate` (centre patch, plain SVD) **and** `survey_surface`
(full-frame RANSAC, dominant plane) → overrun check (`LargeSurfaceRequired` vs crop) →
`classify_compact` (guard band, boundary corroboration, centring, tilt, 5-frame rectangle
identity — re-surveys each raw frame) → hybrid extent (plane from depth, edges from the
SAM/colour polygon undistorted onto that plane, directionally corroborated) →
characterization-age gate → publish `frame` + `gate` → return a lock token.

### Targets → dry run → run → insert
`POST /poses/generate` → cone poses (`generate_calibration_poses`), reachability, collision
screen, coverage-aware diversity → `TasniScan_*`. Five-position locks branch to
`_generate_tiled_scan_targets` (`plan_rect_tour`). `POST /poses/simulate` → `SimTourJob` in
SIMULATE. `POST /run` → `run_robot`, per target `move_j` + `settle_s` + burst CAP, one bulk
GET, TSDF fuse (Open3D legacy `ScalableTSDFVolume`), ROI crop, `work_plane_from_points`,
bounded region, mesh clean, coverage metrics. `POST /insert` → frame (origin = corner
nearest base, +X = longer edge meeting it, +Z = up), rectangle, mesh, `active.json`.

---

## 2. Findings

| ID | Area | Finding | Severity |
|---|---|---|---|
| A1 | accuracy | Two disagreeing camera models in one measurement chain (2.05 % lateral scale) | critical |
| A2 | accuracy | Standoff gate accepts ±150 mm *outside* the accurate band; test is red | high |
| A3 | accuracy | Live and lock choose the surface plane by different rules | high |
| A4 | accuracy | `hole_filling_filter` fabricates depth in the metrology chain | high |
| A5 | accuracy | Temporal filter with no post-motion frame flush | medium |
| A6 | accuracy | Five-position `CaptureRecord.plane_rms_mm` is a whole-scene residual | low |
| A7 | accuracy | Flat surfaces scanned with a hand-eye *calibration* pose distribution | medium |
| B1 | latency | Defect 2 explained: telemetry computed inline on the video thread | high |
| B2 | efficiency | Five TCP connections + five full frames per authoritative measurement | medium |
| B3 | robustness | One shared RealSense pipeline, thread-per-client, no arbitration | medium |
| C1 | UX | Live crop square is 1000 mm while the config/lock use 600 mm | medium |
| C2 | UX | JPEG preview silently disables all scan guidance | medium |
| C3 | UX | Five different tests decide "framed" | low |
| C4 | hygiene | Dead `depth_within_border` in `survey.py` | low |
| C5 | hygiene | `agent-debug-map.md` wrong about `classify_compact` being unwired | low |
| C6 | hygiene | `jetson_deploy.py status` blocks on a sudo prompt | low |

### A1 — Two disagreeing camera models in one measurement chain

The Jetson aligns depth into the colour frame with the **factory** colour intrinsics
(`_DEFAULT_INTRINSICS["1280x720"]`: fx 908.10, cx 650.24, **zero distortion** — the D435i
ships RGB with no distortion coefficients). The host back-projects that aligned depth with
the **ChArUco-calibrated** intrinsics from `tasni.config.json` (fx 889.87, cx 648.98,
k1 +0.1148, k2 −0.2386) — and back-projects with a **plain pinhole, no undistortion**
(`_backproject_depth`, `survey_surface`, `depth_gate.evaluate_depth_gate`,
`_deproject_plane_points_mm`).

Two consequences on every X/Y measured from depth:

- **Scale.** 908.10 / 889.87 = **+2.05 %**. A 600 mm declared region measures ≈612 mm; a
  1000 mm platform measures ≈1020 mm. Z is unaffected (it comes from the depth value), so
  plane RMS and repeatability stay clean — which is why nothing flagged it.
- **Distortion.** Never applied on the back-projection side. At the frame corner the radial
  factor is ≈0.971, i.e. ~20–30 px, ~10–17 mm at 500 mm standoff — largest exactly at the
  boundary, which is where extents and corners are measured.

Meanwhile `_project_color_corners_uv` (service.py:311) and `_corners_from_boundary_on_plane`
(service.py:1110) **do** apply distortion. So projection and back-projection disagree, and
the newest feature — "plane from depth, EDGES from vision" (`69e588e`) — stitches the two
models together at the point of maximum divergence.

**Why the 2026-08-13 characterization passed.** `length_err_mm` (0.003–0.43 mm) is measured
by `_corner_on_plane_mm`: undistort the *colour* corner pixel, ray-intersect the depth-fitted
plane. That validates the colour model plus depth *z*. It never measures a length from the
depth grid's own back-projection, so the depth-grid lateral scale is untested. `plane_rms`
is scale-invariant. The one metric that would catch this is missing.

**Fix.** Make one model authoritative end to end. Either push the calibrated intrinsics to
the Jetson, or (better) stop aligning on the Jetson — ship raw depth + depth intrinsics +
depth→colour extrinsics and do the mapping host-side with the calibrated model. At minimum,
back-project through `cv2.undistortPoints` / a cached `initUndistortRectifyMap` with the same
model everything else uses. Then add the missing acceptance test: a known length measured
**from the depth cloud alone** at three standoffs (VDI/VDE 2634-2 calls this sphere-spacing
error; it is the standard acceptance test for an optical 3D measuring system).

### A2 — Standoff gate accepts 150 mm outside the accurate band

`distance_tol_mm` went 50 → 150 in `5b08e33`. That tolerance is applied around a standoff
already clipped to `[accurate_min_mm=300, accurate_max_mm=800]` (`plan_scan` step 3), so the
effective accept window is now **150–950 mm**:

- 950 mm is past the characterized envelope (furthest trial 795 mm, plane RMS 2.05 mm vs
  0.93 mm at `d* = 310 mm`).
- 150 mm is below the D435i's MinZ at 1280×720 (~175 mm) — there would be no depth at all.

`tests/test_scan_job.py::test_generate_refuses_when_too_far` fails on this branch as a direct
result: a 900 mm standoff is now accepted. The widening was justified by real data (RMS
varies little across ±150 mm), but the *clamp* was lost.

**Fix.** `lo = max(ideal - tol, accurate_min_mm)`, `hi = min(ideal + tol, accurate_max_mm)`.
Then the test passes on its own terms again.

### A3 — Live and lock choose the surface plane by different rules

- Jetson `fit_nearest_plane`: seeded from the **centre patch**, expanded by 8-connectivity.
  Its own comment: *"anchored to the reticle plane rather than 'largest plane wins': floor/walls
  and objects cannot steal the live work-surface measurement."*
- Host `survey_surface` → `plane.fit_plane`: **the plane with the most inliers in the whole
  frame**, no reticle anchoring, no connectivity constraint.

On a platform sitting on a larger table — the exact "any dimension" case — the HUD can track
the platform while the lock snaps to the table. `five_position_capture` already documents this
failure mode verbatim for corner captures (*"survey_surface's RANSAC can lock onto the
BACKGROUND … silently accepted with FLOOR geometry mislabelled as table geometry"*) and
declares plane selection out of scope.

**Fix.** Port the reticle-seeded + connected selection into `survey_surface`. It is ~30 lines
and already written twice (`server/server_unicast_syncronous.py:181,33`). One rule everywhere.

### A4 — `hole_filling_filter` in the metrology chain

`setup_depth_filters()` returns `[disparity, spatial, temporal, depth, hole_filling]`, applied
by `getFrames` to every `MODE FULL` and `MODE BURST` frame — i.e. to the lock measurement,
every five-position capture, and every scan-tour view that gets TSDF-fused.

`hole_filling_filter` **fabricates** depth by copying from neighbours. It is a visualisation
filter. It directly corrupts what this module measures:

- the density-cliff edge trim (`_density_extent_1d`) looks for a cliff hole-filling smooths over;
- `fully_framed` (filled pixels reach the image border);
- the coverage dots, documented as marking *"REAL measured depth"* — they no longer do;
- the TSDF, which fuses invented geometry as if measured.

Compounding it: the **live** telemetry path uses raw, unaligned, **unfiltered** depth
(`frames.get_depth_frame()` inside the h264 feeder) while the lock path uses aligned+filtered
depth. Two different depth products for the same surface.

**Fix.** Drop `hole_filling` from the capture chain (keep a filled variant for display only if
wanted), and make the live and authoritative chains identical.

### A5 — Temporal filter with no post-motion flush

`rs.temporal_filter()` carries state across frames. At each tour pose: `move_j` →
`sleep(settle_s)` → CAP. Nothing discards frames after the move, and the pipeline holds the
most recent frame, so a CAP after an idle can return a frame captured before or during motion.
Standard practice is to discard ~5 frames (or reset the filter) after any motion.

### A6 — Five-position `plane_rms_mm` is a whole-scene residual

`service.py:1333` calls `_plane_rms_mm(frame.depth, K)` **without `outline_uv`** — the exact
defect `777b9a8` fixed for the compact lock. A corner capture is majority background, so this
records the scene depth spread, not surface flatness, into the immutable survey contract. The
coplanarity *gates* use `fit_global_plane`'s per-set RMS (correct), so this is evidence-quality
only — but it is stored and carried forward as if meaningful.

### A7 — Flat surfaces scanned with a calibration pose distribution

`generate_scan_targets` orbits a cone: `flat_cone_deg = 18°`, roll ±30°, distance jitter ±15 % —
`generate_calibration_poses`, where pose diversity *is* the objective. For a flat plate the
opposite holds, and your own 2026-08-13 characterization measured it: **incidence costs ~4×
what distance costs**. 18° of deliberate incidence buys nothing on a plane; the jitter walks
the standoff off `d*`.

The right generator already exists: `plan_rect_tour` + `_generate_tiled_scan_targets`
(overlapping fronto-parallel tiles at a fixed standoff), currently reachable only via the
five-position path. Use it for `surface_type="flat"` too; keep the cone for raised/3D objects.

### B1 — Defect 2 (~10 s HUD lag) is fully explained

The Jetson computes `scan_plane_telemetry` **synchronously on the H.264 feeder thread**
(`server_unicast_syncronous.py:899`). While it runs, nothing pumps `wait_for_frames()` and
nothing feeds the encoder — so the video hitches *and* telemetry cannot be produced faster
than the computation allows.

Measured cadence is 3.4 s against a nominal 1.0 s → ~2.4 s of compute per cycle. Benchmarked
the dominant term: `center_connected_mask` is a pure-Python 8-connected BFS over the strided
plane mask (~120×214 cells at stride 6) — **112 ms on this workstation, so ~0.9–1.7 s on a
Nano** — plus a full-frame `rs.pointcloud().calculate()` and an 11 MB vertex array per cycle.

End to end: 3.4 s production + ~2.5 s transport ≈ 6 s minimum age, worse through the settle
and EMA layers → the reported ~10 s, and the "freezes then jumps" signature.

`f5f8b69` widened the staleness gate 2 s → 10 s "sized to the MEASURED cadence". That stops
healthy payloads being dropped, but the HUD still cannot be fresher than its producer. The
measurement in `live_diag.py`'s candidate #2 is already the answer.

**Fix.** (1) Move telemetry off the feeder thread — feeder stashes the latest depth+colour in
a slot, a worker computes. (2) Make it cheap: fit on a decimated 424×240 depth frame, and
replace the Python BFS with a vectorised label propagation (or install OpenCV on the Jetson
and use `connectedComponents`). (3) Then run at ~5 Hz. Aiming latency drops from ~6 s to
<0.5 s and the video stops hitching.

### B2 — Five TCP connections per authoritative measurement

`_authoritative_acquisition` loops `services.camera.grab(with_depth=True)` ×
`surface_measure_frames = 5`, and `grab()` connects / reads one frame / closes each time.
This is visible in the Jetson journal as bursts of `Connection from …` immediately followed
by `Lost connection … Broken pipe` — the server keeps streaming into a socket the client
already closed, so **every one-shot depth grab logs an error line**, which also makes the log
useless for spotting real faults.

**Fix.** One `MODE FULL` connection, read N frames, close. Optionally a `MODE SNAP <n>` so the
server stops after N and the disconnect is clean.

### B3 — One shared pipeline, thread-per-client, no arbitration

`main()` spawns a thread per connection; all of them call `pipeline.wait_for_frames()` on one
global pipeline (frames get split between concurrent pullers — your own finding, `f2e4f56`).
The only thing preventing it today is the **host-side** `CameraLease`; nothing on the Jetson
enforces it. A second tool (`tools/characterize_distance.py`, a second app instance, or a
previous client's thread still inside its 10 s `sendall` timeout) silently degrades both.
`handle_client`'s comment still claims *"this server is single-threaded with listen(1)"* —
stale since it became threaded with `listen(5)`.

**Fix.** One capture thread owning the pipeline, publishing the latest frame to per-client
queues (a frame broker). That also makes B1 and B2 trivial.

### C1 — The live crop square is the wrong size

`WORK_CROP_MM = 1000.0` is hardcoded on the Jetson; `tasni.config.json` has
`scan.work_crop_mm = [600, 600]`. In crop mode the HUD **draws** the Jetson's 1000×1000 square
while the readout says 600×600 and the lock produces 600×600 — 2.8× the area. `module.py:132`
calls this "intentionally left out of sync"; it is still a false statement in the operator's
primary aiming instrument, and the box visibly jumps on lock.

### C2 — JPEG preview silently disables all scan guidance

The `SCAN` handshake token is parsed for every mode (`server:759`) but only forwarded to
`stream_h264` (`server:797`). The plain-JPEG `while True` loop never calls
`publish_scan_telemetry`. Setting `calibration.preview_codec = "jpeg"` — a documented,
supported setting — therefore produces video with **no telemetry at all**, no error, and a
gate that simply never goes green.

### C3 — Five different tests decide "framed"

Jetson `depth_fully_framed` (border pixels, `margin = max(2*stride, 10)`) → Jetson
`fully_framed` (+ colour-margin corner test, `0.015`) → host override from re-projected corners
(`live_frame_margin_uv = 0.02`) → host lock `_corners_in_frame` → `classify_compact`'s guard
band (`compact_guard_uv = 0.04`). Each was added to fix a real symptom; together they are the
main reason framing behaviour is hard to predict, and A2's clamp interacts with all of them.

### C4 / C5 / C6 — hygiene

- `survey.py:227` computes `depth_within_border` and never uses it (the superseded pixel-border
  framing test). It reads as if the border rule still applies.
- `docs/agent-debug-map.md` states `classify_compact` has "zero callers outside that test file"
  and that `lock_scan_surface` never imports it. It **is** wired now (`service.py:569,596`).
  Anyone reading the map first will mis-model the lock.
- `py -3.10 tools/jetson_deploy.py status` blocks on `[sudo] password for jetson:` partway
  through, so it cannot complete non-interactively.

---

## 3. Tooling currency

| Component | Installed | Current | Verdict |
|---|---|---|---|
| OpenCV | **4.7.0** (Feb 2023) | 4.12 / 4.13 | ~2.5 years behind. The ChArUco API in use — `interpolateCornersCharuco`, `estimatePoseCharucoBoard` — is **deprecated since 4.8** in favour of `cv2.aruco.CharucoDetector`, which detects + interpolates + refines in one pass and handles partial boards better. Upgrading requires that migration; it is also a genuine accuracy win for the calibration the scan depends on. |
| NumPy | 1.24.2 (Dec 2022) | 2.x | Pinned in practice by cv2 4.7 / Open3D 0.17. Upgrade as a set. |
| Open3D | 0.17.0 | 0.19+ | Uses the **legacy** `o3d.pipelines.integration.ScalableTSDFVolume`; the modern path is `o3d.t.geometry.VoxelBlockGrid` (tensor API, much faster, optional GPU). Also **unpinned in `pyproject.toml`** — a fresh install today gets a different Open3D than the one validated. Pin it. |
| SciPy | 1.10.1 | 1.14+ | Behind; only used for reprojection refinement. |
| FastAPI 0.137 · pydantic 2.13 · uvicorn | current | — | Fine. |
| onnxruntime 1.23.2 · PyAV 17.1 | current | — | Fine. |
| React 18.3 · Vite 5.4 · TS 5.5 · three 0.161 | 2024 | React 19 · Vite 7 · TS 5.9 · three ~0.18x | Not urgent. three.js is the most dated. |
| librealsense 2.55.1 · D435i FW 05.16.00.01 | recent, matched pair | — | Healthy. No action. |
| **Jetson Nano / L4T R32.7.6 / Ubuntu 18.04** | **EOL** | — | Hard ceiling: no newer JetPack exists for this board. System Python 3.6, no OpenCV in the server venv, NVENC driven through a `gst-launch` subprocess. **This is the real platform risk** — the Nano is what throttles the whole live loop. An Orin Nano (or even a Pi 5 + USB3) would run the telemetry at 30 Hz and remove B1/B3 structurally. |
| EdgeSAM weights | S-Lab **non-commercial** | — | Licence exposure if Tasni is ever commercial. MobileSAM (Apache-2.0) already drops in via `scan.sam_*` config — switch the default and validate once now, rather than discovering it later. |

---

## 4. Recommended sequence

Ordered by accuracy gained per unit of effort.

1. **Pin down the camera model** — one K, one distortion policy, end to end; add the
   depth-only known-length check to `characterize_distance.py`. (A1) *Gates everything else:
   a 2 % scale error makes every other tolerance in the system meaningless.*
2. **Drop `hole_filling`; flush frames after motion.** (A4, A5) Two small changes, immediate
   quality.
3. **Clamp the standoff window to the accurate band; unfail the test.** (A2)
4. **Move Jetson telemetry off the video thread and make it cheap.** (B1) Biggest UX win —
   aiming becomes real-time.
5. **One plane-selection rule** (reticle-seeded + connected) shared by live and lock. (A3)
6. **Tiled fronto-parallel planner for flat surfaces**; retire the cone for that case. (A7)
7. **Frame broker on the Jetson**; single connection for N-frame grabs. (B2, B3)
8. Cosmetics and hygiene: C1, C2, C3, C4, C5, C6.

---

## 5. On "any dimension, most efficiently"

The five-position survey is honest and well-built, but it is expensive UX: hand-jogging a
KR150 to four corners, with gates that can reject the whole survey at `finish()` after all
five captures.

A cheaper path you already have every part for — **coarse-to-fine**:

- Back off to whatever standoff frames the whole platform, **even past `accurate_max_mm`** —
  metric depth is not needed there.
- Take the **boundary from vision**. SAM already does this, and an undistorted ray is
  angularly exact at any range: 1 px ≈ 1.7 mm at 1.5 m.
- Take the **plane from depth up close**, at `d*`.
- Intersect the vision rays with that plane — which is exactly what
  `_corners_from_boundary_on_plane` already does, just with the two measurements taken at
  different standoffs instead of one.

One jog instead of five, and the boundary is *more* accurate than the corner ritual because it
never depends on cross-position registration (the weakness `rect_fit.py`'s own docstring
documents: `discrepancy_mm` is blind to pure translational error). Keep the five-position
survey for platforms that do not fit even at maximum reach, or that are not visible from one
vantage; keep `LargeSurfaceRequired` as the honest refusal, just give it a second, cheaper
primary action.

One more sizing note: the whole design orbits `d* = 310 mm`, chosen as "the closest distance
that passes every gate". At 310 mm the 1280×720 footprint is only ~440×250 mm — so almost any
real work platform is a "large surface", and the tile count explodes. Re-running the sweep with
the incidence axis (`f6536c5` added it) and choosing `d*` for **coverage-per-view × accuracy**
rather than "closest that passes" would cut tiling substantially.
