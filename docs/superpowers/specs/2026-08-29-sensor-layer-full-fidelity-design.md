# Sensor layer at full fidelity — design

**Status:** design settled with the operator on 2026-08-29 (big-bang approach chosen over
a versioned dual-path; R7 1080p colour included; cell risk before the 1 Sep paper deadline
accepted explicitly). **Next step:** `writing-plans` produces the task plan from §9.
**Scope:** audit sub-projects **A + B + C** of
`docs/realsense-capability-audit-2026-08-29.md` — reproducible sensor config (R4.1, R12,
R9-log), the Release+CUDA rebuild (R1), and the full-fidelity depth wire (R2, R3, R5, R7).
Tuning experiments (R4.2, R10), IR calibration (R6), IMU (R8), exposure lock (R11) and
chrony are **out** — each is its own spec once this has landed and been characterised.

Background: the audit itself (every number in §1 was read off the device); the earlier
workflow audit `docs/scan-audit-2026-08-14.md` (its A1 and A4 close here); the 2026-08-13
cell characterisation (the record every acceptance test compares against).

## 1. Why this exists

The scanner is not delivering what the D435i + librealsense + Jetson can measure:

- **Depth words are 1 mm** while the sensor resolves ~0.2 mm at the 450 mm standoff — a
  120×120 centre patch at 449 mm held **25 distinct values**. Extrusion's 5 mm layers are
  measured on 1 mm terraces; the on-Jetson filters' float smoothing is truncated to 1 mm on
  the way out.
- **`align(color)` on the Jetson discards ~50 % of the depth field** per view (87°×58°
  depth FOV projected into a 69°×42° colour frame), *upsamples* the surviving half by ~1.26×
  laterally into detail that was never measured, and creates the occlusion shadows that
  `hole_filling_filter` then paints over (the live frame was **99.9 % valid**). It is also
  the mechanism behind audit A1: the Jetson aligns with the *factory* colour K (fx 908.10),
  the host back-projects with the *calibrated* one (fx 889.87) → **+2.05 % lateral scale**.
- **The filters run after alignment**, so spatial/temporal smooth across shadows alignment
  just created. Intel's documented order is filters on native depth.
- **The loaded librealsense is an unoptimised debug build** (no `-O`, `-g`, 165 MB, no
  CUDA/OpenMP) — the service idles at 110 % CPU with no client, and align + spatial cost
  "~a second per frame". That cost is why the architecture grew a colour-only fast path,
  burst capture and an H.264 preview.
- **The ASIC configuration is unrecorded**: `visual_preset` = 0 (Custom). The 2026-08-13
  characterisation was measured under a state that nothing in the repo can reproduce.

What "highest-res" buys after this, at 450 mm: 0.1 mm depth words instead of 1 mm, the
full depth field per view instead of half, one camera model instead of two disagreeing by
2 %, a mesh built only from measured points, and a sensor configuration reproducible from
the repo.

## 2. What exists today (verified in code, `main` @ `ed0f742`)

- **Server** `server/server_unicast_syncronous.py` (1422 lines): one shared pipeline,
  depth 1280×720 z16 + colour 1280×720 bgr8 + an infrared stream nobody reads;
  `align = rs.align(rs.stream.color)`; `getFrames()` aligns first then runs
  `disparity → spatial → temporal → depth → hole_filling`. Wire header per frame is
  `<I depth_len><I color_len><d timestamp>` + lz4(`np.save`) depth + JPEG colour; the same
  framing is reused by `stream_burst`'s GET. Handshake modes: `MODE COLOR [Q<n>|H264 [B<kbps>]] [SCAN]`,
  `MODE BURST`, `MODE TELEMETRY`; anything else (or nothing) = full depth+colour.
  `depth_unit_mm` is computed at startup from `get_depth_scale()` and **never sent**.
- **The live HUD path already runs on native, unaligned depth.** `stream_h264`'s feeder
  takes `frames.get_depth_frame()` with `depth_profile.intrinsics`, builds an
  `rs.pointcloud`, and projects into colour through the factory depth→colour extrinsic —
  re-resolving pyrealsense2's column-major `.rotation` **empirically every telemetry frame**
  by projecting 8 sample points both ways (`:1170-1186`). `scan_plane_telemetry(depth,
  intr, depth_unit_mm, …)` already takes the unit as a parameter and emits overlay
  coordinates in pixels of `overlay_size` (the colour frame).
- **Host** `tasni/core/camera.py`: `Frame(color, depth, timestamp, telemetry)`;
  `CameraClient.grab/stream/burst`; full clients send no handshake.
  `CameraConfig.depth_scale = 1000.0` and `resolution = "1280x720"` (comment: "must be the
  720p K"); `intrinsics` keyed by resolution with factory defaults for 640×480 / 1280×720 /
  1920×1080; a single `dist_coeffs` list.
- **Six copies of the back-projection** `raw / depth_scale * 1000`, all against the
  *colour* K on aligned depth: `extrusion/processing.py:45`, `scan/depth_gate.py:100`,
  `scan/reconstruct.py:291`, `scan/service.py:957`, `scan/survey.py:174`,
  `tools/characterize_distance.py:200,205,268`. `depth_scale=1000.0` is a default in
  seven signatures.
- **Hand-eye** is the RoboDK `Realsense` **tool** offset: `RoboDKIO.camera_pose_T() =
  flange_pose_T() @ _tool_pose` (`core/rdk_io.py:181`). It was solved from ChArUco in the
  **colour** image, so it is the *colour* camera's pose.
- **Extrusion** `processing.py` did not read `frame.color` when this was written; since
  `041ad1b` (2026-08-29, after the spec) a saturation gate decides bead vs board from the
  colour frame by same-pixel indexing -- an aligned-depth assumption this design must port
  (see 4.4). The archive manifest already
  has a `provenance.camera_intrinsics` block that `figures.py` reads back. The extrusion
  chain voxel-downsamples at **2 mm** (`ExtrusionConfig.voxel_size_m`).
- **Scan TSDF** `reconstruct.fuse_views` integrates RGB-D with `TSDFVolumeColorType.RGB8`
  and then `measured_mesh_neutral_color=True` discards the colour; it raises if
  `depth.shape != color.shape`.
- **Calibration** uses `MODE COLOR` grabs only. Intrinsics auto-calibration
  (`calibration/intrinsics_calib.py`) refreshes K + distortion from a hand-eye run's views.
- **Uncommitted** on the working tree: an in-flight extrusion feature (ring-arc assembly,
  low-relief capture, 3 tests, a new `ring1_low_relief_20260829.npz` fixture) in
  `extrusion/processing.py` + `tests/test_extrusion_measure.py`. Same file this design
  changes.

## 3. Decisions taken with the operator

| Decision | Chosen | Rejected |
|---|---|---|
| Deadline vs. quality | go all-in before 1 Sep; the paper run happens **after** this lands, on the new chain | safe-items-only first |
| Scope | A + B + C | C only; A + B then measure |
| Wire strategy | **big bang**: one depth format, `align` deleted, every consumer ported in one push | versioned `MODE FULL2` with per-consumer migration |
| 1080p colour (R7) | **in** — no legacy aligned path remains to inflate | defer |

A stale host must still fail *loudly*, not misread (§4.1). That is a refusal, not a
compatibility path.

## 4. Design

### 4.1 The wire

One depth format. A depth client sends `MODE FULL V2\n` (burst: `MODE BURST V2\n`). The
server replies with **one newline-terminated JSON line — the greeting — then frames**
(schema below; numeric values illustrative, the server fills them from the live profile):

```json
{"protocol": 2, "aligned": false, "depth_unit_mm": 0.1,
 "depth": {"width": 1280, "height": 720, "fx": 0, "fy": 0, "ppx": 0, "ppy": 0,
           "model": "brown_conrady", "coeffs": [0, 0, 0, 0, 0]},
 "color": {"width": 1920, "height": 1080, "fx": 0, "fy": 0, "ppx": 0, "ppy": 0},
 "depth_to_color": {"rotation_row_major": [[1,0,0],[0,1,0],[0,0,1]],
                    "translation_mm": [0, 0, 0]},
 "filters": ["threshold", "disparity", "spatial", "temporal", "disparity_inv"],
 "device": {"serial": "", "fw": "5.16.00.01", "librealsense": "2.55.1",
            "visual_preset": 0, "laser_power": 150},
 "temps": {"asic_c": 0.0, "projector_c": 0.0},
 "global_time_enabled": true}
```

- **The per-frame header is byte-identical to today**: `<I depth_len><I color_len><d ts>`
  + lz4(`np.save`) depth + JPEG colour. Burst-GET reuses it. Only the array's *meaning*
  changes: native depth resolution, `depth_unit_mm` units. One host decoder serves all
  paths.
- **Version gate.** A depth or burst client that sends no handshake, or one without the
  `V2` token, is refused: the server logs one line (`client <ip> did not request V2; this
  server speaks protocol 2 only`) and closes. This is what saves a backend that did not
  restart (the module-cache trap) from blocking on a length it misparsed out of JSON
  bytes. `MODE COLOR …` and `MODE TELEMETRY` are unchanged (no depth on them).
- **The greeting is JSON, once per connection**, rather than new header fields: the header
  is fixed-width and shared by three send paths; temps/preset/firmware want room to grow.
  `grab()` connects per call, so it pays ~1 KB per grab — negligible.
- **The extrinsic ships row-major, verified at startup.** librealsense documents
  `rs2_extrinsics.rotation` as column-major; the server transposes once at pipeline start
  and **asserts** against `rs2_transform_point_to_point` on a test point (raise, do not
  guess). The per-frame 8-point probe in the H.264 feeder is deleted.

### 4.2 Server

Two extractions out of `server_unicast_syncronous.py`, each with a real second caller:

| Module | Contents | Callers |
|---|---|---|
| `server/rs_config.py` | preset / laser / **`depth_units` = 0.0001** / `auto_exposure_priority` with the existing read-back logging; advanced-mode `serialize_json()` dump at start (**R4.1**), logged and written to `~/robodk-characterization/asfound-<date>.json`; `asic_temperature` + `projector_temperature` readers (**R12**); `global_time_enabled` read-back (**R9**) | startup logging, the greeting |
| `server/rs_geometry.py` | depth/colour intrinsics, the extrinsic transposed + asserted, greeting dict builder, `depth_to_color_projector` | the V2 handshake, the H.264 telemetry feeder |

Then in the main file:

- `align` removed from the globals, `_rebuild_pipeline`, `getFrames`, `stream_burst`.
  `openPipeline` drops `rs.align` and `enable_stream(infrared, 1)`; colour stream to
  **1920×1080 bgr8 @30** (**R7**). `width/height` split into `DEPTH_SIZE` and `COLOR_SIZE`;
  `stream_h264`'s gst caps and `frame_bytes` follow `COLOR_SIZE`.
- Filter chain `threshold(depth_min..depth_max) → disparity → spatial → temporal →
  disparity_inv`; **`hole_filling` deleted** (**R5**). `getFrames` runs it on the native
  depth frame — the order Intel documents, and the shadows it used to smooth across no
  longer exist.
- `depth_unit_mm` in the greeting comes from `get_depth_scale()` **after** the option is
  set — the achieved value, never the requested one.
- Startup log gains: as-found JSON path, `depth_units` read-back, temps,
  `global_time_enabled`, librealsense version string.

The Jetson-side `scan_overlay.py` / `scan_plane_telemetry` are unchanged in logic; the
feeder passes the (now 0.1) `depth_unit_mm` it already takes, `overlay_size` becomes the
1080p colour size, and the projector comes from `rs_geometry`.

### 4.3 Host core

**`tasni/core/depth_geometry.py`** — new, pure numpy, no sockets, no RoboDK. The six
back-projections collapse into it:

- `CameraGeometry` (frozen dataclass): `depth_unit_mm`, `depth_K` (3×3), `depth_size`,
  `depth_dist`, `color_size`, `T_color_depth` (4×4, mm — maps depth-frame points into the
  colour frame), `temps`, `device`, `raw` (the greeting dict, for archives).
  `CameraGeometry.from_greeting(dict)` validates `protocol == 2`.
- `backproject(depth_u16, geom, *, mask=None) -> (pts_mm[N,3], idx[N])` — depth-frame
  points in mm, zero-depth pixels dropped.
- `to_color_pixels(pts_mm, geom, K_color, dist_color) -> uv[N,2]` — project depth-frame
  points into the **calibrated** colour model. This is the single answer to "which depth
  pixels does a colour-space region mean": back-project everything, project into colour,
  keep what lands inside. The gate's centre patch, the survey's rectangle, and
  `characterize_distance`'s per-corner disc all use it. No inverse mapping exists.
- `depth_pose(T_x_color, geom) = T_x_color @ geom.T_color_depth`. The hand-eye remains the
  **colour** camera's pose; the depth pose is composed at the back-projection site and
  never persisted.

**`tasni/core/camera.py`**: `CameraClient` sends `MODE FULL V2`, reads the greeting line,
parses `CameraGeometry`, attaches it as `Frame.geometry` (and keeps `client.geometry` =
last seen). Burst reads it after `BURST READY`. A refusal/close before the greeting raises
`CameraError("camera server speaks protocol 2 — restart the Tasni backend")`.
`CameraConfig.depth_scale` is **deleted**, not deprecated: nothing may fall back to 1000.

**Config migration** (`core/config.py`, on load): `resolution` default → `"1920x1080"`.
If `intrinsics["1920x1080"]` still equals the factory default while `intrinsics["1280x720"]`
does not, set it to **1.5 × the calibrated 720p entry** (fx, fy, ppx, ppy scaled; the
factory table shows 1080p is an exact 1.5× of 720p: 1362.15/908.10 = 1.5000). Distortion
coefficients are normalised and carry over unchanged. **The hand-eye is a physical
transform and stays valid** — R7 needs a validation run, not a recalibration.

### 4.4 Consumers

| Consumer | Change |
|---|---|
| `extrusion/processing.py::depth_to_work_points` | signature `(depth, geom, T_work_color)`; back-projects with `depth_K` into the colour frame. **Chroma gate (`041ad1b`) moves from depth pixels to registered points**: project each depth point into the calibrated colour model, read the saturation mask there; abstention rules unchanged; points outside the colour image dropped while the gate applies. `ExtrusionConfig.voxel_size_m` **0.002 → 0.001** — R2 reaches the ring numbers only through this (the voxel is point spacing, not accuracy; the 1.26 mm hand-eye floor still governs claims). |
| extrusion archive | `manifest.provenance.camera_geometry` = `geom.raw` (temps, preset, fw, unit come free). **`figures.py` keeps one read-side fallback:** a manifest without `camera_geometry` is rendered as it was captured — aligned, 1 mm, colour K — so ring 1 and the paper fixtures still render. Archived data, not a live path. |
| `scan/reconstruct.fuse_views` | depth-only TSDF: `TSDFVolumeColorType.NoColor` with a constant image at depth size; intrinsic = `depth_K` @ `depth_size`; extrinsic = inv(`depth_pose`); Open3D `depth_scale = 1000 / depth_unit_mm`. The `depth ≠ color shape` guard is deleted. `measured_mesh_neutral_color` becomes the only behaviour. |
| `scan/depth_gate`, `scan/survey`, `scan/service._backproject_depth`, `_save_views` | via `backproject` + `to_color_pixels`; `_save_views` writes `geom.raw` instead of `depth_scale`. `ScanView` carries `geometry`. |
| `tools/characterize_distance.py` | corners detected in colour → depth points whose colour projection lies within `corner_disc_px` (default 6 px at 1080p, a tool argument) of the corner → median. It migrates in this push because it is what **proves** the win. |
| calibration module | no code change (colour-only grabs, now 1080p). |
| `webui/AimHud.tsx` | telemetry already carries `overlay_size`; plan-time check that the HUD divides by it rather than assuming 1280×720 — fix if it does. |

### 4.5 Sequencing (big bang, but sequenced for evidence)

```
0  commit the in-flight extrusion work (ring arcs; same file)          — operator confirms it is ready
1  R4.1  capture the as-found advanced-mode JSON over SSH, commit server/presets/custom-as-found-2026-08-29.json
         BEFORE depth_units changes (the depth table is part of that JSON)
2  R1    rebuild librealsense v2.55.1 Release + CUDA + OpenMP + py3.10 bindings in ~/librealsense/build_cuda/,
         re-point the venv's pyrealsense2, restart; verify the OLD protocol serves identically;
         log idle CPU + per-frame getFrames() time before/after. Old build dir stays = rollback.
3  BIG BANG  server + host + consumers + config migration + tests on one branch; one deploy;
         backend restarted and its StartTime checked against file mtimes
4  cell acceptance (§7)
```

## 5. Error handling, summarised

- Stale backend vs new server → refused at handshake, one log line each side, no hang.
- Greeting missing/invalid (`protocol != 2`, missing keys) → `CameraError` naming the
  field; never silently default a unit.
- Extrinsic assert fails at server start → the service does not serve depth (log the two
  candidate projections); better dark than 2 % wrong again.
- Archive without `camera_geometry` → legacy render, and the figure caption says so.
- `depth_units` unsupported/rejected by the device → greeting reports the achieved value;
  the host uses whatever it says. The acceptance probe (§7) is what notices.

## 6. Testing (proof before the cell)

- **`tests/test_depth_geometry.py`** — the real unit coverage: backproject→`to_color_pixels`
  round trip on a synthetic K with a known extrinsic; `depth_pose` against a hand-built
  case; ROI mask selects exactly the expected pixels; `from_greeting` parses the §4.1
  document and rejects `protocol: 1`.
- **`tests/test_rs_geometry.py`** (server, pure) — greeting builder against fake profile
  objects; the transpose-and-assert fails on a deliberately wrong orientation.
- **`tests/test_camera_client.py`** — greeting parse and the refusal path against a fake
  socket.
- **Extrusion** — the existing ring-1 fixture now exercises the archive fallback; one new
  synthetic native-depth fixture (`geom` with a non-identity extrinsic) exercises the real
  path and must recover a known ring radius.
- **Scan** — `fuse_views` on two synthetic depth-only views of a plane reproduces the plane
  in the base frame.
- `npm run build` for the HUD; `pytest -k` on the touched tests only (the full suite is
  not run — see memory).

## 7. Cell acceptance (from the audit; each gets a dated file)

| Check | Pass |
|---|---|
| Centre-patch probe (audit §6 script) | step 0.1, ≥ 200 distinct values in the 120×120 patch |
| Points per view at the same pose | ≈ 2× today |
| Validity fraction, logged per capture | **drops** from 99.9 % (real support) |
| `characterize_distance` before/after, as-found JSON loaded | plane RMS not worse; `length_err_mm` within noise of true where today it reads +2 % |
| Idle service CPU, per-frame time (R1) | both lower; recorded in the doc |
| Hand-eye **validation** run at 1080p colour | held-out reprojection ≤ ~0.9 px band, board consistency in the same band as before |
| HUD | tilt/level readouts and the rectangle overlay behave as on 2026-07-06 |

## 8. Non-goals

R4.2 preset/laser/disparity-shift tuning, R10 848×480 A/B, R6 IR-frame calibration,
R8 IMU tilt, R11 exposure lock / AE ROI, chrony — each its own spec after §7 has produced
the baseline they must be compared against. Tailscale-path bandwidth. Any host-side hole
filling.

## 9. Implementation tasks (for `writing-plans`)

1. Commit the in-flight extrusion work as its own commit (after the operator confirms).
2. R4.1 as-found JSON: SSH capture script + `server/presets/custom-as-found-2026-08-29.json`.
3. R1 rebuild runbook + execution on the Jetson; before/after CPU + frame-time numbers
   into the audit doc.
4. `server/rs_config.py` (+ depth_units, temps, global_time, as-found dump at start).
5. `server/rs_geometry.py` + tests.
6. Server: V2 handshake + greeting, refusal, `align` removal, filter chain, 1080p colour,
   `stream_h264` sizes, feeder uses `rs_geometry`.
7. `tasni/core/depth_geometry.py` + tests.
8. `CameraClient` V2 + greeting + `Frame.geometry`; delete `depth_scale`; config migration.
9. Extrusion: `depth_to_work_points`, archive provenance, `figures.py` fallback, voxel,
   fixtures.
10. Scan: `fuse_views` depth-only, gate/survey/service back-projection, `_save_views`.
11. `tools/characterize_distance.py` colour-corner → depth-disc.
12. HUD `overlay_size` check; `npm run build`.
13. Deploy, restart backend, run §7; write results into the audit doc; update
    `docs/jetson-scanner.md`, `AGENTS.md`, CLAUDE.md's stale "High-Accuracy preset" line.
