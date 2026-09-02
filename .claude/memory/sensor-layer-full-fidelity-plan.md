---
name: sensor-layer-full-fidelity-plan
description: "Protocol 2 (raw unaligned 0.1 mm depth, 1080p colour, no align/hole_filling) SHIPPED and merged to main 2026-08-30 (54ab1da + cdc91f8); capability review done, 4 findings open, `py -3.10 tools/cell_health.py` is the morning readiness command"
metadata:
  node_type: memory
  type: project
  originSessionId: 32018e29-4e05-40ae-8d33-e56dda4f3e23
  modified: 2026-08-30T07:34:23.937Z
---

**DONE and live.** The 13-task programme shipped: protocol 2 = raw UNALIGNED depth at
0.1 mm words, depth 1280x720, colour 1920x1080, no `align`, no `hole_filling`, one JSON
greeting per connection carrying depth intrinsics + the depth->colour extrinsic.
librealsense rebuilt Release+CUDA+OpenMP (2.53.1 debug -> 2.55.1). Merged `54ab1da`,
post-deploy cell fixes merged `cdc91f8` (reticle-seeded plane fit, branch-guard bead
clamp, 3-D paper figures), review + health check `f6c17e6`. Jetson deployed, verified
live: `device reports 0.0001`, min step 0.10 mm, 92.3% real validity (the old 99.9% was
`hole_filling` fabricating), idle CPU 178.9%->28.2%, depth grab 7688->1055 ms.

**Why it matters:** removing `align` closed [[scan-chain-audit-2026-08-14]]'s +2.05%
lateral-scale defect — registration now happens once, on the host, through one
consistent model, with no resampling.

**How to apply:**
- **`py -3.10 tools/cell_health.py`** is the one-command morning readiness check
  (red/green + exact remedy). It reads device options from the service's JOURNAL
  READ-BACK rather than stopping the camera. `--skip-camera` for no camera traffic.
- Capability verdict (`docs/sensor-layer-capability-review-2026-08-30.md`): resolutions,
  depth word size, laser power and emitter are all at maximum or measured to have no
  headroom. **CUDA is exercised on the hot path** (YUY2->BGR8 at 1080p30,
  `unpack_yuy2_cuda`); librealsense ships only 2 CUDA kernels and the depth filter chain
  has **no GPU and no OpenMP path at all**, so its ~1 s/frame is irreducible in-SDK —
  don't chase more GPU flags. Empirical headroom numbers live in
  `docs/realsense-quality-headroom-2026-08-30.md`; Medium Density preset is the only
  open candidate.
- **All 4 findings FIXED + deployed** (`0c3a2b0` host, `4319c9e` server, 281 tests green):
  (1) `CameraClient.check_color_size` raises on a greeting-vs-config colour-size
  mismatch, at first USE of the geometry (not greeting-parse — a failover test opens a
  connection it never uses). (2) `_density_ratio` now divides by the CALIBRATED K that
  produced `uv`; fully-covered `valid_frac` 1.0376 -> 0.9974. **Watch:** both `>= 0.95`
  consumers (`_planned_surface_aim` quality shortcut, `lock_scan_surface`'s
  `LargeSurfaceRequired`) now fire at TRUE 0.95 not 0.9132, i.e. FEWER large-surface
  refusals. (3) `auto_exposure_priority` moved to the colour endpoint — **read-back
  proved the device was ALREADY 0, so it was a behavioural no-op**; the value is now
  explicitly asserted every open instead of accidentally correct. (4) pointcloud hoisted
  to the feeder thread (not module scope — the host suite imports that file with a
  stubbed pyrealsense2).
- **`tests/test_camera_recovery.py` had been DEAD since the protocol-2 merge** (5 tests,
  `openPipeline` returns one pipeline but the fakes returned the old `(pipeline, align)`
  tuple). Nobody saw it because the suite is too slow to run in full — a whole file can
  rot invisibly. Now 17 tests, with the `align` rebind assertion re-pointed at
  `STATIC_GEOMETRY`/`ACHIEVED_OPTIONS`/`DEVICE_INFO`/`depth_unit_mm` and mutation-proven.
- **Both greeting-coherence defects FIXED + deployed (`697cd84`)**: `_camera_snapshot()`
  reads the five globals under `_camera_lock` (device I/O stays OUTSIDE it — wrapping
  `make_greeting` wholesale would let a hung USB control transfer block the supervisor
  from ever rebuilding, a new permanent-wedge mode); and `greet()` now returns its
  generation so the streaming/burst loops close a connection whose greeting a rebuild
  invalidated, scoped so colour-only/H.264/telemetry clients (the scan HUD) are never
  dropped. **KEY FACT: `_camera_lock` is held across the WHOLE rebuild (~4 s typical,
  ~25 s worst), but a greeted client waits anyway — `read_frames()` takes the same lock
  on the very next statement after the greeting is sent.** In-band re-send was rejected:
  wire-format change needing a host parser change.
  **Behavioural consequence to know:** a rebuild mid-burst now closes the connection, and
  `scan._capture` recovers by falling back to per-pose, which RE-TOURS THE ROBOT from
  target 0. Correct-but-expensive beats silently-wrong. `grab()` callers in
  `scan._capture_per_pose` and `extrusion/measure.py` do NOT catch `CameraError`.
- **Still open, and worse than first assessed** — `read_frames` releases `_camera_lock`
  before `wait_for_frames`, and `_release_pipeline` reads the MODULE GLOBAL `pipeline`,
  so a thread that sampled generation G can be blocked in `wait_for_frames` on precisely
  the object being `stop()`ed. Per the 2026-08-29 note in `_release_pipeline`, stopping a
  wedged D435i once took the USB host controller down and **killed the process on
  SIGSEGV** — so the realistic worst case is a **segfault during recovery**, not a
  spurious timeout, masked by `Restart=always` as a restart loop. Deliberately NOT fixed
  (concurrency, real blast radius); fix needs the supervisor reworked to pass a handle
  rather than re-read the global. Also open: a burst session idle between poses only
  notices greeting staleness at the next `CAP`.
- Fixed since: the `depth_unit_mm` re-read no longer swallows (`13732d9`) — falls back to
  `ACHIEVED_OPTIONS["depth_unit_mm"]` (read off the REOPENED device) and logs a scale
  change; the old comment "same device; the startup value still holds" was wrong twice
  over. `server/README.md` rewritten (it described align + High Accuracy + 720p colour);
  the two dead `rs.align` servers deleted.
- 8 open3d test guards were `except: print; return` — reporting PASS while executing
  nothing (3 in test_scan_reconstruct, 5 in test_scan_job). Now `pytest.importorskip`.
- `server/server_unicast_asyncio.py` and `server_unicast_syncronous_dynamicRes.py` still
  call `rs.align` but are DEAD — the unit runs `server_unicast_syncronous.py` only.
- Related: [[pfh-paper-ring-stack-experiment]], [[scan-chain-audit-2026-08-14]],
  [[camera-intrinsics-distortion]], [[pytest-suite-too-slow-to-run-fully]],
  [[restart-tasni-backend-after-code-edits]].
