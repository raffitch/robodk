# RealSense capability audit — 2026-08-29

Is the scanner using everything the D435i + librealsense + the Jetson can give it for the
highest-resolution 3D scan? **No.** This document records what is *actually* running on the
device (probed live, not read from docs), what is already right, and twelve places where
capability is being left on the table — each with evidence, cost, a fix, and the
acceptance test that proves the fix helped.

**State at audit time:** `main` @ `267cf71`. Jetson `realsense-camera` active, no clients
connected, arm parked. Every number below was read back from the device on 2026-08-29
unless it says "spec".

**Companion:** [scan-audit-2026-08-14.md](scan-audit-2026-08-14.md) audited the scan
*workflow*; this audits the *sensor layer* underneath it. Its findings A1 (two camera models,
2.05 % lateral scale) and A4 (hole filling in the metrology chain) reappear here as R3 and
R5 because the sensor-layer fix is the same fix.

**Update (2026-09-01):** the safe-tier depth-filter chain (spatial, temporal, threshold,
hole-filling) is now runtime-settable via a `SET` command on the burst protocol, with every
achieved value recorded in the per-connection greeting's `filter_options` and every override
dying on a service restart — see the status lines added to R2, R3, R5 and R7 below.

---

## 1. What is actually running (measured)

| | Measured | Source |
|---|---|---|
| Camera | D435i, FW **5.16.00.01**, USB id `8086:0b3a` | `docs/jetson-scanner.md`, unchanged |
| USB link | **5000M** (real USB 3) on `tegra-xusb` Bus 02 | `lsusb -t` |
| IMU | enumerates as HID If 5; kernel has `hid_sensor_accel_3d` / `hid_sensor_gyro_3d` loaded, `/sys/bus/iio/devices/iio:device0..2` present | `lsusb -t`, `lsmod`, `/sys` |
| librealsense **loaded by the service** | **2.53.1** (`~/librealsense/build_py310/librealsense2.so.2.53.1`, built 2023-05-30 from `v2.53.1-5-gb874e4268`, Dec 2022) | `ldd` of the venv's `pyrealsense2.so` |
| librealsense **installed by apt** | 2.55.1 (`/usr/lib/aarch64-linux-gnu/librealsense2.so.2.55.1`, 9.3 MB) — **not the one loaded** | `dpkg -l`, `ls -la` |
| Build type of the loaded lib | `CMAKE_BUILD_TYPE` **empty**; compile flags `-pedantic -g ... -ftree-vectorize` — **no `-O` at all**; `.so` is **165 MB** | `CMakeCache.txt`, `CMakeFiles/realsense2.dir/flags.make` |
| CUDA / OpenMP in the loaded lib | `BUILD_WITH_CUDA=OFF`, `BUILD_WITH_OPENMP=OFF` (CUDA 10.2 toolkit is installed at `/usr/local/cuda`) | `CMakeCache.txt`, `ls /usr/local` |
| Backend | V4L2 (`FORCE_RSUSB_BACKEND=OFF`) — the right one for 720p streaming on a Nano | `CMakeCache.txt` |
| Streams | depth 1280×720 z16 @30, colour 1280×720 bgr8 @30, **infrared 1 enabled and never read** | `server/server_unicast_syncronous.py:940-943` |
| Depth options read back at start | `visual_preset` **0 = Custom**, `laser_power` **150** of 0..360, `emitter_enabled` 1 | journal 15:12:24 |
| Exposure / white balance / AE priority | not set anywhere → device defaults (auto) | grep of the server |
| `depth_units` | supported by this build; not set → 0.001 m. **Live frame: uint16, quantisation step = 1 (mm), 25 distinct values across a 120×120 centre patch at 449 mm** | `probe_depth.py` (§6) |
| Filter chain | disparity → spatial → temporal → depth → **hole_filling**, no decimation | `setup_depth_filters()` |
| Live frame validity | **99.9 %** of 921,600 pixels valid, range 436–1128 mm | `probe_depth.py` |
| Wire timestamp | `frames.get_timestamp()` (device/global time, ms) in the 16-byte header; host compares against the *workstation* clock | server `:823, :1051`; `docs/scan-workframe-cell-defects.md` §4 |
| Service CPU | **110 %** of 400 % (4 cores) with **no client connected** | `top` |
| Power / fan | `nvpmodel` MAXN, fan pinned 255 by the unit | `nvpmodel -q`, unit file |
| Host consumption | `align(color)`-ed depth + JPEG; `depth_scale=1000.0` hard-coded as a default in 7 signatures; `frames_per_pose=3` median; TSDF voxel 1.5 mm; `measured_mesh_neutral_color=True` (colour discarded on the scan mesh) | `tasni/core/config.py`, `modules/scan/reconstruct.py` |

Three of those rows contradict the written record: `docs/jetson-scanner.md` says librealsense
2.55.1 (that is the apt package, not the loaded library); the CLAUDE.md line "RealSense
High-Accuracy preset + filter order live in `server_unicast_syncronous.py`" describes a state
the device has never been in (the code was corrected to leave-alone + read-back; the device
is on Custom); and the server's filter comment still says "full-resolution" when half the
field is cropped by alignment (R3).

---

## 2. What is already right

- **Max depth resolution.** 1280×720 is the D435i's top depth mode. ✅
- **No decimation** on scan data — deliberate, commented. ✅
- **Filter order** disparity→spatial→temporal→depth is Intel's recommended ordering. ✅
- **USB 3** confirmed (the well-known Nano trap is a USB-2 negotiation; not the case). ✅
- **MAXN + fan at 100 %** — no thermal throttling while scanning. ✅
- **Emitter forced on**, laser pinned by `Environment=` so device state is reproducible
  across restarts (the earlier 300 mW drift is closed). ✅
- **Options are read back and logged**, so the journal states the achieved config. ✅
- **Host side:** TSDF fusion, 3-frame median per pose, burst capture so the arm is not
  blocked on Wi-Fi. ✅

The architecture is sound. The waste is *underneath* it: an unoptimised sensor library
forcing throughput compromises, a depth word that throws away sub-millimetre resolution, and
an alignment step that discards half the depth field to get colour the mesh does not keep.

---

## 3. Findings

| ID | Area | Finding | Impact | Effort |
|---|---|---|---|---|
| R1 | throughput | The loaded librealsense is an **unoptimised debug build without CUDA/OpenMP**, two years behind the installed apt version | high — every later item has headroom because of it | medium (one rebuild, zero app code) |
| R2 | resolution | Depth is quantised to **1 mm**; the sensor resolves ~0.2 mm at the working standoff | high — bites extrusion (5 mm layers on 1 mm terraces) hardest | medium (plumb `depth_scale` end to end) |
| R3 | coverage/accuracy | `align(color)` on the Jetson **discards ~50 % of the depth field** per view and is the root of audit A1 | high — ~2× the poses needed; 2.05 % scale bug | high (wire-format change, host mapping) |
| R4 | tuning | **Advanced mode available, never used**; device runs an unrecorded **Custom** preset | high — the 2026-08-13 characterisation was measured under a configuration nobody can reproduce | low to record, medium to tune |
| R5 | accuracy | `hole_filling_filter` fabricates depth (99.9 % valid frame) | medium — audit A4 | low |
| R6 | calibration | **Left IR stream enabled, never consumed** — depth's own frame | medium — removes depth↔colour registration from the hand-eye chain | medium |
| R7 | precision | Colour streamed at **720p** while 1080p is available and its K is already in config | medium — ChArUco corner precision bounds everything downstream | low (after R3) |
| R8 | robustness | **IMU unused**; tilt is inferred from a plane fit that needed three debouncing rounds | medium | medium |
| R9 | robustness | Timestamps cross a machine boundary; `global_time_enabled` not verified/logged | low–medium — the logged HUD-freeze hazard | low |
| R10 | resolution | 1280×720 vs Intel's stated **848×480 optimum** for D435-class depth noise never A/B'd | unknown until measured | low (one characterisation run) |
| R11 | repeatability | Exposure / white balance / AE-priority left on auto across a scan run | low–medium | low |
| R12 | record | ASIC/projector temperature never logged; warm-up drift invisible in the archive | low, high documentary value for the paper | trivial |

### R1 — The loaded librealsense is an unoptimised debug build

```
build_py310/CMakeFiles/realsense2.dir/flags.make
CXX_FLAGS = -pedantic -g -Wno-missing-field-initializers ... -ftree-vectorize -pthread -fPIC -std=gnu++11
```

No optimisation flag at all (`CMAKE_BUILD_TYPE` is empty, so CMake adds none), `-g` on. The
result is a **165 MB** shared object where the optimised apt build of a newer version is
**9.3 MB**. `BUILD_WITH_CUDA=OFF` on a CUDA device: `rs.align`, `rs.pointcloud` and colour
conversion are exactly the paths librealsense offloads to the GPU when built with CUDA.
`BUILD_WITH_OPENMP=OFF` on four cores.

The service sits at **110 % CPU with nobody connected**; the server's own comment on the
colour-only fast path says align + spatial filter "cost the Nano ~a second per frame". That
one-second cost is why the architecture had to grow a filter-skipping fast path, burst
capture and an H.264 preview. Those were the right responses to the cost; the cost itself is
mostly the build.

**Fix.** Rebuild **2.55.1** (the tag matches FW 5.16.00.01's recommendation) as Release with
CUDA + OpenMP + Python bindings for the venv's 3.10, in a *new* build directory so the current
one stays as rollback:

```sh
cd ~/librealsense && git fetch --tags && git checkout v2.55.1
mkdir build_cuda && cd build_cuda
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_WITH_CUDA=true -DBUILD_WITH_OPENMP=true \
  -DBUILD_PYTHON_BINDINGS=true \
  -DPYTHON_EXECUTABLE=/home/jetson/EtherSenseServer/ethenv/bin/python \
  -DBUILD_EXAMPLES=false -DBUILD_GRAPHICAL_EXAMPLES=false -DFORCE_RSUSB_BACKEND=false
make -j2      # 4 GB RAM: -j4 with CUDA units OOMs; add zram/swap first. ~1 h.
```

Then swap the venv's `pyrealsense2*.so` for the new one (today's binding RPATH-links its own
build dir, so the same pattern — new dir, re-point — keeps rollback a one-line change).
Toolchain on the Nano is sufficient (CUDA 10.2, gcc 7.5, cmake 3.10.2, 21 GB free).

**Acceptance.** Same options, same scene: (a) service idle CPU and per-frame `getFrames()`
time logged before/after; (b) plane RMS on the characterisation board unchanged within noise
(CUDA align rounds slightly differently — a change is expected, a *worsening* is a bug);
(c) the fast path's "~a second per frame" comment re-measured and updated.

**Do not** do this in the days before the 1 Sep paper deadline without the rollback path
proven first; do it with the old build dir intact.

### R2 — Depth is quantised to 1 mm; the sensor resolves ~0.2 mm

A live frame with the arm parked came back `uint16`, minimum step **1**, median 450 (mm),
and only **25 distinct values** across a 120×120 centre patch. The D435i's disparity has
5 sub-pixel bits (Δd = 1/32 px); with f ≈ 650 px and B = 50 mm the depth granularity is
Δz = Z²·Δd/(f·B):

| Standoff | Sensor granularity | Delivered (1 mm units) | Lost |
|---|---|---|---|
| 300 mm | 0.09 mm | 1 mm | 11× |
| 450 mm | 0.19 mm | 1 mm | 5× |
| 600 mm | 0.35 mm | 1 mm | 3× |
| 800 mm | 0.62 mm | 1 mm | 1.6× |
| 1000 mm | 0.96 mm | 1 mm | — |

So across the whole `accurate_min..accurate_max` band (300–800 mm) the 1 mm word is the
limiting factor, not the sensor. This is granularity, not accuracy — noise is larger — but
the on-Jetson temporal/spatial filters run in float in disparity space and are **re-quantised
to `depth_units` on the way out**, so their smoothing is truncated to 1 mm too. For extrusion,
a 5 mm layer measured on 1 mm terraces is ±10 % per layer before anything else is counted.

`rs.option.depth_units` is supported by this build. `0.0001` (0.1 mm) keeps a 6.55 m
ceiling (the frame spans 0.44–1.13 m).

**Cost.** The wire header (`<I depth_len><I color_len><d timestamp>`) carries no scale, and
the host defaults `depth_scale=1000.0` in seven signatures (`config.py:647`,
`extrusion/processing.py`, `scan/depth_gate.py`, `scan/reconstruct.py`, `scan/service.py`).
The filter deltas are unaffected (disparity domain); `threshold_filter`, `depth_min_m`,
`depth_max_m`, and any host threshold in raw units must be re-read.

Two extrusion-specific consequences, checked against `e0b78d7` (the 2026-08-29 figures
merge): (a) each take's **archive manifest must carry `depth_unit_mm`**, because
`figures.py` re-runs `depth_to_work_points` on *archived* depth with the default scale —
a post-R2 take rendered without it would come out 10× too far away; (b) the extrusion
chain **voxel-downsamples at 2 mm** (`ExtrusionConfig.voxel_size_m`, `config.py:809`),
which `docs/extrusion-current-handoff.md` correctly calls the measurement's resolution —
so R2 reaches the ring numbers only once that voxel is lowered with it (the multi-view
inspection spec already flagged the same cap).

**Fix.** Add a versioned handshake (`MODE FULL2` → server replies one JSON line with
`depth_unit_mm`, depth intrinsics, depth→colour extrinsics — the same greeting R3 needs);
carry `depth_scale` on the host `Frame`; make the config field the fallback for the old
handshake. Set `depth_units` in `set_high_accuracy_preset()` with the same read-back logging
as the other options.

**Acceptance.** Repeat the centre-patch probe: step 0.1 mm, ≥200 distinct values in the
same patch. Then `tools/characterize_distance.py` before/after — plane RMS should *drop* at
300–500 mm; if it does not, the noise floor was already above the LSB and the win is confined
to the filtered/averaged outputs (still worth having, since the median and TSDF then have
sub-mm structure to average).

**FIXED** (2026-08-29, `54ab1da` on `main`, the sensor-layer full-fidelity batch) —
`depth_units` is now always set and read back to `DEPTH_UNITS_M = 0.0001` (0.1 mm),
unconditionally, unlike the leave-alone default used for laser power/preset:
`server/rs_config.py:18` (constant), `:75-76` (set + read-back). Confirmed against the
current tree, not merely the commit message.

### R3 — `align(color)` on the Jetson discards ~50 % of the depth field

Depth FOV is 87°×58°, RGB is 69°×42° (spec). On the image plane that is
tan(34.5°)/tan(43.5°) × tan(21°)/tan(29°) = 0.72 × 0.69 ≈ **0.50** — every aligned frame
throws away half the depth pixels that were computed. The scan then sets
`measured_mesh_neutral_color=True` and discards the colour it paid for. Alignment is also
where the occlusion "shadows" behind edges come from, which R5's hole filling then paints
over.

This is the same mechanism as audit **A1**: the Jetson aligns with the *factory* colour K
(fx 908.10, zero distortion); the host back-projects with the *calibrated* K (fx 889.87, real
distortion) → +2.05 % lateral scale, up to ~17 mm at the frame corner at 500 mm. A1's
recommended fix and this finding's fix are one change.

**Fix.** Stop aligning on the Jetson. Ship raw z16 depth at depth resolution plus, once per
connection (the R2 greeting), the depth stream's intrinsics and the depth→colour extrinsic
(`profile.get_extrinsics_to()`). On the host: back-project with the depth intrinsics (the
depth image is rectified — effectively zero distortion), fold the extrinsic into the hand-eye
(`T_flange_depth = T_flange_color · T_color_depth`), integrate TSDF from the depth image
directly, and project into the *calibrated* colour model only where colour is needed
(boundary corroboration, ChArUco). The Jetson stops doing the most expensive per-frame
operation it has; the host gets the full field and one camera model.

**Acceptance.** (a) Points per view ≈ 2× at the same pose; (b) the A1 acceptance test —
a known length measured **from the depth cloud alone** at three standoffs (VDI/VDE 2634-2
sphere-spacing style) — reads within noise of true, where today it reads +2 %; (c) a flat
platform locks with fewer `TasniScan_*` poses at the same coverage gate.

**FIXED** (2026-08-29, `54ab1da` on `main`, the sensor-layer full-fidelity batch) — there is
no `align()`/`rs.align` call anywhere left in `server/server_unicast_syncronous.py`; depth
ships native/unaligned and the greeting says so on the wire (`"aligned": False`,
`server/rs_geometry.py:79`), carrying the depth intrinsics and the depth→colour extrinsic
(`server/rs_geometry.py:29-61`, `static_geometry`/`build_greeting`) for the host to
back-project with one camera model. Closes audit A1's 2.05 % scale bug with it.

### R4 — Advanced mode is available and never used; the device runs an unrecorded Custom

`hasattr(rs, 'rs400_advanced_mode')` → True. The journal shows `visual_preset` = **0**, i.e.
**Custom**: the depth ASIC is running whatever was last written to it — not High Accuracy,
not Default, not anything reproducible from this repo. The 2026-08-13 depth characterisation
(the number the standoff gate trusts) was therefore measured under a configuration that a
replacement camera, a firmware reset, or a `realsense-viewer` session on someone's laptop
could change without any diff appearing anywhere.

Untapped in advanced mode:

- **Depth control block** (`get_depth_control()`): `textureCountThreshold`,
  `textureDifferenceThreshold`, `deepSeaSecondPeakThreshold`, `deepSeaMedianThreshold`,
  `scoreThreshA/B`, `lrAgreeThreshold`. These are *precisely* the confidence knobs that
  decide whether a low-texture region near a surface edge yields a point or a hole — the
  2026-08-13 finding "valid depth stopped ~20 mm short of the panel's top and bottom edges"
  lives here, and the server comment already noted High Accuracy is the wrong *direction*
  (it raises confidence thresholds). High Density / Medium Density lower them.
- **Depth table** (`get_depth_table()`): `disparityShift` trades MaxZ for a closer MinZ.
  Z error scales with Z², so if extrusion inspection could sit at 250 mm instead of 440 mm,
  granularity goes 0.19 → 0.06 mm and noise shrinks with it. Also `depthClampMin/Max` and
  `depthUnits` (the same knob as R2).
- **Presets as JSON** (`load_json` / `serialize_json`): Intel ships tuned starting points
  (High Accuracy, High Density, Medium Density, Default). Committing a JSON to the repo and
  loading it at start makes the ASIC configuration a reviewable diff — the same discipline
  `Environment=RS_LASER_POWER=150` brought to laser power.
- **Laser power** sits at 150/360 (42 %). Contrast on the projected pattern is what the
  matcher has to work with on blank surfaces (the ring's extruded surface is blank).

**Fix — in two separate steps, in this order.**
1. **Record before changing anything**: `rs400_advanced_mode(dev).serialize_json()` at
   service start, logged and written to a dated file under `characterization/` (git-ignored
   like the characterisation itself) *and* a copy committed under `server/presets/` as
   `custom-as-found-2026-08-29.json`. This is read-only and makes the 2026-08-13 record
   reproducible for the first time.
2. **Then tune as a dated experiment**: load High Density, re-run `characterize_distance.py`,
   compare edge-fill and plane RMS against the as-found JSON. Same for laser 150 → 300 → 360,
   and for `disparityShift` at the extrusion standoff. Each trial is one JSON + one
   characterisation file.

**Acceptance.** The as-found JSON exists and reloading it reproduces the 2026-08-13 plane RMS
within noise. For the tuning step, the 20 mm edge deficit shrinks with plane RMS not
worsening; otherwise the change is rejected and the JSON stays as-found.

### R5 — `hole_filling_filter` fabricates depth

The live frame was **99.9 % valid** at ranges 436–1128 mm. Real stereo depth of an indoor
scene with a parked arm is not 99.9 % valid; the last filter in the chain is inventing the
rest, and it invents most exactly where the metrology cares — the occlusion shadows R3
creates behind every edge, and the low-confidence band at surface edges R4 describes. Audit
A4 said the same from the workflow side. `rs.threshold_filter` is available and unused;
clipping to the work volume *before* the spatial filter also stops background depth from
being smoothed into surface edges.

**Fix.** Drop `hole_filling` from the scan/extrusion path (keep it, if at all, for the live
preview where a pretty image is the point). Add `threshold_filter(min, max)` around the
current `depth_min_m..depth_max_m` band ahead of `spatial`. If holes must be filled for a
downstream step, do it host-side after R3 with a method that marks filled pixels as such.

**Acceptance.** Raw validity fraction logged per capture; the coverage gate (`min_actual_*`)
now measures real support. Expect the validity number to drop and the mesh's
`measured_mesh_min_support_views` filtering to become *less* necessary.

**FIXED** in two steps. The `threshold_filter` clip ahead of `spatial`
(`RS_DEPTH_MIN_M`/`RS_DEPTH_MAX_M`) and dropping `hole_filling` out of the chain both landed
2026-08-29 in `54ab1da` on `main` — but there `hole_filling` was simply absent from the code,
an omission rather than a recorded decision. This runtime-parameters branch (commit `1ab6dab`;
**not yet merged to `main`**) turns that absence into an explicit runtime knob:
`FILTER_SETTINGS["hole_filling"]` defaults to `-1.0` (off) and is only added to the chain when
`>= 0` (`server/server_unicast_syncronous.py:972` default, `:1822-1825` conditional add). "We
do not fill holes, on purpose" is now a value recorded in every greeting's `filter_options`,
not an absence a reader has to infer.

### R6 — The left IR stream is enabled and never read

`cfg.enable_stream(rs.stream.infrared, 1)` with no consumer costs USB bandwidth and a little
CPU for nothing. But the better move is to *use* it: D400 depth is computed in — and
registered to — the **left IR** frame. A ChArUco detected on IR 1 is detected in the depth
camera's own coordinate frame, so the hand-eye solve for the *depth* camera no longer routes
through the RGB lens, the factory depth→colour extrinsic, and the alignment resampling. For a
depth-only mesh (which the scan already is, see R3) that removes an entire error term from
the chain.

**Cost.** The projector pattern litters the IR image and will break ChArUco detection, so the
IR capture needs the emitter off for that frame: either toggle `emitter_enabled` around the
one-shot grab (simplest; the live path is not affected because one-shots already stop the
preview), or use `emitter_on_off` alternating mode with `sequence_id_filter` (supported by
this build) so depth and clean IR interleave continuously. IR is 8-bit grey — ChArUco
detection is unaffected; the JPEG path needs a single-channel branch.

**Fix.** Add `MODE IR` to the handshake (colour-only-shaped payload, `y8` instead of BGR),
emitter off for the grab, emitter back on after. Offer the calibration module a
"calibrate the depth camera" run that uses it. Keep the colour hand-eye for the boundary/vision
features that genuinely need RGB.

**Acceptance.** Hand-eye on IR reports held-out reprojection and board-consistency in the
same band as the RGB solve (≤ 0.9 px / ≤ 0.8 mm), and the R3 sphere-spacing test improves or
holds.

### R7 — Colour at 720p while the sensor and config already support 1080p

ChArUco corner precision sets the hand-eye quality, which bounds every downstream number.
`_DEFAULT_INTRINSICS` already carries a 1080p K; the server streams 720p. 1080p@30 colour
alongside 720p depth is a supported D435i combination and fits USB 3 (~124 + 55 + 27 MB/s
with IR). Wi-Fi cost only matters for one-shots (2.25× the JPEG bytes at q100).

**Do this after R3.** With alignment still on the Jetson, aligning to a 1080p colour frame
upsamples the depth image to 1080p — 2.25× the lz4 payload for zero information — and the
colour conversion at 1080p is the CPU cost R1 removes.

**Acceptance.** Reprojection RMS on the calibration board drops (expect roughly the pixel-
pitch ratio, ~1.5×); intrinsics auto-calibration coverage unchanged.

**FIXED** (2026-08-29, `54ab1da` on `main`, the sensor-layer full-fidelity batch) — colour
streams at 1080p: `COLOR_SIZE = (1920, 1080)` (`server/server_unicast_syncronous.py:878`),
consumed by `cfg.enable_stream(rs.stream.color, ...)` (`:1155`). Landed together with R3 (raw
depth, no Jetson-side upsampling to 1080p), matching this finding's own "do this after R3"
ordering.

### R8 — The IMU is unused

The D435i's accelerometer gives the gravity vector in the camera frame directly. Today the
LEVEL/TILT HUD derives it from a plane fit on noisy depth, which took three rounds of
hysteresis and pose-gated holds (`c7442c2`, `55f3c27`, `eb523cf`) to stop wiggling. The
kernel side is already there: `hid_sensor_accel_3d`/`gyro_3d` are loaded and `iio:device0..2`
exist, so the V4L2 backend can read it without a rebuild or a kernel patch.

**Fix.** Enable `rs.stream.accel` (63 Hz is enough) on the shared pipeline — note that
framesets then carry motion frames, so `wait_for_frames` consumers must skip them, or the
accel is read on a callback — and publish a smoothed gravity vector in the telemetry payload.
The HUD uses it for tilt; the plane fit stays for *surface* orientation. Uncalibrated IMU
bias is ~1°, which is under the HUD's deadband; `rs-imu-calibration.py` if better is needed.

**Acceptance.** Tilt readout jitter with the arm parked ≤ 0.1° p-p without any hold logic
engaged; RoboDK-reported flange tilt and IMU tilt agree within 1° across the aiming cone.

### R9 — Timestamp domain not pinned or logged

The wire header carries `frames.get_timestamp()`. With `global_time_enabled` (librealsense
defaults it on for D400, but nothing here sets or logs it) that is the *Jetson's* system
clock — and `docs/scan-workframe-cell-defects.md` §4 records the hazard: the host compares it
with the *workstation* clock, on a Jetson with no RTC battery. No RealSense option fixes a
cross-machine skew; but the state should be known, and intra-Jetson latency (sensor →
`getFrames()` → socket) *is* measurable with `frame_metadata_value.sensor_timestamp` /
`frame_timestamp` / `backend_timestamp` and never has been.

**Fix.** Read back and log `global_time_enabled` at start like the other options; add the
metadata timestamps to the telemetry payload so the live-latency probe can split Jetson
pipeline latency from Wi-Fi latency; keep cross-machine staleness arrival-based
(`service.py:1715` already does this for telemetry) and put `chrony` on the Jetson so
`journalctl` timestamps and the archive agree with the workstation.

**Acceptance.** `LiveLatencyProbe` reports sensor→socket latency as its own column; the
"Clock skew tripping the 2 s drop" signature in the cell-defects table cannot occur.

### R10 — 1280×720 vs 848×480 was never A/B'd

Intel's D400 tuning guide states that the D435-class optimum for depth *noise* is 848×480 —
the ASIC's native operating mode — while 1280×720 buys lateral sample density. For a
metrology scan the right choice depends on whether the limiting error is per-pixel depth noise
(then 848×480 wins and also cuts align/filter cost 2.3×) or edge localisation (then 720p wins).
The repo has never measured it. Nothing above should be *assumed* to favour 720p.

**Fix.** One `characterize_distance.py` run at each resolution, same board, same standoffs,
same as-found JSON (R4 step 1 first). Compare plane RMS, edge-fill, and `length_err_mm`.

**Acceptance.** A dated pair of characterisation files and a one-line decision recorded in
this doc.

### R11 — Exposure and white balance float across a scan run

Nothing sets colour exposure, white balance, or `auto_exposure_priority`. Consequences: the
colour AE priority default lets the RGB frame rate drop below 30 in dim light (which stalls
the shared pipeline's `wait_for_frames` for *every* client, including depth), and ChArUco /
boundary contrast differs pose to pose. Depth AE can also be told *where* to expose via the
sensor ROI (`rs.roi_sensor`) so the projector pattern is exposed for the work surface, not the
far wall.

**Fix.** `auto_exposure_priority=0` unconditionally (frame rate is a contract here). Lock
colour exposure + WB for the duration of a scan/inspection run (run start: read the auto
values, write them as manual; run end: restore auto). Set the depth AE ROI to the work
rectangle when a lock exists.

**Acceptance.** Frame interval histogram from the H.264 feeder shows no sub-30 fps tail;
per-pose JPEG mean luminance variance across a run drops to noise.

### R12 — Temperature never logged

`asic_temperature` and `projector_temperature` are read-only options that cost nothing to
read. Depth drifts during warm-up (Intel: allow the ASIC to reach steady state; FW-side
thermal compensation exists on this firmware) and no capture in the archive records whether it
was taken cold. For the paper's ring-stack trials that is a reviewer's first question.

**Fix.** Read both at every full-frame grab, put them in the R2 greeting / telemetry payload,
and have the extrusion archive write them into each take's metadata.

**Acceptance.** Every archived take carries both temperatures; a warm-up curve (first 15 min
after service start) exists as a figure.

---

## 4. Recommended sequence

```
R4.1 record as-found JSON            (read-only, hours, do FIRST — makes the 08-13 record reproducible)
R12  log temperatures                (trivial, same commit)
R1   Release+CUDA rebuild of 2.55.1  (unblocks everything; keep old build dir for rollback)
R2   depth_units 0.1 mm + greeting   (needs the greeting; host depth_scale plumbing;
                                      archive manifest; extrusion 2 mm voxel lowered with it)
R3   raw depth + host-side mapping   (same greeting; closes audit A1; ~2x points per view)
R5   drop hole_filling, add threshold (falls out of R3: fill host-side or not at all)
R10  848x480 A/B                     (one characterisation run, after R4.1 so it is comparable)
R4.2 preset / laser / disparity-shift experiments (each a JSON + a dated characterisation)
R7   1080p colour                    (only after R3)
R6   IR-frame calibration            (after R3; needs emitter toggle)
R11  exposure lock + AE ROI          (independent, any time)
R8   IMU tilt                        (independent, any time)
R9   timestamp domain + chrony       (independent, any time)
```

Dependencies: R2 and R3 share the greeting; R7 depends on R3; R5 is best done with R3; every
tuning experiment (R4.2, R10, laser) must be preceded by R4.1 or it is not comparable with
2026-08-13. None of this should land on the cell in the days before the 1 Sep deadline unless
it is R4.1/R12 (read-only, no behaviour change).

What "highest-res" actually buys after all of it, at a 450 mm standoff: 0.1 mm depth words
instead of 1 mm, the full 87°×58° depth field per view instead of half, one camera model
instead of two disagreeing by 2 %, a mesh built only from measured points, and a sensor
configuration that can be reproduced from the repo.

---

## 5. Things that are NOT worth chasing

- **A newer librealsense than 2.55.1 on this Nano.** JetPack 4.6 / L4T R32.7.6 is end of the
  line for the board; 2.55.1 is the last release Intel validated against it and matches
  the installed firmware's recommendation. Newer tags may build but gain nothing measurable
  here.
- **The RSUSB (libusb) backend.** It removes the kernel dependency but is the known cause of
  frame drops at 720p on Nanos; the V4L2 backend the current build uses is correct.
- **Decimation on scan data.** Correctly refused already.
- **Tailscale-path bandwidth.** Not a sensor limit; unchanged by anything here.

---

## 6. How this was verified (repeatable)

```sh
# what the service actually loads, and how it was built
ssh -i ~/.ssh/jetson_robodk jetson@10.12.171.70 '
  ldd $(/home/jetson/EtherSenseServer/ethenv/bin/python -c "import pyrealsense2 as m;print(m.__file__)") | grep realsense
  grep -E "^BUILD_WITH_CUDA|^BUILD_WITH_OPENMP|^CMAKE_BUILD_TYPE|^FORCE_RSUSB" ~/librealsense/build_py310/CMakeCache.txt
  grep CXX_FLAGS ~/librealsense/build_py310/CMakeFiles/realsense2.dir/flags.make
  ls -la ~/librealsense/build_py310/librealsense2.so.2.53.1 /usr/lib/aarch64-linux-gnu/librealsense2.so.2.55.1
  lsusb -t; lsmod | grep hid_sensor; ls /sys/bus/iio/devices; nvpmodel -q; top -bn1 | grep python
  journalctl -u realsense-camera -n 400 --no-pager | grep "RealSense:"'

# what the build exposes
ssh ... '/home/jetson/EtherSenseServer/ethenv/bin/python -c "
import pyrealsense2 as rs
print(hasattr(rs, \"rs400_advanced_mode\"), hasattr(rs.option, \"depth_units\"))
print([a for a in dir(rs) if \"filter\" in a or \"transform\" in a])"'

# depth quantisation, from the workstation, with no other client connected
# (PYTHONPATH=<repo>; py -3.10)
import numpy as np
from tasni.core.config import load_config
from tasni.core.camera import CameraClient
d = CameraClient(load_config().camera).grab(with_depth=True, timeout=20).depth
v = d[d > 0]; print(d.dtype, d.shape, "valid %.3f" % (v.size / d.size), v.min(), v.max())
cy, cx = d.shape[0] // 2, d.shape[1] // 2; p = d[cy-60:cy+60, cx-60:cx+60]
pv = np.unique(p[p > 0]); print("unique", pv.size, "min step", np.diff(pv).min())
```

Output on 2026-08-29: `uint16 (720, 1280) valid 0.999 436 1128` / `unique 25 min step 1`.
