# Sensor-layer capability review, 2026-08-30

**Scope.** The whole "sensor layer at full fidelity" programme, `82a89c3..cdc91f8`
(~4,500 insertions across 48 files), reviewed with a **capability** lens: is
anything leaving the D435i's sensor capability unused? Areas the operator named:
intrinsics handling and the 1.5x 1080p migration, depth/colour resolutions, laser
power and emitter, stereo/disparity settings, whether the CUDA build is actually
exercised by the paths that matter, and whether `align`/`hole_filling` removal is
complete.

This review **builds on** [`realsense-quality-headroom-2026-08-30.md`](realsense-quality-headroom-2026-08-30.md)
(the completed on-cell sweep at the 400 mm ring standoff) rather than re-measuring
it. Where a question was already settled empirically there, this document says so
and moves on.

Every claim below is either quantified in-session or cites the evidence that
settles it. Claims marked *unverified* say so.

---

## Verdict by capability area

| Area | State | Evidence |
|---|---|---|
| Depth resolution | **At maximum** (1280x720) | Sensor max; headroom study measured 848x480 as a net loss for the bead (crest height -8%) |
| Colour resolution | **At maximum** (1920x1080) | Sensor max; bounds ChArUco corner precision downstream |
| Depth word size | **At maximum useful** (0.1 mm) | Live: `device reports 0.0001`, min step 0.10 mm. Already finer than the D435's stereo granularity (~0.19 mm @ 450 mm) — no longer the limiting factor anywhere |
| Laser power | **No headroom** (150) | Board plane RMS flat 1.16–1.19 mm across 90→360; the only monotonic effect is temporal jitter 0.130→0.096 mm, two orders below the 5 mm bead scale |
| Emitter | **On** (`emitter_enabled=1`) | Journal read-back; laser=0 is clearly worse for the board |
| Stereo / disparity shift | **Correctly left alone** | `disparityShift=60` changes nothing at 447 mm; 120 breaks the board outright (RMS 260 mm) |
| Visual preset | **One open candidate** | Medium Density (5) is the only setting that moved ring crest fidelity in a useful direction (+6% crest frac) for an acceptable board cost (+9% RMS). Wants a dated, caliper-scored on-cell trial — not a default flip |
| CUDA build | **Exercised on the hot path** | See below — this is the one that most needed checking |
| `align` removal | **Complete on the live path** | See below |
| `hole_filling` removal | **Complete** | `setup_depth_filters()` is `threshold → disparity → spatial → temporal → disparity_inv`; no `hole_filling` anywhere in the repo |
| IR (Y8) streams | **Deliberately unused, correctly** | Nothing reads them; enabling two 720p IR streams costs USB bandwidth on top of 1080p colour + 720p depth for no measurement gain |
| IMU | **Deliberately unused, correctly** | Camera pose comes from the robot |

### CUDA: is the rebuild actually being used?

Yes, on the single hottest per-frame path — and there is nothing further to claim.

librealsense 2.55.1 ships exactly **two** CUDA kernels (verified on the Jetson,
`~/librealsense/src/cuda/`): `cuda-conversion.cu` and `cuda-pointcloud.cu`.

- **`cuda-conversion` is on the live path.** `unpack_yuy2` is CUDA-guarded
  (`src/proc/color-formats-converter.cpp:58` → `rscuda::unpack_yuy2_cuda`), and the
  server requests `rs.format.bgr8` at 1920x1080@30, which forces exactly that
  YUY2→BGR8 conversion — ~62 Mpx/s now running on the GPU. This is the main reason
  idle service CPU fell 178.9% → 28.2%.
- **`cuda-pointcloud` is on a cold path only** — `rs.pointcloud()` appears once, in
  the scan-telemetry tick (~1 Hz), not in `getFrames`.
- **The depth filter chain has no GPU path at all, and no OpenMP path either.**
  A grep of `spatial-filter.cpp`, `temporal-filter.cpp`, `disparity-transform.cpp`
  and `threshold.cpp` for `RS2_USE_CUDA` or `omp parallel` returns **nothing**.

So the ~1 s/frame the filter chain costs the Nano is **irreducible inside
librealsense**; the Release/`-O3` half of the rebuild (7688 → 1055 ms depth grab)
was the entire win available there. Any further gain has to come from shortening
the chain or moving filtering to the host — not from more GPU flags.

### `align` removal

Complete on the live path. The systemd unit runs
`ExecStart=.../server/server_unicast_syncronous.py`, which contains no `align`.
Two other files in `server/` still call `rs.align` — `server_unicast_asyncio.py`
and `server_unicast_syncronous_dynamicRes.py` — but neither is run by the unit;
they are dead alternates. Worth deleting or marking as such so a future reader
does not take them for the live server.

Removing `align` also **closed the 2026-08-14 audit's headline defect**: the Jetson
used to align with *factory* intrinsics while the host back-projected with
*calibrated* ones, a +2.05% lateral scale error the characterization could not see.
Registration now happens once, on the host, through one consistent model, with no
resampling — and the TSDF integrates native depth through the true depth K and the
depth camera's own pose (`reconstruct.py:fuse_views`).

---

## Findings, ranked by whether they cost real quality

### 1. Nothing cross-checks `config.camera.resolution` against the greeting's colour size — HIGH (latent)

The host picks its colour `K` **by a config string** (`CameraConfig.K` indexes
`intrinsics[resolution]`), while the actual stream size arrives in the greeting.
Nothing compares them. Measured on a synthetic full-frame plane through the real
config and a real greeting:

```
correct 1080p K : u range  -186.2 .. 2148.1   centre-patch pts  448
stale   720p K  : u range  -124.1 .. 1432.1   centre-patch pts 1035
```

A stale 720p override compresses every point into the top-left ~44% of a 1080p
canvas and changes the centre-patch population by 2.3x. `ColorRegistered` even
holds both halves of the contradiction — `color_size` from the greeting, `uv` from
the config `K` — so the mismatch is one comparison away from being detectable.
Nothing raises and nothing is logged: the chroma gate simply samples the wrong
colour pixels and misclassifies bead vs board.

**Today this is correct** (both are 1920x1080), and the config file carries no
`resolution` key, so it takes the new default. But the only thing that made it
correct was the x1.5 migration firing once. Anyone writing `camera.resolution`
into `tasni.config.json` re-opens it silently.

**Fix:** compare `geometry.color_size` against `config.camera.size` where the
greeting is parsed, and refuse (or loudly warn) on a mismatch. `tools/cell_health.py`
now performs this cross-check as a pre-run gate, but the runtime still has none.

### 2. `_density_ratio` uses the FACTORY colour K while `uv` is built with the CALIBRATED K — MEDIUM

This is the programme's own explicitly-unresolved open question (Ruling R25's
follow-up). `ColorRegistered.uv` is produced by `project_to_color(..., K_color)`
with the host's **calibrated** K, but `_density_ratio()` divides by
`geom.color_K_factory`. Measured:

```
factory    fx,fy = 1362.15, 1362.21
calibrated fx,fy = 1334.81, 1336.21
valid_frac reads HIGH by a constant 1.0403  (+4.03%)
gate min_valid_depth_frac=0.5 therefore trips at a TRUE coverage of 0.4806
measured on a 100%-covered synthetic plane: valid_frac = 1.0297  (truth 1.0)
```

It **errs permissive**, so it cannot cause a false refusal — only a marginally
early DETECT — and `evaluate_depth_gate` clamps with `min(1.0, ...)`. But the
number is also persisted into scan records, where a 4% bias is not obvious to a
later reader. **Fix is one word**: pass the same calibrated K into
`ColorRegistered` that produced `uv`.

### 3. `auto_exposure_priority` is set on the wrong sensor — MEDIUM

`rs_config.configure_depth_sensor` sets `auto_exposure_priority` on the **depth**
sensor. On D400 that option is registered on the **colour** endpoint
(`librealsense/src/ds5/ds5-color.cpp:161`, `color_ep.register_pu(RS2_OPTION_AUTO_EXPOSURE_PRIORITY)`).
The live journal confirms the consequence:

```
RealSense: auto_exposure_priority unsupported on this device/build - skipped
```

So the guard that stops auto-exposure dropping the stream below 30 fps in dim light
is **absent from the colour stream this very programme doubled to 1080p** — the
stream most likely to miss a frame deadline. When AE stretches exposure past the
frame period, `wait_for_frames` stalls and the recovery supervisor rebuilds the
pipeline. Two `Camera stalled; rebuilding` events were logged in the last 24 h;
**the causal link is unproven** and there are other known causes (USB EPROTO).

The log line is also misleading: it reads as "this device cannot do it" when the
truth is "we asked the wrong sensor."

**Fix:** apply it to the colour sensor, and read back what the RGB endpoint's
default actually is (it is a UVC passthrough option, so librealsense does not force
a default — it inherits the device's).

### 4. `rs.pointcloud()` is constructed inside the telemetry loop — LOW

`server_unicast_syncronous.py:1142` builds a fresh `rs.pointcloud()` on every
telemetry tick. It is a processing block with internal allocation/caching that a
per-tick rebuild throws away. At ~1 Hz the cost is small, but hoisting it out of
the loop is free.

---

## Deferred-minor triage (from the programme's own reports)

Carried forward, judged against "does this cost real quality":

- **Worth doing** — the four above. Item 1 was flagged during the programme as
  "WORTH EYEBALLING IN TASK 13" and is still unaddressed in the runtime.
- **Worth doing, cheap** — `SurveyThresholds.border_margin_px` is now dead;
  `clean_measured_surface_mesh` keeps unused `K`/`width`/`height`; the unused
  `Rt_to_T` import in `tests/geometry_fixtures.py`.
- **Real test-quality gap** — the open3d test guard is `except: print; return`, so
  on a machine without open3d the entire TSDF rewrite would report PASS while
  executing nothing. (open3d 0.17.0 *is* installed here, so those tests genuinely
  ran — but the guard should fail, or skip, not pass.)
- **Accept as-is** — `backproject` not validating `depth.shape` against
  `geom.depth_size` (every current call site derives both from one source, and a new
  raise could break an archived take); `CameraGeometry`'s unusable generated
  `eq`/`hash`; the redundant `except (ValueError, json.JSONDecodeError)`;
  `n_grid_total` floor-vs-ceil overstating coverage ~0.5%; the single `conn.recv(64)`
  handshake read (a 13-byte write essentially never splits).
- **Already resolved** — `rs_config`'s unflushed `log=print` default (Task 6 injects
  `print(..., flush=True)`); the `unit_mm=1.0` default is unreachable from any live
  path (all three production call sites pass `frame.geometry.depth_unit_mm`).

## What was verified live during this review

Jetson `main cdc91f8`, service active, listening on 1024, camera free. Journal
read-back: `depth_units → device reports 0.0001`, `depth_unit_mm = 0.0999…`,
`laser_power → device reports 150`, `emitter_enabled → 1`,
`visual_preset left as-is at 0.0`, `depth (1280, 720) colour (1920, 1080)`, and **no**
`extrinsic layout check failed`. Host config migrated: 1080p fx 1334.8113
(= calibrated 889.8742 x 1.5), not the factory 1362.15. A live grab returned
0.1 mm/word, min step 0.10 mm, 92.3% valid — matching the headroom study's 92.2%
baseline. A stale (`MODE FULL`, no V2) client is refused with
`ERR protocol 2 required; send MODE FULL V2`.

All of the above is re-checkable in one command: `py -3.10 tools/cell_health.py`.
