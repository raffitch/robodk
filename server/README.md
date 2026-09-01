# Jetson camera server (vendored into the monorepo)

The **Jetson-side** code that streams the RealSense D435i to the host. It was once a
separate repo (`github.com/raffitch/realsense-ethernet`), vendored here so there is
**one repo** to work in and one place to deploy from.

## The live server is `server_unicast_syncronous.py` — and only that file

The systemd unit `realsense-camera` (`realsense-camera.service` in this directory)
runs exactly one script:

```
ExecStart=/home/jetson/EtherSenseServer/ethenv/bin/python \
          /home/jetson/robodk/server/server_unicast_syncronous.py
```

No other Python file here is executed by anything (the auto-pull shell script has
its own timer unit). `server_unicast_asyncio.py`
and `server_unicast_syncronous_dynamicRes.py` used to sit here looking like
alternates; they were pre-protocol-2 EtherSense variants that still called
`rs.align` and streamed 640x480 depth, could not have served a current host client,
and were **deleted on 2026-08-30** — recover them from git history
(`git show 55ffeac:server/server_unicast_asyncio.py`) if ever needed.

| File | Role |
|------|------|
| **`server_unicast_syncronous.py`** | **The live server.** Everything below describes it. |
| `handshake.py` | `parse_handshake()` — the one line a client sends. Imported by the server AND by the host test suite, so the two sides cannot silently drift. |
| `rs_config.py` | Device configuration **with read-back** (depth units, laser, preset, colour AE priority), temperatures, global time, the as-found advanced-mode dump. |
| `rs_geometry.py` | Depth/colour intrinsics + the depth→colour extrinsic, and the greeting it is packed into. |
| `scan_overlay.py` | Pure-numpy live work-rectangle math (density-cliff trim, colour edge cross-check). Imported by the server; unit-tested on the host. |
| `presets/` | The as-found ASIC JSON captured off this device (`custom-as-found-2026-08-29.json`), written by `tools/jetson_dump_asfound.py`. Reference only — **nothing loads it at runtime**. |
| `realsense-camera.service` | The systemd unit. |
| `jetson-autopull.{sh,service,timer}` | The ~2-minute auto-pull that restarts the camera when `server/` changes and no client is mid-capture. |
| `client_unicast_asyncio.py`, `robodk_3dscanning.py`, `robodk_client_*.py`, `nksr_reconstruct.py` | **Legacy, unreferenced.** Jetson-side counterparts of the pre-`tasni` macros. Nothing imports or runs them; they predate protocol 2. Kept only as historical reference — do not copy patterns out of them. |

## Protocol 2 in one paragraph

The server sends **raw, unaligned depth counts** plus a JSON **greeting** that says
exactly what those counts mean, and the host does all of the registration. This
replaced the old scheme, where the Jetson ran `rs.align(rs.stream.color)` and shipped
depth already resampled onto the colour grid. The alignment happened with the
camera's **factory** intrinsics while the host back-projected with its own
**calibrated** ones — a +2.05% lateral scale error that no characterization could
see (`docs/scan-audit-2026-08-14.md`). There is now **no `rs.align` anywhere in this
directory**, no resampling, and one consistent camera model on the host.

## Streams

| Stream | Format | Notes |
|---|---|---|
| Depth | **1280x720** `z16` @30 | `DEPTH_SIZE` — the D435i's top depth mode. |
| Colour | **1920x1080** `bgr8` @30 | `COLOR_SIZE` — ChArUco corner precision bounds every downstream number (audit R7). |
| Infrared | *not enabled* | Nobody reads it; enabling it costs USB bandwidth. |

Both come from **one** `rs.pipeline`, opened once at startup by `openPipeline()` and
shared by every client thread.

### Depth unit: 0.1 mm per word

`rs_config.DEPTH_UNITS_M = 0.0001` is **set on every open and read back**
(`configure_depth_sensor`). At the RealSense default of 1 mm/count a uint16 covers
65 m at 1 mm granularity — 10x coarser than this cell needs, for range it never uses.
At 0.1 mm the ceiling is 6.55 m, still far beyond the 1.5 m work volume, and
quantization stops being a visible term in a bead-height measurement.

The achieved value is carried in the greeting as `depth_unit_mm` (0.1). The host
multiplies **every** depth by it, so it is the single scale factor the whole
measurement chain hangs on: `_reread_depth_unit_mm()` re-reads it after a recovery
rebuild, falls back to what `openPipeline` just read off the reopened device, and
logs loudly rather than carrying a possibly-stale number forward in silence.

### Filter chain (on native depth, no decimation, no hole filling)

`setup_depth_filters()`:

```
threshold(0.15 .. 1.5 m) -> disparity_transform -> spatial -> temporal -> disparity_transform(inverse)
```

- **Threshold first** (`RS_DEPTH_MIN_M` / `RS_DEPTH_MAX_M`) so background depth is
  never smoothed *into* a surface edge (audit R5).
- **Spatial/temporal in disparity space**, which is where they are well-behaved.
- **No decimation** — full-resolution scan data is the point.
- **No hole filling.** A filled pixel is fabricated depth, and it gets fabricated
  exactly where the metrology cares: at surface edges.

The chain costs the Nano roughly a second per frame and has no CUDA or OpenMP path
inside librealsense, so that cost is irreducible short of shortening the chain
(`docs/sensor-layer-capability-review-2026-08-30.md`).

Two env levers exist for the crest-height A/B, both defaulting to current behaviour:
`RS_SPATIAL=0` drops the spatial filter, `RS_SPATIAL_SMOOTH_DELTA` lowers its
edge-preservation threshold. **Both arms are readable off an archived take**: the
greeting's `filters` list names the chain that ran, and `filter_options.spatial_smooth_delta`
carries the delta it ACTUALLY ran at — read back off the filter object, so an untouched
filter records librealsense's own default (20) rather than the `-1 = don't touch` env
value. `null` there means there was no spatial filter at all (or, if `spatial` IS in
`filters`, that the SDK refused the read-back).

### Depth is NOT aligned

`getFrames()` returns the **native** depth frame and the colour frame side by side.
The greeting carries `"aligned": false`, both intrinsics, and the depth→colour
extrinsic; the host back-projects depth through the **depth** intrinsics and
projects into colour itself when it needs a colour sample.

`rs_geometry.extrinsic_row_major()` transposes librealsense's column-major rotation
and then **checks** it against `rs2_transform_point_to_point` on a probe point. A
mismatch raises and the server refuses to start rather than serve a geometry that
would be wrong for every client.

## The greeting

One `json.dumps(...) + b"\n"` line, sent **once per connection**, before any frame,
to depth-carrying clients only (`MODE FULL V2` / `MODE BURST V2`). Colour-only and
telemetry connections get no greeting and read none.

```json
{"protocol": 2,
 "aligned": false,
 "depth_unit_mm": 0.1,
 "depth":  {"width": 1280, "height": 720, "fx": …, "fy": …, "ppx": …, "ppy": …, "model": …, "coeffs": [...]},
 "color":  {"width": 1920, "height": 1080, …},
 "depth_to_color": {"rotation_row_major": [[…]], "translation_mm": [x, y, z]},
 "filters": ["threshold", "disparity", "spatial", "temporal", "disparity_inv"],
 "device": {"serial": …, "fw": …, "librealsense": …,
            "color_auto_exposure_priority": …, "visual_preset": …, "laser_power": …},
 "temps": {"asic_c": …, "projector_c": …},
 "global_time_enabled": true|false|null}
```

Static geometry comes from `STATIC_GEOMETRY`; temperatures and global-time are read
live off the device at greeting time. Everything is taken from ONE
`_camera_snapshot()` so a concurrent rebuild cannot hand a client a mixture of two
opens.

**The greeting can go stale.** It is sent once and the frame stream carries no
generation marker, so if the camera is rebuilt mid-connection the client would keep
describing new frames with the old open's numbers. There is no way to re-send it in
band, so the server **drops the connection** instead (`_greeting_is_stale` /
`_stale_greeting_close`) and the client's existing reconnect path fetches a fresh
one. Every serving loop checks twice: once at the top, and once again with the frame
in hand immediately before `sendall`, because a filter-chain pass is long enough for
a rebuild to land entirely inside it.

## Handshake

One line, right after connect, parsed by `handshake.parse_handshake` (shared with the
host test suite). Read with a 0.5 s timeout; the socket then goes to a 10 s timeout
for sends.

| Client sends | Server does |
|---|---|
| `MODE FULL V2` | Greeting, then continuous `depth+colour` frames. |
| `MODE BURST V2` | `BURST READY\n`, the greeting, then the CAP/GET/CLEAR loop below. |
| `MODE COLOR [Q<n>] [H264 [B<kbps>]] [SCAN]` (or a bare `C`) | Colour only. **No greeting.** `Q<n>` clamps JPEG quality to 10..100; `H264` switches to the hardware encoder; `B<kbps>` clamps 500..20000 (default 4000); `SCAN` additionally publishes scan telemetry. |
| `MODE TELEMETRY` | Length-prefixed JSON telemetry side-channel; no greeting, no frames. |
| **anything else that wants depth** — no line at all, a bare `MODE FULL`, garbage | **Refused**: `ERR protocol 2 required; send MODE FULL V2\n`, then close. |

That last row is deliberate and is the **protocol-2-only** rule. A host that did not
restart after the protocol change must fail loudly at the handshake rather than
misread the JSON greeting as a 4-byte frame length and hang. It also means the old
embedded macros (`macros/3DScan.py` and friends, which send nothing) can no longer
talk to this server at all — `tasni.core.camera.CameraClient` is the supported client.

### Frame framing

Identical on the streaming and burst paths, so the host reuses one decoder:

```
<I depth_len><I color_len><d timestamp> + depth + colour
    depth  = lz4.frame(np.save buffer of the raw uint16 depth array)   # depth_len=0 for colour-only
    colour = JPEG (TurboJPEG, quality 100 unless Q<n> was given)
```

H.264 is the exception: `stream_h264` relays a raw Annex-B byte-stream with **no
per-frame header**, produced by `gst-launch-1.0` driving the Nano's NVENC
(`nvv4l2h264enc`, baseline, SPS/PPS inlined so a mid-stream client can start at the
next IDR). It is lossy and inter-frame — live preview only; every one-shot capture
takes the lossless JPEG/LZ4 path.

### Burst

`MODE BURST V2` opens an interactive session so a robot tour is not stalled by a
per-pose depth transfer over the cell's Wi-Fi (a full frame can take 6–11 s):

- `CAP` — grab + filter + compress one frame into a **RAM** buffer (max 64); reply
  `<I idx><I thumb_len><thumb JPEG>` for the live per-pose strip (`thumb_len=0` = skip).
- `GET` — `<I count>` then every buffered frame in the framing above.
- `CLEAR` — drop the buffer, reply `<I 0>`.
- `SET [k=v ...]` — one JSON line: `{"ok":true,"filters":[...],"filter_options":{...}}` (achieved values) or `{"ok":false,"error":"..."}`. A successful write retires the generation — sessions greeted before it end (the issuing one too, after the reply) and reconnect into a fresh greeting. A bare `SET` is a read. Overrides never survive a service restart.

The buffer is RAM-only and is dropped in `finally`, so an abandoned burst leaves
nothing on the Jetson. Command reads use a 180 s timeout (the robot is moving
between poses); `GET` uses 120 s.

### Telemetry

`scan_plane_telemetry()` fits the central depth patch, expands that plane across a
sparse full-frame sample anchored to the reticle, and returns live scan-guidance
data (distance, tilt, the trimmed work rectangle, coverage). It is produced **only**
by the `MODE COLOR H264 SCAN` feeder, about once a second, and `publish_scan_telemetry`
hands it to any `MODE TELEMETRY` connection as `<I len> + JSON`.

## Concurrency and recovery

`main()` binds `0.0.0.0:1024` with `listen(5)` and starts a **daemon thread per
client** (`_serve_client`), which always closes the socket — a raised handler used to
leak accepted sockets into CLOSE_WAIT until new clients could no longer connect.

All acquisition funnels through **`read_frames()`**, so one stall detector sees the
timeouts from every client thread. On 2026-08-29 the camera streamed for two hours,
stopped delivering, and then raised `Frame didn't arrive within 5000` on every
acquisition — 57 in a row — while the service stayed `active`, kept LISTENING and
kept accepting clients. The supervisor exists so that cannot recur silently:

| Constant | Value | Meaning |
|---|---|---|
| `FRAME_WEDGE_THRESHOLD` | 3 | Below this a stall is just a dropped frameset; librealsense recovers on its own. |
| `RECOVERY_BACKOFF_S` | 1, 5, 15 s | Pause before each reopen. Re-opening a USB device in a tight loop is how a Tegra host controller dies. |
| `MAX_REBUILDS_WITHOUT_PROGRESS` | 3 | A device can enumerate cleanly and still deliver nothing. |
| `HEALTHY_FRAMES_AFTER_REBUILD` | 30 (~1 s) | A rebuild is credited only once frames actually flow again. |

`_rebuild_pipeline` runs once on behalf of every waiting thread (under
`_camera_lock`), re-opens through `openPipeline()` — which also rebinds
`STATIC_GEOMETRY`, `ACHIEVED_OPTIONS` and `DEVICE_INFO` — re-reads `depth_unit_mm`,
and bumps `_camera_generation` last. If recovery fails, `_give_up()` prints
`CAMERA UNAVAILABLE: …` and exits; `Restart=always` / `RestartSec=3` (with no start
limit) then retries from scratch, which is a visible, diagnosable state rather than
a server that accepts clients it can never feed.

**Known, unfixed:** `read_frames` releases `_camera_lock` before calling
`current.wait_for_frames()`, so a thread can be inside `wait_for_frames` on a
pipeline that `_release_pipeline` then stops. Pre-existing; left alone deliberately.

## Environment variables

These three, and no others (`grep os.environ server/*.py`):

| Variable | Default | Effect |
|---|---|---|
| `RS_LASER_POWER` | `-1` = **leave alone** | IR projector power (0..360 on this device). The unit pins `Environment=RS_LASER_POWER=150`, matching the 2026-08-13 depth characterization. |
| `RS_VISUAL_PRESET` | `-1` = **leave alone** | `rs400` preset ordinal (3 high_accuracy, 4 high_density, 5 medium_density). |
| `RS_ASFOUND_DIR` | `/home/jetson/robodk-characterization` | Where the as-found advanced-mode JSON is written on each open. |

**Why the two acquisition knobs default to leave-alone.** RealSense option state
lives on the **device** and survives a service restart, so a silent default rewrites
the camera on every start — which is how it once ended up at 300 mW while the dated
depth envelope had been measured at 150. Leave-alone preserves the device's state and
the greeting *reports* it (`device.visual_preset`, `device.laser_power`), so the
running configuration is visible rather than assumed. In practice this device runs
the **Custom** preset (`visual_preset` = 0) — see `presets/custom-as-found-2026-08-29.json`.
It is specifically **not** High Accuracy: that preset raises the confidence threshold
and returns fewer but surer points, while the measured defect on this cell was
*missing coverage* at surface edges. Trial a change as an explicit experiment, with
its own before/after measurement (`tools/characterize_distance.py`) — not as a default.

Set unconditionally regardless of the above: `depth_units` (0.1 mm, the whole point
of protocol 2), `emitter_enabled`, and `auto_exposure_priority = 0` on the **colour**
endpoint — the latter lives on the colour sensor on the D400 series, and without it
AE can stretch colour exposure past the frame period in dim light, dropping below
30 fps, stalling `wait_for_frames` and costing a recovery rebuild.

## Deployment

Never edit this code on the Jetson. It clones **this** repo to `~/robodk`, follows
the branch checked out there, and `jetson-autopull.timer` pulls every ~2 minutes,
restarting `realsense-camera` **only** when the pulled diff touched `server/` and no
client is connected on :1024. So any change here restarts the live camera on the next
tick.

```
python tools/jetson_deploy.py status     # active? listening? timer? logs
python tools/jetson_deploy.py deploy     # immediate pull + restart from the current branch
python tools/jetson_deploy.py bootstrap  # (re)install service + auto-pull (idempotent)
```

Host-side tests that pin this file's behaviour (they stub `pyrealsense2`, so they run
on Windows):

```
py -3.10 -m pytest tests/test_camera_recovery.py tests/test_camera_greeting_coherence.py \
                   tests/test_rs_config.py tests/test_handshake.py tests/test_camera_wire.py \
                   tests/test_server_env.py tests/test_scan_telemetry_server.py -q
```

See also [`../docs/jetson-scanner.md`](../docs/jetson-scanner.md) (device details,
SSH, known issues) and
[`../docs/sensor-layer-capability-review-2026-08-30.md`](../docs/sensor-layer-capability-review-2026-08-30.md).
