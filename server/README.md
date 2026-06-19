# Jetson camera server (vendored into the monorepo)

These are the **Jetson-side** scripts that stream the RealSense D435i to the RoboDK
clients in `../macros/`. They were previously a separate repo
(`github.com/raffitch/realsense-ethernet`) — vendored here so there is **one repo** to
work in.

**Source:** branch `RT-3DScanning-D435` @ commit `954c950` (the branch the Jetson runs).

> The old `realsense-ethernet` repo kept variants on feature branches. For reference:
> - `RT-3DScanning-D435` — full 3D-scan production set (vendored here; superset)
> - `RT-AutoCharucoBoardCalibration-V1` — calibration-tuned variant (to reconcile when we
>   build the calibration app)
> - `ethersense-0.1/0.2`, `master` — older generic EtherSense bases

## What runs in production
The Jetson's **"Sync Server"** desktop shortcut runs **`server_unicast_syncronous.py`**
under the Python 3.10 venv (`~/EtherSenseServer/ethenv/bin/python`). That is the
authoritative server.

| File | Role |
|------|------|
| **`server_unicast_syncronous.py`** | **Production server.** Binds `0.0.0.0:1024`, RealSense 1280×720 depth+color @30, aligns depth→color, `spatial` filter only, streams continuously. |
| `server_unicast_syncronous_dynamicRes.py` | Server variant with selectable resolution |
| `server_unicast_asyncio.py` | Asyncio server variant (the "Async Server" shortcut) |
| `robodk_3dscanning.py` | Jetson-side counterpart of `macros/3DScan.py` |
| `robodk_client_syncronous_cv_calibrate*.py` | Counterparts of `macros/AutoCalibrate.py` |
| `robodk_client_ArucoToPoints.py` | Counterpart of `macros/ArucoToPlane.py` |
| `nksr_reconstruct.py` | NKSR neural meshing (also invoked by `3DScan.py` via WSL) |
| `client_unicast_asyncio.py` | Test client |

## Wire protocol (matches `macros/*.receive_data`)
16-byte header `<I depth_len><I color_len><d timestamp>`, then lz4-compressed depth
(`np.save` buffer) + JPEG color (TurboJPEG). Single client, continuous stream.

### Optional stream-mode handshake (backward compatible)
Right after connecting, a client *may* send one line to pick the stream:

| Client sends | Server streams |
|---|---|
| *(nothing)* / `MODE FULL` / unrecognized | **full depth+color** (default — legacy + scan clients are unaffected) |
| `MODE COLOR` (or a bare `C`) | **color-only**: `depth_len=0`, and the server skips depth align + spatial filter + lz4 entirely |

Color-only is for the live aiming preview + calibration (which never use depth). On the
Nano that path is the difference between ~0.5 fps and realtime (~30+ fps) — the cost was
per-frame **align+filter CPU**, not bandwidth. Depth/scan clients that just connect and
read keep getting byte-identical full frames. (`tasni.core.camera.CameraClient` sends
`MODE COLOR` only when `color_only=True`.)

## Known improvement targets (from docs/best-practices-review.md)
- No visual preset is set — apply **High Accuracy** for object scanning.
- Only `spatial` is enabled; decimation/temporal/hole-filling are commented out — adopt
  Intel's disparity-domain filter order.

## Deployment
This code is meant to run on the Jetson. See the planned systemd service + deploy flow
(`tools/jetson_deploy.py`) — the Jetson will `git pull` THIS repo and restart the service,
so you push once, here.
