# 3D Scanner Host — Jetson Nano

The 3D scanner is an **Intel RealSense D435i** attached to an **NVIDIA Jetson Nano**.
The Jetson streams depth+color frames over Ethernet (TCP) and the RoboDK macros in
`macros/` connect to it as clients. This doc records what's actually on the device, as
probed on **2026-06-19** via `tools/jetson_probe.py`.

> Credentials are **not** in this repo. They live in `secrets/jetson.env` (git-ignored).
> An SSH key (`~/.ssh/jetson_robodk`) was installed on the Jetson, so access is now
> **passwordless**: `ssh -i ~/.ssh/jetson_robodk jetson@10.12.171.70`.

## Connection
| | |
|---|---|
| Host (scan network) | `10.12.171.70` — used by `3DScan.py`, `3DScanParam.py` |
| Host (other network) | `10.5.5.19` — used by `AutoCalibrate.py`, `ArucoToPlane.py` (same camera, different subnet? **open question**) |
| User | `jetson` |
| Stream port | TCP **1024** |
| SSH | key-based (`jetson_robodk`); password in `secrets/jetson.env` |

## Software stack (as of 2026-06-19)
| Component | Version |
|---|---|
| Board | Jetson Nano (`t210ref`), hostname `jetson-desktop` |
| OS | Ubuntu 18.04.6 LTS (Bionic) |
| L4T / JetPack | **L4T R32.7.6** (JetPack 4.6.x), build dated 2024-11-05 |
| Kernel | 4.9.337-tegra, aarch64 |
| System Python | 3.6.9 (`/usr/bin/python3`) |
| Built-from-source Python | 3.10.11 (`~/Python-3.10.11`) — used by the server venv `ethenv` |
| **librealsense** | **2.55.1** (`2.55.1-0~realsense.3335`) — recent, late 2024 |
| pyrealsense2 | installed (old binding, no `__version__`) |

> Jetson Nano is EOL at JetPack 4.6.x — R32.7.6 is effectively the latest it can run.
> No newer L4T/JetPack is available for this board.

## Camera
| | |
|---|---|
| Model | **Intel RealSense D435i** |
| Serial | `112222071901` |
| **Firmware** | **05.16.00.01** |
| USB id | `8086:0b3a` (D435i) |

FW `5.16.00.01` is aligned with librealsense 2.55's recommended D435i firmware, so the
SDK/firmware pairing is healthy. A firmware backup exists at
`~/realsense_firmware_backup/` and `~/.150423062145.*.bin`.

## The streaming server (camera side)
Two relevant directories in `~`:

- **`~/EtherSenseServer/`** — based on the open-source *EtherSense* (RealSense-over-
  Ethernet) project. Contains `EtherSenseServer.py`, `server_unicast_asyncio.py`,
  `server.py`, `servercolor.py`, etc. All bind **port 1024**. Has an
  `AlwaysRunningServer.bash` keep-alive wrapper. Runs in venv `ethenv` (Python 3.10).
- **`~/realsense-ethernet/`** — a **git repo** →
  `https://github.com/raffitch/realsense-ethernet.git`. This holds the **server-side
  counterparts to our macros**:
  | Jetson script | Pairs with RoboDK macro |
  |---|---|
  | `robodk_3dscanning.py` | `3DScan.py` |
  | `robodk_client_ArucoToPoints.py` | `ArucoToPlane.py` |
  | `robodk_client_syncronous_cv_calibrate*.py` | `AutoCalibrate.py` |
  | `server_unicast_*.py` | the stream the macros read |
  | `nksr_reconstruct.py` | NKSR neural meshing (also called by `3DScan.py` via WSL) |

**Wire format** (what the macros decode): 16-byte header
`<I depth_len><I color_len><d timestamp>`, then `depth` (lz4-compressed `.npy`) +
`color` (JPEG). Matches `receive_data()` in the macros.

## ⚠️ Operational issues found (affect scanning today)
1. **Server is NOT running** — nothing was listening on port 1024 during the probe.
   A scan/calibration would fail to connect until the server is started.
2. **Autostart is likely broken** — root's crontab launches
   `cd /home/jetson/EtherSense; ./AlwaysRunningServer.bash`, but the directory is
   `EtherSense**Server**` (no `~/EtherSense` exists), so the `cd` fails.
3. **`AlwaysRunningServer.bash` is stale** — it runs `python EtherSenseServer.py` with a
   **Python 2.7** `PYTHONPATH`, but the working stack is Python 3.10 + librealsense 2.55.
   The real production server is almost certainly one of the `server_unicast_*` scripts
   in `~/realsense-ethernet/`, started another way (or manually).
4. **Flaky connectivity** — one SSH attempt timed out during probing; scans depend on a
   stable link to `10.12.171.70`.

### To start the server manually (until autostart is fixed)
```bash
ssh -i ~/.ssh/jetson_robodk jetson@10.12.171.70
cd ~/realsense-ethernet
# identify the live server variant, then run it under the py3.10 venv, e.g.:
# source ~/EtherSenseServer/ethenv/bin/activate && python server_unicast_syncronous_dynamicRes.py
```
(Exact command TBD — needs confirming which variant is the production one.)

## How to re-probe
```bash
python tools/jetson_probe.py          # installs key (idempotent) + full report
python tools/jetson_probe.py --no-key # report only
```
