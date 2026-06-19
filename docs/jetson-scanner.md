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

## How the server is started — NOW a systemd service (2026-06-19)
The camera server runs as a **systemd service** `realsense-camera` (auto-start on boot,
auto-restart on crash). It runs `server/server_unicast_syncronous.py` from the monorepo
clone at `~/robodk` under the Python 3.10 venv, and sets the fan to max on start.

```bash
# from the workstation (uses tools/jetson_deploy.py):
python tools/jetson_deploy.py status     # active? listening on 1024? recent logs
python tools/jetson_deploy.py deploy      # git pull ~/robodk + restart service
python tools/jetson_deploy.py restart     # bounce the service
python tools/jetson_deploy.py bootstrap   # (re)install the service (idempotent)

# on the Jetson directly:
sudo systemctl status realsense-camera
journalctl -u realsense-camera -f
```
Unit file: `server/realsense-camera.service` (installed to
`/etc/systemd/system/`). The Jetson tracks this repo's `main`; server code is in `server/`.
The legacy `/etc/crontab` autostart (39 broken lines) has been removed
(backup: `/etc/crontab.pre-cleanup.bak`).

### Legacy manual start (fallback only — don't run alongside the service)
The old desktop shortcuts still exist; both run the venv Python against a script in
`~/realsense-ethernet/`. Don't use them while the systemd service is active — they'd fight
over port 1024.

- **"Jetson-Realsense Async Server"**
  ```
  ~/EtherSenseServer/ethenv/bin/python ~/realsense-ethernet/server_unicast_asyncio.py
  ```
- **"Jetson-Realsense Sync Server"** (the fuller / primary one):
  ```bash
  echo '<sudo-pw>' | sudo -S sh -c 'echo 255 > /sys/devices/pwm-fan/target_pwm' \  # fan -> 100%
    && cd ~/realsense-ethernet && git pull \                                        # pull latest server code
    && ~/EtherSenseServer/ethenv/bin/python ~/realsense-ethernet/server_unicast_syncronous.py
  ```
  (sudo password embedded in the shortcut; stored in `secrets/jetson.env` as
  `JETSON_SUDO_PASSWORD`.)

So before any scan/calibration: **start the server** (double-click the shortcut on the
Jetson, or run the equivalent over SSH). The sync variant also force-cools the board and
`git pull`s the newest server code first.

## ⚠️ Things to know (affect scanning)
1. **Server now auto-starts at boot** via systemd (`realsense-camera`) and restarts on
   crash. Port 1024 should be open whenever the Jetson is up. ✅ (fixed 2026-06-19)
2. **Legacy `/etc/crontab` autostart removed** — it was 39 broken lines pointing at a
   non-existent `~/EtherSense` dir, spawning failing processes every minute. Gone.
3. **Flaky connectivity** — SSH to `10.12.171.70` times out intermittently (the tooling
   auto-reconnects/retries). Scans depend on a stable link — worth investigating the
   physical/Wi-Fi link as a separate reliability item.
4. **sudo password = login password** (`JETSON_SUDO_PASSWORD` in secrets); the `akuk` in
   the old desktop shortcut was stale/wrong.
4. **Two server variants** — async vs sync; both emit the same port-1024 frame format.
   Confirm which one your macros were validated against.

## How to re-probe
```bash
python tools/jetson_probe.py          # installs key (idempotent) + full report
python tools/jetson_probe.py --no-key # report only
```
