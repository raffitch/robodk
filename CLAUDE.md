# RoboDK station: 241113_AutoScan

This folder is the editable working copy of the Python inside `241113_AutoScan.rdk`,
a ~117 MB RoboDK **binary** station. The station drives a **KUKA KR150 R2700** with an
Intel **RealSense** camera for ArUco-referenced **Open3D** 3D auto-scanning.

The `.rdk` is binary, so its embedded Python cannot be edited directly. The two bridge
scripts below extract that Python to `macros/*.py` (editable here) and push edits back in.

## The editing loop

1. **Extract** embedded macros to `macros/*.py` (overwrites them from the station):
   ```
   python rdk_extract.py "241113_AutoScan.rdk"
   ```
2. **Edit** the files in `macros/` (this is the real work; git tracks it).
3. **Sync** edits back into the station:
   ```
   python rdk_sync.py "241113_AutoScan.rdk"            # -> 241113_AutoScan.synced.rdk (safe)
   python rdk_sync.py "241113_AutoScan.rdk" --inplace  # overwrite the source .rdk
   ```
4. **Open the resulting `.rdk` in RoboDK yourself** to run/simulate.

## Macros (extracted from the station)

| File | Lines | What it does |
|------|-------|--------------|
| `macros/3DScan.py` | 440 | Main scan pipeline: RealSense capture over socket, point clouds via Open3D |
| `macros/3DScanParam.py` | 204 | Parameterized variant of the scan |
| `macros/ArucoToPlane.py` | 284 | Detects ArUco markers, computes the work-plane / reference frame |
| `macros/AutoCalibrate.py` | 199 | Camera/robot calibration routine |
| `macros/AutoScanTargetDefinition.py` | 144 | Generates dome of KUKA scan target poses |

There is also 1 GUI program (instruction list, not Python) in the station — not extracted.

## IMPORTANT: isolation from your open RoboDK window

RoboDK's API attaches to **any** running RoboDK instance. To make sure these scripts
never reach into a station you have open in the GUI, they connect through
`rdk_session.connect()`, which launches a **private, headless** instance
(`-NEWINSTANCE -NOUI -EXIT_LAST_COM`, `quit_on_close=True`) that disappears when the
script finishes. So: a script run does **not** show a window and does **not** disturb
your interactive session.

- `rdk_extract.py` / `rdk_sync.py` are **read-only on the source `.rdk`** unless you pass
  `--inplace`. Default output is a separate `*.synced.rdk`.
- Always keep a backup of `241113_AutoScan.rdk` before using `--inplace`.

## Hardware: the 3D scanner (Jetson + RealSense)
The scan/calibration macros are **clients** of a RealSense D435i on a Jetson Nano that
streams over TCP 1024. Full device details, software/firmware versions, the server-side
repo, and known operational issues are in **[docs/jetson-scanner.md](docs/jetson-scanner.md)**.
- SSH (passwordless, key installed): `ssh -i ~/.ssh/jetson_robodk jetson@10.12.171.70`
- Re-probe the device: `python tools/jetson_probe.py`
- Credentials live in `secrets/jetson.env` (**git-ignored — never commit**).

### Jetson camera server (now a monorepo + systemd service)
The Jetson server code is vendored here in **[server/](server/)** (was a separate repo).
It runs as a systemd service `realsense-camera` (auto-start on boot). Manage from here:
```
python tools/jetson_deploy.py status     # active? listening on 1024? logs
python tools/jetson_deploy.py deploy      # push, then this: git pull on Jetson + restart
python tools/jetson_deploy.py bootstrap   # (re)install the service (idempotent)
```
The Jetson clones THIS repo to `~/robodk` and tracks `main`. So: **one repo** — push here,
then `deploy`. See [docs/jetson-scanner.md](docs/jetson-scanner.md).

## Roadmap / status (updated 2026-06-19)
- ✅ Extract macros → monorepo → GitHub (private: `raffitch/robodk`)
- ✅ Best-practices research → [docs/best-practices-review.md](docs/best-practices-review.md)
- ✅ **#2 Jetson hardening**: monorepo, systemd service, deploy tool, cron cleanup
- ⏭️ **#1 NEXT: calibration app** — refactor `macros/AutoCalibrate.py` into an `rdkscan/`
  shared library (camera client, ChArUco, RoboDK I/O) + a GUI, and **add calibration
  quality metrics** (reprojection error + held-out validation poses) per the research.
  Keep TSAI solver; do NOT switch to PARK. This is the seed the scan app reuses later.
- Deferred: TSDF fusion (biggest scan-quality win), RealSense High-Accuracy preset +
  filter order (both in `server/server_unicast_syncronous.py`), Tailscale for off-LAN.

## Notes
- Requires the `robodk` package (installed under Python 3.10) and RoboDK at `C:\RoboDK`.
- Loading the 117 MB station takes a minute or two per script run — expected.
- `*.synced.rdk` is git-ignored; the source `.rdk` is large — see `.gitignore`.
