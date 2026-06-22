# RoboDK station: Tasni

This folder is the editable working copy of the Python inside `Tasni.rdk`
(renamed from `241113_AutoScan.rdk`),
a ~117 MB RoboDK **binary** station. The station drives a **KUKA KR150 R2700** with an
Intel **RealSense** camera for ArUco-referenced **Open3D** 3D auto-scanning.

The `.rdk` is binary, so its embedded Python cannot be edited directly. The two bridge
scripts below extract that Python to `macros/*.py` (editable here) and push edits back in.

## The editing loop

1. **Extract** embedded macros to `macros/*.py` (overwrites them from the station):
   ```
   python rdk_extract.py "Tasni.rdk"
   ```
2. **Edit** the files in `macros/` (this is the real work; git tracks it).
3. **Sync** edits back into the station:
   ```
   python rdk_sync.py "Tasni.rdk"            # -> 241113_AutoScan.synced.rdk (safe)
   python rdk_sync.py "Tasni.rdk" --inplace  # overwrite the source .rdk
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
- Always keep a backup of `Tasni.rdk` before using `--inplace`.

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
python tools/jetson_deploy.py status        # active? listening on 1024? auto-pull timer? logs
python tools/jetson_deploy.py deploy         # manual: git pull on Jetson + restart now
python tools/jetson_deploy.py setup-autopull # install the auto-pull timer (idempotent)
python tools/jetson_deploy.py bootstrap      # (re)install service + auto-pull (idempotent)
```
The Jetson clones THIS repo to `~/robodk` and tracks `main`. So: **one repo** — and it
**auto-pulls `origin/main` every ~2 min** (systemd timer), restarting the camera only
when `server/` changed and no client is mid-capture. So normally: **just push to `main`**
and the Jetson deploys itself; `deploy` is only for an immediate push-and-restart. A
feature branch reaches the Jetson only once merged to `main`. See
[docs/jetson-scanner.md](docs/jetson-scanner.md).

## North star (the actual goal)
Build **ONE external control-panel app** (Python, drives RoboDK over its API) with a clean
interface where the user picks what to do — **calibrate / scan / locate-ArUco / define
targets** — replacing today's scattered embedded macros, OpenCV windows and tkinter popups.
The app sits on a shared **`rdkscan/`** library (camera client, ChArUco, RoboDK I/O, config)
that every action reuses. RoboDK stays the orchestrator; the Jetson stays the camera server.
**We build it one module at a time, starting with calibration — but design for expansion
from the start** (the calibration module must not be a one-off; its library + GUI shell are
the foundation the scan/aruco/target modules plug into).

### Long-term vision — a robotic-fabrication PLATFORM (à la Aibuild / ai-build.com)
The end goal is bigger than scanning: a **platform that hosts ALL of the user's robot
workflows** — scanning, calibration, **3D printing / additive**, and whatever comes next —
each as a pluggable **workflow module** on a shared core (robot/RoboDK connection, camera,
config, job runner, live monitoring, logging). Inspiration: **Aibuild** — software that
*orchestrates and increasingly automates existing* robotic-manufacturing tools (they target
KUKA arms + gantries: toolpath, printing, simulation) rather than replacing them; over time,
trending toward more automation/AI. So architect the app as a **module registry + shared
services**, not a calibration tool with a few add-ons. The calibration module is module #1.
The shared core (`rdkscan/` + the GUI shell + config) IS the platform; modules plug in.

## The app: `tasni/` (platform) — see [tasni/README.md](tasni/README.md)
The control-panel app lives in **`tasni/`**: a module-registry + shared-services
platform. Backend = FastAPI (`tasni/webapp`, API-only + serves the build);
frontend = **React + Vite + TypeScript** (`tasni/webui`) with a **Dashboard**
landing and per-module pages (calibration is one module, not the front door).
Package name: **`tasni`**. RoboDK connection mode: **`attach`** (binds the running
GUI; if it has no station with the robot, the app opens **`Tasni.rdk`** into it so
you drive the real cell, not an empty station). Run it on Windows with
**`.\start.ps1`** (or `start.bat`); dev = backend + Vite hot-reload on :5173,
`.\start.ps1 prod` builds + serves on :8000. (`start.sh` is the Git Bash equivalent.)

## Roadmap / status (updated 2026-06-22)
- ✅ Extract macros → monorepo → GitHub (private: `raffitch/robodk`)
- ✅ Best-practices research → [docs/best-practices-review.md](docs/best-practices-review.md)
- ✅ **#2 Jetson hardening**: monorepo, systemd service, deploy tool, cron cleanup
- ✅ **#1 Calibration module = the app's first slice** (merged to `main`; local commits
  not yet pushed to `origin`). Refactored `macros/AutoCalibrate.py` into the `tasni`
  core + a React web app, with the missing **quality metrics** (reprojection px,
  held-out validation px, board-consistency mm). Kept TSAI (no PARK) + optional
  reprojection refinement. Now **RealSense-only + real-robot**: forced `Realsense`
  tool, **no taught pose** — a **live aiming HUD** (DETECT·DISTANCE·ANGLE lamps over the
  live camera; `core/livepreview.py` + `core/aiming.py`) gates **Create targets**,
  which **auto-generates** reachable poses (cone+roll, IK-filtered) around the robot's
  *current* pose and leaves `TasniCalib_*` in RoboDK to inspect.
  Single-source-of-truth **printable board** (default 8×6 @ 30 mm fits A4 1:1) + visual
  preview, no "matching" step. Launches as a **standalone app window** (`.\start.ps1`).
  - ✅ **Hardware-tested on the KUKA — verdict PASS** (held-out reproj ~0.9 px, board
    consistency ~0.8 mm). The robot-moving path (Create targets → dry tour → Run →
    return-to-start) runs on the real cell. TSAI finding retained: OpenCV TSAI is
    fragile near a ~180° camera→flange mount (PARK/HORAUD/ANDREFF stay exact), so
    `solver_method="best"` ranks all methods and the metrics expose a bad solve.
  - ✅ **Camera intrinsics handled under the hood.** The D435i ships its RGB stream
    with *zero* distortion (Intel calibrates depth/IR, not the RGB lens), which made
    the hand-eye solve borderline. The first hand-eye Run now **auto-calibrates** K +
    lens distortion from its own captured `TasniCalib_*` views (k3 fixed), applies
    them live (no restart) + persists, and gates on a marker so it runs **once**. An
    optional full-frame capture for best edge accuracy exists (API only; no UI). The
    intrinsics self-check compares *recovered vs configured* distortion (not vs zero).
    `tools/jetson_intrinsics.py` reads the camera's factory intrinsics off the Jetson.
  - ✅ **Workspace guardrails**: collision-screened pose generation + a SIMULATE
    **dry tour** (reachability / collision / return-to-start) before the real run.
  - Next ideas: model the board + table as collision objects (so a pose that would
    bump the physical board is auto-filtered, not just tool↔arm); live 3D viewport.
- Then integrate the rest into the same app: scan (with **TSDF fusion** — biggest quality
  win), ArUco-to-plane, target generation. RealSense High-Accuracy preset + filter order
  live in `server/server_unicast_syncronous.py`. Tailscale (off-LAN) deferred.

## Notes
- Requires the `robodk` package (installed under Python 3.10) and RoboDK at `C:\RoboDK`.
- Loading the 117 MB station takes a minute or two per script run — expected.
- `*.synced.rdk` is git-ignored; the source `.rdk` is large — see `.gitignore`.
