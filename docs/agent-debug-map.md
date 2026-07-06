# Agent Debug Map

Current purpose: give future agents a low-token entry point into the Tasni app,
scan/calibration logic, RoboDK connection, and Jetson camera server.

Last updated: 2026-07-06. Active branch: `calibration-improvements`.

## Recent scan fixes (2026-07-06)

- `55f3c27` — **parked-scan jitter fixed**: the live HUD hold now uses symmetric
  hysteresis (`live_hold_release_frames`), so a parked arm shows zero jitter and only
  sustained motion releases the freeze. Live-verified (parked 124/124 held, 0.00 p-p;
  real-robot A6 move releases+tracks+refreezes).
- `d4e56bd` — **three scan defects**: (1) live overlay draws the density-TRIMMED
  rectangle to match lock/insert (server sends `trimmed_corners_color_mm`; host
  projects it calibrated); (2) reference mode wired (`generate_scan_targets` →
  `_reference_locate`, was dead code); (3) EDGE A lamp populated (advisory — real for
  an elongated platform via `edge_gate_min_aspect`, never blocks lock). Jetson
  auto-deployed the server change.
- Live 6-DOF HUD test (no code change): RANGE + TILT feedback + the hold verified on
  the real arm; X/Y-center/EDGE not testable (large white surface = crop mode);
  jog sign/`jog_invert` mapping to the pendant still to confirm. See
  `docs/live-robot-testing.md`.

## Start Here

Use this file before reading the long handoff docs.

| Need | Read / edit |
|---|---|
| Global agent rules and app overview | `CLAUDE.md` |
| Tasni app architecture | `tasni/README.md` |
| Scan workflow backend | `tasni/modules/scan/module.py`, `tasni/modules/scan/service.py` |
| Scan frontend | `tasni/webui/src/pages/Scan.tsx`, `tasni/webui/src/pages/AimHud.tsx` |
| Scan planner / surface survey | `tasni/modules/scan/planner.py`, `tasni/modules/scan/survey.py` |
| Fusion / mesh / work plane | `tasni/modules/scan/reconstruct.py`, `tasni/modules/scan/plane.py` |
| Calibration workflow | `tasni/modules/calibration/module.py`, `tasni/modules/calibration/service.py` |
| Shared camera transport | `tasni/core/camera.py`, `tasni/core/livepreview.py` |
| RoboDK API wrapper | `tasni/core/rdk_io.py`, `tasni/core/session.py` |
| Config defaults and knobs | `tasni/core/config.py`, `tasni.config.json` |
| Jetson camera server | `server/server_unicast_syncronous.py`, `server/scan_overlay.py` |
| Jetson deploy / restart | `tools/jetson_deploy.py`, `server/jetson-autopull.sh` |
| Tests | `tests/test_scan_job.py`, `tests/test_scan_planner.py`, `tests/test_calibration_job.py`, `tests/test_collision_guard.py` |

## Current Scan UX Contract

The intended scan workflow is:

1. Scan page auto-connects to RoboDK station and the real robot link.
2. Live camera feed starts automatically.
3. Operator jogs in TOOL frame using X/Y/Z and A/B/C guidance.
4. Finite platforms must be centered, level, edge-aligned, and framed before lock.
5. Oversized/crop surfaces can remain unframed; the reticle defines the fixed work crop.
6. `Lock & create targets` freezes one authoritative depth/color snapshot, then creates `TasniScan_*` targets.
7. Targets are inspected/dry-run in RoboDK, then scan run captures depth/color and fuses the work plane/mesh.

Important: display overlays are not the source of truth. Target creation uses the locked
snapshot and current RoboDK pose. The HUD should help the operator aim without changing
the actual lock data.

## Live Overlay And Dots

There are two producers of surface overlay coordinates:

- Live aiming: Jetson telemetry from `server/server_unicast_syncronous.py`.
- Lock snapshot: host survey from `tasni/modules/scan/survey.py`.

Recent fixes:

- `e452247 Keep scan lock overlay stable`: the frontend keeps the last live color-space
  overlay for display when lock publishes a snapshot, avoiding a visible rescale/jump.
- `0248178 Stabilize scan telemetry projection`: Jetson H.264 telemetry self-checks
  RealSense depth-to-color rotation orientation before vectorized projection.
- `d1e38e8 Throttle scan telemetry during preview`: scan telemetry runs at 1 Hz during
  preview so the Nano encoder is less likely to stall.

If dots look horizontally compressed:

1. Check the Jetson is actually on `calibration-improvements`: `py -3.10 tools/jetson_deploy.py status`.
2. Confirm `server/server_unicast_syncronous.py` on Jetson includes `SCAN_TELEMETRY_PERIOD_S`.
3. Check whether H.264 preview is active via `tasni.config.json` / `calibration.preview_codec`.
4. Inspect `server/server_unicast_syncronous.py::stream_h264` projection code before touching frontend scaling.

If FPS/no-signal dips (or the scan gate goes silent — no `gate` events on `/ws`):

1. **First remedy: restart the preview** — POST `/api/modules/scan/live/stop` then
   `/api/modules/scan/live/start`. The scan-telemetry channel stalls intermittently on
   Wi-Fi (and after robot-state changes); a restart reliably kicks it. See
   `docs/live-robot-testing.md` §5.
2. Check Jetson logs: `py -3.10 tools/jetson_deploy.py logs`.
3. Look for repeated broken pipes or reconnect loops.
4. H.264 path requires PyAV on the workstation and Nano NVENC. JPEG fallback is possible,
   but the JPEG server path must publish scan telemetry if dots/guidance are needed.

## Jetson Deploy Reality

The Jetson clones this repo at `/home/jetson/robodk`.

Current behavior:

- `tools/jetson_deploy.py deploy` pulls the current local branch, unless `JETSON_BRANCH`
  is set.
- `server/jetson-autopull.sh` follows the branch checked out on the Jetson, falling back
  to `main` if that branch has no remote.
- Camera service: `realsense-camera`.
- Stream port: `1024`.

Useful commands:

```powershell
py -3.10 tools\jetson_deploy.py status
py -3.10 tools\jetson_deploy.py deploy
py -3.10 tools\jetson_deploy.py restart
py -3.10 tools\jetson_deploy.py logs
```

## Local App Restart

Headless production server:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\serve.ps1 -Stop
powershell -NoProfile -ExecutionPolicy Bypass -File .\serve.ps1 -NoBuild -Port 8000
```

The app is at `http://127.0.0.1:8000`.

Useful probes:

```powershell
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/api/health
Invoke-WebRequest -UseBasicParsing -Method POST http://127.0.0.1:8000/api/modules/scan/connect
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8000/api/modules/scan/config
```

## Verification Sets

Focused scan:

```powershell
py -3.10 -m py_compile tasni\core\config.py tasni\modules\scan\service.py
pytest tests\test_scan_job.py tests\test_scan_planner.py
cd tasni\webui; npm run typecheck; npm run build
```

Broader regression:

```powershell
pytest tests\test_collision_guard.py tests\test_calibration_job.py tests\test_scan_job.py tests\test_scan_planner.py tests\test_sim_tour.py
```

## Existing Long Docs

Read only when needed:

- `docs/live-robot-testing.md`: **how to drive the real KUKA from a script safely**
  and read the live HUD — SIMULATE-vs-RUN_ROBOT trap, telemetry stalls, stale-hold
  reads, camera-tool IK, the continuous-monitor pattern. Read before any move script.
- `docs/jetson-scanner.md`: Jetson hardware/software/server details.
- `docs/scan-workbox-handoff.md`: scan dots and rectangle trim history.
- `docs/flat-workframe-validation-handoff.md`: current recommendation for the next
  milestone after fitted flat scan mesh.
- `docs/scan-coverage-dots-handoff.md`: older coverage-dot investigation.
- `docs/scan-survey-planner-handoff.md`: original surface-aware planner design; some status is stale.
- `docs/calibration-aiming-guidance-handoff.md`: calibration aiming UX.
- `docs/best-practices-review.md`: broader calibration/scan review.

Many handoff docs contain old commit hashes and status lines. Prefer this map for
current navigation, then use the long docs for reasoning history.
