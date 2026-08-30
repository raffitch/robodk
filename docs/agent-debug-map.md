# Agent Debug Map

Current purpose: give future agents a low-token entry point into the Tasni app,
scan/calibration logic, RoboDK connection, and Jetson camera server.

Last updated: 2026-08-28. Active branch: `main` (`calibration-improvements` merged in
`51849b1`, 2026-08-25).

> **OPEN BLOCKER — live print, the arm does not move.** If you are picking that up, read
> **[live-print-next-session.md](live-print-next-session.md)** first; the evidence is in
> [live-print-dispatch-handoff-2026-08-28.md](live-print-dispatch-handoff-2026-08-28.md).
> New agents: start at [../AGENTS.md](../AGENTS.md).

## Two-path workframe survey (2026-08-13)

Implements `docs/scan-workframe-two-path-plan.md` (design) via
`docs/scan-workframe-implementation-plan.md` (17-task plan, all merged; 393 tests
green). Three ways to establish a scan workframe now exist side by side, all
producing the same `LockedWorkframeSurvey` contract:

- **Compact** (`mode="compact"`): the whole platform fits in one camera view at
  the aiming standoff — the existing single-lock path (`lock_scan_surface` in
  `service.py`), which still selects compact-vs-crop via the pre-existing
  `surface_mode` heuristic (`"full"` if the fitted rectangle stays in frame, else
  `"crop"` — `service.py:1112`), NOT via `classify_compact` (see the
  `classifier.py` bullet below — **known gap**).
- **Five-position survey** (`mode="five_position"`): a platform too large for one
  view — guided CENTER + four-corner pendant-jog capture sequence, driven by
  `SurveyPanel.tsx` and the `/survey/*` routes.
- **User-specified region** (`mode="user_specified"`): the old fixed-crop fast
  path, relabeled with honest provenance (`user specified - plane measured,
  boundary declared`) — operator-entered rectangle dimensions projected onto the
  measured plane, via `POST /surface/region`.

New backend modules (`tasni/modules/scan/`):

- `survey_contract.py` — the immutable `LockedWorkframeSurvey` /
  `CaptureRecord` / `RobotStateSnapshot` frozen dataclasses (spec §11) that ALL
  three paths funnel into; also `PROVENANCE_*` / `MODE_*` constants, clockwise
  corner ordering (`order_corners_clockwise`), and capture-freshness checks.
- `classifier.py` — pure decision logic (no camera/robot/RoboDK access) for
  compact-vs-large eligibility at `d*` (spec §6): guard band, boundary
  detection, centering, tilt, rectangle-identity-across-frames, and predicted
  scan-coverage (predict-only planner run). Returns `CompactEligibility`. **Not
  wired into any production caller as of Task 17** — `classify_compact` has full
  unit coverage (`tests/test_scan_classifier.py`) but grep finds zero callers
  outside that test file; `lock_scan_surface` never imports it. The actual
  compact/crop decision at lock time is still the older `surface_mode`
  full/rectangle-in-frame heuristic noted above. Confirm whether this is
  intentional (spec §6's eligibility gates folded into that heuristic some other
  way) or a genuine missed-wiring gap before relying on `classify_compact`'s
  guard/identity/coverage checks actually protecting a real lock.
- `rect_fit.py` — global plane + constrained-rectangle fit for the
  five-position survey (spec §7): `fit_global_plane` (RANSAC over all five
  captures' plane points) and `solve_constrained_rectangle` (closed-form
  total-least-squares over one shared orientation `theta` + 4 independent edge
  offsets — 5 DOF vs. the unconstrained 8 DOF). Its module docstring is the
  canonical statement of the `discrepancy_mm` limitation corrected in
  `docs/scan-workframe-two-path-plan.md` §7 below — read it before trusting
  either diagnostic field.
- `corner_evidence.py` — extracts base-frame corner point + two adjacent edges'
  points from one corner capture (boundary polygon for direction, depth +
  camera pose for metric geometry). Interior direction is derived from the
  polygon's own winding order (not a vertex-mean), so it stays correct on
  non-convex frame-clipped contours.
- `five_position.py` — the guided capture state machine (`SURVEY_STEPS =
  ("center", "corner1", "corner2", "corner3", "corner4")`): `add_capture`,
  `recapture`, `finish` (fits the global plane + rectangle, gates on
  coplanarity/discrepancy/corner-agreement, emits the quality report and a
  `LockedWorkframeSurvey`). Capture orchestration (camera/robot I/O) lives in
  `service.py`; this module is pure geometry/state.
- `tasni/core/characterize.py` — pure-geometry (mm, no hardware) metrics for
  picking the validated optimal standoff `d*` (spec §5, Phase 0):
  `DistanceTrial`, `plane_metrics`, `choose_dstar` (closest distance that
  passes every gate, not the best-scoring one). Defines `CHARACTERIZATION_DIR`
  (`characterization/`, git-ignored) and `latest_characterization()` — the
  dependency runs `tools/ -> tasni/core/`, never the reverse, because `tools/`
  is excluded from packaging.

New CLI tool: `tools/characterize_distance.py` — interactive, sweeps operator-
jogged standoffs over the ChArUco board, writes a dated
`characterization/characterization-YYYYMMDD.json`. Headless-importable (no
camera/robot/RoboDK touched above `main()`) so `tests/test_characterize.py` runs
without hardware. `lock_scan_surface` reads the latest file back and
warns/refuses on missing or stale characterization via `ScanConfig`'s
`calibration_max_age_days` / `calibration_expiry_hard_fail`.

New routes under `/api/modules/scan` (`tasni/modules/scan/module.py`):

- `POST /surface/region` — user-specified region fast path (§2, §4 above).
- `POST /survey/begin` — start a five-position survey (clears any prior state).
- `GET /survey/state` — current step, accepted captures, accumulating
  `corners_base`, warnings (polled by `SurveyPanel.tsx` and pushed on the
  `survey` websocket event).
- `POST /survey/capture` — authoritative jog→stop→measure capture for the
  current step; advances the step on acceptance.
- `POST /survey/recapture` — redo a weak/rejected corner without restarting.
- `POST /survey/finish` — fit the global plane + constrained rectangle, run the
  coplanarity/discrepancy/corner-agreement gates, return the quality report and
  (on pass) lock the `LockedWorkframeSurvey`.
- `POST /survey/cancel` — abandon the in-progress survey.

New frontend: `tasni/webui/src/pages/SurveyPanel.tsx` — offered by `Scan.tsx`
whenever `gate.surface_mode === "crop"` (platform too large for one view): a
"Begin guided survey" button starts it, mounting the panel in place of the
normal lock controls (`Scan.tsx:845`, `surveyActive` state; also auto-resumes
on page mount if a survey is already in progress server-side, so a refresh
can't silently discard captured corners). Renders the CENTER→C1→C2→C3→C4→REVIEW
step sequence, a top-down SVG polygon diagram (see the chirality caveat in the
two-path plan's Hardware validation TODO), and the quality report gate before
`Create targets`.

New `ScanConfig` keys (`tasni/core/config.py`), grouped by the section comment
that introduced them:

- Compact-eligibility classifier (§6): `compact_guard_uv`,
  `compact_center_tol_uv`, `compact_identity_frames`, `compact_identity_tol_uv`.
- Five-position survey (§7): `survey_capture_max_age_s`,
  `survey_coplanar_warn_mm`, `survey_coplanar_reject_mm`, `survey_edge_band_mm`,
  `survey_rect_discrepancy_mm`, `survey_corner_agreement_mm`,
  `survey_min_edge_points`, `survey_plane_inlier_band_mm`,
  `survey_corner_min_plane_coverage_frac`.
- Tiled close-range tour over a five-position-surveyed rectangle:
  `survey_tour_overlap`, `survey_tour_views_per_tile`,
  `survey_tour_max_contiguous_empty_tiles`.
- Distance characterization / calibration-age gate (§5, §10, Phase 0):
  `calibration_max_age_days`, `calibration_expiry_hard_fail`.

Read `docs/scan-workframe-two-path-plan.md` for the design and its Hardware
validation TODO (deferred, cell-only checks); `docs/scan-workframe-
implementation-plan.md` for the 17-task build log (each task's brief/report
pair is under `.superpowers/sdd/scan-workframe-implementation-plan/`, and
`progress.md` there records every review finding, including the
`discrepancy_mm` translational-blindness discovery and the corner-aiming /
SVG-chirality items later moved into the two-path plan's TODO section).

## Recent scan fixes (2026-07-06)

- **SAM (point-prompted) live work boundary**: for low-contrast scenes the colour layer
  abstains on (the green-mat-on-gray-table), the blue rectangle is now segmented by a
  learned point-prompted model (`tasni/modules/scan/sam_boundary.py`, ONNX on the HOST —
  the Jetson never runs SAM). Verified on the real cell: EdgeSAM hugged the mat (score
  0.98) where colour could only fall back to depth. Model-agnostic (reads the ONNX graph
  signature → EdgeSAM's simplified decoder AND MobileSAM's standard SAM decoder both drop
  in). Runs in a **background thread** (`SamBoundaryWorker`) so the ~450 ms/frame inference
  never hitches the ~6 fps video; publishes the SAME `boundary` /ws event (no frontend
  change). Config: `scan.boundary_engine` (`color` | `sam` | `sam_then_color`, default
  `sam_then_color` = SAM primary, colour fallback) + `sam_*` knobs. Shared mask→rectangle
  tail factored into `color_boundary.mask_to_boundary`. **Default weights = EdgeSAM
  (S-Lab non-commercial license)**; MobileSAM (Apache-2.0) swaps in via config. Weights are
  NOT committed — `py -3.10 -m pip install -e .[sam]` then `py -3.10 tools/download_sam.py`
  (see `models/README.md`). **Windows gotcha:** onnxruntime must load BEFORE PySide2/Qt
  (RoboDK `robolink` pulls Qt in; shiboken2's import hook otherwise breaks onnxruntime's
  DLL init) → pre-loaded in `tasni/__init__.py` + `tests/conftest.py`. Full handoff +
  status: `docs/scan-boundary-sam-handoff.md`.
- **Live COLOR work boundary (`1dd8d21`)**: the blue work rectangle is now segmented from
  the color frame at video rate (`tasni/modules/scan/color_boundary.py`, reticle-seeded
  Lab distance) and published on a `boundary` /ws event — bypassing the noisy 1 Hz depth
  telemetry + the anti-jitter freeze (fixes "rectangle only updates on Refresh / laggy").
  Abstain-safe: falls back to the depth outline on low-contrast scenes. **Next: add SAM
  (point-prompted) for hard scenes — see `docs/scan-boundary-sam-handoff.md`** (the
  `boundary` event is the ready drop-in seam; no frontend change needed).
- **"Refresh view" button / `POST /live/refresh`**: manual escape hatch for a stale live
  projection (driver not mirroring the arm + a lateral jog slips past the hold + vision
  escape). Drops the anti-jitter hold + pose anchor so the reading re-settles at the
  current pose; keeps video streaming + the distance target continuous. `module.py`
  (`threading.Event` consumed by the analyze loop) + `Scan.tsx`. See
  `docs/live-robot-testing.md` §4.
- **RANGE target moving-goalpost fixed**: the distance target (`ideal_distance_mm`) is
  now a STABLE framing standoff (physical-size based, `frame_margin` bumped 1.05→1.12 so
  the aim point sits inside the frame, not on the crop edge). In `crop` mode the target
  HOLDS the value latched while framed instead of collapsing to `accurate_min` — a small
  over-nudge into overrun no longer snaps the goal to 300 mm and drives the operator even
  closer (the "target 592 → jumps to 300 → unreachable" report). Only a genuinely
  oversized/never-framed surface falls to `accurate_min`. Host-only
  (`tasni/modules/scan/service.py::live_scan_telemetry_payload`, `module.py` latch on
  `surface_mode=="full"`, `config.py` frame_margin). See `docs/live-robot-testing.md` §4.
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
| Global agent rules and app overview | `AGENTS.md` (any tool), then `CLAUDE.md` |
| **Live print blocker — what to run next** | `docs/live-print-next-session.md` |
| Live print blocker — the evidence | `docs/live-print-dispatch-handoff-2026-08-28.md` |
| Dispatch diagnostics / bisect ladder | `tasni/core/rdk_io.py` `dispatch_program`, `tools/dispatch_bisect.py` |
| Tasni app architecture | `tasni/README.md` |
| Cylinder/extrusion current state and status -5 diagnosis | `docs/extrusion-current-handoff.md` |
| Cylinder original requirements (historical) | `docs/HANDOFF_EXTRUSION_CYLINDER.md` |
| Cylinder legacy behavior and valve mapping | `docs/extrusion-legacy-trace.md` |
| Scan workflow backend | `tasni/modules/scan/module.py`, `tasni/modules/scan/service.py` |
| Scan frontend | `tasni/webui/src/pages/Scan.tsx`, `tasni/webui/src/pages/AimHud.tsx` |
| Scan planner / surface survey | `tasni/modules/scan/planner.py`, `tasni/modules/scan/survey.py` |
| Two-path workframe survey (compact/five-position/user-specified) | `tasni/modules/scan/survey_contract.py`, `classifier.py`, `rect_fit.py`, `corner_evidence.py`, `five_position.py`; UI `tasni/webui/src/pages/SurveyPanel.tsx`; see the section above |
| `d*` distance characterization (spec §5) | `tasni/core/characterize.py`, `tools/characterize_distance.py` |
| Fusion / mesh / work plane | `tasni/modules/scan/reconstruct.py`, `tasni/modules/scan/plane.py` |
| Calibration workflow | `tasni/modules/calibration/module.py`, `tasni/modules/calibration/service.py` |
| Shared camera transport | `tasni/core/camera.py`, `tasni/core/livepreview.py` |
| RoboDK API wrapper | `tasni/core/rdk_io.py`, `tasni/core/session.py` |
| Config defaults and knobs | `tasni/core/config.py`, `tasni.config.json` |
| Jetson camera server | `server/server_unicast_syncronous.py`, `server/scan_overlay.py` |
| Jetson deploy / restart | `tools/jetson_deploy.py`, `server/jetson-autopull.sh` |
| Tests | `tests/test_scan_job.py`, `tests/test_scan_planner.py`, `tests/test_calibration_job.py`, `tests/test_collision_guard.py`, `tests/test_survey_contract.py`, `tests/test_scan_classifier.py`, `tests/test_rect_fit.py`, `tests/test_corner_evidence.py`, `tests/test_five_position.py`, `tests/test_characterize.py` |

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

- `docs/deposit-segmentation-handoff-2026-08-30.md`: **read before touching the
  extrusion chroma gate, `deposit_floor_mm`, or `floor_profile`.** The gate's
  bead-vs-board saturation separation has inverted on the cell (bead 25, board 28)
  and its 1 mm-quantisation justification is obsolete at protocol 2's 0.1 mm --
  but the 1.5 mm floor it unlocks is still load-bearing, so it cannot just be
  deleted. Also carries the `floor_profile`-is-None-in-production defect and the
  `assemble_arcs` three-way divergence.
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
- `docs/extrusion-current-handoff.md`: authoritative Cylinder Test implementation,
  live-test result, status -5 placement fix, retained artifacts, and next steps.
- `docs/HANDOFF_EXTRUSION_CYLINDER.md`: original cylinder requirements; its final
  pre-implementation status is historical.
- `docs/extrusion-legacy-trace.md`: legacy pipeline, valve mapping, and deliberate changes.

Many handoff docs contain old commit hashes and status lines. Prefer this map for
current navigation, then use the long docs for reasoning history.
