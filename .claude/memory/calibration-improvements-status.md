---
name: calibration-improvements-status
description: "Tasni calibration effort — 4 phases + review follow-up + NVENC H.264 preview + Jetson auto-pull ALL SHIPPED to origin/main (HEAD 27a38f2) and deployed/hardware-validated (2026-06-21)."
metadata: 
  node_type: memory
  type: project
  originSessionId: bb71efaa-98dd-4274-a9a3-a96f17fb4912
---

A 4-phase improvement of the **calibration module** (`tasni/`) is underway, from a review of hand-eye
best-practices + UX/UI. The full sequenced plan (phases, files, reuse map, verification) lives at
`C:\Users\User\.claude\plans\frolicking-wibbling-platypus.md`.

**Status (as of 2026-06-20):** **Phase 1 (calibration accuracy) is COMPLETE** on branch
`calibration-improvements` — **local, NOT pushed** (5 commits, `1e59649`…`b461359`). It added:
multi-method `solve_best` (removes TSAI's ~180° mount singularity), median-of-N frame capture,
seeded-shuffle holdout + k-fold cross-val, a motion-diversity/conditioning metric, intrinsics
verify-and-warn, the pose cone widened 32°→45° (axis-spread 0.10→0.17; roll does NOT help diversity),
and a 3D bird's-eye "Target spread" diagram (board on pedestal, camera cone above). Also fixed a
pre-existing `tasni.cli --apply` crash. All Python suites green; `tsc` + `vite build` clean.

**Phase 2 (safety & operator trust) is COMPLETE** on the same branch (commit `53fdc51`, NOT pushed):
`SimTourJob` simulated dry-tour soft gate (`POST /poses/simulate`, per-pose reachable/collision +
return-to-start, restores run mode in finally; new `rdk_io` run-mode + collision helpers),
confirm-before-motion dialog (replaces `window.confirm`), live per-pose board-lock thumbnail strip,
print-scale hard gate (100 mm-ruler ack gates Run) + measure-a-square calculator. New
`tests/test_sim_tour.py`; all Python suites green, `tsc`+`vite` clean.

**Phase 3 (metrics interpretation & accessibility) is COMPLETE** on the same branch (commit `e8dbb7d`,
NOT pushed): `quality.diagnose()` metrics→verdict (pass/borderline/fail + cause attribution — camera
model vs robot-pose/depth, intrinsics-warn, weak diversity, overfit) rendered as a verdict banner;
colour-blind-safe lamps (✓/✗/· glyph + OK/NO word) and AimHud ✓ IN/✗ OUT readout tags; HUD jog-frame
clarity hint (camera/TOOL frame + live `jog_invert_*` state). New `tests/test_diagnose.py`; all suites
green, `tsc`+`vite` clean.

**Phase 4 (last phase) is COMPLETE** on the same branch (commits `0c5b2b3`…`312c397`, NOT pushed) —
platform plumbing that benefits module #2 (scan):
- **#16** moved the aiming gate `calibration/gate.py` → `core/aiming.py`; lifted `ViewDetection` →
  `core/charuco_types.py` (re-exported from `charuco.py`) so core needs no `modules.*`.
- **#17** `tests/test_camera_wire.py` — round-trips the Jetson server's `<IId>`+lz4+JPEG framing through an
  in-memory fake socket; asserts MODE COLOR⇒depth_len=0 + turbojpeg/cv2-fallback agree.
- **#13+#14** new `core/runs.py` (run_dir/load_report/load_meta/list_runs + path-traversal guard + `root=`
  seam + atomic write_active/read_active). Job writes `meta.json`; `service.apply_calibration(job=|run_id=)`
  unifies apply (in-memory fast path OR load from disk → survives restart) and writes
  `runs/calibration/active.json`. `module.ApplyBody{run_id}`, `GET /api/runs/active?module=`, Home.tsx
  "Cell calibrated: <date> · <verdict> · <px> · tool". New `tests/test_runs.py`.
- **#15** new `core/camera_lease.py` — labelled single-owner mutex; **non-blocking** job acquire (never
  deadlocks the single JobRunner worker), safe release. `ServiceContainer.camera_lease`; LivePreview
  acquires on start / releases after thread join; capture grabs go via `_camera_hold`; `/api/health`
  reports the owner; `/live/start`→409 on CameraBusy. New `tests/test_camera_lease.py`.
- **#18** (not dropped) `core/config.py` dataclasses → Pydantic models (validate-on-load/assignment,
  extra=forbid; JSON-override semantics preserved; `vars()`→`model_dump()`).

All 10 Python suites green; `tsc`+`vite build` clean. **Locked decisions held:** intrinsics = verify-and-warn;
dry-run = soft gate. **Robot-motion + real-camera paths are structure+fake tested only — final cell
validation is the user's.** Handoff `phase4-handoff.md` can be deleted (phase done).

**The ENTIRE 4-phase effort is committed locally on `calibration-improvements` but UNPUSHED to `origin`
(private `raffitch/robodk`).**

**Review follow-up (2026-06-21) — 2 more local commits on the same branch:**
- `3773510` feat(calibration): UX hardening + robust solver pass — run/connect error banner;
  reload-resilience (rehydrate solved run via `/status` on mount, target count after Connect — `/targets`
  only post-Connect because `session.rdk` lazily triggers the 117 MB station load); holdout pre-flight;
  dialog a11y (Esc/aria/focus-Cancel). Conservative outlier rejection in the solve
  (`handeye.reject_outliers`, config `reject_outliers/outlier_px/outlier_factor`, surfaced in report+UI);
  `cross_validate(method="best")` re-selects per fold. New synthetic tests.
- `7b92081` perf(stream): TCP_NODELAY (server + both client sockets) + backward-compatible
  `MODE COLOR Q<n>` JPEG-quality handshake; preview sends Q60 (`calibration.preview_jpeg_quality`),
  captures stay high-q. **Hardware-validated on the Jetson Nano**: color frame 195.5→100.8 KB at Q60
  (−48%), 72.2 KB at Q40; full depth+color path + old-server back-compat confirmed; service stopped,
  tested, and **restored healthy**. NVENC/H.264 hybrid + wired-link recommendation in `docs/jetson-scanner.md`.

**2026-06-21 — FULL SHIP COMPLETED.** The whole branch (4 phases + the 2 review-follow-up commits) was
pushed and **fast-forward-merged to `origin/main` (now at `7b92081`)**, and the Jetson stream change was
**deployed via `tools/jetson_deploy.py deploy` and verified live** (production server: 195.7 KB → 100.9 KB
at Q60). A fresh `python -m tasni --port 8000` app was started so its client sends Q60 (the earlier stray
instance had exited). **The "ask before pushing" guardrail is now moot — the branch IS on origin/main.**

**Git auth note:** GCM/`wincredman` is broken on this workstation, so a dedicated GitHub key
`~/.ssh/github_robodk` (ed25519, passphraseless, comment `robodk-claude-deploy`) was generated and added
to the user's GitHub; the repo's `origin` was switched to SSH (`git@github.com:raffitch/robodk.git`) with
repo-local `core.sshCommand` pointing at that key. Future pushes from here work over SSH without GCM.

**Still real-cell-untested:** the robot-MOTION calibration paths (capture tour / apply) remain
structure+fake tested only — the streaming + camera-read path is now the only part hardware-validated.

**2026-06-21 — NVENC H.264 PREVIEW: BUILT + HARDWARE-VALIDATED (UNCOMMITTED on `calibration-improvements`).**
Transport decision (user-chosen): **H.264 over the EXISTING TCP :1024 connection** (NOT RTSP — rtsp-server
lib isn't installed on the EOL Nano, and the browser goes through the Windows client anyway, so RTSP's
direct-player payoff doesn't apply). New handshake `MODE COLOR H264 [B<kbps>]` (default 4000). Server
(`server/server_unicast_syncronous.py` `stream_h264`) drives `gst-launch-1.0` as a **subprocess** (raw BGR
in on stdin → `rawvideoparse → videoconvert → nvvidconv NV12 → nvv4l2h264enc → h264parse → fdsink` H.264 out
on stdout) — because GStreamer `gi` bindings exist only in system py3.6, not the server's 3.10 venv. No
per-frame header in H264 mode: server relays the raw Annex-B byte-stream; client decodes with **PyAV**
(`tasni/core/camera.py` `_H264Stream`). Wired via `calibration.preview_codec` ("jpeg" default | "h264") +
`preview_h264_bitrate_kbps`, threaded through `LivePreview.start/_loop/stream` → calibration module. `av`
added as optional dep (`pip install -e .[h264]`; already installed on THIS workstation). Captures + grab()
stay JPEG/lossless. **Measured live: first frame ~0.4 s, ~37 fps @ 1280×720** (vs ~0.5 fps old depth path).
Tested by running the new server from /tmp on the Jetson (service stopped → tested → **restored healthy**).

**2026-06-21 — JETSON AUTO-PULL: BUILT + VALIDATED (UNCOMMITTED).** A systemd timer
(`server/jetson-autopull.{sh,service,timer}`, every ~2 min) makes the Jetson follow `origin/main` itself —
**push to main ⇒ Jetson self-deploys**. Fetch → if main moved, hard-reset to it; **defers if a client is on
:1024** (no mid-capture interruption), **restarts `realsense-camera` only when `server/` changed**. Puller
installed to `/usr/local/bin/jetson-autopull.sh` (root for systemctl, git as `jetson`). Install via
`python tools/jetson_deploy.py setup-autopull` (also folded into `bootstrap`; `status` shows the timer).
**Installed + enabled on the Jetson NOW**, and the pull+restart path proven by rewinding the clone one commit
and watching it catch back up + restart the camera. Docs: `docs/jetson-scanner.md` (NVENC results + auto-pull
section), `CLAUDE.md` deploy commands updated.

**2026-06-21 — SHIPPED + DEPLOYED + DETECTION-VALIDATED.** Both features committed (`27a38f2`, +`.gitattributes`
forcing LF on `*.sh`/`*.service`/`*.timer` so Windows CRLF can't break the Jetson scripts), ff-merged to
`origin/main`, and pushed. **The push deployed itself via auto-pull**: the Jetson timer detected
`7b92081 -> 27a38f2`, saw `server/` changed, restarted the camera — confirmed live (this is the real
push→pull→restart flow, not a simulation). Client flipped via `tasni.config.json` (git-ignored,
machine-local) `calibration.preview_codec="h264"` — `av` is installed here. **Head-to-head ChArUco detection
vs the live production server: H.264@4000kbps = 25/25 frames, avg 34.8/35 corners; JPEG Q60 = 25/25, avg
35.0/35** — H.264 loses essentially nothing, so the "H.264 softens corners" worry is a non-issue at this
bitrate; the aiming gate works identically. NVENC preview is now the default on this workstation.

**Still real-cell-untested (unchanged):** the robot-MOTION calibration paths (capture tour / apply). The
camera-read + streaming path (JPEG and now H.264) is hardware-validated; robot motion is not.

**2026-06-21 — POSE-GENERATOR DIVERSITY + GUARDRAILS (UNCOMMITTED on `calibration-improvements`).**
Added `tools/robot_probe.py` — loads `Tasni.rdk` in a headless RoboDK and dumps the live robot's joint
limits + replays the pose generator against real IK. Confirmed: **KUKA KR150 R2700**, A2 range only 135°
and A5 only 250° are the binding axes. **Key non-obvious finding:** the generator's old "keep first N
reachable" clustered the kept set at the *narrow* end of the cone (poses are spiral-ordered centre-outward),
AND at the robot's current seed (TCP `[360, -1507, 394]`, far out in -Y = workspace edge) only **16/45
candidates are reachable** — so reachability, not holdout/pose-count, is the binding constraint on
hand-eye diversity. Fixes: (1) `poses.select_diverse` = farthest-point sampling on the camera +Z axis
(picks a spread instead of first-N; big gain at OPEN seeds, near-neutral at edge seeds where the reachable
set is itself narrow — proven +1° only at the current edge seed); (2) `poses.viewing_angle_span` →
service logs *effective* cone vs configured 45° and **warns when effective < half** so the operator
re-seeds BEFORE a capture run (this is the lever that actually helps at edge seeds); (3) `MIN_TRAIN_VIEWS=6`
floor — solve-time holdout auto-scales DOWN to keep ≥6 training views (was a too-low `holdout+3` floor that
could leave 3 train poses). Also hardened `RdkSession.close()` against the Free-license popup tearing the
socket. New tests in `tests/test_pose_generation.py`; all pose/calibration suites green. NOTE: this
workstation's RoboDK is a **Free (expired) license — "API calls are limited"** popup appears but queries
still return.

**2026-06-22 — CONNECT + COLLISION ROOT-CAUSE FIXES (UNCOMMITTED on `calibration-improvements`).**
User-reported on the real cell: (a) first "Connect" click always failed ("Connection problem"), second
worked; (b) created targets still had the **spindle colliding with A4**, and target creation was suspiciously
*instant*.
- **THE collision bug — a `robomath.Mat` gotcha:** `Mat.Cols()` / `Mat.Rows()` return the column/row **LISTS,
  not integer counts** (`pose_to_T` relies on this: `np.array(pose.Rows())`). So `screen_collisions`' guard
  `ik.Cols()==1 and ik.Rows()>=6` was **always False** → `MoveJ_Test` (the per-pose collision sweep) NEVER
  ran → nothing was filtered → every target stored as an un-checked **cartesian** target (RoboDK reaches it
  in a colliding IK branch). Fixed by discriminating on element COUNT (`np.asarray(ik.list()).size >= 6`, new
  `RdkIO._ik_to_joints`). Same gotcha also silently broke `robot_dof` (always hit its `=6` fallback) and the
  new `target_joints` (always returned None) — both fixed to use `.list()` size. **`is_reachable` was already
  correct** (it used `.list()`), which is why reachability worked but collision filtering didn't.
- **Connect bug:** the earlier "widen `TIMEOUT`" fix set only the attribute; `Item().Valid()` never re-applies
  it to the live socket, so the first heavy query still ran at robolink's 10 s default. Fixed: `RdkSession._prime`
  pushes the timeout onto the live socket via `rdk._setTimeout()` AND disables global collision checking BEFORE
  the first query; `/connect` now **polls `robot().Valid()` until detected** (config `robodk.connect_timeout_s`
  120 s, `disable_collisions_on_connect`), resetting the session on transient errors — "Connected" only once the
  robot is actually present. Handle is `Finish()`-ed if `_ensure_station` raises (no leak).
- **Tool discovery hardened:** `mounted_tool_items` now uses `ItemList(ITEM_TYPE_TOOL)` (finds the spindle
  regardless of parentage — `robot.Childs()` was missing it) + recursive subtree object walk; warns if 0
  tool↔arm pairs enabled. RoboDK's default map EXCLUDES a tool↔own-robot, so `ensure_mounted_tool_collision_pairs`
  force-enables tool↔arm links `0..dof-collision_skip_wrist_links` (default skip 2 → A1..A4 checked, A5/A6
  skipped to avoid the camera-at-wrist false positive; tunable to 1). Dry tour (`SimTourJob`) now **sweeps
  inter-target paths** (MoveJ_Test between consecutive joint configs), not just resting poses. SolveIK anchored
  with `joints_approx=seed` for deterministic branch.
- New `tools/collision_probe.py` (attach-mode diagnostic: dumps discovered tools, per-target collisions +
  colliding items). New `tests/test_collision_guard.py` (incl. regression that MoveJ_Test actually runs);
  extended sim-tour/job tests. All Python suites green (61 passed; 6 pre-existing `test_runs.py` `tmp`-fixture
  errors are unrelated), `tsc` clean. **Still real-cell-unverified** — awaiting the user's probe run / retest.
  A prior adversarial-review workflow flagged the live-socket-timeout + inter-target-sweep + A5 gaps (all
  addressed here).

**2026-06-22 — JOINT-LOCKED TARGETS (camera-TCP, not flange) (UNCOMMITTED on `calibration-improvements`).**
User-reported on the real cell: selecting a created `TasniCalib_*` target moved the **flange** to the
viewpoint instead of the calibrated **Realsense camera** TCP. Root cause: `add_target` stored a target as a
bare **cartesian** target (just a TCP pose) whenever `screen_collisions` handed back no locked joints —
which happens for cone-edge poses where the *seed-anchored* `SolveIK` finds no branch near the seed (they
pass the seedless `is_reachable`), or whenever `collision_filter` is off. A cartesian target drives whichever
tool the RoboDK GUI has active, so with the flange selected the flange visits the viewpoint. Fix: new
`RdkIO.solve_joints_for_pose(T, seed)` (seeded IK → seedless-from-seed fallback, reusing `_ik_to_joints`);
`generate_calibration_targets` reads `seed_joints` and **back-fills joints for any chosen pose left unlocked**
so EVERY target is a joint target locked to the camera TCP — reproduces the camera at the viewpoint regardless
of the GUI's active tool. Also surfaced `camera_tool_offset_mm` + `targets_joint_locked`/`targets_cartesian`
in the `/poses/generate` payload and the UI log (warns if camera TCP < 15 mm ⇒ calibration not actually
applied ⇒ flange would visit the viewpoint anyway — the second possible cause of the same symptom). New
assertions in `test_calibration_job.py` (joint-locked back-fill); all 11 Python suites green, `tsc --noEmit` clean.

**REAL ROOT CAUSE (joint-lock alone did NOT fix it — user still saw flange-at-target):** the whole pose
pipeline trusted RoboDK's **active tool** (`robot.Pose()`, `SolveIK`, the run's flange math all read the
active TCP). In an **attach** session `robot.setPoseTool(Realsense)` does NOT reliably make it the active
TCP, so `Pose()`/`SolveIK` silently used the **FLANGE** — the generated poses, the IK, and therefore the
joint configs were all flange-based (joint-locking just locked the *wrong* config). The original working
macro `AutoScanTargetDefinition.py` avoided this only because the user picked the tool interactively.
**Fix = stop trusting the active tool; drive everything off the Realsense tool's mounting pose EXPLICITLY:**
`RdkIO` now stores `self._tool_pose` (= `tool.PoseTool()`, read off the tool ITEM, set in `use_camera_tool`/
`use_tool_and_frame`); new `RdkIO._solve_ik(T, seed)` passes that mount as SolveIK's explicit `tool=` arg so
joints place the **camera** (not flange) at `T` (used by `is_reachable`, `solve_joints_for_pose`,
`screen_collisions`); new `flange_pose_T()` = `Pose() @ inv(PoseTool())` and `camera_pose_T()` =
`flange_pose_T() @ _tool_pose` derive flange/camera independent of active-tool state. Service: seed =
`rdk.camera_pose_T()` (was `tcp_pose_T()`), run capture flange = `rdk.flange_pose_T()` (was
`tcp_pose_T() @ inv(tool_pose)`). Net: correct whether or not setPoseTool actually activates the tool; if the
symptom persists after this, the Realsense tool's `PoseTool` offset is genuinely ~0 (calibration not applied
to the tool) — the surfaced `camera_tool_offset_mm` reveals that. **Requires backend restart
(`python -m tasni`) + Create-targets re-run to take effect.** All 11 suites green. **Still real-cell-unverified.**
USER CONFIRMED camera positioning fixed after this (2026-06-22).

**2026-06-22 — COLLISION-GUARD ORDER BUG (spindle↔A4 / "target 12") (UNCOMMITTED on `calibration-improvements`).**
After the camera fix, the user reported the generation collision filter still let the spindle hit A4 on
target 12 (deterministic across fresh sessions). Root cause confirmed against the **robodk 5.6.4** API:
`setCollisionActive(COLLISION_ON)` **rebuilds the collision map from RoboDK's defaults — which EXCLUDE a tool
from colliding with its own robot.** `generate_calibration_targets` enabled the tool↔arm pairs via
`ensure_mounted_tool_collision_pairs` **before** `screen_collisions` turned checking ON, so turning it on
**wiped** those pairs → `MoveJ_Test` swept the spindle↔A4 pose clean → it became target 12. (The dry tour was
unaffected — it already enables pairs *after* `set_collision_checking(True)`.) Fix: `screen_collisions(poses,
*, guard_skip=None)` now (re)enables the mounted-tool↔arm pairs **inside, right after** turning checking on
(the dry tour's order); service passes `guard_skip=collision_skip_wrist_links`. Also added a resting-config
**`Collisions()` backstop** after each `MoveJ_Test` (MoveJ_Test leaves the robot AT the destination when the
path is clear; a coarse 8° step can skip a thin endpoint contact). The early `ensure_*` call in the service is
kept only for discovery/logging (the guard dict). `screen_collisions` SolveIK now also passes the camera tool
explicitly (`_solve_ik`) so the collision-checked config = the camera-at-viewpoint config that gets stored.
New regression tests in `test_collision_guard.py` (`..._enables_guard_pairs_after_checking_on` asserts
COLLISION_ON precedes the pair-enables; `..._no_guard_when_skip_none`). All 11 suites green; **requires backend
restart + Create-targets re-run.** Still real-cell-unverified for this specific target-12 case.

**2026-06-24 — TARGET-GEN BEST-PRACTICE TRIO, "option A" (UNCOMMITTED on `calibration-improvements`).**
From a hand-eye best-practice review of how calibration targets relate to the cone/board/size, user chose
**option A** (keep hand-eye poses purely for rotation diversity; push intrinsics edge-coverage to the
*dedicated* capture, NOT into the hand-eye tour). Shipped 3 library/wiring changes + tests + webui, all green:
- **#2 roll-aware selection:** `poses.select_diverse` now does farthest-point sampling on the **full camera
  rotation geodesic** (`_rotation_geodesic`), not the +Z viewing axis only — so roll (the 3rd rotation axis
  the cone tilt can't give) is part of the diversity maximization. Measured: axis_spread 0.398→0.436, min
  inter-pose rotation 14°→25°, viewing span unchanged (45°). Old "roll doesn't help diversity" note (and the
  config comment) corrected.
- **B visibility pre-filter:** new pure-numpy `poses.project_pinhole`/`board_visible_fraction`; `service.
  generate_calibration_targets` projects the board (corners placed in base from the SEED detection) into each
  reachable+collision-free candidate and drops poses where the board would clip the frame — closes the
  "reachable & safe but board-not-in-frame ⇒ wasted real-robot motion" gap. Config `visibility_filter`,
  `min_board_visible_frac` 0.85, `board_visible_margin_frac` 0.04; surfaced in `/poses/generate` payload
  (`visibility_checked`, `poses_offframe_dropped`) + the generate log + UI. Degrades to no-op if it would
  starve <MIN_TRAIN_VIEWS. New `CharucoTarget.all_obj_points` accessor.
- **#3 decoupled corner floors:** new config `min_charuco_corners_solve` (12) used ONLY for per-pose capture
  acceptance (`_capture` `detect_median`); aiming gate + generate-grab stay on `min_charuco_corners` (6). A
  weak 6-corner view detects fine for aiming but its noisy PnP pose dragged the solve.
- Tests: `test_pose_generation.py` +2 (roll-spread / visibility); `test_calibration_job.py` FakeBoard got
  `all_obj_points`. All 122 Python tests green, `vite build` clean.
- **DEFERRED FOLLOW-UP (option A's intrinsics half — NOT started):** build the **"Camera intrinsics (Step 0)"
  React panel**. The backend is ALREADY complete — `intrinsics_calib.IntrinsicCalibSession` (4×3 coverage
  grid, novelty filter, auto-capture, `draw_overlay`) + routes `/intrinsics/live/start|status|solve|apply|
  reset` in `module.py`. CLAUDE.md's "API only; no UI" is literal: the only gap is the frontend so operators
  actually run the dedicated capture (the accurate edge-coverage path) instead of the thin auto-from-hand-eye.
- **Robot-motion paths still real-cell-unverified** (this round only removes/relaxes poses; worst case a pose
  is dropped, never added).
- **COMMITTED + PUSHED to `origin/calibration-improvements` (2026-06-24)** as TWO commits (user chose to split
  the entangled working tree): `f502161` = the calibration trio (split out of the 3 shared files config.py /
  calibration/service.py / Calibration.tsx via per-hunk `git apply --cached`), `e00f43c` = the pre-existing
  in-progress **scan survey/planner** feature (survey.py/planner.py + ScanConfig + Scan.tsx + collision_pairs
  dry-tour reporting + `docs/scan-survey-planner-handoff.md`) that was sitting uncommitted in the same tree.
  Branch HEAD `e00f43c`; 122 tests green. NOTE: the scan-survey/planner memory ([[scan-survey-planner-plan]])
  said "not started" but it WAS implemented + test-green and is now committed in `e00f43c`.
- **STILL TODO — option A's intrinsics half:** the "Camera intrinsics (Step 0)" React panel (backend +
  routes already exist; UI only). Deferred so the user can verify the trio on the real KUKA first.
  → **DONE 2026-06-24 — see the Step-0 intrinsics panel entry at the bottom of this file.**

**2026-06-24 — COLLISION SAFETY (Part 1): baseline-relative screening + modeled platform keep-out.
COMMITTED + PUSHED `725e9dd` on origin/calibration-improvements.** Triggered by the user: target 9 + the
8→9 transit bump the platform the ChArUco sits on. Live-probed the cell (KUKA KR150, 7-tool turret) and
found the REAL cause: (a) the station reports **6 CONSTANT collisions even at the safe aimed pose** (robot
base↔Pedestal, 3× tool↔wrist L6, Positioner↔wallall, Realsense↔spindle) — counting TOTAL collisions made
every pose look colliding → the old soft fallback **shipped all 15 incl. a genuine spindle↔arm
self-collision (target 13)**; (b) **the platform is NOT in the RoboDK tree at all** (the `Pedestal` object
is something else), so the 8→9 neck-dip bump was invisible to every check. Fixes in `rdk_io` +
`calibration/service` + `module` + `config`:
- **Baseline-relative screening**: `collision_pair_keys()` (canonical order-independent pair set);
  `screen_collisions()` records the seed pair-set and drops a pose only if its SWEPT path
  (`collision_path_samples`, default 6) adds a pair NOT in baseline; legacy total-count path under
  `baseline_relative=False`. `ensure_obstacle_collision_pairs()` enables tool↔object pairs.
  `new_collisions_here()`/`path_new_collisions()` shared by generation + dry tour (so the dry tour catches
  the 8→9 transit). **Live-validated: drops 13, ignores the 6 artifacts.**
- **Modeled platform keep-out**: `add_keepout_box()` builds a box at board footprint + margin (config
  `board_keepout`, `board_keepout_margin_mm`=**300** — set to the platform's real overhang, `_above_mm`=10,
  `_depth_mm`=600) as `TasniBoardKeepout`, created in generate before screening, removed by Clear targets.
  **Live-validated: margin 300 catches the 8→9 transit dip (150 misses); flange dips to Z=126 mid-transit,
  board top ≈ −187, long turret tools hang into the platform.**
- **No more silent bypass**: colliding poses always dropped; too-few-clean → REFUSE with guidance (user
  chose refuse-with-guidance). `collision_filter_hard_fail` now only governs the can't-evaluate case.
- 124 Python tests green (reworked collision_guard/sim_tour/calibration_job fakes for the new semantics +
  a keep-out-box geometry test).
- **NUANCE / next:** with the box, the **dry tour flags** the 8→9 transit, but GENERATION may still create
  target 9 (its resting pose is baseline-clean; generation screens seed→pose, not the consecutive 8→9).
  To stop low poses being created at all → **post-selection consecutive-tour screening** in generate (drop
  a pose whose tour transit enters the keep-out) — NOT yet done. Also: the board low at Z≈−187 (below the
  robot base); board centre est. [−18,−1427,−187] from look-at rays.
- **STILL TODO (Part 2, user-prioritized) — HANDOFF WRITTEN, start in a new session:** calibration HUD
  **tilt-direction guidance** (which way to rotate TOOL B/C to level the board). Full spec in
  **`docs/calibration-aiming-guidance-handoff.md`** (committed `f1df7b6`). Crux: emit `tilt_b_deg`/
  `tilt_c_deg` from `evaluate_gate` (core/aiming.py) using the board normal `det.R_target2cam[:,2]` +
  the scan gate's `atan2(nx,-nz)`/`atan2(ny,-nz)` decomposition (depth_gate.py:116-127); `AimHud` TiltFix
  already renders them. The handoff also captures the optional target-9 tour-drop, the keepout margin
  note, and the intrinsics Step-0 UI leftover.
- **Target-9 decision (2026-06-24):** user is OK leaving it — with `725e9dd` the platform is modeled +
  the dry tour FLAGS the 8→9 transit; generation may still CREATE a low pose (screens seed→pose, not the
  consecutive transit), so "never created" needs the tour-drop in the handoff §6. User closed the app;
  not pursued now.

**2026-06-24 — STEP-0 CAMERA-INTRINSICS UI PANEL (closes option-A's intrinsics half). DONE, COMMITTED +
PUSHED `ddcbe6c` on origin/`calibration-improvements`.** Triggered by a real run (`runs/calibration/20260624-150722`) whose verdict was
**borderline — driven SOLELY by the intrinsics self-check** (recovered k2=0.363 vs configured 0.137, cx/cy
~3px off; everything else PASS: holdout 0.789px < train 0.908px, board consistency 0.733mm, 3 solvers agree
within 0.02px). Diagnose() ignores cross_val (the 1.9px CV "warn" is just the UI colour band, NOT part of the
verdict). The fix for the warning is a dedicated edge-covered intrinsic capture — the backend
(`intrinsics_calib.IntrinsicCalibSession` 4×3 grid + `/intrinsics/live/start|stop|status|reset|solve|apply`)
already existed; only the React panel was missing. Built `tasni/webui/src/pages/IntrinsicsPanel.tsx`
(self-contained: own WS subscription, reacts only to `frame` while its capture is live + `gate` events tagged
`mode:"intrinsics"`; coverage-grid widget mirrors `status.cells`; Start/Stop/Reset/Solve(fix_k3)/Apply;
result shows fit RMS + fx/fy/cx/cy with Δ-vs-config + dist coeffs; thin-coverage warning <60%). Wired into
`Calibration.tsx` as a collapsible "Step 0" card above the aiming card; the main aiming subscription now
ignores intrinsics frames/gate (`intrinsicsLiveRef`, run-aware so board-lock thumbnails still flow), aiming
reclaims the stream in `beginLive`, `onApplied→loadConfig`. CSS in `index.css` (`.intr-*`). Camera-only —
works before Connect (no RoboDK/robot motion); `disabled` only while a calibration job runs. NO backend
change (handoff was right: UI-only gap). `tsc --noEmit` + `vite build` clean; Python suite unchanged (133).
Not yet exercised on the cell. NOTE: the aiming `/live/start` returns "already running" if intrinsics preview
is up, so the operator must Stop/Solve intrinsics before aiming (the panel auto-stops aiming when capture
starts, but not vice-versa — acceptable first cut).

**2026-06-24 — AUTO-CONNECT THE PHYSICAL ROBOT (no more manual "Connect robot"). DONE, COMMITTED + PUSHED
`26c3091` on origin/`calibration-improvements`.** User pain: every calibration run said "robot offline" so they had to link the
KUKA controller by hand in RoboDK's "Connect robot" panel; the RoboDK model also didn't track the real arm.
ROOT: nothing in the app ever called RoboDK's real-robot driver link — `run_mode="run_robot"` only tells
RoboDK to *target* hardware; it still needs `robot.Connect()` to the controller or it reports "offline" and
won't move. (The macros only ever socket-connect to the CAMERA server, never the robot.) Fix:
- `core/rdk_io.py`: `connect_robot(ip="", timeout_s, poll_s)` = `robot.Connect(ip, blocking=False)` then poll
  `ConnectedState()` until `ROBOTCOM_READY` (==0) or timeout; idempotent (early-returns if already ready);
  NEVER raises. Plus `robot_connected()->(ready,msg)`, `robot_connection_params()->{ip,port}` (reads the IP
  stored on the robot item — blank `robot_ip` config uses it). Module-level `link_real_robot(rdk,cfg)` =
  best-effort connect + summary dict for the connect response.
- `/connect` (calibration + scan module.py): after the robot is detected, calls `link_real_robot`, returns
  `robot_link:{connected,message,ip,configured}`. Best-effort — readiness is NOT gated on it (controller may
  be off); the operator can still aim. Frontend `robotLinkNote()` (exported from Calibration.tsx, reused by
  Scan.tsx) appends "Real robot ONLINE/OFFLINE/not linked" to the connect status line.
- Run jobs (CalibrationJob + ScanCaptureJob): `ensure_real_robot_link(rdk, cfg.robodk)` right after
  `apply_run_mode("run_robot")` — auto-connects, and RAISES a clear "real robot offline — check controller/
  driver/network" error if it can't (instead of RoboDK silently refusing). No-op when run mode is simulate or
  `connect_robot_on_connect=False`.
- Config (RoboDKConfig): `connect_robot_on_connect=True`, `robot_ip=""` (blank=use robot's stored panel IP),
  `robot_connect_timeout_s=10.0`.
- New `tests/test_robot_link.py` (8 tests, fake robot scripting ConnectedState). **133 Python tests green**
  (was 125), `tsc --noEmit` + `vite build` clean.
- **CAVEAT / real-cell-unverified:** the *position sync* half ("fetch current position and update") relies on
  RoboDK's driver MONITORING to mirror the controller into the model once linked — that's a RoboDK driver
  setting, not an API call (no separate "get real joints" exists; `robot.Joints()` returns the model, which
  the driver updates). If the model still doesn't track after this, check RoboDK Connection→Synchronize/monitor
  on the robot. The auto-link half (no manual connect before runs) is the solid part. Robot-motion still
  real-cell-untested.

**2026-06-24 — AIMING GUIDANCE (Part 2): board tilt-DIRECTION on the calibration HUD. DONE, COMMITTED +
PUSHED `790bfde` on origin/`calibration-improvements`.** Implemented the
`docs/calibration-aiming-guidance-handoff.md` §1–4 spec. The
calibration gate (`core/aiming.py`) now emits `tilt_b_deg`/`tilt_c_deg` so the HUD's existing ROTATE-TOOL
panel (`AimHud` `TiltFix`, previously scan-only) shows WHICH way to rotate to level the ChArUco board, not
just the tilt magnitude. New helper `board_tilt_bc_deg(R_target2cam)` mirrors the scan gate's normal→B/C
decomposition EXACTLY (orient board normal `R[:,2]` toward camera, `B=atan2(nx,-nz)`, `C=atan2(ny,-nz)`) so
signs are consistent with scan by construction; `evaluate_gate` carries the two fields, `GateReading.to_dict()`
emits them. Both emit paths (`module.py` live preview, `service.py` authoritative grab) ship `to_dict()`, so
the fields ride along with no other backend change; frontend needed NO change (TiltFix already renders when
either field is non-null). Skipped the §3b copy tweak — TiltFix has no "surface" wording, reads fine for the
board. Test `test_tilt_direction_bc` added (+`tilt_axis` param on `_det`): +20° about cam X ⇒ C=+20/B≈0;
+20° about cam Y ⇒ B=−20/C≈0. **125 pytest green** (was 124), `vite build` green. Handoff doc updated to
mark §1–4 DONE. **STILL TODO:** real-board SIGN verification on the cell (handoff §3b/§7 — code matches the
already-hardware-validated scan gate, but the board-normal sign should be eyeballed); plus the secondary
Part-2 ideas (§5) and collision follow-ups (§6: target-9 tour-drop, intrinsics Step-0 UI).
