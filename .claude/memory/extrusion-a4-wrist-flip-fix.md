---
name: extrusion-a4-wrist-flip-fix
description: "Extrusion axis-4 wrist flip - root cause FOUND by spike; native Curve Follow works with an un-mirrored path-to-tool seed, no targets"
metadata: 
  node_type: memory
  type: project
  originSessionId: 2ff6eec9-f6a9-4a86-966c-a6d3bdcda9a4
  modified: 2026-08-27T18:38:15.024Z
---

Root cause of the extrusion A4 wrist flip, measured on the cell 2026-08-27 by
`tools/probe_extrusion_branch.py` (branch `extrusion-inline-program` `4b25531`).
Requirement stands: a program like `myprog`, NO per-point targets; never work on
`main` (worktrees only). See `docs/extrusion-wrist-flip-handoff.md` for the
history, but note the corrections below — that doc's recommended fix is wrong.

**The flip is the path-to-tool SEED (`project.setPose`), not the tool.**
`rdk_io.py:1543` only ever tries two seeds, and `:1547-1554` breaks on the first
that GENERATES — which is the flipped one — despite its own comment claiming it
keeps the one that verifies against neutral.

- RoboDK **mirrors** the seeded roll: `generated RPW = [-r, p, 180 - w]`
  (equivalently `R_out = Rz(180)·S·R_in·S`, `S = diag(1,-1,1)`). Seed from the
  pre-un-mirrored orientation, then `·rotx(pi)·rotz(pi)` → pose error 0.00° at
  every yaw tested, zero targets, path start lands on the parked joints.
- `·rotx(pi)·rotz(pi)` on the RAW commanded orientation is a **coincidence** that
  only works because this cell's yaw is ~90°. At yaw 135° it is 91° wrong.
- Handoff §6's mirrored TCP (`TasniPrintTCP`) is a **no-op** — it produces
  bit-identical joints to `LongCalibTool`. Don't build on it.
- The emitted flipped pose is wrong *as a pose* (178.63° roll about the tool
  axis), not merely a wrong IK branch — so a one-target prefix cannot rescue it
  (measured: still dA4 178.59).
- `Robolink.AddFile` on a `.src` opens a **modal dialog** — never completes
  headless, and the dialog lands on the operator's screen. Writing a `.src` via
  `Item.MakeProgram` works; the post refuses `LIN` to a joint target.

**IMPLEMENTED + pushed 2026-08-27** on branch `extrusion-inline-program`
(`bf103dd`, `40afe49`, `6fc0461`, `175d3e6`, `7506f20`, `1333943`; 84 targeted tests green).
`curve_follow_seed_T` (matrix form, exact at tilt — no threshold/refusal needed), curve
normal = frame +Z, RoboDK's own program kept, then four gates: per-move pose error ≤ 1°,
wrist-flip sample check, valve placement, target count unchanged. Headless e2e on the real
station: 195 instructions, targets +0, pose error 0.000°, dA4/5/6 = 4.97/4.42/1.64°,
`update_program(collisions=True)` = 195/195 OK, no collisions; path start = the parked pose.
RoboDK mirrors its approach on the way out, so ONE zero-length plane move sits between
AirOff and the retract — don't assert "AirOff is immediately followed by the retract".
TRAP: `ExtrusionConfig` is `extra="forbid"`, so any `tasni.config.json` still carrying the
removed `max_path_targets_per_layer` now fails startup (the operator's live file is clean).
**MERGED to `main` @ `534b9b5` and pushed (2026-08-27, --no-ff).** Independently reviewed
(seed map is an involution; gates block the observed failures; no stale refs). Cell
validation (preflight → GUI dry run) still pending; the backend must be RESTARTED first.
Worktree `../RoboDkClaude-extrusion-inline` + branch `extrusion-inline-program` left in
place (not deleted).

**2026-08-27 INSPECTION A4 flip (same class of bug, different mechanism) — root cause
measured read-only on the GUI station:** `inspection.py:pose_from_aim` hard-codes the roll-0
camera X = work-frame +X, but this cell's Realsense X at the PARKED pose = work-frame **−X**
(Z = −Z). So "roll 0" is a 180° roll; for that pose SolveIK_All has NO neutral branch (all 4
have |dA4| or |dA6| ≈ 178) and `create_inspection_target` uses plain seeded SolveIK with no
wrist gate → stores dA4 = −178, collision passes, accepted. Roll-180 solves to
`[89.8, −62.5, 147.8, 0.9, −54.1, −0.2]` (≤ 12° from parked). Fix delegated to the peer on
branch `extrusion-inspection-roll`: robot-referenced roll (`reference_x` = camera X at
`start_joints`) + `solve_joints_on_neutral_branch` gate in `create_inspection_target`
(reject → candidate walk continues) + joints/deltas in the report. Preview endpoint is
station-less → labels `roll_reference`.

**2026-08-27, second instance of the SAME class of bug — the INSPECTION move.**
Fixed on branch `extrusion-inspection-roll` (`ea4ccb7`, `755ae95`; 96 tests green;
MERGED to `main` @ `9ed159c` on 2026-08-27, 100 targeted tests green on the merged tree). **CELL-VALIDATED 2026-08-27: operator reports cylinder "perfect and smooth" and, after the backend restart, inspection "all good now". Both A4-flip issues CLOSED.** `pose_from_aim` hard-coded its roll reference to the work frame's +X,
but the Realsense TCP at the parked joints reads X=[-1,0,0] Y=[0,1,0] Z=[0,0,-1] in
`Tasni Work Frame` — so "roll 0" was 179.7° off, RoboDK had only 4 IK branches for
it and ALL were flipped, and the seeded `SolveIK` stored dA4 = −178.1 which then
PASSED collision validation (a flipped wrist is not a collision). Fix: roll is
referenced to the camera at the job's start joints via the new read-only
`RdkIO.camera_axes_in_frame`, and the target is solved with
`solve_joints_on_neutral_branch` instead of a seeded `SolveIK`. Verified on the
cell: old pose refused by the gate; new one solves to
[89.89, -62.47, 147.78, 1.33, -54.08, -0.25], all deltas ≤ 11.8°, collisions 100% OK.

**Generalisable lesson: any fixed orientation convention in this cell needs a gate
that compares against the ROBOT's neutral pose — a wrong convention still passes
collision validation.** Both bugs were exactly 180° conventions that nothing checked.


Related: [[restart-tasni-backend-after-code-edits]], [[windows-python-and-encoding-traps]].
