# Extrusion cylinder: the axis-4 wrist flip — handoff

## 2026-08-27 — RESOLVED

Fixed on branch `extrusion-inline-program`. The layer is now ONE native Curve Follow
program with **zero station targets**, on the parked wrist branch. Everything below
this section is kept as the historical record, but **four of its conclusions are
wrong** — read this first.

**Root cause: the path-to-tool seed.** A Curve Follow Project does not reproduce the
roll of its `project.setPose` seed — it MIRRORS it. Measured: a project seeded with
`X · rotx(π) · rotz(π)` generates the rotation `Rz(180°) · S · X · S`, `S = diag(1,-1,1)`.
That map is an involution, so `curve_follow_seed_T` feeds it the orientation we want
and gets back the seed that produces it. Verified at eight commanded orientations
(four yaws plus tilted ones): **0.000° pose error** at every one.

**Corrections to the sections below:**

1. **§5.1 is wrong** that the seed was tried and failed. Only TWO seeds were ever
   tried — `orientation` and `orientation · rotx(π)` — and the loop kept whichever
   GENERATED first, not whichever verified. The winner was the flipped one. The naive
   `· rotx(π)` seed scores 1.4° at this cell's ~90.7° commanded yaw purely because the
   mirror is near-identity there; at 135.7° yaw the same seed is **91.4° wrong**.
2. **§6's recommended fix (a mirrored `TasniPrintTCP`) is a no-op.** `LongCalibTool`
   and a tool rotated 180° about local X produce *bit-identical joints* for the
   corresponding seed. The tool item never influenced the flip; only the seed did.
   `LongCalibTool` was never modified.
3. **§3 is only half right.** The generated tool AXIS is identical (`toolZ = [0,0,1]`
   in both), but the emitted POSE is rotated 178.63° about that axis — so the pose
   itself is wrong, not just the IK branch, and RoboDK realises that roll by flipping
   axis 4 rather than axis 6. **§5.2's conclusion does not follow**: a one-target
   neutral prefix cannot rescue a wrong pose (measured: still dA4 = 178.59°).
4. **The `.src` import route is blocked, not merely untried.** `Robolink.AddFile` on a
   `.src` opens a MODAL import dialog, so headless it never returns. Writing a `.src`
   via `Item.MakeProgram` works (the post refuses `LIN` to a joint target).

**What the fix does now:** curve normal = the SURFACE normal (frame +Z), not the
commanded tool Z; one computed seed instead of a search; RoboDK's generated program
kept exactly as emitted; then four gates — per-instruction pose error ≤ 1°, the
interpolated wrist-flip sample check, valve-call placement (the extruder may not open
at the approach standoff nor stay open while the nozzle lifts), and an unchanged
station target count.

All of it is reproducible with **`tools/probe_extrusion_branch.py`** (probes A, R, V,
B, C) against a private headless licensed instance.

**The INSPECTION move had the same class of bug** (fixed on
`extrusion-inspection-roll`). Not the seed this time, but the same shape: an
arbitrary orientation convention that happens to be exactly 180° off on this cell.
`pose_from_aim` hard-coded its roll reference to the work frame's +X, while the
Realsense TCP at the parked joints reads X=[-1,0,0] Y=[0,1,0] Z=[0,0,-1] in
`Tasni Work Frame` — so "roll 0" was 179.7° from the robot's own camera
orientation. RoboDK returned only four IK branches for it and every one was flipped
(|dA4| or |dA6| ≈ 178°); the seeded `SolveIK` stored
`[91.9, -50.4, 117.1, -177.9, -82.2, 1.9]` (dA4 = −178.1) and it passed collision
validation, because a flipped wrist is not a collision. The roll-180 candidate is
0.8° from the parked camera orientation and solves to
`[89.8, -62.5, 147.8, 0.9, -54.1, -0.2]`. Fix: roll is referenced to the camera at
the job's start joints, and the target is solved with
`solve_joints_on_neutral_branch` instead of a seeded `SolveIK`. Lesson worth
generalising: **any fixed orientation convention in this cell deserves a gate that
compares against the robot's own neutral pose**, because a wrong one still
collision-validates.

---


Written 2026-08-27 after a long unsuccessful session. Everything here was measured on
the live cell, not inferred. **Read this before touching `create_extrusion_layer_program`.**

Current branch/HEAD when written: `main` @ `fe0ebb3`.

---

## 1. The symptom

Simulating the cylinder, the generated path turns **axis 4 by ~180°** away from the
operator's parked pose. The operator sees *"the flange facing the robot instead of facing
away from it as in its neutral state"*.

## 2. What the operator wants (stated explicitly)

- **No per-point station targets.** Said twice, the second time bluntly. The module's
  documented design is *"one native XYZ+IJK curve per layer, not hundreds of targets"*
  (`docs/extrusion-current-handoff.md`). Targets violate that and clutter the tree.
- **`LongCalibTool` is the extruder and its settings must not be changed.**
- A program that simulates, like their known-good `myprog` does.

## 3. Measured cell facts (do not re-derive these)

Robot: KUKA KR150 R2700. Parked joints (what the code calls "neutral"):

```
[89.22, -74.25, 147.96, 0.21, -42.52, 0.63]
```

Declared travel: **A4 and A6 are ±350°** — which is precisely what puts the flipped
branch within reach. The `NEUTRAL` target is essentially the parked pose (all deltas < 3.3°).

**Tool TCP Z direction at the identical parked pose, measured in `Tasni Work Frame`:**

| tool | toolZ | roll | TCP offset from flange |
|---|---|---|---|
| **LongCalibTool** | `[0, 0, +1]` **UP, away from the table** | −0.01° | `[388.7, 8.0, 440.7]` |
| penholder | `[0, 0, −1]` DOWN, into the table | 179.99° | `[117.1, 4.7, 367.0]` |
| spindle | `[−0.01, 0.856, −0.517]` DOWN | −177.99° | `[0, 0, 200]` |
| Realsense | `[0.012, 0.001, −1]` DOWN | 179.95° | `[116.2, −31.9, 190.8]` |

`penholder`'s `toolZ_in_flange` is `[0.86, 0.02, 0.52]`; `LongCalibTool`'s is
`[-0.86, -0.02, -0.52]` — **exact negatives**. LongCalibTool is the only tool in the
station whose Z points away from the work.

**A neutral-branch IK solution EXISTS for every pose the job needs.** Probed with
`SolveIK_All` at 8 angles plus the +40 approach and +60 retract standoffs — 9–11 branches
per pose, exactly one passing a ±90° wrist filter:

```
path (all angles), APPROACH +40, RETRACT +60  ->  accepted=1
  e.g. [93.8, -67.8, 151.8, 5.1, -52.9, 0.1]   dA4=+4.9 dA5=-10.3 dA6=-0.6
  the flip is the sibling branch of the SAME pose:
       [93.8, -67.8, 151.8, -174.9, 52.9, -179.9]
```

So **the geometry, the reachability and the commanded orientation are all fine.** This has
never been a reachability problem.

## 4. Root cause

A RoboDK **Curve Follow Project aligns the tool Z *into* the work** (approach along
+normal). `create_extrusion_layer_program` sets the curve normal to the commanded
orientation's `+toolZ`. Because `LongCalibTool`'s Z points *away* from the work, satisfying
RoboDK's convention requires physically rotating the wrist ~180° → **A4 = −179°**.

This single fact explains every observed symptom: `setMachiningParameters` returning `-5`
on the identity path-to-tool seed, only the local-X-flipped seed generating, the wrist
clamp making the path ungeneratable, and pose-pinning being unable to rescue it.

## 5. Two hard API constraints (both proven, not assumed)

### 5.1 A Curve Follow Project cannot be steered onto a wrist branch

Tried and failed, all on the live cell:

| lever | result |
|---|---|
| `project.setJoints(start_joints)` (preferred start joints) | still flips |
| deterministic params (`TurntableActive/FollowAngleOn/FollowRealignOn/RotZ_Range = 0`) | still flips |
| `project.setPose(...)` path-to-tool seed, identity **and** local-X-flipped | seed 1 produces **no program at all**; seed 2 generates at A4 = −179 |
| `robot.setJointLimits(...)` clamping A4/A5/A6 to neutral ±90 | **stops generating** ("no linked generated program") rather than switching to the branch that demonstrably exists |

### 5.2 `setInstruction` CANNOT pin joints

Proven with a throwaway plain program (`AddProgram` + `AddTarget`, then `setInstruction`
with `is_joint_target=True`, then read back):

```
[0] jointTarget=1 joints=[89.2, -74.3, 148.0, 0.2, -42.5, 0.6]
                    want=[89.2, -74.3, 148.0, 11.0, -42.5, 22.0]  -> LOST
VERDICT: setInstruction on a PLAIN program DOES NOT retain joint vectors
```

The written vector is discarded; the instruction keeps reading back **its target item's**
joints. True for plain and machining-linked programs alike. `Item.MoveL`'s own docstring
confirms the other half: *"Important note when adding new movement instructions to
programs: only target items supported, not poses."*

**Consequence:** with the current architecture, one target per waypoint is the only way to
pin a wrist configuration — which is exactly what the operator does not want. That is the
impasse this handoff exists to break.

> `myprog` holds 306 inline cartesian moves with only ONE target in the whole station,
> so RoboDK *can* store inline poses — but it authored those internally. The Python API
> cannot reproduce it.

## 6. RECOMMENDED FIX — a mirrored TCP item (never tried)

**This is the most promising untried route and it respects "don't touch LongCalibTool".**

Create a *separate, additional* tool item — say `TasniPrintTCP` — at the **same TCP origin**
as `LongCalibTool` but with its frame rotated 180° about local X, so its Z points **into**
the work. `LongCalibTool` itself is not modified.

Then hand the Curve Follow Project that tool. RoboDK's convention is now satisfied by the
operator's parked orientation, so the native solve should produce the un-flipped path with:

- **zero station targets**
- the native one-curve program the design calls for
- no joint clamping, no seed guessing, no pose pinning

Why it should work: at the parked flange pose, `LongCalibTool` Z reads `[0,0,+1]`.
Rotating the TCP frame by Rx(180°) negates Y and Z, so `TasniPrintTCP` Z reads `[0,0,-1]`
at that *same flange pose* — exactly "tool Z into the work". The physical motion is
identical; only the frame used to describe it changes.

**To verify before building on it:** create the item, run one layer, and read back the
generated program's joints (see §9). Confirm `dA4` stays near +5° and that the approach
standoff lands **above** the build plane.

Caveats to check: the generated program must run with a TCP whose origin matches the real
nozzle (it does — only the rotation differs), and the operator must be shown that a second
tool item appears in the tree.

### Alternatives, in order of preference

1. **Correct `LongCalibTool`'s frame directly** — cleanest of all, but the operator has
   explicitly forbidden it. Only revisit with their consent.
2. **Flip the curve normal** (`normal = -orientation[:3,2]`) so RoboDK's anti-parallel
   convention yields the wanted orientation. Risk: the approach/retract params
   (`"Approach": "NTS <mm> 0 0"`) are normal-relative, so the standoffs would likely need
   their signs flipped too. Verify where the approach actually lands before trusting it.
3. **Accept RoboDK's branch** — no clutter, simplest code, but the flange sits 180° from
   parked. Only viable if that orientation turns out to be harmless for the extruder.
4. Per-waypoint targets — works, rejected by the operator. This is what `fe0ebb3` does.

## 7. Current state of the code

`main` @ `fe0ebb3`. `create_extrusion_layer_program` builds the curve and project (for
visibility), **discards the project's own solve**, and emits the program from
branch-locked waypoint targets:

- waypoints thinned to `extrusion.max_path_targets_per_layer` (default **60**) → 62 targets
  a layer instead of 183; chord error 0.053 mm at r=37.5 mm against a 6 mm bead
- `check_cancel` runs between waypoints (the old loop held ~1800 RPCs, so cancelling looked
  dead and targets kept appearing after the request)
- waypoint targets are deleted **even on failure**; only curve/project/program are kept

**If targets are unacceptable in the interim, `git revert fe0ebb3`** — but note nothing
else currently pins the branch, so the flip returns.

Relevant commits, newest first:

| commit | what |
|---|---|
| `fe0ebb3` | branch-locked waypoint targets, thinned + cancellable (current) |
| `554a379` | build-staleness reporting (§8) |
| `d5ac40b` | pin generated position, not RoboDK's rotation — superseded |
| `e4d7ae8` | `setInstruction` pinning as joint moves — **does not work**, see §5.2 |
| `0d2fc29` | path-to-tool seed selection — superseded |
| `db949cc` | native path + wrist clamp — clamp makes generation fail |
| `fdb3441` | preflight side effects (still good, unrelated) |
| `c4246ad` | the original Codex neutral-branch work |

Still-good, unrelated fixes worth keeping: `fdb3441` (the reachability preflight no longer
moves the robot via `setJoints`, and restores the operator's active tool/frame) and
`554a379` (build staleness).

## 8. Operating notes that cost this session real time

- **The backend caches imported modules.** Editing `tasni/**.py` does nothing until the app
  restarts, and a cell test against stale code is indistinguishable from a failed fix. This
  invalidated **two** test cycles. `554a379` added detection: `/api/health` → `build.stale`,
  a `STALE CODE:` line in the job log, and a `build` block in run reports. The old
  `git_commit` field was renamed `git_commit_checked_out` because it reported
  `git rev-parse HEAD` at *report* time, not what was loaded.
- **Never restart while a fix is still being pushed** — that wasted a third cycle.
- `python` is not on PATH here; use **`py -3.10`**.
- **Never round-trip a source file through PowerShell `Get-Content`/`Set-Content`.** It
  decodes BOM-less UTF-8 as ANSI and adds a BOM; it mangled every em-dash in `rdk_io.py`.
  Repair with `iconv -f UTF-8 -t WINDOWS-1252`.
- Don't run the full pytest suite (too slow). Use the targeted set in §9.

## 9. How to investigate (read-only recipes)

All of these attach to the running RoboDK GUI and only query — they never move the robot.

```python
import robolink as rl
rdk = rl.Robolink(); robot = rdk.Item("", rl.ITEM_TYPE_ROBOT)

robot.Joints()                       # parked pose
robot.JointLimits()                  # (lower, upper, type) -- A4/A6 are +/-350
robot.SolveIK_All(pose, toolT)       # every branch for a pose; N x M, columns = solutions
robot.JointsConfig(joints)           # front/elbow/wrist flags, first 3 values
prog.Instruction(i)                  # (name, type, movetype, is_joint_target, pose, joints)
prog.InstructionListJoints(mm_step=20, deg_step=5, collision_check=0)
prog.getLink(rl.ITEM_TYPE_MACHINING) # invalid == a plain program
```

Reading the **generated program's actual joints** is the single most useful check — that is
how the `jointTarget=0` fingerprint and the −113°/−179° deviations were found. A failed run
keeps `<name>_Curve` and `<name>_Settings`; the program survives only if the failure was
after generation.

Test command:

```
py -3.10 -m pytest tests/test_rdk_io_extrusion.py tests/test_rdk_io_side_effects.py \
  tests/test_extrusion.py tests/test_extrusion_job.py tests/test_build_info.py -q
```

## 10. Open questions for the next session

1. Does the mirrored-TCP approach (§6) actually produce an un-flipped native path? **Test
   this first** — it is the only route that satisfies every stated constraint at once.
2. If it works, where does the approach standoff land — above or below the build plane?
   RoboDK computes it from the curve normal, and this was never confirmed on a good run.
3. Is the ±90° "neutral window" the right acceptance test at all? `myprog` runs at
   **A5 +117° / A6 −109°** from the parked pose, so that window would reject the operator's
   own working program. It is only meaningful when the print pose really is near parked.
4. The wrist verifier bounds A4/A5/A6 only; A1–A3 are deliberately free.

## 11. How the requirement drifted (so it does not repeat)

The operator rejected per-point targets. When the API constraint in §5.2 was proven, they
were offered a choice and selected "accept targets, but make them tidy" — and then, on
seeing 62 targets per layer appear in practice, rejected targets outright again.

**Read that as the real requirement: no per-point targets, at all.** Do not re-litigate it
with a tidier variant; solve §6 instead.
