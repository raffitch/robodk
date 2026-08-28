# Live print blocker — continuation handoff (written 2026-08-28)

**For the next agent, whichever tool.** Read [AGENTS.md](../AGENTS.md) first (5 min: working
agreement + environment traps). This file is *where we are and what to do next*. The deep
investigation record is
[live-print-dispatch-handoff-2026-08-28.md](live-print-dispatch-handoff-2026-08-28.md) —
read that only when you need the evidence behind a claim here.

---

## TL;DR — measured outcome and next cut

On the cell, `jog` physically moved A6 by 2 deg; immediately afterwards `trivial`
returned 1/1 accepted but never became busy and did not physically move. There was one
RoboDK process, run mode read 6, the kept layer program's run type read 2 (run on robot),
and the driver stayed READY. Right-click Run on the kept layer program still moved the
arm. The fault is therefore **RoboDK v6.0.5's API program-execution path**, not the
driver, controller, program content, or Tasni app state.

The documented program item command `program.setParam("Start", 0)` was then tested on
the same bare one-MoveJ fixture: it returned `"OK"`, RoboDK reported the robot busy, and
the operator confirmed the physical arm moved. The narrow fix is implemented:
real-robot programs use item Start; simulation retains `RunCode()`. A complete app
Characterize job then physically moved to the inspection pose, captured, archived and
returned, validating the dispatch fix without a valve.

That first real frame exposed a second, processing-only issue: the broad ChArUco-board
depth residual was the largest DBSCAN cluster, so the old rule reported a nonsensical
52.77 mm radius and 51.12 mm bead. Ring characterization now scores every cluster for
angular coverage and radial compactness. Offline replay selects the visible ring and
returns radius 39.17 mm, centre (217.94, 150.44) mm, bead footprint 13.26 mm and top Z
6.14 mm. A second 300 mm attempt was correctly rejected but saw no independent ring
cluster although the operator confirmed the ring was present. Depth is already native
1280×720, whose MinZ is ~280 mm, so the 300 mm clamp is the sensor floor, not a tunable
margin — the ring-selector fix is what the next run tests. **Next cell test:** restart,
Characterize at 300 mm, verify the archived mask is annular, then Apply. Failed attempts
now archive raw RGB-D. Live print remains a separate guarded test after measurement and
continues to use 300 mm.

---

## 1. The symptom (unchanged)

Clicking **Print & record** in the Tasni app: RoboDK accepts the layer program, the cell
makes one audible click, the arm does not move. Right-clicking the *same* program in
RoboDK afterwards **does** move it. `AirOn`/`AirOff` also work by hand.

```
layer 1: dispatched — RunCode returned 195 of 195 instructions, run mode 6
layer 1: program ran 0.0 s (NEVER OBSERVED RUNNING); flange camera says the arm did NOT move
```

## 2. State as of this handoff

- Branch **`main`**, working tree clean before this handoff update; push every update.
- Targeted tests green: 126 across `test_dispatch_report`, `test_extrusion_job`,
  `test_extrusion_wait`, `test_extrusion`, `test_rdk_io_run_mode`,
  `test_extrusion_runtime`, `test_extrusion_motion_witness`, `test_extrusion_standoff`,
  `test_valve_outputs`. (Do **not** run the whole suite — it is too slow and has been
  interrupted repeatedly. Run the files you touch.)
- The item-Start fix is implemented and app-level Characterize motion/capture/return is
  cell-validated. A fresh characterization is pending only to validate the new geometric
  ring selector before Apply.
- Cell bisect measured 2026-08-28: direct `MoveJ` physically moved; API-created one-MoveJ
  program was accepted but never busy and did not move. One RoboDK process was running;
  RoboDK `6.0.5.26883`; station run mode 6; program run type 2. The external Python API
  package was upgraded from `robodk 5.6.4` to `6.0.1`; `RunCode()` still failed
  identically, eliminating the version mismatch. `pyproject.toml` now requires 6.0.1+.

## 3. The decision tree — this is the actual work

### Rung 1: `py -3.10 tools/dispatch_bisect.py jog`

**Measured outcome: ARM MOVED physically by 2 deg on A6.** Continue to Rung 2; do not
re-check the controller path.

| Outcome | Meaning | Next |
|---|---|---|
| **Arm moves** | Driver + KUKAVARPROXY + `RoboDKsync570` KRL loop + pendant are all fine. The fault is **RoboDK's program executor**. | Rung 2 |
| **Arm does not move** | The fault is **below RoboDK**, and `ConnectedState() == READY` only ever meant the TCP socket was up. | Go to the pendant: is `RoboDKsync570` *selected and running*, with its pointer cycling in `WHILE COM_ACTION >= 0`? What is `$OV_PRO` (program override)? Any unacknowledged `KSS…` message? Nothing in this repo can fix that; report it to the operator. |

> Why `jog` is the right first cut: the live job's `finally` block already calls
> `rdk.move_j_joints(start_joints)` on every failed run (`service.py:1078`) — a direct
> driver move. Nobody has ever seen it move the arm, but that proves nothing, because the
> arm is already at `start_joints`, so it is a zero-length move. `jog` is the first real
> test of that path.

### Rung 2: `py -3.10 tools/dispatch_bisect.py trivial`

Builds a 2-instruction program, dispatches it exactly as the app does, deletes it after.

**Measured outcome: ARM DID NOT MOVE.** `RunCode()` returned 1 for the one-instruction
program; `program.Busy()` and `robot.Busy()` remained false for 10 seconds; model joints
did not change; the operator directly confirmed no physical motion. This eliminates
app-specific state/timing and all layer-program content.

| Outcome | Meaning | Next |
|---|---|---|
| **Arm moves** | API program dispatch works *from a bare script*. The difference is the **app's** state or timing. | Diff the script's numbers against the app's log for the same program. Suspects, in order: something the app leaves on the `Robolink` connection; the concurrent `/api/rdk` status poll hitting the same connection mid-run; a second RoboDK process (`Get-Process RoboDK` must be exactly 1). |
| **Arm does not move**, but a manual right-click → Run does | **API-dispatched program execution is broken in this RoboDK build/station**, for any program. | Rung 3 |

### Rung 3: what the driver itself reports

The cheapest driver witness has now been read: it stayed READY during both the app layer
dispatch and the bare-program dispatch. Combined with a successful direct-driver jog,
that means RoboDK's RunProg executor never sent the driver a command. The alternate
`setParam("Start", 0)` command returned `"OK"` and physically moved A6; production
real-robot dispatch now uses it. The routes below remain useful only if the app-level
cell validation differs (installed version: **RoboDK v6.0.5**, 2026-06-15):

1. **`/DEBUG=` — documented** (`C:\RoboDK\Notes.txt`, v3.4.2: "It is possible to pass a
   specific file for debugging"). Close RoboDK, relaunch as
   `"C:\RoboDK\bin\RoboDK.exe" /DEBUG=C:\Users\User\Desktop\robodk-debug.txt`, reproduce,
   read the file. This is general RoboDK debug output, not driver-only.
2. **Run the driver in a visible console — most direct.** The KUKA driver is a separate
   process RoboDK spawns: `C:\RoboDK\api\robot\apikuka.exe`. The shipped
   `apikuka-start.bat` only sets Qt paths and launches it, so it runs fine in a console
   window where you can see the traffic. It takes commands on **stdin** — that is how
   RoboDK drives it — so it also lets you command the KUKA with RoboDK out of the loop.

The connection panel (double-click the robot, or **Connect → Connect robot**) shows the
driver status text live; keep it open during a dispatch. *Caveat: the exact GUI labels
could not be verified from the install — the binary's strings are packed and the `.qm`
translations are hashed — so look for the status area, not a specific button name.*

Also re-run the app once and read the **new** line added in `37d4f18`:

```
layer 1: driver READY -> WORKING -> READY          <- RoboDK did dispatch; controller-side fault
layer 1: driver stayed READY throughout — it was never asked to work, so RoboDK
         accepted the program and then dispatched nothing   <- RoboDK-side fault
layer 1: driver READY -> PROBLEMS (…message…)      <- the controller says why
```

## 4. Measured dead — do not re-chase any of these

| Claim | Killed by |
|---|---|
| `RunCode()` refused the program | Returns **195 of 195** instructions |
| Station not in `RUN_ROBOT` | Run mode reads back **6** every dispatch |
| The layer program / Curve Follow project is malformed | The **2-instruction valve program** dispatches identically (`2 of 2`, mode 6) and is equally silent |
| The valve mapping (`$OUT[0]`, `KSS014444`) | Fixed in `deaad43`; IOs renamed to bare numbers; AirOn/AirOff run by hand |
| Wrong driver `.src` version | The two copies in `Downloads` are **byte-identical** (`cmp`) |
| Stale work frame / stale scan / wrong inspection tool / bad scan / 140 mm offset | All measured on the cell — see the investigation doc §8 |
| Camera / depth / hand-eye | Verified: TCP says 471.1 mm, camera measures 467.5 mm |

**The single most important reframe:** the trivial no-motion valve program fails the same
way as the 195-instruction layer program. This is **not** about what the program contains.
Stop looking at the toolpath, the machining project, the seed, the orientation, the valve
mapping.

## 5. Where the code is

| What | Where |
|---|---|
| Dispatch + all its diagnostics | `tasni/core/rdk_io.py:1890` `dispatch_program()` |
| Driver's own state (`READY`/`WORKING`/`PROBLEMS`) | `tasni/core/rdk_io.py:1953` `driver_state()` |
| Thin int wrapper for old callers | `tasni/core/rdk_io.py:1941` `start_program()` |
| Valve programs (the simple-case control) | `tasni/core/rdk_io.py:2005` `run_station_program()` |
| Direct driver move — the fallback primitive | `tasni/core/rdk_io.py:250` `move_j_joints()` |
| Busy-wait, fast polling, `on_poll` hook | `tasni/modules/extrusion/service.py:279` `_wait_program()` |
| Log-line formatters | `service.py:183` `describe_dispatch()`, `service.py:256` `describe_driver_states()` |
| Flange-camera motion witness | `service.py:156` `view_changed()` |
| Live layer dispatch (the loop under investigation) | `service.py:892` |
| Bisect ladder | `tools/dispatch_bisect.py` |
| Tests for all of the above | `tests/test_dispatch_report.py` |

## 6. If dispatch cannot be fixed — the fallback, with honest costs

Only after rungs 1–3. Two options, both real:

**A. Drive the path directly through the driver** (no generated program). `move_j_joints`
/ `move_j_pose` already exist. Removes this entire class of problem.
*Cost:* the driver is blocking per command (`$ADVANCE = 0` in `RoboDKsync570.src`), so
every point is a round trip and you lose RoboDK's blending/rounding — a dense extrusion
path would print jerkily, or slowly, or both. Acceptable for **inspection** (one pose, no
path), which is the natural place to start. Probably not acceptable for the print itself.

**B. Post-process to a native KUKA `.src` and run it on the controller**
(`RUNMODE_MAKE_ROBOTPROG_AND_START`). Preserves blending and runs at full speed with no
PC in the motion loop.
*Cost:* a file has to get onto the controller, and a previous attempt at `.src` import was
blocked by a modal dialog in RoboDK (see the `extrusion-a4-wrist-flip-fix` history). Needs
the operator at the pendant.

Do not start either without telling the operator — B especially changes how the cell is
operated.

## 7. Ground rules for this repo

Full list in [AGENTS.md](../AGENTS.md). The four that bite hardest:

1. **`py -3.10`** — there is no `python` on PATH.
2. **Never run the full pytest suite** — run the files you touched.
3. **Restart the app after backend edits** and check `/api/health` `build.stale` *before*
   asking for a cell test. Two cell runs were wasted on stale code.
4. **Commit and push everything.** The operator reviews from pushed history; the Jetson
   deploys from it. Report the hashes.

Plus, specific to this bug: **measure before theorising**, and keep test fakes physically
coherent — a fake whose driver never leaves `READY` models the exact fault under
investigation and will pass the guard meant to catch it (that happened; see `37d4f18`).

## 8. What to write back

Update, in this order:

1. This file — the decision-tree outcome you got, with the actual log lines.
2. The investigation doc's §3 table — move whatever you measured from *inferred* to
   *measured*, or add a new dead row.
3. `AGENTS.md` §4 — if the "what's open" summary changed.

And say plainly which of the two branches at Rung 1 you landed in. That one fact is worth
more to the next person than any amount of narrative.
