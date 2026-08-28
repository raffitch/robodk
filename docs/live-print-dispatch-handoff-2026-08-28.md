# Live print: the arm does not move — handoff (2026-08-28, revised)

**Status: UNRESOLVED**, but narrowed by measurement. Clicking **Print & record** makes the
controller click once and the arm never moves. Running the same program by hand in RoboDK
moves it. As of the instrumented run in §0b, *every RoboDK-side explanation is dead*: the
program is fully accepted (195/195 instructions), the run mode is really 6, and the
trivial valve program behaves the same way. Next: `tools/dispatch_bisect.py jog`.

**Revision note.** The first version of this handoff (commit `c02838a`) was written at the
end of the day that produced 16 fixes; re-reading it against the code and against the
latest failed run showed that several of its "ruled out" rows were *inferences*, that
its "next step 1" is already answered by the newest log, and that the cheapest evidence
(the pendant message window, RoboDK's driver log, three numbers the job discards) has
never been looked at. This version separates **measured** from **inferred**, decodes the
latest log against the code, and orders the next steps by cost × decisiveness. Nothing
in the code changed for this revision.

**Continuing this work? Start with [live-print-next-session.md](live-print-next-session.md)**
— what to run next and what each outcome means. This file is the evidence behind it.

Read with [docs/agent-debug-map.md](agent-debug-map.md). The extrusion module's
background is [docs/extrusion-current-handoff.md](extrusion-current-handoff.md).

---

## 0. Start here: what the latest run actually says

Latest run (app restarted after `5fcb46a` — every line below exists only in that code):

```
inspection camera ready: depth frame received before robot motion
real robot linked (Ready)
valve OFF: job startup before motion
valve OFF: before layer 1 approach
layer 1: validating program in RoboDK — this is the slow step on a large station
layer 1: validated in 0.0 s (RoboDK predicts 1.9 s of robot motion)
layer 1: program ran 0.0 s; flange camera says the arm did NOT move
valve OFF: job exit/cancellation/fault
ERROR: RuntimeError: layer 1 the flange camera saw an unchanged view before and after:
the arm did not move, though RoboDK predicted 1.9 s of motion. ...
```

Decoded against `tasni/modules/extrusion/service.py` and `tasni/core/rdk_io.py`:

| Line | What it really means |
|---|---|
| `program ran 0.0 s` | **Execution** time, not wall time (`_wait_program` returns `0.0` when `running_since is None`). So during the whole 5 s start grace (`program_start_grace_s`), `program.Busy()` **and** `robot.Busy()` never returned True for more than one 50 ms poll. This is the answer to the old "next step 1": **neither the program nor the robot was ever seen running.** Caveat: the log does not print `saw_busy`, so "never busy" and "busy for exactly one poll" print identically — log it (§5 C). |
| the same line | A program RoboDK really handed to the driver would be busy ≥ 1.9 s (the arm moves). A program RoboDK merely *simulated* at 1× would also be busy ≈ 1.9 s (the model animates; the live job does not touch simulation speed). **Neither happened.** What is left: RoboDK's pre-run check rejected the program (`RunCode()` return value — the job discards anything ≥ 0), or the run aborted within ~50 ms (a driver error — visible in RoboDK's connection log and/or on the pendant), or the run mode is not what we assume (`RunMode()` has never been read back). All three are readable; none has been read. |
| `valve OFF: …` (×3) | **Not evidence the controller did anything.** `run_station_program` logs after RoboDK *accepts* the program (`RunCode() >= 0`), and its `while program_busy()` has the same start race. Nobody has confirmed the physical valve responds to an API-dispatched AirOff. `hardware_io_test_approved` is an operator attestation in config — the app dispatches no I/O test. |
| `real robot linked (Ready)` | `ConnectedState() == ROBOTCOM_READY`: the driver process has its KUKAVARPROXY socket. It does **not** prove `RoboDKsync570` is selected and running on the pendant (§2). |
| `validated in 0.0 s` | Normal since live collision validation defaults off (`46a7770`); the "slow step" wording is stale. Not a lead. |

---

## 0b. Instrumented run — both software hypotheses ELIMINATED

Second cell run, with the dispatch report live:

```
valve OFF: job startup before motion — RunCode returned 2 of 2 instructions, run mode 6
valve OFF: before layer 1 approach — RunCode returned 2 of 2 instructions, run mode 6
layer 1: validated in 0.0 s (RoboDK predicts 1.9 s of robot motion)
layer 1: dispatched — RunCode returned 195 of 195 instructions, run mode 6
layer 1: program ran 0.0 s (NEVER OBSERVED RUNNING); flange camera says the arm did NOT move
```

**Dead, by measurement:**

| Hypothesis | Killed by |
|---|---|
| `RunCode()` returned 0 — RoboDK's pre-run check refused the program | It returned **195 of 195**. Every instruction cleared. |
| The station was not really in `RUN_ROBOT` | Read back as **6** on every dispatch. `92f2d1d` works. |
| The layer program is special / the machining program is malformed | The 2-instruction valve program dispatches **identically** (2 of 2, mode 6) and is equally silent. |
| The camera hold, validation, or collision screening corrupts state | The very first dispatch of the run — the startup AirOff, before any program is built — already looks like this. |

**What this leaves.** RoboDK fully accepted the program, in the right mode, and then
either sent the driver nothing, or sent it something the driver dropped inside 50 ms.
The log could not tell those apart, because *both* busy probes are RoboDK-side.
Two instruments were added for exactly this (`37d4f18`):

- **`driver_state()`** — `ConnectedState()` is the **driver's** view: `ROBOTCOM_WORKING`
  while executing, `ROBOTCOM_PROBLEMS` with the controller's message. Sampled across the
  whole wait. If it never leaves `READY`, **RoboDK dispatched nothing to the driver** and
  the fault is in RoboDK's program executor, not the cell. If it goes `WORKING` or
  `PROBLEMS`, the command reached the driver and the message says what happened.
- **~250 Hz polling for the first 0.5 s** — a flat 50 ms poll cannot distinguish "never
  started" from "started and died in 20 ms", and that reading is what the diagnosis rests on.

**Note the asymmetry that now dominates:** the *valve* programs are dispatched exactly
like the layer program and are equally never-observed-running — yet `AirOn`/`AirOff` are
confirmed to work when run by hand, just like the layer program. So this is **not** about
what the program contains. It is about API-dispatched execution as such, in this station.
Stop looking at the toolpath, the machining project, and the valve mapping.

---

## 1. The one open problem

Per layer, RoboDK reports the program accepted, the controller makes a single audible
click, the arm does not move, the flange camera confirms it. Running the *same* program
by hand in RoboDK (right-click → Run, program marked "Run on robot") **does** move it, and
`AirOn`/`AirOff` run clean by hand.

The click appeared only after `92f2d1d` (assert `RUNMODE_RUN_ROBOT` right before
`RunCode()`); before it the signature was "program ran 0.5 s, no click, no motion"
(`deaad43`, `92f2d1d` commit messages). So the run-mode fix changed *something* about
what reaches the cell — but nobody has identified what clicks (§2).

---

## 2. The click is unidentified — and it decides the diagnosis

The first handoff treated the click as proof that "dispatch reaches the controller".
That is one of three readings, and they point in different directions:

| If the click is… | It means… | Then look at… |
|---|---|---|
| **Arm brake release** (from the arm itself) | The KRC started a motion command (`PTP COM_E6AXIS` / `LIN COM_E6POS`, driver module CASE 2/3) and immediately stopped: a zero-length move (target == current joints), program override `$OV_PRO` at 0 %, or a stop/halt. | Pendant message window at that instant; `$OV_PRO`; the driver log's first move target vs `$AXIS_ACT`. |
| **Cabinet relay / pneumatic solenoid** | The KRC executed a `$OUT[]` write (CASE 10). Every generated layer program starts with **moves** (rapid approach → path start) before its first valve call (`CallPathStart` = AirOn), so a valve click with no motion would mean the moves are being *skipped or executed as no-ops*, not refused. Also consistent with the API AirOff at job start reaching the KRC. | Whether the click coincides with `valve OFF: job startup` (before any program) or with the layer dispatch. |
| **Drives enabling / mains contactor**, tied to no instruction | The KRC is being asked to go to "drives on" and nothing follows. | Pendant: drives state, mode, enabling. |

**Stand next to the arm during one dispatch and settle this.** It costs nothing.

---

## 3. What is ruled out — measured vs inferred

| Item | Status | Evidence / gap |
|---|---|---|
| KUKA driver module version | **Measured** | `RoboDKsync570.src` (Nov 2024) and `RoboDKsync570 (1).src` (Jun 2026) in the operator's `Downloads` are byte-identical (`cmp`, re-verified in this revision). |
| `$OUT[0]` valve fault (`KSS014444`, module line 152) | **Measured** | Pendant message; fixed by renaming station IOs `IO_508`→`508` (`deaad43`); AirOn/AirOff clean by hand. |
| Camera / depth / work frame / hand-eye / tool TCPs | **Measured** | §7. |
| Pendant mode / drives / safety stop | **Inferred only** | From "a manual run moves it". Holds *only if* the manual run was done in the same session, seconds after the app failure, with no pendant interaction in between — record whether that was the case. Note the app's own error text tells the operator to check exactly these; the doc and the app currently disagree. **Never checked:** whether `RoboDKsync570` is *selected and running* (program pointer cycling in the `WHILE COM_ACTION >= 0` loop) at the moment of the app's dispatch. That KRL loop executes every driver command; KUKAVARPROXY accepts writes whether or not it runs, and RoboDK's "Ready" reflects only the socket. After the earlier `KSS014444` runtime error the pointer stopped at line 152 and needs acknowledge + Start to re-enter the loop. Also never checked: `$OV_PRO` (program override — the driver module does not set it; 0 % gives exactly "brakes click, no motion"). |
| Station run mode at `RunCode()` | **Inferred** | `92f2d1d` asserts it before every `RunCode()`; `RunMode()` has never been read back to confirm RoboDK honoured it. The commit's premise that `Program.Update()` "leaves the station in SIMULATE" is a hypothesis, not an observation. |
| Two RoboDK processes | **Not checked** | The app runs in `attach` mode and binds to whatever answers on the API port first. A leftover headless `-NEWINSTANCE -NOUI` process (from `rdk_extract.py` / `rdk_sync.py`) is invisible. `Get-Process RoboDK` must report exactly one. If the operator right-clicked the app's own `TasniCylinder_LIVE_*` program in their window for the manual run, this is excluded — record that. |
| API dispatch reaching the KRC at all | **Not checked** | The only evidence is the click (§2). Watch the valve at `valve OFF: job startup before motion`. |

### Dead ends already chased (do not repeat)

Stale work-frame centre; stale scan frame; wrong inspection tool; a bad scan; a 140 mm
build-plane offset (that was the *unsettled pose*, fixed in `b55ef5c`); an older
`RoboDKsync` driver file.

---

## 4. Facts about the dispatch path the first handoff left out

- **Layer programs are Curve Follow machining-project outputs**, not hand-built programs:
  `create_extrusion_layer_program` → `AddMachiningProject` + `setMachiningParameters`
  (`AutoUpdate` 0, inline cartesian moves, zero station targets).
  `CallPathStart`/`CallPathFinish` are program-call events to `AirOn`/`AirOff`. So each
  program is: frame/tool/speed → rapid approach → path start → **AirOn** → path → **AirOff**
  → retract. The moves come first.
- **Per-layer sequence** (`CylinderPrintJob.__call__`): `apply_run_mode("run_robot")`
  once at job start → `ensure_real_robot_link` → per layer: `valve_off` (AirOff via
  `RunCode`) → build program → `update_program` (`Program.Update`) → `_witness_frame()`
  (camera grab, up to `grab_timeout_s`) → `start_program` (`setRunType(RUN_ON_ROBOT)`,
  `setRunMode(6)`, `RunCode()`) → `_wait_program` → witness → `program_runtime_fault`.
- **RoboDK's own documentation** (`robolink.py`, `Item.RunCode`): with
  `setRunMode(RUNMODE_RUN_ROBOT)` + `program.setRunType(PROGRAM_RUN_ON_ROBOT)`, `RunCode()`
  runs "the same way as if we right clicked the program and selected Run on robot". So
  "manual works, API doesn't" means either a state we have not read differs
  (`RunMode()`, the return value, the driver log), or the two runs were not done under
  the same conditions (pendant state, RoboDK instance, timing).
- **`RunCode()` return value** is "the number of instructions that can be executed
  successfully (a quick program check is performed before the program starts)".
  `start_program` returns it; the job rejects only `< 0`. A `0` passes as success.
- **Driver**: `C:\RoboDK\api\robot\apikuka.exe` (KUKAVARPROXY) ↔ `RoboDKsync570.src`
  running on the KRC. Protocol: driver writes `COM_E6AXIS`/`COM_E6POS`/`COM_VALUE*`, then
  `COM_ACTION = n`; the KRL loop executes and sets `COM_ACTION = 0`; `$ADVANCE = 0`, so
  every move is blocking. CASE 2 = PTP joints, 3 = LIN, 5 = `$TOOL`, 6/7 = speed
  (`$VEL.CP`, `$VEL_AXIS[]`; values ≤ 0 are ignored), 8 = rounding, 10 = `$OUT[]`,
  12 = wait `$IN[]`. It never touches `$OV_PRO`.
- **Concurrency**: the frontend's status poll (`/api/rdk` → `robot_connected()` →
  `ConnectedState()`) hits the same `Robolink` while a print runs. `robolink` serialises
  calls with a lock; not known to interfere, but the standalone script in §5 E removes
  the question.

---

## 5. Next steps, ordered by cost × decisiveness

C and E are now built, so the whole ladder is: restart the app, press Print & record,
read the new log lines, and — if they do not settle it — run `tools/dispatch_bisect.py`
while watching the arm. One cell visit, a few minutes, no material printed.

**A. One dispatch, observed (0 code).** Start Print & record and, for that one run:
1. At `valve OFF: job startup before motion` — does the **physical valve** respond
   (LED / solenoid)? Yes → API→driver→KRC works for DO and the fault is motion-specific.
   No → nothing from the API reaches the KRL loop; the click is something else.
2. Identify the click (§2).
3. Read the **pendant message window** the moment it clicks (any `KSS…`, "Start key
   required", "enabling switch", "$OV_PRO", "command velocity", acknowledge prompts).
4. Read the pendant state: operating mode, drives, `$OV_PRO`, and whether
   `RoboDKsync570` is **selected and running** with its pointer cycling in the loop.
5. Then — **without touching the pendant** — right-click → Run the *same* kept program
   (`keep_artifacts` on) and note whether it moves. Record all five.

**B. RoboDK's driver log (0 code).** Open Connect → Connect robot, show the log
("More options" / "Show log"), and keep it visible during the dispatch. It shows every
command RoboDK sent to the driver and the driver's replies. Compare with a manual run of
the same program. This is the single most informative artifact available and it has
never been looked at.

**C. The three numbers the job threw away — ✅ DONE (`RdkIO.dispatch_program`).** Every
dispatch now logs `RunCode()`'s return value against the program's instruction count,
the run mode **read back** after setting it, and `saw_busy`. Both valve programs and
layer programs report, and both land in `provenance.dispatch`. **Just restart the app
and press Print & record** — the log now says which of these it is:

| New log line | Meaning |
|---|---|
| `RunCode returned 0 of 12 instructions … <-- RoboDK cleared ZERO instructions` | RoboDK's pre-run check refused the program. The fault is RoboDK-side and the message is in its driver log (B). |
| `… — station run mode is 1, NOT the 6 we set` | `92f2d1d`'s premise is wrong; the station will not hold RUN_ROBOT. |
| `RunCode returned 12 of 12 …, run mode 6` **and** `NEVER OBSERVED RUNNING` | RoboDK genuinely dispatched and the controller declined → pendant/driver (A.3–A.4). |
| valve OFF reports healthy, layer reports 0 | The generated Curve Follow program is the fault, not the dispatch path. |

The valve line is the **control**: AirOff is a two-instruction, no-motion program through
the identical path. Comparing the two reports splits "the API never reaches the
controller" from "the layer program specifically is refused" in the log alone.

**D. `Get-Process RoboDK` (10 s).** Must be exactly one. (Likely already excluded: the
operator right-clicks the app's own kept `TasniCylinder_*` program in their own window,
which means the app is attached to that instance.)

**E. Standalone bisect — ✅ written (`tools/dispatch_bisect.py`).** Runs outside the app
against the RoboDK window you have open, four rungs, ~2 min, no printing and no valve:

```
py -3.10 tools/dispatch_bisect.py link              # state only, no motion
py -3.10 tools/dispatch_bisect.py jog               # MOVES ~2° on A6 via the DRIVER, no program
py -3.10 tools/dispatch_bisect.py trivial           # MOVES: 2-instruction program it builds+deletes
py -3.10 tools/dispatch_bisect.py program <NAME>    # MOVES: the kept layer program
```

| Result | Conclusion |
|---|---|
| `jog` does not move the arm | Fault is **below RoboDK** — driver, KUKAVARPROXY, or `RoboDKsync570` not cycling on the pendant. Check the pendant before anything else. |
| `jog` moves, `trivial` does not | Driver + KRL loop + pendant are **fine**; API *program* execution is the fault. |
| `trivial` moves, `program` does not | The generated **Curve Follow program** is the fault, not dispatch. |
| all move | The fault is in the **app's** state/timing, not the API — diff this script's numbers against the app's log for the same program. |

`jog` is the important one and it is the cheapest: it is a direct `MoveJ` through the
driver (KRL `CASE 2`), so it tests the whole controller path with no program, no
machining project, and no valve.

**F. Only after A–E: the architectural fallback.** Drive motion through the driver
(`MoveJ`/`MoveL` per pose) instead of running a generated program per layer. The
operator raised this instinct too. Keep it as the fallback, not the first move — it
would hide a pendant/driver-state cause rather than fix it.

---

## 6. What was fixed today (all on `main`, pushed)

Baseline `9101aa1` → head `5fcb46a` (+ this doc).

| Commit | Fix |
|---|---|
| `1158106` | Camera prefers direct LAN (`10.12.171.70`), falls back to Tailscale; winner cached |
| `d62c294` | `/api/health` resolves the route itself instead of echoing the configured fallback |
| `a63696b` | Empty-ROI error names **which** band rejected the points, with counts |
| `4473aec` | **server/**: client socket no longer leaked when the handler dies (`CLOSE_WAIT` pile-up) |
| `b55ef5c` | Settled-pose read + commanded-vs-measured standoff cross-check before measuring |
| `7365e83` | Wait for a program to *become* busy (start race); retry the arrival check |
| `e7c16e3` | Catch a program accepted but never executed; log the validation phase |
| `69d24c2`→`46a7770` | Live collision validation is the operator's toggle (default **off**) |
| `e9a990e` | **Flange camera witnesses real motion** — the key diagnostic |
| `32dfd94` | Operator can keep generated RoboDK items after a run |
| `deaad43`, `9359e6d` | Valve outputs reach the KUKA driver as **indices**, not names; preflight always shows actual instructions |
| `92f2d1d` | **`setRunMode(RUNMODE_RUN_ROBOT)` before `RunCode()`** |
| `a019daf` | Poll the **robot's** busy state, not only the program item's; grace 0.5→5 s |
| `5fcb46a` | A run whose arm never moved must not report success |

New test files: `test_valve_outputs`, `test_rdk_io_run_mode`, `test_extrusion_wait`,
`test_extrusion_runtime`, `test_extrusion_motion_witness`, `test_extrusion_standoff`,
`test_camera_failover`, `test_server_client_lifecycle`.

---

## 7. Diagnostics now available (and what each can and cannot say)

- **Flange-camera motion witness** — `service.view_changed()`. Camera bolted to the
  flange: if the arm moves the view *must* change. Decisive when it says the arm did not
  move. It cannot say *why*.
- **`program ran N s`** — execution time from first observed busy to last. `0.0` means
  never observed busy for more than one poll (§0). It does not print `saw_busy` yet.
- **`valve OFF/ON` log lines** — RoboDK accepted the program. Not proof of execution.
- **Standoff cross-check** — `inspection.standoff_report()` / `standoff_fault()`; archived
  in `provenance.standoff` on every layer.
- **ROI band diagnostics** — per-band counts and Z distribution on the empty-ROI error.
- **Runtime guard** — `service.program_runtime_fault()` against RoboDK's predicted
  `time_s`.
- **Offline replay** — every layer archives `depth.npy`, `color.png`,
  `provenance.T_work_camera`.
- **Dispatch report** — `RdkIO.dispatch_program()` / `service.describe_dispatch()`.
  `RunCode()`'s return value vs the instruction count, the run mode read **back**, and
  `saw_busy`. Logged for valve and layer programs, archived in `provenance.dispatch`.
- **Bisect ladder** — `tools/dispatch_bisect.py` (`link` / `jog` / `trivial` / `program`).
- **Still not used**: RoboDK's driver/connection log, the pendant message window,
  `$OV_PRO`, whether `RoboDKsync570` is cycling. None of these live in software.

---

## 8. Verified-correct on the cell (do not re-investigate)

Measured 2026-08-28, all by direct observation:

- **Scan plane / work frame is correct.** Parked pose: `Realsense` TCP 471.1 mm above the
  work frame; camera measures the board at 467.5 mm — agree within 4 mm.
- **Depth is trustworthy.** Scan and extrusion inspection independently measure ~470 mm
  to the same board.
- **Camera is healthy** on the direct LAN path: one-shot grabs ~150–200 ms, held-open
  stream ~75 ms after a ~2 s first frame.
- **Tool TCPs**: `LongCalibTool` − `Realsense` = `(−34.7, 70.4, −363.5)` mm. Not 144 mm;
  the "wrong tool" theory is dead.
- **Valve mapping**: station IOs renamed to bare numbers; AirOn/AirOff run on the robot
  *by hand*.

---

## 9. Housekeeping

- **Discard `runs/extrusion/20260828-163731-f088cf48`** and every later
  `20260828-*-f088cf48` trial: all are measurements of an empty board (RMS 5.92 → 11.28 →
  32.10 mm is the plane drifting out of the ROI, not a print). None may reach the PFH
  paper.
- **Restart the app after every code change.** `/api/health` reports `build.stale`;
  check it before asking for a cell test. This wasted two cell runs today.
- **Jetson**: on In5 Wi-Fi at `10.12.171.70`, `autoconnect-priority 10`. `nmcli` over SSH
  needs `sudo -S`. See the `jetson-wifi-network-ops` memory.
- **Commit the KUKA driver module.** `RoboDKsync570.src` exists only in the operator's
  `Downloads` (not in `C:\RoboDK`). It cost most of a day to reverse-engineer `$OUT[0]`
  from a pendant code; the command map is in §4 now, but the file itself should live in
  the repo (e.g. `station/kuka/`).
- **Make the app's error text and this doc agree** about the pendant once §3's row is
  actually measured.

---

## 10. Process note for the next session

Three regressions were introduced today while chasing this, all from changing behaviour
without re-checking what depended on it:

1. Raising the start grace to 5 s made every program "outlast" its prediction, silently
   disarming the runtime guard.
2. The camera witness could only *clear* a suspicion, never raise one — so the signal
   that was right every time had no authority, and a run archived three measurements of
   an empty board.
3. Test fakes were physically incoherent (a camera 6 mm from its aim point serving a
   500 mm depth frame; a 2 s program completing instantly), and new guards fired on them
   correctly before firing on the real fault.

And one from writing the first version of this doc: three "ruled out" rows were
inferences presented as measurements, and the cheapest evidence on the cell — the pendant
message window, the driver log, the valve LED, three integers the job already has in
hand — was never collected. Several confident diagnoses were wrong because they reasoned
from partial evidence. **The measurements that resolved things were quick** — a depth
grab at the parked pose, a TCP comparison, reading the driver `.src`. Collect §5 A–D
before proposing a fix.
