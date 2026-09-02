---
name: live-print-dispatch-blocker
description: "Live print dispatch blocker (controller clicks, arm does not move) — RESOLVED 2026-08-28 per operator ('a lot of fixes and now the program works', commit 4ac5c3f); kept for the evidence-first lesson and the bisect tool"
metadata: 
  node_type: memory
  type: project
  originSessionId: c89acaf0-4c5f-442c-8061-0baf8798a920
  modified: 2026-08-28T13:00:11.570Z
---

**RESOLVED 2026-08-28** — operator reported "a lot of fixes and now the program works"; the
fix landed as `4ac5c3f` "start real programs through working RoboDK API path" (another
session; I have not read its diff). The measure-only inspection move ran on the cell that
evening via `rdk.start_program(..., real_robot=True)`. The history below is kept for the
bisect tool and the evidence-first lesson.

**2026-08-28 (earlier), was UNRESOLVED.** Extrusion live print (Print & record): RoboDK accepts the
dispatched layer program, the cell makes one audible click, the arm does not move. A
manual right-click → Run on the same program DOES move it; AirOn/AirOff work by hand.

**Handed off for Codex/any tool (`2b7d901`,`e29c668`):** repo-root **`AGENTS.md`** is now the
tool-agnostic entry point (working agreement + the env traps that were ONLY in this memory:
`py -3.10`, no full pytest, PowerShell UTF-8 mojibake, restart-for-build.stale), and
**`docs/live-print-next-session.md`** is the continuation handoff (decision tree per bisect
outcome, file:line map, fallback designs). Cross-linked from the debug map.

**Read [docs/live-print-dispatch-handoff-2026-08-28.md] for the evidence** (revised 2026-08-28 by a
second agent; code untouched). Key corrections vs the first version:
- The latest log's `program ran 0.0 s` = neither `program.Busy()` nor `robot.Busy()` was
  ever seen True in the 5 s grace → RoboDK never really ran it (on robot OR in sim).
- `valve OFF:` log lines only mean RoboDK accepted the program — not that the KRC did it.
- The "click" is unidentified (arm brakes vs cabinet relay vs valve solenoid) and each
  reading points somewhere different.
- "Pendant mode ruled out" and "run mode fixed" were INFERENCES, not measurements.
  Never checked: RoboDKsync570 selected+running on the pendant at dispatch time, `$OV_PRO`,
  `RunMode()` read-back, `RunCode()` return value (job ignores anything ≥ 0), RoboDK's
  driver/connection log, RoboDK process count.

**Why:** most of the day was lost to confident diagnoses reasoned from partial evidence;
the first handoff then presented some of those inferences as rule-outs.

**Instrumentation now EXISTS (`89e1345`/`bc7c53a`, main):** `RdkIO.dispatch_program()`
logs RunCode's return vs instruction count + run mode READ BACK + saw_busy, for valve AND
layer programs (valve = the no-motion control: healthy valve line beside a refused layer
line localises it to the generated Curve Follow program, from the log alone).
`tools/dispatch_bisect.py` = link/jog/trivial/program ladder; `jog` (direct driver MoveJ,
no program) is the decisive rung: it fails => fault is BELOW RoboDK (pendant/driver).

**INSTRUMENTED RUN DONE — both software hypotheses DEAD:** `RunCode returned 195 of 195
instructions, run mode 6`, and the 2-instruction VALVE program dispatches identically
(`2 of 2, run mode 6`) and is equally never-observed-running. So: nothing was refused,
run mode is genuinely 6, and it is NOT about what the program contains (stop looking at
the Curve Follow program, toolpath, valve mapping). It is API-dispatched execution as
such. Added `37d4f18`: `driver_state()` samples the DRIVER's ConnectedState
(WORKING/PROBLEMS vs READY) across the wait — never leaving READY = RoboDK dispatched
nothing to the driver — plus ~250 Hz polling for the first 0.5 s (a flat 50 ms poll
cannot tell "never started" from "died in 20 ms").

**How to apply:** run `py -3.10 tools/dispatch_bisect.py jog` FIRST (30 s, no app, direct
driver MoveJ, no program). Arm moves => driver/KRL/pendant fine => RoboDK's program
executor is the fault. Arm does not move => fault is BELOW RoboDK and "Ready" is only the
socket — check whether RoboDKsync570 is actually cycling on the pendant. Then re-run the
app for the new driver-state line. Do not add more guesses first.
Do not re-chase: stale frame/centre, wrong tool, bad scan, driver file version, `$OUT[0]`.
Check `/api/health` `build.stale` before ANY cell test ([[restart-tasni-backend-after-code-edits]]).
Discard every `runs/extrusion/20260828-*-f088cf48` trial — empty-board measurements, must
not reach the PFH paper ([[pfh-paper-ring-stack-experiment]]).
Driver: `C:\RoboDK\api\robot\apikuka.exe` ↔ `RoboDKsync570.src` (only in operator's
Downloads; command map now in the doc §4). Jetson network: [[jetson-wifi-network-ops]].
