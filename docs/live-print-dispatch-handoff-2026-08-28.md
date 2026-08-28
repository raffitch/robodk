# Live print: the arm does not move — handoff (2026-08-28)

**Status: UNRESOLVED.** The controller accepts the dispatched layer program, makes an
audible click, and does not execute the motion. Everything downstream of that has been
fixed, instrumented, or ruled out.

Read this with [docs/agent-debug-map.md](agent-debug-map.md). The extrusion module's
own background is [docs/extrusion-current-handoff.md](extrusion-current-handoff.md).

---

## 1. The one open problem

Clicking **Print & record** produces, per layer:

```
layer 1: validated in 0.0 s (RoboDK predicts 1.9 s of robot motion)
layer 1: program ran 5.0 s; flange camera says the arm did NOT move
```

The robot makes a click as if about to move. It does not move. Running the *same*
program by hand in RoboDK (right-click → Run) **does** move it, and `AirOn`/`AirOff`
now work by hand too.

So: dispatch reaches the controller (the click is new — it appeared only after
`92f2d1d`), and the controller then declines to execute.

### What is NOT the cause (each checked, with evidence)

| Ruled out | Evidence |
|---|---|
| Pendant mode / drives / safety stop | A manual run of the same program moves the arm |
| KUKA driver module version | `RoboDKsync570.src` (Nov 2024) and `RoboDKsync570 (1).src` (Jun 2026) are **byte-identical** |
| `$OUT[0]` valve fault | Fixed — operator renamed station IOs `IO_508`→`508`; AirOn/AirOff run clean |
| Station run mode at `RunCode()` | Fixed in `92f2d1d`; the click appeared as a result |
| Camera / depth / work frame / hand-eye | Verified correct on the cell (§4) |

### Next steps, in order

1. **Read whether the robot ever goes busy.** `_wait_program` now polls
   `rdk.robot_busy()` as well as `program_busy(name)`. If neither ever reports busy
   while the controller clicks, the driver is accepting and discarding the program —
   instrument `RunCode()`'s **return value** (it is documented as "the number of
   instructions that can be executed successfully"; the job only checks `< 0`, so a
   `0` return currently passes as success).
2. **Compare a manual run against ours at the API level.** The difference is now
   narrow: same program item, same run type, same run mode. Candidates left are
   RoboDK's machining-project program needing a different execution entry point, or
   the driver requiring the robot item to be the active one.
3. **Consider bypassing generated programs entirely.** Commanding motion through the
   driver (`MoveJ` to joints) instead of building and running a program per layer
   removes this whole class of problem. The operator raised this instinct
   independently. Inspection is the natural place to start (single pose, no path).

---

## 2. What was fixed today (all on `main`, pushed)

Baseline `9101aa1` → head `5fcb46a`.

| Commit | Fix |
|---|---|
| `1158106` | Camera prefers direct LAN (`10.12.171.70`), falls back to Tailscale; winner cached |
| `d62c294` | `/api/health` resolves the route itself instead of echoing the configured fallback |
| `a63696b` | Empty-ROI error names **which** band rejected the points, with counts |
| `4473aec` | **server/**: client socket no longer leaked when the handler dies (`CLOSE_WAIT` pile-up) |
| `b55ef5c` | Settled-pose read + commanded-vs-measured standoff cross-check before measuring |
| `7365e83` | Wait for a program to *become* busy (start race); retry the arrival check |
| `e7c16e3` | Catch a program accepted but never executed; log the slow validation phase |
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

## 3. Diagnostics now available (use these before theorising)

- **Flange-camera motion witness** — `service.view_changed()`. The camera is bolted to
  the flange, so if the arm moves the view *must* change. RoboDK's model cannot witness
  its own error; this can. It logs `the arm MOVED / did NOT move / motion unknown` per
  layer and is **decisive** when it says the arm did not move.
- **Standoff cross-check** — `inspection.standoff_report()` / `standoff_fault()`.
  Compares the distance the pose implies against the distance the camera reports.
  Archived in `provenance.standoff` on every layer, pass or fail.
- **ROI band diagnostics** — the empty-ROI error carries per-band counts and the Z
  distribution of points inside the radial band.
- **Runtime guard** — `service.program_runtime_fault()`. Compares execution time
  against RoboDK's own prediction (`update_program` → `time_s`).
- **Offline replay** — every layer archives `depth.npy`, `color.png` and
  `provenance.T_work_camera`, so any capture can be reprocessed without the cell.

---

## 4. Verified-correct on the cell (do not re-investigate)

Measured 2026-08-28, all confirmed by direct observation:

- **Scan plane / work frame is correct.** At the parked pose the `Realsense` TCP reads
  471.1 mm above the work frame and the camera measures the board at 467.5 mm — agree
  within 4 mm.
- **Depth is trustworthy.** Scan and extrusion inspection independently measure ~470 mm
  to the same board.
- **Camera is healthy** on the direct LAN path: one-shot grabs ~150–200 ms, held-open
  stream ~75 ms after a ~2 s first frame.
- **Tool TCPs**: `LongCalibTool` − `Realsense` = `(−34.7, 70.4, −363.5)` mm. Not 144 mm;
  the "wrong tool" theory is dead.
- **Valve mapping**: station IOs renamed to bare numbers; AirOn/AirOff run on the robot.

### Dead ends already chased (do not repeat)

Stale work-frame centre; stale scan frame; wrong inspection tool; a bad scan; a
140 mm build-plane offset (that was the *unsettled pose*, fixed in `b55ef5c`); an older
`RoboDKsync` driver file.

---

## 5. Housekeeping

- **Discard `runs/extrusion/20260828-163731-f088cf48`.** All three layers are
  measurements of an empty board (RMS 5.92 → 11.28 → 32.10 mm is the plane drifting out
  of the ROI, not a print). It must not reach the PFH paper.
- **Restart the app after every code change.** `/api/health` reports `build.stale`;
  check it before asking for a cell test. This wasted two cell runs today.
- **Jetson**: now on In5 Wi-Fi at `10.12.171.70`, `autoconnect-priority 10`. `nmcli`
  over SSH needs `sudo -S` (polkit denies a non-console session). See the
  `jetson-wifi-network-ops` memory.
- **Commit the KUKA driver module.** `RoboDKsync570.src` exists only in the operator's
  `Downloads`. It cost most of a day to reverse-engineer `$OUT[0]` from a pendant code;
  `CASE 10` is the digital-output handler and line 152 is `$OUT[io_id] = TRUE`.

---

## 6. Process note for the next session

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

Several confident diagnoses were also wrong because they reasoned from partial evidence
rather than measuring first. **The measurements that actually resolved things were
quick** — a depth grab at the parked pose, a TCP comparison, a radius profile, reading
the driver `.src`. Reach for them before theorising.
