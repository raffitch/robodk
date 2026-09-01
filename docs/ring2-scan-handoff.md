# Scan ring 2 from `ScanAtRing2` — task page

Opened 2026-09-01, revised same day (runtime-SET A-B-A folded in; §1 baseline
refreshed to 0/5). Three presses at one already-taught pose, no new code.

The operator has positioned the arm at a pose that frames **ring 2** of the stack and
saved it as the RoboDK target **`ScanAtRing2`**. The stack has **not** moved since the
2026-09-01 13:13 trial, so everything below is directly comparable to it.

The job: capture ring 2 from that pose and find out **what the camera actually returns
there**, because every layer-2 measurement so far has been invalid and nobody has yet
looked at ring 2 from a pose chosen for ring 2.

---

## 1. Why this is worth doing

Layer 1 measures beautifully and layer 2 does not, on the same stack, in the same trial:

| 2026-09-01 13:13 | layer 1 (8 takes) | layer 2 (5 takes) |
|---|---|---|
| valid | **8 / 8** | **0 / 5** |
| path completeness | 0.992–0.993 | **0.223–0.260** |
| max angular gap | 2.6–2.8° | **266.5–279.7°** |
| fitted radius | 41.7–42.2 mm | 39.0–42.4 mm |
| substrate sigma | 0.49 mm | 0.48–0.55 mm |
| separation margin | 1.57 mm | 14.39 mm |
| camera Z | 308.1 mm | 310.7 mm |

*(Updated 2026-09-01 evening: the table originally said 0/1 with take 2 dying in the
backend crash. Takes 03–05 were captured at 13:36, after the `bcd05cd` fix, as **one
parked excursion with `repeats=3`** — and they reproduce the dropout cleanly:
completeness 0.240 / 0.254 / 0.260, gaps 273.7° / 268.7° / 266.5°. So the dropout is
stable and reproducible, not a crash artifact. Takes 04/05 are the settled ones (§4);
the within-excursion climb 0.240→0.260 is the filter chain converging. The baseline to
beat is therefore **completeness ≈ 0.25–0.26, gap ≈ 267°, from takes 04/05** — not the
take-1 row, which was the unsettled first burst at that pose.)*

The inspection pose barely moves between the two layers — **2.6 mm** of camera Z — and
the substrate fit is just as clean. So the difference is not standoff and not the
substrate. Something about ring 2 *as seen from a ring-1-shaped pose* is not coming back.

> ### ⚠️ CORRECTION 2026-09-01 15:0x — `ScanAtRing2` IS the automatic pose
>
> This page was opened on the premise that `ScanAtRing2` is "the first pose chosen to
> frame ring 2 on its own terms". **It is not.** Read out of the running station against
> the 13:13 layer-002 archive:
>
> | | position (work frame) | orientation | baseline |
> |---|---|---|---|
> | archived automatic layer-2 pose | 208.940, 144.867, 310.700 | nadir | 179.8° |
> | `ScanAtRing2` (arm parked on it) | 208.940, 144.867, 310.700 | nadir | 179.8° |
> | **delta** | **0.000 mm** | **0.000°** | **0.00°** |
>
> Bit-identical. The taught target reproduces the automatic viewpoint exactly, so
> **layer-002 takes 03–05 of the 13:13 trial already ARE the capture this page asked
> for**, and a "does ring 2 measure better from `ScanAtRing2`?" run cannot answer
> anything — it re-measures a configuration already on disk.
>
> What survives: the pose lever is **tested and dead**, not untested. The remaining
> levers are the spatial filter (§3 arm B) and the camera roll (§4.1) — and since the
> arm is already parked on this exact pose, the filter A/B costs **zero robot motion**.
> Arm A1 is retained, but as a *control* (does the cell still reproduce 13:13 today?),
> never as a test of the viewpoint.

---

## 2. What the dropout looks like right now

Measured from the 13:13 archive with `tools/probe_roll_readings.py`'s sector counter
(shared centre, ring band r 32.6–52.6 mm). ROI points per 10° sector:

```
  0-59    397  357  224  189  200  188
 60-119   230  177  148  139  206   99
120-179     0   74   29  215  226  440
180-239   433  307   59   79  329  312
240-299    97    0    9  101  175   23
300-359    28  279   90  169  347  411
```

Two sectors read **literally zero**: 120–130° and 250–260°. The wider collapse runs
**120–140°** and **250–300°**, and the maximum elsewhere is 440.

Note this has **moved** since 2026-08-31, when the collapse was at 140–190° with lesser
lows at 30°, 240°, 260° and 330°. The camera baseline did **not** move: 179.1° then,
179.8° now. But the operator also moved and re-scanned the platform between the two, so
this is **not** a controlled comparison and must not be read as one — it is the reason
the roll probe exists.

### 2.1 What the archive already settles (added 2026-09-01 evening, offline, no robot)

Four things were measured off the 13:13 archive before spending any robot time. They
narrow the experiment but do **not** replace it.

**(a) The dead sectors are not dark. The shadow explanation is REFUTED.** The colour
image shows the taller stack casting a much bigger shadow than ring 1 did, which is a
tempting story. It is wrong: projecting every ring-band point into the colour image and
taking the mean luma per sector, the *deadest* sector of all — 120–129°, at 6–8 ROI
points — is one of the **brightest** (luma 92.4 against a 63–105 range). Correlation
between sector brightness and ROI yield is only +0.32, and it is carried entirely by
250–269°, which happens to be both dark and dead. Do not spend a lighting excursion.

**(b) The camera returns depth in the dead sectors — it just reads the TABLE there.**
In 120–140° the raw back-projected cloud still has 756–764 points in the ring band
(against 1040–1228 in a live sector), but only 75–86 of them sit above the substrate
floor, and their median height is **−0.3 to +0.15 mm**, i.e. substrate level, where a
live sector reads 5.4–7.7 mm. So this is not "no data". The ring's crest specifically
is not being resolved as elevated at those angles.

**(c) The dropout appeared WITH ring 2, at an unchanged pose.** The layer-1 and layer-2
inspection poses are the same XY, both nadir (`zaxis = -Z`), 2.6 mm apart in Z. At that
same pose, ring 1 alone yields 340 ROI points at 120–129°; the two-ring stack yields 8.

**(d) This stack's yield is strongly anisotropic about the stereo baseline — but that
is NOT a fixed camera property.** Folding sector yield onto |angle − baseline| gives:

| take | along-baseline | across-baseline | ratio |
|---|---|---|---|
| 13:13 layer-001 (ring 1 alone) | 225 | 247 | **0.91–0.95** |
| 13:13 layer-002 take04/05 (settled) | 340 | 148 | **2.27–2.30** |
| 2026-08-30 20:24 layer-002 (also 2 rings, baseline 179.1°) | 277–298 | 254–274 | **1.05–1.11** |

The 2-ring stack of 08-30 is the counter-example that matters: **same baseline, same
kind of scene, ratio 1.09** — so a pure stereo-aperture artifact of the baseline cannot
be the whole story, or that take would show it too. The anisotropy is highly
reproducible *within* a trial (2.27 vs 2.30) and absent in the same trial's layer 1, so
it is a real property of *this stack seen from this viewpoint* — and the archive cannot
say which of the two owns it. **That is precisely the camera-vs-scene question, and only
a controlled roll on the untouched stack answers it.** The one archived capture at a
genuinely different baseline (119.1°, 2026-08-31 17:12) is a different trial and stack,
so it is confounded exactly as the roll-probe handoff warns.

---

## 3. Protocol

**Do not touch the stack.** The whole value of this run is that it is comparable to the
13:13 trial.

1. **Check the cell**: `py -3.10 tools/cell_health.py`. It must say CELL OK. Start the
   backend with `.\start.ps1` if it is not running.
2. **Confirm the pose is still there** — `ScanAtRing2` in RoboDK — and that the arm can
   reach it without collision. If the operator has jogged since, re-teach it before
   anything else; a stale target is worse than no target. Capture uses **manual
   inspection mode** (`inspection_auto=false`, `inspection_target="ScanAtRing2"` —
   `tasni/modules/extrusion/models.py`), so no new code.
3. **Arm A1 — stock chain, the CONTROL.** Measure **layer 2** from `ScanAtRing2`, one
   press, `repeats=3` (what the 13:36 baseline triple used). Read takes 2–3; the first
   burst is not settled (§4). Since the target is the automatic pose (see the correction
   in §1), this arm asks only *"does the cell still reproduce 13:36 today?"* — it must
   land near completeness 0.25–0.26, gap ≈267°, along/across ≈2.3. If it does not, the
   stack or the cell moved and **nothing else on this page is interpretable**.
4. **Arm B — spatial filter off, via the runtime SET (no deploy, no restart).** With
   the robot home and **no capture in flight**, send `SET spatial=0` on TCP 1024 and
   confirm the `{"ok":true,...}` reply. Then the same press again (`repeats=3`). A
   successful SET rebuilds the filter chain, so the first take after ANY SET is a
   throwaway regardless of how familiar the pose is — read takes 2–3.
5. **Arm A2 — restore and control.** Send a **full explicit restore** (every
   `FILTER_SETTINGS` key with its stock value, per runtime-parameters spec 4.1 — do
   not trust leftover state), confirm with a bare `SET` read-back, and press once
   more. A2-vs-A1 is the drift control; if they disagree materially, the B arm is
   not interpretable and says so.
6. **Immediately compare** arm A1 against the settled 13:36 baseline in §1
   (completeness ≈ 0.25–0.26, gap ≈ 267°): completeness, maximum angular gap, fitted
   radius, substrate sigma. Then run the sector counter (§4, **fixed centre**) on all
   three arms and compare against the profile in §2.

SET guardrails (all live-verified 2026-09-01, `server/server_unicast_syncronous.py`):

- Never send SET while a capture is in flight — a successful SET **retires the camera
  generation** and closes every greeted session, killing the take.
- Read each take's arm off its own manifest,
  `provenance.camera_geometry.filter_options.spatial_smooth_delta` (**20** = stock,
  **`null`** = spatial off) — never off what you sent. A Jetson service restart
  mid-A/B silently reverts to the unit file's boot defaults; the per-take read-back
  is how you catch it.
- There is no host-side SET sender in `tasni/` or `tools/` yet — it is a manual
  socket line (connect, consume the greeting, send `SET key=value ...\n`, read one
  JSON reply). Bare `SET` is read-only and echoes the achieved state.

---

## 3.5 RESULT — run 2026-09-01 15:38, trial `20260901-153855-8fe68421`

All three arms ran, 16 takes, **zero arm motion** (the arm was already parked on the
target). Every take's arm is verified from its own
`provenance.camera_geometry.filter_options.spatial_smooth_delta`.

### The spatial filter is FALSIFIED as the mechanism

The A/B that `docs/inspection-roll-probe-handoff.md` §3.1 had been deferring since
2026-08-31, run properly at last on an untouched stack:

| arm | n | ROI points | across-baseline | along/across ratio |
|---|---|---|---|---|
| A1 stock (`delta 20`) | 5 | 8023 | 145.7 | 2.25–2.35 |
| **B spatial OFF** (`null`) | 6 | **7684** | **141.9** | **2.18–2.28** |
| A2 stock, restored | 3 | 8007 | 144.2 | 2.24–2.41 |

Switching the spatial filter off did **not** recover the missing sectors. It returned
*fewer* points everywhere (−4%) and left the anisotropy intact. The dead sectors
(120–130°, 250–270°, 290–300°) sit in exactly the same places in all three arms. The
`smooth_delta=20` vs "<4 disparity pixels" story was the leading hypothesis and it is
now dead — **do not spend anything more on it.**

**A1 versus A2 is the drift control and it PASSES** (2.25–2.35 against 2.24–2.41), so
the B arm is interpretable rather than a comparison against a moving baseline.

### The stack has NOT moved since 13:13

A1 reproduces the 13:13 settled takes on every stable measure (ratio 2.27/2.30 then,
2.25–2.35 now; ROI n ≈ 8000 both). Four and a half hours and a backend restart apart.
The anisotropy is a real, reproducible property of this configuration.

### ⚠️ `path_completeness` is NOT a stable readout — do not tune against it

The same 16 takes, same pose, same stack, same chain, report completeness anywhere from
**0.253 to 0.891** while the underlying sector coverage varies by only a few percent:

| take | completeness | gap | fitted radius | ROI points | ratio |
|---|---|---|---|---|---|
| A1 take02 | 0.253 | 268.8° | 37.68 | 7721 | 2.25 |
| A1 take03 | 0.624 | 135.4° | 43.37 | 8039 | 2.32 |
| B take07 | **0.891** | **39.2°** | 42.21 | 7762 | 2.18 |
| B take10 | 0.263 | 265.2° | 41.89 | 7664 | 2.26 |

Take 07 is the best ring-2 measurement ever recorded — and take 10, six presses later on
the *same* filter arm, is back at the old 0.26. The swing tracks where the circle fit
lands (radius 37.7–44.2 against a nominal 42.0), not what the camera saw: on a ring with
a 100°+ gap the fit is under-constrained, and completeness is measured against the
nominal circle, so a fit that drifts a few mm in radius swings completeness by a factor
of three. **Reading take 07 alone would have "proved" spatial-off is a huge win. The
coverage statistic says it is not.** Judge levers on ROI coverage and the along/across
ratio; treat a single take's completeness as noise.

This also explains the layer-2 invalidity in a way the earlier framing missed: two
compounding faults, a real sensor-level anisotropy AND a fit that cannot stabilise on a
partial ring. Fixing only one will not produce a valid layer-2 measurement.

### Branch-guard aborts are now routine on this stack

6 of 16 takes died with "branch guard exhausted after 3 attempts" — the frame measures
the bead at 4.2–4.6 mm against the recipe's 8.1 mm, giving `spur_limit` 7 px. The take
still archives raw RGB-D, so the coverage analysis above survives it; only the pipeline's
own metrics are lost. Do not loosen the guard (see the run-day note in
`docs/pfh-paper-handoff.md`) — reprocess instead.

---

## 4. Reading it

**Settling first, before any conclusion.** Measured 2026-09-01 on the 2026-08-31
archive: a settled burst's first and last frames correlate at **+0.937**, while the
first burst at a fresh pose manages **+0.045** — it does not even agree with itself,
because the Jetson's filter chain is still converging through it. `settle_s` does not
help; it elapses before the host opens the depth stream, and the chain only advances
when a client pulls frames. A burst is already settled by the *second* one at the same
pose. Two settled bursts of one ring reproduce each other's static field at **+0.960**.

Get the sector profile with the existing probe's own function — no new code — but
**hold the centre fixed**. With a 270° gap an arc fit's centre is poorly constrained,
and the outcome discrimination below hinges on whether the dropout's work-frame ANGLES
move — so per-take `fit_ring` centres would shift the sector labels between captures.
Use the trial's work centre and the same band §2's profile used:

```python
from tools.probe_roll_readings import load_take, sector_counts
import numpy as np
# session.json's APPLIED setup centre -- what every take was actually scored
# against. NOT trial.json's setup centre (209.45, 147.10), which is the
# pre-characterization placement and sits 2.3 mm away in Y.
c = np.array([208.940, 144.867])
t = load_take("runs/extrusion/<trial>/layer-002")
p = sector_counts(t["points"][t["roi"]][:, :2], c, inner_mm=32.6, outer_mm=52.6)
```

That centre is independently corroborated: ring 1's eight takes fit their centre to
(209.0–209.3, 144.8–145.2), agreeing with it to **0.3 mm**. Per-take `fit_ring` centres
are not usable for this — on `layer-001-take08` the same call returns (196.05, 143.69),
**12.9 mm** away, which would rotate every sector label.

Recomputed on the pinned centre, the §2 profile reproduces across all four layer-002
takes, so the dropout is stable rather than noise (take05, ROI points per 10°):

```
  0-59    473  484  345  172  199  210
 60-119   253  195  177  132  231  144
120-179     6   69   48  203  277  459     <- 120-129 is the deep hole
180-239   495  306  100  173  372  371
240-299   191   36   12  187  184   27     <- 250-269 the second
300-359    48  293  172  174  317  440
```

Outcomes, and they lead different places (arm letters from §3):

- **Arm A1 does NOT reproduce the 13:36 numbers.** Stop. The stack or the cell moved
  since 13:13 and every comparison on this page is void — re-establish a baseline
  before spending anything else. (This is the only outcome A1 can now produce that
  changes what you do: it is the automatic pose, so "ring 2 measures cleanly from
  `ScanAtRing2`" is not on the table — that configuration is already on disk at
  completeness 0.25.)
- **A1 drops out, B (spatial off) restores it.** Mechanism found in one excursion:
  `smooth_delta=20` against a ring spanning under 4 disparity pixels. Stop and tune
  the spatial path before asking multi-view or more poses to repair it.
- **A1 and B both drop out, and the dropout MOVES with the pose** (vs §2's angles).
  The spatial filter is exonerated at this pose; the artifact is viewpoint-tied.
  That is the camera-locked signature, and it makes the roll probe (§5) the right
  next instrument rather than more excursions.
- **The dropout stays at the same work-frame angles in every arm.** Something is
  genuinely there — or genuinely absent — on the ring itself. Then stop scanning and
  go look at the physical ring at 120–140° and 250–300°.

If **A2 disagrees materially with A1**, the comparison was not stable and the B arm
is not interpretable — say so rather than picking the convenient reading.

**Report the along/across-baseline ratio for every arm** (§2.1d), not just completeness.
It is the statistic that carries the camera-vs-scene question: this stack reads **2.27–2.30**
from the automatic pose, where its own ring 1 reads 0.91–0.95. A ratio that stays near
2.3 through a pose change and a filter change is evidence the geometry is at fault; one
that collapses toward 1.0 names whichever lever moved it.

### 4.1 Why the roll arm is still the decisive one

The archive can show the anisotropy but cannot own it (§2.1d): the 2026-08-30 two-ring
stack read 1.09 at the same baseline, so "the baseline does it" is already falsified as
a complete explanation, and no capture at a *different* baseline exists that is not also
a different stack. Only rotating the camera about its optical axis, on this untouched
stack, separates them — and it makes a quantitative prediction rather than a vague one:

> **If the anisotropy is camera-locked, a roll of θ moves the dead sectors by θ and the
> along/across ratio stays ≈2.3 about the NEW baseline. If it is scene-locked, the dead
> sectors stay at 120–129° and 250–269° in the work frame and the ratio about the new
> baseline collapses toward 1.0.**

### 4.2 Running the roll arm — protocol

Arm and disarm it with `tools/inspection_roll.py`; it exists because three separate
mistakes here each produce a run that looks fine and answers nothing.

**Adding 90° to the candidate list does nothing.** The default
`inspection_roll_candidates_deg` is the *fallback ladder* `[0, 180, 90, 270]`, and the
generator accepts the FIRST candidate with IK that passes validation — roll 0 always
does, so 90 is never reached. A forced roll must be the **only** candidate. And it must
be the only one for a second reason: with `[90, 0]`, a refused 90 quietly captures at
roll 0 and archives a take that looks ordinary. Armed as a single value, a refusal fails
the run loudly instead.

**90° sits exactly ON the wrist limit.** A roll about a nadir optical axis is
essentially that much A6, `max_tool_axis_spin_deg` defaults to 90, and the gate is
`abs(delta) > limit` — so exact-90 passes only if floating point cooperates. It was
refused on this cell before. Give it headroom in the **trial's setup**
(`maximum_tool_axis_spin_deg`, e.g. 110), never in the global default: the looser limit
then applies to one measure-only trial, is recorded in its fingerprint and `trial.json`,
and never reaches a print path — which matters on a cell with a documented wrist-flip
history. If you would rather not touch the limit at all, **85° answers the same question**:
the effect is 180°-periodic, so 85° still swings along↔across almost fully, and 10°
sectors resolve it trivially.

**The config is read at startup**, so arming changes nothing until the backend restarts.
The plan survives that restart — `_restore_plan_from_session` rebuilds it from
`session.applied` — so a roll can be swapped mid-trial without losing the session.

Sequence, on one untouched stack:

1. Build the stack and measure normally (roll 0, ladder disarmed). This is the
   comparable arm — the ring-1 layer doubles as a null control, since ring 1 alone
   showed no anisotropy (0.91–0.95).
2. `py -3.10 tools/inspection_roll.py 90`, restart, re-measure **the same layer**.
3. `py -3.10 tools/inspection_roll.py --disarm`, restart, when finished.

**Verify the achieved roll before believing anything.** Read the first rolled take's
`provenance.T_work_camera` and check its X-axis angle in the work frame actually moved
(179.8° → ~269.8° for a 90° roll). Never trust that a roll happened because it was
requested — that is the same rule the filter arms follow.

Whatever happens, **record which**. A run that is not written down costs the same robot
time and buys nothing.

---

## 5. Traps

- **The KUKA must be in EXT/AUT, not T1, or the capture hangs uncancellably.**
  (2026-09-01, cost 16 minutes.) `_move_to_inspection` dispatches the inspection move
  as a RoboDK program with `real_robot=True` and then blocks in `_wait_program`
  (`modules/extrusion/service.py:295`), which has **no timeout** and only tests
  `ctx.cancelled` *between* polls. In T1 the controller needs the enabling switch held
  and will not execute a driver-dispatched program, so RoboDK waits for a completion
  that mode cannot produce and `POST /cancel` returns `cancelling` and does nothing.
  This happens even when the arm is **already standing on the target** and the move is
  zero-distance. How to recognise it, in the order that separates causes:
  `/api/health` keeps answering in ~200 ms with `robodk: ok` (it proves nothing);
  **`/station-options` blocks**, because the wedged job holds the shared `services.rdk`;
  the Jetson shows **no ESTABLISHED connection on 1024** (`ss -tn state established
  '( sport = :1024 )'`) while health still says `camera: in_use`, since the lease flag
  outlives the socket; and RoboDK's CPU barely advances while `Responding` stays True.
  Do **not** kill RoboDK or the backend to escape it: a program already dispatched runs
  on the controller, so killing the host does not stop the arm. Switch the robot to
  EXT/AUT and re-run. `robot.ConnectedState()` reporting `(0, 'Ready')` does **not**
  mean the robot will accept a dispatched program — it only means the driver is
  connected.
- **Layer ≥ 2 has a known history.** Before `bd455a7` the chain measured the largest
  surviving arc of the stack rather than layer N, so any completeness number from
  before 2026-08-31 21:56 is not comparable. The archived `report.json` of older takes
  still holds those pre-fix values.
- **The backend crashed on 2026-09-01 during layer-002 take 2** and left an all-zero
  report. Root cause is now found and fixed (`bcd05cd`: two OpenBLAS runtimes
  multithreading in one process). If it dies again, read the **Windows
  Application-Error log** first — `Get-WinEvent ... Id -eq 1000` names the faulting
  DLL. The faulthandler dump does not survive a multi-thread fault.
- **`spatial_smooth_delta` is now archived** (`e06a2c5`) and read **20.0** on the 13:13
  takes. Any take captured by a Jetson running that server records what the filter
  really ran at; takes older than it record `null`, meaning UNKNOWN, never 20.
- **Do not reprocess the 2026-08-30 archive** — `tests/test_extrusion_golden.py` uses
  its `report.json` as a "has anyone reprocessed this?" cross-check. (Reading it with
  `probe_roll_readings.load_take`, as §2.1d does, is not reprocessing: it writes
  nothing.)
- **Manual inspection changes the plan fingerprint.** `plan_fingerprint` hashes the
  whole setup, `inspection_auto` and `inspection_target` included
  (`modules/extrusion/toolpath.py:13`), so pointing the run at `ScanAtRing2` yields a
  different fingerprint from the 13:13 session's `applied` one and `/measure/layer`
  refuses it as the stale-plan artifact. The correct move is a **new session** carrying
  the SAME characterization-applied recipe and placement (radius 42.0, layer height 2.6,
  bead 8.1, centre 208.940/144.867 — from `session.json`'s `applied`), changing ONLY the
  two inspection fields. The nominal circle is then bit-identical, so completeness, gap
  and radius stay directly comparable. Do **not** re-characterize: that would move the
  centre and break comparability with the 13:13 numbers.
- **The spatial-filter A/B is DONE and NEGATIVE** (§3.5, 2026-09-01). Turning the
  filter off does not recover the sectors; it loses 4% of the points and leaves the
  anisotropy at 2.2. Do not re-run it, and do not treat `smooth_delta` as an open
  lever. Note `docs/inspection-roll-probe-handoff.md` §3.1's "deploy the server before
  the sweep" **predates the SET tier** — an arm is now one socket line, no deploy;
  only the per-take `filter_options` read-back rule there still binds.
- **Driving an arm is now one command** — `tools/camera_set.py`, added 2026-09-01
  because the merged runtime-parameters work was server-side only and the A/B above
  had to be hand-rolled:
  ```
  py -3.10 tools/camera_set.py              # READ-ONLY: prints the achieved arm
  py -3.10 tools/camera_set.py spatial=0    # one arm, no deploy, no restart
  py -3.10 tools/camera_set.py --restore    # the full spec-4.1 explicit restore
  ```
  Confirm the arm with the bare read-only form BEFORE and AFTER every run, and read
  each take's actual arm from its archived `filter_options` — never from what you
  sent, because a Jetson service restart mid-sweep silently reverts to the unit
  file's defaults.
