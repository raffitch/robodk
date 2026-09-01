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

`ScanAtRing2` breaks that: it is the first pose chosen to frame ring 2 on its own terms.
If ring 2 measures cleanly from there, the fault is the automatic inspection pose for
layer N, which is a bounded, fixable thing. If it still drops out, the fault is in the
ring or the sensor, and the roll probe (§5) is the next instrument, not more poses.

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
3. **Arm A1 — stock chain, the comparable capture.** Measure **layer 2** from
   `ScanAtRing2`, one press, `repeats=3` (what the 13:36 baseline triple used). Read
   takes 2–3; the first burst at a fresh pose is not settled (§4). **Every vs-13:13
   conclusion comes from this arm only.**
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
c = np.array([209.45, 147.10])          # trial.json setup centre, shared by all takes
t = load_take("runs/extrusion/<trial>/layer-002")
p = sector_counts(t["points"][t["roi"]][:, :2], c, inner_mm=32.6, outer_mm=52.6)
```

Outcomes, and they lead different places (arm letters from §3):

- **Ring 2 measures cleanly from `ScanAtRing2` in arm A1** (completeness > 0.9,
  gap < 30°). The automatic layer-N inspection pose is the fault. Compare that pose's
  `T_work_camera` against `ScanAtRing2`'s and find what differs — the standoff is
  nearly identical, so look at where the ring sits in frame and at incidence, not at
  distance. Arms B/A2 still deliver the never-run crest-height A/B for free.
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

Whatever happens, **record which**. A run that is not written down costs the same robot
time and buys nothing.

---

## 5. Traps

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
  its `report.json` as a "has anyone reprocessed this?" cross-check.
- **The spatial-filter A/B is folded into §3 as arms B/A2** (updated 2026-09-01: the
  runtime SET made it a one-line arm change, so it no longer waits for a separate
  excursion). It is the only untried lever with a measured mechanism for losing a
  ring: `smooth_delta` 20 against a ring spanning under 4 disparity pixels at 300 mm.
  Note `docs/inspection-roll-probe-handoff.md` §3.1's "deploy the server before the
  sweep" **predates the SET tier** — arming an arm needs no deploy now; only the
  per-take `filter_options` read-back rule there still binds.
