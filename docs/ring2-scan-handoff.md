# Scan ring 2 from `ScanAtRing2` — task page

Opened 2026-09-01. One robot excursion, one already-taught pose, no new code.

The operator has positioned the arm at a pose that frames **ring 2** of the stack and
saved it as the RoboDK target **`ScanAtRing2`**. The stack has **not** moved since the
2026-09-01 13:13 trial, so everything below is directly comparable to it.

The job: capture ring 2 from that pose and find out **what the camera actually returns
there**, because every layer-2 measurement so far has been invalid and nobody has yet
looked at ring 2 from a pose chosen for ring 2.

---

## 1. Why this is worth doing

Layer 1 measures beautifully and layer 2 does not, on the same stack, in the same trial:

| 2026-09-01 13:13 | layer 1 (8 takes) | layer 2 |
|---|---|---|
| valid | **8 / 8** | **0 / 1** (take 2 died with the backend) |
| path completeness | 0.992–0.993 | **0.223** |
| max angular gap | 2.6–2.8° | **279.7°** |
| fitted radius | 41.7–42.2 mm | 40.40 mm |
| substrate sigma | 0.49 mm | 0.55 mm |
| separation margin | 1.57 mm | 14.39 mm |
| camera Z | 308.1 mm | 310.7 mm |

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
   anything else; a stale target is worse than no target.
3. **Capture from `ScanAtRing2`**, measuring **layer 2**, with the same `repeats` the
   13:13 trial used.
4. **Take at least two takes at that pose and read the SECOND.** The first burst at any
   fresh pose is not settled — see §4. This is not optional; it is the difference
   between measuring the ring and measuring the filter's transient.
5. **Immediately compare** against the 13:13 layer-002 numbers in §1: completeness,
   maximum angular gap, fitted radius, substrate sigma. Then run the sector counter
   (§4) and compare against the profile in §2.

---

## 4. Reading it

**Settling first, before any conclusion.** Measured 2026-09-01 on the 2026-08-31
archive: a settled burst's first and last frames correlate at **+0.937**, while the
first burst at a fresh pose manages **+0.045** — it does not even agree with itself,
because the Jetson's filter chain is still converging through it. `settle_s` does not
help; it elapses before the host opens the depth stream, and the chain only advances
when a client pulls frames. A burst is already settled by the *second* one at the same
pose. Two settled bursts of one ring reproduce each other's static field at **+0.960**.

Get the sector profile with the existing probe's own function — no new code:

```python
from tools.probe_roll_readings import load_take, fit_ring, sector_counts
t = load_take("runs/extrusion/<trial>/layer-002")
c, r = fit_ring(t)
p = sector_counts(t["points"][t["roi"]][:, :2], c, inner_mm=r-10, outer_mm=r+10)
```

Three outcomes, and they lead different places:

- **Ring 2 measures cleanly from `ScanAtRing2`** (completeness > 0.9, gap < 30°). The
  automatic layer-N inspection pose is the fault. Compare that pose's `T_work_camera`
  against `ScanAtRing2`'s and find what differs — the standoff is nearly identical, so
  look at where the ring sits in frame and at incidence, not at distance.
- **The dropout survives but MOVES with the pose.** It is tied to the viewpoint, not to
  the ring. That is the camera-locked signature, and it makes the roll probe (§5) the
  right next instrument rather than more excursions.
- **The dropout stays at the same work-frame angles.** Something is genuinely there — or
  genuinely absent — on the ring itself. Then stop scanning and go look at the physical
  ring at 120–140° and 250–300°.

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
- **The spatial filter is still the untested lever.** The A/B in
  `docs/inspection-roll-probe-handoff.md` §3.1 (`RS_SPATIAL=0` on the same untouched
  stack) has never been run, and it is the only untried thing with a measured mechanism
  for losing a ring: `smooth_delta` 20 against a ring spanning under 4 disparity pixels
  at 300 mm. If ring 2 drops out from `ScanAtRing2` as well, run that before anything
  more elaborate.
