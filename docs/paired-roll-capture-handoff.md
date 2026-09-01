# Paired vertical/horizontal capture — task page

Opened 2026-09-01 17:1x, at `033c70b`. Continues
**[ring2-scan-handoff.md](ring2-scan-handoff.md)**, which is the analysis of record
for *why* any of this matters. Read that page's §2.1 and §3.5 first; this page is
only the state of the instrument and what to do next.

---

## 0. State in one paragraph

The app can now capture **every frame twice from one trip out** — once as the pose
generator chooses ("vertical") and once with the camera rolled 90° about its own
optical axis ("horizontal"). It is ON by default and applies to the whole guided
sequence, characterization included. It has run on the cell **once**, at 16:57, and
that attempt exposed two bugs which are now fixed but **NOT yet re-run**. The backend
at the time of writing is up, idle, and already running the fix (process started
17:06:08, last source edit 17:03:49) — no restart needed.

**The immediate next action is to re-run a characterization and check the pair is
clean.** That is one press, and it is the acceptance test for everything below.

---

## 1. What to do next

1. In the Extrusion page, tick **Hands clear** and leave **"Vertical + horizontal
   (roll the camera 90° for a second view)"** ticked (it now defaults on).
2. Press **Characterize ring 1**.
3. Read the log. You want two `ring (...)` lines — one `vertical`, one `horizontal` —
   and **no** `WARNING ... SKIPPED` / `invalid`.
4. Run the acceptance check in §2 before trusting anything.

If the horizontal view is skipped, the message now names the reason and the run keeps
the vertical view. That is the designed behaviour and it is not a crash — report the
reason rather than working around it.

---

## 2. Acceptance check — is the pair actually a pair?

**A label is not proof.** The only proof of what was captured is
`provenance.T_work_camera` in each take's own report. Run this on the newest trial:

```python
import json, math
from pathlib import Path
import numpy as np

TRIAL = Path("runs/extrusion/<newest>")
for d in sorted(p for p in TRIAL.iterdir() if p.is_dir()):
    rep = d / "report.json"
    if not rep.is_file():
        continue
    data = json.loads(rep.read_text(encoding="utf-8"))
    prov = data.get("provenance") or {}
    T = prov.get("T_work_camera")
    if T is None:
        continue
    T = np.asarray(T, float)
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(-T[2, 2])))))
    base = math.degrees(math.atan2(T[1, 0], T[0, 0])) % 360.0
    print(f"{d.name:18} {prov.get('orientation'):10} tilt={tilt:5.2f}  "
          f"baseline={base:6.1f}  xyz=({T[0,3]:7.2f},{T[1,3]:7.2f},{T[2,3]:7.2f})")
```

The pair is valid **only** if all three hold:

| check | required | why |
|---|---|---|
| `tilt` | **0.00° on BOTH** | a tilted view is a different measurement; incidence costs ~4x what distance costs |
| `xyz` | identical on both | the pair must differ by roll alone |
| `baseline` | differs by **90°** (179.8 → 89.8 **or** 269.8) | 89.8 and 269.8 are the SAME axis — either is correct |

If tilt is non-zero on the rolled view, the tilt-substitution bug is back — stop and
read §4.1.

---

## 3. What was wrong on the 16:57 run (both now fixed, both unverified)

The rolled view rotated correctly but came back **tilted**, and its characterization
was invalid: `substrate separation collapsed: -15.784 mm ... board p99 26.233 mm`.
`runs/extrusion/20260901-165716-8c5ee676/characterize-02` is the evidence.

The pose record named the cause outright: `roll 90 / tilt 0` was **rejected** ("no IK
solution on the neutral wrist branch"), and the candidate walk then fell through the
tilt ladder to `tilt 10 / azimuth 270`. That moved the camera 52 mm sideways
(y 147.10 → 95.01) and tilted the axis 10°, which is what collapsed the substrate fit.

1. **Tilt ladder applied to a commanded roll** (`033c70b`). A commanded roll now gets
   `tilt 0 / azimuth 0` only. The one permitted alternative is the **opposite** roll:
   θ and θ−180 put the camera X axis on the same *line*, reversed, and a stereo
   baseline is an axis rather than a direction — so they are interchangeable, and
   trying both is what makes 90° reachable when only one wrist direction solves.
2. **The branch guard refused what was requested** (`033c70b`). The operator confirms
   a 90° roll IS reachable, so the refusal was the gate. `solve_joints_on_neutral_branch`
   required the neutral `JointsConfig` triple (front/rear, elbow, **wrist-flip**), and a
   90° tool-axis roll legitimately changes the wrist-flip flag. `allow_wrist_flip` now
   drops **only** that flag and **only** for a commanded roll. Front/rear and elbow stay
   locked — those swing the arm through the cell — and the axis-4/6 magnitude bound is
   unchanged.

---

## 4. Traps

### 4.1 Anything that silently substitutes a different viewpoint
This has now bitten twice in one day, in two different ways. A capture that looks
completely ordinary in the archive but was taken from somewhere else is the failure
mode to design against, not an edge case.

- The roll ladder `inspection_roll_candidates_deg` defaults to `[0, 180, 90, 270]` and
  is a **fallback list whose first entry always wins** — so 90 sitting in it is never
  used. Adding a roll to that list does nothing. A commanded roll must be forced.
- A two-entry forced list (`[90, 0]`) would turn a refused roll into a silent roll-0
  capture. Forced rolls are `[θ, θ−180]` only, which are the same axis.
- The tilt ladder must never apply to a commanded roll (§3).

### 4.2 `path_completeness` is not a stable readout
Same pose, same stack, same chain: **0.253 to 0.891**. It tracks where the circle fit
lands (radius 37.7–44.2 vs nominal 42.0), not what the camera saw. Judge any lever on
**ROI coverage** and the **along/across-baseline ratio**, never on one take's
completeness. Full evidence in ring2-scan-handoff.md §3.5.

### 4.3 Cell and process
- **The KUKA must be in EXT/AUT, not T1**, and `RoboDKsync570` must be running, or the
  inspection move hangs *uncancellably* (`_wait_program` has no timeout and only checks
  cancel between polls). `/api/health` still answers `robodk: ok` — the real test is
  that **`/station-options` blocks**. Do not kill RoboDK to escape it: a dispatched
  program runs on the controller.
- **`build_stale`** must be `false` before trusting any cell result. Check the backend
  process start time against source mtimes; the app caches imported modules.
- The **branch guard aborts** (`branch guard exhausted`) are routine on a thin ring and
  still archive raw RGB-D — the coverage analysis survives them. Do not loosen it.
- Reading the robot via a second `robolink` client fails with *"Robot busy / not ready"*
  while the operator is at the pendant. Do not thrash it; ask.

---

## 5. The open question, and the prediction

Everything else is ruled out. From ring2-scan-handoff.md: the **pose** lever is dead
(`ScanAtRing2` is bit-identical to the automatic pose), and the **spatial filter** is
falsified (a proper A-B-A left the anisotropy at 2.2 and returned 4% *fewer* points).

What is real: layer 2 shows a reproducible anisotropy about the stereo baseline —
yield along it ≈ **2.3×** yield across it, dead sectors at 120–130°, 250–270°, 290–300°
— where the same stack's ring 1 reads 0.91–0.95 (flat). In the dead sectors the camera
still returns points, but at *substrate* height: the crest is unresolved, not missing.
Brightness does not explain it (the deadest sector is among the brightest).

It cannot be attributed from the archive, because the 2026-08-30 two-ring stack read
1.09 at the same baseline. **The roll is the only instrument that separates
camera-locked from scene-locked:**

> If **camera-locked**, the dead sectors move *with* the roll and the ratio stays ≈2.3
> about the NEW baseline. If **scene-locked**, they stay at the same work-frame angles
> and the ratio about the new baseline collapses toward 1.0.

Either answer is worth having: camera → fuse two rolled views, and multi-view stops
being speculative. Scene → stop blaming the sensor and go look at the deposition.
Note that **neither alone yields a valid layer-2 measurement** — the circle fit is also
unstable on a gappy ring (§4.2). Two faults; this only tells you which to attack first.

---

## 6. Reference

**Analysis tools** (all read-only on archives, no robot):
- `tools/probe_roll_readings.py` — `load_take`, `sector_counts`. Pin the sector centre
  to the session's **applied** centre; per-take `fit_ring` lands up to 12.9 mm off.
- `tools/camera_set.py` — read/set the Jetson filter chain at runtime. Bare = read-only.
- `tools/inspection_roll.py` — arm a forced roll globally (an alternative to the UI
  checkbox; needs a restart). Not required for the paired capture.

**Where the paired capture lives:**
- `modules/extrusion/measure.py` — `RingMeasureJob.rolls`, `RingCharacterizeJob.rolls`,
  `_roll_label`
- `modules/extrusion/service.py` — `_build_inspection_move(roll_deg=...)`,
  `wrist_allowance_deg`
- `modules/extrusion/inspection.py` — `pose_candidates(rolls=, tilts=, azimuths=)`
- `core/rdk_io.py` — `allow_wrist_flip`
- `webui/src/pages/Extrusion.tsx` — `pairedOrientations` (defaults **on**)
- `tests/test_paired_orientations.py` — 20 tests

**Trials on disk (2026-09-01):** `131341-2b12355c` the layer-1/layer-2 baseline;
`153855-8fe68421` the spatial A-B-A (16 takes); `165716-8c5ee676` the first paired
attempt (the tilted one).

**A characterization seeds the recipe**, so a paired characterization applies its
**vertical** view — applying the rolled one would derive radius/centre/height from the
orientation under test. Sessions predating paired capture carry no `orientation` and
keep the old last-one behaviour.
