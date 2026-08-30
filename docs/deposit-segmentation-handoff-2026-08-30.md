# Deposit segmentation: why the chroma gate is failing, and the universal replacement

Last updated: 2026-08-30 (evening cell session). Branch: `main`, HEAD `d48edf9`.
**Working tree is clean — nothing from this session was committed.**

This is the handoff for one question: *how should the extrusion pipeline tell
deposited bead from the surface it sits on, in a way that does not care what the
background is or what colour the bead is?* It exists because the current answer —
a saturation gate — has been measured to no longer work, and because the reason it
was introduced has been obsoleted by a hardware protocol change.

Everything below was measured offline from the archive with no robot time. The
reproduction commands are in the last section.

---

## 1. What happened on the cell tonight (trial `20260830-202416-293b208d`)

Sequence: characterize ring 1 → apply recipe → noise floor → re-place → ring 2.

| stage | takes | verdict |
|---|---|---|
| `characterize-01` | 1 | **valid**, completeness 0.992, r 40.38, bead 8.94, branch guard clean |
| `layer-001` (noise floor + re-place) | 8 | **all valid** |
| `layer-002` (ring 2) | 3 | **all invalid** |

Layer 1 is excellent data and should be kept:

| metric | spread across 8 takes |
|---|---|
| completeness | 0.992 – 0.993 |
| max angular gap | 2.6 – 2.7° (limit 30°) |
| measured radius | 40.94 – 41.09 mm (**0.15 mm**) |
| centre offset | 0.98 – 1.17 mm (**0.19 mm**) |

Layer 2 (all three takes): completeness 0.563 – 0.624, gap 135 – 157°, radius
~41–42 mm. Raw RGB-D and the 5-frame bursts are archived, so every one of them is
reprocessable — **do not re-run the robot to investigate these.**

Earlier trials that session are also useful evidence and are all archived:
- `20260830-190622-925d7178` — 5 takes, thin clay bead (5.7 mm) on the green mat,
  all invalid, completeness 0.31–0.45.
- `20260830-201254-e0e5c0b0` — the white-foam ring, characterization crashed.

---

## 2. The chroma gate: what it is, and the measurement that condemns it

### Why it exists

Added in `041ad1b` (2026-08-29) for a real, well-documented failure. Depth was
quantised at **1 mm**, so bare black ChArUco squares read 2.9–5.9 mm — one to three
LSB above the 2.5 mm deposit floor, facing straight up like any bead crest. No
height floor a real bead cleared could exclude them. A 22-point checker patch 12 mm
outside the ring dilated into 17–22 px skeleton arms against a 15 px spur limit and
exhausted the branch guard. Takes 1–3 carried the same patch and *passed*, with
radii biased 0.6–0.7 mm large — larger than the offsets the experiment exists to
resolve. **The crash was the good outcome; a loosened guard would have produced
silently wrong numbers.**

The gate worked because saturation separated bead from board ~20:1 (bead S 106–114,
board 5–25, nothing above 60). Result: 0/4 valid → 4/4 valid.

Its real function is to **earn a 1.5 mm deposit floor instead of 2.5 mm** — see
`deposit_floor_mm()` in `tasni/modules/extrusion/processing.py`. The two are
deliberately coupled: an abstaining gate restores the conservative floor.

### The premise has inverted

Measured on `layer-001/color.png` from tonight's run:

| region | when built (08-29) | tonight |
|---|---|---|
| clay bead | S 106–114 | **S median 25** |
| board black squares | S 5–25 | **S median 28** |
| board white paper | — | S median 7 |
| separation | ~20:1 | **none — the board is more saturated than the bead** |

`frac(S > 60)`: bead 0.206, black squares 0.115. Since the board covers far more
area than the ring, **the gate now admits more board than bead in absolute terms.**

This is visible directly in the stage renders (§5): what survives the gate is a
recognisable checkerboard.

Likely cause: exposure. HSV saturation is `(max − min) / max`, which is unstable at
low V and undefined at V = 0. These frames are dim (V median ~105–118). The bead's
saturation fell by a factor of four; the board's did not. **Check what changed in
the camera exposure/preset between 08-29 and now — that may be the cheapest fix of
all, and it should be ruled out before any code is written.**

### Its original justification is also obsolete

```
depth quantisation when the gate was written : 1.0 mm/word
depth quantisation now (protocol 2)          : 0.1 mm/word
```

Re-measured on tonight's frame, height above a locally-fitted substrate plane:

| region | p50 | p99 | σ |
|---|---|---|---|
| board black squares | 0.14 mm | 1.52 mm | 7.91 mm (heavy flying-pixel tail) |
| board white paper | 0.00 mm | 1.24 mm | 0.72 mm |
| clay bead | 0.63 mm | **10.0 mm** | 5.22 mm |

The black squares now sit *at* the plane, not 2.9–5.9 mm above it. **Height alone
separates board from bead today.** The argument that motivated the gate no longer
holds.

### But removing it naively does not work

Tested on four real takes, gate ON (floor 1.5) vs OFF (abstain, floor 2.5):

| take | gate ON | gate OFF |
|---|---|---|
| `layer-001` | 0.993 / 2.6° ✓ | 0.993 / 2.5° ✓ |
| `layer-001-take05` | 0.993 / 2.6° ✓ | **0.842 / 56.9° ✗** |
| `layer-002` | 0.624 / 136° ✗ | 0.616 / 138° ✗ |
| `layer-002-take03` | 0.563 / 157° ✗ | 0.588 / 148° ✗ |

The gate is no longer discriminating, but the 1.5 mm floor it unlocks is still
load-bearing. It is being kept alive by the downstream ROI crop throwing the board
away geometrically, not by doing its own job. **Do not simply delete it.**

---

## 3. The second defect: `floor_profile` is `None` in production

`test_repeat_takes_and_the_floor_from_the_previous_layer` asserts that layer 2
receives ring 1's measured top as its `floor_profile`. The cell recorded
`floor: {"source": "build_plane", ...}` on **both** layers, which per
`processing.py` means `floor_profile` was `None`. `session.json` had `tops` keys
`1` and `2`, so the data was present.

**This is unexplained and is the highest-value thing to chase.** Start here.

Two related observations:

- Ring 1's archived "measured top" spans z 1.50 – 10.86 mm (mean 4.91), and the
  minimum is *exactly* the deposit floor — board points are being recorded as ring
  crest. So even when the floor profile is wired up it is a poor reference surface.
- Feeding it as layer 2's floor makes things **worse**, not better: 0.624 → 0.541.
  Because `layer_floor_margin_mm` is added on top of that noisy surface and cuts
  into ring 2.

---

## 4. The proposed universal design

Principle: **discriminate on the invariant (geometry), not the accident
(appearance).** Height above the surface the bead sits on is physically true
regardless of pigment, background or lighting.

1. **Reference the surface, never absolute Z.** The board sits 0.49–0.59° off
   work-frame Z — ~3.4 mm across the field, larger than the detection threshold
   itself. Fit the substrate locally (robust plane, or low-order surface if the
   paper curls) and measure height above *that*.

2. **Better: capture the reference.** Scan the empty build surface at the same pose
   before depositing and difference against it. The checkerboard's depth bias,
   tilt, warp and print-induced noise are then in both frames and cancel exactly.
   Background-agnostic by construction. ChArUco detection already exists from the
   calibration module and can re-register the reference rigidly if the board shifts.

3. **Threshold from measured noise, not a constant.** Take σ from the substrate's
   own fit residuals in the same frame and cut at k·σ. White paper measured
   σ = 0.72 mm, so the existing 1.5 mm floor is ≈2σ — a defensible value that is
   currently hard-coded and therefore silently wrong at a different standoff or
   incidence.

4. **Shape and topology as the selection stage, not colour.** The bead is a
   connected curve of roughly known width. The 22-point checker patch that crashed
   the branch guard in August would be rejected by "must belong to a component
   longer than a few bead widths" with no colour at all.

5. **Active IR for material independence in the sensing itself.** The white foam
   failed partly because untextured white gives the stereo matcher nothing — two
   opposite dead arcs aligned with the baseline (§6). The D435i's IR projector is
   blind to visible colour and exists for this. Raising it, and reducing exposure so
   specular white stops clipping, addresses the half no software filter can.

6. **If colour is kept at all, make it self-calibrating and demoted.** Use a
   chromaticity space with illumination normalised out (CIELAB a\*b\*), learn the
   two populations from the frame itself (Otsu/GMM on the ROI), and use it only as
   a tie-breaker. Never a fixed threshold on HSV saturation.

### The unification worth noticing

`floor_profile` (layer N−1 as layer N's reference) and the chroma gate are **the
same problem solved twice, ad hoc**: *what surface is this bead sitting on?* One
returns `None` in production while its test asserts otherwise; the other's premise
has inverted. Reference-surface subtraction unifies them — layer 1's reference is
the empty plate, layer N's is layer N−1's measured surface, captured through the
same code path.

The change deletes two fragile mechanisms rather than adding a third.

**Not yet verified:** that a reference-subtraction pipeline actually recovers the
failing layer-2 takes. That is testable offline against the archive with no robot
time and should be done before committing to the rewrite.

---

## 5. Stage-by-stage evidence

Rendered from the real chain via `process_observation(..., stages=...)` — not a
re-implementation. Same light, viewpoint and scale across all six; only membership
changes. Images in `C:\Users\User\Desktop\tasni_stages\` (`ALL_STAGES.png` is the
labelled contact sheet).

| stage | points | kept |
|---|---|---|
| 0 every valid depth pixel (of 921,600; 4.7% invalid) | 438,711 | — |
| 1 after the chroma gate | 42,079 | 9.6% |
| 2 after height + radial ROI | 6,915 | 16% |
| 3 voxel + outliers + largest cluster | 2,410 | 35% |
| 4 after radial trim | 2,410 | 100% (no-op on this frame) |
| 5 after the upward-normal filter | **924** | **38%** |

What the pictures show that the numbers do not:

- **Stage 1 is a checkerboard.** The gate is holding the board's black squares in a
  clear checker pattern while the ring is barely distinguishable.
- **Stage 5 is where the ring stops being a surface.** Stages 2–4 show a solid
  closed band; stage 5 keeps only the crown, visibly thinner and breaking up. On
  this frame that is fine (0.993, valid). On a thinner or tilted ring it is the
  difference between 0.99 and layer 2's 0.56.

The sensor is not the problem: the ring is fully and densely captured in **every**
frame, including the layer-2 ones that failed.

---

## 6. Related findings from the same session

**White foam rings are unusable** (trial `20260830-201254-e0e5c0b0`). Two
independent failures: the chroma gate keeps **0.0%** of the ring (S ≈ 6 against a
threshold of 60; V = 254, clipped white), and even with the gate off the IR stereo
resolves only ~55% of the circumference — two dead arcs at 60–120° and 240–330°,
aligned with the horizontal baseline, because untextured white gives no horizontal
gradient. Use chromatic, textured clay.

**`assemble_arcs` diverges across three code paths** (introduced by `6e5b5ed`):

| path | value |
|---|---|
| live layer-1 measure (`measure.py:873`) | `True` |
| `reprocess_saved_layer` (`service.py:1197`) | `False` |
| take figure (`figures.py:830`) | `False` |
| characterize + its figure | `True` (consistent) |

So a layer-1 take that passes live can change its numbers — possibly flipping to
invalid — the moment anyone presses reprocess, and its method figure shows a
different segmentation than the metrics it is captioned with. This also means the
archived takes cannot be rescued with the reprocess button, which is the one lever
that avoids robot time.

**Per-take cycle is ~13 s**, of which ~8.7 s is the arm commuting:

| phase | 19:06 | 20:24 | Δ |
|---|---|---|---|
| move to pose | 2695 | 4509 | +1814 |
| settle | 1000 | 1000 | 0 |
| capture | 958 | 2425 | +1467 (the 5-frame fusion, expected) |
| processing | 836 | 924 | +88 |
| return home | 2532 | 4185 | +1653 |
| **cycle** | **8021** | **13042** | **+5021** |

The return leg is nothing but `move_j_joints(start_joints)`, and the inspection
joints are near-identical between runs — so the arm is starting from a farther-away
park pose. Jog it closer before pressing Run. Use `repeats` (N frames per trip)
rather than `excursions` to add samples cheaply. **The start pose is not archived
anywhere**, so this can only be inferred from the return timing; recording it would
make slow cycles self-explaining.

---

## 7. What NOT to do

- **Do not lower `upwards_normal_z`** from 0.92 as a quick fix. It lifts layer 2
  from 0.56–0.62 to 0.85–0.99, but **breaks 4 tests** — the branch guard exhausts
  because the fatter mask grows more skeleton branches. This is the hazard the
  run-day note warns about.
- **Do not loosen the branch guard or `minimum_angular_coverage`.** `041ad1b`
  documents that a loosened guard converts a loud crash into a silently wrong
  number, and the coverage figure is what the paper reports.
- **Do not delete the chroma gate outright** — the 1.5 mm floor it unlocks is still
  load-bearing (§2).
- **Do not re-run the robot** to investigate layer 2. Every failing take is
  archived with raw RGB-D and its 5-frame burst.

## 8. Suggested order of work

1. Check the camera exposure/preset delta between 08-29 and now. Cheapest possible
   explanation for the saturation collapse; rule it out first.
2. Fix the `floor_profile` production/test divergence (§3). Unexplained, and it
   sits underneath everything else.
3. Fix ring 1's "measured top" so it is a crest, not crest-plus-board.
4. Prototype reference-surface subtraction offline against the archived takes and
   check whether it recovers layer 2 before touching the live chain.
5. Propagate `assemble_arcs` to `reprocess_saved_layer` and the take figure, and
   update the test assertion that pins the old layer-2 behaviour.

## 9. Reproducing the evidence

All read-only on `runs/extrusion/`, no robot, no backend.

```
# stage renders + contact sheet
py -3.10 <scratch>/stages.py

# chroma gate on vs off, real chain
py -3.10 <scratch>/gate.py

# arc assembly / normal-threshold sweeps
py -3.10 <scratch>/l2.py
py -3.10 <scratch>/sat_sweep.py
```

The scratch scripts from this session are throwaway; the archive is the durable
artifact. Key paths:

- `runs/extrusion/20260830-202416-293b208d/` — tonight's trial (8 good layer-1
  takes, 3 failed layer-2 takes, characterization)
- `runs/extrusion/20260830-201254-e0e5c0b0/characterize-01/` — the white-foam failure
- `runs/extrusion/20260830-190622-925d7178/` — thin bead on the green mat

Each take carries `color.png`, `depth.npy` (5-frame fused), `depth-frames.npy` (the
raw burst), and a manifest with intrinsics, `T_work_camera`, camera geometry and the
full processing config — enough to reprocess exactly.
