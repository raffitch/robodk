# Roll probe: is what we are missing locked to the camera, or to the scene?

Task page. Opened 2026-09-01; corrected after review the same day. The roll sequence
uses one unchanged ring/stack and **three measurement excursions (A-B-A)**; the preceding
spatial-filter A/B adds its own control captures. Existing code can capture the evidence,
but the layer-2/dropout and residual-correlation readings still need a small offline probe
extension (§4).

This is a **re-run**. The probe exists (`tools/probe_roll_pair.py`), a pair is already
on disk, and the probe **refused to read it** — see "What happened last time". Nothing
below is a first attempt; the job is to get a *controlled* pair.

---

## 1. Why this is worth an excursion

Three separate problems on this cell could all have one cause, and one manipulation
separates that cause from the alternatives:

| symptom | measured | camera-locked? | scene-locked? |
|---|---|---|---|
| layer-2 dropout | ~50° sector, 22–121 ROI pts per 10° vs 250–466 elsewhere | stereo shadow → **rolls with the camera** | self-shadowing → **stays put** |
| board halo | shelf 1.7–2.2 mm over the plane at work ~200° and ~353° | stereo edge artifact → **rolls** | something on the board → **stays** |
| depth noise | **static 1.176 mm vs temporal 0.130 mm** | sensor pattern → **decorrelates with pose** | surface texture → **does not** |

A **roll** — rotating the camera about its own optical axis — is the closest thing to a
single-variable move that separates these. The viewpoint does not change, so true
geometric self-shadowing and real board features are untouched; but the stereo baseline
direction and the sensor's own fixed noise pattern rotate with the camera.

It is not *perfectly* single-variable, and that is what killed the first attempt: the
**IR projector rolls with the camera too**, and turning its dot pattern against the
printed board moved the noise floor itself (§2). The protocol in §3 is designed around
that confound — not merely around operator care.

The third row is the one that matters most. The 1.176 / 0.130 mm numbers came from the
2026-08-30 headroom scene at ~447 mm, not from this two-ring stack at 300 mm. A direct
2026-09-01 replay of the latest layer-2 burst still found the same qualitative result:
robust spatial substrate sigma 0.608 → 0.569 → 0.549 → 0.506 → 0.496 mm over its five
frames, against 0.126 mm mean per-pixel temporal sigma on 246,917 common board pixels.
Static structure dominates, but the monotonic change also says the production burst was
**still settling**. More frames at one settled pose buy little against static structure;
the current first five are not proof of what a settled burst buys.

Static noise can average down *if it decorrelates with pose* — and the entire case for
multi-view rests on that assumption, which has never been tested. The captures below make
that test possible, but the existing halo-only pair statistic does not perform it (§4).

---

## 2. What happened last time (read this before planning the run)

A pair was captured on 2026-08-31 for the halo question:

- **A** — `runs/extrusion/20260831-163156-5bf38c80/characterize-01`, baseline 179.1°
- **B** — `runs/extrusion/20260831-171203-24d21bab/characterize-01`, baseline 119.1°

```
py -3.10 tools/probe_roll_pair.py \
  runs/extrusion/20260831-163156-5bf38c80/characterize-01 \
  runs/extrusion/20260831-171203-24d21bab/characterize-01
```

```
VERDICT: NOT A CONTROLLED COMPARISON. The two captures do not share a noise floor,
so neither 'baseline-locked' nor 'real geometry' can be read off them.
```

**Three facts to carry into the re-run:**

1. **90° was requested; 60° was delivered.** The probe's own protocol says to set
   `inspection_roll_candidates_deg [90, 60, 0]`; the planner could not reach 90 and
   fell through to 60. (`max_tool_axis_spin_deg` defaults to 90, so a 90° roll sits
   exactly on the limit.) 60° is *plenty* — it rotates the baseline 60° — but you must
   **read the achieved roll off `T_work_camera`, never off the config.** The
   characterize records do not carry `roll_deg` at all: `report.json`'s
   `inspection_pose.roll_deg` came back `None` for capture B. No hand analysis is
   needed — the probe itself prints each capture's baseline axis in the work frame and
   the pair's gap ("the two cameras' baselines differ by … deg"), all read off
   `provenance.T_work_camera`. Run the probe on the pair the moment capture B lands;
   that line IS the achieved-roll check.

2. **Capture B was a worse capture; the roll is a candidate cause, not an isolated
   cause.** Substrate
   sigma 0.866 mm against A's 0.591 (ratio 1.46, the gate is 25%), the derived floor
   **pinned at its 2.0 mm clamp ceiling**, the skirt pattern gone from two clean lobes
   to four at three times the amplitude, and a fitted radius of 44.10 mm where A gave
   42.64 on the same untouched ring. Rolling the IR projector's dot pattern against the
   printed board is a plausible cause, but the pair also changed recipe and absolute
   viewpoint (fact 3). The refused pair cannot assign causality to any one of them.

3. **The old pair was not the same viewpoint.** Capture A's trial used bead diameter
   8.9 mm; B used 15.0 mm. Both inspection records said 300 mm standoff, but standoff is
   measured from the recipe-derived aim height. `T_work_camera` therefore placed the
   optical centres at work Z 313.159 and 319.259 mm — **6.100 mm apart** — while their
   optical axes agreed. That exactly follows the 6.1 mm recipe change. Freeze the whole
   applied plan/recipe/toolpath fingerprint in the re-run, not merely the standoff field.

The raw axis numbers the probe printed (work-frame disagreement 19.8°,
baseline-relative 79.8°) **are not an answer** and must not be quoted. The probe prints
them under an explicit refusal for exactly this reason.

---

## 3. Protocol

### 3.1 First decision gate: test the spatial filter

Before spending the roll excursions, run the already-built **current/default versus
`RS_SPATIAL=0`** A/B on the same untouched two-ring stack at roll 0. This is the only
untried lever with a measured direct mechanism for losing the ring: the camera read the
crest ~1.5 mm low against the operator's five ruler readings, and the stock disparity-domain
spatial filter uses `smooth_delta=20` while the 4–11 mm relief spans only ~1.4–3.9 disparity
pixels at 300 mm (`server/server_unicast_syncronous.py`, commit `f9c4a53`).

- If spatial-off materially restores the low crest/dropout without turning board noise
  into a false ring, stop and tune that path before asking multi-view to repair it.
- If it does not restore the sector, restore the default chain and continue below.
- The greeting archives whether `spatial` was present, so on/off is distinguishable.
  It does **not** archive `RS_SPATIAL_SMOOTH_DELTA`; do not run a delta sweep until that
  value is added to provenance (or each arm is otherwise immutably labelled).

### 3.2 Controlled roll capture

Place one ring — or leave the current 2-ring stack in place if the dropout is the
target — and **do not touch it for the whole protocol.**

**Use A-B-A: A1 at +30°, B at −30°, A2 at +30°.** Do not repeat the
0-vs-60 design that failed. Symmetric rolls still rotate the baseline 60° between the
two orientations — the discriminating variable is intact — while A1 versus A2 measures
ordinary drift/repeatability. Calling +30 and −30 the same *class* of projector interaction
does not make them equal; only the returning A2 control can show whether the comparison
was stable. The roll-0 takes already on disk
(`runs/extrusion/20260831-195459-19838507/layer-002*`, tilt 0 / azimuth 0 / standoff
300 mm) remain the work-frame reference for where the dropout sector sits; they are
**not** capture A of the pair.

Before A1, set `measure_depth_fusion_frames` to **10** for this diagnostic only. The
archive retains all ten raw frames; analyze the **last five** after the filter chain has
advanced, rather than treating the visibly converging first five as settled. Apply the
same setting to every arm and restore it afterwards.

1. **Capture A1 (+30°).** In `tasni.config.json`'s `extrusion` block set
   `inspection_roll_candidates_deg` to `[30, 0]`, **restart the backend**, confirm
   `GET /api/health` → `build.stale == false`, run the measurement.
2. **Capture B (−30°).** Set `[-30, 0]` (candidates are plain floats tried in list
   order at tilt 0 first — negative is legal), restart, confirm, run with the same
   `repeats`. Do **not** ask for 90° anywhere: it sits exactly on
   `max_tool_axis_spin_deg`'s default limit of 90 and was refused last time. Leave that
   limit alone and keep every candidate comfortably inside it.
3. **Capture A2 (+30° again).** Restore `[30, 0]`, restart, confirm, and repeat without
   touching the stack or changing the applied plan.
4. **Verify achieved rolls immediately.** Run the existing pair probe twice: A1-vs-B
   and A2-vs-B. Each should report a baseline gap near 60°; A1 and A2 should return to
   the same baseline. Under 15° apart from B is inconclusive and means a rolled pose
   probably fell through to 0. Do not read any refused verdict.
5. **Restore both startup overrides** (`inspection_roll_candidates_deg` and
   `measure_depth_fusion_frames`) afterwards. A forgotten value silently changes every
   later take.

**Hold everything else equal anyway.** Same applied plan, recipe, toolpath fingerprint,
work frame, search centre, absolute camera position, standoff, same tilt (0 — a 15° tilt costs
3–4 mm of plane noise against 0.65 mm at zero), same ambient light, same laser power
(pinned to 150 in the systemd unit; confirm it did not move). After A2, compare
`report["substrate"]["sigma_mm"]` between all three captures.
**Within 25% and every floor off the 2.0 mm clamp, or the comparison is void** — the
existing probe enforces this per pair. Also compare substrate p99/inlier fraction, valid
depth fraction, `T_work_camera` translation/optical axis, temperatures, depth unit,
filter list, preset and laser. Equal global sigma alone cannot prove that two angular
noise patterns are comparable. Depth exposure/gain are not currently archived, which is
a disclosed residual confound.

Most importantly, **A1 and A2 must agree with one another**. If the returning control
does not reproduce the first +30° capture, drift/repeatability is already as large as the
effect being assigned to roll and B cannot be interpreted.

**If the A-B-A set still fails the comparability gates, stop — do not iterate on the cell.**
That outcome is itself a finding: the noise floor depends on roll angle on this board,
and no roll pair can answer the question here. Bring it back to the desk. The remaining
levers are not more excursions of the same kind: an unprinted matte board under the same
ring (removes the printed-pattern interaction, though it also changes the very surface
the halo lives on), or accepting the tilted-star offline A/B (`tools/multiview_ab.py`,
§5) as the discriminating experiment despite its higher noise cost.

---

## 4. Reading it

`tools/probe_roll_pair.py` implements the decision rule for the **halo** (it measures
lifted board points in the skirt annulus). It also fits each capture's ring centre/radius
independently, so a biased fit moves the annulus being compared. A controlled reader must
use one shared centre/radius (or at least report sensitivity to that choice). Its frame rule
generalises only to the quantity actually measured:

- **baseline-relative peaks agree, work-frame peaks rotate → CAMERA-LOCKED.** Two rolls
  close the gap at zero tilt cost. Strictly, the probe measures one specific static
  artifact — the lifted-board axis — not per-pixel noise decorrelation and not the ring
  dropout. This verdict says the **halo** is camera-locked; it cannot prove that all static
  noise averages down.
- **work-frame peaks agree, baseline-relative peaks rotate → SCENE-LOCKED.** Extra views
  of the same kind will not help; the tilted star (a genuinely different viewpoint) is
  the only thing that can remove that **particular scene-locked quantity**. It does not
  prove that the separate static residual field is scene-locked too.

**For the layer-2 dropout the measurement is different and the probe does not do it
yet.** The dropout is an absence, not a lifted shelf: count ROI points per 10° sector
and find where they collapse, then express that sector twice — in work-frame angle and
relative to that capture's own baseline axis (`T_work_camera[:3, 0]`). Exactly one of the
two should agree between A and B. Same rule, different quantity; extend the probe or
write a sibling, and delete it once the answer is in.

**For the static-noise question, neither angular statistic is enough.** Using the archived
raw bursts, rasterise the surrounding substrate into the work frame, remove a fitted plane
and common low-frequency shape, then compare:

- within-pose residual correlation (A1 frames with A1, B with B, A2 with A2);
- A1-vs-A2 correlation in work coordinates — the repeatability/drift control;
- cross-roll correlation in work coordinates; and
- cross-roll correlation after expressing the residual maps in each camera's own
  baseline coordinates.

That is the direct test of whether the 0.5–0.6 mm static residual decorrelates. The halo
axis can support it, not substitute for it. Capturing A-B-A can happen before this sibling
exists because `depth-frames.npy` preserves the evidence; **declaring the result cannot**.

Baseline for the current stack, so a re-run is comparable — 2026-08-31 layer-002,
reprocessed offline with the deposit-floor fix (`bd455a7`, which landed at 21:56 — two
hours **after** the 19:54 capture). **The archived `report.json` files still hold the
pre-fix output** — completeness 0.294 / 0.272 / 0.363, gaps 229–262°, radii
40.6–47.9 mm — so do not open the archive, see numbers wildly unlike this table, and
conclude the ring moved. (`tests/test_extrusion_golden.py`'s comments record the same
reprocess: 0.294 → 0.515, radius spread 7.24 → 0.29 mm.) To regenerate the table, run
the chain read-only the way the golden harness does (`processing.measure_take`);
pressing reprocess in the app overwrites `report.json` in place — tolerable for this
archive, banned for 2026-08-30 (§6).

| | take 1 | take 2 | take 3 |
|---|---|---|---|
| completeness | 0.515 | 0.517 | 0.509 |
| max angular gap | 174.6° | 174.0° | 176.6° |
| fitted radius | 43.29 mm | 43.40 mm | 43.58 mm |
| centre offset | 2.52 mm | 2.43 mm | 2.94 mm |

Two numbers describe the dropout and they are **not the same quantity** — do not let
them collide. The **140–190°** figure (≈50° wide, §1's table) is where per-10°-sector
ROI *counts* collapse (22–121 points against 250–466 elsewhere). The **~175° max
angular gap / ~0.51 completeness** above is a *measured-path* metric: about half the
path is missing — presumably a wider sparse arc around that severe core, though the
reconciliation has not been checked. The sibling probe counts raw ROI points per
sector, so 140–190° on the roll-0 takes is the number it must reproduce first;
explaining the wider path gap is part of reading the result, not a discrepancy to be
alarmed by.

---

## 5. The multi-view branch nobody ever used

**If the answer comes back CAMERA-LOCKED, do not start building from scratch.** Most of
the registration/archive machinery exists — but the branch's shipped pose set is tilted,
while a baseline-locked dropout first calls for cheaper **zero-tilt roll fusion**. Reuse
the machinery; do not assume the Mercedes-star pose set is the answer.

`origin/worktree-multiview-inspection` @ **`96a17f6`** — worktree already checked out at
`.claude/worktrees/multiview-inspection`. **17 commits, +3659 lines**, never merged, never
run on the cell. It is not a stub:

| | |
|---|---|
| `tasni/modules/extrusion/multiview.py` | 311 lines — level, register, merge |
| `tests/test_extrusion_multiview.py` | 1356 lines |
| `tools/multiview_ab.py` | offline A/B: reprocess a star take merged vs top-only, **no robot time** |
| `docs/extrusion-current-handoff.md` (branch copy) | the cell A/B protocol and its decision rule |
| spec / plan | `docs/superpowers/specs/2026-08-30-multiview-inspection-design.md`, `docs/superpowers/plans/2026-08-30-multiview-inspection.md` |

Design: top view plus three **15°**-tilted views at 120° azimuths (a "Mercedes star"),
each levelled against the surrounding annulus, all four registered against **one shared
circle** — gauge-fixed joint solve, no ICP, no ChArUco (banned here by operator decision;
it belongs to hand-eye only and the board is not always under the rings). Opt-in,
default OFF, and `multiview=False` provably reduces to today's exact single-view path.

**Four things about it that are now stale or worth knowing:**

- It is **45 commits behind main** and predates everything from 2026-08-31: the halved
  voxel (`a4015e1`), the radial-trim fixed point (`50d0b34`), and the layer-N deposit
  floor (`bd455a7`). Rebase before trusting any number in it.
- Its "**count trap**" note says the chain voxel-downsamples at **1 mm**. Main is now
  **0.5 mm**. That note is wrong as written and the merged-cloud arithmetic behind it
  needs redoing.
- It carries a fix main still lacks: `depth_plane_check` computing incidence from the
  actual pose (`cos = -T[2,2]`) instead of assuming a straight-down view. On main that
  gate still rejects tilted frames above roughly 18° at 300 mm standoff — **any tilted
  capture on main will be refused by it.** Worth cherry-picking regardless of what this
  probe says; tilt 0 reduces to the old check exactly.
- Its protocol opens with "do not start before the PFH paper's single-view cell run is
  finished", because multi-view redefines `acquisition_to_path_ms`. The operator has
  since said the paper timeline is **not** a sequencing constraint (2026-08-31) — but
  the `acquisition_to_path_ms` hazard is real on its own terms and still applies.

Known gap, disclosed on the branch: `RingCharacterizeJob` accepts `multiview` for API
symmetry but **does not act on it** — characterize's capture path never reaches the seam
the star merges at. The UI withholds the toggle there for that reason.

---

## 6. Traps

- **The backend hard-crashes.** Seven times over 2026-08-30/31, native heap corruption,
  cause unknown. More excursions is more exposure. If it dies mid-run, read
  `%TEMP%\tasni-backend.crash.log` **before restarting** — it names the thread and the
  call, and it is the only thing that does.
- **Restart the backend after any config edit** and check `build.stale == false`. The app
  caches imported modules; otherwise you will test stale code and not know it.
- **Do not reprocess the 2026-08-30 archive.** `tests/test_extrusion_golden.py` asserts
  its `report.json` still holds the original 2026-08-30 values as a
  "has anyone reprocessed this?" cross-check. Reprocessing it breaks that permanently.
- **Do not read a refused verdict.** The probe prints its axis numbers even when it
  refuses the pair. They are diagnostics, not results.
- **Robot settling is not filter settling.** `settle_s` elapses before the host opens
  the depth stream; the Jetson's spatial/temporal blocks advance only when `getFrames()`
  is called. The latest five-frame burst was still converging, which is why this protocol
  archives ten and reads the last five.
- **A 300 mm standoff is relative to the aim point, not an absolute camera-Z invariant.**
  Changing bead diameter moved the old pair's camera by 6.1 mm. Compare
  `T_work_camera`, and freeze the complete applied plan.
