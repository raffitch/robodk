# Roll probe: is what we are missing locked to the camera, or to the scene?

Task page. Opened 2026-09-01; corrected after review the same day. The roll sequence
uses one unchanged ring/stack and **three measurement captures (A-B-A)** — three config
edits and backend restarts, which can all happen in one cell session, not three trips;
the preceding spatial-filter A/B adds its own control captures. Both offline readings
now exist (`tools/probe_roll_readings.py`, §4), so no code is needed before capturing.

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

A second 2026-09-01 replay, by a different estimator, puts a much sharper edge on the
settling half of that. Correlating each burst's **residual field** (the fitted plane
removed, rasterised into polar cells around the ring) frame-to-frame:

| take | within-pose, frame 0 vs frame 4 | vs the other takes, work frame |
|---|---|---|
| layer-002 (first burst at the pose) | **+0.045** | +0.053 / +0.052 |
| layer-002-take02 | **+0.937** | +0.960 vs take03 |
| layer-002-take03 | **+0.937** | +0.960 vs take02 |

Two settled bursts of the same ring reproduce each other's static field at **+0.96** —
the structure is extremely repeatable, and the measurement is sensitive enough to see it
go away. The first burst at a fresh pose does not even reproduce **itself** (+0.045), and
is uncorrelated with both settled takes. So the unsettled burst is not merely noisier;
its residual field is *transient filter state*, carrying almost none of the static
structure. Nothing observed lands between those two populations.

(The two replays use different estimators and regions and their absolute numbers differ —
the sigma series above is a robust spatial sigma over the fit disc, the correlations here
are over a polar annulus outside the bead. Both say the same thing about settling; quote
whichever, but do not mix them into one series.)

Static noise can average down *if it decorrelates with pose* — and the entire case for
multi-view rests on that assumption, which has never been tested. The captures below make
that test possible, and `tools/probe_roll_readings.py` (§4) now performs the test; the
existing halo-only pair statistic still does not.

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
  Since 2026-09-01 it also archives the delta the filter ACTUALLY ran at, as
  `provenance.camera_geometry.filter_options.spatial_smooth_delta` (read back off the
  filter, so an untouched chain records 20, not the `-1` env default; `null` = no spatial
  filter). A delta sweep is therefore readable afterwards — **but only from takes captured
  by a Jetson running that server**: every take already on disk has no `filter_options`
  key at all, and the host reports those as unknown rather than as 20. Deploy the server
  before the sweep, and check the first take's manifest actually carries the field.

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

Two things to know about that setting. First, the *production* median still fuses all
ten, unsettled frames included — the comparability gate and the reported metrics are
therefore computed over a partly-unsettled burst. That is the same in all three arms so
the comparison survives, but do not quote a take's own `sigma_mm` as if it described the
settled camera. Second, `probe_roll_readings.py` enforces settling directly: it
correlates each burst's first and last frame and **refuses any capture below +0.5**
(measured populations: +0.937 settled, +0.045 unsettled — nothing in between). If you
would rather not spend ten frames per pose, the cheap alternative is a throwaway warm-up
capture at each pose that you discard; the two repeat takes on disk show a burst is
already settled by the *second* one at the same pose.

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
4. **Verify achieved rolls immediately** — `py -3.10 tools/probe_roll_readings.py <A1>
   <B> <A2>`. It prints the baseline separation A1–B (should be near 60°) and the
   baseline return A1–A2 (should be near 0°), and refuses the set if either the noise
   floors disagree, a burst never settled, or the rolls came out under 15° apart —
   which means a rolled pose fell through to 0. Do not read any refused verdict.
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

Two readers, one per question.

**`tools/probe_roll_pair.py` — the halo.** Lifted board points in the skirt annulus.
The independent-fit defect the review found is **fixed**: it now takes one shared
centre/radius from capture A and prints how far each capture's own fit sits from it.
That number is worth reading on its own — on the refused 2026-08-31 pair the two fits
were **3.90 mm apart on the same untouched ring**, so the two arms had been compared
through annuli 3.9 mm out of register. Its frame rule generalises only to the quantity
it actually measures:

- **baseline-relative peaks agree, work-frame peaks rotate → CAMERA-LOCKED.** Two rolls
  close the gap at zero tilt cost. Strictly, the probe measures one specific static
  artifact — the lifted-board axis — not per-pixel noise decorrelation and not the ring
  dropout. This verdict says the **halo** is camera-locked; it cannot prove that all static
  noise averages down.
- **work-frame peaks agree, baseline-relative peaks rotate → SCENE-LOCKED.** Extra views
  of the same kind will not help; the tilted star (a genuinely different viewpoint) is
  the only thing that can remove that **particular scene-locked quantity**. It does not
  prove that the separate static residual field is scene-locked too.

**`tools/probe_roll_readings.py` — the dropout and the static field.** Run it on the
whole set at once:

```
py -3.10 tools/probe_roll_readings.py <A1> <B> <A2>
```

It gates before it reads (comparability, then settling, then whether the rolls actually
separated), uses one shared centre/radius for every capture, and reports two independent
quantities:

- **The dropout**, as ROI points per 10° sector in a band around the ring. It does **not**
  report "where the dropout is": measured against the real archive, the layer-2 deficit is
  multi-lobed — a deep collapse at 140–190° (counts 6, 13, 65, 80 per 10°) plus lesser
  lows at 30°, 240°, 260° and 330° — so any single "dropout axis" averages them into a
  number pointing at none of them (it returns 249°). The verdict instead comes from
  **registering the two whole profiles**: a scene-locked dropout registers near 0°, a
  camera-locked one near the change in baseline angle. That is robust to the extra lobes,
  and it was always the real question — not where the dropout is, only whether it moved
  with the camera.
- **The static residual field**, rasterised into polar cells on the board annulus, plane
  removed, radially-symmetric component removed (it carries no angular information, so it
  can only inflate a correlation toward 1). Low angular frequency is deliberately **kept**:
  the stereo artifact being hunted is itself a 2-cycle pattern, so an angular high-pass
  would delete the signal along with the nuisance. It reports the within-pose floor, the
  A1-vs-A2 repeatability ceiling, and the cross-roll correlation in work versus baseline
  coordinates — the direct test of whether the 0.5–0.6 mm static residual decorrelates.

Sanity anchors from the archive, so a re-run has something to land against: two settled
same-roll takes correlate at **+0.960**, and no cross-roll number can legitimately beat
that ceiling. The probe says so if one does.

Both readings are one quantity each. Agreement between them is the result; a split verdict
means neither is safe to build on. Delete the probe and its tests together once the answer
is in.

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
ROI *counts* collapse. The **~175° max angular gap / ~0.51 completeness** above is a
*measured-path* metric: about half the path is missing, a wider sparse arc around that
severe core.

The count figure is now confirmed directly off the archive (`probe_roll_readings.py` on
the three roll-0 takes, one shared centre, ring band r 31.8–51.8 mm). Per 10° sector,
take 1: the collapse runs **80, 6, 65, 122, 13** across 140–190° against a maximum of
455 elsewhere — §1's "22–121 against 250–466" holds. What §1 does *not* say, and the
sibling reading had to learn, is that the deficit is **multi-lobed**: further lows sit at
30° (28), 240° (69), 260° (51) and 330° (54). That is why the verdict registers whole
profiles instead of locating an axis (§4). The three same-roll takes reproduce each
other's profile to within 2.4–6.5°, which is the drift floor any cross-roll shift must
beat.

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
