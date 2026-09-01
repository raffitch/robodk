# Roll probe: is what we are missing locked to the camera, or to the scene?

Task page. Opened 2026-09-01. One robot excursion, one ring, no new code required.

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

A **roll** — rotating the camera about its own optical axis — is the only single-variable
move that separates these. The viewpoint does not change, so true geometric
self-shadowing and real board features are untouched; but the stereo baseline direction
and the sensor's own fixed noise pattern rotate with the camera.

The third row is the one that matters most. Your noise is **static, not temporal**,
which is why more frames per pose buy nothing. Static noise still averages down *if it
decorrelates with pose* — and the entire case for multi-view rests on that assumption,
which has never been tested. A roll pair tests it for one excursion.

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

**Two facts to carry into the re-run:**

1. **90° was requested; 60° was delivered.** The probe's own protocol says to set
   `inspection_roll_candidates_deg [90, 60, 0]`; the planner could not reach 90 and
   fell through to 60. (`max_tool_axis_spin_deg` defaults to 90, so a 90° roll sits
   exactly on the limit.) 60° is *plenty* — it rotates the baseline 60° — but you must
   **read the achieved roll off `T_work_camera`, never off the config.** The
   characterize records do not carry `roll_deg` at all: `report.json`'s
   `inspection_pose.roll_deg` came back `None` for capture B. Recover it with
   `np.asarray(report["provenance"]["T_work_camera"])[:3, 0]` — the camera's +X in work
   coordinates — and compare its angle against capture A's.

2. **Capture B was a worse capture, and that is what killed it.** Substrate sigma
   0.866 mm against A's 0.591 (ratio 1.46, the gate is 25%) and the derived floor
   **pinned at its 2.0 mm clamp ceiling**. Its fitted radius came back 44.10 mm where A
   gave 42.64 on the same untouched ring — a 1.5 mm disagreement that is itself the
   tell. Do not re-run until you have a plan for holding the noise floor equal.

The raw axis numbers the probe printed (work-frame disagreement 19.8°,
baseline-relative 79.8°) **are not an answer** and must not be quoted. The probe prints
them under an explicit refusal for exactly this reason.

---

## 3. Protocol

Place one ring — or leave the current 2-ring stack in place if the dropout is the
target — and **do not touch it for the whole protocol.**

1. **Capture A (roll 0).** Baseline already exists for the stack:
   `runs/extrusion/20260831-195459-19838507/layer-002`, takes 1–3, tilt 0 / azimuth 0 /
   **roll 0** / standoff 300 mm. Re-capture only if the ring has moved since.
2. **Capture B (rolled).** In `tasni.config.json`'s `extrusion` block set
   `inspection_roll_candidates_deg` to `[90, 60, 0]`, **restart the backend**, confirm
   `GET /api/health` → `build.stale == false`, then run the same measurement with the
   same `repeats`.
3. **Verify the achieved roll immediately** (§2.1). If it came out 0, the run is a
   no-op — stop and fix the reachability before spending more cell time.
4. **Restore `inspection_roll_candidates_deg`** afterwards. It is a startup value; a
   forgotten override silently changes every later take.

**Hold the noise floor equal.** This is the whole difficulty, and it is what failed last
time. Same standoff, same tilt (0 — a 15° tilt costs 3–4 mm of plane noise against
0.65 mm at zero), same ambient light, same laser power (pinned to 150 in the systemd
unit; confirm it did not move). After capture B, before anything else, compare
`report["substrate"]["sigma_mm"]` against A's. **Within 25% and off the 2.0 mm clamp, or
the pair is void** — the probe will refuse it and it will have cost an excursion for
nothing.

---

## 4. Reading it

`tools/probe_roll_pair.py` implements the decision rule for the **halo** (it measures
lifted board points in the skirt annulus). Its rule generalises:

- **baseline-relative peaks agree, work-frame peaks rotate → CAMERA-LOCKED.** Two rolls
  close the gap at zero tilt cost, the static noise decorrelates with pose, and the
  multi-view case is confirmed on evidence rather than assumption.
- **work-frame peaks agree, baseline-relative peaks rotate → SCENE-LOCKED.** Extra views
  of the same kind will not help; the tilted star (a genuinely different viewpoint) is
  the only thing that can, and the static noise will *not* average down.

**For the layer-2 dropout the measurement is different and the probe does not do it
yet.** The dropout is an absence, not a lifted shelf: count ROI points per 10° sector
and find where they collapse, then express that sector twice — in work-frame angle and
relative to that capture's own baseline axis (`T_work_camera[:3, 0]`). Exactly one of the
two should agree between A and B. Same rule, different quantity; extend the probe or
write a sibling, and delete it once the answer is in.

Baseline for the current stack, so a re-run is comparable — 2026-08-31 layer-002,
after the deposit-floor fix (`bd455a7`):

| | take 1 | take 2 | take 3 |
|---|---|---|---|
| completeness | 0.515 | 0.517 | 0.509 |
| max angular gap | 174.6° | 174.0° | 176.6° |
| fitted radius | 43.29 mm | 43.40 mm | 43.58 mm |
| centre offset | 2.52 mm | 2.43 mm | 2.94 mm |

The missing sector is at work-frame **140–190°**.

---

## 5. The multi-view branch nobody ever used

**If the answer comes back CAMERA-LOCKED, do not start building.** Most of it exists.

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
