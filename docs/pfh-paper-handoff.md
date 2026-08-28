# PFH paper — handoff (ring-stack measure-only evidence)

**Deadline: 1 September 2026.** Written 2026-08-28. This is the single page for the
paper task: what the paper is still missing, what already exists, and the exact order
to get the rest. Everything else is background:

- Design of the experiment: `docs/superpowers/specs/2026-08-27-ring-stack-measure-only-design.md`
- Module state and traps: `docs/extrusion-current-handoff.md`
- Cell-run status: `AGENTS.md`

## 1. What the paper needs

Prototypes for Humanity 2026 short paper, *"Real-time Error Mapping, Correction, and
Archival …"* (Tchakerian / Hayek / Daneluzzo). The co-author package
(`C:\Users\User\Desktop\desktop\RoboArch_to_PFH_Coauthor_Package\`) says the text is
written and compliant at 1,887 / 2,000 words and is waiting on numbers, under the
authors' rule: **do not invent or estimate values that were not measured.**

| # | Number | Status | Comes from |
|---|---|---|---|
| 1 | Count of recipes / trials in the legacy Firebase archive | **out of scope** — not in this repo | — |
| 2 | Geometric deviation nominal ↔ extracted centreline: mean abs / RMS / max | ❌ **needs the cell run** | `paper-summary` deviation columns |
| 2b | **Detection error**: how well a known introduced offset is recovered | ❌ **needs the cell run** | `paper-summary` *detection error* column |
| 3 | Scan-to-feedback time, mean ± sd over ~10 runs | ⚠️ one sample (3,124 ms); needs ≥ 10 | `acquisition_to_path_ms` |
| — | Path-extraction success X/N | ⚠️ 1/1 so far | `valid` count |
| — | Layer height / bead width variation | ✅ **have it** (6.1 mm mean, 2.9–11.0; bead 9.9 mm) | `geometry` in the manifest |

### The wording constraint — do not lose this

The rings are **hand-placed dried beads**, not a print. Nothing is extruded, no valve
switches, no material recipe is recorded. So the paper can claim a **controlled
validation of the sensing-and-comparison chain against a known introduced offset** —
**not** the deposition deviation of a printed cylinder. `paper_summary()` already
phrases it that way and the figure captions carry it too; keep it.

### The error floor — do not claim below it

Hand-eye calibration verdict is `borderline`, board consistency **1.26 mm**, work-plane
RMS 1.39 mm. Do not report sub-millimetre accuracy on top of that.

## 2. What already exists

**A first real capture succeeded 2026-08-28**: `runs/extrusion/20260828-192115-47fb78ea`
(300 mm standoff, collisions OFF). Characterize picked the ring over the ChArUco board
residual (r 40.5 mm, centre (197.5, 152.5), bead 10.4 mm) and a separate Measure re-found
it 0.5 mm away — that repeatability is itself evidence.

**The paper session exists: `runs/extrusion/20260828-204846-5b455377`** (2026-08-28
evening, Layers 3, New session, Characterize → *Apply* done correctly: r 42.6 mm, bead
12.8 mm, height 2.8–13.0 mm, centre (214.6, 146.7)). Its layer-001 take 1 failed on
the cell — ChArUco-board depth noise fused to the ring, fixed in code the same night
(see *Board noise* below) — and was **reprocessed offline from its archived frame**:
r 42.31, centre offset 1.28 mm, mean/RMS/max 1.51/1.85/3.91 mm, shape RMS 1.58,
completeness 0.992. That is the first zero-offset baseline take. Continue in this
session.

**Figures are done** (`tasni/modules/extrusion/figures.py`, merged `ba8d2b3`). Four per
take plus one per trial, 300 dpi PNG **and vector PDF**, rendered from the archive with
no robot. They draw automatically on every take from now on, and older takes render on
first view.

| Figure | Use in the paper |
|---|---|
| `plan` | the headline: cloud + extracted centreline + nominal ring, mm axes, scale bar |
| `heightmap` | bird's-eye relief with a z colourbar — shows the wavy bead at a glance |
| `iso` | oblique cloud + centreline, vertical exaggeration stated on the axis |
| `profile` | unrolled z(θ) and Δr(θ); a pure shift shows as a clean cosine |
| `stack` | per trial: every layer's latest take, plan + oblique — where offsets read |

Paths: `runs/extrusion/<trial>/layer-NNN/figures/*.pdf` and
`runs/extrusion/<trial>/figures/stack.pdf`.

## 3. What is left — the cell run

**The app now carries this protocol.** The Extrusion page's *Ring stack — measure
only* card opens with a **Run guide** listing every step below with live progress
(`3/5` takes done, etc.), the traps beside the buttons, and the ground truth echoed
back before each press. What follows is the same thing on paper.

**The one thing that went wrong last time — *Apply to recipe & placement* was
skipped** between Characterize and Measure. Layer-001's 15.38 mm centre offset and
11.31 mm RMS are the ring measured against a stale plan: **an artifact, not a
result.** That class of mistake is now refused rather than archived — see *Gates*
below — but the order still matters.

Restart the backend first (it caches imported modules; check `/api/health` →
`build.stale`), then, in the Extrusion page:

0. **Set Layers to 3** before anything else, and press **New session**. *Apply to
   recipe & placement* keeps whatever layer count the recipe already has — it only
   rewrites radius, bead, layer height and centre — and *Measure layer 2* is
   refused with `layer_index must be 1..1` on a one-layer plan.
1. Scan surface applied → **Center on scanned surface** → **Generate**.
2. Place ring 1 within ~50 mm of the table centre → **Characterize ring**.
3. **Apply to recipe & placement** → **Generate**. ← *the step that was missed*
4. **Noise floor** (phase `noise floor`): Measure layer 1 **five times** without
   touching the ring.
5. **Placement repeatability** (phase `re-placed`): lift and re-place ring 1
   **three times**, measure each.
6. **Stack** (phase `stacked true`): ring 2 placed true → Measure L2 (×3); ring 3
   true → Measure L3 (×3).
7. **Introduced offsets — on the TOP ring only** (phase `top ring shifted`). Type
   the value into *introduced offset X/Y* **before** pressing Measure, so ground
   truth is archived beside the result. Shift the top ring **10 mm** along frame +X
   → Measure ×3; then **15 mm** → Measure ×3. Optional: prop one side for a tilt case.
8. **Paper summary** → copy the Markdown block.

**Set the Phase selector on every take.** The summary groups by *layer + phase +
introduced offset*, so the phase is what separates sensing repeatability (step 4)
from placement repeatability (step 5) — pooled into one row they hide each other,
and the paper wants both.

### Gates: what the app now refuses

None of these need a robot to detect, so they fire before the motion checks.

- **Measuring against a plan that is not the one this session applied.** Once
  *Apply* has run, the session is bound to that plan's fingerprint. Pressing
  *Center on scanned surface → Generate* (or regenerating for any other reason)
  makes Measure refuse and name the fix: press **Apply to recipe & placement**
  again. Re-centring while a session is bound also asks for confirmation first.
- **Measuring layer N before layer N−1 has a measured top.** Layer N's ROI floor
  IS layer N−1's latest measured take; without it a stacked ring blends into the
  ring beneath (the synthetic proof exhausts the branch guard outright).
- **Invalid takes** are shown in the table with their reason and a **Reprocess**
  button (no robot motion — it re-runs the current processing on the archived
  RGB-D frame). They are counted as takes but never averaged into any statistic.

### Resuming after a backend restart

The plan lives in memory, but the session records what it applied, so **the app
restores it on its own** — the card says *"Plan restored from session …"* and you
can measure straight away. If a session predates that record (the 2026-08-28 paper
session does), press **Apply to recipe & placement** once: it now applies the
characterization onto the plan the *session* was created with, and reproduces the
exact fingerprint its takes were measured against (verified offline for
`20260828-204846-5b455377`: `7465b81877`, r 42.6 / bead 12.8 / layers 3). Do
**not** press *Center on scanned surface → Generate* to recover — that rebuilds
the pre-Apply plan.

### Board depth noise — fixed in code, do not work around it in config

The `branch guard exhausted` on layer-001 take 1 was the board's own depth noise
(z p99 +4.8 mm on a bare board; 22.7 % of it clears the 2.5 mm floor) joining the
ring's cluster and dilating into a lobe. Fixed by a radial trim about the fitted
ring (`extrusion.radial_trim_schedule_mm`, see `docs/extrusion-current-handoff.md`).
Do **not** raise `deposit_min_height_mm` (a 3 mm floor read r 36.7 for this 42.6 mm
ring and called it valid) and do not narrow `radial_roi_margin_mm` (it caps the
introduced offset at ~18 mm). If a take still fails, its raw RGB-D is archived and
**Reprocess** scores it against the take's own plan.

### Why displacements go on the top ring, and why three takes

- **Top ring only.** Layer N's ROI floor is the *latest take* of layer N−1
  (`MeasureSession.floor_profile`). Displacing a ring that something else is
  measured on top of corrupts that floor for every take above it. Displace the
  highest ring in the stack and nothing downstream is affected.
- **Three takes per condition, not one.** `paper-summary` reports mean ± sd per
  condition; a single take has no sd, and requirement #3 needs ≥ 12 measurements in
  total anyway.

### Getting the ground truth right — the weakest link

"Shift it 10 mm" done by eye is the least trustworthy number in the experiment, and
every detection-error figure is measured against it. The work frame came from the
board rectangle, so **its axes are parallel to the board edges** — a steel rule laid
along an edge IS the frame axis. Mark where the ring sits (two pencil ticks against
its outer edge), slide it along the rule, and **type the distance you actually
achieved**: 12 mm scores exactly as well as 10 mm, because the summary compares the
measurement against what you typed, not against a round number.

**Do not use the ChArUco square pitch as the ruler on this cell.** The board in
`tasni.config.json` is A3, 8×6 at **40 mm** squares, and one square exceeds the
25 mm cap — at 40 mm the displaced ring falls outside the ±30 mm radial search band
and the take comes back invalid. (The 30 mm A4 board would be just as unusable.)
There is no printed feature at a usable pitch: the markers are 29.3 mm inside their
40 mm squares, so the only printed intervals are 5.35, 29.3 and 40 mm.

Take one throwaway measurement first to learn which edge is frame **+X** and its
sign (the reported `center_offset_mm` tells you), then re-take with the annotation
right — the summary groups by what you typed, so a sign error puts
good data in a mislabelled group. The offset fields are **sticky between presses**:
the card echoes *"This press records: layer 2 · top ring shifted · introduced offset
(10, 0) mm"* above the button, and there is a **Clear offset** button — read that
line before every press.

### Watch this on layer 2's first take

Layer N keeps only points above the measured top of layer N−1 plus
`layer_floor_margin_mm` (2.0). These rings run **2.9–11.0 mm** tall, so where ring 2
sits only ~3 mm above ring 1 there is under 1 mm of margin and its low stretches can
be clipped — showing up as reduced `path_completeness` or an angular gap, i.e.
`valid: false`. The synthetic proof used a *uniform* ring, so this case is untested
on real geometry. **Check completeness on L2 take 1 before continuing.** If it
clips, lower `extrusion.layer_floor_margin_mm` and press **Reprocess** — every take
archives its raw RGB-D, so no cell time is lost.

Constraints and expectations:

- Keep every offset **≤ 25 mm** — the radial ROI is ±30 mm.
- For a pure shift `d`: centre offset ≈ `d`, max ≈ `d`, mean ≈ `0.64 d`, RMS ≈ `0.71 d`.
  A 10 mm shift should read ≈ 10 / 10 / 6.4 / 7.1 mm. **If it does not, stop and
  investigate — that relation is the built-in sanity check.** `paper-summary` now checks
  it for you and prints a `WARNING` naming which statistic disagrees.
- ≥ 12 measurements total gives requirement #3 its mean ± sd. Takes reprocessed
  offline do not contribute to it (their processing time is a desktop number); the
  summary says how many it left out.
- 300 mm is the sensor floor at 1280×720 (MinZ ≈ 280 mm). Closer needs a lower depth
  profile on the Jetson — a separate, deliberate change, not something to try mid-run.
  The camera already **climbs with the stack**: it aims at the cylinder axis at the top
  of the layer being measured and holds 300 mm above *that*, so L1/L2/L3 are measured
  from the same standoff, not from a fixed Z.
- Collisions default OFF for measure-only: the hand-placed stack is not in the station
  model. IK/reachability screening still runs.

## 4. After the run

- Figures for every take appear automatically; pick the ones the paper wants and use
  the **PDF**. A take with an introduced offset also draws the **ground truth ring**
  (nominal + that offset, teal dash-dot) in `plan` and in the trial `stack`, so the
  figure shows the extracted centreline landing on where the ring was actually moved
  to rather than merely displaced from nominal.
- **Paper summary** gives the deviation table grouped by **layer, phase and
  introduced offset**, timing mean ± sd, height/bead stats and valid X/N, already
  worded correctly. Invalid takes are counted but excluded from every average, and
  offline-reprocessed takes are kept out of the cycle-time statistic. It also reports
  the **detection error** — `|measured centre offset − the offset you typed|` — which is
  the claim the paper actually makes, and a sentence per condition ("a 10.0 mm introduced
  offset was recovered as 10.05 ± 0.28 mm"). A take archived before that field existed
  is left out of the score rather than counted as a perfect measurement.
- Copy the first good capture's `color.png` + `depth.npy` into
  `tests/fixtures/extrusion/ring1/` as a regression fixture if it is better than the
  one already there.
- Discard every `runs/extrusion/20260828-*-f088cf48` trial: those predate the ring
  selector fix and measure an empty board. Keep their raw frames as evidence only.

## 5. Known-good numbers you can already cite

From `20260828-192115-47fb78ea` (characterization + one measure):

- Ring, characterized from its own scan: **radius 40.5 mm**, bead **10.4 mm**,
  height **2.9–10.8 mm**.
- Capture-to-capture repeatability of the ring centre: **0.5 mm**.
- Acquisition → reconstructed 3-D path: **3,124 ms** (capture 2,874 + processing 250).
  *One sample — needs ~10 for mean ± sd.*
- Layer height along the ring: mean **6.1 mm**, min 2.9, max 11.0; bead footprint
  **9.9 mm**.
- Path extraction: **1/1 valid**, completeness 0.99, max angular gap 2.9°.
