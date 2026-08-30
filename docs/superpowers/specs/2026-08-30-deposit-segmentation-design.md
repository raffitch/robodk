# Deposit segmentation without colour — design

**Status:** design agreed with the operator on 2026-08-30, after an offline validation
run against the archive. Revised the same day by a code-verifying review — see §12. **Supersedes** the chroma gate introduced in `041ad1b`
(2026-08-29) and the handoff that condemned it,
`docs/deposit-segmentation-handoff-2026-08-30.md` (`8b2db77`).

**Code facts below are verified against `main @ 8b2db77` (2026-08-30).**

**Every number in §2 was measured offline** from
`runs/extrusion/20260830-202416-293b208d` with no robot motion and no re-capture, using
a prototype front end spliced onto the *shipped* downstream chain. Nothing was tuned to
make the result look good; where the measurement contradicted the proposed design, the
design changed (§3.2, §3.4, §11).

Background to read first: `docs/deposit-segmentation-handoff-2026-08-30.md` (what
failed on the cell and why), `docs/superpowers/specs/2026-08-27-ring-stack-measure-only-design.md`
(the chain this modifies).

**Not blocked by the PFH paper.** The paper stands on the eight layer-1 takes already
archived and already measured by the old chain; this design does not touch them.

---

## 1. Why this exists

`process_observation` decides, per point, whether it is deposited material or the
surface underneath. Today it decides with two mechanisms, and both are broken:

**The saturation gate has inverted.** It was built when the clay separated from the
printed board ~20:1 in HSV saturation. Measured on 2026-08-30, the bead's median S is
25 and the board's black squares' is 28 — the board is now *more* chromatic than the
bead. This is not a drift to re-tune. Colour auto-exposure runs free on the Jetson
(`server_unicast_syncronous.py:1082` pins only `auto_exposure_priority`, and per-frame
exposure is never recorded), so a fixed threshold on saturation was never a calibrated
quantity. It is measured downstream of a gain loop that responds to whatever is in
frame — a white ring, a different mat, the room lights.

**The work frame is not the datum.** `min_z = deposit_floor_mm(...)` is a constant Z in
the work frame, justified in-code as "deterministic height subtraction is more
reproducible than fitting a new plane per frame". Measured, the board sits **1.2 mm
below work-frame Z = 0 and tilted 0.48–0.62°**, adding ±0.75 mm across the ±70 mm radial
ROI band. That is roughly 2 mm of systematic error spent before sensor noise, against a
1.5 mm threshold.

The two are the same problem solved twice — *what surface is this bead sitting on?* —
and `floor_profile` (layer N−1 as layer N's reference) is a third instance of it.

**The principle:** discriminate on the invariant (geometry), not the accident
(appearance). Height above the surface the deposit rests on is physically true
regardless of pigment, substrate or lighting.

---

## 2. The evidence this design rests on

Prototype front end, shipped downstream chain (`_filter_deposit`, `_radial_trim`,
`_top_surface`, the rasteriser, the branch guard, `compare_circle`), so every difference
is attributable to segmentation alone.

### 2.1 Layer 1, eight takes — colour is not needed

| front end | valid | completeness | radius mean | radius σ | offset mean | offset σ |
|---|---|---|---|---|---|---|
| old — chroma + 1.5 mm constant | 8/8 | 0.992–0.993 | 41.01 | 0.056 | 1.08 | 0.057 |
| new — fitted substrate + 3σ | 8/8 | 0.992–0.993 | 40.98 | 0.107 | 1.08 | 0.070 |

Completeness, validity and centre-offset repeatability are unchanged with **no colour
input at all**. The radius σ difference (0.056 → 0.107 mm) **cannot be resolved at
n = 8**: two-sided F = 3.66 on 7 and 7 df, p = 0.109 against 4.99 needed at 5%; in the
one direction that matters (new worse than old) p ≈ 0.054. Eight takes have little
power, so the honest statement is "not distinguishable here, and below the 0.132 mm
raster-quantisation floor (§7)" — not "proven absent". The golden harness therefore
pins radius σ across the eight takes at ≤ 0.15 mm (§5), so a real repeatability cost
surfaces instead of hiding under an unchanged mean. See §11 for what the
investigation did find.

### 2.2 Substrate separation

Height above a fitted substrate versus above work-frame Z = 0, at a 1.5 mm cut:

| take | reference | substrate admitted | ring kept |
|---|---|---|---|
| `layer-001` | work-frame Z = 0 | 0.78% | 78.2% |
| `layer-001` | fitted substrate | **0.00%** | 73.2% |
| `layer-002` | work-frame Z = 0 | 2.34% | 83.3% |
| `layer-002` | fitted substrate | **0.10%** | 82.3% |

Substrate false positives fall 8–80× for 1–5 points of ring retention. Those few hundred
stray board points are what feed the branch guard and bias the radial trim — the failure
`041ad1b` was written to prevent.

### 2.3 The board is invisible in height

Under camera protocol 2 the depth quantisation is 0.1 mm, versus the 1.0 mm the gate was
written against. Height above a locally-fitted substrate, **measured in the previous
session and reported in the handoff** (not re-derived here):

| region | p50 | p99 |
|---|---|---|
| board black squares | 0.14 mm | 1.52 mm |
| board white paper | 0.00 mm | 1.24 mm |
| clay bead | 0.63 mm | 10.0 mm |

Independently corroborated by this session's own fit: substrate p99 is 0.79 mm on
`layer-001` and 1.09 mm on `layer-002`, against a ring core p50 of 2.80 mm and 8.04 mm.

Either way the black squares sit *at* the plane. The original justification for a colour
gate — that no height floor could exclude them — no longer holds.

### 2.4 Layer 2 is not a segmentation failure

Every 10° sector around ring 2 carries 200–530 valid depth pixels; nothing is unsensed.
The median height of raised material per sector swings **2 to 16 mm**, and all three
takes trace the same profile to within a line width. The notches near 130° and 250° read
2–4 mm — ring 1 exposed, ring 2 absent above it. Completeness of 0.56–0.62 is an honest
measurement of a badly stacked physical ring.

This retires the handoff's open question ("not yet verified: that reference subtraction
recovers the failing layer-2 takes"). It does not, and it should not: feeding layer 1's
measured top as the reference correctly leaves nothing where ring 2 is not, taking
completeness 0.62 → 0.50.

---

## 3. The design

### 3.1 One contract: `SubstrateModel`

New module `tasni/modules/extrusion/substrate.py`:

```python
class SubstrateModel(Protocol):
    source: str            # "fitted_plane" | future providers
    sigma_mm: float        # robust scale of the substrate's own residuals, this frame
    def height(self, xyz: np.ndarray) -> np.ndarray: ...   # mm above the substrate
    def floor_mm(self, k: float) -> float: ...
    def to_report(self) -> dict: ...
```

`deposit_floor_mm`, `chroma_gate_mask` and the `floor_profile` special case all collapse
into it. The interface admits further providers (a captured empty-plate reference, layer
N−1's measured top); **none ships in this change** — see §11.

### 3.2 The estimator: deterministic IRLS, one-sided

`PlaneSubstrate` fits `z = ax + by + c` by iteratively reweighted least squares with a
**one-sided Tukey biweight** — positive residuals down-weighted harder (`c⁺ = 2.0`) than
negative (`c⁻ = 4.685`), because the deposit is the only thing that can sit above the
surface. Seeded with plain least squares, 12 iterations, no randomness.

**RANSAC was tried first and rejected on measurement, not taste.** It is the textbook
primitive (`Open3D segment_plane`), and it fails two requirements:

- **Determinism.** Open3D 0.17 ignores `o3d.utility.random.seed`; the same frame refits
  to a different plane on every run (intercept ±0.055 mm, tilt ±0.008°). A chain that
  cannot reprocess a frame to the same number twice is not a measurement chain.
- **Accuracy.** Against a substrate-only ground truth over four cell frames, mean |error|
  is 0.064 mm for one-sided IRLS versus 0.191 mm for RANSAC + refit, 0.113 mm for
  trimmed least squares and 0.178 mm for two-sided Tukey.

The fit is guarded: if the recovered normal is more than ~25° off the work frame's up
axis, the frame is refused rather than measured against a wall or a fixture.

The fit region is the neighbourhood of the ROI (`r < 150 mm` about the plan centre,
configurable as `substrate_fit_radius_mm` — the §8 mitigation "the fit region can be
widened" has to be operable without a code change) — the work frame supplies the
up-axis and the search band. **No ring geometry is used**, so the estimator carries no
assumption about what is being deposited.

### 3.3 Scale from the uncontaminated half

`sigma_mm` is `median(residual) − p15.87(residual)`. Deposit contaminates only the
positive side, so the lower half is pure sensor noise and a one-sided estimator is both
robust and unbiased here, where a two-sided MAD is inflated by the bead.

Measured on the four ground-truth cell frames of §3.2: `sigma_low` returns
0.55–0.61 mm against a ground-truth substrate σ of 0.44–0.52 — conservative in the
safe direction, and stable across takes.

### 3.4 The threshold is derived, not constant

```
floor = clamp(k · substrate.sigma_mm, floor_min, floor_max)     # k default 3.0
```

On the archive this yields 1.55–1.74 mm across the eight layer-1 takes — it
*reproduces* today's hard-coded 1.5 mm on today's geometry, and adapts at a different
standoff or incidence where the constant is silently wrong. (That floor range implies
per-take σ down to 0.52 mm — a different frame set from §3.3's four ground-truth
frames, which is why the two ranges do not coincide exactly; the golden harness
records per-take `sigma_mm` baselines and is the authoritative reconciliation.) The
clamp exists so a pathological fit cannot open the floor to everything or close it to
nothing. **Defaults: `substrate_floor_clamp_mm = [1.0, 2.0]`** — the ceiling sits
safely under the measured k = 4 cliff at 2.25 mm, the lower bound under anything the
archive's noise justified (3σ never measured below 1.55 mm).

`k = 4.0` was measured to fall off a cliff (completeness 0.358): at a 2.25 mm floor the
cut eats into a ring whose crest is 2.9–4.9 mm over much of its circumference. `k` is
bounded accordingly.

**The substrate is refitted every frame.** This was tested against the alternative —
fitting once and reusing — which is far worse: radius σ 0.107 → 0.234 mm, offset σ
0.070 → 0.254 mm, completeness floor 0.992 → 0.821. Refitting absorbs pose-dependent
depth bias that any stale reference carries straight into the measurement.

### 3.5 Topology replaces the gate's safety role

The gate's one defensible function was killing the 22-point checker patch that dilated
into skeleton arms and exhausted the branch guard. That is a **compactness** property,
not a colour one.

Before the deposit filter, raster the ROI to a 2.5D height map at
`raster_mm_per_pixel`, close speckle at half a bead width, label 8-connected components,
and drop any component whose **principal-axis extent** is below
`deposit_min_length_beads` (default 3.0) bead widths. Measured on the archive: layer-001
keeps 1 of 1 component; layer-002 keeps 2 of 4 and 1 of 9 — the rejected ones are exactly
the compact patches.

The filter is **fail-open**: if it would leave fewer than `cluster_min_points`, it is
bypassed and recorded as bypassed, so a thin or fragmented real ring is never starved by
topology alone.

### 3.6 What is deleted

- `chroma_gate_mask` and its call sites in `process_observation` and `characterize_ring`.
- Config: `deposit_min_saturation`, `deposit_min_chroma_fraction`,
  `deposit_min_height_no_chroma_mm`.
- `deposit_floor_mm`, and the `plane_distance_threshold_m` / `deposit_min_height_mm`
  coupling that fed it. With the function gone both fields are dead — no other
  consumer exists in the extrusion chain — so they are removed too, not left as
  clutter.
- `K` and `dist` from `process_observation`'s signature: in the extrusion module
  `ColorRegistered` exists *only* to serve the gate (`processing.py:707`, `:956`). The
  front of the chain becomes `depth_to_work_points`, which already exists and is already
  what `figures.py` uses. The scan module's use of `ColorRegistered` is untouched.
- The `floor_profile` parameter, replaced by a `SubstrateModel` argument — **end to
  end**, because it has consumers beyond `process_observation`: `ring_geometry`'s
  reference argument, `Session.floor_profile` (`measure.py:134`), the
  measure-layer-N−1-first 409 guard and its `allow_missing_floor` escape
  (`module.py:768–776`, `:93`), `layer_floor_margin_mm`, and the web UI's client-side
  copy of the same gate (`Extrusion.tsx:998`, `:1217`). `Session.tops` **stays
  recorded**: the UI renders it, and the future layer-N−1 provider (§11) would be
  built from it — only its consumers go.

One trap the deletions must clear: `ExtrusionConfig` is `extra="forbid"`, and the
archive's per-take `processing_config` is re-validated on every reprocess and figure
build (`service.py:1213`, `figures.py:837`, `:874`). Deleting fields therefore breaks
every existing archive unless retired keys are stripped first — a `from_archive()`
constructor drops (never reinterprets) the retired keys, and those three call sites go
through it.

The colour frame keeps being captured and archived. It is evidence and figure material;
it takes no part in any decision.

Consequence worth noting: the gate discards **45.6%** of the valid depth cloud
(400,830 of 878,222 on `layer-001`) because those points project outside the narrower
colour FOV — depth is 90° horizontal, colour 70°, an area ratio of ~0.49 that this
matches. They stop being discarded for a reason that has nothing to do with them. Most
sit far outside any ring ROI, so this is a cost and an honesty gain rather than a
recovered signal.

### 3.7 One seam for live, reprocess and figures

`measure.py`, `service.reprocess_saved_layer` and `figures.py` currently call
`process_observation` with **different arguments for the same take** — `assemble_arcs`
is `layer_index == 1` live and `False` in both others, and `reprocess_saved_layer`
passes no `floor_profile` at all. (Verified against the code, there are five direct
callers, not three: `measure.py:866` live measure, `service.py:1051` live-print
inspection, `service.py:1197` reprocess, `figures.py:830` take figure and
`figures.py:912` characterization figure — plus `characterize_ring`'s own refined
pass at `processing.py:1005`. All of them route through the seam.)

This is the whole of the handoff's "`floor_profile` is `None` in production, and the test
asserts otherwise" mystery (§3 of that document, flagged as the highest-value thing to
chase). There is no production/test contradiction: **all three archived layer-2 reports
carry `offline_reprocess: True`**, so they are reprocess output that overwrote the live
measurement. The live path did receive its floor.

Introduce one `measure_take(...)` entry point that every caller uses. `assemble_arcs`
stops being caller-chosen: the seam derives it from the take itself
(`layer_index == 1`, the isolated-ring case), so the same take gets the same answer on
every path. Deliberate behaviour change: the live-print layer-1 inspection and the
layer-1 reprocess/figure paths flip to assembly ON, aligning them with the live
measure — the divergence was the defect. This lands **first**, because without it the
offline validation in §5 proves nothing.

---

## 4. What gets reported

Per take, in `report["substrate"]`: source, `sigma_mm`, derived floor, plane tilt versus
work Z, plane offset at the ring centre, inlier fraction, and the component tally from
§3.5.

Plus one derived number, **separation margin** = bead p50 − substrate p99. It is the
surface- and material-agnostic answer to "is segmentation healthy on this setup" — 3.0 mm
against 1.47 mm on `layer-001`. A take whose margin collapses says so, instead of quietly
returning a low completeness that reads like a placement fault.

---

## 5. Testing

- **Unit**: `PlaneSubstrate` against synthetic tilted and warped surfaces with a known
  bead — assert recovered tilt, recovered σ, and height accuracy. Assert **bit-identical
  output across repeated fits** (the RANSAC failure mode, pinned).
- **Unit**: `compactness_filter` rejects a compact patch and keeps an arc of equal pixel
  count; asserts the fail-open bypass.
- **Golden**: the eleven archived takes reprocessed (read-only — the harness must
  never write into `runs/`), metrics compared against recorded baselines. Layer-1
  acceptance: 8/8 valid, completeness ≥ 0.990, radius mean within 0.10 mm of 41.0,
  **radius σ across the eight takes ≤ 0.15 mm** (§2.1 — the σ question is
  underpowered, so the harness holds the line instead). Layer-2 acceptance:
  completeness within 0.05 of the archived value — these takes are *expected to
  remain invalid* (§2.4) and the test pins that, so a future change that "fixes" them
  is caught as the false positive it would be. The harness lands right after the seam
  and records the OLD chain's numbers first, so the front-end swap is judged against
  a baseline the same harness produced. `runs/` is git-ignored: the golden tests skip,
  loudly, on a machine without the archive.
- **Contract**: the three call sites of §3.7 return identical results for one take.

Per `~/.claude/memory`, the full suite is too slow to run: use `pytest -k` on the
extrusion tests plus an import check, and `npm run build` if any frontend touches this.

---

## 6. Files touched

| file | change |
|---|---|
| `tasni/modules/extrusion/substrate.py` | new — `SubstrateModel`, `PlaneSubstrate`, `compactness_filter` |
| `tasni/modules/extrusion/processing.py` | remove gate + `deposit_floor_mm`; take a `SubstrateModel`; drop `K`/`dist` |
| `tasni/modules/extrusion/measure.py` | build the substrate, call the seam |
| `tasni/modules/extrusion/service.py` | `reprocess_saved_layer` through the seam |
| `tasni/modules/extrusion/figures.py` | stage renders through the seam; `_chroma_dist` and the K/dist plumbing die with the gate |
| `tasni/modules/extrusion/module.py` | remove the measure-layer-N−1-first 409 guard + `allow_missing_floor` (obsolete with `floor_profile`) |
| `tasni/modules/extrusion/paper_docx.py` | methods text stops describing the chroma gate / constant floor; renders `report["substrate"]` (legacy archives keep their `.get` fallbacks) |
| `tasni/webui/src/pages/Extrusion.tsx` | drop the client-side previous-layer gate (`tops`-presence checks); keep the `tops` rendering |
| `tasni/core/config.py` | remove 6 fields (`deposit_min_saturation`, `deposit_min_chroma_fraction`, `deposit_min_height_no_chroma_mm`, `plane_distance_threshold_m`, `deposit_min_height_mm`, `layer_floor_margin_mm`); add `substrate_sigma_k`, `substrate_floor_clamp_mm`, `substrate_fit_radius_mm`, `deposit_min_length_beads`; add `ExtrusionConfig.from_archive` (strips retired keys — `extra="forbid"` otherwise refuses every existing archive) |
| `tests/test_extrusion_processing.py`, `tests/test_extrusion_measure.py` | ~9 gate tests replaced; the `ring1`/`ring2` fixture READMEs re-scoped |

---

## 7. Out of scope, deliberately

- **Raster quantisation.** Radius is read off a 1 mm-quantised skeleton; re-rasterising
  the same crest cloud on a sub-pixel-shifted grid moves it 0.132 mm — more than
  segmentation contributes. It affects the old chain equally, halving the pixel size
  exhausts the branch guard on 3 of 8 takes, and the real fix (fit the circle to crest
  points, which is already more repeatable at σ 0.044 but reads 0.10–0.25 mm high) needs
  its own design. Bundling it would make both changes impossible to judge.
- **Sensing.** Untextured white and specular materials under-resolve in IR stereo — the
  foam ring resolved ~55% of its circumference with two dead arcs on the stereo baseline.
  This design *reports* that (§4); it cannot fix it. Named follow-up: pin colour
  exposure/gain and record them per frame, and re-characterize the depth envelope on
  white and dark materials.
- **A captured-reference provider.** See §11.
- **`upwards_normal_z`.** §5 of the handoff shows the crest filter is where a thin ring
  stops being a surface. The golden harness makes it measurable; changing it is a
  separate decision with its own evidence.

---

## 8. Risks

- **A substrate that is not planar.** The estimator assumes one dominant plane in the ROI
  neighbourhood. A curled sheet or a curved build surface breaks that. Mitigation: the
  normal guard (§3.2) refuses rather than mismeasures, and `sigma_mm` rises visibly
  before it refuses. A low-order surface provider is the escalation, and it plugs into
  the same interface.
- **A deposit that covers most of the ROI.** IRLS assumes the substrate is the majority.
  For a dense raster or a wide print this inverts. Mitigation: the fit region is a
  neighbourhood, not the ROI, and can be widened; `inlier_fraction` is reported.
- **The 2026-08-29 crash frame is unvalidated.** The offline run used the 2026-08-30
  archive; the branch-guard fixture
  (`tests/fixtures/extrusion/ring1/ring1_take04_branchguard_20260829.npz`) — a
  pre-protocol-2 capture whose board patch sits *raised* 2.9–5.9 mm and welded near
  the ring's flank — is the hard case for §3.5 and has not been run through the new
  chain. The implementation plan measures it before pinning, and stops the change if
  the patch is silently included in a valid measurement (a refusal, by contrast, is
  an acceptable outcome — the 2026-08-29 ruling stands: the crash was the good one).
- **Layer N on layer N−1.** With `floor_profile` deleted, a stacked layer is measured
  against the base plane, so its ROI ceiling must accommodate the full stack height.
  §2.4 shows previous-layer referencing made things worse on the only stacked data we
  have — but that data is a badly stacked ring, so the question is not settled, only
  unevidenced. Recorded as an open question, not a solved one.

---

## 9. Open questions

1. Does a well-stacked ring 2 need a previous-layer reference? Cannot be answered from
   this archive (§2.4). Answer it the next time a good stack is measured.
2. Is `k = 3.0` right at a different standoff? The 300 mm inspection distance is the only
   one measured. `sigma_mm` should track standoff; the golden harness will show it when a
   different-distance take exists.

---

## 10. Non-goals

This is not a general segmentation library, and it does not attempt to be
material-agnostic in the *sensing*. It answers one question — bead or substrate —
using geometry only, for a chain that already knows roughly where to look.

---

## 11. What the offline run changed in this design

Recorded because these were the proposed design until the archive said otherwise:

1. **RANSAC → deterministic IRLS** (§3.2). Determinism and accuracy, both measured.
2. **Per-frame refit is required, not incidental** (§3.4). The shared-substrate variant
   was tested and is much worse.
3. **The captured-reference provider does not ship.** Both things it was meant to buy
   failed to appear: previous-layer referencing made layer 2 worse (0.62 → 0.50), and a
   reused reference hurt repeatability. The interface stays so a provider can be added
   against evidence; building it now would be speculative.
4. **The radius regression is not a regression** (§2.1, §7). p = 0.109 at n = 8. It was
   about to become the first task in the plan.

---

## 12. What the 2026-08-30 design review added

A second pass verified every code fact against `main` and amended, without changing the
direction:

1. **The σ claim was over-stated as settled** — reframed as underpowered at n = 8, and
   the golden harness now pins radius σ ≤ 0.15 mm so the question stays measured (§2.1,
   §5).
2. **The `floor_profile` deletion was under-scoped** — its consumers reach
   `ring_geometry`, `Session.floor_profile`, the `module.py` 409 guard +
   `allow_missing_floor`, `layer_floor_margin_mm` and the web UI's client-side gate;
   `Session.tops` recording deliberately survives (§3.6, §6).
3. **The clamp had no stated defaults** — now `[1.0, 2.0]` mm, with the reasoning (§3.4).
4. **§3.3's and §3.4's σ ranges do not exactly coincide** — attributed to their
   different frame sets; the golden baselines are the authoritative per-take record
   (§3.4).
5. **Two facts the implementation would have tripped over**: there are five
   `process_observation` callers plus `characterize_ring`'s internal pass, not three
   (§3.7); and `ExtrusionConfig` is `extra="forbid"`, so field retirement without a
   `from_archive` key-stripper breaks every archived take's reprocess (§3.6, §6).
