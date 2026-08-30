# Multi-view ring inspection + side photo — design (revamp)

**Status:** design agreed with the operator on 2026-08-30. **Supersedes**
`docs/superpowers/specs/2026-08-29-multiview-inspection-design.md` (`a1cafa0`) and its
plan `docs/superpowers/plans/2026-08-29-multiview-inspection.md` (`c31c720`), both of
which were written against `main @ f4f06e6` and are retired by this document —
see §2 for what invalidated them.

**Code facts below are verified against `main @ da5f7a4` (2026-08-30).**

**Scope:** the ring-stack **measure-only** experiment and its paper export. The live
print reuses the same capture function later (§11). **Multi-view is an opt-in toggle,
default OFF**, so the validated single-view chain and every number already archived stay
exactly as they are until someone asks for otherwise.

**Do not start implementing this before the PFH paper's cell run is done.**
`docs/pfh-paper-handoff.md` carries a **1 September 2026** deadline and still needs
numbers #2, #2b and #3 measured on the *single-view* chain. This design deliberately
touches the shared capture path, and it redefines `acquisition_to_path_ms`, which is
the paper's number #3.

Background to read first: `docs/pfh-paper-handoff.md` (the experiment, its traps, the
paired detection error), `docs/superpowers/specs/2026-08-27-ring-stack-measure-only-design.md`
(the chain this extends), `docs/realsense-quality-headroom-2026-08-30.md` (why sensor
tuning is not the lever), and `characterization/characterization-20260813.json` (the
incidence numbers in §3).

---

## 1. Why this exists

The mock extruded rings are **thin**. Ring 1 characterizes at radius 40.5 mm, bead
10.4 mm, height 2.9–10.8 mm — so over much of its circumference the crest stands only
2.9–4.9 mm above the board, against bare-board depth noise that reaches +4.8 mm p99 at
300 mm. One straight-down RGB-D frame sees the crest at grazing signal-to-noise and the
flanks hardly at all.

Repeating frames from the same pose (the existing `repeats` count) averages the same
missing flanks. The operator's goal is **more resolution and a truer profile of the
ring** — which means seeing the parts of the bead a top-down camera structurally cannot
see.

**Sensor tuning is not the lever.** `docs/realsense-quality-headroom-2026-08-30.md`
swept laser power (0–360), visual preset, depth resolution and `disparityShift` at this
exact 300–500 mm standoff and found no usable headroom: laser power is flat across its
whole range, 848×480 is quieter on a flat board but 8 % *worse* on the bead, and
disparity shift is actively harmful this close. The single surviving candidate
(`visual_preset=Medium Density`, ring crest fraction +6 %) is a separate, unvalidated
one-line experiment. Geometry — more viewpoints — is the remaining lever.

### What multi-view will and will not buy (say this in the paper too)

- **It adds the flanks, and the flanks are where the profile lives.**
  `processing.py:378 _filter_deposit` keeps "everything the deposit IS, flanks
  included"; `processing.py:599 _top_surface` then selects only upward-facing points
  (`upwards_normal_z = 0.92`, i.e. normals within ~23° of vertical) to get the crest the
  centreline is read from. So `processing.py:292 bead_width_profile` and
  `processing.py:320 ring_geometry` — bead width, layer-height range, cross-section —
  read the flanks *before* that selection throws them away. A view tilted by θ rotates
  the visible arc of the bead's cross-section by θ toward the near flank; three azimuths
  put every flank point within 60° of some camera.
- **It helps the centreline much less.** The top view already sees the crest best, and
  `_top_surface` is going to discard everything the tilted views uniquely contribute to
  the crest anyway. Expect the merged centreline to gain dropout-filling and
  per-voxel averaging, not a step change. Claim the profile; do not oversell the centre.
- **It does not raise lateral resolution.** That is fixed by the sensor: ≈0.4 mm/px at
  300 mm, and 300 mm is the D435i depth floor at 1280×720 (`measure_close_range_min_mm`,
  and MinZ scales with depth width, so only a *lower* depth profile would open real
  headroom — see the config comment at `config.py`).
- **It does not average depth noise unless the views are registered to better than that
  noise.** §5.3 is the registration that makes the merge safe; §10 is the A/B that
  decides whether merged takes go in the paper at all.
- **Tilt costs board quality**, and §3 has the measured price.

---

## 2. What changed since the 2026-08-29 spec, and why it was retired

Roughly twenty-five commits landed on `main` between `f4f06e6` (what the old spec was
verified against) and `da5f7a4`. Four of them invalidate load-bearing parts of it.

| # | The old spec said | Current reality |
|---|---|---|
| 1 | Registration must correct a **≈2 % lateral scale mismatch** — "the Jetson aligns depth with factory intrinsics, the host back-projects with calibrated ones" — and "views from opposite sides disagree by 1–2 mm; a naive merge blurs it". That mismatch is the *reason* the old spec's §4.3 existed. | **Gone.** Protocol 2 (`8e8ba41`, `dfa12a5`, `f662dae`, `54ab1da`) removed alignment on the Jetson entirely: it now sends raw unaligned 0.1 mm depth plus a JSON greeting, and the host back-projects through the frame's own geometry (`tasni/core/depth_geometry.py`, `ColorRegistered.build`). Inter-view disagreement now reduces to hand-eye plus pose error. Registration still earns its place — but as a **measurement** of that error, not as a correction for a known systematic. |
| 2 | (silent — the gate did not exist) | **`measure.py:203 depth_plane_check` now rejects every tilted view.** Added after the old spec (`926e5d0`, `1e16ff1`, `e2c7ee7`). See §5.2 — this is the single hard blocker. |
| 3 | "Point counts are capped by the chain's **2 mm** voxel." | The voxel is **1 mm** (`extrusion.voxel_size_m = 0.001`, lowered in `ea9ef7b`). The A/B guidance built on 2 mm is wrong as written. |
| 4 | Task 6 specified a **derived** side pose (`side_pose_from_crest`, `side_view_plan`) and `side_view_*` config keys, default OFF. | **The side photo is already built, differently and better** (`b94593c`, `82a89c3`): `measure.py:443 capture_side_photo` moves to **taught** targets (`SideCapture` / `TowardsSideCapture`) by their **stored joints**, defaults **ON**, and never fails a measurement. The taught-target choice is a hard-won cell lesson, not a shortcut — see §4. That task is superseded, not pending. |

Two further changes shape this design rather than invalidate the old one: the **chroma
gate** became the bead-vs-board discriminator (§5.2), and `_filter_deposit` gained arc
assembly for characterization.

### ChArUco is out of scope, deliberately

An earlier draft of this revamp proposed anchoring the views to the ChArUco board under
the rings. **Rejected by the operator: the board will not always be there, and ChArUco
in this codebase belongs to hand-eye calibration only.** The measurement chain must not
acquire a ChArUco dependency. Registration is ring-first, by the geometry of the ring
itself (§5.3). Do not reintroduce it.

---

## 3. The measured price of tilt

From `characterization/characterization-20260813.json` (`incidence_sweep`, `d* = 310 mm`,
the same cell). These are measured, not interpolated:

| target tilt | measured tilt | standoff | board plane RMS | board plane max | board length err | passed budget |
|---:|---:|---:|---:|---:|---:|:--|
| 0° | 0.99° | 310 mm | **0.650 mm** | 2.59 mm | 0.036 mm | yes |
| 10° | 9.14° | 315 mm | 2.006 mm | 20.01 mm | 0.096 mm | yes |
| 20° | 19.59° | 333 mm | **4.969 mm** | 25.12 mm | 0.447 mm | **no** |
| 30° | 29.42° | 362 mm | 7.430 mm | 24.92 mm | 0.691 mm | **no** |

The standoff grows with tilt in this sweep, but distance is not what is driving the
RMS: the distance-only trials in the same file give 0.934 mm at 310 mm and 0.982 mm at
400 mm. The growth is tilt.

**Two conclusions, both binding on this design:**

1. **The old spec's 20° default is wrong for these rings.** 4.97 mm plane RMS against a
   crest that is 2.9 mm proud in its thin sectors is not a measurement, it is noise with
   a ring-shaped bias. **Default tilt is 15°, capped at 25°.**
2. **The `length_err` column is a systematic warp, and it is removable.** It grows
   0.036 → 0.096 → 0.447 → 0.691 mm — far faster than random noise would. Per-view
   levelling (§5.3 step 1) removes the plane-offset-and-tilt part of exactly this term.
   What survives is why the tilt stays modest.

Rather than defend 15° by argument, §10 sweeps it: because every view is archived raw,
one capture session at 10/15/20° is reprocessed offline and compared with no second cell
run.

---

## 4. What exists today (verified in code, `main @ da5f7a4`)

**Poses.** `inspection.py:129 pose_from_aim(aim, standoff, *, tilt_deg, azimuth_deg,
roll_deg, reference_x)` already places the camera on a cone of half-angle `tilt_deg`
about the surface normal, with the aim point exactly on the optical axis at exactly the
standoff — so tilt trades incidence for reach and never moves centring or distance.
OpenCV convention, `z_axis = -away` where
`away = [sinθ·cosφ, sinθ·sinφ, cosθ]`. **Everything the star needs is already here**;
the star is a *choice of angles*, not new pose maths. `inspection.py:161 pose_candidates`
orders roll → tilt → azimuth as *fallbacks* when the fronto-parallel pose is unreachable;
`inspection.py:191 inspection_plan` publishes descriptors for preview.

**Capture.** `measure.py:255 _move_to_inspection` builds one target + program, starts it,
settles, reselects the inspection TCP and work frame, and reads `T_work_camera` back from
RoboDK. `measure.py:290 _capture_at_pose` grabs one validated RGB-D frame with the arm
already parked. `measure.py:356 _inspect_and_capture` is the two composed.
`measure.py:633 _one_excursion` deliberately separates them so `repeats` frames can share
one trip out.

**Processing.** `processing.py:645 process_observation` runs, in order:
`ColorRegistered.build` (back-project native depth through the frame's greeting geometry)
→ `chroma_gate_mask` → `transform_points(T_work_camera, …)` → work ROI (height band ×
radial band) → optional `floor_profile` → `_filter_deposit` (voxel → statistical →
radius → DBSCAN) → `_radial_trim` → `_top_surface` → skeleton → spline.
`processing.py:914 characterize_ring` is the sibling entry point with `assemble_arcs=True`.

**The two arrival gates.** `measure.py:203 depth_plane_check` (whole-frame median depth
vs camera height above the plane) and `inspection.py:257 standoff_report` /
`inspection.py:290 standoff_fault` (central-patch median vs `|camera − aim|`; used from
`service.py:1005`).

**The chroma gate.** `processing.py:64 chroma_gate_mask` projects registered depth points
into the colour image and keeps the chromatic ones (`deposit_min_saturation = 60`); when
the colour frame carries less than `deposit_min_chroma_fraction` (0.005) chromatic
content it **abstains**. `processing.py:48 deposit_floor_mm` then picks the floor:
1.5 mm when the gate held, **2.5 mm when it abstained**. That floor is not cosmetic —
the config comment records that at 2.5 mm on the 2026-08-29 cell run a 45° sector reading
2.9–4.9 mm fell out of the deposit cluster and **all four takes came back invalid**.

**Side photo (already shipped).** `measure.py:427 side_capture_requirements`,
`measure.py:443 capture_side_photo`, config `side_capture_target = "SideCapture"`,
`side_capture_approach_target = "TowardsSideCapture"`, `side_capture_enabled = True`,
`side_capture_settle_s = 0.6`; per-request override `side_photo: bool | None` on
`module.py MeasureLayerBody`; `SideViewRecord` and `LayerManifest.side_view` in
`models.py`. It moves via the **taught** approach target in both directions, by
**stored joints**, and never raises.

> **Do not "improve" the side photo into a derived pose.** Two cell facts are encoded
> here. (a) The approach target exists because the direct joint move between neutral and
> the side pose sweeps the arm through things standing in the cell that the station model
> does not contain. (b) `move_j(name)` on a *cartesian* target resolves against the tool
> and frame active *now*, not the ones it was taught with — and since the last take
> leaves `Realsense` + the work frame selected, that sent the arm somewhere wrong on
> 2026-08-29 (137.8 s excursion against 2.7 s for an inspection move). Stored joints have
> no such dependency.

**Archive.** `archive.py:59 write_layer` writes `color.png`, `depth.npy`,
`nominal_path.json`, `commanded_path.json`, `measured_path.json`, the point cloud, the
derived images and `manifest.json` into one take directory.
`service.py:1134 reprocess_saved_layer` rebuilds derived artifacts from
`layer_dir/color.png` + `depth.npy` + `provenance.T_work_camera`.

**Statistics.** `measure.py:993 capture_style` already refuses to conflate takes that
shared one trip with takes that each re-approached — the precedent this design extends
rather than duplicates.

---

## 5. Design

### 5.1 The star (poses)

Four views per take:

| name | tilt | azimuth (work frame, from +X) |
|---|---|---|
| `top` | 0° | — |
| `star-000` | `multiview_tilt_deg` (15°) | 0° |
| `star-120` | `multiview_tilt_deg` | 120° |
| `star-240` | `multiview_tilt_deg` | 240° |

All four share the **same aim point and the same standoff**, so the ring fills the frame
identically from every side and no view is nearer or better-framed than another.

Azimuth is measured **in the work frame from +X** — the same axis the paired-detection
offset is expressed along — so a star has a reproducible orientation across takes and
sessions. Roll is measured from the camera-as-parked axis, exactly as the top view does
today (`_roll_reference_axis(reference_x=…)`); getting this wrong costs a wrist flip.

New pure functions in `inspection.py`, alongside the existing ones:

- `star_view_angles(config) -> list[tuple[str, float, float]]` — the (name, tilt,
  azimuth) table above, from config.
- `star_view_candidates(aim_mm, standoff_mm, config, *, tilt_deg, azimuth_deg,
  reference_x)` — per-view ordered candidates, reusing `pose_from_aim`. A view's
  candidate list varies **roll only**; tilt and azimuth are what the view *is* and must
  not be silently substituted by a fallback. A view with no reachable candidate is
  **dropped** (§8), never quietly replaced by a different geometry.
- `multiview_plan(recipe, setup, *, K, size_px, config)` — descriptors for preview and
  for the dry tour, mirroring `inspection_plan`.

### 5.2 Capture, and the gates that reject tilted frames

**One excursion per view.** The arm must physically move between azimuths, so a star take
is four moves: `<stem>_Inspect` (top, unchanged) then `<stem>_Inspect_star000` /
`_star120` / `_star240`. `_inspect_and_capture` is called once per view. Each view's
frame, pose and greeting geometry are archived independently *before* anything is merged,
so a merge that goes wrong is always recoverable offline.

**How `repeats` composes with the star.** Today `repeats` does **not** average anything:
`_one_excursion` grabs `repeats` frames with the arm parked and each one becomes its own
**take** — its own directory, manifest and result — so their spread measures the sensing
chain alone with the robot's re-approach excluded by construction. That meaning must
survive.

So a star take grabs `repeats` frames **at each pose while parked there**, and take *k*
is assembled from the *k*-th frame of every view. `repeats = 3` with multi-view ON
therefore yields **3 independent 3-view-plus-top takes from 4 moves**, not 12 moves —
the arm still visits each pose exactly once. Per-view frames are never averaged together;
frame *k* belongs to take *k*, which is what keeps the repeat spread an honest
repeatability number rather than a smoothed one.

#### Gate 1: `depth_plane_check` — the blocker

Today (`measure.py:203`):

```
camera_z = T_work_camera[2, 3]
accepted = [camera_z - characterize_max_height_mm(40), camera_z + depth_plane_slack_mm(15)]
agrees   = accepted[0] <= median(valid depth) * unit_mm <= accepted[1]
```

The docstring is explicit that this holds "looking straight down at the work plane",
where the frame's median depth *is* the camera's height above the plane. Off-axis the
two quantities separate: the camera drops to `aim_z + standoff·cos θ` while the median
depth stays at roughly the standoff, so the median runs **above** `camera_z` by
`standoff·(1 − cos θ)`. The gate's high side has only `depth_plane_slack_mm = 15` mm of
budget, and tilt spends it.

Computed against the real constants (`aim_z ≈ 5` mm, ceiling 40, slack 15) — the signed
column is how much of the 15 mm high-side budget the tilt alone consumes:

| standoff | 10° | 15° | 18° | 20° | 25° | 30° |
|---:|---:|---:|---:|---:|---:|---:|
| 300 mm | pass (−0.4) | pass (**+5.2**) | pass (**+9.7**) | pass (**+13.1**) | **FAIL** (+23.1) | **FAIL** (+35.2) |
| 400 mm | pass (+1.1) | pass (**+8.6**) | pass (**+14.6**) | **FAIL** (+19.1) | **FAIL** (+32.5) | **FAIL** (+48.6) |
| 500 mm | pass (+2.6) | pass (**+12.0**) | **FAIL** (+19.5) | **FAIL** (+25.2) | **FAIL** (+41.8) | **FAIL** (+62.0) |

Three things follow, and all three argue for the same fix:

1. **The 25° cap fails everywhere, and 20° fails at 400 mm and beyond.** Those views are
   discarded, `depth_stale_retries` burns its retries, and the take dies with a message
   blaming a frozen depth stream or a wrong work frame — none of which is what happened.
2. **Even where it passes, the gate goes nearly blind.** At the 15° default a tilted view
   spends 5–12 mm of a 15 mm budget on pure geometry, leaving 3–10 mm to catch the actual
   fault this gate exists for. A stalled stream returning a plausible-but-wrong depth
   would now slip through on exactly the views we trust least.
3. **The threshold moves with the standoff**, which is derived per ring from
   `framing_standoff`. So the same tilt silently works on a small ring and fails on a
   large one. That is the worst failure mode available here: geometry-dependent,
   intermittent, and misattributed to the camera.

**Fix — derive the incidence from the pose, add no knob:**

```
cos_incidence = -T_work_camera[2, 2]          # exactly cos(tilt) for pose_from_aim
expected      = camera_z / cos_incidence
accepted      = [expected - ceiling / cos_incidence, expected + slack / cos_incidence]
```

Because `pose_from_aim` sets `z_axis = -away` with `away_z = cos(tilt)`, `-T[2,2]` **is**
`cos(tilt)` — no convention ambiguity, and nothing to pass in that could disagree with
where the arm actually went. At tilt 0, `cos_incidence = 1` and every expression above
collapses to today's, **byte-for-byte**; the existing single-view tests are the
regression and must stay green untouched.

The band widens by the same factor because a tilted view of a plane spreads depth across
the frame, and the same `characterize_max_height_mm` deposit allowance is measured along
a longer ray. Guard `cos_incidence` against a degenerate/near-horizontal pose (refuse
below `multiview_min_cos_incidence`, 0.5 → 60°) rather than dividing by something near
zero.

**Verified before writing this.** Sweeping the corrected formula over standoffs
300/400/500/800 mm × tilts 0–30°, the gap between `expected` and the true median stays at
**+5.0 to +5.8 mm** everywhere — that is the `aim_z` term, and today's tilt-0 gate already
carries exactly +5.0 of it. So the correction does not merely make tilted views pass: it
keeps the gate's *sensitivity to a real fault* essentially constant across the whole
working envelope, instead of letting it decay with tilt and standoff as it does now. The
small residual is absorbed by the 40 mm ceiling side, as it already is today.

#### Gate 2: `standoff_report` — already correct

It medians a **central patch** and compares against `|camera − aim|`. Since
`pose_from_aim` guarantees the aim point sits on the optical axis at exactly the
standoff for every tilt, this is tilt-invariant by construction. **No change.** It is
worth running per view and archiving the result: `delta_mm` per view is an independent
read on whether the arm reached each star pose.

#### The chroma gate needs a per-view rule

The gate is per-frame by nature (it needs *that* view's colour and *that* view's
registration), but the deposit floor it selects is global to the cloud the ROI is
applied to. Since the merge happens before the ROI, **one abstaining view would drag the
whole merged cloud to the 2.5 mm floor** and reproduce the 2026-08-29 all-takes-invalid
failure.

**Rule: gate each view against its own colour frame; a view whose gate abstains is
dropped from the merge, with its `chroma_fraction` and the reason recorded.** Because
abstainers are dropped, every *contributing* view is gated by construction and the merged
cloud always earns the 1.5 mm floor. If so many abstain that fewer than
`multiview_min_views` survive, the take falls back to top-only (§8) and the floor is
whatever that single frame earns — precisely today's behaviour.

### 5.3 Levelling and registration — ring-first, no fiducial

**Why not ICP.** The ring is a torus. Sliding one view tangentially around it costs
almost nothing in point-to-point distance, so ICP has a nearly-free degree of freedom it
will fill with whatever the noise suggests. When a shape has a degenerate DOF, the right
move is to stop registering *points* and start fitting the *model the object has* — here,
a circle.

**Why not anchor to the top view.** Translating each tilted view so its own fitted ring
centre lands on the top view's centre makes the merged centre, by construction, the top
view's centre. The merged cloud would then be the measurement of record while the
headline number (centre offset, paired detection error) was still a single-view answer.
That is circular and this design rejects it.

**Step 1 — level each view on the surface the ring rests on.** Fit a plane (RANSAC) to
the points in an **annulus outside the radial ROI band** — i.e. surface, not deposit —
and apply the rigid transform that takes that plane to `z = 0`. This is the surface the
ring sits on, present whether or not anything is printed on it, and it removes the
plane-offset-and-tilt part of the systematic warp that §3's `length_err` column measures.
A view whose annulus yields too few inliers, or whose fitted plane sits more than
`multiview_max_level_mm` from `z = 0`, is dropped (§8).

**Step 2 — solve every view against ONE shared circle.** The merge happens before the
ROI, so at this point a view holds its whole chroma-gated cloud, not a deposit cluster.
For **fitting purposes only**, take the subset inside the chain's own height × radial
band (the same `min_z`/`max_z`, `r_lo`/`r_hi` computed at `processing.py:645`) and fit a
circle to that. Nothing is discarded by the fit: the offsets it solves are applied to the
view's **full** gated cloud, which is what gets concatenated. Fit each view independently
to seed, then solve jointly:

- **Unknowns:** a lateral offset `(dx_i, dy_i)` per view, plus one shared centre
  `(cx, cy)` and one shared radius `r`.
- **Residual:** for each point `p` of view `i`, `| ‖p_xy + d_i − c‖ − r |`.
- **One shared radius, not one per view.** A per-view radius would let a view carrying
  residual scale error absorb it silently; a shared radius forces it to surface as a
  large residual instead.
- **Gauge fix:** `Σ d_i = 0`. Without it the problem is translation-degenerate. With it,
  no view is privileged and the merged centre is the **consensus of all views**, not the
  top view's answer wearing a four-view costume.
- Robustify with a soft-L1 / Huber loss so one bad arc cannot drag the solution.

The solved `d_i` **are** the inter-view disagreement, and they get archived. Also archive
the **pre-correction** spread (each view's independently fitted centre against their mean)
— that is the raw measurement of the cell's hand-eye + pose error at the ring, and it is
what makes the A/B interpretable rather than merely favourable. It is an independent read
on the 1.26 mm board-consistency figure the paper already cites.

**A common bias is invisible to this, and that is fine.** If all views are wrong the same
way, the consensus cannot see it — but that is the same bias the single view already
carries, so the merge makes nothing worse. Say so in the paper rather than implying the
merge removes it.

**Z is handled by levelling only.** No per-view Z offset is solved: step 1 already put
every view's surface at `z = 0`, and a free Z per view would let the flanks slide
vertically against each other, which is exactly the profile we are trying to measure.

### 5.4 Where the merge sits in the chain

Right after the work-frame transform and **before** the ROI:

```
per view:  ColorRegistered.build -> chroma_gate_mask -> transform_points   [5.2]
           -> level on the annulus                                          [5.3 step 1]
merged:    joint circle solve, apply d_i, concatenate                       [5.3 step 2]
then:      work ROI -> floor_profile -> _filter_deposit -> _radial_trim
           -> _top_surface -> skeleton -> spline                            [UNCHANGED]
```

One ROI, one voxel, one DBSCAN, one crest extraction, over one cloud. Everything
downstream of the merge is the cell-validated chain, untouched.

This requires splitting `process_observation` at that seam — the old spec's Task 3, still
the right seam, into a bigger function than it was:

- `observation_points(*, color, depth, geometry, T_work_camera, K, dist, config, counts)
  -> (points_work_mm, chroma_diag)` — back-project, gate, transform.
- `process_points(points, *, plan, layer, config, floor_profile, stages, assemble_arcs,
  chroma_gated) -> ProcessingResult` — everything from the ROI onward.
- `process_observation` and `characterize_ring` become thin wrappers over the two.

The refactor must be **behaviour-preserving**: the existing
`tests/test_extrusion_processing.py` and the archived-fixture tests are the proof, and
they change by not one line.

### 5.5 Archive and manifest

**The take root does not move.** `color.png` and `depth.npy` at the take root stay the
**top** view. That single decision keeps `reprocess_saved_layer`, `figures.py`, the
archived `tests/fixtures/extrusion/ring*/` fixtures and every existing reader working
with no change at all.

The extra views land beside them:

```
<take>/
  color.png  depth.npy            <- top view, exactly as today
  views/
    top/         pose.json                       (frame is the take root's)
    star-000/    color.png  depth.npy  pose.json
    star-120/    color.png  depth.npy  pose.json
    star-240/    color.png  depth.npy  pose.json
  merged_points.npy               <- the merged work-frame cloud
  manifest.json
```

`pose.json` per view carries `T_work_camera`, the requested and achieved tilt/azimuth/roll,
the `standoff_report`, and the frame's own `camera_geometry` greeting dict. Directory
names keep the hyphen (`star-120`); the RoboDK item name drops it (`_star120`).

New typed records in `models.py` (all defaulted — `LayerManifest` is `extra="forbid"`
and every field must be optional so old archives still validate):

- `ViewRecord`: `name`, `tilt_deg`, `azimuth_deg`, `roll_deg`, `T_work_camera`,
  `standoff_delta_mm`, `chroma_fraction`, `chroma_gated`, `points_before_merge`,
  `fitted_center_mm`, `solved_offset_mm`, `residual_rms_mm`, `dropped` (bool),
  `drop_reason` (str | None).
- `CaptureRecord`: `style` (`"single"` | `"star"`), `views: list[ViewRecord]`,
  `consensus_center_mm`, `consensus_radius_mm`, `spread_before_mm` (pre-correction),
  `residual_after_mm` (post-correction), `merged_points_file`, `timings_ms` per view
  and total.
- `LayerManifest.capture: CaptureRecord | None`.

**Timings.** `acquisition_to_path_ms` is the paper's number #3 and its meaning changes
under a star. The manifest records per-view `move_ms` / `settle_ms` / `capture_ms`, a
`views_total_ms`, and the merge cost separately, so a reader can always recover both "what
one frame cost" and "what this take cost".

### 5.6 Reprocess, and why the A/B is free

`reprocess_saved_layer(root, trial_id, layer_index, take, *, views="as_archived")` gains a
`views` switch:

- `"as_archived"` — merge whatever `views/` holds (default; a single-view take is
  unaffected).
- `"top_only"` — rebuild from the top view alone, ignoring `views/`.

**Every star take is therefore its own control.** One arm-time cost, two reconstructions,
no cell re-run to compare them, and the comparison is paired on the *identical* physical
ring placement — which is far stronger evidence than two separate captures could ever be,
because the operator cannot re-place a ring exactly.

The same mechanism sweeps the tilt: capture once at 10/15/20°, reprocess and compare
offline.

### 5.7 Figures and the paper summary

- `figures.py` gains a `views` figure: the four colour frames with each view's fitted
  ring and solved offset drawn on, plus the merged cloud. It renders from the archive
  with no robot, like every other figure.
- The existing per-take figures read the merged cloud when there is one and the top view
  otherwise; nothing else about them changes.
- `capture_style()` is **extended, not duplicated**, to report `"star"` alongside
  `"parked"` / `"re-approach"` / `"single"`, and `paper_summary()` must **refuse to pool
  merged takes with single-view takes** in any statistic — the same discipline it already
  applies to parked-vs-re-approach and to offline-reprocessed takes.

### 5.8 API and UI — two independent toggles

**Multi-view** mirrors the side photo's existing shape exactly:

- config `extrusion.multiview_enabled: bool = False`
- per-request `multiview: bool | None = None` on `MeasureLayerBody` **and**
  `CharacterizeBody` (`None` = follow config).

**Side photo** keeps everything it has (`side_capture_enabled = True`,
`side_photo: bool | None`). **Nothing couples the two**: merged with no photo, single-view
with a photo, both, or neither are all reachable.

**The UI gap to close.** `side_photo` exists in the API but appears nowhere in
`tasni/webui/src/pages/Extrusion.tsx` — it only ever runs on its config default. Both
toggles get real controls on the measure card, side by side, each showing what it costs
(multi-view: "4 trips instead of 1"; side photo: "one extra excursion after the capture").

### 5.9 Configuration (all new keys defaulted; `ExtrusionConfig` is `extra="forbid"`)

| key | default | why |
|---|---|---|
| `multiview_enabled` | `False` | Opt-in. The single-view chain is the validated one. |
| `multiview_tilt_deg` | `15.0` | §3: 10° costs 2.0 mm plane RMS, 20° costs 4.97 mm. |
| `multiview_max_tilt_deg` | `25.0` | Hard cap; 30° measured 7.43 mm. |
| `multiview_azimuths_deg` | `[0, 120, 240]` | Minimum that puts every flank within 60° of a camera. |
| `multiview_min_cos_incidence` | `0.5` | Refuse a near-horizontal pose rather than divide by ~0 in §5.2. |
| `multiview_level_annulus_width_mm` | `60.0` | The levelling annulus runs from the OUTER edge of the chain's radial ROI band (`recipe.radius_mm + radial_roi_margin_mm`) outward by this width — surface, never deposit. |
| `multiview_level_min_points` | `500` | Below this the plane fit is not trustworthy → drop the view. |
| `multiview_max_level_mm` | `10.0` | A fitted surface further than this from `z=0` means the view is wrong, not tilted. |
| `multiview_min_view_points` | `200` | Too little ring to fit → drop the view. |
| `multiview_min_arc_deg` | `90.0` | A view seeing less arc than this cannot constrain a centre. |
| `multiview_max_offset_mm` | `5.0` | A solved offset beyond this is a failed registration → drop the view. |
| `multiview_min_views` | `2` | Below this, fall back to top-only rather than "merge" one cloud. |

---

## 6. Operator-facing behaviour

1. Two checkboxes on the measure card: **Multi-view capture** (off by default) and
   **Side photo** (on by default). Each says what it costs.
2. With multi-view on, a Measure or Characterize press moves the camera four times
   instead of once — about 12–15 s of arm time rather than about 3 s — and the progress
   log names each view as it goes.
3. Any view that cannot be reached, cannot see enough ring, or whose colour gate abstains
   is announced in the log and dropped. **The take still completes**, on the views that
   remain.
4. If only the top view survives, the result is exactly today's single-view result and
   the log says so plainly.
5. The archive gains a `views/` directory and a `views` figure. Everything already on
   disk is untouched and still reprocesses.

---

## 7. Non-goals

- **No ChArUco.** §2. Not as registration, not as an advisory check, not at all.
- **No ICP or feature-based cloud registration.** §5.3.
- **No change to the live print.** §11.
- **No *behavioural* change to the single-view path.** Task 3 edits `depth_plane_check`
  and task 4 refactors `process_observation`, both on the shared path — but both must
  reduce to today's behaviour exactly, and the existing tests are the proof.
  `multiview=False` must produce what it produces today.
- **No new capture protocol on the Jetson.** This is entirely host-side; `server/` is not
  touched, so nothing here can self-deploy to the camera.
- **No derived side-photo pose.** §4.
- **Not a lateral-resolution improvement.** §1.

---

## 8. Error handling

Every row **drops a view and continues**; none fails the take. Every drop is recorded in
the `ViewRecord` with its reason and surfaced in the job log.

| condition | detected where | behaviour |
|---|---|---|
| No reachable/collision-free candidate for a view | `star_view_candidates` + the existing screening | Drop the view. Never substitute a different tilt/azimuth. |
| Program refuses to start, or the move times out | `_move_to_inspection` | Drop the view, stop the program, continue to the next. |
| `depth_plane_check` fails after retries | `_capture_at_pose` | Drop the view. (For the **top** view this still fails the take, exactly as today.) |
| `standoff_fault` beyond tolerance | per-view `standoff_report` | Drop the view; record `delta_mm`. |
| Chroma gate abstains | `chroma_gate_mask` per view | Drop the view (§5.2). Never let it move the merged floor. |
| Plane fit fails / too few annulus inliers / level > `multiview_max_level_mm` | §5.3 step 1 | Drop the view. |
| Too few points, or arc < `multiview_min_arc_deg` | §5.3 step 2 seed | Drop the view. |
| Solved offset > `multiview_max_offset_mm` | §5.3 step 2 | Drop the view, re-solve without it. |
| Fewer than `multiview_min_views` survive | merge entry | Fall back to **top-only**; mark `style="single"` with a warning naming what was lost. |
| Top view itself fails | as today | The take fails, raw frames archived, exactly as today. |
| Cancel mid-star | `ctx.check_cancel()` between views | Return the arm home; archive what was captured. |

**The arm always comes home.** `_one_excursion`'s `finally` already guarantees the return
and the artifact cleanup; the star adds three more programs to the same cleanup list and
must not add a second return path.

---

## 9. Testing (proof before the cell)

RoboDK is never touched by tests — reuse the fakes in `tests/test_extrusion_job.py`.
Synthetic RGB-D comes from `tests/extrusion_synthetic.py`, whose `render_scene` already
renders from **any** camera pose, so multi-view scenes need no new renderer.

New `tests/test_extrusion_multiview.py`:

1. **Pose table** — `star_view_angles` returns the configured tilt at 0/120/240°;
   `star_view_candidates` varies roll only, never tilt or azimuth.
2. **Gate reduction** — `depth_plane_check` on a tilt-0 pose reproduces values pinned
   from the current implementation *before* the change (capture them first, assert them
   after). This is the byte-for-byte claim, and it is what protects the single-view path.
3. **Gate at tilt** — a synthetic 25° view of a plane at a 300 mm standoff passes the new
   gate and *fails* the old one; and the standoff dependence is asserted directly (20° at
   400 mm fails the old gate, passes the new). This is the regression that proves the
   blocker is fixed and that it was standoff-dependent, not a fixed angle.
4. **Degenerate incidence** — a pose below `multiview_min_cos_incidence` is refused, not
   divided by.
5. **Levelling** — a synthetic view with a known injected plane tilt is levelled to
   `z = 0` within tolerance.
6. **Joint solve recovers injected offsets** — inject known `(dx, dy)` per view; assert
   the solved offsets recover them to sub-0.1 mm and that `Σ d_i = 0` holds.
7. **Gauge is a consensus, not an anchor** — displacing the *top* view moves the
   consensus centre by `1/n` of the displacement, not by zero. This is the test that the
   circularity of the old spec is actually gone.
8. **Shared radius surfaces scale error** — a view with a scaled ring produces a large
   residual rather than a quietly-absorbed fit.
9. **Chroma abstention drops the view** and does *not* move the merged floor; all-abstain
   does move it.
10. **Every drop reason** in §8 maps to a `ViewRecord` with that reason, and the take
    still completes.
11. **`min_views` fallback** produces a genuinely top-only result equal to the
    single-view path's.
12. **Seam is behaviour-preserving** — `process_observation` before and after the refactor
    produce identical `ProcessingResult` on the archived fixture.
13. **Manifest round-trips** — old manifests without `capture` still validate.
14. **`reprocess(views="top_only")`** on a star take equals the single-view result.

Tests reaching `_filter_deposit` (anything through `process_points`, the circle solve on
real clusters, `characterize_*`) need Open3D: guard with
`pytest.importorskip("open3d")`. Levelling, the pose functions, the gate maths and the
archive are pure numpy — no skip.

Targeted command (**do not run the full suite** — it is too slow and the operator
interrupts it):

```
py -3.10 -m pytest tests/test_extrusion_multiview.py tests/test_extrusion_measure.py \
  tests/test_extrusion.py tests/test_extrusion_job.py tests/test_extrusion_processing.py \
  tests/test_extrusion_figures.py tests/test_extrusion_standoff.py -q
```

---

## 10. Cell protocol: the A/B that decides whether merged takes go in the paper

Run **after** the PFH paper's single-view cell run is complete.

1. Place one ring. Do not move it for the whole protocol.
2. Measure with multi-view ON, `repeats = 3`, at `multiview_tilt_deg` = 10, then 15, then
   20. Nine takes, one ring placement, no re-placement between them.
3. Reprocess every take **both ways** (`views="as_archived"` and `views="top_only"`).
   This is the paired comparison; no further cell time is needed.

**Read these, in this order:**

| question | metric | where |
|---|---|---|
| Did the profile improve? | bead width profile spread, crest height range | `geometry` (`RingGeometry`) |
| Did coverage improve? | completeness, max angular gap | `metrics` |
| Was the merge trustworthy? | pre-correction inter-view spread, post-correction residual | `capture.spread_before_mm`, `capture.residual_after_mm` |
| Did repeatability improve? | centre spread across the 3 repeats | `centre_spread()` |
| Which tilt wins? | all of the above across 10/15/20° | the sweep |

**The count trap, stated loudly.** `_deposit_clusters` voxel-downsamples at **1 mm**
(`voxel_size_m = 0.001`). Merging four views does **not** multiply the surviving point
count — it multiplies the samples each voxel averages and it fills dropouts. Reading
`after_voxel` and concluding "no gain" is reading the wrong number. **The pre-voxel
`counts["after_work_roi"]` is the one that moves**, together with validity, completeness
and the profile metrics above.

**Decide:**

- Profile metrics improve and the post-correction residual is below the cell's own
  hand-eye floor (1.26 mm board consistency) → merged takes may go in the paper, with the
  residual reported alongside them, and the tilt that won becomes the default.
- Profile improves but the residual is at or above that floor → report the improvement as
  qualitative (a better-sampled cross-section), do **not** claim improved accuracy, and
  keep the default OFF.
- No improvement → keep it OFF, keep the code (it is opt-in and costs nothing switched
  off), and write the negative result down. A measured negative result about view
  geometry is worth having.

**The error floor still binds.** Hand-eye verdict `borderline`, board consistency
1.26 mm, work-plane RMS 1.39 mm. Multi-view does not lower it, and nothing here may claim
sub-millimetre accuracy on top of it.

---

## 11. Later: the live print

`CylinderPrintJob` / `CylinderDryRunJob` are cell-validated and **this design does not
touch them**. Once the A/B settles the geometry, the live print can reuse
`capture_views` unchanged — but that is a separate design, because a print's inspection
happens between layers with material curing, and four excursions per layer is a different
cost question from four per measurement.

---

## 12. Implementation tasks (for `writing-plans`)

Plan: `docs/superpowers/plans/2026-08-30-multiview-inspection.md`.

1. Config keys and manifest records (§5.9, §5.5) — pure data, no behaviour.
2. Star poses in `inspection.py` (§5.1) — pure numpy.
3. Tilt-aware `depth_plane_check` (§5.2) — the blocker, with the byte-for-byte reduction
   test. **Independently useful and independently mergeable.**
4. The processing seam: `observation_points` / `process_points` (§5.4) —
   behaviour-preserving refactor, proved by the untouched existing tests.
5. `multiview.py`: levelling, the joint circle solve, merge, diagnostics (§5.3).
6. Capture the views: `capture_views`, archive layout, job wiring, timings (§5.2, §5.5).
7. Reprocess `views=` + the offline A/B tool (§5.6).
8. Figures, `capture_style`, `paper_summary` guards (§5.7).
9. API + both UI toggles, including the side photo's missing control (§5.8).
10. Docs: `docs/pfh-paper-handoff.md`, `docs/extrusion-current-handoff.md`, `AGENTS.md`.

Tasks 1–6 are the capture-and-merge core. Tasks 7–10 depend on nothing in the A/B and can
follow immediately; only the **default flip** in §10 waits for cell evidence.
