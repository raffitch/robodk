---
name: multiview-inspection-spec
description: "REVAMPED 2026-08-30: design spec + 10-task plan for optional multi-view (top + 3 tilted 15deg at 120deg) merged ring capture. The 2026-08-29 pair is RETIRED (4 things invalidated it). Registration is ring-first (joint fit to ONE shared circle, gauge-fixed) - ChArUco is BANNED here by the operator. BUILT after all -- 17 commits on origin/worktree-multiview-inspection @ 96a17f6, never merged, never run on the cell."
metadata: 
  node_type: memory
  type: project
  modified: 2026-08-30T07:39:34.978Z
  originSessionId: a40f845f-b0a2-408f-99d3-853c7191c680
---

Spec: `docs/superpowers/specs/2026-08-30-multiview-inspection-design.md` (`95485bd`).
Plan (all 10 tasks): `docs/superpowers/plans/2026-08-30-multiview-inspection.md`
(`bfeb952`). Both pushed to `main`. The **2026-08-29 pair is deleted** (`a1cafa0` /
`c31c720` in history) — do not resurrect it.

**Why the old one died (4 things):** (1) protocol 2 removed the ≈2 % lateral scale
mismatch its whole registration existed to correct — the Jetson no longer aligns, the host
back-projects through the frame's own greeting geometry; (2) `depth_plane_check` did not
exist then and it rejects tilted frames; (3) the chain voxel moved 2 mm → **1 mm**, which
its A/B guidance was built on; (4) the side photo **shipped** meanwhile with *taught*
targets (`SideCapture`/`TowardsSideCapture`, stored joints, default ON), not the derived
pose it specified.

**Operator decisions, binding:**
- **ChArUco is ONLY for hand-eye calibration.** The board will not always be under the
  rings. It is out of this design entirely — not registration, not an advisory check.
- The merged cloud **replaces** the single frame end-to-end (chosen over profile-only),
  which is why the registration must not anchor to the top view.
- Two independent toggles: multi-view (default OFF) and side photo (default ON).
- Goal in the operator's words: improve the **resolution and profile** of the ring, *if it
  helps* — so the A/B decides, and a negative result gets written down.

**Design decisions to keep:** same aim point + standoff for every view; per-view levelling
on the surface annulus **outside** the radial ROI band; then a **gauge-fixed joint solve**
(`Σ offsets = 0`) of per-view XY offsets against **ONE shared centre and ONE shared
radius** — shared radius so a view with scale error surfaces as residual instead of
absorbing it; ICP rejected (torus tangential DOF is free). Merge sits after the work-frame
transform and **before** the ROI, so the unchanged chain runs once. Top view stays
`color.png`/`depth.npy` at the take root so reprocess/figures/fixtures keep working.

**Four findings worth more than the doc (verified, not guessed):**
1. **`depth_plane_check` is standoff-dependent, and it is a latent bug on the EXISTING
   single-view path.** Median depth runs above `camera_z` by `standoff·(1−cos θ)` against a
   15 mm one-sided budget → fails above ~18° at 300 mm, ~14° at 500 mm, and even at 15° it
   spends 5–12 mm of the budget on geometry. Fix reads incidence off the pose
   (`cos = -T[2,2]`, exactly `cos(tilt)` for `pose_from_aim`), tilt 0 reduces byte-for-byte.
   Swept 300–800 mm × 0–30°: residual bias holds at 5.00–5.77 mm, so sensitivity stays flat.
   **Task 3 is independently mergeable and should land even if the rest is dropped.**
2. **Tilt is expensive and it is measured**, in `characterization/characterization-20260813.json`
   `incidence_sweep`: board plane RMS 0.650 mm @1°, 2.006 @9.1°, **4.969 @19.6°**, 7.430
   @29.4°. The old 20° default was noise against a 2.9 mm crest → **default is 15°, cap 25°**.
   `length_err` grows 0.036→0.447 mm over the same range = the systematic warp levelling removes.
3. **Synthetic fixtures pass an ALL-ZERO colour frame**, so `chroma_gate_mask` abstains.
   Under the per-view drop rule that drops every view in every test — green and worthless.
   The plan adds `render_color` to `tests/extrusion_synthetic.py` FIRST.
4. **Multi-view pays on the PROFILE, not the centreline.** `_top_surface` keeps only
   normals within ~23° of vertical, so the tilted views' unique contribution is discarded
   for the centreline; `bead_width_profile`/`ring_geometry` read the flanks *before* that.
   Also: sensor tuning is a dead end — `docs/realsense-quality-headroom-2026-08-30.md`
   found no headroom in laser/preset/resolution/disparity shift. Geometry is the lever.

**IT WAS BUILT (discovered 2026-09-01).** The "docs only" line above was wrong.
`origin/worktree-multiview-inspection` @ **`96a17f6`** (worktree at
`.claude/worktrees/multiview-inspection`) carries the whole 10-task plan executed:
**17 commits, +3659 lines** — `tasni/modules/extrusion/multiview.py` (311),
`tests/test_extrusion_multiview.py` (1356), `tools/multiview_ab.py` (offline merged-vs-
top-only A/B from archived frames, no robot time), the UI toggles, and a written cell A/B
protocol in the branch's `docs/extrusion-current-handoff.md`. **Never merged, never run on
the cell.** Do not rebuild any of it.

What has gone stale on it: 45 commits behind main, predating the halved voxel (`a4015e1`),
the radial-trim fixed point (`50d0b34`) and the layer-N deposit floor (`bd455a7`) — so its
"count trap" note saying the chain voxels at **1 mm** is wrong (main is 0.5 mm) and that
arithmetic needs redoing. Finding 1 above (`depth_plane_check` incidence) is FIXED ON THE
BRANCH and still absent from main, so **main's arrival gate refuses any frame tilted past
~18° at 300 mm** — cherry-pick it before attempting any tilted capture, whatever else
happens. Known disclosed gap: `RingCharacterizeJob` takes `multiview` but does not act on
it (characterize's capture path never reaches the merge seam).

**Test the premise before merging any of it:** the case for multi-view rests on the STATIC
depth noise (1.176 mm spatial vs 0.130 temporal) decorrelating with pose, which has never
been measured. [[roll-probe-camera-vs-scene]] is the one-excursion test.

**How to apply:** the PFH paper is no longer a sequencing constraint
([[paper-timeline-not-a-constraint]]), but the `acquisition_to_path_ms` hazard it named is
real on its own terms: it edits the shared capture path and redefines
`acquisition_to_path_ms`, so every take captured after it means something different by that
number. The work is NOT "execute the plan" any more — it is rebase the branch onto main,
redo the voxel arithmetic, then run the branch's own cell A/B protocol. Stage paths
explicitly — never `git add -A` (another session was editing this repo concurrently on
2026-08-30). Restart the backend before any cell test.
See [[pfh-paper-ring-stack-experiment]], [[cell-characterization-2026-08-13]],
[[extrusion-take-figures]].
