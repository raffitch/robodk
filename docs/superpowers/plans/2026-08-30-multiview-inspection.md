# Multi-view ring inspection — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a Measure or Characterize press capture the ring from the top view plus three 15°-tilted views at 120° azimuths, level and register them against one shared circle, and measure the merged cloud — because the mock rings are thin (crest 2.9–4.9 mm over much of the ring) and one straight-down frame sees the flanks hardly at all. Opt-in, default OFF, so every archived single-view number stays exactly as it is.

**Architecture:** `inspection.py` gains the star angles (pure numpy, reusing the existing `pose_from_aim` cone). `measure.py`'s `depth_plane_check` becomes incidence-aware by reading the pose, which is what unblocks tilted frames at all. `processing.py` splits at the back-projection seam (`observation_points` → `process_points`), behaviour-preserving, so a new `multiview.py` can level each view on its board annulus, solve per-view XY offsets against one shared circle with the gauge fixed, concatenate, and hand the merged work-frame cloud to the *unchanged* chain. `measure.py` captures the views in order (one RoboDK target + program per view, as today) and archives every raw frame under `views/`.

**Tech Stack:** Python 3.10 (`py -3.10`), pydantic v2, numpy/scipy/OpenCV, Open3D (the chain's deposit filter; lazily imported), matplotlib (`figures` extra), React + Vite + TypeScript for the UI. RoboDK is never touched by tests — fakes from `tests/test_extrusion_job.py`. Synthetic RGB-D from `tests/extrusion_synthetic.py` (`render_scene` renders from **any** camera pose, so multi-view scenes need no new geometry renderer — but they DO need a new colour renderer, Task 5 Step 1).

**Spec:** `docs/superpowers/specs/2026-08-30-multiview-inspection-design.md` — read §1 (why), §2 (what the previous spec got wrong and why it was retired), §3 (the measured price of tilt), §4 (code facts), §5 (design), §8 (error table), §9 (tests) before starting. This plan implements all ten of the spec's §12 tasks.

---

## Global Constraints

- **Do not start before the PFH paper's cell run is finished.** `docs/pfh-paper-handoff.md` carries a **1 September 2026** deadline and still needs numbers #2, #2b and #3 measured on the single-view chain. This plan edits the shared capture path and redefines `acquisition_to_path_ms`, which is number #3.
- Work in a **git worktree on branch `multiview-inspection`** (`git worktree add ../RoboDkClaude-multiview -b multiview-inspection main`). Never commit on `main`. Push every commit (`git push -u origin multiview-inspection`) — the operator reads progress from the pushed history. Merge `--no-ff` at the end.
- **Another session may be working in this repo concurrently.** Stage your own paths explicitly (`git add <path> <path>`); never `git add -A` or `git commit -a`.
- Python is **`py -3.10`** (no `python` on PATH). Never round-trip a source file through PowerShell `Get-Content`/`Set-Content` — it silently mojibakes this repo's UTF-8. Edit with the Edit tool.
- **Do not run the full pytest suite** (too slow; the operator interrupts it). The targeted command, used in every task:
  ```
  py -3.10 -m pytest tests/test_extrusion_multiview.py tests/test_extrusion_measure.py tests/test_extrusion.py tests/test_extrusion_job.py tests/test_extrusion_processing.py tests/test_extrusion_figures.py tests/test_extrusion_standoff.py -q
  ```
- `ExtrusionConfig` (`tasni/core/config.py`) and every model in `tasni/modules/extrusion/models.py` are `extra="forbid"`: **every new field needs a default**, and no field is ever removed or renamed. Old archives must still validate.
- `CylinderPrintJob` and `CylinderDryRunJob` in `service.py` are cell-validated — **do not change their behaviour.**
- The single-view path of `RingMeasureJob` / `RingCharacterizeJob` / `process_observation` / `characterize_ring` is cell-validated (paper takes on 2026-08-28/29). `multiview=False` must produce what it produces today; the existing tests in `tests/test_extrusion_measure.py` are the regression and must stay green **untouched**.
- **ChArUco is out of scope by operator decision.** It belongs to hand-eye calibration only, and the board will not always be under the rings. Do not reintroduce it as registration, as an advisory check, or at all.
- Tests reaching `_filter_deposit` (anything through `process_points`, `merge_views` on real clusters, `characterize_*`) need Open3D: start them with `pytest.importorskip("open3d")`. `level_points`, the circle solve, the pose functions, the gate maths and the archive are pure numpy/scipy — no skip.
- The RoboDK item name for a view's program is `<stem>_Inspect` (top, unchanged) or `<stem>_Inspect_star000` / `_star120` / `_star240` (hyphen dropped). Archive directory names keep the hyphen: `views/star-120/`.
- Every task ends with a commit **and a push**. Commit messages end with `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>` plus your harness's `Claude-Session:` line if it gives you one.
- The Tasni backend caches imported modules: before any cell test, **restart Tasni** and check `GET /api/health` → `build.stale == false`.
- **Verify every call signature against the real code before transcribing it.** The code blocks in this plan call existing repo functions, and the plan's author can get a signature wrong. This already happened once: Task 2's `multiview_plan` was written calling `framing_standoff(..., config=config)` and treating its return as a float, when the real signature takes seven explicit keyword arguments (no `config`) and returns a **dict** — a guaranteed `TypeError` that survived authoring *and* review because no test in the task ever called the function. Before you transcribe a call to anything you did not write in this plan, open it and check. And if a task's code introduces a function, make sure at least one test **invokes** it: a function no test calls is a function no one has run.

---

## File map

| File | Responsibility in this plan |
|---|---|
| `tasni/core/config.py` | 12 `multiview_*` fields on `ExtrusionConfig` (Task 1) |
| `tasni/modules/extrusion/models.py` | `ViewRecord`, `CaptureRecord`; `LayerManifest.capture` (Task 1) |
| `tasni/modules/extrusion/inspection.py` | `star_view_angles`, `star_view_candidates`, `multiview_plan` (Task 2) |
| `tasni/modules/extrusion/measure.py` | incidence-aware `depth_plane_check` (Task 3); `capture_views`, job wiring, timings (Task 6); `capture_style` (Task 8) |
| `tasni/modules/extrusion/processing.py` | seam refactor: `observation_points`, `process_points`; `process_observation` / `characterize_ring` become wrappers (Task 4) |
| `tasni/modules/extrusion/multiview.py` (new) | `ViewCloud`, `MergeResult`, `level_points`, `fit_circle`, `solve_view_offsets`, `merge_views` (Task 5) |
| `tasni/modules/extrusion/archive.py` | `write_layer(views=…, merged_points_xyz=…)` (Task 6) |
| `tasni/modules/extrusion/service.py` | `reprocess_saved_layer(views=…)` (Task 7) |
| `tasni/modules/extrusion/figures.py` | the `views` figure (Task 8) |
| `tasni/modules/extrusion/module.py` | `multiview` on `MeasureLayerBody` / `CharacterizeBody` (Task 9) |
| `tasni/webui/src/pages/Extrusion.tsx` | both toggles, including the side photo's missing control (Task 9) |
| `tests/extrusion_synthetic.py` | `render_color` — chromatic frames so the gate holds (Task 5) |
| `tests/test_extrusion_multiview.py` (new) | Tasks 1–7 tests |
| `tools/multiview_ab.py` (new) | offline A/B over an archived trial (Task 7) |

---

## Interfaces used across tasks (names are binding)

```python
# inspection.py (Task 2)
def star_view_angles(config) -> list[tuple[str, float, float]]:
    """[(name, tilt_deg, azimuth_deg), ...] — ("top", 0.0, 0.0) first."""

def star_view_candidates(aim_mm, standoff_mm: float, config, *, tilt_deg: float,
                         azimuth_deg: float, reference_x=None) -> list[dict]:
    """Candidates for ONE view; roll varies, tilt/azimuth never do."""

def multiview_plan(recipe, setup, *, K, size_px, config) -> dict:
    """{"standoff_mm": float, "aim_mm": [x,y,z], "views": [{...}, ...]}"""

# measure.py (Task 3)
def depth_plane_check(depth, T_work_camera, config, *, unit_mm: float = 1.0) -> dict:
    """Adds 'cos_incidence' and 'expected_depth_mm' to the existing keys."""

# processing.py (Task 4)
def observation_points(*, color, depth, geometry, T_work_camera, K, dist, config,
                       counts: dict | None = None) -> tuple[np.ndarray, bool]:
    """(work-frame points Nx3 mm, chroma_gated)."""

def process_points(points, *, plan, layer, config, chroma_gated: bool,
                   floor_profile=None, stages=None,
                   assemble_arcs: bool = False) -> ProcessingResult: ...

# multiview.py (Task 5)
@dataclass
class ViewCloud:
    name: str; points: np.ndarray; chroma_gated: bool
    tilt_deg: float; azimuth_deg: float; T_work_camera: np.ndarray

@dataclass
class MergeResult:
    points: np.ndarray                 # merged work-frame cloud
    chroma_gated: bool
    used: list[str]; dropped: dict[str, str]
    consensus_center_mm: tuple[float, float]; consensus_radius_mm: float
    offsets_mm: dict[str, tuple[float, float]]
    residual_rms_mm: dict[str, float]
    spread_before_mm: float; residual_after_mm: float

def level_points(points, *, r_inner_mm, r_outer_mm, center_xy, config
                 ) -> tuple[np.ndarray, dict]: ...
def fit_circle(xy) -> tuple[float, float, float]:      # (cx, cy, r)
def solve_view_offsets(view_xy: dict[str, np.ndarray], config) -> dict: ...
def merge_views(views: list[ViewCloud], *, plan, layer, config) -> MergeResult: ...

# archive.py (Task 6)
#   write_layer(..., views=None, merged_points_xyz=None)
#     views: list[{"name", "color", "depth", "pose": dict}]

# service.py (Task 7)
#   reprocess_saved_layer(root, trial_id, layer_index, take=1, *, views="as_archived")
```

---

### Task 1: Config keys and manifest records

Pure data. No behaviour changes anywhere; the point is that every later task has its knobs and its record types already defined and validated.

**Files:**
- Modify: `tasni/core/config.py` (in `ExtrusionConfig`, after `characterize_max_height_mm`)
- Modify: `tasni/modules/extrusion/models.py` (after `SideViewRecord`, ~line 150)
- Test: `tests/test_extrusion_multiview.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces: the 12 `multiview_*` config fields; `ViewRecord`, `CaptureRecord`, `LayerManifest.capture`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_extrusion_multiview.py`:

```python
"""Multi-view ring capture: poses, gates, levelling, the joint circle solve, merge."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tasni.core.config import ExtrusionConfig  # noqa: E402
from tasni.modules.extrusion.models import (CaptureRecord, LayerManifest,  # noqa: E402
                                            ViewRecord)


def test_multiview_config_defaults_match_the_spec():
    c = ExtrusionConfig()
    assert c.multiview_enabled is False           # opt-in; single view is the validated one
    assert c.multiview_tilt_deg == 15.0           # spec section 3: 20 deg costs 4.97 mm plane RMS
    assert c.multiview_max_tilt_deg == 25.0
    assert c.multiview_azimuths_deg == [0.0, 120.0, 240.0]
    assert c.multiview_min_cos_incidence == 0.5
    assert c.multiview_level_annulus_width_mm == 60.0
    assert c.multiview_level_min_points == 500
    assert c.multiview_max_level_mm == 10.0
    assert c.multiview_min_view_points == 200
    assert c.multiview_min_arc_deg == 90.0
    assert c.multiview_max_offset_mm == 5.0
    assert c.multiview_min_views == 2


def test_view_record_defaults_and_drop_reason():
    v = ViewRecord(name="star-120", tilt_deg=15.0, azimuth_deg=120.0)
    assert v.dropped is False and v.drop_reason is None
    assert v.solved_offset_mm is None and v.chroma_gated is None
    dropped = ViewRecord(name="star-240", tilt_deg=15.0, azimuth_deg=240.0,
                         dropped=True, drop_reason="chroma gate abstained")
    assert dropped.drop_reason == "chroma gate abstained"


def test_capture_record_defaults_to_single_view():
    c = CaptureRecord()
    assert c.style == "single" and c.views == [] and c.merged_points_file is None


def test_old_manifest_without_capture_still_validates(tmp_path):
    """extra='forbid' plus a new required field would break every archived take."""
    from tasni.modules.extrusion.models import CylinderRecipe
    recipe = CylinderRecipe(radius_mm=40.0, layer_count=1, layer_height_mm=6.0,
                            bead_diameter_mm=10.0, robot_speed_mm_s=75,
                            extrusion_rate_pct=0)
    m = LayerManifest(trial_id="t", layer_index=1, recipe=recipe,
                      toolpath_fingerprint="abc")
    assert m.capture is None
    assert LayerManifest.model_validate_json(m.model_dump_json()).capture is None
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `py -3.10 -m pytest tests/test_extrusion_multiview.py -q`
Expected: FAIL — `ImportError: cannot import name 'CaptureRecord'`.

- [ ] **Step 3: Add the config fields**

In `tasni/core/config.py`, inside `ExtrusionConfig`, immediately after `characterize_max_height_mm`:

```python
    # -- multi-view ring capture (modules/extrusion/multiview.py) ------------
    # Opt-in. The single-view chain is the cell-validated one and every number
    # in the PFH archive came from it; turning this on changes what a take IS.
    multiview_enabled: bool = False
    # The cell's own incidence sweep (characterization/characterization-20260813.json)
    # measured board plane RMS 0.650 mm at 1 deg, 2.006 at 9.1, 4.969 at 19.6 and
    # 7.430 at 29.4. The rings are 2.9-4.9 mm proud over much of their length, so
    # 20 deg buys a noise floor comparable to the signal. 15 is the compromise;
    # the on-cell A/B sweeps 10/15/20 offline from one capture session.
    multiview_tilt_deg: float = Field(default=15.0, ge=0.0, le=45.0)
    multiview_max_tilt_deg: float = Field(default=25.0, ge=0.0, le=45.0)
    # Three azimuths is the minimum that puts every flank point within 60 deg of
    # some camera. Measured in the WORK frame from +X -- the same axis the
    # paired-detection offset is expressed along -- so a star is reproducible.
    multiview_azimuths_deg: list[float] = Field(
        default_factory=lambda: [0.0, 120.0, 240.0])
    # depth_plane_check divides by cos(incidence); refuse a degenerate pose
    # rather than divide by something near zero. 0.5 = 60 deg.
    multiview_min_cos_incidence: float = Field(default=0.5, gt=0.0, le=1.0)
    # The levelling annulus runs from the OUTER edge of the chain's radial ROI
    # band (recipe.radius_mm + radial_roi_margin_mm) outward by this width. It
    # must contain surface and never deposit: it is what defines z = 0.
    multiview_level_annulus_width_mm: float = Field(default=60.0, gt=0, le=500)
    multiview_level_min_points: int = Field(default=500, ge=50)
    # A fitted surface further than this from z = 0 means the view is wrong, not
    # tilted -- a bad pose, a wrong work frame -- so drop it rather than "level"
    # the whole scene onto a fiction.
    multiview_max_level_mm: float = Field(default=10.0, gt=0, le=200)
    multiview_min_view_points: int = Field(default=200, ge=10)
    # A view seeing less arc than this cannot constrain a centre: fitting a
    # circle to a short arc trades centre against radius almost freely.
    multiview_min_arc_deg: float = Field(default=90.0, gt=0, le=360)
    # A solved offset beyond this is a failed registration, not a measurement.
    multiview_max_offset_mm: float = Field(default=5.0, gt=0, le=100)
    # Below this, "merging" is one cloud with extra steps: fall back to top-only.
    multiview_min_views: int = Field(default=2, ge=2)
```

- [ ] **Step 4: Add the manifest records**

In `tasni/modules/extrusion/models.py`, after `SideViewRecord`:

```python
class ViewRecord(_Record):
    """One camera view of the ring inside a multi-view take.

    Every field beyond the identity three is optional because a view can be
    dropped at any stage -- unreachable pose, failed arrival gate, abstaining
    colour gate, too little ring to fit -- and the record still has to say what
    happened. ``dropped`` with a ``drop_reason`` is a normal outcome, not an
    error: the take completes on the views that survived (spec section 8).
    """

    name: str
    tilt_deg: float
    azimuth_deg: float
    roll_deg: float | None = None
    T_work_camera: list[list[float]] | None = None
    standoff_delta_mm: float | None = None
    chroma_fraction: float | None = None
    chroma_gated: bool | None = None
    points_before_merge: int | None = None
    fitted_center_mm: list[float] | None = None
    solved_offset_mm: list[float] | None = None
    residual_rms_mm: float | None = None
    dropped: bool = False
    drop_reason: str | None = None


class CaptureRecord(_Record):
    """How the take's cloud was acquired, and how much the views disagreed.

    ``spread_before_mm`` is the raw scatter of the per-view fitted centres about
    their mean BEFORE any correction -- the cell's hand-eye plus pose error read
    at the ring. It is archived because it is the evidence that makes the merge
    interpretable; ``residual_after_mm`` is what survived the joint solve.
    """

    style: str = "single"                      # "single" | "star"
    views: list[ViewRecord] = Field(default_factory=list)
    consensus_center_mm: list[float] | None = None
    consensus_radius_mm: float | None = None
    spread_before_mm: float | None = None
    residual_after_mm: float | None = None
    merged_points_file: str | None = None
    timings_ms: dict[str, Any] = Field(default_factory=dict)
```

And add one field to `LayerManifest`, next to `side_view`:

```python
    # How this take's cloud was acquired. None on every take archived before
    # multi-view existed, which is what keeps extra="forbid" safe here.
    capture: CaptureRecord | None = None
```

- [ ] **Step 5: Run the tests and make sure they pass**

Run: `py -3.10 -m pytest tests/test_extrusion_multiview.py -q`
Expected: 4 passed.

Then confirm nothing else moved:
Run: `py -3.10 -m pytest tests/test_extrusion_measure.py tests/test_extrusion.py -q`
Expected: all pass, same counts as before the task.

- [ ] **Step 6: Commit and push**

```bash
git add tasni/core/config.py tasni/modules/extrusion/models.py tests/test_extrusion_multiview.py
git commit -m "feat(extrusion): multi-view config keys and manifest records"
git push -u origin multiview-inspection
```

---

### Task 2: Star poses

Pure numpy on top of the cone `pose_from_aim` already builds. No new pose maths — the star is a *choice of angles*.

**Files:**
- Modify: `tasni/modules/extrusion/inspection.py` (after `pose_candidates`, ~line 190)
- Test: `tests/test_extrusion_multiview.py`

**Interfaces:**
- Consumes: Task 1's config fields; the existing `pose_from_aim`, `_roll_reference_axis`, `framing_standoff`, `aim_point_mm`.
- Produces: `star_view_angles`, `star_view_candidates`, `multiview_plan`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_extrusion_multiview.py`:

```python
from tasni.modules.extrusion.inspection import (pose_from_aim,  # noqa: E402
                                                star_view_angles,
                                                star_view_candidates)


def test_star_angles_are_top_first_then_the_configured_azimuths():
    names = star_view_angles(ExtrusionConfig())
    assert names[0] == ("top", 0.0, 0.0)
    assert [n for n, _, _ in names] == ["top", "star-000", "star-120", "star-240"]
    assert [round(t, 3) for _, t, _ in names[1:]] == [15.0, 15.0, 15.0]
    assert [a for _, _, a in names[1:]] == [0.0, 120.0, 240.0]


def test_star_angles_honour_the_tilt_cap():
    c = ExtrusionConfig(multiview_tilt_deg=40.0, multiview_max_tilt_deg=25.0)
    assert all(t == 25.0 for _, t, _ in star_view_angles(c)[1:])


def test_star_candidates_vary_roll_only_never_tilt_or_azimuth():
    """A fallback may re-roll the wrist. It may NOT quietly become another view."""
    cands = star_view_candidates([0.0, 0.0, 5.0], 300.0, ExtrusionConfig(),
                                 tilt_deg=15.0, azimuth_deg=120.0)
    assert len(cands) == len(ExtrusionConfig().inspection_roll_candidates_deg)
    assert {c["tilt_deg"] for c in cands} == {15.0}
    assert {c["azimuth_deg"] for c in cands} == {120.0}
    assert [c["roll_deg"] for c in cands] == ExtrusionConfig().inspection_roll_candidates_deg


def test_every_star_view_keeps_the_aim_point_on_axis_at_the_standoff():
    aim = np.array([10.0, -20.0, 5.0])
    for _, tilt, azimuth in star_view_angles(ExtrusionConfig()):
        T = pose_from_aim(aim, 300.0, tilt_deg=tilt, azimuth_deg=azimuth)
        camera, axis = T[:3, 3], T[:3, 2]
        assert abs(np.linalg.norm(camera - aim) - 300.0) < 1e-6      # exact standoff
        to_aim = (aim - camera) / np.linalg.norm(aim - camera)
        assert np.dot(axis, to_aim) > 1.0 - 1e-9                     # exactly on axis


def test_star_views_sit_on_a_cone_and_are_120_deg_apart_in_the_work_frame():
    aim = np.array([0.0, 0.0, 5.0])
    cams = [pose_from_aim(aim, 300.0, tilt_deg=t, azimuth_deg=a)[:3, 3]
            for _, t, a in star_view_angles(ExtrusionConfig())[1:]]
    offsets = [c[:2] - aim[:2] for c in cams]
    radii = [float(np.linalg.norm(o)) for o in offsets]
    assert max(radii) - min(radii) < 1e-6                            # one cone
    angles = sorted(round(float(np.degrees(np.arctan2(o[1], o[0]))) % 360.0, 3)
                    for o in offsets)
    assert angles == [0.0, 120.0, 240.0]
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `py -3.10 -m pytest tests/test_extrusion_multiview.py -q -k star`
Expected: FAIL — `ImportError: cannot import name 'star_view_angles'`.

- [ ] **Step 3: Implement**

In `tasni/modules/extrusion/inspection.py`, after `pose_candidates`:

```python
def star_view_angles(config) -> list[tuple[str, float, float]]:
    """The (name, tilt, azimuth) table for one multi-view star, top view first.

    A "Mercedes star": the top view as today plus one ring of the scan module's
    dome at a single cone angle. All views share the aim point and the standoff
    (pose_from_aim guarantees both), so the ring fills the frame identically
    from every side and no view is better framed than another.

    Azimuth is measured in the WORK frame from +X -- the axis the paired
    detection offset is expressed along -- so a star has the same orientation
    across takes and across sessions.
    """
    tilt = min(float(config.multiview_tilt_deg), float(config.multiview_max_tilt_deg))
    views = [("top", 0.0, 0.0)]
    for azimuth in config.multiview_azimuths_deg:
        views.append((f"star-{int(round(float(azimuth))):03d}", tilt, float(azimuth)))
    return views


def star_view_candidates(aim_mm, standoff_mm: float, config, *, tilt_deg: float,
                         azimuth_deg: float, reference_x=None) -> list[dict]:
    """Ordered candidate poses for ONE view of the star.

    Roll varies; tilt and azimuth never do. That asymmetry is the point:
    ``pose_candidates`` treats tilt and azimuth as FALLBACKS for an unreachable
    fronto-parallel pose, but here they are what the view IS. Substituting them
    would silently hand the merge two clouds of the same flank and none of the
    other, which no diagnostic downstream could detect.
    """
    candidates = []
    for roll in config.inspection_roll_candidates_deg:
        candidates.append({
            "T_work_camera": pose_from_aim(aim_mm, standoff_mm, tilt_deg=tilt_deg,
                                           azimuth_deg=azimuth_deg, roll_deg=roll,
                                           reference_x=reference_x),
            "tilt_deg": float(tilt_deg), "azimuth_deg": float(azimuth_deg),
            "roll_deg": float(roll)})
    return candidates


def multiview_plan(recipe, setup, *, K: np.ndarray, size_px: tuple[int, int],
                   config) -> dict:
    """Descriptors for the whole star: preview, the dry tour, and the job log."""
    aim = aim_point_mm(recipe, setup, 1)
    diameter = cylinder_diameter_mm(recipe)
    # framing_standoff takes the band explicitly and returns a DICT -- mirror
    # inspection_plan (inspection.py:252-256) exactly; it has no `config` kwarg.
    framing = framing_standoff(
        width_mm=diameter, height_mm=diameter, K=K, size_px=size_px,
        frame_margin=config.inspection_frame_margin,
        near_mm=config.inspection_min_mm, far_mm=config.inspection_max_mm)
    standoff = framing["standoff_mm"]
    return {"standoff_mm": float(standoff), "aim_mm": [float(v) for v in aim],
            "views": [{"name": name, "tilt_deg": tilt, "azimuth_deg": azimuth}
                      for name, tilt, azimuth in star_view_angles(config)]}
```

- [ ] **Step 4: Add the test that actually CALLS `multiview_plan`**

The other tests exercise the pose maths but never invoke `multiview_plan`, which is how a
wrong `framing_standoff` call survived both authoring and review on the first pass. Invoke it:

```python
def test_multiview_plan_executes_and_returns_its_documented_shape():
    import extrusion_synthetic as syn
    import test_extrusion_measure as tem          # repo idiom; there is no tests/__init__.py
    from tasni.modules.extrusion.inspection import multiview_plan

    plan = tem.scene_plan()
    config = ExtrusionConfig()
    out = multiview_plan(plan.recipe, plan.setup, K=syn.K_720P,
                         size_px=syn.SIZE_720P, config=config)
    assert isinstance(out["standoff_mm"], float)
    assert config.inspection_min_mm <= out["standoff_mm"] <= config.inspection_max_mm
    assert len(out["aim_mm"]) == 3
    assert [v["name"] for v in out["views"]] == ["top", "star-000", "star-120", "star-240"]
    assert [v["tilt_deg"] for v in out["views"]] == [0.0, 15.0, 15.0, 15.0]
```

- [ ] **Step 5: Run the tests and make sure they pass**

Run: `py -3.10 -m pytest tests/test_extrusion_multiview.py -q`
Expected: 10 passed.

- [ ] **Step 6: Commit and push**

```bash
git add tasni/modules/extrusion/inspection.py tests/test_extrusion_multiview.py
git commit -m "feat(extrusion): star view angles and per-view pose candidates"
git push
```

---

### Task 3: The tilt-aware arrival gate

**This is the blocker, and it is independently useful — it can merge on its own.** Without it no tilted view can ever be captured.

**Files:**
- Modify: `tasni/modules/extrusion/measure.py:203-238` (`depth_plane_check`)
- Test: `tests/test_extrusion_multiview.py`

**Interfaces:**
- Consumes: `config.multiview_min_cos_incidence` (Task 1); `pose_from_aim` (Task 2, for the tests).
- Produces: `depth_plane_check` with two extra result keys, `cos_incidence` and `expected_depth_mm`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_extrusion_multiview.py`:

```python
from tasni.modules.extrusion.measure import depth_plane_check  # noqa: E402

AIM = np.array([0.0, 0.0, 5.0])          # a layer top 5 mm above the work plane


def _frame(mm: float, shape=(48, 64)):
    """A depth frame whose every pixel reads the same distance, in 1 mm words."""
    return np.full(shape, float(mm), dtype=float)


def test_top_view_gate_is_unchanged_byte_for_byte():
    """Pinned from the implementation BEFORE this task. The single-view path is
    cell-validated and this refactor must not move it by one millimetre."""
    T = pose_from_aim(AIM, 300.0, tilt_deg=0.0)
    out = depth_plane_check(_frame(300.0), T, ExtrusionConfig(), unit_mm=1.0)
    assert out["camera_z_mm"] == pytest.approx(305.0, abs=1e-6)
    assert out["observed_depth_mm"] == pytest.approx(300.0, abs=1e-6)
    assert out["accepted_range_mm"] == [265.0, 320.0]
    assert out["agrees"] is True
    assert out["cos_incidence"] == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("standoff, tilt", [(300.0, 25.0), (400.0, 20.0), (500.0, 18.0)])
def test_tilted_views_that_the_old_gate_rejected_now_pass(standoff, tilt):
    """Each row is a case computed against the real constants where the
    height-based gate fails: the median sits standoff*(1-cos t) above camera_z
    and the high side has only depth_plane_slack_mm = 15 mm of budget."""
    T = pose_from_aim(AIM, standoff, tilt_deg=tilt)
    config = ExtrusionConfig()
    camera_z = float(T[2, 3])
    old_high = camera_z + config.depth_plane_slack_mm
    assert standoff > old_high                      # the old gate WOULD have failed
    out = depth_plane_check(_frame(standoff), T, config, unit_mm=1.0)
    assert out["agrees"] is True
    assert out["cos_incidence"] == pytest.approx(np.cos(np.radians(tilt)), abs=1e-9)


def test_the_gate_keeps_its_sensitivity_across_the_whole_envelope():
    """The correction must not merely let tilted views through: the gap between
    expected and the true median has to stay at the aim_z term (+5.0 mm at tilt
    0) instead of growing with tilt, or the gate goes blind exactly where the
    data is worst."""
    for standoff in (300.0, 400.0, 500.0, 800.0):
        for tilt in (0.0, 10.0, 15.0, 20.0, 25.0):
            T = pose_from_aim(AIM, standoff, tilt_deg=tilt)
            out = depth_plane_check(_frame(standoff), T, ExtrusionConfig(), unit_mm=1.0)
            bias = out["expected_depth_mm"] - standoff
            assert 5.0 <= bias <= 5.8, (standoff, tilt, bias)
            assert out["agrees"] is True


def test_a_genuinely_wrong_depth_is_still_refused_at_tilt():
    """The frozen-stream fault the gate exists for (cell 2026-08-29: colour at
    312 mm, depth stuck at 447) must still be caught on a tilted view."""
    T = pose_from_aim(AIM, 300.0, tilt_deg=15.0)
    out = depth_plane_check(_frame(447.0), T, ExtrusionConfig(), unit_mm=1.0)
    assert out["agrees"] is False


def test_a_degenerate_near_horizontal_pose_is_refused_not_divided_by():
    T = pose_from_aim(AIM, 300.0, tilt_deg=80.0)          # cos = 0.17 < 0.5
    out = depth_plane_check(_frame(300.0), T, ExtrusionConfig(), unit_mm=1.0)
    assert out["agrees"] is False
    assert "incidence" in (out.get("refused") or "")


def test_native_depth_units_still_scale():
    T = pose_from_aim(AIM, 300.0, tilt_deg=15.0)
    out = depth_plane_check(_frame(3000.0), T, ExtrusionConfig(), unit_mm=0.1)
    assert out["observed_depth_mm"] == pytest.approx(300.0)
    assert out["agrees"] is True
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `py -3.10 -m pytest tests/test_extrusion_multiview.py -q -k "gate or tilt or envelope or degenerate or native"`
Expected: FAIL — `KeyError: 'cos_incidence'`, and the tilted cases fail `agrees is True`.

- [ ] **Step 3: Implement**

Replace the body of `depth_plane_check` in `tasni/modules/extrusion/measure.py`. Keep the whole existing docstring and **add** the paragraphs below to it:

```python
def depth_plane_check(depth, T_work_camera, config, *, unit_mm: float = 1.0) -> dict:
    """Does this depth frame describe the pose it was taken at?

    ... (KEEP the entire existing docstring verbatim) ...

    **Off-axis.** The paragraph above holds looking straight DOWN, where the
    frame's median depth IS the camera's height above the plane. Tilt separates
    them: the camera drops to ``aim_z + standoff*cos(t)`` while the median stays
    at roughly the standoff, so the median runs ABOVE camera_z by
    ``standoff*(1 - cos t)`` -- and the high side has only
    ``depth_plane_slack_mm`` (15 mm) of budget. Computed against the real
    constants that fails above ~18 deg at a 300 mm standoff and above ~14 deg at
    500 mm, and even where it passes it spends 5-12 mm of that 15 mm budget on
    geometry, leaving almost nothing to catch the fault this gate exists for.

    So the expectation is scaled by the incidence, read from the pose itself:
    ``pose_from_aim`` sets ``z_axis = -away`` with ``away_z = cos(tilt)``, so
    ``-T[2, 2]`` IS ``cos(tilt)`` -- no convention to get wrong, and nothing to
    pass in that could disagree with where the arm actually went. At tilt 0
    every expression below collapses to the height-based form exactly, which is
    what keeps the cell-validated single-view path unmoved. Swept over
    300-800 mm x 0-30 deg the residual holds at +5.0..+5.8 mm (the ``aim_z``
    term the tilt-0 gate already carries), so sensitivity stays flat instead of
    decaying with tilt.
    """
    T = np.asarray(T_work_camera, dtype=float)
    camera_z = float(T[2, 3])
    values = np.asarray(depth)
    valid = values[values > 0]
    observed = float(np.median(valid)) * float(unit_mm) if valid.size else float("nan")
    ceiling = float(getattr(config, "characterize_max_height_mm", 40.0))
    slack = float(getattr(config, "depth_plane_slack_mm", 15.0))
    # -T[2,2] is cos(tilt) for every pose_from_aim pose; clamp guards a
    # hand-built or degenerate transform rather than trusting the caller.
    cos_incidence = float(np.clip(-T[2, 2], -1.0, 1.0))
    floor_cos = float(getattr(config, "multiview_min_cos_incidence", 0.5))
    base = {"camera_z_mm": camera_z, "observed_depth_mm": observed,
            "valid_pixels": int(valid.size), "cos_incidence": cos_incidence}
    if cos_incidence < floor_cos:
        # Dividing by this would manufacture an expectation from nothing. A pose
        # this far off-axis is a bug upstream, not a view worth gating.
        return {**base, "expected_depth_mm": float("nan"),
                "accepted_range_mm": [float("nan"), float("nan")], "agrees": False,
                "refused": (f"camera incidence {np.degrees(np.arccos(cos_incidence)):.0f} deg "
                            f"exceeds the {np.degrees(np.arccos(floor_cos)):.0f} deg limit")}
    expected = camera_z / cos_incidence
    low, high = expected - ceiling / cos_incidence, expected + slack / cos_incidence
    return {**base, "expected_depth_mm": expected,
            "accepted_range_mm": [round(low, 1), round(high, 1)],
            "agrees": bool(valid.size and low <= observed <= high)}
```

Note `np` is already imported in `measure.py`.

- [ ] **Step 4: Run the tests and make sure they pass**

Run: `py -3.10 -m pytest tests/test_extrusion_multiview.py -q`
Expected: all pass.

- [ ] **Step 5: Prove the single-view path did not move**

Run: `py -3.10 -m pytest tests/test_extrusion_measure.py tests/test_extrusion.py tests/test_extrusion_standoff.py -q`
Expected: all pass, **same counts as before this task**. If any test here needed editing, stop — the reduction claim is false and the implementation is wrong.

- [ ] **Step 6: Commit and push**

```bash
git add tasni/modules/extrusion/measure.py tests/test_extrusion_multiview.py
git commit -m "fix(extrusion): the arrival gate assumed a straight-down view"
git push
```

---

### Task 4: The processing seam

Behaviour-preserving refactor. The proof is that `tests/test_extrusion_processing.py`, `tests/test_extrusion_measure.py` and the archived-fixture tests pass **unedited**.

**Files:**
- Modify: `tasni/modules/extrusion/processing.py:645` (`process_observation`) and `:914` (`characterize_ring`)
- Test: existing suites (the regression) + one new seam test

**Interfaces:**
- Consumes: nothing new.
- Produces: `observation_points(...) -> (points, chroma_gated)` and `process_points(...) -> ProcessingResult`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_extrusion_multiview.py`:

```python
def test_the_seam_reproduces_process_observation_exactly():
    """observation_points + process_points must equal process_observation. If it
    does not, every archived number silently changes meaning."""
    pytest.importorskip("open3d")
    import extrusion_synthetic as syn
    import geometry_fixtures as gf
    from tasni.modules.extrusion.inspection import aim_point_mm
    from tasni.modules.extrusion.processing import (observation_points,
                                                    process_observation,
                                                    process_points)
    from tests.test_extrusion_measure import scene_plan     # same fixture as today

    plan = scene_plan()
    layer = plan.layers[0]
    T = syn.inspection_camera_T(aim_point_mm(plan.recipe, plan.setup, 1), 300.0)
    rings = [syn.RingSpec(60.0, 8.0, (200.0, 150.0), height_fn=syn.flat(6.0))]
    depth = syn.render_scene(rings, T, plane_center_xy_mm=(200.0, 150.0))
    color = np.zeros((syn.SIZE_720P[1], syn.SIZE_720P[0], 3), np.uint8)
    geom, config = gf.aligned(syn.K_720P, syn.SIZE_720P), ExtrusionConfig()

    whole = process_observation(color=color, depth=depth, geometry=geom,
                                T_work_camera=T, K=syn.K_720P, dist=None,
                                plan=plan, layer=layer, config=config)
    points, gated = observation_points(color=color, depth=depth, geometry=geom,
                                       T_work_camera=T, K=syn.K_720P, dist=None,
                                       config=config)
    split = process_points(points, plan=plan, layer=layer, config=config,
                           chroma_gated=gated)
    np.testing.assert_allclose(split.measured_xyz, whole.measured_xyz)
    assert split.metrics.model_dump() == whole.metrics.model_dump()
```

- [ ] **Step 2: Run it to make sure it fails**

Run: `py -3.10 -m pytest tests/test_extrusion_multiview.py -q -k seam`
Expected: FAIL — `ImportError: cannot import name 'observation_points'`.

- [ ] **Step 3: Split the function**

In `processing.py`, extract the first block of `process_observation` (everything from `mark = time.perf_counter()` through `keep("backprojected", points)`) into:

```python
def observation_points(*, color: np.ndarray, depth: np.ndarray,
                       geometry: CameraGeometry, T_work_camera: np.ndarray,
                       K: np.ndarray, dist: np.ndarray | None, config,
                       counts: dict | None = None) -> tuple[np.ndarray, bool]:
    """One frame's chroma-gated points, in the work frame. Nothing ROI-specific.

    This is the seam multi-view merges at (multiview.py): everything ABOVE it is
    per-frame by nature -- it needs that frame's own colour, its own registration
    and its own hand-eye pose -- and everything below it is per-CLOUD and must
    run exactly once over the merged result, or the ROI, the voxel, the DBSCAN
    and the crest extraction would each run n times and disagree.
    """
    counts = {} if counts is None else counts
    reg = ColorRegistered.build(depth, geometry, K, dist)
    keep_mask, chroma_gated = chroma_gate_mask(color, reg, config, counts)
    points = transform_points(T_work_camera, reg.pts_mm[keep_mask])
    counts["raw_depth_pixels"] = int(keep_mask.sum())
    return points, chroma_gated
```

Then rename the remainder to `process_points`, taking the already-transformed cloud:

```python
def process_points(points: np.ndarray, *, plan: CylinderPlan, layer: LayerPath,
                   config, chroma_gated: bool,
                   floor_profile: np.ndarray | None = None,
                   stages: dict | None = None, assemble_arcs: bool = False,
                   counts: dict | None = None,
                   timings: dict | None = None) -> ProcessingResult:
    """Everything from the work ROI onward, over ONE cloud of any provenance."""
```

Its body is the current `process_observation` body from `setup, recipe = plan.setup, plan.recipe` to the end, unchanged, with `min_z = deposit_floor_mm(config, chroma_gated)` now reading the passed-in flag.

Finally make the old entry points thin wrappers that preserve their exact signatures and their timing keys:

```python
def process_observation(*, color, depth, geometry, T_work_camera, K, dist,
                        plan, layer, config, floor_profile=None, stages=None,
                        assemble_arcs=False) -> ProcessingResult:
    """... (KEEP the entire existing docstring verbatim) ..."""
    counts: dict = {}
    timings: dict = {}
    mark = time.perf_counter()
    points, chroma_gated = observation_points(
        color=color, depth=depth, geometry=geometry, T_work_camera=T_work_camera,
        K=K, dist=dist, config=config, counts=counts)
    timings["backproject_ms"] = (time.perf_counter() - mark) * 1000
    if stages is not None:
        stages["backprojected"] = np.asarray(points, dtype=float).copy()
    return process_points(points, plan=plan, layer=layer, config=config,
                          chroma_gated=chroma_gated, floor_profile=floor_profile,
                          stages=stages, assemble_arcs=assemble_arcs,
                          counts=counts, timings=timings)
```

Apply the same treatment to `characterize_ring` (it keeps `assemble_arcs=True`).

- [ ] **Step 4: Run the seam test**

Run: `py -3.10 -m pytest tests/test_extrusion_multiview.py -q -k seam`
Expected: PASS.

- [ ] **Step 5: Prove the refactor changed nothing**

Run: `py -3.10 -m pytest tests/test_extrusion_processing.py tests/test_extrusion_measure.py tests/test_extrusion.py tests/test_extrusion_figures.py -q`
Expected: all pass, **unedited**. Editing any of these to make them pass means the refactor was not behaviour-preserving — revert and redo.

- [ ] **Step 6: Commit and push**

```bash
git add tasni/modules/extrusion/processing.py tests/test_extrusion_multiview.py
git commit -m "refactor(extrusion): split processing at the back-projection seam"
git push
```

---

### Task 5: `multiview.py` — levelling, the joint circle solve, the merge

The core. Pure numpy/scipy except where it reads the chain's ROI constants.

**Files:**
- Create: `tasni/modules/extrusion/multiview.py`
- Modify: `tests/extrusion_synthetic.py` (add `render_color`)
- Test: `tests/test_extrusion_multiview.py`

**Interfaces:**
- Consumes: Task 1's config; Task 4's `observation_points`.
- Produces: `ViewCloud`, `MergeResult`, `level_points`, `fit_circle`, `solve_view_offsets`, `merge_views`.

- [ ] **Step 1: Give the synthetic scenes a chromatic colour frame**

Without this every multi-view test is worthless: `render_scene`'s companion colour frame is all zeros, `chroma_gate_mask` abstains, and the drop rule in `merge_views` would discard **every** view, so the tests would exercise only the fallback.

Add to `tests/extrusion_synthetic.py`:

```python
def render_color(rings: list[RingSpec], T_work_camera: np.ndarray, *,
                 K: np.ndarray = K_720P, size_px: tuple[int, int] = SIZE_720P,
                 ring_bgr=(40, 90, 190), board_bgr=(180, 180, 180)) -> np.ndarray:
    """A colour frame where the rings are chromatic and the board is not.

    The real discriminator is saturation (deposit_min_saturation = 60): the clay
    is chromatic, the printed board is not, and they separate ~20:1 on the cell.
    A test that leaves colour at zeros makes the gate ABSTAIN, which restores the
    2.5 mm floor and -- under multiview's per-view rule -- drops the view. So any
    test that means to exercise the gate-held path must render colour here.
    """
    w, h = size_px
    image = np.full((h, w, 3), board_bgr, np.uint8)          # S ~ 0: reads "board"
    for ring in rings:
        pts = ring.surface_points()
        cam = np.linalg.inv(T_work_camera) @ np.c_[pts, np.ones(len(pts))].T
        cam = cam[:3].T
        forward = cam[cam[:, 2] > 1.0]
        uv = (K @ forward.T).T
        uv = uv[:, :2] / uv[:, 2:3]
        u = np.rint(uv[:, 0]).astype(int)
        v = np.rint(uv[:, 1]).astype(int)
        ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        image[v[ok], u[ok]] = ring_bgr                        # S ~ 200: reads "bead"
    return image
```

Verify the intent directly:

```python
def test_synthetic_colour_actually_holds_the_chroma_gate():
    """If this fails, every merge test below is silently testing the fallback."""
    import cv2
    import extrusion_synthetic as syn
    rings = [syn.RingSpec(60.0, 8.0, (200.0, 150.0), height_fn=syn.flat(6.0))]
    T = syn.inspection_camera_T(np.array([200.0, 150.0, 6.0]), 300.0)
    color = syn.render_color(rings, T)
    sat = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)[:, :, 1]
    fraction = float((sat > 60).mean())
    assert fraction > ExtrusionConfig().deposit_min_chroma_fraction
```

- [ ] **Step 2: Write the failing tests for the solve**

```python
from tasni.modules.extrusion.multiview import (ViewCloud, fit_circle,  # noqa: E402
                                               level_points, merge_views,
                                               solve_view_offsets)


def _ring_xy(cx, cy, r, n=720, arc_deg=360.0, seed=0):
    rng = np.random.default_rng(seed)
    theta = np.radians(np.linspace(0.0, arc_deg, n, endpoint=False))
    rad = r + rng.normal(0.0, 0.2, n)
    return np.column_stack((cx + rad * np.cos(theta), cy + rad * np.sin(theta)))


def test_fit_circle_recovers_a_known_circle():
    cx, cy, r = fit_circle(_ring_xy(200.0, 150.0, 40.5))
    assert (cx, cy, r) == pytest.approx((200.0, 150.0, 40.5), abs=0.05)


def test_the_joint_solve_recovers_injected_offsets():
    truth = {"top": (0.0, 0.0), "star-000": (1.2, -0.7),
             "star-120": (-0.9, 0.4), "star-240": (0.3, 1.1)}
    mean = np.mean(list(truth.values()), axis=0)
    views = {name: _ring_xy(200.0, 150.0, 40.5, seed=i) + np.array(d)
             for i, (name, d) in enumerate(truth.items())}
    out = solve_view_offsets(views, ExtrusionConfig())
    for name, d in truth.items():
        # Recovered up to the gauge: the solve removes the MEAN displacement,
        # which is unobservable from the ring alone.
        expected = np.array(d) - mean
        assert np.allclose(out["offsets_mm"][name], -expected, atol=0.12), name


def test_the_gauge_is_a_consensus_not_an_anchor():
    """THE test that the old spec's circularity is gone. Displacing the TOP view
    must move the consensus centre by 1/n of the displacement -- not by zero,
    which is what anchoring every view to the top view would give."""
    base = {n: _ring_xy(200.0, 150.0, 40.5, seed=i)
            for i, n in enumerate(["top", "star-000", "star-120", "star-240"])}
    before = solve_view_offsets(base, ExtrusionConfig())["consensus_center_mm"]
    moved = dict(base)
    moved["top"] = base["top"] + np.array([4.0, 0.0])
    after = solve_view_offsets(moved, ExtrusionConfig())["consensus_center_mm"]
    assert after[0] - before[0] == pytest.approx(1.0, abs=0.15)      # 4.0 / 4 views
    assert after[1] - before[1] == pytest.approx(0.0, abs=0.15)


def test_offsets_sum_to_zero():
    views = {n: _ring_xy(200.0, 150.0, 40.5, seed=i) + np.array([i * 0.6, -i * 0.4])
             for i, n in enumerate(["top", "star-000", "star-120", "star-240"])}
    out = solve_view_offsets(views, ExtrusionConfig())
    total = np.sum(list(out["offsets_mm"].values()), axis=0)
    assert np.allclose(total, [0.0, 0.0], atol=1e-6)


def test_a_scaled_view_surfaces_as_residual_not_absorbed():
    """One shared radius is the point: a per-view radius would let a view with
    residual scale error fit itself perfectly and hide the problem."""
    views = {n: _ring_xy(200.0, 150.0, 40.5, seed=i)
             for i, n in enumerate(["top", "star-000", "star-120"])}
    views["star-240"] = _ring_xy(200.0, 150.0, 44.0, seed=9)         # +8.6% scale
    out = solve_view_offsets(views, ExtrusionConfig())
    assert out["residual_rms_mm"]["star-240"] > 5 * max(
        out["residual_rms_mm"][n] for n in ("top", "star-000", "star-120"))


def test_levelling_removes_an_injected_plane_tilt():
    rng = np.random.default_rng(0)
    xy = rng.uniform(-150.0, 150.0, (6000, 2))
    tilt = np.radians(8.0)
    z = xy[:, 0] * np.tan(tilt) + 3.0
    levelled, diag = level_points(np.column_stack((xy, z)), r_inner_mm=90.0,
                                  r_outer_mm=150.0, center_xy=(0.0, 0.0),
                                  config=ExtrusionConfig())
    inside = np.linalg.norm(levelled[:, :2], axis=1) < 150.0
    assert abs(float(np.median(levelled[inside, 2]))) < 0.3
    assert diag["level_mm"] == pytest.approx(3.0, abs=0.5)
```

- [ ] **Step 3: Run them to make sure they fail**

Run: `py -3.10 -m pytest tests/test_extrusion_multiview.py -q -k "circle or solve or gauge or gauge or scaled or level"`
Expected: FAIL — `ModuleNotFoundError: tasni.modules.extrusion.multiview`.

- [ ] **Step 4: Implement `multiview.py`**

```python
"""Merge several camera views of one ring into a single work-frame cloud.

The rings are thin: a top-down frame sees the crest at grazing signal-to-noise
and the flanks hardly at all, and the flanks are what the bead-width and
cross-section numbers are read from. Tilted views add them; this module makes
the tilted views agree with each other well enough to be worth adding.

Two things it deliberately does NOT do.

ICP. The ring is a torus, so sliding one view tangentially around it costs
almost nothing in point-to-point distance and ICP has a nearly free degree of
freedom it will fill with noise. When a shape has a degenerate DOF you stop
registering points and start fitting the model the object actually has: a circle.

Anchor to the top view. Translating each tilted view so its own fitted centre
lands on the top view's makes the merged centre, by construction, the top view's
centre -- so the merged cloud would be the measurement of record while the
headline number was still a single-view answer. The joint solve below fixes the
gauge as ``sum(offsets) == 0`` instead, so no view is privileged and the centre
is a consensus.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from .processing import deposit_floor_mm

MAX_FIT_POINTS = 2000        # per view; the solve is O(points), the fit is not


@dataclass
class ViewCloud:
    """One view's chroma-gated work-frame points, before any merge."""

    name: str
    points: np.ndarray
    chroma_gated: bool
    tilt_deg: float = 0.0
    azimuth_deg: float = 0.0
    T_work_camera: np.ndarray | None = None


@dataclass
class MergeResult:
    points: np.ndarray
    chroma_gated: bool
    used: list[str] = field(default_factory=list)
    dropped: dict[str, str] = field(default_factory=dict)
    consensus_center_mm: tuple[float, float] | None = None
    consensus_radius_mm: float | None = None
    offsets_mm: dict[str, tuple[float, float]] = field(default_factory=dict)
    residual_rms_mm: dict[str, float] = field(default_factory=dict)
    spread_before_mm: float | None = None
    residual_after_mm: float | None = None


def fit_circle(xy) -> tuple[float, float, float]:
    """Least-squares circle through 2-D points (Kasa), returned as (cx, cy, r).

    Algebraic, not geometric: it is a seed for the joint solve, and it is linear,
    so it cannot fail to converge on a partial arc the way an iterative fit can.
    """
    p = np.asarray(xy, dtype=float).reshape(-1, 2)
    A = np.column_stack((2.0 * p[:, 0], 2.0 * p[:, 1], np.ones(len(p))))
    b = (p ** 2).sum(axis=1)
    cx, cy, c = np.linalg.lstsq(A, b, rcond=None)[0]
    return float(cx), float(cy), float(np.sqrt(max(c + cx * cx + cy * cy, 0.0)))


def arc_span_deg(xy, center_xy) -> float:
    """Angular extent actually covered, as the largest gap's complement.

    A view seeing two short opposite arcs covers 360 deg of RANGE but constrains
    a centre about as well as one arc, so span is measured as 360 minus the
    biggest empty wedge.
    """
    p = np.asarray(xy, dtype=float).reshape(-1, 2) - np.asarray(center_xy, dtype=float)
    if len(p) < 3:
        return 0.0
    ang = np.sort(np.degrees(np.arctan2(p[:, 1], p[:, 0])) % 360.0)
    gaps = np.diff(np.concatenate((ang, [ang[0] + 360.0])))
    return float(360.0 - gaps.max())


def level_points(points, *, r_inner_mm: float, r_outer_mm: float, center_xy,
                 config) -> tuple[np.ndarray, dict]:
    """Rotate/translate one view so the surface its ring sits on is z = 0.

    The plane is fitted in an ANNULUS outside the chain's radial ROI band, so it
    sees surface and never deposit. This removes the plane-offset-and-tilt part
    of the systematic warp that grows with incidence -- the cell's own sweep
    measured board length error growing 0.036 -> 0.447 mm from 0 to 20 deg, far
    faster than random noise would.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    centre = np.asarray(center_xy, dtype=float)
    radius = np.linalg.norm(pts[:, :2] - centre, axis=1)
    annulus = pts[(radius >= r_inner_mm) & (radius <= r_outer_mm)]
    if len(annulus) < int(config.multiview_level_min_points):
        raise ValueError(f"levelling annulus held {len(annulus)} points, "
                         f"need {config.multiview_level_min_points}")
    # Plane through the annulus by SVD on the centred points: robust to the
    # annulus being wide, and it needs no RANSAC because the annulus is chosen
    # to exclude the deposit in the first place.
    mean = annulus.mean(axis=0)
    normal = np.linalg.svd(annulus - mean)[2][-1]
    if normal[2] < 0:
        normal = -normal
    level_mm = float(abs(mean[2]))
    if level_mm > float(config.multiview_max_level_mm):
        raise ValueError(f"fitted surface sits {level_mm:.1f} mm from z=0, "
                         f"limit {config.multiview_max_level_mm:.1f} mm")
    # Rotation taking the fitted normal onto +Z (Rodrigues about their cross).
    target = np.array([0.0, 0.0, 1.0])
    axis = np.cross(normal, target)
    s, c = float(np.linalg.norm(axis)), float(np.dot(normal, target))
    if s < 1e-9:
        R = np.eye(3)
    else:
        k = axis / s
        Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        R = np.eye(3) + Kx * s + Kx @ Kx * (1 - c)
    levelled = (R @ (pts - mean).T).T + np.array([mean[0], mean[1], 0.0])
    return levelled, {"level_mm": level_mm,
                      "tilt_removed_deg": float(np.degrees(np.arccos(np.clip(c, -1, 1)))),
                      "annulus_points": int(len(annulus))}


def solve_view_offsets(view_xy: dict[str, np.ndarray], config) -> dict:
    """Per-view lateral offsets that make every view fit ONE shared circle.

    Unknowns: one (dx, dy) per view, plus a shared centre and a shared radius.
    One shared radius, not one per view: a per-view radius would let a view
    carrying residual scale error absorb it silently, where a shared one forces
    it out as a large residual.

    Gauge: sum(offsets) == 0, imposed as two heavily weighted residual rows.
    Without it the problem is translation-degenerate (slide everything together
    and nothing changes). With it, no view is privileged and the recovered
    centre is the consensus of all views rather than the first one's answer.

    Soft-L1 loss so one bad arc cannot drag the solution.
    """
    names = list(view_xy)
    subs, seeds = [], []
    rng = np.random.default_rng(0)
    for name in names:
        p = np.asarray(view_xy[name], dtype=float).reshape(-1, 2)
        if len(p) > MAX_FIT_POINTS:
            p = p[rng.choice(len(p), MAX_FIT_POINTS, replace=False)]
        subs.append(p)
        seeds.append(fit_circle(p))
    seed_centres = np.array([[s[0], s[1]] for s in seeds])
    c0 = seed_centres.mean(axis=0)
    r0 = float(np.mean([s[2] for s in seeds]))
    spread_before = float(np.sqrt(np.mean(
        np.sum((seed_centres - c0) ** 2, axis=1)))) if len(names) > 1 else 0.0

    gauge_weight = 1e3 * max(1.0, float(np.mean([len(p) for p in subs])))

    def residuals(x):
        cx, cy, r = x[0], x[1], x[2]
        d = x[3:].reshape(len(names), 2)
        out = [np.linalg.norm(p + d[i] - [cx, cy], axis=1) - r
               for i, p in enumerate(subs)]
        out.append(np.array([gauge_weight * d[:, 0].sum(),
                             gauge_weight * d[:, 1].sum()]))
        return np.concatenate(out)

    x0 = np.concatenate(([c0[0], c0[1], r0], np.zeros(2 * len(names))))
    sol = least_squares(residuals, x0, loss="soft_l1", f_scale=1.0, max_nfev=200)
    cx, cy, r = float(sol.x[0]), float(sol.x[1]), float(sol.x[2])
    d = sol.x[3:].reshape(len(names), 2)
    per_view, offsets = {}, {}
    for i, name in enumerate(names):
        res = np.linalg.norm(subs[i] + d[i] - [cx, cy], axis=1) - r
        per_view[name] = float(np.sqrt(np.mean(res ** 2)))
        offsets[name] = (float(d[i, 0]), float(d[i, 1]))
    return {"consensus_center_mm": (cx, cy), "consensus_radius_mm": r,
            "offsets_mm": offsets, "residual_rms_mm": per_view,
            "spread_before_mm": spread_before,
            "residual_after_mm": float(np.sqrt(np.mean(
                [v ** 2 for v in per_view.values()])))}


def merge_views(views: list[ViewCloud], *, plan, layer, config) -> MergeResult:
    """Level, register and concatenate. Never raises; degrades to the top view.

    Drops are normal, not errors (spec section 8): a tilted view can miss the
    ring, land badly, or lose its colour gate, and the take must still complete
    on what is left. A view whose CHROMA GATE ABSTAINED is dropped rather than
    contributed, because the deposit floor is a property of the merged cloud --
    letting one abstainer through would drag the whole merge to the 2.5 mm floor,
    which on 2026-08-29 cost a 45 deg sector and made all four takes invalid.
    """
    recipe, setup = plan.recipe, plan.setup
    centre = np.array([float(setup.center_x_mm), float(setup.center_y_mm)])
    r_hi = float(recipe.radius_mm) + float(config.radial_roi_margin_mm)
    r_lo = float(recipe.radius_mm) - float(config.radial_roi_margin_mm)
    max_z = (float(layer.nominal_z_mm) + float(recipe.bead_diameter_mm) / 2
             + float(config.deposit_height_margin_mm))

    top = next((v for v in views if v.name == "top"), views[0] if views else None)
    if top is None:
        raise ValueError("merge_views needs at least the top view")

    levelled: dict[str, np.ndarray] = {}
    fit_xy: dict[str, np.ndarray] = {}
    dropped: dict[str, str] = {}
    for view in views:
        if not view.chroma_gated:
            dropped[view.name] = "colour gate abstained; would move the merged floor"
            continue
        try:
            pts, _ = level_points(view.points, r_inner_mm=r_hi,
                                  r_outer_mm=r_hi + config.multiview_level_annulus_width_mm,
                                  center_xy=centre, config=config)
        except ValueError as exc:
            dropped[view.name] = f"levelling failed: {exc}"
            continue
        # The fit subset only: the chain's own height x radial band, so the
        # circle is fitted to deposit rather than to the whole gated cloud. The
        # offsets it solves are applied to the FULL cloud, so nothing is lost.
        #
        # The lower bound MUST be the chain's own deposit floor, not zero. The
        # surface reads a millimetre or so either side of z = 0 after levelling,
        # so a zero floor pulls a ring-shaped slab of board into the circle fit
        # and biases the centre it is trying to measure.
        radius = np.linalg.norm(pts[:, :2] - centre, axis=1)
        min_z = deposit_floor_mm(config, True)      # gated: abstainers already dropped
        band = pts[(radius >= r_lo) & (radius <= r_hi)
                   & (pts[:, 2] >= min_z) & (pts[:, 2] <= max_z)]
        if len(band) < int(config.multiview_min_view_points):
            dropped[view.name] = f"only {len(band)} ring points, need {config.multiview_min_view_points}"
            continue
        seed = fit_circle(band[:, :2])
        span = arc_span_deg(band[:, :2], seed[:2])
        if span < float(config.multiview_min_arc_deg):
            dropped[view.name] = f"sees {span:.0f} deg of arc, need {config.multiview_min_arc_deg:.0f}"
            continue
        levelled[view.name] = pts
        fit_xy[view.name] = band[:, :2]

    def top_only(reason: str | None = None) -> MergeResult:
        if reason:
            dropped.setdefault("__merge__", reason)
        return MergeResult(points=np.asarray(top.points, dtype=float),
                           chroma_gated=bool(top.chroma_gated),
                           used=["top"], dropped=dropped)

    if len(fit_xy) < int(config.multiview_min_views):
        return top_only(f"only {len(fit_xy)} view(s) usable, "
                        f"need {config.multiview_min_views}")

    solved = solve_view_offsets(fit_xy, config)
    far = [n for n, d in solved["offsets_mm"].items()
           if float(np.hypot(*d)) > float(config.multiview_max_offset_mm)]
    if far:
        for name in far:
            dropped[name] = (f"solved offset {np.hypot(*solved['offsets_mm'][name]):.1f} mm "
                             f"exceeds {config.multiview_max_offset_mm:.1f} mm")
            fit_xy.pop(name, None)
            levelled.pop(name, None)
        if len(fit_xy) < int(config.multiview_min_views):
            return top_only("registration rejected too many views")
        solved = solve_view_offsets(fit_xy, config)

    merged = []
    for name, pts in levelled.items():
        dx, dy = solved["offsets_mm"][name]
        merged.append(pts + np.array([dx, dy, 0.0]))
    return MergeResult(
        points=np.vstack(merged), chroma_gated=True,
        used=list(levelled), dropped=dropped,
        consensus_center_mm=solved["consensus_center_mm"],
        consensus_radius_mm=solved["consensus_radius_mm"],
        offsets_mm=solved["offsets_mm"], residual_rms_mm=solved["residual_rms_mm"],
        spread_before_mm=solved["spread_before_mm"],
        residual_after_mm=solved["residual_after_mm"])
```

- [ ] **Step 5: Run the tests and make sure they pass**

Run: `py -3.10 -m pytest tests/test_extrusion_multiview.py -q`
Expected: all pass.

- [ ] **Step 6: Add the drop-behaviour tests**

```python
def _cloud(name, *, gated=True, cx=200.0, cy=150.0, r=40.5, n=1500, seed=0):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, 2 * np.pi, n)
    rad = r + rng.normal(0.0, 1.0, n)
    ring = np.column_stack((cx + rad * np.cos(theta), cy + rad * np.sin(theta),
                            rng.normal(6.0, 0.4, n)))
    board = np.column_stack((rng.uniform(cx - 160, cx + 160, 8000),
                             rng.uniform(cy - 160, cy + 160, 8000),
                             rng.normal(0.0, 0.3, 8000)))
    return ViewCloud(name=name, points=np.vstack((ring, board)), chroma_gated=gated)


def test_an_abstaining_view_is_dropped_not_contributed():
    from tests.test_extrusion_measure import scene_plan
    plan = scene_plan(radius=40.5, bead=10.0)
    views = [_cloud("top", seed=0), _cloud("star-000", seed=1),
             _cloud("star-120", seed=2), _cloud("star-240", gated=False, seed=3)]
    out = merge_views(views, plan=plan, layer=plan.layers[0], config=ExtrusionConfig())
    assert "star-240" in out.dropped and "colour gate" in out.dropped["star-240"]
    assert out.chroma_gated is True                # the merge keeps the 1.5 mm floor
    assert set(out.used) == {"top", "star-000", "star-120"}


def test_all_abstaining_falls_back_to_the_top_view():
    from tests.test_extrusion_measure import scene_plan
    plan = scene_plan(radius=40.5, bead=10.0)
    views = [_cloud(n, gated=False, seed=i) for i, n in
             enumerate(["top", "star-000", "star-120", "star-240"])]
    out = merge_views(views, plan=plan, layer=plan.layers[0], config=ExtrusionConfig())
    assert out.used == ["top"] and out.chroma_gated is False
    np.testing.assert_array_equal(out.points, views[0].points)


def test_a_wildly_misregistered_view_is_rejected_and_the_rest_survive():
    from tests.test_extrusion_measure import scene_plan
    plan = scene_plan(radius=40.5, bead=10.0)
    views = [_cloud("top", seed=0), _cloud("star-000", seed=1),
             _cloud("star-120", seed=2), _cloud("star-240", cx=230.0, seed=3)]
    out = merge_views(views, plan=plan, layer=plan.layers[0], config=ExtrusionConfig())
    assert "star-240" in out.dropped
    assert set(out.used) == {"top", "star-000", "star-120"}
```

Run: `py -3.10 -m pytest tests/test_extrusion_multiview.py -q`
Expected: all pass.

- [ ] **Step 7: Commit and push**

```bash
git add tasni/modules/extrusion/multiview.py tests/extrusion_synthetic.py tests/test_extrusion_multiview.py
git commit -m "feat(extrusion): level and register ring views against one shared circle"
git push
```

---

### Task 6: Capture the views

Wires the star into the jobs and onto disk. **No behaviour change when `multiview` is off.**

**Files:**
- Modify: `tasni/modules/extrusion/measure.py` (`_one_excursion`, `_one_take`, `RingMeasureJob.__init__`, `RingCharacterizeJob`)
- Modify: `tasni/modules/extrusion/archive.py` (`write_layer`)
- Test: `tests/test_extrusion_multiview.py`

**Interfaces:**
- Consumes: Tasks 1–5.
- Produces: `capture_views`, `write_layer(views=…, merged_points_xyz=…)`, `RingMeasureJob(multiview=…)`, `RingCharacterizeJob(multiview=…)`.

- [ ] **Step 1: Write the failing tests**

```python
def test_archive_writes_the_views_directory_and_the_merged_cloud(tmp_path):
    from tasni.modules.extrusion.archive import ExtrusionArchive
    from tasni.modules.extrusion.models import CaptureRecord, LayerManifest, ViewRecord
    from tests.test_extrusion_measure import scene_plan
    plan = scene_plan()
    archive = ExtrusionArchive(tmp_path)
    archive.create_trial("t1", plan)
    color = np.zeros((8, 8, 3), np.uint8)
    depth = np.ones((8, 8), np.uint16)
    manifest = LayerManifest(
        trial_id="t1", layer_index=1, recipe=plan.recipe,
        toolpath_fingerprint=plan.fingerprint,
        capture=CaptureRecord(style="star", merged_points_file="merged_points.npy",
                              views=[ViewRecord(name="star-120", tilt_deg=15.0,
                                                azimuth_deg=120.0)]))
    layer_dir = archive.write_layer(
        manifest, nominal_xyz=np.zeros((3, 3)), commanded_xyz=np.zeros((3, 3)),
        color=color, depth=depth,
        views=[{"name": "star-120", "color": color, "depth": depth,
                "pose": {"tilt_deg": 15.0}}],
        merged_points_xyz=np.zeros((5, 3)))
    assert (layer_dir / "color.png").is_file()          # top view stays at the root
    assert (layer_dir / "views" / "star-120" / "color.png").is_file()
    assert (layer_dir / "views" / "star-120" / "depth.npy").is_file()
    assert (layer_dir / "views" / "star-120" / "pose.json").is_file()
    assert (layer_dir / "merged_points.npy").is_file()


def test_single_view_take_writes_no_views_directory(tmp_path):
    from tasni.modules.extrusion.archive import ExtrusionArchive
    from tasni.modules.extrusion.models import LayerManifest
    from tests.test_extrusion_measure import scene_plan
    plan = scene_plan()
    archive = ExtrusionArchive(tmp_path)
    archive.create_trial("t1", plan)
    manifest = LayerManifest(trial_id="t1", layer_index=1, recipe=plan.recipe,
                             toolpath_fingerprint=plan.fingerprint)
    layer_dir = archive.write_layer(manifest, nominal_xyz=np.zeros((3, 3)),
                                    commanded_xyz=np.zeros((3, 3)),
                                    color=np.zeros((8, 8, 3), np.uint8),
                                    depth=np.ones((8, 8), np.uint16))
    assert not (layer_dir / "views").exists()
```

- [ ] **Step 2: Run them to make sure they fail**

Run: `py -3.10 -m pytest tests/test_extrusion_multiview.py -q -k archive`
Expected: FAIL — `TypeError: write_layer() got an unexpected keyword argument 'views'`.

- [ ] **Step 3: Extend the archive**

In `archive.py`, add two keyword-only parameters to `write_layer` and write them after the existing `color`/`depth` block:

```python
        # Extra views of the SAME take. The top view deliberately stays at the
        # layer root as color.png/depth.npy so reprocess_saved_layer, figures.py
        # and the archived ring fixtures keep working with no change at all.
        if views:
            import cv2
            for view in views:
                out = layer / "views" / _segment(str(view["name"]), "view name")
                out.mkdir(parents=True, exist_ok=False)
                if view.get("color") is not None:
                    if not cv2.imwrite(str(out / "color.png"), np.asarray(view["color"])):
                        raise OSError(f"failed to write {out / 'color.png'}")
                if view.get("depth") is not None:
                    np.save(out / "depth.npy", np.asarray(view["depth"]))
                (out / "pose.json").write_text(
                    json.dumps(view.get("pose") or {}, indent=2), encoding="utf-8")
        if merged_points_xyz is not None:
            np.save(layer / "merged_points.npy", np.asarray(merged_points_xyz, float))
```

- [ ] **Step 4: Add `capture_views` to `measure.py`**

```python
def capture_views(services, ctx: JobContext, plan: CylinderPlan, layer, *,
                  inspection_name: str, start_joints, seed_pose, collisions: bool,
                  artifacts: list[str], near_mm=None, repeats: int = 1) -> dict:
    """Visit every star pose once, grabbing ``repeats`` frames at each.

    The arm visits each pose EXACTLY once: take k is later assembled from the
    k-th frame of every view, so repeats = 3 costs 4 moves, not 12. That keeps
    what ``repeats`` has always meant -- frames with the arm parked, robot
    re-approach excluded by construction -- instead of quietly turning it into
    an averaging window.

    A view that cannot be reached or cannot be trusted is recorded and skipped;
    only the TOP view failing fails the take, exactly as today.
    """
    from .inspection import star_view_angles
    ecfg = services.config.extrusion
    out = {"views": [], "records": []}
    for name, tilt, azimuth in star_view_angles(ecfg):
        program = (inspection_name if name == "top"
                   else f"{inspection_name}_{name.replace('-', '')}")
        record = {"name": name, "tilt_deg": tilt, "azimuth_deg": azimuth}
        try:
            ctx.check_cancel()
            moved = _move_to_inspection(
                services, ctx, plan, layer, inspection_name=program,
                start_joints=start_joints, seed_pose=seed_pose,
                collisions=collisions, artifacts=artifacts, near_mm=near_mm,
                tilt_deg=tilt, azimuth_deg=azimuth)
            frames = [_capture_at_pose(services, ctx, moved["T_work_camera"])
                      for _ in range(max(1, int(repeats)))]
        except Exception as exc:
            if name == "top":
                raise
            record.update({"dropped": True, "drop_reason": str(exc)[:200]})
            ctx.log(f"view {name} dropped: {exc}")
            out["records"].append(record)
            continue
        record.update({
            "T_work_camera": np.asarray(moved["T_work_camera"], float).tolist(),
            "roll_deg": moved["inspect"]["pose"].get("roll_deg")})
        out["views"].append({"name": name, "moved": moved, "frames": frames,
                             "tilt_deg": tilt, "azimuth_deg": azimuth})
        out["records"].append(record)
    return out
```

`_move_to_inspection` gains `tilt_deg=0.0, azimuth_deg=0.0` keyword arguments passed through to `_build_inspection_move`, which selects `star_view_candidates(...)` instead of `pose_candidates(...)` whenever `tilt_deg` is non-zero. **When both are 0 it must take exactly the path it takes today.**

- [ ] **Step 5: Wire the jobs**

`RingMeasureJob.__init__` and `RingCharacterizeJob.__init__` gain `multiview: bool | None = None`, resolved exactly like `side_photo` already is:

```python
        self.multiview = (services.config.extrusion.multiview_enabled
                          if multiview is None else bool(multiview))
```

`_one_excursion` branches once at the top: single view calls `_move_to_inspection` as today; multi-view calls `capture_views`. `_one_take` gains a `views=None` argument; when given it builds `ViewCloud`s via `observation_points`, calls `merge_views`, and passes the merged cloud to `process_points`. When `views is None` it calls `process_observation` exactly as today.

Build the `CaptureRecord` from the merge result and attach it to the manifest.

- [ ] **Step 6: Run everything**

Run the full targeted command from Global Constraints.
Expected: all pass, and `tests/test_extrusion_measure.py` **unedited**.

- [ ] **Step 7: Commit and push**

```bash
git add tasni/modules/extrusion/measure.py tasni/modules/extrusion/archive.py tests/test_extrusion_multiview.py
git commit -m "feat(extrusion): capture and merge the star views"
git push
```

---

### Task 7: Reprocess `views=` and the offline A/B

This is what makes the cell protocol cheap: every star take becomes its own paired control.

**Files:**
- Modify: `tasni/modules/extrusion/service.py:1134` (`reprocess_saved_layer`)
- Create: `tools/multiview_ab.py`
- Test: `tests/test_extrusion_multiview.py`

- [ ] **Step 1: Write the failing test**

```python
def test_reprocess_top_only_equals_the_single_view_result(tmp_path):
    """The A/B's control arm. Same capture, two reconstructions -- paired on the
    identical ring placement, which the operator cannot reproduce by hand."""
    pytest.importorskip("open3d")
    from tasni.modules.extrusion.service import reprocess_saved_layer
    # (build a star take on disk via the archive, then:)
    merged = reprocess_saved_layer(tmp_path, "t1", 1, views="as_archived")
    top = reprocess_saved_layer(tmp_path, "t1", 1, views="top_only")
    assert merged["capture"]["style"] == "star"
    assert top["capture"]["style"] == "single"
    assert top["metrics"]["measured_center_mm"] != merged["metrics"]["measured_center_mm"]
```

- [ ] **Step 2: Run it, see it fail** — `TypeError: unexpected keyword argument 'views'`.

- [ ] **Step 3: Implement.** `reprocess_saved_layer` gains `views: str = "as_archived"`. When `"as_archived"` and `layer_dir/views/` exists, load every view's `color.png`/`depth.npy`/`pose.json`, rebuild `ViewCloud`s through `observation_points`, and run `merge_views` → `process_points`. When `"top_only"`, ignore `views/` entirely and take today's path.

- [ ] **Step 4: Write `tools/multiview_ab.py`** — walks a trial, reprocesses every take both ways, and prints one table: bead-width spread, crest-height range, completeness, max angular gap, `after_work_roi` (**pre-voxel** — say so in the header), centre spread across repeats, `spread_before_mm` and `residual_after_mm`. It must print the voxel warning verbatim:

```
NOTE: the chain voxel-downsamples at 1 mm. Merging four views does NOT multiply
the surviving point count -- it multiplies the samples each voxel averages and
fills dropouts. Read after_work_roi (pre-voxel), never after_voxel.
```

- [ ] **Step 5: Run the targeted command.** Expected: all pass.

- [ ] **Step 6: Commit and push**

```bash
git add tasni/modules/extrusion/service.py tools/multiview_ab.py tests/test_extrusion_multiview.py
git commit -m "feat(extrusion): reprocess a star take top-only, and an offline A/B"
git push
```

---

### Task 8: Figures and the paper's statistics guards

**Files:**
- Modify: `tasni/modules/extrusion/figures.py`, `tasni/modules/extrusion/measure.py` (`capture_style`, `paper_summary`)
- Test: `tests/test_extrusion_multiview.py`, `tests/test_extrusion_figures.py`

- [ ] **Step 1: Write the failing test**

```python
def test_capture_style_reports_star_and_paper_summary_refuses_to_pool():
    """Merged and single-view takes measure the same ring differently, and
    acquisition_to_path_ms means something different for each. Pooling them
    would put a number in the paper that describes neither."""
    from tasni.modules.extrusion.measure import capture_style
    star = [{"provenance": {"excursion_index": 1, "repeats_in_excursion": 1},
             "capture": {"style": "star"}} for _ in range(3)]
    assert capture_style(star) == "star"
    mixed = star + [{"provenance": {"excursion_index": 2, "repeats_in_excursion": 1},
                     "capture": {"style": "single"}}]
    assert capture_style(mixed) == "mixed"
```

- [ ] **Step 2: Run it, see it fail.**

- [ ] **Step 3: Implement.** `capture_style` reads `manifest["capture"]["style"]` first: all `"star"` → `"star"`; a mix of styles → `"mixed"`; otherwise fall through to today's parked/re-approach logic unchanged. `paper_summary` groups by capture style and never averages across groups, exactly as it already does for offline-reprocessed takes. Add the `views` figure to `figures.py`: the four colour frames with each view's fitted ring and solved offset drawn on, plus the merged cloud.

- [ ] **Step 4: Run the targeted command.** Expected: all pass.

- [ ] **Step 5: Commit and push**

```bash
git add tasni/modules/extrusion/figures.py tasni/modules/extrusion/measure.py tests/test_extrusion_multiview.py
git commit -m "feat(extrusion): a views figure, and keep merged takes out of single-view statistics"
git push
```

---

### Task 9: API and both UI toggles

**Files:**
- Modify: `tasni/modules/extrusion/module.py`, `tasni/webui/src/pages/Extrusion.tsx`

- [ ] **Step 1: Add `multiview` to the request bodies**

```python
    # Capture the ring from the top view plus three tilted views merged into one
    # cloud. None = follow the configured default (off). Independent of
    # side_photo: either, both or neither.
    multiview: bool | None = None
```

on `MeasureLayerBody` **and** `CharacterizeBody`, passed straight to the jobs.

- [ ] **Step 2: Surface BOTH toggles in the UI**

`side_photo` currently exists in the API and appears **nowhere** in `Extrusion.tsx` — it only ever runs on its config default. Add two independent checkboxes to the measure card:

- "Multi-view capture — 4 trips instead of 1 (~15 s of arm time)", default unchecked.
- "Side photo — one extra excursion after the capture", default checked.

Both flow into the measure/characterize request bodies. Neither disables the other.

- [ ] **Step 3: Build the frontend**

Run: `cd tasni/webui && npm run build`
Expected: clean build, no TypeScript errors.

- [ ] **Step 4: Run the targeted command.** Expected: all pass.

- [ ] **Step 5: Commit and push**

```bash
git add tasni/modules/extrusion/module.py tasni/webui/src/pages/Extrusion.tsx
git commit -m "feat(extrusion ui): independent multi-view and side-photo toggles"
git push
```

---

### Task 10: Docs, and the handoff for the cell A/B

**Files:**
- Modify: `docs/pfh-paper-handoff.md`, `docs/extrusion-current-handoff.md`, `AGENTS.md`, `CLAUDE.md`

- [ ] **Step 1: Write the cell protocol into `docs/extrusion-current-handoff.md`** — spec §10 verbatim: one ring, unmoved, multi-view ON, `repeats = 3`, tilt 10 then 15 then 20, then `py -3.10 tools/multiview_ab.py <trial>`. Include the decision rule (profile improves **and** residual below the 1.26 mm hand-eye floor → may go in the paper and the winning tilt becomes the default; profile improves but residual at or above it → qualitative claim only, default stays OFF; no improvement → keep OFF, keep the code, **write the negative result down**).

- [ ] **Step 2: Update `AGENTS.md`** — move multi-view from DESIGNED to BUILT (opt-in, default OFF, awaiting the on-cell A/B), and keep the three traps: do not start before the paper run; the arrival gate was straight-down-only; ChArUco is out of scope by operator decision.

- [ ] **Step 3: Update `docs/pfh-paper-handoff.md` §4b** to point at the built feature and its A/B.

- [ ] **Step 4: Add a `multiview` line to `CLAUDE.md`'s extrusion section.**

- [ ] **Step 5: Commit, push, and merge**

```bash
git add docs/ AGENTS.md CLAUDE.md
git commit -m "docs(extrusion): the multi-view cell A/B protocol and its decision rule"
git push
git checkout main && git merge --no-ff multiview-inspection && git push
```

---

## Self-review (done while writing)

**The numeric assertions in Tasks 3 and 5 were executed, not estimated.** The
`solve_view_offsets`, `fit_circle`, `level_points` and `depth_plane_check` code in this
plan was run standalone before the plan was committed, and the tolerances written into the
tests are what it actually produces:

| assertion | tolerance in the test | measured |
|---|---|---|
| `fit_circle` recovers (200, 150, 40.5) | ±0.05 | (200.007, 149.995, 40.497) |
| injected per-view offsets recovered | ±0.12 mm | worst 0.014 mm |
| offsets sum to zero (the gauge) | ±1e-6 | exactly 0 |
| displacing the top view moves the consensus by 4.0/4 | ±0.15 | **1.0000** / dy −0.0000 |
| a +8.6 % scaled view's residual vs the others | > 5× | **7.8×** |
| levelling `level_mm` on an injected 3.0 mm offset | ±0.5 | 2.772 |
| levelling removes an injected 8° tilt | — | 8.0000° removed, median z 0.0 |
| gate bias across 300–800 mm × 0–30° | 5.0–5.8 mm | 5.00–5.77 mm |

If an implementer's run disagrees with a row above, the implementation has drifted from
this plan — do not loosen the tolerance to make it pass.

**Spec coverage.** §5.1 → Task 2. §5.2 gates → Task 3; capture → Task 6. §5.3 → Task 5. §5.4 seam → Task 4. §5.5 → Tasks 1 and 6. §5.6 → Task 7. §5.7 → Task 8. §5.8 → Task 9. §5.9 → Task 1. §8 error table → Task 5 (merge-side drops) and Task 6 (capture-side drops). §9 tests → distributed across every task. §10 cell protocol → Task 10. §11 live print is explicitly a non-goal. Every spec section maps to a task.

**Three things worth flagging to whoever executes this.**

1. **Task 3 is independently mergeable and independently useful.** It fixes a latent bug on the *existing* single-view path: the gate's sensitivity already decays with standoff, so a large ring measured at 500 mm has less fault margin than a small one at 300 mm. If the rest of this plan is deferred, Task 3 should still land.
2. **Task 5 Step 1 is not optional scaffolding.** The existing synthetic fixtures pass an all-zero colour frame, which makes `chroma_gate_mask` abstain. Under the new per-view drop rule that would drop every view in every test, so the merge tests would silently exercise only the fallback and pass while proving nothing.
3. **Task 4's proof is the tests you do *not* edit.** If `tests/test_extrusion_processing.py` or `tests/test_extrusion_measure.py` needs a single line changed to go green, the refactor changed behaviour and every archived number's meaning moved with it. Revert and redo rather than adjusting the test.
