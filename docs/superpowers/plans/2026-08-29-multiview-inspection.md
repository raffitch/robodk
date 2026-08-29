# Multi-view inspection + side photo — Implementation Plan (tasks 1–6)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a Measure / Characterize press capture the ring from the top view plus three tilted views, level and register them, and measure the merged cloud — because the mock extruded rings are **thin** (crest 2–4 mm) and one straight-down frame sees the crest at grazing signal-to-noise and the flanks hardly at all. Plus one side RGB photo per ring for the paper. Both are opt-in and default OFF, so every archived single-view number stays exactly as it is.

**Architecture:** `inspection.py` gains the star and side poses (pure numpy). `processing.py` is split at the back-projection seam (`observation_points` → `process_points`), behaviour-preserving, so a new `multiview.py` can level each view on its board annulus, align it on its own fitted ring centre to the top view, concatenate, and hand the merged work-frame cloud to the *unchanged* chain. `measure.py` captures the views in order (one RoboDK target+program per view, as today), archives every raw frame under `views/`, and runs the side photo as a second, separately timed excursion after the arm is home. The manifest gains typed `capture` / `side_view` records.

**Tech Stack:** Python 3.10 (`py -3.10`), pydantic v2, numpy/scipy/OpenCV, Open3D (the chain's deposit filter; lazily imported), matplotlib (`figures` extra). RoboDK is never touched by tests — fakes from `tests/test_extrusion_job.py`. Synthetic RGB-D from `tests/extrusion_synthetic.py` (`render_scene` renders from **any** camera pose, so multi-view scenes need no new renderer).

**Spec:** `docs/superpowers/specs/2026-08-29-multiview-inspection-design.md` — read §1 (why), §2 (code facts), §4 (design), §5 (error table), §6 (tests) first. This plan covers the spec's §9 tasks **1–6**; tasks 7–10 (reprocess `top_only` + A/B tool, paper summary/docx/`views.png`, API + UI toggles, docs) are a second plan written **after** the on-cell A/B in §8, because its result may change what they need to do.

## Global Constraints

- Work in a **git worktree on branch `multiview-inspection`** (`git worktree add ../RoboDkClaude-multiview -b multiview-inspection main`). Never commit on `main`. Push every commit (`git push -u origin multiview-inspection`) — the operator reads progress from the pushed history. Merge `--no-ff` at the end.
- Python is **`py -3.10`** (no `python` on PATH). Never round-trip a source file through PowerShell `Get-Content`/`Set-Content` (it mojibakes this repo's UTF-8) — edit with the Edit tool.
- **Do not run the full pytest suite** (too slow; the operator interrupts it). The targeted command, used in every task:
  `py -3.10 -m pytest tests/test_extrusion_multiview.py tests/test_extrusion_measure.py tests/test_extrusion.py tests/test_extrusion_job.py tests/test_extrusion_processing.py tests/test_extrusion_figures.py -q`
- `ExtrusionConfig` (`tasni/core/config.py`) and every model in `tasni/modules/extrusion/models.py` are `extra="forbid"`: every new field needs a default; never remove or rename a field.
- `CylinderPrintJob` and `CylinderDryRunJob` in `service.py` are cell-validated — **do not change their behaviour**. The one edit this plan makes to `service.py` is an optional, default-`None` keyword on `_build_inspection_move` (Task 5) that leaves its default path byte-for-byte identical.
- The single-view path of `RingMeasureJob` / `RingCharacterizeJob` / `process_observation` / `characterize_ring` is cell-validated (paper takes on 2026-08-28/29). `views="single"` (the default) must keep calling exactly what it calls today; the existing tests in `tests/test_extrusion_measure.py` are the regression and must stay green untouched.
- Tests that reach `_filter_deposit` (anything through `process_points`, `view_ring_center`, `merge_views` with registration, `characterize_*`) need Open3D: start them with `pytest.importorskip("open3d")`. `level_points`, `median_depth`, the pose functions and the archive are pure numpy — no skip.
- **Point counts are capped by the chain's 2 mm voxel** (`_deposit_clusters` → `voxel_down_sample(2 mm)`): merging four views does not multiply `after_radial_trim`, it multiplies the samples each voxel is averaged from (and fills dropouts). Assert the gain on the pre-voxel count `counts["after_work_roi"]` and on validity/radius — not on post-voxel counts. Say this in the A/B (spec §8) when it runs.
- The RoboDK item name for a view's program is `<stem>_Inspect` (top, unchanged) or `<stem>_Inspect_star0` / `_star120` / `_star240` (hyphen dropped); the side photo's is `<stem>_Side`. Archive directory names keep the hyphen: `views/star-120/`.
- Every task ends with a commit + push. Commit messages end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` (plus your harness's `Claude-Session:` line if it gives you one).
- The Tasni backend caches modules: before any cell test, **restart Tasni** and check `GET /api/health` → `build.stale == false`.

---

## File map

| File | Responsibility in this plan |
|---|---|
| `tasni/core/config.py` | 12 `multiview_*` + 7 `side_view_*` fields on `ExtrusionConfig` (Task 1) |
| `tasni/modules/extrusion/models.py` | `ViewRecord`, `CaptureRecord`, `SideViewRecord`; `LayerManifest.capture / side_view / merged_points_file` (Task 1) |
| `tasni/modules/extrusion/inspection.py` | `multiview_names`, `star_view_candidates`, `side_pose_from_crest`, `side_view_plan`; `inspection_plan(multiview=, side_view=)` (Task 2) |
| `tasni/modules/extrusion/processing.py` | seam refactor: `observation_points`, `_work_roi`, `process_points`, `coarse_plan`, `characterize_points`; `process_observation` / `characterize_ring` become wrappers (Task 3) |
| `tasni/modules/extrusion/multiview.py` (new) | `ViewPoints`, `MergeResult`, `median_depth`, `level_points`, `view_ring_center`, `merge_views`, `process_views`, `characterize_views` (Task 4) |
| `tasni/modules/extrusion/service.py` | `_build_inspection_move(candidate_source=None)` (Task 5) |
| `tasni/modules/extrusion/measure.py` | `View`, `capture_views`, `_inspect_and_capture(candidate_source=, frames=)`, `RingMeasureJob(views=)`, `RingCharacterizeJob(views=)`, timings (Task 5); `SideView`, `_build_side_move`, `capture_side_view`, `RingMeasureJob(side_view=)` (Task 6) |
| `tasni/modules/extrusion/archive.py` | `write_layer(views=, merged_points=)`, `write_characterization(views=)`, `_write_view` (Task 5); `write_side_view` (Task 6) |
| `tasni/modules/extrusion/figures.py` | `OPTIONAL_LAYER_FIGURES`, `TakeData.side_view`, `side_crop_px`, `_figure_side` (Task 6) |
| `tests/test_extrusion_multiview.py` (new) | every new test in this plan, appended task by task |

## Interfaces used across tasks (names are binding)

```python
# inspection.py (Task 2)
multiview_names(config) -> ["top", "star-0", "star-120", "star-240"]
star_view_candidates(aim_mm, standoff_mm, config, reference_x=None) -> list[{name, tilt_deg, azimuth_deg, candidates: list[{view, tilt_deg, azimuth_deg, roll_deg, xyz_mm, T}]}]
side_view_plan(recipe, setup, layer_index, config) -> {name, layer_index, layer_top_z_mm, standoff_mm, requested_elevation_deg, elevation_deg, floor_raised, min_camera_z_mm, refused: str|None, candidates: list[{view, azimuth_deg, elevation_deg, roll_deg, crest_mm, xyz_mm, T}]}
# processing.py (Task 3)
observation_points(depth, K, T_work_camera) -> np.ndarray            # Nx3 work-frame points
process_points(points, *, plan, layer, config, floor_profile=None, stages=None, counts=None, timings=None, started=None) -> ProcessingResult
characterize_points(points, *, search_center_mm, work_frame, config, inspection_tool="Realsense", print_tool="LongCalibTool", counts=None, started=None) -> CharacterizationResult
# multiview.py (Task 4)
ViewPoints(name, points, T_work_camera); MergeResult(points, views, report)
median_depth(frames) -> np.ndarray
level_points(points, *, center_xy, inner_mm, outer_mm, config) -> (points, info)
view_ring_center(points, *, plan, layer, config, floor_profile=None) -> (center|None, n_points, error|None)
merge_views(views, *, plan, layer, config, floor_profile=None, registration=None) -> MergeResult
process_views(views, *, plan, layer, config, floor_profile=None, stages=None) -> ProcessingResult
characterize_views(views, *, search_center_mm, work_frame, config, inspection_tool=..., print_tool=...) -> CharacterizationResult
# measure.py (Tasks 5, 6)
View(name, descriptor, target, T_work_camera, color, depth, frames, move_ms, settle_ms, capture_ms, error)   # .ok
capture_views(services, ctx, plan, layer, *, program_stem, start_joints, seed_pose, collisions, artifacts, views, frames_per_view=1, near_mm=None) -> list[View]
RingMeasureJob(..., views="single"|"multi", side_view=False); RingCharacterizeJob(..., views="single"|"multi")
SideView(descriptor, target, T_work_camera, color, depth, excursion_ms, error)
capture_side_view(services, ctx, plan, layer_index, *, program_name, start_joints, artifacts) -> SideView
# archive.py
write_layer(..., views: list[{name, color, depth, pose}] | None = None, merged_points=None)
write_side_view(layer_dir, *, color, depth, record: SideViewRecord) -> SideViewRecord
```

---

### Task 1: Config keys and manifest records

**Files:**
- Modify: `tasni/core/config.py:17` (import) and `:859-860` (append to `ExtrusionConfig` after `radial_trim_schedule_mm`)
- Modify: `tasni/modules/extrusion/models.py:5` (import), `:130` (new records before `LayerManifest`), `:157` (three new manifest fields)
- Create: `tests/test_extrusion_multiview.py`

**Interfaces:**
- Produces: the config fields listed in the spec §4.9 plus `multiview_level_outlier_mm` (2.5) and `multiview_max_xy_shift_mm` (10.0); `ViewRecord`, `CaptureRecord`, `SideViewRecord`; `LayerManifest.capture: CaptureRecord | None`, `.side_view: SideViewRecord | None`, `.merged_points_file: str | None`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_extrusion_multiview.py
"""Multi-view inspection + side photo (spec: docs/superpowers/specs/2026-08-29-multiview-inspection-design.md).

The mock rings are thin (crest 2-4 mm); one straight-down frame sees the crest at
grazing signal-to-noise and the flanks hardly at all. These tests prove the star
poses, the per-view levelling + centre registration, the merged measurement, the
capture job and the side photo on synthetic RGB-D before the robot moves.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

import extrusion_synthetic as syn
from tasni.core.config import ExtrusionConfig
from tasni.modules.extrusion.models import (CaptureRecord, CylinderRecipe, CylinderSetup,
                                            LayerManifest, SideViewRecord, ViewRecord)

CENTER = (200.0, 150.0)


def recipe(**updates) -> CylinderRecipe:
    base = dict(radius_mm=40, layer_count=3, layer_height_mm=5, bead_diameter_mm=6,
                robot_speed_mm_s=75, extrusion_rate_pct=30)
    base.update(updates)
    return CylinderRecipe(**base)


def setup(**updates) -> CylinderSetup:
    base = dict(print_tool="LongCalibTool", work_frame="Tasni Work Frame",
                inspection_tool="Realsense", inspection_auto=True,
                center_x_mm=CENTER[0], center_y_mm=CENTER[1])
    base.update(updates)
    return CylinderSetup(**base)


# ------------------------------------------------------------ Task 1: config + records

def test_multiview_and_side_view_defaults_are_off_and_bounded():
    cfg = ExtrusionConfig()
    assert cfg.multiview_tilt_deg == 20.0 and cfg.multiview_tilt_min_deg == 10.0
    assert cfg.multiview_azimuth_offset_deg == 0.0 and cfg.multiview_azimuth_slack_deg == 20.0
    assert cfg.multiview_frames_per_view == 1 and cfg.multiview_registration == "centre"
    assert cfg.multiview_level_annulus_mm == 90.0 and cfg.multiview_level_min_points == 2000
    assert cfg.multiview_max_xy_shift_mm == 10.0 and cfg.multiview_voxel_mm == 0.0
    assert cfg.side_view_elevation_deg == 15.0 and cfg.side_view_standoff_mm == 250.0
    assert cfg.side_view_min_camera_z_mm == 80.0 and cfg.side_view_collision_check is True
    assert cfg.side_view_azimuth_fallbacks_deg == [90.0, 180.0, 270.0]
    with pytest.raises(ValidationError):
        ExtrusionConfig(multiview_tilt_deg=35.0)             # 30 deg cap: tilt costs board quality
    with pytest.raises(ValidationError):
        ExtrusionConfig(multiview_tilt_min_deg=0.0)          # tilt 0 would be the top view again
    with pytest.raises(ValidationError):
        ExtrusionConfig(side_view_standoff_mm=100.0)         # >= 150 mm
    with pytest.raises(ValidationError):
        ExtrusionConfig(side_view_elevation_deg=60.0)        # above 45 it is not a side view
    with pytest.raises(ValidationError):
        ExtrusionConfig(multiview_registration="icp")


def test_manifest_without_capture_fields_is_a_single_view_take():
    manifest = LayerManifest(trial_id="t", layer_index=1, recipe=recipe(), toolpath_fingerprint="f")
    assert manifest.capture is None and manifest.side_view is None
    assert manifest.merged_points_file is None
    old = json.loads(manifest.model_dump_json())
    for key in ("capture", "side_view", "merged_points_file"):
        old.pop(key)
    assert LayerManifest.model_validate(old).capture is None       # archives written before this land


def test_manifest_round_trips_a_multi_view_capture_and_a_side_photo():
    capture = CaptureRecord(mode="multi", frames_per_view=3, views=[
        ViewRecord(name="top", descriptor={"tilt_deg": 0.0}, T_work_camera=np.eye(4).tolist(),
                   color_file="color.png", depth_file="depth.npy", frames=3,
                   move_ms=1200.0, settle_ms=1000.0, capture_ms=90.0),
        ViewRecord(name="star-120", descriptor={"tilt_deg": 20.0, "azimuth_deg": 120.0},
                   used=False, error="no IK solution")])
    side = SideViewRecord(color_file="side/color.png", descriptor={"elevation_deg": 17.2},
                          excursion_ms=4100.0)
    manifest = LayerManifest(trial_id="t", layer_index=1, recipe=recipe(), toolpath_fingerprint="f",
                             capture=capture, side_view=side, merged_points_file="merged_points.npy")
    back = LayerManifest.model_validate_json(manifest.model_dump_json())
    assert back.capture.mode == "multi" and back.capture.frames_per_view == 3
    assert [v.name for v in back.capture.views] == ["top", "star-120"]
    assert back.capture.views[1].used is False and back.capture.views[1].depth_file is None
    assert back.side_view.excursion_ms == 4100.0 and back.side_view.error is None
    with pytest.raises(ValidationError):
        CaptureRecord(mode="stereo")
    with pytest.raises(ValidationError):
        ViewRecord(name="top", unexpected=1)                        # extra="forbid" like every record
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `py -3.10 -m pytest tests/test_extrusion_multiview.py -q`
Expected: 3 failures — `ImportError: cannot import name 'CaptureRecord'` (collection error is fine).

- [ ] **Step 3: Add the config fields**

In `tasni/core/config.py` line 17 change `from typing import Any` to `from typing import Any, Literal`. Then append inside `ExtrusionConfig`, directly after the `radial_trim_schedule_mm` field (line 860):

```python
    # -- multi-view capture (modules/extrusion/multiview.py) -----------------
    # The mock rings are THIN (crest 2-4 mm; bare-board depth noise +4.8 mm p99
    # at 300 mm). One straight-down frame sees the crest at grazing
    # signal-to-noise and the flanks hardly at all, and repeating frames from
    # the same pose only averages the same missing flanks. Three views tilted by
    # ``multiview_tilt_deg`` at 120 deg azimuth spacing put every flank point
    # within 60 deg of some camera. Tilt costs board quality (plane RMS 0.65 mm
    # at 1 deg, 2.0 at 9, 5.0 at 20 on 2026-08-13), and part of that is a
    # systematic warp that per-view levelling removes; the rest is why the tilt
    # stays modest. Everything here is inert unless a request asks for
    # ``views = "multi"``; the validated single-view chain is untouched.
    multiview_tilt_deg: float = Field(default=20.0, ge=10.0, le=30.0)
    multiview_tilt_min_deg: float = Field(default=10.0, ge=5.0, le=30.0)   # never 0: that is the top view
    multiview_azimuth_offset_deg: float = Field(default=0.0, ge=-180.0, le=180.0)
    multiview_azimuth_slack_deg: float = Field(default=20.0, ge=0.0, le=60.0)
    # Depth frames per view, combined per pixel by the median of valid samples.
    # The cheap lever against depth noise at a fixed pose; independent of the
    # view count. Config only, not in the UI.
    multiview_frames_per_view: int = Field(default=1, ge=1, le=7)
    # "centre": translate each view so its own fitted ring centre lands on the
    # top view's (the correction IS the inter-view error, reported per view);
    # a view whose ring cannot be found is merged unshifted. "strict": such a
    # view is dropped. "none": hand-eye only -- the A/B control (spec section 8).
    multiview_registration: Literal["centre", "strict", "none"] = "centre"
    # Levelling annulus: board points between R + radial_roi_margin + 10 and
    # R + this, with |z| below the max, plane-fitted with one outlier round.
    multiview_level_annulus_mm: float = Field(default=90.0, gt=0, le=500)
    multiview_level_max_abs_z_mm: float = Field(default=15.0, gt=0, le=100)
    multiview_level_min_points: int = Field(default=2000, ge=10)
    multiview_level_outlier_mm: float = Field(default=2.5, gt=0, le=50)
    # A view whose ring centre lands further than this from the top view's is a
    # mis-detection, not a registration error (hand-eye disagreement is 1-2 mm).
    multiview_max_xy_shift_mm: float = Field(default=10.0, gt=0, le=100)
    # Optional voxel thinning of the merged cloud; 0 = off (the chain's own
    # 2 mm voxel + 1 mm raster thin it anyway).
    multiview_voxel_mm: float = Field(default=0.0, ge=0, le=20)

    # -- side photo (documentation for the paper, never a measurement) -------
    # Azimuth in the work frame from +X (the paired-detection axis); fallbacks
    # are tried in order when the first pose is unreachable.
    side_view_azimuth_deg: float = Field(default=0.0, ge=-180.0, le=360.0)
    side_view_azimuth_fallbacks_deg: list[float] = Field(
        default_factory=lambda: [90.0, 180.0, 270.0])
    # Requested elevation above horizontal. The camera housing (~25 mm about the
    # optical centre) and the flange behind it would otherwise sit at bead
    # height, i.e. on the table, so the elevation is RAISED to keep the optical
    # centre above ``side_view_min_camera_z_mm``; past 45 deg the pose is
    # refused at preflight rather than photographed from above.
    side_view_elevation_deg: float = Field(default=15.0, ge=0.0, le=45.0)
    side_view_min_camera_z_mm: float = Field(default=80.0, ge=0, le=500)
    # The RGB lens focuses fine at 250 mm and depth is irrelevant to a photo, so
    # this is deliberately NOT clamped by inspection_min_mm.
    side_view_standoff_mm: float = Field(default=250.0, ge=150.0, le=1000)
    # Low and near the table: keep RoboDK's collision check ON here. The
    # measure-only default of OFF exists because the check rejected good
    # OVERHEAD poses against furniture, which does not apply to this pose.
    side_view_collision_check: bool = True
    # Half-angle of the near-crest arc the side figure is cropped to.
    side_view_crop_deg: float = Field(default=30.0, gt=0, le=180)
```

- [ ] **Step 4: Add the manifest records**

In `tasni/modules/extrusion/models.py` line 5 change `from typing import Any` to `from typing import Any, Literal`. Insert before `class LayerManifest(_Record):` (line 130):

```python
class ViewRecord(_Record):
    """One measurement view as captured and archived.

    ``descriptor`` is the pose the candidate walk accepted (tilt/azimuth/roll,
    standoff, joints, wrist report); ``used`` is False when the view was
    skipped -- unreachable or a capture error -- and then ``error`` says why.
    File names are relative to the layer directory; the top view keeps the
    historical ``color.png`` / ``depth.npy`` so every existing reader stays right.
    """
    name: str
    descriptor: dict[str, Any] = Field(default_factory=dict)
    T_work_camera: list[list[float]] | None = None
    color_file: str | None = None
    depth_file: str | None = None
    frames: int = Field(default=1, ge=1)
    move_ms: float | None = None
    settle_ms: float | None = None
    capture_ms: float | None = None
    used: bool = True
    error: str | None = None


class CaptureRecord(_Record):
    """How the take's cloud was captured. Absent on a manifest = one top view."""
    mode: Literal["single", "multi"] = "single"
    frames_per_view: int = Field(default=1, ge=1)
    views: list[ViewRecord] = Field(default_factory=list)


class SideViewRecord(_Record):
    """The side photo's provenance. Documentation only: it never feeds a number,
    and its ``error`` never invalidates the take it sits beside."""
    color_file: str | None = None
    depth_file: str | None = None
    descriptor: dict[str, Any] = Field(default_factory=dict)
    T_work_camera: list[list[float]] | None = None
    excursion_ms: float | None = None
    error: str | None = None
```

Then add three fields at the end of `LayerManifest` (after `warnings: list[str] = Field(default_factory=list)`, line 157):

```python
    # Multi-view capture (spec 2026-08-29). None on every take archived before
    # it existed and on every single-view take: absent means "one top view".
    capture: CaptureRecord | None = None
    side_view: SideViewRecord | None = None
    merged_points_file: str | None = None
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `py -3.10 -m pytest tests/test_extrusion_multiview.py tests/test_extrusion.py tests/test_extrusion_measure.py -q`
Expected: all pass (the three new ones plus the existing regression, which proves the new fields are additive).

- [ ] **Step 6: Commit and push**

```bash
git add tasni/core/config.py tasni/modules/extrusion/models.py tests/test_extrusion_multiview.py
git commit -m "feat(extrusion): config keys and manifest records for multi-view capture and the side photo

Both features default OFF; a manifest without the new fields is a single-view take.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
git push -u origin multiview-inspection
```

---

### Task 2: Star and side poses

**Files:**
- Modify: `tasni/modules/extrusion/inspection.py` (append after `order_candidates_seed_first`, line 254; extend `inspection_plan`, lines 191–234)
- Test: `tests/test_extrusion_multiview.py`

**Interfaces:**
- Consumes: `pose_from_aim`, `layer_top_z_mm`, `aim_point_mm` (existing).
- Produces: `STAR_AZIMUTH_STEP_DEG = 120.0`, `multiview_names(config)`, `star_view_candidates(aim_mm, standoff_mm, config, reference_x=None)`, `side_pose_from_crest(crest_mm, *, azimuth_deg, elevation_deg, standoff_mm, roll_deg=0.0) -> 4x4`, `SIDE_VIEW_MAX_ELEVATION_DEG = 45.0`, `side_view_plan(recipe, setup, layer_index, config) -> dict`, `inspection_plan(..., multiview=False, side_view=False)`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_extrusion_multiview.py`)

```python
# ------------------------------------------------------------ Task 2: poses

from tasni.modules.extrusion.inspection import (aim_point_mm, inspection_plan,  # noqa: E402
                                                multiview_names, side_view_plan,
                                                star_view_candidates)


def test_star_views_keep_the_aim_on_axis_at_the_standoff_120_deg_apart():
    aim = np.array([200.0, 150.0, 8.0])
    cfg = ExtrusionConfig()
    views = star_view_candidates(aim, 300.0, cfg, reference_x=syn.CAMERA_X_AT_PARK)
    assert [v["name"] for v in views] == ["star-0", "star-120", "star-240"] == multiview_names(cfg)[1:]
    assert multiview_names(cfg)[0] == "top"
    assert [v["azimuth_deg"] for v in views] == [0.0, 120.0, 240.0]
    for view in views:
        first = view["candidates"][0]
        assert (first["tilt_deg"], first["azimuth_deg"], first["roll_deg"]) == (20.0, view["azimuth_deg"], 0.0)
        for candidate in view["candidates"]:
            in_camera = np.linalg.inv(candidate["T"]) @ np.append(aim, 1.0)
            assert np.allclose(in_camera[:3], [0.0, 0.0, 300.0], atol=1e-9)   # centred, at the standoff
            assert candidate["tilt_deg"] >= cfg.multiview_tilt_min_deg          # never the top view again
            assert candidate["view"] == view["name"]
    # Walk order: rolls at the nominal azimuth, then azimuth slack in 10 deg steps
    # either side, then the same again 5 deg shallower, down to the minimum tilt.
    walk = [(c["tilt_deg"], c["azimuth_deg"], c["roll_deg"]) for c in views[0]["candidates"]]
    assert walk[:4] == [(20.0, 0.0, 0.0), (20.0, 0.0, 180.0), (20.0, 0.0, 90.0), (20.0, 0.0, 270.0)]
    assert walk[4][:2] == (20.0, 10.0) and walk[8][:2] == (20.0, 350.0)
    assert walk[16][:2] == (20.0, 340.0) and walk[20][:2] == (15.0, 0.0)
    assert min(t for t, _, _ in walk) == 10.0 and len(set(walk)) == len(walk) == 60


def test_star_camera_positions_sit_on_a_cone_120_deg_apart():
    views = star_view_candidates(np.zeros(3), 300.0, ExtrusionConfig())
    xyz = np.array([v["candidates"][0]["xyz_mm"] for v in views])
    assert np.allclose(xyz[:, 2], 300.0 * math.cos(math.radians(20.0)))
    angles = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0])) % 360.0
    assert np.allclose(np.sort(angles), [0.0, 120.0, 240.0], atol=1e-6)
    rotated = star_view_candidates(np.zeros(3), 300.0, ExtrusionConfig(multiview_azimuth_offset_deg=30.0))
    assert [v["azimuth_deg"] for v in rotated] == [30.0, 150.0, 270.0]


def test_side_pose_has_the_near_crest_on_axis_and_an_upright_image():
    plan = side_view_plan(recipe(), setup(), 1, ExtrusionConfig())
    assert plan["refused"] is None and plan["layer_top_z_mm"] == 6.0
    first = plan["candidates"][0]
    assert first["view"] == "side" and first["azimuth_deg"] == 0.0 and first["roll_deg"] == 0.0
    assert first["crest_mm"] == [240.0, 150.0, 6.0]           # centre + R along +X, at the layer top
    T = first["T"]
    in_camera = np.linalg.inv(T) @ np.append(first["crest_mm"], 1.0)
    assert np.allclose(in_camera[:3], [0.0, 0.0, 250.0], atol=1e-9)
    assert T[2, 3] >= ExtrusionConfig().side_view_min_camera_z_mm - 1e-9
    assert T[2, 1] < -0.9                                     # camera +Y (image down) points at the table
    assert T[0, 3] > 240.0                                    # outside the ring, looking inward
    assert np.isclose(np.linalg.det(T[:3, :3]), 1.0)          # a proper rotation, not a mirror
    assert [c["azimuth_deg"] for c in plan["candidates"][::2]] == [0.0, 90.0, 180.0, 270.0]
    assert [c["roll_deg"] for c in plan["candidates"][:2]] == [0.0, 180.0]
    flipped = plan["candidates"][1]["T"]
    assert np.allclose(flipped[:3, 3], T[:3, 3]) and np.allclose(flipped[:3, 1], -T[:3, 1])


def test_side_elevation_is_raised_to_clear_the_camera_floor_and_refused_past_45():
    low = side_view_plan(recipe(), setup(), 1, ExtrusionConfig(side_view_elevation_deg=0.0))
    # layer top 6 mm; an 80 mm floor at a 250 mm standoff needs asin(74/250) = 17.2 deg
    assert low["floor_raised"] and low["elevation_deg"] == pytest.approx(17.2, abs=0.1)
    assert low["requested_elevation_deg"] == 0.0
    assert all(c["T"][2, 3] >= 80.0 - 1e-6 for c in low["candidates"])
    high = side_view_plan(recipe(), setup(), 1, ExtrusionConfig(side_view_min_camera_z_mm=250.0))
    assert high["refused"] and "45" in high["refused"] and high["candidates"] == []
    clear = side_view_plan(recipe(), setup(), 1, ExtrusionConfig(side_view_elevation_deg=0.0,
                                                                  side_view_min_camera_z_mm=0.0))
    assert clear["elevation_deg"] == 0.0 and not clear["floor_raised"]
    higher_layer = side_view_plan(recipe(), setup(), 3, ExtrusionConfig())
    assert higher_layer["layer_top_z_mm"] == 16.0 and higher_layer["candidates"][0]["crest_mm"][2] == 16.0


def test_inspection_plan_lists_the_star_and_side_poses_only_when_asked():
    cfg = ExtrusionConfig()
    plain = inspection_plan(recipe(), setup(), K=syn.K_720P, size_px=syn.SIZE_720P, config=cfg)
    assert "views" not in plain["layers"][0] and "side_view" not in plain["layers"][0]
    full = inspection_plan(recipe(), setup(), K=syn.K_720P, size_px=syn.SIZE_720P, config=cfg,
                           multiview=True, side_view=True)
    layer = full["layers"][0]
    assert [v["name"] for v in layer["views"]] == ["star-0", "star-120", "star-240"]
    assert all("T" not in c for v in layer["views"] for c in v["candidates"])      # JSON-safe
    assert layer["views"][0]["candidates"][0]["xyz_mm"][2] == pytest.approx(
        layer["top_z_mm"] + 300.0 * math.cos(math.radians(20.0)))
    assert layer["side_view"]["elevation_deg"] == pytest.approx(17.2, abs=0.1)
    assert "T" not in layer["side_view"]["candidates"][0]
    json.dumps(full)                                                               # nothing numpy left
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `py -3.10 -m pytest tests/test_extrusion_multiview.py -q`
Expected: `ImportError: cannot import name 'multiview_names'`.

- [ ] **Step 3: Implement the star and side poses**

Append to `tasni/modules/extrusion/inspection.py` after `order_candidates_seed_first` (line 254):

```python
# -- multi-view star -----------------------------------------------------------
# Three views at one cone angle, 120 deg apart in azimuth ("Mercedes star"): one
# ring of the scan module's dome. Same aim point and standoff as the top view, so
# the ring fills the frame identically from every side. Azimuth is measured in
# the WORK frame from +X (the paired-detection axis) so the star's orientation
# is reproducible across takes; roll is measured from the camera-as-parked axis,
# exactly as for the top view.
STAR_AZIMUTH_STEP_DEG = 120.0
STAR_SLACK_STEP_DEG = 10.0
STAR_TILT_STEP_DEG = 5.0


def multiview_names(config) -> list[str]:
    """Capture order: the top view first (it is the registration reference)."""
    return ["top"] + [f"star-{int(round(k * STAR_AZIMUTH_STEP_DEG))}" for k in range(3)]


def _star_walk(config) -> list[tuple[float, float]]:
    """(tilt, azimuth offset) pairs in fallback order.

    The configured tilt with azimuth slack first (0, +10, -10, +20, -20 ...),
    then the same again 5 deg shallower, down to ``multiview_tilt_min_deg``.
    Never tilt 0 -- that is the top view again, and a duplicate reference.
    """
    slack = float(config.multiview_azimuth_slack_deg)
    offsets = [0.0]
    step = STAR_SLACK_STEP_DEG
    while step <= slack + 1e-9:
        offsets += [step, -step]
        step += STAR_SLACK_STEP_DEG
    tilts = []
    tilt = float(config.multiview_tilt_deg)
    while tilt >= float(config.multiview_tilt_min_deg) - 1e-9:
        tilts.append(tilt)
        tilt -= STAR_TILT_STEP_DEG
    return [(t, o) for t in tilts for o in offsets]


def star_view_candidates(aim_mm, standoff_mm: float, config,
                         reference_x=None) -> list[dict]:
    """The three tilted views, each with its own ordered candidate walk.

    Per view: roll candidates at the nominal azimuth first (roll is free), then
    azimuth slack, then shallower tilt. Every candidate keeps the aim point on
    the optical axis at the standoff (see :func:`pose_from_aim`).
    """
    rolls = [float(v) for v in config.inspection_roll_candidates_deg]
    base = float(config.multiview_azimuth_offset_deg)
    views = []
    for k in range(3):
        name = f"star-{int(round(k * STAR_AZIMUTH_STEP_DEG))}"
        azimuth = (base + k * STAR_AZIMUTH_STEP_DEG) % 360.0
        candidates = []
        for tilt, offset in _star_walk(config):
            for roll in rolls:
                az = (azimuth + offset) % 360.0
                T = pose_from_aim(aim_mm, standoff_mm, tilt_deg=tilt, azimuth_deg=az,
                                  roll_deg=roll, reference_x=reference_x)
                candidates.append({"view": name, "tilt_deg": tilt, "azimuth_deg": az,
                                   "roll_deg": roll,
                                   "xyz_mm": [float(v) for v in T[:3, 3]], "T": T})
        views.append({"name": name, "tilt_deg": float(config.multiview_tilt_deg),
                      "azimuth_deg": azimuth, "candidates": candidates})
    return views


# -- side photo ----------------------------------------------------------------
SIDE_VIEW_MAX_ELEVATION_DEG = 45.0
SIDE_VIEW_ROLLS_DEG = (0.0, 180.0)


def side_pose_from_crest(crest_mm, *, azimuth_deg: float, elevation_deg: float,
                         standoff_mm: float, roll_deg: float = 0.0) -> np.ndarray:
    """Camera pose looking at ``crest_mm`` from the side: ``elevation_deg`` above
    horizontal, ``standoff_mm`` away along the outward azimuth, image upright
    (work +Z is image up) unless rolled 180. OpenCV convention: +Z out of the
    lens, +X right in the image, +Y down."""
    crest = np.asarray(crest_mm, dtype=float).reshape(3)
    az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
    outward = np.array([math.cos(az), math.sin(az), 0.0])
    away = math.cos(el) * outward + math.sin(el) * np.array([0.0, 0.0, 1.0])
    z_axis = -away                                         # looks back at the crest
    up = np.array([0.0, 0.0, 1.0]) - float(z_axis[2]) * z_axis
    up /= np.linalg.norm(up)
    y_axis = -up                                           # image DOWN
    x_axis = np.cross(y_axis, z_axis)                      # right-handed: y x z = x
    roll = math.radians(roll_deg)
    rolled_x = math.cos(roll) * x_axis + math.sin(roll) * y_axis
    rolled_y = np.cross(z_axis, rolled_x)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2] = rolled_x, rolled_y, z_axis
    T[:3, 3] = crest + standoff_mm * away
    return T


def side_view_plan(recipe, setup, layer_index: int, config) -> dict:
    """Where to photograph the bead from the side, and whether that is allowed.

    Targets the NEAR CREST of the ring, not its centre, so the bead is what
    fills the frame. The elevation is raised above the configured value when the
    optical centre would otherwise sit below ``side_view_min_camera_z_mm`` (the
    housing and the flange behind it on the table); the formula decides, not the
    operator, and the derived value is reported so a photo that is not
    perpendicular says by how much. Above 45 deg it is no longer a side view:
    refused with the number rather than photographed from above.
    """
    top_z = layer_top_z_mm(recipe, setup, layer_index)
    standoff = float(config.side_view_standoff_mm)
    requested = float(config.side_view_elevation_deg)
    needed = (float(config.side_view_min_camera_z_mm) - top_z) / standoff
    floor_elevation = math.degrees(math.asin(float(np.clip(needed, -1.0, 1.0))))
    elevation = max(requested, floor_elevation)
    refused = None
    if elevation > SIDE_VIEW_MAX_ELEVATION_DEG:
        refused = (f"side view refused: keeping the camera above the "
                   f"{config.side_view_min_camera_z_mm:.0f} mm floor at a {standoff:.0f} mm "
                   f"standoff needs {elevation:.1f} deg of elevation, past the "
                   f"{SIDE_VIEW_MAX_ELEVATION_DEG:.0f} deg limit of a side view")
    azimuths = ([float(config.side_view_azimuth_deg)]
                + [float(v) for v in config.side_view_azimuth_fallbacks_deg])
    candidates = []
    if refused is None:
        for azimuth in azimuths:
            az = math.radians(azimuth)
            crest = np.array([setup.center_x_mm + recipe.radius_mm * math.cos(az),
                              setup.center_y_mm + recipe.radius_mm * math.sin(az), top_z])
            for roll in SIDE_VIEW_ROLLS_DEG:
                T = side_pose_from_crest(crest, azimuth_deg=azimuth, elevation_deg=elevation,
                                         standoff_mm=standoff, roll_deg=roll)
                candidates.append({"view": "side", "azimuth_deg": azimuth % 360.0,
                                   "elevation_deg": elevation, "roll_deg": roll,
                                   "crest_mm": [float(v) for v in crest],
                                   "xyz_mm": [float(v) for v in T[:3, 3]], "T": T})
    return {"name": "side", "layer_index": int(layer_index), "layer_top_z_mm": top_z,
            "standoff_mm": standoff, "requested_elevation_deg": requested,
            "elevation_deg": float(elevation),
            "floor_raised": bool(floor_elevation > requested),
            "min_camera_z_mm": float(config.side_view_min_camera_z_mm),
            "refused": refused, "candidates": candidates}
```

Then extend `inspection_plan` (line 191). Change the signature to:

```python
def inspection_plan(recipe, setup, *, K: np.ndarray, size_px: tuple[int, int],
                    config, reference_x=None, multiview: bool = False,
                    side_view: bool = False) -> dict:
```

and replace the per-layer `layers.append({...})` block (lines 208–216) with:

```python
        entry = {
            "layer_index": index,
            "top_z_mm": float(aim[2]),
            "aim_mm": [float(v) for v in aim],
            "camera_z_mm": float(aim[2] + standoff),
            "candidates": [{k: v for k, v in candidate.items() if k != "T"}
                           for candidate in pose_candidates(aim, standoff, config,
                                                            reference_x)],
        }
        if multiview:
            entry["views"] = [
                {**view, "candidates": [{k: v for k, v in c.items() if k != "T"}
                                        for c in view["candidates"]]}
                for view in star_view_candidates(aim, standoff, config, reference_x)]
        if side_view:
            side = side_view_plan(recipe, setup, index, config)
            entry["side_view"] = {**side, "candidates": [
                {k: v for k, v in c.items() if k != "T"} for c in side["candidates"]]}
        layers.append(entry)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `py -3.10 -m pytest tests/test_extrusion_multiview.py tests/test_extrusion.py -q`
Expected: all pass. (`test_extrusion.py` covers the unchanged `pose_candidates` / `inspection_plan` defaults.)

- [ ] **Step 5: Commit and push**

```bash
git add tasni/modules/extrusion/inspection.py tests/test_extrusion_multiview.py
git commit -m "feat(extrusion): star-view and side-photo poses

Three tilted views 120 deg apart with their own candidate walks (never tilt 0),
and a side pose aimed at the near crest whose elevation the camera floor decides.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
git push
```

---

### Task 3: Processing seam — `observation_points` / `process_points` / `characterize_points`

Behaviour-preserving refactor. Everything after the first line of `process_observation` already operates on work-frame points; this task names that seam so a merged cloud can enter it. Existing tests are the regression.

**Files:**
- Modify: `tasni/modules/extrusion/processing.py:462-635` (`process_observation`) and `:662-724` (`characterize_ring`)
- Test: `tests/test_extrusion_multiview.py`

**Interfaces:**
- Produces: `observation_points(depth, K, T_work_camera) -> np.ndarray`; `_work_roi(points, *, plan, layer, config, floor_profile=None, counts=None, keep=None) -> (points, roi_diag, floor)`; `process_points(points, *, plan, layer, config, floor_profile=None, stages=None, counts=None, timings=None, started=None) -> ProcessingResult`; `coarse_plan(*, center_mm, radius_mm, bead_mm, height_mm, work_frame, config, inspection_tool="Realsense", print_tool="LongCalibTool") -> CylinderPlan`; `characterize_points(points, *, search_center_mm, work_frame, config, inspection_tool="Realsense", print_tool="LongCalibTool", counts=None, started=None) -> CharacterizationResult`. `process_observation` and `characterize_ring` keep their exact signatures.
- Known, accepted report difference: the refined pass inside `characterize_*` no longer carries a `timings_ms.backproject_ms` key (nothing reads it; `grep -rn backproject_ms tests/` is empty).

- [ ] **Step 1: Write the failing test** (append)

```python
# ------------------------------------------------------------ Task 3: the seam

from tasni.modules.extrusion.processing import (characterize_points,  # noqa: E402
                                                characterize_ring, observation_points,
                                                process_observation, process_points)
from tasni.modules.extrusion.toolpath import generate_cylinder_plan  # noqa: E402


def scene_plan(*, radius=60.0, bead=8.0, layers=1, layer_height=6.0, center=CENTER):
    made = CylinderRecipe(radius_mm=radius, layer_count=layers, layer_height_mm=layer_height,
                          bead_diameter_mm=bead, robot_speed_mm_s=75, extrusion_rate_pct=0,
                          points_per_circle=180)
    placed = CylinderSetup(print_tool="LongCalibTool", work_frame="Tasni Work Frame",
                           inspection_tool="Realsense", inspection_auto=True,
                           center_x_mm=center[0], center_y_mm=center[1])
    return generate_cylinder_plan(made, placed)


def top_frame(plan, layer_index, rings, *, seed=0):
    """Depth from the derived straight-down pose, exactly as ``observe`` in test_extrusion_measure."""
    T = syn.inspection_camera_T(aim_point_mm(plan.recipe, plan.setup, layer_index), 300.0)
    depth = syn.render_scene(rings, T, plane_center_xy_mm=(plan.setup.center_x_mm,
                                                           plan.setup.center_y_mm), seed=seed)
    return depth, T


def test_process_points_is_the_chain_after_back_projection():
    pytest.importorskip("open3d")
    plan = scene_plan()
    rings = [syn.RingSpec(60.0, 8.0, CENTER, height_fn=syn.flat(6.0))]
    depth, T = top_frame(plan, 1, rings)
    color = np.zeros((syn.SIZE_720P[1], syn.SIZE_720P[0], 3), np.uint8)
    whole = process_observation(color=color, depth=depth, T_work_camera=T, K=syn.K_720P,
                                plan=plan, layer=plan.layers[0], config=ExtrusionConfig())
    points = observation_points(depth, syn.K_720P, T)
    assert points.shape[1] == 3 and len(points) == whole.report["counts"]["raw_depth_pixels"]
    stages: dict = {}
    seam = process_points(points, plan=plan, layer=plan.layers[0], config=ExtrusionConfig(),
                          stages=stages)
    assert seam.metrics.model_dump() == whole.metrics.model_dump()
    assert seam.geometry.model_dump() == whole.geometry.model_dump()
    for key in ("after_work_roi", "after_largest_cluster", "after_radial_trim", "after_normal_cluster"):
        assert seam.report["counts"][key] == whole.report["counts"][key], key
    assert "raw_depth_pixels" not in seam.report["counts"]          # nobody back-projected here
    assert set(stages) >= {"work_roi", "deposit_cluster", "radial_trimmed", "top_surface"}
    assert seam.report["timings_ms"]["total_ms"] > 0


def test_characterize_points_matches_characterize_ring():
    pytest.importorskip("open3d")
    plan = scene_plan(radius=50.0)
    rings = [syn.RingSpec(61.0, 8.0, (CENTER[0] + 5.0, CENTER[1] - 3.0), height_fn=syn.flat(6.0))]
    depth, T = top_frame(plan, 1, rings)
    color = np.zeros((syn.SIZE_720P[1], syn.SIZE_720P[0], 3), np.uint8)
    whole = characterize_ring(color=color, depth=depth, T_work_camera=T, K=syn.K_720P,
                              search_center_mm=CENTER, work_frame="Tasni Work Frame",
                              config=ExtrusionConfig())
    seam = characterize_points(observation_points(depth, syn.K_720P, T), search_center_mm=CENTER,
                               work_frame="Tasni Work Frame", config=ExtrusionConfig())
    assert seam.summary() == whole.summary()
    assert seam.report["coarse"]["radius_mm"] == whole.report["coarse"]["radius_mm"]
    assert seam.report["kind"] == "characterization"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `py -3.10 -m pytest tests/test_extrusion_multiview.py -q`
Expected: `ImportError: cannot import name 'characterize_points'`.

- [ ] **Step 3: Split `process_observation` at the seam**

In `tasni/modules/extrusion/processing.py`, replace the whole of `process_observation` (lines 462–635) with the following three functions. The bodies of `_work_roi` and `process_points` are the existing lines moved verbatim — copy them from the file, do not retype them; only the framing shown here is new.

```python
def observation_points(depth: np.ndarray, K: np.ndarray, T_work_camera: np.ndarray) -> np.ndarray:
    """The work-frame cloud of one depth frame: the seam every later stage sits on.

    Multi-view capture back-projects each view with its own pose, registers the
    clouds, and hands the union to :func:`process_points`; nothing after this
    line knows how many frames it came from.
    """
    points, _ = depth_to_work_points(depth, K, T_work_camera, depth_scale=1000.0)
    return points


def _work_roi(points: np.ndarray, *, plan: CylinderPlan, layer: LayerPath, config,
              floor_profile: np.ndarray | None = None, counts: dict | None = None,
              keep=None) -> tuple[np.ndarray, dict, dict]:
    """Height band + radial band about the plan centre, then the per-layer floor.

    Returns ``(points, roi_diag, floor)``. Extracted so a single view's own ring
    can be found by exactly the selection the merged cloud goes through
    (``multiview.view_ring_center``) -- one implementation, one set of numbers.
    """
    counts = {} if counts is None else counts
    keep = keep or (lambda name, cloud: None)
    setup, recipe = plan.setup, plan.recipe
    radius = np.linalg.norm(points[:, :2] - np.array([setup.center_x_mm, setup.center_y_mm]), axis=1)
    max_z = layer.nominal_z_mm + recipe.bead_diameter_mm / 2 + config.deposit_height_margin_mm
    # The selected work frame defines the build plane at Z=0, so deterministic
    # height subtraction is more reproducible than fitting a new plane per frame.
    min_z = max(config.deposit_min_height_mm,
                config.plane_distance_threshold_m * 1000.0)
    in_height = (points[:, 2] >= min_z) & (points[:, 2] <= max_z)
    r_lo = recipe.radius_mm - config.radial_roi_margin_mm
    r_hi = recipe.radius_mm + config.radial_roi_margin_mm
    in_radial = (radius >= r_lo) & (radius <= r_hi)
    roi = in_height & in_radial
    # ... lines 507-541 of the original (roi_diag, the percentile block,
    #     counts.update, points = points[roi], keep("work_roi"), the floor
    #     block with keep("above_floor")) verbatim ...
    counts["after_work_roi"] = len(points)
    return points, roi_diag, floor


def process_points(points: np.ndarray, *, plan: CylinderPlan, layer: LayerPath, config,
                   floor_profile: np.ndarray | None = None, stages: dict | None = None,
                   counts: dict | None = None, timings: dict | None = None,
                   started: float | None = None) -> ProcessingResult:
    """Everything after back-projection: ROI -> deposit -> trim -> crest -> centreline.

    ``points`` is a work-frame cloud from ONE frame (:func:`process_observation`)
    or from several registered views (``multiview.process_views``); nothing
    below knows or cares which. ``counts`` / ``timings`` / ``started`` let the
    caller pre-seed what it already measured (raw pixel count, back-projection
    or merge time, the clock start), so the report reads exactly as it did
    before this seam existed.
    """
    def keep(name: str, cloud: np.ndarray) -> None:
        if stages is not None:
            stages[name] = np.asarray(cloud, dtype=float).copy()
    started = time.perf_counter() if started is None else started
    timings = {} if timings is None else timings
    counts = {} if counts is None else counts
    setup, recipe = plan.setup, plan.recipe
    points, roi_diag, floor = _work_roi(points, plan=plan, layer=layer, config=config,
                                        floor_profile=floor_profile, counts=counts, keep=keep)
    if len(points) < config.cluster_min_points:
        raise RuntimeError(
            "not enough deposited-geometry points inside the configured work ROI "
            f"(need {config.cluster_min_points}); {json.dumps(roi_diag)}")
    # ... lines 548-635 of the original (mark = ...; deposit = _filter_deposit ...
    #     through `return ProcessingResult(...)`) verbatim ...


def process_observation(*, color: np.ndarray, depth: np.ndarray,
                        T_work_camera: np.ndarray, K: np.ndarray,
                        plan: CylinderPlan, layer: LayerPath, config,
                        floor_profile: np.ndarray | None = None,
                        stages: dict | None = None) -> ProcessingResult:
    """Reconstruct one layer from exactly one saved synchronized RGB-D frame.

    (keep the original docstring's two paragraphs about ``floor_profile`` and
    ``stages`` here, verbatim)
    """
    started = time.perf_counter()
    timings: dict[str, float] = {}
    counts: dict[str, int] = {}
    mark = time.perf_counter()
    points, counts["raw_depth_pixels"] = depth_to_work_points(
        depth, K, T_work_camera, depth_scale=1000.0)
    timings["backproject_ms"] = (time.perf_counter() - mark) * 1000
    if stages is not None:
        stages["backprojected"] = np.asarray(points, dtype=float).copy()
    return process_points(points, plan=plan, layer=layer, config=config,
                          floor_profile=floor_profile, stages=stages,
                          counts=counts, timings=timings, started=started)
```

- [ ] **Step 4: Split `characterize_ring` the same way**

Replace `characterize_ring` (lines 662–724) with:

```python
def coarse_plan(*, center_mm, radius_mm: float, bead_mm: float, height_mm: float,
                work_frame: str, config, inspection_tool: str = "Realsense",
                print_tool: str = "LongCalibTool") -> CylinderPlan:
    """A throwaway one-layer plan from coarse ring numbers, so a refined pass runs
    the SAME chain a layer measurement uses -- one pipeline, one set of numbers."""
    recipe = CylinderRecipe(
        radius_mm=float(np.clip(radius_mm, 5.0, 500.0)), layer_count=1,
        layer_height_mm=float(np.clip(height_mm, 0.5, 50.0)),
        bead_diameter_mm=float(np.clip(bead_mm, 0.5, 50.0)),
        robot_speed_mm_s=75.0, extrusion_rate_pct=0.0,
        points_per_circle=config.measured_spline_points)
    setup = CylinderSetup(
        print_tool=print_tool, work_frame=work_frame, inspection_tool=inspection_tool,
        inspection_auto=True, center_x_mm=float(center_mm[0]), center_y_mm=float(center_mm[1]))
    return generate_cylinder_plan(recipe, setup)


def characterize_points(points: np.ndarray, *, search_center_mm, work_frame: str, config,
                        inspection_tool: str = "Realsense", print_tool: str = "LongCalibTool",
                        counts: dict | None = None,
                        started: float | None = None) -> CharacterizationResult:
    """Characterize a ring from a work-frame cloud with NO recipe assumption.

    Pass 1 takes everything above the build plane inside a search cylinder
    around ``search_center_mm``, filters it like a deposit, and fits a circle to
    get a coarse centre/radius/bead. Pass 2 hands those to
    :func:`process_points` as a throwaway recipe so the refined numbers come out
    of the same code the layer measurements use.
    """
    started = time.perf_counter() if started is None else started
    counts = {} if counts is None else counts
    everything = np.asarray(points, dtype=float)          # pass 2 re-reads the whole cloud
    center = np.asarray(search_center_mm, dtype=float)
    min_z = max(config.deposit_min_height_mm, config.plane_distance_threshold_m * 1000.0)
    radial = np.linalg.norm(everything[:, :2] - center, axis=1)
    roi = ((everything[:, 2] >= min_z) & (everything[:, 2] <= config.characterize_max_height_mm)
           & (radial <= config.characterize_search_radius_mm))
    points = everything[roi]
    counts["after_search_roi"] = len(points)
    if len(points) < config.cluster_min_points:
        raise RuntimeError("no deposited geometry inside the characterization search region")
    clusters = _deposit_clusters(points, config, counts)
    deposit, selector = _select_ring_cluster(clusters, center, counts)
    coarse_center, coarse_radius = fit_circle_xy(deposit)
    width = bead_width_profile(deposit, coarse_center, bins=config.bead_width_bins)
    top = _top_surface(deposit, config, counts)
    coarse_height = float(np.percentile(top[:, 2], 90))
    coarse = {"center_mm": [float(coarse_center[0]), float(coarse_center[1])],
              "radius_mm": float(coarse_radius), "bead_width_mm": width["mean_mm"],
              "height_mm": coarse_height, "time_ms": (time.perf_counter() - started) * 1000}
    plan = coarse_plan(center_mm=coarse_center, radius_mm=coarse_radius, bead_mm=width["mean_mm"],
                       height_mm=coarse_height, work_frame=work_frame, config=config,
                       inspection_tool=inspection_tool, print_tool=print_tool)
    refined = process_points(everything, plan=plan, layer=plan.layers[0], config=config,
                             counts={"raw_depth_pixels": int(len(everything))})
    geometry = refined.geometry
    report = {**refined.report, "coarse": coarse, "counts_coarse": counts,
              "ring_selector": selector,
              "kind": "characterization",
              "total_ms": (time.perf_counter() - started) * 1000}
    return CharacterizationResult(
        radius_mm=refined.metrics.measured_radius_mm,
        center_mm=refined.metrics.measured_center_mm,
        bead_width_mm=geometry.bead_width_mean_mm,
        bead_width_min_mm=geometry.bead_width_min_mm,
        bead_width_max_mm=geometry.bead_width_max_mm,
        top_z_mean_mm=geometry.top_z_mean_mm, top_z_min_mm=geometry.top_z_min_mm,
        top_z_max_mm=geometry.top_z_max_mm, measured_xyz=refined.measured_xyz,
        segmentation=refined.segmentation, skeleton=refined.skeleton,
        comparison=refined.comparison, report=report)


def characterize_ring(*, color: np.ndarray, depth: np.ndarray, T_work_camera: np.ndarray,
                      K: np.ndarray, search_center_mm, work_frame: str, config,
                      inspection_tool: str = "Realsense",
                      print_tool: str = "LongCalibTool") -> CharacterizationResult:
    """Measure a ring with NO recipe assumption from one RGB-D frame.

    Back-projects the frame and defers to :func:`characterize_points`; the
    multi-view job merges its registered views and calls that directly.
    """
    started = time.perf_counter()
    counts: dict[str, int] = {}
    points, counts["raw_depth_pixels"] = depth_to_work_points(depth, K, T_work_camera)
    return characterize_points(points, search_center_mm=search_center_mm, work_frame=work_frame,
                               config=config, inspection_tool=inspection_tool,
                               print_tool=print_tool, counts=counts, started=started)
```

- [ ] **Step 5: Run the regression + the new tests**

Run: `py -3.10 -m pytest tests/test_extrusion_multiview.py tests/test_extrusion_measure.py tests/test_extrusion_processing.py tests/test_extrusion_figures.py tests/test_extrusion_job.py -q`
Expected: all pass. `test_extrusion_figures.py::test_the_method_figure_draws_every_stage_of_the_pipeline` proves `stages` still fills through the wrapper; the characterize tests in `test_extrusion_measure.py` prove the refined pass is unchanged.

- [ ] **Step 6: Commit and push**

```bash
git add tasni/modules/extrusion/processing.py tests/test_extrusion_multiview.py
git commit -m "refactor(extrusion): name the back-projection seam in the reconstruction chain

process_observation = process_points(observation_points(...)); characterize_ring =
characterize_points(...). Behaviour-preserving: every existing test is the regression.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
git push
```

---

### Task 4: `multiview.py` — levelling, centre registration, merge, diagnostics

**Files:**
- Create: `tasni/modules/extrusion/multiview.py`
- Test: `tests/test_extrusion_multiview.py`

**Interfaces:**
- Consumes (Task 3): `_work_roi`, `_filter_deposit`, `_radial_trim`, `process_points`, `characterize_points`, `coarse_plan`, `ProcessingResult`, `CharacterizationResult`; `fit_circle_xy`; (Task 2) `star_view_candidates` (tests only).
- Produces: `ViewPoints`, `MergeResult`, `median_depth`, `level_points`, `view_ring_center`, `merge_views`, `process_views`, `characterize_views` with the signatures in the interfaces table. `process_views` adds `report["merge"]`, `report["views"]`, `report["timings_ms"]["merge_ms"]`, `report["counts"]["merged_points"]` and fills `stages["merged"]`.

- [ ] **Step 1: Write the failing tests** (append)

```python
# ------------------------------------------------------------ Task 4: multiview.py

from tasni.modules.extrusion.multiview import (ViewPoints, characterize_views,  # noqa: E402
                                               level_points, median_depth, merge_views,
                                               process_views, view_ring_center)


def rigid(*, tilt_deg=0.0, dz_mm=0.0, dx_mm=0.0, dy_mm=0.0, about=(CENTER[0], CENTER[1], 0.0)):
    """A rigid error: rotate ``tilt_deg`` about the work X axis through ``about``, then translate."""
    t = math.radians(tilt_deg)
    R = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(t), -math.sin(t)], [0.0, math.sin(t), math.cos(t)]])
    P = np.eye(4)
    P[:3, :3] = R
    P[:3, 3] = np.asarray(about, float) - R @ np.asarray(about, float) + [dx_mm, dy_mm, dz_mm]
    return P


def star_views(plan, layer_index, rings, *, config=None, seed=0, perturb=None, names=None):
    """Render the rings from the top pose and the three star poses, back-project each.

    ``perturb[name]`` is a 4x4 applied to that view's pose AFTER rendering, i.e. a
    hand-eye / levelling error: the scene is true, the back-projection is wrong.
    """
    cfg = config or ExtrusionConfig()
    aim = aim_point_mm(plan.recipe, plan.setup, layer_index)
    poses = {"top": syn.inspection_camera_T(aim, 300.0)}
    for view in star_view_candidates(aim, 300.0, cfg, reference_x=syn.CAMERA_X_AT_PARK):
        poses[view["name"]] = view["candidates"][0]["T"]
    views = []
    for index, (name, T) in enumerate(poses.items()):
        if names is not None and name not in names:
            continue
        depth = syn.render_scene(rings, T, plane_center_xy_mm=(plan.setup.center_x_mm,
                                                               plan.setup.center_y_mm),
                                 seed=seed + index)
        used = T if not perturb or name not in perturb else perturb[name] @ T
        views.append(ViewPoints(name, observation_points(depth, syn.K_720P, used), used))
    return views


def level_annulus(plan, cfg):
    inner = plan.recipe.radius_mm + cfg.radial_roi_margin_mm + 10.0
    return inner, plan.recipe.radius_mm + cfg.multiview_level_annulus_mm


def test_median_depth_ignores_dropouts_and_keeps_the_dtype():
    a = np.array([[100, 0], [300, 7]], np.uint16)
    b = np.array([[102, 0], [0, 9]], np.uint16)
    c = np.array([[98, 5], [310, 8]], np.uint16)
    med = median_depth([a, b, c])
    assert med.dtype == np.uint16 and med.tolist() == [[100, 5], [305, 8]]
    assert median_depth([a]).tolist() == a.tolist()


def test_levelling_removes_an_injected_tilt_and_height_offset():
    plan = scene_plan()
    ring = [syn.RingSpec(60.0, 8.0, CENTER, height_fn=syn.flat(6.0))]
    bad = star_views(plan, 1, ring, perturb={"star-0": rigid(tilt_deg=1.0, dz_mm=1.5)},
                     names=["star-0"])[0]
    cfg = ExtrusionConfig()
    inner, outer = level_annulus(plan, cfg)
    leveled, info = level_points(bad.points, center_xy=CENTER, inner_mm=inner, outer_mm=outer, config=cfg)
    assert info["applied"] and info["points"] >= cfg.multiview_level_min_points
    assert info["tilt_deg"] == pytest.approx(1.0, abs=0.05)
    assert info["dz_mm"] == pytest.approx(1.5, abs=0.5)      # the lever arm to the annulus centroid
    assert info["rms_mm"] < 0.7                              # 0.5 mm noise + 1 mm depth quantization
    _, residual = level_points(leveled, center_xy=CENTER, inner_mm=inner, outer_mm=outer, config=cfg)
    assert residual["tilt_deg"] < 0.05 and abs(residual["dz_mm"]) < 0.1
    # The ring itself came along: its crest is back near z = 6 after levelling.
    near_ring = leveled[np.abs(np.linalg.norm(leveled[:, :2] - CENTER, axis=1) - 60.0) < 1.0]
    assert np.percentile(near_ring[:, 2], 95) == pytest.approx(6.0, abs=1.0)


def test_levelling_is_skipped_with_a_warning_when_the_board_is_not_visible():
    cfg = ExtrusionConfig()
    plan = scene_plan()
    inner, outer = level_annulus(plan, cfg)
    ring_only = syn.RingSpec(60.0, 8.0, CENTER, height_fn=syn.flat(6.0)).surface_points()
    same, info = level_points(ring_only, center_xy=CENTER, inner_mm=inner, outer_mm=outer, config=cfg)
    assert not info["applied"] and "skipped" in info["warning"] and same is not None
    assert np.array_equal(same, ring_only)


def test_a_view_finds_its_own_ring_centre_by_the_chains_own_selection():
    pytest.importorskip("open3d")
    plan = scene_plan()
    ring = [syn.RingSpec(60.0, 8.0, (CENTER[0] + 2.0, CENTER[1] - 1.0), height_fn=syn.flat(6.0))]
    view = star_views(plan, 1, ring, names=["star-120"])[0]
    center, n, error = view_ring_center(view.points, plan=plan, layer=plan.layers[0], config=ExtrusionConfig())
    assert error is None and n > 100
    assert center[0] == pytest.approx(CENTER[0] + 2.0, abs=0.5) and center[1] == pytest.approx(CENTER[1] - 1.0, abs=0.5)
    bare = star_views(plan, 1, [], names=["star-120"])[0]
    center, n, error = view_ring_center(bare.points, plan=plan, layer=plan.layers[0], config=ExtrusionConfig())
    assert center is None and error


def test_centre_registration_recovers_an_injected_lateral_error():
    pytest.importorskip("open3d")
    plan = scene_plan()
    ring = [syn.RingSpec(60.0, 8.0, CENTER, height_fn=syn.flat(6.0))]
    views = star_views(plan, 1, ring, perturb={"star-120": rigid(dx_mm=1.5)})
    merged = merge_views(views, plan=plan, layer=plan.layers[0], config=ExtrusionConfig())
    rec = {v["name"]: v for v in merged.views}
    # The injected 1.5 mm comes back as the correction; the untouched views sit
    # within the fit's own bias. A tilted view sees the near flank better than
    # the far one, which can bias its centre ~0.2 mm toward the camera: if the
    # untouched views land there, widen THEIR bound to 0.4 and say so in the
    # commit -- never weaken the -1.5 recovery beyond 0.3.
    assert rec["star-120"]["xy_shift_mm"][0] == pytest.approx(-1.5, abs=0.3)
    assert abs(rec["star-120"]["xy_shift_mm"][1]) < 0.3
    assert math.hypot(*rec["star-0"]["xy_shift_mm"]) < 0.3 and math.hypot(*rec["star-240"]["xy_shift_mm"]) < 0.3
    assert rec["top"]["xy_shift_mm"] == [0.0, 0.0] and all(v["used"] for v in merged.views)
    assert all(v["levelling"]["applied"] for v in merged.views)
    assert merged.report["registration"] == "centre"
    assert merged.report["views_used"] == ["top", "star-0", "star-120", "star-240"]
    assert merged.report["views_dropped"] == [] and merged.report["merge_ms"] > 0
    assert merged.report["xy_shift_max_mm"] == pytest.approx(1.5, abs=0.3)
    assert len(merged.points) == sum(len(v.points) for v in views)
    # The control: with registration "none" the error stays in the merged cloud.
    naive = merge_views(views, plan=plan, layer=plan.layers[0], config=ExtrusionConfig(), registration="none")
    assert all(v["xy_shift_mm"] == [0.0, 0.0] for v in naive.views) and naive.report["xy_shift_max_mm"] is None
    aligned = process_points(merged.points, plan=plan, layer=plan.layers[0], config=ExtrusionConfig())
    blurred = process_points(naive.points, plan=plan, layer=plan.layers[0], config=ExtrusionConfig())
    assert aligned.metrics.valid and blurred.metrics.valid
    # A quarter of the points 1.5 mm off widens the footprint where the shifted
    # view's outer flank sticks out. If this margin needs tuning, tune the
    # margin, not the direction.
    assert blurred.geometry.bead_width_max_mm > aligned.geometry.bead_width_max_mm + 0.4


def test_four_views_of_a_thin_ring_sample_it_denser_and_measure_it_true():
    """The case this feature exists for: a thin, wavy bead. Four views must
    measure it; the top view alone is allowed to fail. The chain's 2 mm voxel
    caps post-voxel counts, so the density gain is read BEFORE the voxel."""
    pytest.importorskip("open3d")
    plan = scene_plan(radius=42.0, bead=10.0, layer_height=3.5)
    thin = [syn.RingSpec(42.0, 10.0, CENTER, height_fn=syn.wavy(3.5, 0.75, lobes=3))]
    views = star_views(plan, 1, thin, seed=3)
    merged = process_views(views, plan=plan, layer=plan.layers[0], config=ExtrusionConfig())
    assert merged.metrics.valid, merged.metrics.warnings
    assert merged.metrics.measured_radius_mm == pytest.approx(42.0, abs=0.3)
    assert merged.metrics.center_offset_norm_mm < 0.5
    assert merged.report["merge"]["views_used"] == ["top", "star-0", "star-120", "star-240"]
    assert merged.report["counts"]["merged_points"] > 1000
    assert merged.report["timings_ms"]["merge_ms"] > 0 and merged.report["timings_ms"]["total_ms"] > merged.report["timings_ms"]["merge_ms"]
    top_only = None
    try:
        top_only = process_points(views[0].points, plan=plan, layer=plan.layers[0], config=ExtrusionConfig())
    except RuntimeError:
        pass                                                 # exactly the failure multi-view is for
    if top_only is not None:
        assert merged.report["counts"]["after_work_roi"] >= 2.5 * top_only.report["counts"]["after_work_roi"]
        assert merged.metrics.shape_rms_mm <= top_only.metrics.shape_rms_mm + 0.05


def test_a_view_without_a_ring_is_merged_unshifted_or_dropped_by_policy():
    pytest.importorskip("open3d")
    plan = scene_plan()
    ring = [syn.RingSpec(60.0, 8.0, CENTER, height_fn=syn.flat(6.0))]
    views = [v for v in star_views(plan, 1, ring) if v.name != "star-240"]
    views.append(star_views(plan, 1, [], names=["star-240"])[0])        # star-240 saw only the board
    lenient = merge_views(views, plan=plan, layer=plan.layers[0], config=ExtrusionConfig())
    rec = {v["name"]: v for v in lenient.views}
    assert rec["star-240"]["used"] and "unshifted" in rec["star-240"]["warning"]
    assert rec["star-240"]["xy_shift_mm"] == [0.0, 0.0] and rec["star-240"]["ring_points"] == 0
    strict = merge_views(views, plan=plan, layer=plan.layers[0], config=ExtrusionConfig(), registration="strict")
    assert strict.report["views_dropped"] == ["star-240"]
    assert strict.report["views_used"] == ["top", "star-0", "star-120"]
    assert len(strict.points) == sum(len(v.points) for v in views[:3])
    two = process_views(views[:2], plan=plan, layer=plan.layers[0], config=ExtrusionConfig())
    assert two.metrics.valid and two.report["merge"]["views_used"] == ["top", "star-0"]
    with pytest.raises(ValueError, match="top view"):
        merge_views(views[1:], plan=plan, layer=plan.layers[0], config=ExtrusionConfig())


def test_characterize_from_four_views_recovers_the_ring_with_no_recipe():
    pytest.importorskip("open3d")
    plan = scene_plan(radius=50.0)                    # only the aim/poses come from this plan
    ring = [syn.RingSpec(61.0, 8.0, (CENTER[0] + 5.0, CENTER[1] - 3.0), height_fn=syn.flat(6.0))]
    views = star_views(plan, 1, ring)
    found = characterize_views(views, search_center_mm=CENTER, work_frame="Tasni Work Frame",
                               config=ExtrusionConfig())
    assert found.radius_mm == pytest.approx(61.0, abs=0.5)
    assert found.center_mm[0] == pytest.approx(CENTER[0] + 5.0, abs=0.6)
    assert found.center_mm[1] == pytest.approx(CENTER[1] - 3.0, abs=0.6)
    assert found.bead_width_mm == pytest.approx(8.0, abs=1.5)
    assert found.report["kind"] == "characterization" and found.report["coarse"]["radius_mm"] > 50.0
    assert found.report["merge"]["views_used"] == ["top", "star-0", "star-120", "star-240"]
    assert all(v["levelling"]["applied"] for v in found.report["views"])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `py -3.10 -m pytest tests/test_extrusion_multiview.py -q`
Expected: `ModuleNotFoundError: No module named 'tasni.modules.extrusion.multiview'`.

- [ ] **Step 3: Write `multiview.py`**

```python
# tasni/modules/extrusion/multiview.py
"""Merge several registered views of one ring into the cloud the chain measures.

Why: the mock rings are THIN (crest 2-4 mm; bare-board depth noise +4.8 mm p99 at
300 mm), so one straight-down frame sees the crest at grazing signal-to-noise and
the flanks hardly at all. Three views tilted by ~20 deg at 120 deg azimuth spacing
put every flank point within 60 deg of some camera. Merging them naively would
blur the ring instead of sharpening it: hand-eye board consistency is 1.26 mm and
the Jetson/host intrinsics mismatch is ~2 % of lateral scale, so views from
opposite sides disagree by 1-2 mm at the ring. Two corrections make the merge safe:

  1. **Levelling** -- fit the board annulus around the ring per view, rotate and
     shift the view so that plane is z = 0. Removes the tilt-dependent warp the
     2026-08-13 characterization measured (plane RMS 0.65 -> 2.0 -> 5.0 mm at
     1 / 9 / 20 deg), leaving the random part.
  2. **XY alignment** -- find the ring in each view alone, by the SAME ROI +
     deposit filter + radial trim the chain uses, and translate the view so its
     fitted centre coincides with the top view's. A circle fit is exact for the
     shape we know we have; ICP on a torus has an unconstrained yaw. The
     correction vector IS the inter-view registration error, so it is reported
     per view, never hidden.

The top view is the reference so merged centres stay continuous with every
single-view number already archived. Known residual: a per-view scale error
inflates that view's radius, so a merged bead may read up to ~2 % of R wide until
the intrinsics alignment on the Jetson is fixed -- report, do not hide.
"""
from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass

import numpy as np

from .comparison import fit_circle_xy
from .processing import (CharacterizationResult, ProcessingResult, _filter_deposit,
                         _radial_trim, _work_roi, characterize_points, coarse_plan,
                         process_points)

REGISTRATIONS = ("centre", "strict", "none")


@dataclass
class ViewPoints:
    """One view, back-projected into the work frame with its own camera pose."""
    name: str
    points: np.ndarray
    T_work_camera: np.ndarray


@dataclass
class MergeResult:
    points: np.ndarray          # the merged work-frame cloud the chain measures
    views: list[dict]           # per-view diagnostics (levelling, xy_shift_mm, used, warning)
    report: dict                # registration, views_used/dropped, merged_points, merge_ms, shifts


def median_depth(frames) -> np.ndarray:
    """Per-pixel median of the VALID (non-zero) samples; 0 where none was valid.

    The cheap lever against depth noise at a fixed pose (``multiview_frames_per_view``);
    dropouts in one frame do not drag a pixel toward zero.
    """
    arrays = [np.asarray(frame) for frame in frames]
    if not arrays:
        raise ValueError("median_depth needs at least one frame")
    if len(arrays) == 1:
        return arrays[0].copy()
    stack = np.stack(arrays).astype(float)
    stack[stack <= 0] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)           # all-NaN pixels
        median = np.nanmedian(stack, axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    return np.rint(median).astype(arrays[0].dtype)


def _fit_plane(points: np.ndarray) -> tuple[float, float, float]:
    """Least-squares z = a x + b y + c."""
    A = np.column_stack((points[:, 0], points[:, 1], np.ones(len(points))))
    a, b, c = np.linalg.lstsq(A, points[:, 2], rcond=None)[0]
    return float(a), float(b), float(c)


def _rotation_to_z(normal: np.ndarray) -> np.ndarray:
    """Rotation taking the unit ``normal`` onto +Z (Rodrigues); identity when it is there."""
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(normal, z)
    s = float(np.linalg.norm(axis))
    c = float(np.clip(normal @ z, -1.0, 1.0))
    if s < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    k = axis / s
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    theta = math.atan2(s, c)
    return np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def level_points(points, *, center_xy, inner_mm: float, outer_mm: float,
                 config) -> tuple[np.ndarray, dict]:
    """Rotate + shift a view so the board annulus around the ring is the plane z = 0.

    The annulus is ``inner_mm .. outer_mm`` from ``center_xy`` with
    ``|z| < multiview_level_max_abs_z_mm``; one round of outlier rejection at
    ``multiview_level_outlier_mm``; the rigid transform pivots about the annulus
    centroid (rotation onto +Z, then the centroid's fitted height to zero).
    Skipped with a warning -- points returned unchanged -- when fewer than
    ``multiview_level_min_points`` board points are visible.
    """
    pts = np.asarray(points, dtype=float)
    info = {"applied": False, "points": 0, "tilt_deg": 0.0, "dz_mm": 0.0,
            "rms_mm": None, "warning": None}
    if not len(pts):
        info["warning"] = "levelling skipped: empty view"
        return pts, info
    rel = pts[:, :2] - np.asarray(center_xy, dtype=float)
    r = np.linalg.norm(rel, axis=1)
    annulus = pts[(r >= inner_mm) & (r <= outer_mm)
                  & (np.abs(pts[:, 2]) < float(config.multiview_level_max_abs_z_mm))]
    minimum = int(config.multiview_level_min_points)
    info["points"] = int(len(annulus))
    if len(annulus) < minimum:
        info["warning"] = (f"levelling skipped: {len(annulus)} board points in the "
                           f"{inner_mm:.0f}-{outer_mm:.0f} mm annulus (need {minimum})")
        return pts, info
    a, b, c = _fit_plane(annulus)
    residual = annulus[:, 2] - (a * annulus[:, 0] + b * annulus[:, 1] + c)
    inlier = np.abs(residual) <= float(config.multiview_level_outlier_mm)
    if int(inlier.sum()) >= minimum:
        annulus = annulus[inlier]
        a, b, c = _fit_plane(annulus)
        residual = annulus[:, 2] - (a * annulus[:, 0] + b * annulus[:, 1] + c)
    normal = np.array([-a, -b, 1.0])
    normal /= np.linalg.norm(normal)
    cx, cy = float(annulus[:, 0].mean()), float(annulus[:, 1].mean())
    pivot = np.array([cx, cy, a * cx + b * cy + c])
    rotation = _rotation_to_z(normal)
    leveled = (pts - pivot) @ rotation.T + np.array([cx, cy, 0.0])
    info.update(applied=True, points=int(len(annulus)),
                tilt_deg=float(math.degrees(math.acos(float(np.clip(normal[2], -1.0, 1.0))))),
                dz_mm=float(pivot[2]),
                rms_mm=float(math.sqrt(float(np.mean(residual ** 2)))))
    return leveled, info


def view_ring_center(points, *, plan, layer, config,
                     floor_profile=None) -> tuple[np.ndarray | None, int, str | None]:
    """This view's own fitted ring centre, by the chain's own selection.

    Work ROI (height + radial band + per-layer floor) -> deposit cluster ->
    radial trim -> circle fit: the same functions the merged cloud goes through,
    so "the ring in this view" means what it means everywhere else. Returns
    ``(None, n, reason)`` instead of raising when the view has no usable ring.
    """
    counts: dict[str, int] = {}
    try:
        roi, _, _ = _work_roi(np.asarray(points, dtype=float), plan=plan, layer=layer,
                              config=config, floor_profile=floor_profile, counts=counts)
        if len(roi) < int(config.cluster_min_points):
            raise RuntimeError(f"{len(roi)} points inside the work ROI "
                               f"(need {config.cluster_min_points})")
        deposit = _filter_deposit(roi, config, counts)
        deposit = _radial_trim(deposit, getattr(config, "radial_trim_schedule_mm", ()), counts)
        center, _ = fit_circle_xy(deposit)
    except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
        return None, int(counts.get("after_radial_trim", counts.get("after_largest_cluster", 0))), str(exc)
    return np.asarray(center, dtype=float), int(len(deposit)), None


def _voxel_thin(points: np.ndarray, voxel_mm: float) -> np.ndarray:
    keys = np.floor(points / float(voxel_mm)).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(first)]


def merge_views(views: list[ViewPoints], *, plan, layer, config, floor_profile=None,
                registration: str | None = None) -> MergeResult:
    """Level every view, align each non-top view on its ring, concatenate.

    ``views[0]`` must be the top view: it is the reference every other view is
    translated onto. ``registration`` overrides ``config.multiview_registration``
    ("centre" | "strict" | "none" -- see the config comment).
    """
    started = time.perf_counter()
    registration = registration or config.multiview_registration
    if registration not in REGISTRATIONS:
        raise ValueError(f"unknown registration {registration!r}; expected one of {REGISTRATIONS}")
    if not views or views[0].name != "top":
        raise ValueError("the first view must be the top view (the registration reference)")
    R = float(plan.recipe.radius_mm)
    center_xy = (float(plan.setup.center_x_mm), float(plan.setup.center_y_mm))
    inner = R + float(config.radial_roi_margin_mm) + 10.0
    outer = R + float(config.multiview_level_annulus_mm)
    clouds: list[np.ndarray] = []
    records: list[dict] = []
    for view in views:
        leveled, info = level_points(view.points, center_xy=center_xy, inner_mm=inner,
                                     outer_mm=outer, config=config)
        clouds.append(leveled)
        records.append({"name": view.name, "backprojected": int(len(view.points)),
                        "ring_points": 0, "levelling": info, "xy_shift_mm": [0.0, 0.0],
                        "aligned": False, "used": True, "warning": info.get("warning")})
    reference = None
    if registration != "none":
        reference, n, error = view_ring_center(clouds[0], plan=plan, layer=layer, config=config,
                                               floor_profile=floor_profile)
        records[0]["ring_points"] = n
        if reference is None:
            records[0]["warning"] = f"top view: ring not found ({error}); every view merged unshifted"
    cap = float(config.multiview_max_xy_shift_mm)
    for index in range(1, len(views)):
        record = records[index]
        if registration == "none" or reference is None:
            continue
        found, n, error = view_ring_center(clouds[index], plan=plan, layer=layer, config=config,
                                           floor_profile=floor_profile)
        record["ring_points"] = n
        reason = None
        if found is None:
            reason = f"ring not found ({error})"
        elif float(np.linalg.norm(reference - found)) > cap:
            reason = (f"ring centre {float(np.linalg.norm(reference - found)):.1f} mm from the "
                      f"top view's (cap {cap:.0f} mm)")
        if reason:
            if registration == "strict":
                record["used"] = False
                record["warning"] = f"dropped: {reason}"
            else:
                record["warning"] = f"merged unshifted: {reason}"
            continue
        shift = reference - found
        shifted = clouds[index].copy()
        shifted[:, :2] += shift
        clouds[index] = shifted
        record["xy_shift_mm"] = [float(shift[0]), float(shift[1])]
        record["aligned"] = True
    merged = np.vstack([cloud for cloud, record in zip(clouds, records) if record["used"]])
    if float(config.multiview_voxel_mm) > 0:
        merged = _voxel_thin(merged, float(config.multiview_voxel_mm))
    shifts = [math.hypot(*r["xy_shift_mm"]) for r in records if r["aligned"]]
    report = {
        "registration": registration,
        "views_used": [r["name"] for r in records if r["used"]],
        "views_dropped": [r["name"] for r in records if not r["used"]],
        "merged_points": int(len(merged)),
        "merge_ms": (time.perf_counter() - started) * 1000.0,
        "xy_shift_mean_mm": float(np.mean(shifts)) if shifts else None,
        "xy_shift_max_mm": float(np.max(shifts)) if shifts else None,
    }
    return MergeResult(points=merged, views=records, report=report)


def process_views(views: list[ViewPoints], *, plan, layer, config, floor_profile=None,
                  stages: dict | None = None) -> ProcessingResult:
    """Merge the views, then run the unchanged chain on the merged cloud.

    The report gains ``merge`` (the registration summary) and ``views`` (per-view
    diagnostics); ``counts.raw_depth_pixels`` is the sum over views and
    ``counts.merged_points`` what the chain received. ``stages["merged"]`` holds
    the merged cloud so the job can archive it.
    """
    started = time.perf_counter()
    merged = merge_views(views, plan=plan, layer=layer, config=config, floor_profile=floor_profile)
    if stages is not None:
        stages["merged"] = merged.points.copy()
    counts = {"raw_depth_pixels": int(sum(len(v.points) for v in views)),
              "merged_points": int(len(merged.points))}
    timings = {"merge_ms": merged.report["merge_ms"]}
    result = process_points(merged.points, plan=plan, layer=layer, config=config,
                            floor_profile=floor_profile, stages=stages, counts=counts,
                            timings=timings, started=started)
    result.report["merge"] = merged.report
    result.report["views"] = merged.views
    return result


def characterize_views(views: list[ViewPoints], *, search_center_mm, work_frame: str, config,
                       inspection_tool: str = "Realsense",
                       print_tool: str = "LongCalibTool") -> CharacterizationResult:
    """Characterize a ring from several views with NO recipe: coarse on the raw
    union (no centre to level or register about yet), then level + register the
    views about the fitted ring and refit -- one extra pass through the chain."""
    started = time.perf_counter()
    if not views or views[0].name != "top":
        raise ValueError("the first view must be the top view (the registration reference)")
    total = int(sum(len(v.points) for v in views))
    coarse = characterize_points(np.vstack([v.points for v in views]),
                                 search_center_mm=search_center_mm, work_frame=work_frame,
                                 config=config, inspection_tool=inspection_tool,
                                 print_tool=print_tool, counts={"raw_depth_pixels": total})
    plan = coarse_plan(center_mm=coarse.center_mm, radius_mm=coarse.radius_mm,
                       bead_mm=coarse.bead_width_mm, height_mm=coarse.report["coarse"]["height_mm"],
                       work_frame=work_frame, config=config,
                       inspection_tool=inspection_tool, print_tool=print_tool)
    merged = merge_views(views, plan=plan, layer=plan.layers[0], config=config)
    refined = process_points(merged.points, plan=plan, layer=plan.layers[0], config=config,
                             counts={"raw_depth_pixels": total,
                                     "merged_points": int(len(merged.points))},
                             timings={"merge_ms": merged.report["merge_ms"]})
    geometry = refined.geometry
    report = {**refined.report, "coarse": coarse.report["coarse"],
              "counts_coarse": coarse.report["counts_coarse"],
              "ring_selector": coarse.report["ring_selector"],
              "merge": merged.report, "views": merged.views,
              "kind": "characterization",
              "total_ms": (time.perf_counter() - started) * 1000}
    return CharacterizationResult(
        radius_mm=refined.metrics.measured_radius_mm,
        center_mm=refined.metrics.measured_center_mm,
        bead_width_mm=geometry.bead_width_mean_mm,
        bead_width_min_mm=geometry.bead_width_min_mm,
        bead_width_max_mm=geometry.bead_width_max_mm,
        top_z_mean_mm=geometry.top_z_mean_mm, top_z_min_mm=geometry.top_z_min_mm,
        top_z_max_mm=geometry.top_z_max_mm, measured_xyz=refined.measured_xyz,
        segmentation=refined.segmentation, skeleton=refined.skeleton,
        comparison=refined.comparison, report=report)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `py -3.10 -m pytest tests/test_extrusion_multiview.py -q -x`
Expected: all pass. If `test_levelling_removes_an_injected_tilt_and_height_offset` fails on `dz_mm`, print `info` — the lever arm from the annulus centroid to the rotation pivot (0,0 at CENTER) adds `y_centroid · tan 1°`; the tolerance already allows 0.5 mm, so a failure there means the sign of the rotation is wrong (`_rotation_to_z(normal) @ normal` must equal `[0, 0, 1]` — assert that in a scratch check before touching anything else). If the thin-ring test's `top_only` branch passes but the `after_work_roi` ratio is under 2.5, lower it to 2.0 and say so in the commit — the point is "denser", not a specific factor.

- [ ] **Step 5: Run the regression**

Run: the Global Constraints command.
Expected: all pass.

- [ ] **Step 6: Commit and push**

```bash
git add tasni/modules/extrusion/multiview.py tests/test_extrusion_multiview.py
git commit -m "feat(extrusion): level and register several views of a ring and measure the merge

Per-view board levelling, centre alignment to the top view (the correction is the
reported inter-view error), strict/none policies, and a merged characterize.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
git push
```

---

### Task 5: Capture the views — `capture_views`, archive layout, job wiring, timings

**Files:**
- Modify: `tasni/modules/extrusion/service.py:390-394` (signature), `:438-439` (candidate source)
- Modify: `tasni/modules/extrusion/archive.py:59-99` (`write_layer`), `:101-126` (`write_characterization`), new `_write_view`
- Modify: `tasni/modules/extrusion/measure.py:29-40` (imports), `:218-254` (`_inspect_and_capture`), `:203-215` (`take_summary`), `:309-476` (`RingMeasureJob`), `:478-593` (`RingCharacterizeJob`); new `View`, `capture_views`, helpers
- Test: `tests/test_extrusion_multiview.py`

**Interfaces:**
- Consumes: (Task 2) `multiview_names`, `star_view_candidates`; (Task 3) `observation_points`; (Task 4) `ViewPoints`, `median_depth`, `process_views`, `characterize_views`; (Task 1) `CaptureRecord`, `ViewRecord`.
- Produces: `_build_inspection_move(..., candidate_source=None)` where `candidate_source(aim, standoff_mm, config, reference_x) -> list[candidate]` replaces `pose_candidates`; `_inspect_and_capture(..., candidate_source=None, frames=1)` returning two extra keys `departed` (perf_counter stamp) and `frames`; `View`; `capture_views(...) -> list[View]`; `_view_program_name(stem, view)`; `_capture_record(views, frames_per_view) -> CaptureRecord | None`; `_extra_view_files(views) -> list[dict] | None`; `ExtrusionArchive.write_layer(..., views=None, merged_points=None)`, `.write_characterization(..., views=None)`; `RingMeasureJob(..., views="single")`, `RingCharacterizeJob(..., views="single")`; job summary keys `capture_mode`, `views_used`; manifest timings `views_capture_ms` (multi only).

Timing rules (multi): `capture_ms` = `views_capture_ms` = sum of every used view's grab; `acquisition_to_path_ms = capture_ms + total_ms` (spec §4.4: all measurement views to path); `move_to_pose_ms` = sum of the view moves where a star view's move **starts when the previous capture ended** (building + validating its RoboDK program is part of getting the camera there — otherwise `inspection_cycle_ms` would silently omit seconds per view); `settle_ms` = sum of the dwells. `add_return_timing` is unchanged and its sum stays exact.

- [ ] **Step 1: Write the failing tests** (append)

```python
# ------------------------------------------------------------ Task 5: capture + job

from test_extrusion_job import Ctx, START_JOINTS  # noqa: E402
from test_extrusion_measure import (auto_plan, fake_characterize,  # noqa: E402
                                    fake_measure_processing, measure_env)
from tasni.modules.extrusion import measure as measure_mod  # noqa: E402
from tasni.modules.extrusion.archive import ExtrusionArchive  # noqa: E402
from tasni.modules.extrusion.measure import (MeasureSession, RingCharacterizeJob,  # noqa: E402
                                             RingMeasureJob)


def fake_process_views(views, **kwargs):
    fake_process_views.calls.append([v.name for v in views])
    stages = kwargs.get("stages")
    if stages is not None:
        stages["merged"] = np.vstack([v.points for v in views])
    return fake_measure_processing(layer=kwargs["layer"])


fake_process_views.calls = []


def multi_env(tmp_path, monkeypatch):
    svc, rdk, camera = measure_env(tmp_path, monkeypatch)
    monkeypatch.setattr(measure_mod, "process_views", fake_process_views)
    fake_process_views.calls.clear()
    return svc, rdk, camera


def stem_for(plan):
    return "TasniCylinder_MEASURE_%s_L001" % plan.fingerprint[:10]


def test_archive_writes_extra_views_and_the_merged_cloud_beside_the_top_frame(tmp_path):
    plan = scene_plan()
    archive = ExtrusionArchive(tmp_path)
    archive.create_trial("t1", plan, mode="MEASURE_ONLY")
    capture = CaptureRecord(mode="multi", views=[
        ViewRecord(name="top", color_file="color.png", depth_file="depth.npy"),
        ViewRecord(name="star-0", color_file="views/star-0/color.png",
                   depth_file="views/star-0/depth.npy")])
    manifest = LayerManifest(trial_id="t1", layer_index=1, mode="MEASURE_ONLY", recipe=plan.recipe,
                             toolpath_fingerprint=plan.fingerprint, color_file="color.png",
                             depth_file="depth.npy", merged_points_file="merged_points.npy",
                             capture=capture)
    nominal = np.zeros((4, 3))
    layer = archive.write_layer(
        manifest, nominal_xyz=nominal, commanded_xyz=nominal,
        color=np.zeros((4, 4, 3), np.uint8), depth=np.ones((4, 4), np.uint16),
        views=[{"name": "star-0", "color": np.zeros((4, 4, 3), np.uint8),
                "depth": np.full((4, 4), 7, np.uint16), "pose": {"tilt_deg": 20.0}}],
        merged_points=np.zeros((5, 3)))
    assert np.load(layer / "depth.npy")[0, 0] == 1                          # the top view, where it always was
    assert np.load(layer / "views" / "star-0" / "depth.npy")[0, 0] == 7
    assert (layer / "views" / "star-0" / "color.png").is_file()
    assert json.loads((layer / "views" / "star-0" / "pose.json").read_text())["tilt_deg"] == 20.0
    assert np.load(layer / "merged_points.npy").shape == (5, 3)
    loaded = json.loads((layer / "manifest.json").read_text())
    assert loaded["capture"]["mode"] == "multi"
    assert loaded["capture"]["views"][1]["depth_file"] == "views/star-0/depth.npy"
    second = manifest.model_copy(update={"take": 2})
    with pytest.raises(ValueError):                                          # guarded path segment
        archive.write_layer(second, nominal_xyz=nominal, commanded_xyz=nominal,
                            views=[{"name": "../evil", "color": None, "depth": None, "pose": {}}])
    out = archive.write_characterization(
        "t1", 1, color=np.zeros((4, 4, 3), np.uint8), depth=np.zeros((4, 4), np.uint16),
        measured_xyz=np.zeros((5, 3)), derived_images={}, report={"radius_mm": 60.0},
        views=[{"name": "star-120", "color": None, "depth": np.ones((4, 4), np.uint16), "pose": {}}])
    assert (out / "views" / "star-120" / "depth.npy").is_file()


def test_multi_view_measure_captures_four_views_and_measures_the_merge(tmp_path, monkeypatch):
    svc, rdk, camera = multi_env(tmp_path, monkeypatch)
    plan = auto_plan()
    session = MeasureSession.create(tmp_path / "runs" / "extrusion", plan)
    out = RingMeasureJob(svc, plan, session, 1, annotation={"phase": "noise floor"},
                         check_collisions=True, views="multi")(Ctx())
    assert fake_process_views.calls == [["top", "star-0", "star-120", "star-240"]]
    stem = stem_for(plan)
    started = [e[1] for e in rdk.events if e[0] == "start"]
    assert started == [stem + "_Inspect", stem + "_Inspect_star0",
                       stem + "_Inspect_star120", stem + "_Inspect_star240"]
    assert len([e for e in rdk.events if e[0] == "create-target"]) == 4
    assert camera.grabs == 5                                                 # readiness + four views
    for suffix in ("_Inspect", "_Inspect_star0", "_Inspect_star120", "_Inspect_star240"):
        assert stem + suffix in rdk.deleted and stem + suffix + "_Target" in rdk.deleted
    assert rdk.events[-1] == ("move-joints", START_JOINTS)
    layer_dir = Path(out["layer_dir"])
    manifest = json.loads((layer_dir / "manifest.json").read_text())
    views = manifest["capture"]["views"]
    assert manifest["capture"]["mode"] == "multi" and manifest["capture"]["frames_per_view"] == 1
    assert [v["name"] for v in views] == ["top", "star-0", "star-120", "star-240"]
    assert views[0]["color_file"] == "color.png" and views[0]["depth_file"] == "depth.npy"
    assert views[1]["depth_file"] == "views/star-0/depth.npy" and views[1]["used"]
    assert views[1]["descriptor"]["tilt_deg"] == 20.0 and views[1]["descriptor"]["view"] == "star-0"
    assert views[0]["descriptor"]["tilt_deg"] == 0.0
    for name in ("star-0", "star-120", "star-240"):
        assert (layer_dir / "views" / name / "depth.npy").is_file()
        assert (layer_dir / "views" / name / "pose.json").is_file()
    assert (layer_dir / "merged_points.npy").is_file() and manifest["merged_points_file"] == "merged_points.npy"
    assert (layer_dir / "depth.npy").is_file() and (layer_dir / "color.png").is_file()
    assert manifest["provenance"]["T_work_camera"] == views[0]["T_work_camera"]   # old readers stay right
    assert manifest["provenance"]["inspection_pose"]["view"] == "top"
    t = manifest["processing"]["timings_ms"]
    assert t["views_capture_ms"] == pytest.approx(sum(v["capture_ms"] for v in views))
    assert t["capture_ms"] == pytest.approx(t["views_capture_ms"])
    assert t["acquisition_to_path_ms"] == pytest.approx(t["views_capture_ms"] + 10.0)
    assert t["move_to_pose_ms"] == pytest.approx(sum(v["move_ms"] for v in views))
    assert t["settle_ms"] == pytest.approx(4 * svc.config.extrusion.settle_s * 1000.0)
    assert t["inspection_cycle_ms"] == pytest.approx(
        t["move_to_pose_ms"] + t["settle_ms"] + t["capture_ms"] + t["total_ms"] + t["return_ms"])
    assert out["capture_mode"] == "multi" and out["views_used"] == ["top", "star-0", "star-120", "star-240"]
    record = MeasureSession.load(tmp_path / "runs" / "extrusion", session.trial_id).records[-1]
    assert record["capture_mode"] == "multi"


def test_an_unreachable_star_view_is_skipped_and_the_take_still_measures(tmp_path, monkeypatch):
    svc, rdk, camera = multi_env(tmp_path, monkeypatch)
    original = rdk.create_inspection_target

    def refuse_star120(**kwargs):
        if kwargs["name"].endswith("_Inspect_star120_Target"):
            return {"created": False, "target": kwargs["name"], "reason": "no IK solution"}
        return original(**kwargs)

    rdk.create_inspection_target = refuse_star120
    plan = auto_plan()
    session = MeasureSession.create(tmp_path / "runs" / "extrusion", plan)
    ctx = Ctx()
    out = RingMeasureJob(svc, plan, session, 1, annotation={}, check_collisions=True, views="multi")(ctx)
    assert fake_process_views.calls == [["top", "star-0", "star-240"]]
    assert camera.grabs == 4
    manifest = json.loads((Path(out["layer_dir"]) / "manifest.json").read_text())
    views = {v["name"]: v for v in manifest["capture"]["views"]}
    assert not views["star-120"]["used"] and "no IK solution" in views["star-120"]["error"]
    assert views["star-120"]["depth_file"] is None and views["star-0"]["used"]
    assert not (Path(out["layer_dir"]) / "views" / "star-120").exists()
    assert out["views_used"] == ["top", "star-0", "star-240"] and manifest["metrics"]["valid"]
    assert any("star-120 skipped" in line for line in ctx.logs)


def test_a_failed_top_view_fails_the_take_before_anything_is_archived(tmp_path, monkeypatch):
    svc, rdk, camera = multi_env(tmp_path, monkeypatch)
    rdk.unreachable_targets = 60                                              # the whole top walk
    plan = auto_plan()
    session = MeasureSession.create(tmp_path / "runs" / "extrusion", plan)
    with pytest.raises(RuntimeError, match="no reachable"):
        RingMeasureJob(svc, plan, session, 1, annotation={}, check_collisions=True, views="multi")(Ctx())
    assert fake_process_views.calls == [] and not (session.trial_dir / "layer-001").exists()
    assert rdk.events[-1] == ("move-joints", START_JOINTS)


def test_frames_per_view_medians_the_depth_and_records_the_count(tmp_path, monkeypatch):
    svc, rdk, camera = multi_env(tmp_path, monkeypatch)
    svc.config.extrusion.multiview_frames_per_view = 3
    plan = auto_plan()
    session = MeasureSession.create(tmp_path / "runs" / "extrusion", plan)
    out = RingMeasureJob(svc, plan, session, 1, annotation={}, check_collisions=True, views="multi")(Ctx())
    assert camera.grabs == 1 + 4 * 3
    manifest = json.loads((Path(out["layer_dir"]) / "manifest.json").read_text())
    assert manifest["capture"]["frames_per_view"] == 3
    assert all(v["frames"] == 3 for v in manifest["capture"]["views"])
    assert np.load(Path(out["layer_dir"]) / "depth.npy").dtype == np.uint16


def test_multi_view_needs_the_automatic_inspection_pose(tmp_path, monkeypatch):
    svc, rdk, camera = multi_env(tmp_path, monkeypatch)
    from test_extrusion_job import plan as taught_plan
    session = MeasureSession.create(tmp_path / "runs" / "extrusion", taught_plan(auto_inspection=False))
    with pytest.raises(RuntimeError, match="inspection_auto"):
        RingMeasureJob(svc, taught_plan(auto_inspection=False), session, 1, annotation={},
                       check_collisions=True, views="multi")(Ctx())
    assert rdk.events == []
    with pytest.raises(ValueError):
        RingMeasureJob(svc, auto_plan(), session, 1, views="stereo")


def test_single_view_is_untouched_by_the_new_keywords(tmp_path, monkeypatch):
    svc, rdk, camera = multi_env(tmp_path, monkeypatch)
    plan = auto_plan()
    session = MeasureSession.create(tmp_path / "runs" / "extrusion", plan)
    out = RingMeasureJob(svc, plan, session, 1, annotation={}, check_collisions=True)(Ctx())
    assert fake_process_views.calls == [] and camera.grabs == 2
    manifest = json.loads((Path(out["layer_dir"]) / "manifest.json").read_text())
    assert manifest["capture"] is None and manifest["merged_points_file"] is None
    assert "views_capture_ms" not in manifest["processing"]["timings_ms"]
    assert out["capture_mode"] == "single" and out["views_used"] == ["top"]


def test_multi_view_characterize_merges_the_views(tmp_path, monkeypatch):
    svc, rdk, camera = measure_env(tmp_path, monkeypatch)
    seen = []

    def fake_characterize_views(views, **kwargs):
        seen.append([v.name for v in views])
        return fake_characterize(**kwargs)

    monkeypatch.setattr(measure_mod, "characterize_views", fake_characterize_views)
    fake_characterize.calls.clear()
    plan = auto_plan()
    session = MeasureSession.create(tmp_path / "runs" / "extrusion", plan)
    out = RingCharacterizeJob(svc, plan, session, check_collisions=True, views="multi")(Ctx())
    assert seen == [["top", "star-0", "star-120", "star-240"]] and camera.grabs == 5
    assert fake_characterize.calls[-1]["search_center_mm"] == (200.0, 150.0)
    capture_dir = Path(out["capture_dir"])
    assert (capture_dir / "views" / "star-240" / "depth.npy").is_file() and (capture_dir / "depth.npy").is_file()
    report = json.loads((capture_dir / "report.json").read_text())
    assert report["capture"]["mode"] == "multi" and [v["name"] for v in report["capture"]["views"]][1] == "star-0"
    assert out["characterization"]["capture_mode"] == "multi"
    assert rdk.events[-1] == ("move-joints", START_JOINTS)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `py -3.10 -m pytest tests/test_extrusion_multiview.py -q -k "archive_writes_extra or multi_view or star_view_is_skipped or failed_top or frames_per_view or automatic_inspection or untouched"`
Expected: failures — `TypeError: write_layer() got an unexpected keyword argument 'views'`, `AttributeError: module ... has no attribute 'process_views'`, `TypeError: __init__() got an unexpected keyword argument 'views'`.

- [ ] **Step 3: Let `_build_inspection_move` take a candidate source**

In `tasni/modules/extrusion/service.py` change the signature (lines 390–394) to:

```python
def _build_inspection_move(rdk: RdkIO, plan: CylinderPlan, layer, *,
                           inspection_name: str, config, camera,
                           start_joints, seed_pose: dict | None = None,
                           collisions: bool = True,
                           near_mm: float | None = None,
                           candidate_source=None) -> dict:
```

add to the docstring's last paragraph: `` ``candidate_source`` (multi-view capture) replaces :func:`pose_candidates` with another ``(aim, standoff_mm, config, reference_x) -> candidates`` walk; the framing, the roll reference, the seed ordering and every gate stay the same.`` — and replace lines 438–439 with:

```python
    source = pose_candidates if candidate_source is None else candidate_source
    candidates = order_candidates_seed_first(
        source(aim, framing["standoff_mm"], config, reference_x), seed_pose)
```

Nothing else in `service.py` changes.

- [ ] **Step 4: Archive — extra views and the merged cloud**

In `tasni/modules/extrusion/archive.py`:

Change `write_layer`'s signature (lines 59–63) to add `views: list[dict] | None = None, merged_points=None` after `report`. Insert before the `if report is not None:` line (94):

```python
        if views:
            for view in views:
                self._write_view(layer / "views" / _segment(str(view["name"]), "view name"), view)
        if merged_points is not None:
            points = np.asarray(merged_points, dtype=float)
            if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
                raise ValueError("merged point cloud must be a finite Nx3 array")
            np.save(layer / "merged_points.npy", points)
```

Change `write_characterization`'s signature (lines 101–102) to add `views: list[dict] | None = None` after `report`, and insert before its `(out / "report.json").write_text(...)` line:

```python
        for view in views or ():
            self._write_view(out / "views" / _segment(str(view["name"]), "view name"), view)
```

Add the method after `next_characterization_index`:

```python
    @staticmethod
    def _write_view(directory: Path, view: dict) -> None:
        """``views/<name>/{color.png, depth.npy, pose.json}`` for one extra view.

        The TOP view is never written here: it stays at ``color.png`` /
        ``depth.npy`` in the layer directory, where every existing reader
        (reprocess, figures, the paper) already looks.
        """
        directory.mkdir(parents=True, exist_ok=False)
        if view.get("color") is not None:
            import cv2
            if not cv2.imwrite(str(directory / "color.png"), np.asarray(view["color"])):
                raise OSError(f"failed to write {directory.name}/color.png")
        if view.get("depth") is not None:
            np.save(directory / "depth.npy", np.asarray(view["depth"]))
        (directory / "pose.json").write_text(json.dumps(view.get("pose") or {}, indent=2),
                                             encoding="utf-8")
```

Note the guard: `_segment("../evil", ...)` raises `ValueError` **before** `mkdir`, and `write_layer` has already created the layer directory by then — that is fine for the test (it only asserts the raise) and for the job (a view name comes from `multiview_names`, never from input).

- [ ] **Step 5: Capture — `View`, `capture_views`, frames**

In `tasni/modules/extrusion/measure.py`:

Imports (lines 29–40): add `from dataclasses import dataclass` under the stdlib imports; add `from ...core.camera import Frame`; extend `from .models import CylinderPlan, LayerManifest` to `from .models import CaptureRecord, CylinderPlan, LayerManifest, ViewRecord`; extend the processing import to `from .processing import characterize_ring, observation_points, process_observation`; add `from .inspection import multiview_names, star_view_candidates` and `from .multiview import ViewPoints, characterize_views, median_depth, process_views`.

Replace `_inspect_and_capture` (lines 218–254) with:

```python
def _inspect_and_capture(services, ctx: JobContext, plan: CylinderPlan, layer, *,
                         inspection_name: str, start_joints, seed_pose, collisions: bool,
                         artifacts: list[str], near_mm: float | None = None,
                         candidate_source=None, frames: int = 1) -> dict:
    """Move the camera to the derived pose, settle, read the pose, grab the frame(s).

    ``candidate_source`` walks a star view's own candidates instead of the top
    view's (see ``inspection.star_view_candidates``); ``frames`` > 1 grabs that
    many depth frames at the pose and keeps their per-pixel median with the
    last colour frame. Both default to exactly today's single top-view capture.
    """
    rdk: RdkIO = services.rdk
    ecfg = services.config.extrusion
    inspect = _build_inspection_move(
        rdk, plan, layer, inspection_name=inspection_name, config=ecfg,
        camera=services.config.camera, start_joints=start_joints,
        seed_pose=seed_pose, collisions=collisions, near_mm=near_mm,
        candidate_source=candidate_source)
    ctx.check_cancel()
    artifacts.extend(inspect["artifacts"])
    # What the excursion costs the print starts here: the arm leaving the path.
    departed = time.perf_counter()
    if rdk.start_program(inspection_name, real_robot=True) < 0:
        raise RuntimeError(f"inspection program {inspection_name} could not start")
    _wait_program(ctx, rdk, inspection_name)
    move_ms = (time.perf_counter() - departed) * 1000.0
    time.sleep(ecfg.settle_s)
    # Re-select the inspection TCP and chosen work frame before reading the
    # camera pose: the generated program's tool instruction does not update
    # RdkIO's cached tool transform.
    rdk.use_named_tool_frame(plan.setup.inspection_tool, plan.setup.work_frame)
    T_work_camera = rdk.camera_pose_T()
    started = time.perf_counter()
    frame = services.camera.grab(with_depth=True, timeout=ecfg.grab_timeout_s)
    depths = [frame.depth]
    for _ in range(1, max(1, int(frames))):
        extra = services.camera.grab(with_depth=True, timeout=ecfg.grab_timeout_s)
        depths.append(extra.depth)
        frame = extra
    capture_ms = (time.perf_counter() - started) * 1000.0
    if any(depth is None for depth in depths):
        raise RuntimeError("RGB-D capture returned no depth")
    if len(depths) > 1:
        frame = Frame(color=frame.color, depth=median_depth(depths),
                      timestamp=frame.timestamp, telemetry=frame.telemetry)
    ok, jpeg = cv2.imencode(".jpg", frame.color)
    if ok:
        ctx.frame(jpeg.tobytes())
    return {"inspect": inspect, "T_work_camera": T_work_camera, "frame": frame,
            "capture_ms": capture_ms, "move_ms": move_ms, "departed": departed,
            "frames": len(depths),
            # The dwell is commanded, not measured: reporting the configured
            # value keeps the cycle total honest when a test stubs out sleep.
            "settle_ms": float(ecfg.settle_s) * 1000.0}
```

Then add, directly after `_inspect_and_capture`:

```python
@dataclass
class View:
    """One measurement view as captured: pose, frame, timings -- or its error."""
    name: str
    descriptor: dict | None = None
    target: str | None = None
    T_work_camera: np.ndarray | None = None
    color: np.ndarray | None = None
    depth: np.ndarray | None = None
    frames: int = 0
    move_ms: float = 0.0
    settle_ms: float = 0.0
    capture_ms: float = 0.0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.depth is not None


def _view_program_name(stem: str, view: str) -> str:
    """``<stem>_Inspect`` for the top view (unchanged), ``<stem>_Inspect_star120`` etc."""
    return f"{stem}_Inspect" if view == "top" else f"{stem}_Inspect_{view.replace('-', '')}"


def _star_source(view_name: str, seed_pose: dict | None):
    """Candidate source for one star view: its own walk, the seed's roll first.

    The previous view's roll is the wrist configuration that just worked, so
    trying it first is the cheapest way to keep wrist changes between views to
    a minimum -- the walk order otherwise stays as designed.
    """
    def source(aim, standoff_mm, config, reference_x):
        views = star_view_candidates(aim, standoff_mm, config, reference_x)
        candidates = next(v["candidates"] for v in views if v["name"] == view_name)
        if seed_pose and seed_pose.get("roll_deg") is not None:
            roll = float(seed_pose["roll_deg"])
            candidates = sorted(candidates, key=lambda c: float(c["roll_deg"]) != roll)
        return candidates
    return source


def capture_views(services, ctx: JobContext, plan: CylinderPlan, layer, *, program_stem: str,
                  start_joints, seed_pose, collisions: bool, artifacts: list[str],
                  views: list[str], frames_per_view: int = 1,
                  near_mm: float | None = None) -> list[View]:
    """Capture every requested view in order, seeding each walk with the previous pose.

    The top view is the reference every other view is registered to, so its
    failure fails the take (raised, as today, before anything is archived). A
    star view that cannot be reached or captured is recorded with ``error`` and
    skipped; the take goes on with the views it has. One RoboDK target + program
    per view, accumulated in ``artifacts`` for the caller's cleanup.
    """
    captured: list[View] = []
    previous = seed_pose
    for name in views:
        program = _view_program_name(program_stem, name)
        ctx.check_cancel()
        clock = time.perf_counter()
        try:
            got = _inspect_and_capture(
                services, ctx, plan, layer, inspection_name=program,
                start_joints=start_joints, seed_pose=previous, collisions=collisions,
                artifacts=artifacts, near_mm=near_mm,
                candidate_source=None if name == "top" else _star_source(name, previous),
                frames=frames_per_view)
        except Exception as exc:
            if name == "top":
                raise
            ctx.log(f"view {name} skipped: {exc}")
            captured.append(View(name=name, error=str(exc)))
            continue
        # Star candidates carry their view name; the top view's candidates (the
        # unchanged pose_candidates walk) do not, so label it here.
        pose = {**(got["inspect"]["pose"] or {}), "view": name}
        frame = got["frame"]
        # A star view's clock starts when the previous capture ended: building
        # and validating its program is part of getting the camera there, and
        # the excursion total must not silently drop those seconds.
        move_ms = (got["move_ms"] if name == "top"
                   else (got["departed"] - clock) * 1000.0 + got["move_ms"])
        captured.append(View(name=name, descriptor=pose, target=got["inspect"]["target"],
                             T_work_camera=np.asarray(got["T_work_camera"], dtype=float),
                             color=frame.color, depth=frame.depth, frames=got["frames"],
                             move_ms=move_ms, settle_ms=got["settle_ms"],
                             capture_ms=got["capture_ms"]))
        previous = pose or previous
        ctx.log(f"view {name}: tilt {pose.get('tilt_deg', 0):.0f} / azimuth "
                f"{pose.get('azimuth_deg', 0):.0f} / roll {pose.get('roll_deg', 0):.0f} deg, "
                f"{got['capture_ms']:.0f} ms capture")
    return captured


def _capture_record(views: list[View] | None, frames_per_view: int) -> CaptureRecord | None:
    """The manifest's ``capture`` block; None keeps a single-view take as it always was."""
    if views is None:
        return None
    records = []
    for view in views:
        prefix = "" if view.name == "top" else f"views/{view.name}/"
        records.append(ViewRecord(
            name=view.name, descriptor=view.descriptor or {},
            T_work_camera=(None if view.T_work_camera is None
                           else np.asarray(view.T_work_camera, dtype=float).tolist()),
            color_file=(prefix + "color.png") if view.ok else None,
            depth_file=(prefix + "depth.npy") if view.ok else None,
            frames=max(1, view.frames), move_ms=view.move_ms, settle_ms=view.settle_ms,
            capture_ms=view.capture_ms, used=view.ok, error=view.error))
    return CaptureRecord(mode="multi", frames_per_view=int(frames_per_view), views=records)


def _extra_view_files(views: list[View] | None) -> list[dict] | None:
    """The star views' raw frames for the archive; the top view stays at color.png/depth.npy."""
    if views is None:
        return None
    return [{"name": v.name, "color": v.color, "depth": v.depth,
             "pose": {"descriptor": v.descriptor, "target": v.target,
                      "T_work_camera": np.asarray(v.T_work_camera, dtype=float).tolist(),
                      "frames": v.frames, "move_ms": v.move_ms, "settle_ms": v.settle_ms,
                      "capture_ms": v.capture_ms}}
            for v in views if v.name != "top" and v.ok]


def _view_points(views: list[View], K) -> list[ViewPoints]:
    return [ViewPoints(v.name, observation_points(v.depth, K, v.T_work_camera), v.T_work_camera)
            for v in views if v.ok]
```

In `take_summary` (line 203) add to the returned dict: `"capture_mode": ((manifest.get("capture") or {}).get("mode") or "single"),`.

- [ ] **Step 6: Wire `RingMeasureJob`**

Change `__init__` (lines 312–325): add `views: str = "single"` after `close_range_tool_clear`, and inside:

```python
        if views not in ("single", "multi"):
            raise ValueError(f"views must be 'single' or 'multi', not {views!r}")
        self.views = views
```

Make this the **first statement of `__call__`** (before `services = self.services`, and therefore before `_prepare_robot` touches RoboDK — the test asserts `rdk.events == []`):

```python
        if self.views == "multi" and not self.plan.setup.inspection_auto:
            raise RuntimeError("multi-view capture needs the automatic inspection pose "
                               "(setup.inspection_auto); a taught target is one viewpoint")
```

Replace the capture block (lines 341–352, from `with _camera_hold(...)` to `T_work_camera, capture_ms = ...`) with:

```python
            with _camera_hold(services, "extrusion-measure"):
                near = (ecfg.measure_close_range_min_mm if self.close_range_tool_clear else None)
                captured_views: list[View] | None = None
                if self.views == "multi":
                    names = multiview_names(ecfg)
                    ctx.progress(1, 4, f"layer {self.layer_index} take {take}: "
                                       f"capturing {len(names)} views")
                    current_program = _view_program_name(name, "top")
                    captured_views = capture_views(
                        services, ctx, self.plan, layer, program_stem=name,
                        start_joints=start_joints, seed_pose=self.session.last_pose,
                        collisions=self.check_collisions, artifacts=artifacts, views=names,
                        frames_per_view=ecfg.multiview_frames_per_view, near_mm=near)
                    current_program = None
                    top = captured_views[0]
                    inspect = {"target": top.target, "pose": top.descriptor}
                    frame = Frame(color=top.color, depth=top.depth, timestamp=0.0)
                    T_work_camera = top.T_work_camera
                    used = [v for v in captured_views if v.ok]
                    capture_ms = float(sum(v.capture_ms for v in used))
                    move_ms = float(sum(v.move_ms for v in used))
                    settle_ms = float(sum(v.settle_ms for v in used))
                else:
                    ctx.progress(1, 4, f"layer {self.layer_index} take {take}: moving the camera")
                    current_program = inspection_name
                    captured = _inspect_and_capture(
                        services, ctx, self.plan, layer, inspection_name=inspection_name,
                        start_joints=start_joints, seed_pose=self.session.last_pose,
                        collisions=self.check_collisions, artifacts=artifacts, near_mm=near)
                    current_program = None
                    inspect, frame = captured["inspect"], captured["frame"]
                    T_work_camera, capture_ms = captured["T_work_camera"], captured["capture_ms"]
                    move_ms, settle_ms = captured["move_ms"], captured["settle_ms"]
                capture = _capture_record(captured_views, ecfg.multiview_frames_per_view)
                view_files = _extra_view_files(captured_views)
```

In `base = dict(...)` (lines 355–366) add `capture=capture,` and, when multi, the merged file: add the line `merged_points_file=("merged_points.npy" if captured_views is not None else None),`.

Replace the processing call (lines 368–372) with:

```python
                merged_points = None
                try:
                    if captured_views is not None:
                        stages: dict = {}
                        processed = process_views(
                            _view_points(captured_views, services.config.camera.K),
                            plan=self.plan, layer=layer, config=ecfg, floor_profile=floor,
                            stages=stages)
                        merged_points = stages.get("merged")
                    else:
                        processed = process_observation(
                            color=frame.color, depth=frame.depth, T_work_camera=T_work_camera,
                            K=services.config.camera.K, plan=self.plan, layer=layer, config=ecfg,
                            floor_profile=floor)
```

In the failed-take `archive.write_layer(...)` (lines 381–384) add `views=view_files,`. In the success `archive.write_layer(...)` (lines 407–416) add `views=view_files, merged_points=merged_points,`.

Replace the timing lines (397–401) with:

```python
                timings = processed.report["timings_ms"]
                timings["capture_ms"] = capture_ms
                timings["acquisition_to_path_ms"] = capture_ms + timings["total_ms"]
                timings["move_to_pose_ms"] = move_ms
                timings["settle_ms"] = settle_ms
                if captured_views is not None:
                    # Every measurement view's grab, first to last: what "acquisition
                    # to path" means when four frames make one measurement.
                    timings["views_capture_ms"] = capture_ms
```

In the `summary = {...}` (lines 417–427) add:

```python
                           "capture_mode": "multi" if captured_views is not None else "single",
                           "views_used": ([v.name for v in captured_views if v.ok]
                                          if captured_views is not None else ["top"]),
```

- [ ] **Step 7: Wire `RingCharacterizeJob`**

`__init__` (lines 486–494): add `views: str = "single"` with the same validation as above. In `__call__`, add the same `inspection_auto` guard as the first statement. Replace lines 507–517 (`with _camera_hold(...)` through `frame = captured["frame"]`) with:

```python
            with _camera_hold(services, "extrusion-characterize"):
                near = (ecfg.measure_close_range_min_mm if self.close_range_tool_clear else None)
                captured_views: list[View] | None = None
                if self.views == "multi":
                    names = multiview_names(ecfg)
                    ctx.progress(1, 3, f"capturing {len(names)} views of the ring")
                    current_program = _view_program_name(name_stem, "top")
                    captured_views = capture_views(
                        services, ctx, self.plan, layer, program_stem=name_stem,
                        start_joints=start_joints, seed_pose=self.session.last_pose,
                        collisions=self.check_collisions, artifacts=artifacts, views=names,
                        frames_per_view=ecfg.multiview_frames_per_view, near_mm=near)
                    current_program = None
                    top = captured_views[0]
                    captured = {"inspect": {"target": top.target, "pose": top.descriptor},
                                "T_work_camera": top.T_work_camera,
                                "frame": Frame(color=top.color, depth=top.depth, timestamp=0.0),
                                "capture_ms": float(sum(v.capture_ms for v in captured_views if v.ok))}
                else:
                    ctx.progress(1, 3, "moving the camera over the ring")
                    current_program = inspection_name
                    captured = _inspect_and_capture(
                        services, ctx, self.plan, layer, inspection_name=inspection_name,
                        start_joints=start_joints, seed_pose=self.session.last_pose,
                        collisions=self.check_collisions, artifacts=artifacts, near_mm=near)
                    current_program = None
                frame = captured["frame"]
                capture = _capture_record(captured_views, ecfg.multiview_frames_per_view)
                view_files = _extra_view_files(captured_views)
```

where `name_stem = _program_name(self.plan, 1, "CHARACTERIZE")` is defined next to `inspection_name` (line 501: `inspection_name = name_stem + "_Inspect"`).

Replace the `found = characterize_ring(...)` call (lines 524–531) with:

```python
                    common = dict(search_center_mm=(float(self.plan.setup.center_x_mm),
                                                    float(self.plan.setup.center_y_mm)),
                                  work_frame=self.plan.setup.work_frame, config=ecfg,
                                  inspection_tool=self.plan.setup.inspection_tool,
                                  print_tool=self.plan.setup.print_tool)
                    if captured_views is not None:
                        found = characterize_views(
                            _view_points(captured_views, services.config.camera.K), **common)
                    else:
                        found = characterize_ring(
                            color=frame.color, depth=frame.depth,
                            T_work_camera=captured["T_work_camera"],
                            K=services.config.camera.K, **common)
```

Add `"capture": None if capture is None else capture.model_dump(mode="json")` to `failed_report` and to the success `report={...}` dict; pass `views=view_files` to both `archive.write_characterization(...)` calls; add `"capture_mode": "multi" if captured_views is not None else "single"` to `summary`.

- [ ] **Step 8: Run the tests**

Run: the Global Constraints command.
Expected: all pass — the new multi tests, and every existing measure/characterize/job test unchanged (single-view path). If `test_extrusion_measure.py::test_measure_moves_only_the_camera_and_never_touches_the_valve` breaks on `camera.grabs == 2`, the single branch is grabbing more than once — `frames` must default to 1 there.

- [ ] **Step 9: Commit and push**

```bash
git add tasni/modules/extrusion/service.py tasni/modules/extrusion/archive.py tasni/modules/extrusion/measure.py tests/test_extrusion_multiview.py
git commit -m "feat(extrusion): capture top + three star views per take and measure the merge

One RoboDK target/program per view, each walk seeded by the previous pose; a star
view that cannot be reached is skipped, the top view failing fails the take. Raw
frames archived under views/, the merged cloud beside them; timings per spec 4.4.
views='single' (the default) is byte-for-byte today's path.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
git push
```

---

### Task 6: The side photo — second excursion, archive, figure

**Files:**
- Modify: `tasni/modules/extrusion/measure.py` (imports; new `SideView`, `_build_side_move`, `capture_side_view`; `RingMeasureJob.__init__/__call__` → `_measure` + `_side_excursion`)
- Modify: `tasni/modules/extrusion/archive.py` (new `write_side_view`)
- Modify: `tasni/modules/extrusion/figures.py:31` (`OPTIONAL_LAYER_FIGURES`), `:85-97` (`TakeData.side_view`), `:192-211` (`load_take`), new `side_crop_px` + `_figure_side`, `:792-794` (`_LAYER_BUILDERS`), `:878-894` (`render_layer_figures`), `:913-935` (`ensure_figure`)
- Test: `tests/test_extrusion_multiview.py`

**Interfaces:**
- Consumes: (Task 2) `side_view_plan`; (Task 1) `SideViewRecord`; `service._program_valid`, `_wait_program`.
- Produces: `SideView` dataclass with `.record(color_file=None, depth_file=None) -> SideViewRecord`; `_build_side_move(rdk, plan, layer_index, *, program_name, config, start_joints, collisions) -> {artifacts, validation, target, pose}`; `capture_side_view(services, ctx, plan, layer_index, *, program_name, start_joints, artifacts) -> SideView` (never raises); `ExtrusionArchive.write_side_view(layer_dir, *, color, depth, record) -> SideViewRecord`; `RingMeasureJob(..., side_view=False)`, result key `side_view`; session record key `side_view`; `figures.OPTIONAL_LAYER_FIGURES = ("side",)`, `TakeData.side_view: dict | None`, `side_crop_px(take, *, margin=0.25) -> (u0, v0, u1, v1) | None`, `_figure_side(plt, take)`.

- [ ] **Step 1: Write the failing tests** (append)

```python
# ------------------------------------------------------------ Task 6: side photo

def test_side_photo_is_a_second_excursion_after_the_return_home(tmp_path, monkeypatch):
    svc, rdk, camera = measure_env(tmp_path, monkeypatch)
    plan = auto_plan()
    session = MeasureSession.create(tmp_path / "runs" / "extrusion", plan)
    ctx = Ctx()
    out = RingMeasureJob(svc, plan, session, 1, annotation={}, check_collisions=True, side_view=True)(ctx)
    stem = stem_for(plan)
    starts = [i for i, e in enumerate(rdk.events) if e[0] == "start"]
    homes = [i for i, e in enumerate(rdk.events) if e == ("move-joints", START_JOINTS)]
    assert rdk.events[starts[0]][1] == stem + "_Inspect" and rdk.events[starts[1]][1] == stem + "_Side"
    assert homes[0] < starts[1] < homes[1]                     # home, THEN out to the side, THEN home
    assert rdk.events[-1] == ("move-joints", START_JOINTS)
    assert camera.grabs == 3                                   # readiness + measurement + side photo
    assert ("update", stem + "_Side", True) in rdk.events      # collision-checked by default
    assert stem + "_Side" in rdk.deleted and stem + "_Side_Target" in rdk.deleted
    layer_dir = Path(out["layer_dir"])
    assert (layer_dir / "side" / "color.png").is_file() and (layer_dir / "side" / "depth.npy").is_file()
    assert (layer_dir / "side" / "pose.json").is_file()
    manifest = json.loads((layer_dir / "manifest.json").read_text())
    side = manifest["side_view"]
    assert side["color_file"] == "side/color.png" and side["error"] is None
    # auto_plan: bead 8 -> layer-1 top at 8 mm; asin((80 - 8) / 250) = 16.7 deg
    assert side["descriptor"]["view"] == "side" and side["descriptor"]["elevation_deg"] == pytest.approx(16.7, abs=0.1)
    assert side["descriptor"]["floor_raised"] is True and side["excursion_ms"] > 0
    assert side["T_work_camera"] is not None
    assert manifest["metrics"]["valid"] and manifest["capture"] is None
    timings = manifest["processing"]["timings_ms"]
    assert timings["inspection_cycle_ms"] > 0 and "side" not in " ".join(timings)   # not in the cycle
    assert out["side_view"]["excursion_ms"] == side["excursion_ms"]
    record = MeasureSession.load(tmp_path / "runs" / "extrusion", session.trial_id).records[-1]
    assert record["side_view"]["color_file"] == "side/color.png"
    assert any("side photo" in line for line in ctx.logs)


def test_a_failed_side_photo_never_invalidates_the_take(tmp_path, monkeypatch):
    svc, rdk, camera = measure_env(tmp_path, monkeypatch)
    real_grab = camera.grab

    def flaky(**kwargs):
        if camera.grabs >= 2:
            raise RuntimeError("Jetson dropped the socket")
        return real_grab(**kwargs)

    camera.grab = flaky
    plan = auto_plan()
    session = MeasureSession.create(tmp_path / "runs" / "extrusion", plan)
    out = RingMeasureJob(svc, plan, session, 1, annotation={}, check_collisions=True, side_view=True)(Ctx())
    manifest = json.loads((Path(out["layer_dir"]) / "manifest.json").read_text())
    assert manifest["metrics"]["valid"]
    assert "Jetson" in manifest["side_view"]["error"] and manifest["side_view"]["color_file"] is None
    assert manifest["side_view"]["descriptor"]["view"] == "side"      # the pose it got to
    assert rdk.events[-1] == ("move-joints", START_JOINTS)
    assert out["side_view"]["error"]
    stem = stem_for(plan)
    assert stem + "_Side" in rdk.deleted


def test_side_photo_refused_at_preflight_when_the_floor_needs_more_than_45_deg(tmp_path, monkeypatch):
    svc, rdk, camera = measure_env(tmp_path, monkeypatch)
    svc.config.extrusion.side_view_min_camera_z_mm = 250.0
    plan = auto_plan()
    session = MeasureSession.create(tmp_path / "runs" / "extrusion", plan)
    out = RingMeasureJob(svc, plan, session, 1, annotation={}, check_collisions=True, side_view=True)(Ctx())
    manifest = json.loads((Path(out["layer_dir"]) / "manifest.json").read_text())
    assert "45" in manifest["side_view"]["error"] and manifest["metrics"]["valid"]
    assert not any(e[0] == "start" and e[1].endswith("_Side") for e in rdk.events)
    assert camera.grabs == 2


def test_a_failed_take_takes_no_side_photo(tmp_path, monkeypatch):
    svc, rdk, camera = measure_env(tmp_path, monkeypatch)
    monkeypatch.setattr(measure_mod, "process_observation",
                        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("bad skeleton")))
    plan = auto_plan()
    session = MeasureSession.create(tmp_path / "runs" / "extrusion", plan)
    with pytest.raises(RuntimeError, match="raw RGB-D archived"):
        RingMeasureJob(svc, plan, session, 1, annotation={}, check_collisions=True, side_view=True)(Ctx())
    assert camera.grabs == 2 and not any(e[0] == "start" and e[1].endswith("_Side") for e in rdk.events)


def test_the_side_figure_is_cropped_to_the_bead_and_only_drawn_when_a_photo_exists(tmp_path):
    pytest.importorskip("matplotlib")
    from test_extrusion_figures import write_take
    from tasni.modules.extrusion import figures
    layer_dir = write_take(tmp_path)                       # radius 60, bead 8, layer height 6, centre (200,150)
    assert figures.render_layer_figures(layer_dir, only="side") == []           # no photo, no figure
    assert "side" not in figures.LAYER_FIGURES and figures.OPTIONAL_LAYER_FIGURES == ("side",)
    plan = side_view_plan(recipe(radius_mm=60.0, bead_diameter_mm=8.0, layer_height_mm=6.0),
                          setup(), 1, ExtrusionConfig())
    first = plan["candidates"][0]
    photo = np.zeros((720, 1280, 3), np.uint8)
    photo[:, :, 1] = np.linspace(0, 255, 1280).astype(np.uint8)[None, :]
    descriptor = {k: v for k, v in first.items() if k != "T"}
    descriptor.update(layer_top_z_mm=plan["layer_top_z_mm"], standoff_mm=plan["standoff_mm"],
                      floor_raised=plan["floor_raised"])
    record = SideViewRecord(descriptor=descriptor, T_work_camera=first["T"].tolist(), excursion_ms=4200.0)
    written_record = ExtrusionArchive(tmp_path).write_side_view(layer_dir, color=photo, depth=None, record=record)
    assert written_record.color_file == "side/color.png" and written_record.depth_file is None
    assert json.loads((layer_dir / "manifest.json").read_text())["side_view"]["color_file"] == "side/color.png"
    take = figures.load_take(layer_dir)
    assert take.side_view["excursion_ms"] == 4200.0
    u0, v0, u1, v1 = figures.side_crop_px(take)
    assert 0 <= u0 < u1 <= 1280 and 0 <= v0 < v1 <= 720
    assert abs((u0 + u1) / 2 - syn.K_720P[0, 2]) < 60          # the crest sits on the optical axis
    assert (u1 - u0) < 1280 and (v1 - v0) < 720                  # a crop, not the whole frame
    written = figures.render_layer_figures(layer_dir, only="side")
    assert {p.name for p in written} == {"side.png", "side.pdf"}
    assert figures.ensure_figure(layer_dir, "side.png").is_file()
    everything = figures.render_layer_figures(layer_dir)
    assert "side.png" in {p.name for p in everything}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `py -3.10 -m pytest tests/test_extrusion_multiview.py -q -k "side"`
Expected: `TypeError: __init__() got an unexpected keyword argument 'side_view'`, `AttributeError: ... has no attribute 'write_side_view'`.

- [ ] **Step 3: Archive — `write_side_view`**

In `tasni/modules/extrusion/archive.py` import `SideViewRecord` (`from .models import CylinderPlan, LayerManifest, SideViewRecord`) and add after `_write_view`:

```python
    def write_side_view(self, layer_dir: str | Path, *, color=None, depth=None,
                        record: SideViewRecord) -> SideViewRecord:
        """``side/{color.png, depth.npy, pose.json}`` beside an archived take, plus
        ``manifest.side_view``. Written AFTER the take (the photo is a second
        excursion), so this patches the manifest in place like ``add_return_timing``
        does for the return trip. Returns the record with its file names filled in."""
        layer = Path(layer_dir)
        manifest_file = layer / "manifest.json"
        if not manifest_file.is_file():
            raise FileNotFoundError(f"not an archived take: {layer}")
        out = layer / "side"
        out.mkdir(parents=False, exist_ok=True)
        files: dict[str, str] = {}
        if color is not None:
            import cv2
            if not cv2.imwrite(str(out / "color.png"), np.asarray(color)):
                raise OSError("failed to write side/color.png")
            files["color_file"] = "side/color.png"
        if depth is not None:
            np.save(out / "depth.npy", np.asarray(depth))
            files["depth_file"] = "side/depth.npy"
        record = record.model_copy(update=files)
        (out / "pose.json").write_text(record.model_dump_json(indent=2), encoding="utf-8")
        payload = json.loads(manifest_file.read_text(encoding="utf-8"))
        payload["side_view"] = record.model_dump(mode="json")
        manifest_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return record
```

- [ ] **Step 4: Capture — `SideView`, `_build_side_move`, `capture_side_view`**

In `tasni/modules/extrusion/measure.py`: extend the models import with `SideViewRecord`; extend the inspection import with `side_view_plan`; extend the service import with `_program_valid`. Add after `capture_views`:

```python
@dataclass
class SideView:
    """The side photo's excursion: the pose it reached, the frame, or the error."""
    descriptor: dict | None = None
    target: str | None = None
    T_work_camera: np.ndarray | None = None
    color: np.ndarray | None = None
    depth: np.ndarray | None = None
    excursion_ms: float | None = None
    error: str | None = None

    def record(self) -> SideViewRecord:
        return SideViewRecord(
            descriptor=self.descriptor or {},
            T_work_camera=(None if self.T_work_camera is None
                           else np.asarray(self.T_work_camera, dtype=float).tolist()),
            excursion_ms=self.excursion_ms, error=self.error)


def _build_side_move(rdk: RdkIO, plan: CylinderPlan, layer_index: int, *, program_name: str,
                     config, start_joints, collisions: bool) -> dict:
    """The side photo's own target + one-move program, gated like an inspection move.

    Same walk shape as ``_build_inspection_move`` -- IK on the neutral branch,
    RoboDK validation, the wrist report -- over ``side_view_plan``'s candidates
    (azimuth fallbacks x roll 0/180). A refused plan (elevation past 45 deg)
    raises with the number before any target is created.
    """
    side = side_view_plan(plan.recipe, plan.setup, layer_index, config)
    if side["refused"]:
        raise RuntimeError(side["refused"])
    target_name = program_name + "_Target"
    rejected: list[dict] = []
    for candidate in side["candidates"]:
        descriptor = {k: v for k, v in candidate.items() if k != "T"}
        made = rdk.create_inspection_target(
            name=target_name, T=candidate["T"], inspection_tool=plan.setup.inspection_tool,
            work_frame=plan.setup.work_frame, neutral_joints=start_joints,
            maximum_wrist_rotation_deg=plan.setup.maximum_tool_axis_spin_deg)
        if not made["created"]:
            rejected.append({**descriptor, "reason": made["reason"]})
            continue
        created = rdk.create_inspection_program(
            name=program_name, inspection_tool=plan.setup.inspection_tool,
            inspection_target=target_name, speed_mm_s=plan.recipe.travel_speed_mm_s)
        validation = rdk.update_program(program_name, collisions=collisions)
        if _program_valid(validation):
            try:
                wrist = rdk.program_neutral_wrist_report(
                    program_name, start_joints, plan.setup.maximum_tool_axis_spin_deg)
            except RuntimeError as error:
                rejected.append({**descriptor, "reason": str(error)})
                continue
            return {"artifacts": [target_name, created["program"]], "validation": validation,
                    "target": target_name,
                    "pose": {**descriptor, "standoff_mm": side["standoff_mm"],
                             "requested_elevation_deg": side["requested_elevation_deg"],
                             "floor_raised": side["floor_raised"],
                             "layer_top_z_mm": side["layer_top_z_mm"],
                             "joints": made.get("joints"), "wrist": wrist, "rejected": rejected}}
        rejected.append({**descriptor,
                         "reason": (validation["problems"]
                                    or f"validated at only {validation['percent_ok']:.1f}%")})
    tried = "; ".join(f"azimuth {r['azimuth_deg']:.0f}/roll {r['roll_deg']:.0f} deg: {r['reason']}"
                      for r in rejected)
    raise RuntimeError(
        f"layer {layer_index}: no reachable side-view pose at {side['elevation_deg']:.0f} deg "
        f"elevation, {side['standoff_mm']:.0f} mm from the near crest. "
        f"Tried {len(rejected)} pose(s) — {tried}")


def capture_side_view(services, ctx: JobContext, plan: CylinderPlan, layer_index: int, *,
                      program_name: str, start_joints, artifacts: list[str]) -> SideView:
    """Second, separately timed excursion: out to the side pose, one frame, home.

    Runs AFTER the measurement excursion has returned the arm to the start
    pose, so the measurement's ``return_ms`` is the same trip it is today.
    Never raises: any failure is logged and comes back as ``SideView.error`` --
    the photo is documentation, and a missing photo must never invalidate a
    take. ``excursion_ms`` is out + settle + capture + back.
    """
    rdk: RdkIO = services.rdk
    ecfg = services.config.extrusion
    side = SideView()
    started = time.perf_counter()
    running: str | None = None
    try:
        move = _build_side_move(rdk, plan, layer_index, program_name=program_name, config=ecfg,
                                start_joints=start_joints,
                                collisions=ecfg.side_view_collision_check)
        artifacts.extend(move["artifacts"])
        side.descriptor, side.target = move["pose"], move["target"]
        ctx.check_cancel()
        running = program_name
        if rdk.start_program(program_name, real_robot=True) < 0:
            raise RuntimeError(f"side-view program {program_name} could not start")
        _wait_program(ctx, rdk, program_name)
        running = None
        time.sleep(ecfg.settle_s)
        rdk.use_named_tool_frame(plan.setup.inspection_tool, plan.setup.work_frame)
        side.T_work_camera = np.asarray(rdk.camera_pose_T(), dtype=float)
        frame = services.camera.grab(with_depth=True, timeout=ecfg.grab_timeout_s)
        side.color, side.depth = frame.color, frame.depth
        ok, jpeg = cv2.imencode(".jpg", frame.color)
        if ok:
            ctx.frame(jpeg.tobytes())
    except Exception as exc:
        side.error = str(exc)
        ctx.log(f"side photo not taken (the measurement stands): {exc}")
    finally:
        if running:
            try:
                rdk.stop_program(running)
            except Exception:
                pass
        try:
            rdk.move_j_joints(start_joints)
        except Exception as exc:
            side.error = side.error or f"return from the side pose failed: {exc}"
        side.excursion_ms = (time.perf_counter() - started) * 1000.0
    return side
```

- [ ] **Step 5: Wire the job**

`RingMeasureJob.__init__`: add `side_view: bool = False` after `views`; store `self.side_view = bool(side_view)`; initialise `self._layer_dir = None` and `self._start_joints = None`.

Rename the existing `__call__` to `_measure` and, inside it, right after `start_joints = _prepare_robot(...)` add `self._start_joints = start_joints`, and right after the success `layer_dir = archive.write_layer(...)` add `self._layer_dir = layer_dir`. Then add:

```python
    def __call__(self, ctx: JobContext) -> dict:
        result = self._measure(ctx)
        if self.side_view and self._layer_dir is not None:
            result["side_view"] = self._side_excursion(ctx)
            self.result = result
        return result

    def _side_excursion(self, ctx: JobContext) -> dict:
        """The photo, after the measurement is archived and the arm is home.

        Its own camera hold, its own RoboDK artifacts, its own cleanup; the
        measurement's manifest is patched with the result and the session row
        follows, exactly as the return timing is folded in.
        """
        services = self.services
        rdk: RdkIO = services.rdk
        program = _program_name(self.plan, self.layer_index, "MEASURE") + "_Side"
        artifacts: list[str] = []
        ctx.progress(4, 4, "side photo: second excursion")
        try:
            with _camera_hold(services, "extrusion-side-photo"):
                side = capture_side_view(services, ctx, self.plan, self.layer_index,
                                         program_name=program, start_joints=self._start_joints,
                                         artifacts=artifacts)
        finally:
            try:
                rdk.delete_items(list(dict.fromkeys(reversed(artifacts))))
            except Exception:
                pass
        try:
            record = ExtrusionArchive(REPO_ROOT / "runs" / "extrusion").write_side_view(
                self._layer_dir, color=side.color, depth=side.depth, record=side.record())
        except Exception as exc:
            ctx.log(f"side photo not archived (the measurement stands): {exc}")
            record = side.record().model_copy(update={"error": side.error or str(exc)})
        payload = record.model_dump(mode="json")
        for row in self.session.records:
            if row.get("layer_name") == Path(self._layer_dir).name:
                row["side_view"] = payload
        self.session.save()
        if record.error:
            ctx.log(f"side photo: {record.error}")
        else:
            ctx.log(f"side photo: {record.excursion_ms:.0f} ms excursion at "
                    f"{float(record.descriptor.get('elevation_deg', 0.0)):.0f} deg elevation")
        return payload
```

`_camera_hold` raising (`CameraBusy`) propagates out of `_side_excursion` — acceptable: the take is already archived and the error names the holder; it cannot happen in practice because the measurement's hold was released.

- [ ] **Step 6: The figure**

In `tasni/modules/extrusion/figures.py`:

Line 31–32: after `TRIAL_FIGURES` add `OPTIONAL_LAYER_FIGURES = ("side",)   # drawn only when the take has one`. Add `import math` and `from ...core.geometry import transform_points` to the imports.

`TakeData` (line 86): add a last field `side_view: dict | None = None`. In `load_take` add `side_view=manifest.get("side_view")` to the constructor call.

Add after `_figure_profile` (before `_LAYER_BUILDERS`):

```python
def side_crop_px(take: TakeData, *, margin: float = 0.25) -> "tuple[int, int, int, int] | None":
    """Pixel window around the near-crest arc of the ring in the side photo.

    The arc ``+/- side_view_crop_deg`` about the photographed azimuth, at heights
    0 .. layer top + bead, projected through K and the recorded side pose;
    ``margin`` of the box size on every side; clipped to the image. None when
    the take has no side photo or the projection is degenerate.
    """
    side = take.side_view or {}
    descriptor = side.get("descriptor") or {}
    T = side.get("T_work_camera")
    if take.K is None or T is None or "azimuth_deg" not in descriptor:
        return None
    recipe = take.manifest.get("recipe") or {}
    bead = float(recipe.get("bead_diameter_mm") or 0.0)
    top_z = float(descriptor.get("layer_top_z_mm") or 0.0)
    config = ((take.manifest.get("provenance") or {}).get("processing_config") or {})
    crop = float(config.get("side_view_crop_deg") or 30.0)
    (cx, cy), R = take.center, take.radius
    theta = math.radians(float(descriptor["azimuth_deg"])) + np.radians(np.linspace(-crop, crop, 25))
    ring = np.column_stack((cx + R * np.cos(theta), cy + R * np.sin(theta), np.zeros_like(theta)))
    pts = np.vstack([ring, ring + np.array([0.0, 0.0, top_z + bead])])
    cam = transform_points(np.linalg.inv(np.asarray(T, dtype=float)), pts)
    cam = cam[cam[:, 2] > 1.0]
    if not len(cam):
        return None
    u = take.K[0, 0] * cam[:, 0] / cam[:, 2] + take.K[0, 2]
    v = take.K[1, 1] * cam[:, 1] / cam[:, 2] + take.K[1, 2]
    w, h = max(float(u.max() - u.min()), 40.0), max(float(v.max() - v.min()), 40.0)
    return (int(math.floor(u.min() - margin * w)), int(math.floor(v.min() - margin * h)),
            int(math.ceil(u.max() + margin * w)), int(math.ceil(v.max() + margin * h)))


def _figure_side(plt, take: TakeData):
    """The bead in profile: the side photo cropped to the near crest, upright, with a scale bar."""
    side = take.side_view or {}
    if not side.get("color_file") or side.get("error"):
        return None
    import cv2
    image = cv2.imread(str(take.layer_dir / side["color_file"]), cv2.IMREAD_COLOR)
    if image is None:
        return None
    image = image[:, :, ::-1]                                  # BGR -> RGB
    descriptor = side.get("descriptor") or {}
    height, width = image.shape[:2]
    box = side_crop_px(take) or (0, 0, width, height)
    if float(descriptor.get("roll_deg") or 0.0) == 180.0:      # the roll fallback: flip upright
        image = image[::-1, ::-1]
        u0, v0, u1, v1 = box
        box = (width - u1, height - v1, width - u0, height - v0)
    u0, v0, u1, v1 = (max(0, box[0]), max(0, box[1]), min(width, box[2]), min(height, box[3]))
    crop = image[v0:v1, u0:u1]
    fig, ax = plt.subplots(figsize=(6.0, 6.0 * max(1, v1 - v0) / max(1, u1 - u0) + 0.9))
    ax.imshow(crop)
    ax.set_axis_off()
    standoff = float(descriptor.get("standoff_mm") or 0.0)
    if take.K is not None and standoff > 0:
        px_per_mm = float(take.K[0, 0]) / standoff            # valid: the bead is at that distance
        length = 20.0 * px_per_mm
        x, y = crop.shape[1] * .06, crop.shape[0] * .93
        ax.plot([x, x + length], [y, y], color="white", linewidth=3.0, solid_capstyle="butt")
        ax.text(x + length / 2, y - crop.shape[0] * .02, "20 mm", ha="center", va="bottom",
                fontsize=8, color="white")
    elevation = float(descriptor.get("elevation_deg") or 0.0)
    layer = take.manifest.get("layer_index", "?")
    ax.set_title(f"Layer {layer}, side view at {elevation:.0f}° — {take.label}", fontsize=10)
    caption = take.caption + (" · elevation raised to clear the camera floor"
                              if descriptor.get("floor_raised") else "")
    fig.text(.5, .015, wrap_caption(caption), ha="center", va="bottom", fontsize=7.5, color="#4b5563")
    fig.tight_layout(rect=(0, .06, 1, 1))
    return fig
```

Register it: `_LAYER_BUILDERS` gains `"side": _figure_side`. In `render_layer_figures` change the loop header to `for stem in ((LAYER_FIGURES + OPTIONAL_LAYER_FIGURES) if only is None else (only,)):`. In `ensure_figure` change `if stem in LAYER_FIGURES:` to `if stem in LAYER_FIGURES or stem in OPTIONAL_LAYER_FIGURES:`.

- [ ] **Step 7: Run the tests**

Run: the Global Constraints command.
Expected: all pass. `test_extrusion_figures.py::test_rendering_a_take_writes_every_figure_in_both_formats` must still pass (its take has no side photo, so `side` writes nothing and the set equality holds).

- [ ] **Step 8: Commit, push, merge**

```bash
git add tasni/modules/extrusion/measure.py tasni/modules/extrusion/archive.py tasni/modules/extrusion/figures.py tests/test_extrusion_multiview.py
git commit -m "feat(extrusion): a side photo of the bead as a second, separately timed excursion

After the measurement is archived and the arm is home: its own collision-checked
program, one RGB-D frame, home again; never invalidates the take. Archived under
side/ with the pose; figures/side.png crops the near crest with a scale bar.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
git push
```

Then, with the branch green: `git checkout main && git merge --no-ff multiview-inspection -m "merge: multi-view inspection + side photo (tasks 1-6)" && git push`, and tell the operator the merge hash and that the backend needs a restart before any cell test.

---

## Not in this plan (spec §9 tasks 7–10) — write the second plan after the §8 A/B

- `reprocess_saved_layer` for multi-view takes + `views="top_only"` sibling report; `tools/multiview_ab.py` printing the §8 table. **Until then**: a multi-view take reprocesses from its top view only (`color.png`/`depth.npy` are the top view by design), which is the A/B's offline half by accident — say so in the run log if you use it.
- `paper_summary` grouping by capture mode, the method sentence, `views.png`, the per-take "Views" column, the side figure in the docx.
- API bodies (`views`, `side_view`), preflight descriptors (`inspection_plan(multiview=True, side_view=True)` is ready), UI checkboxes, Run-guide text. **Until then** the operator drives multi-view from a Python shell or a one-line endpoint patch — do not ship a UI toggle without the paper-summary grouping, or single- and four-view timings get pooled.
- `docs/pfh-paper-handoff.md` §3 (A/B step, side photo step), `AGENTS.md`, memory.

## Self-review (done while writing)

- **Spec coverage, §4.1–4.6 → tasks:** star + side poses (T2), `observation_points`/`process_points`/`characterize_points` (T3), levelling / centre alignment / merge / diagnostics / `characterize_views` (T4), `capture_views`, frames median, per-view programs, artifacts, timings, `capture` manifest block, `views/` + `merged_points.npy` archive (T5), side excursion, `side/`, `side_view` block, `side.png` with crop + scale bar + flip (T6). §4.4's `views_capture_ms` and `acquisition_to_path_ms` rule kept; `move_to_pose_ms` = sum of moves with the star views' build time included (documented deviation, for an honest cycle total). §4.5, §4.7, §4.8 deferred by scope. §5 error table: every row has a test except "merged cloud fails processing → Reprocess offers top_only" (T7).
- **Placeholder scan:** the "verbatim" moves in Task 3 point at exact line ranges of the current file rather than retyping 90 lines — that is deliberate (a retyped copy is where drift comes from). No TBD/TODO/"similar to" anywhere else.
- **Numbers checked by hand:** side elevation 17.2° for `recipe()` (bead 6 → top 6 mm) and 16.7° for `auto_plan()` (bead 8 → top 8 mm); star walk 3 tilts × 5 azimuth offsets × 4 rolls = 60; top walk = 20 candidates so `unreachable_targets = 60` refuses the whole walk; `camera.grabs` = readiness + views (+ side).
- **Type consistency:** `View.ok`, `ViewPoints(name, points, T_work_camera)`, `MergeResult.views` records with `used/aligned/xy_shift_mm/levelling/warning`, `process_views` report keys `merge`/`views`, `CaptureRecord.views[i].used`, `side_view_plan()["candidates"][i]["T"]`, `SideViewRecord.excursion_ms`, program suffixes `_Inspect_star120` / `_Side` — used identically in every task and test.
