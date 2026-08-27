# Ring-Stack Measure-Only Experiment — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the operator measure hand-placed dried rings (no printing) through the existing extrusion inspection chain, so the PFH paper gets its deviation, timing and height/bead numbers — proven first on synthetic RGB-D, then on the cell.

**Architecture:** A new `MEASURE_ONLY` mode in `tasni/modules/extrusion/measure.py` reuses the cell-validated `_build_inspection_move` + `process_observation` + `ExtrusionArchive` chain per button press (inspect → capture → process → archive → return), with **no** layer program, valve call, or hardware-I/O gate. Processing gains a per-layer floor (previous ring's measured top), centre-offset / shape-error metrics, a height-profile + bead-width `RingGeometry`, and a two-pass `characterize_ring` that derives the recipe from the physical ring. A numpy z-buffer renderer in `tests/extrusion_synthetic.py` makes every number testable before the robot moves.

**Tech Stack:** Python 3.10 (`py -3.10`), pydantic v2, numpy/scipy/OpenCV, Open3D (processing only), FastAPI + TestClient, React/TypeScript (Vite). RoboDK is never touched by tests (fakes from `tests/test_extrusion_job.py`).

**Spec:** `docs/superpowers/specs/2026-08-27-ring-stack-measure-only-design.md` — read it first; §2 lists the code facts, §3 the operator protocol, §4 the design.

## Global Constraints

- Work in a **git worktree on branch `extrusion-ring-stack`** (`git worktree add ../RoboDkClaude-ring-stack -b extrusion-ring-stack main`). Never commit on `main`. Merge `--no-ff` at the end; push every commit (`git push -u origin extrusion-ring-stack`).
- Python is **`py -3.10`** (no `python` on PATH). Never round-trip a source file through PowerShell `Get-Content`/`Set-Content` (it mojibakes UTF-8) — edit with the Edit tool.
- **Do not run the full pytest suite.** Targeted command, used in every task:
  `py -3.10 -m pytest tests/test_extrusion_measure.py tests/test_extrusion.py tests/test_extrusion_job.py tests/test_extrusion_processing.py -q`
- `ExtrusionConfig` is `extra="forbid"`: every new field needs a default; never remove a field.
- `CylinderPrintJob` and `CylinderDryRunJob` in `service.py` are cell-validated — **do not change their behaviour**. Import their helpers; do not refactor them.
- Tests that call `process_observation`/`characterize_ring` need Open3D: start them with `pytest.importorskip("open3d")`.
- Every task ends with a commit; commit messages end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- The Tasni backend caches modules: before any cell test, **restart Tasni** and check `GET /api/health` → `build.stale == false`.

---

## File map

| File | Responsibility |
|---|---|
| `tests/extrusion_synthetic.py` (new) | numpy z-buffer renderer: rings on a plane → uint16 depth |
| `tests/test_extrusion_measure.py` (new) | every test in this plan |
| `tasni/modules/extrusion/models.py` | `DeviationMetrics` offset/shape fields, `RingGeometry`, `LayerManifest.take/annotation/geometry` |
| `tasni/modules/extrusion/comparison.py` | centre offset + shape error in `compare_circle` |
| `tasni/modules/extrusion/processing.py` | `_filter_deposit`, `_top_surface`, `floor_profile`, `bead_width_profile`, `ring_geometry`, `characterize_ring` |
| `tasni/core/config.py` | 4 new `ExtrusionConfig` fields |
| `tasni/modules/extrusion/archive.py` | trial `mode`/`experiment`, takes, `write_characterization` |
| `tasni/modules/extrusion/measure.py` (new) | `measure_station_requirements`, `MeasureSession`, `RingMeasureJob`, `RingCharacterizeJob`, `paper_summary` |
| `tasni/modules/extrusion/module.py` | `/measure/*` endpoints, `/trials` counts, `/trials/{id}/paper-summary` |
| `tasni/webui/src/pages/Extrusion.tsx` | "Ring stack — measure only" card |
| `docs/extrusion-current-handoff.md` | pointer + cell protocol |

---

### Task 1: Synthetic RGB-D renderer

**Files:**
- Create: `tests/extrusion_synthetic.py`
- Create: `tests/test_extrusion_measure.py`

**Interfaces:**
- Produces: `RingSpec(radius_mm, bead_mm, center_xy_mm, z_base_mm, height_fn)`, `flat(h)`, `wavy(mean, amp, lobes)`, `inspection_camera_T(aim_xyz_mm, standoff_mm=300.0) -> 4x4`, `render_scene(rings, T_work_camera, *, plane_z_mm, plane_center_xy_mm, noise_mm, seed) -> uint16 HxW (mm)`, constants `K_720P`, `SIZE_720P`.

- [ ] **Step 1: Write the renderer**

```python
# tests/extrusion_synthetic.py
"""Synthetic RGB-D rendering of dried rings on a flat work surface.

Point-splat + z-buffer, numpy only: dense surface samples in the WORK frame are
moved into the camera frame with inv(T_work_camera), projected with K, and the
nearest depth per pixel is kept. Depth is uint16 millimetres, 0 = no return —
exactly what ``processing.depth_to_work_points`` expects (depth_scale=1000).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from tasni.core.geometry import transform_points
from tasni.modules.extrusion.inspection import pose_from_aim

# The cell's calibrated 1280x720 colour intrinsics (tasni.config.json, 2026-08).
K_720P = np.array([[889.8742117827221, 0.0, 648.9804252459749],
                   [0.0, 890.8099396048351, 362.00464151468503],
                   [0.0, 0.0, 1.0]])
SIZE_720P = (1280, 720)
# The camera's +X at the parked joints reads [-1, 0, 0] in Tasni Work Frame
# (measured on the cell 2026-08-27; see inspection.py).
CAMERA_X_AT_PARK = [-1.0, 0.0, 0.0]

HeightFn = Callable[[np.ndarray], np.ndarray]


def flat(height_mm: float) -> HeightFn:
    return lambda theta: np.full_like(theta, float(height_mm), dtype=float)


def wavy(mean_mm: float, amplitude_mm: float, lobes: int = 2) -> HeightFn:
    """A 'snake' ring: height oscillates ``lobes`` times around the circumference."""
    return lambda theta: mean_mm + amplitude_mm * np.sin(lobes * theta)


@dataclass
class RingSpec:
    """One dried ring: circular centreline, a flattened rounded cross-section.

    Cross-section at angle theta: half-width ``bead_mm / 2``, height
    ``height_fn(theta)``, profile ``z = h * sin(phi) ** 0.5`` for phi in [0, pi]
    (flatter crest than a semi-ellipse — a slumped bead, which is what the
    upward-normal filter sees on the real material).
    """
    radius_mm: float
    bead_mm: float
    center_xy_mm: tuple[float, float] = (0.0, 0.0)
    z_base_mm: float = 0.0
    height_fn: HeightFn = field(default_factory=lambda: flat(6.0))
    crest_exponent: float = 0.5

    def surface_points(self, step_mm: float = 0.25) -> np.ndarray:
        n_theta = max(64, int(np.ceil(2 * np.pi * (self.radius_mm + self.bead_mm) / step_mm)))
        n_phi = max(8, int(np.ceil(np.pi * self.bead_mm / 2 / step_mm)))
        theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
        phi = np.linspace(0.0, np.pi, n_phi)
        T, P = np.meshgrid(theta, phi, indexing="ij")
        h = self.height_fn(T)
        r = self.radius_mm + (self.bead_mm / 2.0) * np.cos(P)
        x = self.center_xy_mm[0] + r * np.cos(T)
        y = self.center_xy_mm[1] + r * np.sin(T)
        z = self.z_base_mm + h * np.sin(P) ** self.crest_exponent
        return np.column_stack((x.ravel(), y.ravel(), z.ravel()))


def plane_points(*, extent_mm: float = 220.0, step_mm: float = 1.0, z_mm: float = 0.0,
                 center_xy_mm: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    axis = np.arange(-extent_mm, extent_mm + step_mm, step_mm)
    X, Y = np.meshgrid(axis + center_xy_mm[0], axis + center_xy_mm[1], indexing="ij")
    return np.column_stack((X.ravel(), Y.ravel(), np.full(X.size, float(z_mm))))


def inspection_camera_T(aim_xyz_mm, standoff_mm: float = 300.0) -> np.ndarray:
    """Camera pose in the work frame, straight down at ``aim``, as the job derives it."""
    return pose_from_aim(np.asarray(aim_xyz_mm, dtype=float), standoff_mm,
                         reference_x=CAMERA_X_AT_PARK)


def render_depth(points_work: np.ndarray, T_work_camera: np.ndarray, *,
                 K: np.ndarray = K_720P, size_px: tuple[int, int] = SIZE_720P,
                 noise_mm: float = 0.5, seed: int = 0) -> np.ndarray:
    """uint16 depth in mm, z-buffered (nearest surface wins); 0 where nothing was hit."""
    width, height = int(size_px[0]), int(size_px[1])
    cam = transform_points(np.linalg.inv(T_work_camera), points_work)
    cam = cam[cam[:, 2] > 1.0]
    u = np.rint(K[0, 0] * cam[:, 0] / cam[:, 2] + K[0, 2]).astype(int)
    v = np.rint(K[1, 1] * cam[:, 1] / cam[:, 2] + K[1, 2]).astype(int)
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    u, v, z = u[inside], v[inside], cam[inside, 2]
    depth = np.full((height, width), np.inf)
    np.minimum.at(depth, (v, u), z)
    hit = np.isfinite(depth)
    rng = np.random.default_rng(seed)
    depth[hit] += rng.normal(0.0, noise_mm, size=int(hit.sum()))
    depth[~hit] = 0.0
    return np.clip(np.rint(depth), 0, 65535).astype(np.uint16)


def render_scene(rings: list[RingSpec], T_work_camera: np.ndarray, *,
                 plane_z_mm: float = 0.0,
                 plane_center_xy_mm: tuple[float, float] = (0.0, 0.0),
                 noise_mm: float = 0.5, seed: int = 0,
                 K: np.ndarray = K_720P, size_px: tuple[int, int] = SIZE_720P) -> np.ndarray:
    parts = [plane_points(z_mm=plane_z_mm, center_xy_mm=plane_center_xy_mm)]
    parts += [ring.surface_points() for ring in rings]
    return render_depth(np.vstack(parts), T_work_camera, K=K, size_px=size_px,
                        noise_mm=noise_mm, seed=seed)
```

- [ ] **Step 2: Write the failing smoke test**

```python
# tests/test_extrusion_measure.py
"""Ring-stack measure-only experiment: synthetic proof, processing, jobs, API."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import extrusion_synthetic as syn
from tasni.modules.extrusion.processing import depth_to_work_points


def test_renderer_puts_a_ring_where_it_says_at_the_height_it_says():
    center = (200.0, 150.0)
    T = syn.inspection_camera_T([center[0], center[1], 6.0], 300.0)
    depth = syn.render_scene([syn.RingSpec(60.0, 8.0, center, height_fn=syn.flat(6.0))], T,
                             plane_center_xy_mm=center, noise_mm=0.0)
    assert depth.dtype == np.uint16 and depth.shape == (720, 1280)
    points, raw = depth_to_work_points(depth, syn.K_720P, T)
    assert raw > 100_000                                  # plane + ring both rendered
    ring = points[points[:, 2] > 3.0]
    radii = np.linalg.norm(ring[:, :2] - np.array(center), axis=1)
    assert 55.0 < radii.min() and radii.max() < 65.0     # 60 +/- bead/2 (+ rounding)
    assert 5.0 < ring[:, 2].max() < 7.0                   # crest at 6 mm
    plane = points[points[:, 2] <= 1.0]
    assert len(plane) > 50_000
```

- [ ] **Step 3: Run it, expect a pass** (this task's test validates the helper itself)

Run: `py -3.10 -m pytest tests/test_extrusion_measure.py -q`
Expected: 1 passed. If the ring radii assertion fails, check that `inspection_camera_T` looks *down* (T[:3, 2] ≈ [0, 0, −1]).

- [ ] **Step 4: Commit**

```bash
git add tests/extrusion_synthetic.py tests/test_extrusion_measure.py
git commit -m "tests: synthetic RGB-D renderer for dried rings on the work surface"
```

---

### Task 2: Centre offset and shape error in `compare_circle`

**Files:**
- Modify: `tasni/modules/extrusion/models.py:79-89` (`DeviationMetrics`)
- Modify: `tasni/modules/extrusion/comparison.py:28-60`
- Test: `tests/test_extrusion_measure.py`

**Interfaces:**
- Produces: `DeviationMetrics.center_offset_mm: tuple[float, float]`, `.center_offset_norm_mm`, `.shape_rms_mm`, `.shape_max_mm` (all defaulted so old manifests still validate).

- [ ] **Step 1: Failing test**

```python
from tasni.modules.extrusion.comparison import compare_circle


def test_shifted_circle_reports_its_offset_and_zero_shape_error():
    theta = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    shifted = np.column_stack((40 * np.cos(theta) + 10, 40 * np.sin(theta), np.full(360, 5.0)))
    m = compare_circle(shifted, 40.0, nominal_center_mm=(0.0, 0.0))
    assert m.center_offset_mm == pytest.approx((10.0, 0.0), abs=1e-6)
    assert m.center_offset_norm_mm == pytest.approx(10.0, abs=1e-6)
    assert m.shape_rms_mm < 1e-6 and m.shape_max_mm < 1e-6
    # Deviation is still measured from the NOMINAL centre (the paper's number).
    assert m.mean_absolute_mm == pytest.approx(6.35, abs=0.05)
    assert m.rms_mm == pytest.approx(7.06, abs=0.05)
    assert m.maximum_mm == pytest.approx(10.0, abs=0.05)


def test_old_metrics_payload_without_offset_fields_still_validates():
    from tasni.modules.extrusion.models import DeviationMetrics
    old = DeviationMetrics(mean_absolute_mm=1, rms_mm=1, maximum_mm=1,
                           measured_center_mm=(0, 0), measured_radius_mm=40,
                           path_completeness=1, maximum_angular_gap_deg=2, valid=True)
    assert old.center_offset_norm_mm == 0.0 and old.shape_rms_mm == 0.0
```

- [ ] **Step 2: Run, expect failure** — `AttributeError: ... center_offset_mm`.

- [ ] **Step 3: Implement**

In `models.py`, extend `DeviationMetrics` (after `maximum_angular_gap_deg`):

```python
    # Fitted-circle centre minus the NOMINAL centre: the direct readout of a
    # bodily displacement of the ring (the paper's introduced-offset check).
    center_offset_mm: tuple[float, float] = (0.0, 0.0)
    center_offset_norm_mm: float = 0.0
    # Radial scatter about the FITTED circle: "ring is not round", separated
    # from "ring placed wrong".
    shape_rms_mm: float = 0.0
    shape_max_mm: float = 0.0
```

In `comparison.py:compare_circle`, after `deviation = ...`:

```python
    offset = center - nominal_center
    shape = np.linalg.norm(pts[:, :2] - center, axis=1) - measured_radius
```

and in the returned `DeviationMetrics(...)` add:

```python
        center_offset_mm=(float(offset[0]), float(offset[1])),
        center_offset_norm_mm=float(np.linalg.norm(offset)),
        shape_rms_mm=float(np.sqrt(np.mean(shape * shape))),
        shape_max_mm=float(np.max(np.abs(shape))),
```

- [ ] **Step 4: Run targeted tests, expect all green** (including `tests/test_extrusion.py::test_circle_metrics_and_bounded_correction`).

- [ ] **Step 5: Commit** — `feat(extrusion): centre offset and shape error in circle metrics`

---

### Task 3: Extract the filter chain and prove `process_observation` end to end

**Files:**
- Modify: `tasni/modules/extrusion/processing.py:222-285`
- Test: `tests/test_extrusion_measure.py`

**Interfaces:**
- Produces: `_filter_deposit(points, config, counts) -> np.ndarray` (voxel → statistical → radius → largest DBSCAN cluster), `_top_surface(points, config, counts) -> np.ndarray` (upward normals → largest cluster). Behaviour of `process_observation` unchanged.

- [ ] **Step 1: Failing e2e tests** (first end-to-end test of the pipeline)

```python
from tasni.core.config import ExtrusionConfig
from tasni.modules.extrusion.inspection import aim_point_mm
from tasni.modules.extrusion.models import CylinderRecipe, CylinderSetup
from tasni.modules.extrusion.processing import process_observation
from tasni.modules.extrusion.toolpath import generate_cylinder_plan

CENTER = (200.0, 150.0)


def scene_plan(*, radius=60.0, bead=8.0, layers=1, layer_height=6.0, center=CENTER):
    recipe = CylinderRecipe(radius_mm=radius, layer_count=layers, layer_height_mm=layer_height,
                            bead_diameter_mm=bead, robot_speed_mm_s=75, extrusion_rate_pct=0,
                            points_per_circle=180)
    setup = CylinderSetup(print_tool="LongCalibTool", work_frame="Tasni Work Frame",
                          inspection_tool="Realsense", inspection_auto=True,
                          center_x_mm=center[0], center_y_mm=center[1])
    return generate_cylinder_plan(recipe, setup)


def observe(plan, layer_index, rings, *, config=None, floor_profile=None, seed=0):
    """Render the rings from the derived inspection pose and process that frame."""
    layer = plan.layers[layer_index - 1]
    T = syn.inspection_camera_T(aim_point_mm(plan.recipe, plan.setup, layer_index), 300.0)
    depth = syn.render_scene(rings, T, plane_center_xy_mm=(plan.setup.center_x_mm,
                                                           plan.setup.center_y_mm), seed=seed)
    color = np.zeros((syn.SIZE_720P[1], syn.SIZE_720P[0], 3), np.uint8)
    kwargs = {} if floor_profile is None else {"floor_profile": floor_profile}
    return process_observation(color=color, depth=depth, T_work_camera=T, K=syn.K_720P,
                               plan=plan, layer=layer, config=config or ExtrusionConfig(),
                               **kwargs)


def test_true_ring_measures_as_true():
    pytest.importorskip("open3d")
    plan = scene_plan()
    out = observe(plan, 1, [syn.RingSpec(60.0, 8.0, CENTER, height_fn=syn.flat(6.0))])
    m = out.metrics
    assert m.valid, m.warnings
    assert m.mean_absolute_mm < 1.0 and m.rms_mm < 1.0
    assert abs(m.measured_radius_mm - 60.0) < 1.0
    assert m.center_offset_norm_mm < 1.0
    assert m.path_completeness >= 0.95
    assert out.report["timings_ms"]["total_ms"] > 0


def test_ring_shifted_10mm_reports_the_shift():
    pytest.importorskip("open3d")
    plan = scene_plan()
    out = observe(plan, 1, [syn.RingSpec(60.0, 8.0, (CENTER[0] + 10.0, CENTER[1]),
                                          height_fn=syn.flat(6.0))])
    m = out.metrics
    assert m.valid, m.warnings
    assert m.center_offset_mm[0] == pytest.approx(10.0, abs=1.0)
    assert abs(m.center_offset_mm[1]) < 1.0
    assert m.maximum_mm == pytest.approx(10.0, abs=1.5)
    assert m.mean_absolute_mm == pytest.approx(6.36, abs=1.0)
    assert m.rms_mm == pytest.approx(7.06, abs=1.0)
    assert m.shape_rms_mm < 1.0
```

- [ ] **Step 2: Run** `py -3.10 -m pytest tests/test_extrusion_measure.py -q -k "true_ring or shifted"`.
  These may already pass on the untouched pipeline — that is the point: they are the regression net for the refactor. If they FAIL on the untouched code, do not touch `processing.py` yet: print `out.report["counts"]` and `["branch_guard_attempts"]` (or the exception), and tune **only** through `ExtrusionConfig(...)` overrides passed by the test's `config=` argument (candidates: `upwards_normal_z` 0.92→0.85, `voxel_size_m` 0.002→0.0015). If a default must change for the synthetic slumped-bead ring to pass, change it in `config.py` with a comment citing this test, and say so in the commit message — the real ring is flatter-topped than the render, so a default that fails here will fail on the cell.

- [ ] **Step 3: Refactor** — cut the two blocks out of `process_observation` verbatim into:

```python
def _filter_deposit(points: np.ndarray, config, counts: dict) -> np.ndarray:
    """Voxel -> statistical -> radius outliers -> largest DBSCAN cluster (Open3D)."""
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("Open3D is required for extrusion processing; install tasni[scan]") from exc
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    cloud = cloud.voxel_down_sample(config.voxel_size_m * 1000.0)
    counts["after_voxel"] = len(cloud.points)
    cloud, _ = cloud.remove_statistical_outlier(
        nb_neighbors=config.statistical_neighbors, std_ratio=config.statistical_std_ratio)
    counts["after_statistical"] = len(cloud.points)
    cloud, _ = cloud.remove_radius_outlier(
        nb_points=config.radius_neighbors, radius=config.radius_m * 1000.0)
    counts["after_radius"] = len(cloud.points)
    if len(cloud.points) < config.cluster_min_points:
        raise RuntimeError("deposited cloud was removed by outlier filtering")
    labels = np.asarray(cloud.cluster_dbscan(
        eps=config.cluster_eps_m * 1000.0, min_points=config.cluster_min_points,
        print_progress=False))
    points = _largest_label(np.asarray(cloud.points), labels)
    counts["after_largest_cluster"] = len(points)
    return points


def _top_surface(points: np.ndarray, config, counts: dict) -> np.ndarray:
    """Upward-facing points of the deposit, then the largest cluster of those."""
    import open3d as o3d
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=100.0, max_nn=30))
    cloud.orient_normals_to_align_with_direction(np.array([0.0, 0.0, 1.0]))
    normals = np.asarray(cloud.normals)
    points = points[normals[:, 2] > config.upwards_normal_z]
    counts["after_upward_normals"] = len(points)
    if len(points) < config.cluster_min_points:
        raise RuntimeError("too few upward-facing deposited points")
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    labels = np.asarray(cloud.cluster_dbscan(
        eps=config.normal_cluster_eps_m * 1000.0,
        min_points=config.cluster_min_points, print_progress=False))
    points = _largest_label(points, labels)
    counts["after_normal_cluster"] = len(points)
    return points
```

and in `process_observation` replace the block between `mark = time.perf_counter()` (the one before `import open3d`) and `timings["filter_ms"] = ...` with:

```python
    mark = time.perf_counter()
    deposit = _filter_deposit(points, config, counts)
    points = _top_surface(deposit, config, counts)
    timings["filter_ms"] = (time.perf_counter() - mark) * 1000
```

(`deposit` is used by Task 5.)

- [ ] **Step 4: Run the targeted suite** — all green, same numbers as Step 2.

- [ ] **Step 5: Commit** — `refactor(extrusion): name the deposit filter and top-surface stages; first e2e processing tests`

---

### Task 4: Per-layer floor from the previous ring's measured top

**Files:**
- Modify: `tasni/core/config.py` (`ExtrusionConfig`, after `measured_spline_points`)
- Modify: `tasni/modules/extrusion/processing.py:process_observation`
- Test: `tests/test_extrusion_measure.py`

**Interfaces:**
- Produces: `process_observation(..., floor_profile: np.ndarray | None = None)`; `report["floor"] = {"source", "margin_mm", "mean_mm"}`; config `layer_floor_margin_mm`.

- [ ] **Step 1: Failing test**

```python
def test_floor_from_previous_layer_keeps_the_ring_below_out_of_the_measurement():
    pytest.importorskip("open3d")
    plan = scene_plan(layers=2, layer_height=6.0)
    ring1 = syn.RingSpec(60.0, 8.0, CENTER, height_fn=syn.flat(6.0))
    first = observe(plan, 1, [ring1])
    assert first.metrics.valid and first.report["floor"]["source"] == "build_plane"

    ring2 = syn.RingSpec(60.0, 8.0, (CENTER[0] + 10.0, CENTER[1]), z_base_mm=6.0,
                         height_fn=syn.flat(6.0))
    floored = observe(plan, 2, [ring1, ring2], floor_profile=first.measured_xyz)
    assert floored.metrics.valid, floored.metrics.warnings
    assert floored.report["floor"]["source"] == "previous_layer_measured"
    assert floored.metrics.center_offset_norm_mm == pytest.approx(10.0, abs=1.5)

    # Without the floor the exposed crescent of ring 1 contaminates the answer:
    # either the branch guard rejects it, or the offset is pulled well under 10.
    try:
        blended = observe(plan, 2, [ring1, ring2])
    except RuntimeError:
        return
    assert (abs(blended.metrics.center_offset_norm_mm - 10.0)
            > abs(floored.metrics.center_offset_norm_mm - 10.0) + 1.0)
```

- [ ] **Step 2: Run, expect failure** — `TypeError: unexpected keyword argument 'floor_profile'`.

- [ ] **Step 3: Implement**

`config.py`, `ExtrusionConfig`, after `measured_spline_points`:

```python
    # -- ring-stack measure-only experiment (modules/extrusion/measure.py) ----
    # Layer N keeps only points above the PREVIOUS layer's measured top at the
    # nearest XY sample plus this margin, so a displaced ring cannot drag the
    # exposed crescent of the ring beneath it into the skeleton.
    layer_floor_margin_mm: float = Field(default=2.0, ge=0, le=20)
    characterize_search_radius_mm: float = Field(default=150.0, gt=0, le=1000)
    characterize_max_height_mm: float = Field(default=40.0, gt=0, le=200)
    bead_width_bins: int = Field(default=36, ge=4, le=360)
```

`processing.py`: signature `def process_observation(*, color, depth, T_work_camera, K, plan, layer, config, floor_profile=None) -> ProcessingResult:` and replace the ROI block with:

```python
    setup, recipe = plan.setup, plan.recipe
    radius = np.linalg.norm(points[:, :2] - np.array([setup.center_x_mm, setup.center_y_mm]), axis=1)
    max_z = layer.nominal_z_mm + recipe.bead_diameter_mm / 2 + config.deposit_height_margin_mm
    # The selected work frame defines the build plane at Z=0, so deterministic
    # height subtraction is more reproducible than fitting a new plane per frame.
    min_z = max(config.deposit_min_height_mm,
                config.plane_distance_threshold_m * 1000.0)
    roi = ((points[:, 2] >= min_z) & (points[:, 2] <= max_z) &
           (radius >= recipe.radius_mm - config.radial_roi_margin_mm) &
           (radius <= recipe.radius_mm + config.radial_roi_margin_mm))
    points = points[roi]
    floor = {"source": "build_plane", "margin_mm": 0.0, "mean_mm": float(min_z)}
    if floor_profile is not None and len(points):
        profile = np.asarray(floor_profile, dtype=float).reshape(-1, 3)
        _, nearest = cKDTree(profile[:, :2]).query(points[:, :2])
        local = profile[nearest, 2] + config.layer_floor_margin_mm
        points = points[points[:, 2] >= local]
        floor = {"source": "previous_layer_measured",
                 "margin_mm": float(config.layer_floor_margin_mm),
                 "mean_mm": float(local.mean())}
    counts["after_work_roi"] = len(points)
```

and add `"floor": floor,` to the `report` dict at the end.

- [ ] **Step 4: Run targeted tests — green.** If the floored offset lands short of 10 by more than 1.5, print `floored.report["counts"]`; the usual cause is the floor being too low (raise `layer_floor_margin_mm` in the test's config to 3.0 and, if that is what it takes, in the default with a comment).

- [ ] **Step 5: Commit** — `feat(extrusion): per-layer floor from the previous ring's measured top`

---

### Task 5: `RingGeometry` — height profile and bead width

**Files:**
- Modify: `tasni/modules/extrusion/models.py` (new `RingGeometry`; `LayerManifest.geometry`)
- Modify: `tasni/modules/extrusion/processing.py` (`bead_width_profile`, `ring_geometry`, `ProcessingResult.geometry`)
- Test: `tests/test_extrusion_measure.py`

**Interfaces:**
- Produces: `RingGeometry` (pydantic), `bead_width_profile(cluster_xyz, center_xy, *, bins, low_pct=2.5, high_pct=97.5) -> dict`, `ring_geometry(measured_xyz, cluster_xyz, center_xy, *, floor_profile, build_plane_z_mm, bins) -> RingGeometry`, `ProcessingResult.geometry: RingGeometry | None`.

- [ ] **Step 1: Failing tests**

```python
def test_wavy_ring_height_profile_is_measured():
    pytest.importorskip("open3d")
    plan = scene_plan(layer_height=7.5)
    out = observe(plan, 1, [syn.RingSpec(60.0, 8.0, CENTER, height_fn=syn.wavy(7.5, 2.5, lobes=2))])
    g = out.geometry
    assert g is not None and g.height_reference == "build_plane"
    assert g.top_z_min_mm == pytest.approx(5.0, abs=1.5)
    assert g.top_z_max_mm == pytest.approx(10.0, abs=1.5)
    assert g.top_z_std_mm > 1.0
    assert g.height_mean_mm == pytest.approx(7.5, abs=1.0)


def test_bead_width_is_the_rings_radial_footprint():
    pytest.importorskip("open3d")
    plan = scene_plan()
    out = observe(plan, 1, [syn.RingSpec(60.0, 8.0, CENTER, height_fn=syn.flat(6.0))])
    g = out.geometry
    assert g.bead_width_mean_mm == pytest.approx(8.0, rel=0.25)
    assert g.bead_width_bins == 36


def test_bead_width_profile_on_an_ideal_annulus():
    from tasni.modules.extrusion.processing import bead_width_profile
    rng = np.random.default_rng(1)
    theta = rng.uniform(0, 2 * np.pi, 20000)
    r = rng.uniform(36.0, 44.0, 20000)                # annulus 40 +/- 4 -> width 8
    pts = np.column_stack((r * np.cos(theta), r * np.sin(theta), np.zeros(20000)))
    w = bead_width_profile(pts, (0.0, 0.0), bins=36)
    assert w["bins_with_data"] == 36
    assert w["mean_mm"] == pytest.approx(8.0, abs=0.6)   # p97.5 - p2.5 of a uniform 8 mm band
```

- [ ] **Step 2: Run, expect failure** — `AttributeError: 'ProcessingResult' object has no attribute 'geometry'` / ImportError.

- [ ] **Step 3: Implement**

`models.py` (after `DeviationMetrics`):

```python
class RingGeometry(_Record):
    """What the ring looks like, not how far it is from nominal.

    ``top_z_*`` are the measured centreline heights in the work frame;
    ``height_*`` subtract the reference surface — the build plane for the first
    ring, or the previous ring's measured top at the nearest sample.
    ``bead_width_*`` is the radial footprint of the deposit cloud per angular bin
    (percentile extent, so the bead's flanks count, outliers do not).
    """
    top_z_mean_mm: float
    top_z_min_mm: float
    top_z_max_mm: float
    top_z_std_mm: float
    height_mean_mm: float
    height_min_mm: float
    height_max_mm: float
    height_reference: str
    bead_width_mean_mm: float
    bead_width_min_mm: float
    bead_width_max_mm: float
    bead_width_bins: int
```

and on `LayerManifest` add `geometry: RingGeometry | None = None` (after `metrics`).

`processing.py`:

```python
from .models import CylinderPlan, DeviationMetrics, LayerPath, RingGeometry
...
@dataclass
class ProcessingResult:
    ...
    filtered_xyz: np.ndarray | None = None
    geometry: RingGeometry | None = None


def bead_width_profile(cluster_xyz, center_xy, *, bins: int = 36,
                       low_pct: float = 2.5, high_pct: float = 97.5) -> dict:
    """Radial extent of the deposit per angular bin, about ``center_xy``."""
    pts = np.asarray(cluster_xyz, dtype=float)
    rel = pts[:, :2] - np.asarray(center_xy, dtype=float)
    radii = np.linalg.norm(rel, axis=1)
    angle = np.mod(np.arctan2(rel[:, 1], rel[:, 0]), 2 * math.pi)
    edges = np.linspace(0.0, 2 * math.pi, bins + 1)
    which = np.clip(np.digitize(angle, edges) - 1, 0, bins - 1)
    widths = np.full(bins, np.nan)
    for index in range(bins):
        r = radii[which == index]
        if len(r) >= 8:
            widths[index] = np.percentile(r, high_pct) - np.percentile(r, low_pct)
    valid = widths[np.isfinite(widths)]
    if not len(valid):
        raise RuntimeError("bead width: no angular bin had enough deposit points")
    return {"bins": bins, "bins_with_data": int(len(valid)),
            "per_bin_mm": [None if not np.isfinite(w) else float(w) for w in widths],
            "mean_mm": float(valid.mean()), "min_mm": float(valid.min()),
            "max_mm": float(valid.max())}


def ring_geometry(measured_xyz, cluster_xyz, center_xy, *, floor_profile=None,
                  build_plane_z_mm: float = 0.0, bins: int = 36) -> RingGeometry:
    measured = np.asarray(measured_xyz, dtype=float)
    top = measured[:, 2]
    if floor_profile is None:
        reference = np.full(len(top), float(build_plane_z_mm))
        reference_name = "build_plane"
    else:
        profile = np.asarray(floor_profile, dtype=float).reshape(-1, 3)
        _, nearest = cKDTree(profile[:, :2]).query(measured[:, :2])
        reference = profile[nearest, 2]
        reference_name = "previous_layer_measured"
    height = top - reference
    width = bead_width_profile(cluster_xyz, center_xy, bins=bins)
    return RingGeometry(
        top_z_mean_mm=float(top.mean()), top_z_min_mm=float(top.min()),
        top_z_max_mm=float(top.max()), top_z_std_mm=float(top.std()),
        height_mean_mm=float(height.mean()), height_min_mm=float(height.min()),
        height_max_mm=float(height.max()), height_reference=reference_name,
        bead_width_mean_mm=width["mean_mm"], bead_width_min_mm=width["min_mm"],
        bead_width_max_mm=width["max_mm"], bead_width_bins=bins)
```

In `process_observation`, after `metrics = compare_circle(...)`:

```python
    geometry = ring_geometry(measured, deposit, metrics.measured_center_mm,
                             floor_profile=floor_profile,
                             build_plane_z_mm=setup.build_plane_z_mm,
                             bins=config.bead_width_bins)
```

add `"geometry": geometry.model_dump(mode="json"),` to `report`, and pass `geometry=geometry` to the `ProcessingResult(...)` constructor.

- [ ] **Step 4: Run targeted tests — green.** `tests/test_extrusion_job.py::fake_processing` builds `ProcessingResult` positionally without `geometry` — it still works because the field defaults to `None`.

- [ ] **Step 5: Commit** — `feat(extrusion): ring geometry — height profile and bead width per measured layer`

---

### Task 6: `characterize_ring` — recipe from the physical ring

**Files:**
- Modify: `tasni/modules/extrusion/processing.py`
- Test: `tests/test_extrusion_measure.py`

**Interfaces:**
- Produces: `CharacterizationResult` dataclass and `characterize_ring(*, color, depth, T_work_camera, K, search_center_mm, work_frame, config, inspection_tool="Realsense", print_tool="LongCalibTool") -> CharacterizationResult`.

- [ ] **Step 1: Failing test**

```python
from tasni.modules.extrusion.processing import characterize_ring


def test_characterize_recovers_a_ring_the_recipe_got_wrong():
    pytest.importorskip("open3d")
    # The recipe/plan says 75 mm radius, 6 mm bead. The physical ring is 60 / 8,
    # 6 mm tall, and sits 15 mm off the table centre.
    plan = scene_plan(radius=75.0, bead=6.0, layer_height=5.0)
    true_center = (CENTER[0] + 15.0, CENTER[1] - 10.0)
    T = syn.inspection_camera_T(aim_point_mm(plan.recipe, plan.setup, 1), 300.0)
    depth = syn.render_scene([syn.RingSpec(60.0, 8.0, true_center, height_fn=syn.flat(6.0))], T,
                             plane_center_xy_mm=CENTER)
    color = np.zeros((720, 1280, 3), np.uint8)
    found = characterize_ring(color=color, depth=depth, T_work_camera=T, K=syn.K_720P,
                              search_center_mm=CENTER, work_frame="Tasni Work Frame",
                              config=ExtrusionConfig())
    assert found.radius_mm == pytest.approx(60.0, abs=1.0)
    assert found.center_mm[0] == pytest.approx(true_center[0], abs=1.0)
    assert found.center_mm[1] == pytest.approx(true_center[1], abs=1.0)
    assert found.bead_width_mm == pytest.approx(8.0, abs=2.0)
    assert found.top_z_mean_mm == pytest.approx(6.0, abs=1.5)
    assert found.report["coarse"]["radius_mm"] == pytest.approx(60.0, abs=3.0)
    assert found.measured_xyz.shape[1] == 3
```

- [ ] **Step 2: Run, expect ImportError.**

- [ ] **Step 3: Implement** (append to `processing.py`; add `from .toolpath import generate_cylinder_plan` and `from .models import CylinderRecipe, CylinderSetup` to the imports; `fit_circle_xy` from `.comparison`)

```python
@dataclass
class CharacterizationResult:
    radius_mm: float
    center_mm: tuple[float, float]
    bead_width_mm: float
    bead_width_min_mm: float
    bead_width_max_mm: float
    top_z_mean_mm: float
    top_z_min_mm: float
    top_z_max_mm: float
    measured_xyz: np.ndarray
    segmentation: np.ndarray
    skeleton: np.ndarray
    comparison: np.ndarray
    report: dict

    def summary(self) -> dict:
        return {k: getattr(self, k) for k in (
            "radius_mm", "center_mm", "bead_width_mm", "bead_width_min_mm",
            "bead_width_max_mm", "top_z_mean_mm", "top_z_min_mm", "top_z_max_mm")}


def characterize_ring(*, color: np.ndarray, depth: np.ndarray, T_work_camera: np.ndarray,
                      K: np.ndarray, search_center_mm, work_frame: str, config,
                      inspection_tool: str = "Realsense",
                      print_tool: str = "LongCalibTool") -> CharacterizationResult:
    """Measure a ring with NO recipe assumption: coarse fit, then the normal pipeline.

    Pass 1 takes everything above the build plane inside a search cylinder around
    ``search_center_mm``, filters it like a deposit, and fits a circle to get a
    coarse centre/radius/bead. Pass 2 hands those to ``process_observation`` as a
    throwaway recipe so the refined centreline, radius and height profile come out
    of the same code the layer measurements use.
    """
    started = time.perf_counter()
    counts: dict[str, int] = {}
    points, counts["raw_depth_pixels"] = depth_to_work_points(depth, K, T_work_camera)
    center = np.asarray(search_center_mm, dtype=float)
    min_z = max(config.deposit_min_height_mm, config.plane_distance_threshold_m * 1000.0)
    radial = np.linalg.norm(points[:, :2] - center, axis=1)
    roi = ((points[:, 2] >= min_z) & (points[:, 2] <= config.characterize_max_height_mm)
           & (radial <= config.characterize_search_radius_mm))
    points = points[roi]
    counts["after_search_roi"] = len(points)
    if len(points) < config.cluster_min_points:
        raise RuntimeError("no deposited geometry inside the characterization search region")
    deposit = _filter_deposit(points, config, counts)
    coarse_center, coarse_radius = fit_circle_xy(deposit)
    width = bead_width_profile(deposit, coarse_center, bins=config.bead_width_bins)
    top = _top_surface(deposit, config, counts)
    coarse_height = float(np.percentile(top[:, 2], 90))
    coarse = {"center_mm": [float(coarse_center[0]), float(coarse_center[1])],
              "radius_mm": float(coarse_radius), "bead_width_mm": width["mean_mm"],
              "height_mm": coarse_height, "time_ms": (time.perf_counter() - started) * 1000}

    recipe = CylinderRecipe(
        radius_mm=float(np.clip(coarse_radius, 5.0, 500.0)), layer_count=1,
        layer_height_mm=float(np.clip(coarse_height, 0.5, 50.0)),
        bead_diameter_mm=float(np.clip(width["mean_mm"], 0.5, 50.0)),
        robot_speed_mm_s=75.0, extrusion_rate_pct=0.0,
        points_per_circle=config.measured_spline_points)
    setup = CylinderSetup(
        print_tool=print_tool, work_frame=work_frame, inspection_tool=inspection_tool,
        inspection_auto=True, center_x_mm=float(coarse_center[0]),
        center_y_mm=float(coarse_center[1]))
    plan = generate_cylinder_plan(recipe, setup)
    refined = process_observation(color=color, depth=depth, T_work_camera=T_work_camera,
                                  K=K, plan=plan, layer=plan.layers[0], config=config)
    geometry = refined.geometry
    report = {**refined.report, "coarse": coarse, "counts_coarse": counts,
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

- [ ] **Step 4: Run targeted tests — green.** If pass 2 raises "not enough deposited-geometry points inside the configured work ROI", the coarse radius is off by > 30 mm: assert on `report["coarse"]` first and inspect whether `_top_surface` kept the plane (raise `min_z`).

- [ ] **Step 5: Commit** — `feat(extrusion): characterize a ring with no recipe assumption (two-pass)`

---

### Task 7: Archive — trial mode, takes, characterization directory

**Files:**
- Modify: `tasni/modules/extrusion/models.py` (`LayerManifest.take`, `.annotation`)
- Modify: `tasni/modules/extrusion/archive.py`
- Modify: `tasni/modules/extrusion/service.py:reprocess_saved_layer` (take-aware, default 1)
- Test: `tests/test_extrusion_measure.py`

**Interfaces:**
- Produces: `ExtrusionArchive.create_trial(trial_id, plan, *, provenance=None, mode="LIVE_PRINT", experiment=None)`, `layer_dir(trial_id, layer_index, *, take=1, require=True)` → `layer-NNN` / `layer-NNN-takeMM`, `write_characterization(trial_id, index, *, color, depth, measured_xyz, derived_images, report) -> Path` → `characterize-NN/`; `LayerManifest.take: int = 1`, `.annotation: dict = {}`; `reprocess_saved_layer(root, trial_id, layer_index, take=1)`.

- [ ] **Step 1: Failing tests**

```python
from tasni.modules.extrusion.archive import ExtrusionArchive
from tasni.modules.extrusion.models import LayerManifest


def test_archive_keeps_every_take_of_a_layer_and_records_the_mode(tmp_path):
    plan = scene_plan()
    archive = ExtrusionArchive(tmp_path)
    trial = archive.create_trial("t1", plan, mode="MEASURE_ONLY",
                                 experiment={"note": "dried rings, hand-placed"})
    data = json.loads((trial / "trial.json").read_text())
    assert data["mode"] == "MEASURE_ONLY" and data["experiment"]["note"].startswith("dried")
    nominal = np.zeros((4, 3))
    for take in (1, 2):
        manifest = LayerManifest(trial_id="t1", layer_index=2, take=take, mode="MEASURE_ONLY",
                                 recipe=plan.recipe, toolpath_fingerprint=plan.fingerprint,
                                 annotation={"introduced_offset_mm": [10, 0]})
        archive.write_layer(manifest, nominal_xyz=nominal, commanded_xyz=nominal)
    assert (tmp_path / "t1" / "layer-002" / "manifest.json").is_file()
    assert (tmp_path / "t1" / "layer-002-take02" / "manifest.json").is_file()
    assert archive.layer_dir("t1", 2, take=2).name == "layer-002-take02"
    loaded = json.loads((tmp_path / "t1" / "layer-002-take02" / "manifest.json").read_text())
    assert loaded["take"] == 2 and loaded["annotation"]["introduced_offset_mm"] == [10, 0]


def test_archive_writes_a_characterization_directory(tmp_path):
    plan = scene_plan()
    archive = ExtrusionArchive(tmp_path)
    archive.create_trial("t1", plan, mode="MEASURE_ONLY")
    out = archive.write_characterization(
        "t1", 1, color=np.zeros((4, 4, 3), np.uint8), depth=np.zeros((4, 4), np.uint16),
        measured_xyz=np.zeros((5, 3)),
        derived_images={"comparison.png": np.zeros((4, 4, 3), np.uint8)},
        report={"radius_mm": 60.0})
    assert out.name == "characterize-01"
    for name in ("color.png", "depth.npy", "measured_path.json", "comparison.png", "report.json"):
        assert (out / name).is_file(), name
```

- [ ] **Step 2: Run, expect failure** — `TypeError: create_trial() got an unexpected keyword argument 'mode'`.

- [ ] **Step 3: Implement**

`models.py`, `LayerManifest`: after `mode`, add

```python
    take: int = Field(default=1, ge=1)
    # Operator ground truth for the measure-only experiment, e.g.
    # {"introduced_offset_mm": [10.0, 0.0], "note": "ring 3 shifted +X"}.
    annotation: dict[str, Any] = Field(default_factory=dict)
```

`archive.py`:

```python
    @staticmethod
    def _layer_name(layer_index: int, take: int) -> str:
        if layer_index < 1 or take < 1:
            raise ValueError("layer index and take must be positive")
        return (f"layer-{layer_index:03d}" if take == 1
                else f"layer-{layer_index:03d}-take{take:02d}")

    def create_trial(self, trial_id: str, plan: CylinderPlan, *, provenance: dict | None = None,
                     mode: str = "LIVE_PRINT", experiment: dict | None = None) -> Path:
        ...  # add to payload:
            "mode": mode, "experiment": experiment or {},

    def layer_dir(self, trial_id: str, layer_index: int, *, take: int = 1,
                  require: bool = True) -> Path:
        layer = self.root / _segment(trial_id, "trial id") / self._layer_name(layer_index, take)
        if require and not (layer / "manifest.json").is_file():
            raise FileNotFoundError(f"archived layer does not exist: {trial_id}/{layer.name}")
        return layer
```

In `write_layer` replace `layer = trial / f"layer-{manifest.layer_index:03d}"` with `layer = trial / self._layer_name(manifest.layer_index, manifest.take)`; in `rewrite_processing` use `self.layer_dir(manifest.trial_id, manifest.layer_index, take=manifest.take)`. Add:

```python
    def write_characterization(self, trial_id: str, index: int, *, color, depth, measured_xyz,
                               derived_images: dict[str, np.ndarray], report: dict) -> Path:
        trial = self.root / _segment(trial_id, "trial id")
        if not (trial / "trial.json").is_file():
            raise FileNotFoundError(f"trial does not exist: {trial_id}")
        out = trial / f"characterize-{index:02d}"
        out.mkdir(parents=False, exist_ok=False)
        import cv2
        if not cv2.imwrite(str(out / "color.png"), np.asarray(color)):
            raise OSError("failed to write color.png")
        np.save(out / "depth.npy", np.asarray(depth))
        self._json_path(out / "measured_path.json", measured_xyz)
        allowed = {"segmentation.png", "skeleton.png", "comparison.png"}
        for name, image in derived_images.items():
            if name not in allowed:
                raise ValueError(f"unsupported derived image name: {name!r}")
            if not cv2.imwrite(str(out / name), np.asarray(image)):
                raise OSError(f"failed to write {name}")
        (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return out
```

`service.py:reprocess_saved_layer(root, trial_id, layer_index, take: int = 1)`: pass `take=take` to `archive.layer_dir(...)`. In `module.py` the `reprocess` endpoint gains `take: int = 1` as a query parameter and forwards it.

- [ ] **Step 4: Run targeted tests — green** (`tests/test_extrusion.py::test_archive_writes_reprocessable_layer` and the job archive tests must still pass — take defaults to 1 so directory names are unchanged).

- [ ] **Step 5: Commit** — `feat(extrusion): archive trial mode, repeat takes, and characterization captures`

---

### Task 8: `measure.py` — session and `RingMeasureJob`

**Files:**
- Create: `tasni/modules/extrusion/measure.py`
- Test: `tests/test_extrusion_measure.py`

**Interfaces:**
- Consumes: `_build_inspection_move`, `_wait_program`, `_program_name`, `_utcnow`, `_git_commit`, `_warn_if_stale` from `service.py`; `_camera_hold`, `ensure_real_robot_link` from `..calibration.service`; `process_observation(..., floor_profile=)`; `ExtrusionArchive` from Task 7.
- Produces: `MODE = "MEASURE_ONLY"`, `measure_station_requirements(rdk, plan, config) -> dict`, `MeasureSession` (`create`, `load`, `latest`, `next_take`, `floor_profile`, `record_take`, `save`, `to_json`), `RingMeasureJob(services, plan, session, layer_index, *, annotation, check_collisions)` returning `{"kind": "ring_measure", ...}`.

- [ ] **Step 1: Failing job tests**

```python
from test_extrusion_job import Ctx, FakeCamera, FakeRdk, services  # noqa: F401
from tasni.modules.extrusion import measure as measure_mod
from tasni.modules.extrusion.measure import MeasureSession, RingMeasureJob
from tasni.modules.extrusion.models import DeviationMetrics, RingGeometry
from tasni.modules.extrusion.processing import ProcessingResult


def fake_measure_processing(**kwargs):
    layer = kwargs["layer"]
    pts = np.array([[p.x_mm, p.y_mm, p.z_mm + 6.0] for p in layer.points])
    metrics = DeviationMetrics(mean_absolute_mm=6.4, rms_mm=7.1, maximum_mm=10.0,
                               measured_center_mm=(10.0, 0.0), measured_radius_mm=40.0,
                               path_completeness=0.99, maximum_angular_gap_deg=5, valid=True,
                               center_offset_mm=(10.0, 0.0), center_offset_norm_mm=10.0,
                               shape_rms_mm=0.3, shape_max_mm=0.8)
    geometry = RingGeometry(top_z_mean_mm=6, top_z_min_mm=5, top_z_max_mm=7, top_z_std_mm=0.5,
                            height_mean_mm=6, height_min_mm=5, height_max_mm=7,
                            height_reference="build_plane", bead_width_mean_mm=8,
                            bead_width_min_mm=7, bead_width_max_mm=9, bead_width_bins=36)
    image = np.zeros((12, 12), np.uint8)
    fake_measure_processing.calls.append(kwargs)
    return ProcessingResult(pts, None, metrics, image, image, np.zeros((12, 12, 3), np.uint8),
                            {"counts": {"raw_depth_pixels": 256}, "timings_ms": {"total_ms": 10.0},
                             "branch_guard_attempts": [{"attempt": 1}]},
                            filtered_xyz=pts.copy(), geometry=geometry)


fake_measure_processing.calls = []


def measure_env(tmp_path, monkeypatch, *, hardware_approved=False):
    svc, rdk, camera = services(tmp_path)
    svc.config.extrusion.hardware_io_test_approved = hardware_approved
    monkeypatch.setattr(measure_mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(measure_mod, "process_observation", fake_measure_processing)
    monkeypatch.setattr(measure_mod, "_git_commit", lambda: "abc123")
    monkeypatch.setattr(measure_mod.time, "sleep", lambda _: None)
    monkeypatch.setattr(measure_mod.runs, "read_active", lambda module: {"run_id": "cal-1"})
    fake_measure_processing.calls.clear()
    return svc, rdk, camera


def auto_plan(layers=3):
    recipe = CylinderRecipe(radius_mm=40, layer_count=layers, layer_height_mm=6,
                            bead_diameter_mm=8, robot_speed_mm_s=75, extrusion_rate_pct=0,
                            points_per_circle=24)
    setup = CylinderSetup(print_tool="LongCalibTool", work_frame="Tasni Work Frame",
                          inspection_tool="Realsense", inspection_auto=True,
                          center_x_mm=200, center_y_mm=150)
    return generate_cylinder_plan(recipe, setup)


def test_measure_moves_only_the_camera_and_never_touches_the_valve(tmp_path, monkeypatch):
    svc, rdk, camera = measure_env(tmp_path, monkeypatch, hardware_approved=False)
    plan = auto_plan()
    session = MeasureSession.create(tmp_path / "runs" / "extrusion", plan, note="rings")
    out = RingMeasureJob(svc, plan, session, 1, annotation={"introduced_offset_mm": None},
                         check_collisions=True)(Ctx())
    kinds = [e[0] for e in rdk.events]
    assert "station-program" not in kinds and "create" not in kinds     # no valve, no layer program
    assert "create-target" in kinds and "create-inspection" in kinds
    assert ("start", "TasniCylinder_MEASURE_%s_L001_Inspect" % plan.fingerprint[:10], True) in rdk.events
    assert rdk.events[-1] == ("move-joints", "START")
    assert any(name.endswith("_Inspect") for name in rdk.deleted)
    assert camera.grabs == 2                                             # readiness + one measurement
    assert out["kind"] == "ring_measure" and out["mode"] == "MEASURE_ONLY"
    layer_dir = Path(out["layer_dir"])
    assert layer_dir.name == "layer-001"
    manifest = json.loads((layer_dir / "manifest.json").read_text())
    assert manifest["mode"] == "MEASURE_ONLY" and manifest["take"] == 1
    assert manifest["geometry"]["bead_width_mean_mm"] == 8
    timings = manifest["processing"]["timings_ms"]
    assert timings["capture_ms"] >= 0
    assert timings["acquisition_to_path_ms"] == pytest.approx(timings["capture_ms"] + 10.0)
    assert (layer_dir / "depth.npy").is_file() and (layer_dir / "color.png").is_file()


def test_repeat_takes_and_the_floor_from_the_previous_layer(tmp_path, monkeypatch):
    svc, rdk, camera = measure_env(tmp_path, monkeypatch)
    plan = auto_plan()
    root = tmp_path / "runs" / "extrusion"
    session = MeasureSession.create(root, plan)
    RingMeasureJob(svc, plan, session, 1, annotation={}, check_collisions=True)(Ctx())
    second = RingMeasureJob(svc, plan, session, 1, annotation={"note": "re-placed"},
                            check_collisions=True)(Ctx())
    assert Path(second["layer_dir"]).name == "layer-001-take02"
    assert fake_measure_processing.calls[-1].get("floor_profile") is None   # layer 1: build plane
    third = RingMeasureJob(svc, plan, session, 2, annotation={"introduced_offset_mm": [10, 0]},
                           check_collisions=True)(Ctx())
    floor = fake_measure_processing.calls[-1]["floor_profile"]
    assert floor is not None and np.asarray(floor).shape[1] == 3          # layer 2: ring 1's top
    assert json.loads((Path(third["layer_dir"]) / "manifest.json").read_text())["annotation"] == {"introduced_offset_mm": [10, 0]}
    # Session survives a restart.
    reloaded = MeasureSession.load(root, session.trial_id)
    assert reloaded.takes == {1: 2, 2: 1}
    assert MeasureSession.latest(root).trial_id == session.trial_id
    assert reloaded.last_pose is not None


def test_measure_archives_the_raw_frame_when_processing_fails(tmp_path, monkeypatch):
    svc, rdk, _ = measure_env(tmp_path, monkeypatch)
    monkeypatch.setattr(measure_mod, "process_observation",
                        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("bad skeleton")))
    plan = auto_plan()
    session = MeasureSession.create(tmp_path / "runs" / "extrusion", plan)
    with pytest.raises(RuntimeError, match="raw RGB-D archived"):
        RingMeasureJob(svc, plan, session, 1, annotation={}, check_collisions=True)(Ctx())
    layer = session.trial_dir / "layer-001"
    assert (layer / "depth.npy").is_file() and "bad skeleton" in (layer / "report.json").read_text()
    assert rdk.events[-1] == ("move-joints", "START")                     # still returns home


def test_measure_blocks_before_motion_when_the_camera_is_offline(tmp_path, monkeypatch):
    from tasni.core.camera import CameraError
    svc, rdk, camera = measure_env(tmp_path, monkeypatch)
    def offline(**kwargs): raise CameraError("camera timeout (100.123.63.127:1024)")
    camera.grab = offline
    plan = auto_plan()
    session = MeasureSession.create(tmp_path / "runs" / "extrusion", plan)
    with pytest.raises(RuntimeError, match="camera is not ready"):
        RingMeasureJob(svc, plan, session, 1, annotation={}, check_collisions=True)(Ctx())
    assert rdk.events == []
```

- [ ] **Step 2: Run, expect ImportError** for `tasni.modules.extrusion.measure`.

- [ ] **Step 3: Implement `measure.py`**

```python
"""Ring-stack measure-only experiment: inspect -> capture -> process -> archive.

Nothing here prints. The operator places dried rings by hand; each press moves
ONLY the camera (the same derived, collision-validated, wrist-gated inspection
move the live print uses), takes one RGB-D frame, measures it and returns to
the start pose. No layer program, no AirOn/AirOff, no hardware-I/O gate.
Trials are archived with ``mode = "MEASURE_ONLY"`` and never counted as prints.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from ...core import runs
from ...core.build_info import build_info
from ...core.jobrunner import JobContext
from ...core.logging import REPO_ROOT
from ...core.rdk_io import RdkIO
from ..calibration.service import _camera_hold, ensure_real_robot_link
from .archive import ExtrusionArchive
from .models import CylinderPlan, LayerManifest
from .processing import characterize_ring, process_observation
from .service import (_build_inspection_move, _git_commit, _program_name, _utcnow,
                      _wait_program, _warn_if_stale)
from .toolpath import points_array

MODE = "MEASURE_ONLY"


def measure_station_requirements(rdk: RdkIO, plan: CylinderPlan, config) -> dict:
    """Only what the camera move needs: no print tool, no valve programs."""
    selected = plan.setup
    checks = [("work_frame", selected.work_frame, "frame"),
              ("inspection_tool", selected.inspection_tool, "tool")]
    if not selected.inspection_auto:
        checks.append(("inspection_target", selected.inspection_target, "target"))
    items = [{"role": role, "name": name, "type": kind,
              "present": rdk.item_exists_as(name, kind)} for role, name, kind in checks]
    return {"ready": all(item["present"] for item in items), "items": items,
            "missing": [item for item in items if not item["present"]]}


def _provenance(services) -> dict:
    return {"git_commit": _git_commit(), "build": build_info(),
            "calibration": runs.read_active("calibration"),
            "camera_resolution": services.config.camera.resolution,
            "camera_intrinsics": {
                "K": np.asarray(services.config.camera.K, dtype=float).tolist(),
                "dist_coeffs": list(services.config.camera.dist_coeffs)},
            "processing_config": services.config.extrusion.model_dump(mode="json")}


class MeasureSession:
    """One MEASURE_ONLY trial and everything measured in it (persisted as session.json)."""

    def __init__(self, root: Path, trial_id: str):
        self.root = Path(root)
        self.trial_id = trial_id
        self.trial_dir = self.root / trial_id
        self.takes: dict[int, int] = {}
        self.tops: dict[int, list[list[float]]] = {}      # layer -> latest measured_xyz
        self.last_pose: dict | None = None
        self.characterizations: list[dict] = []
        self.records: list[dict] = []

    # -- persistence --------------------------------------------------------
    @classmethod
    def create(cls, root: Path, plan: CylinderPlan, *, note: str = "",
               provenance: dict | None = None) -> "MeasureSession":
        trial_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{plan.fingerprint[:8]}"
        ExtrusionArchive(root).create_trial(
            trial_id, plan, provenance=provenance or {}, mode=MODE,
            experiment={"note": note, "kind": "hand-placed dried rings",
                        "created_at": _utcnow()})
        session = cls(root, trial_id)
        session.save()
        return session

    @classmethod
    def load(cls, root: Path, trial_id: str) -> "MeasureSession":
        session = cls(root, trial_id)
        data = json.loads((session.trial_dir / "session.json").read_text(encoding="utf-8"))
        session.takes = {int(k): int(v) for k, v in data.get("takes", {}).items()}
        session.tops = {int(k): v for k, v in data.get("tops", {}).items()}
        session.last_pose = data.get("last_pose")
        session.characterizations = list(data.get("characterizations", []))
        session.records = list(data.get("records", []))
        return session

    @classmethod
    def latest(cls, root: Path) -> "MeasureSession | None":
        root = Path(root)
        if not root.is_dir():
            return None
        for path in sorted(root.iterdir(), reverse=True):
            trial_file, session_file = path / "trial.json", path / "session.json"
            if not (trial_file.is_file() and session_file.is_file()):
                continue
            if json.loads(trial_file.read_text(encoding="utf-8")).get("mode") == MODE:
                return cls.load(root, path.name)
        return None

    def save(self) -> None:
        (self.trial_dir / "session.json").write_text(
            json.dumps(self.to_json(), indent=2), encoding="utf-8")

    def to_json(self) -> dict:
        return {"trial_id": self.trial_id, "mode": MODE, "takes": self.takes,
                "tops": self.tops, "last_pose": self.last_pose,
                "characterizations": self.characterizations, "records": self.records}

    # -- experiment state ---------------------------------------------------
    def next_take(self, layer_index: int) -> int:
        return self.takes.get(layer_index, 0) + 1

    def floor_profile(self, layer_index: int) -> np.ndarray | None:
        below = self.tops.get(layer_index - 1)
        return None if below is None else np.asarray(below, dtype=float)

    def record_take(self, *, layer_index: int, take: int, measured_xyz, pose: dict | None,
                    summary: dict) -> None:
        self.takes[layer_index] = take
        if measured_xyz is not None:
            self.tops[layer_index] = np.asarray(measured_xyz, dtype=float).tolist()
        if pose:
            self.last_pose = pose
        self.records.append(summary)


def _inspect_and_capture(services, ctx: JobContext, plan: CylinderPlan, layer, *,
                         inspection_name: str, start_joints, seed_pose, collisions: bool,
                         artifacts: list[str]) -> dict:
    """Move the camera to the derived pose, settle, read the pose, grab ONE frame."""
    rdk: RdkIO = services.rdk
    ecfg = services.config.extrusion
    inspect = _build_inspection_move(
        rdk, plan, layer, inspection_name=inspection_name, config=ecfg,
        camera=services.config.camera, start_joints=start_joints,
        seed_pose=seed_pose, collisions=collisions)
    ctx.check_cancel()
    artifacts.extend(inspect["artifacts"])
    if rdk.start_program(inspection_name, real_robot=True) < 0:
        raise RuntimeError(f"inspection program {inspection_name} could not start")
    _wait_program(ctx, rdk, inspection_name)
    time.sleep(ecfg.settle_s)
    rdk.use_named_tool_frame(plan.setup.inspection_tool, plan.setup.work_frame)
    T_work_camera = rdk.camera_pose_T()
    started = time.perf_counter()
    frame = services.camera.grab(with_depth=True, timeout=ecfg.grab_timeout_s)
    capture_ms = (time.perf_counter() - started) * 1000.0
    if frame.depth is None:
        raise RuntimeError("RGB-D capture returned no depth")
    ok, jpeg = cv2.imencode(".jpg", frame.color)
    if ok:
        ctx.frame(jpeg.tobytes())
    return {"inspect": inspect, "T_work_camera": T_work_camera, "frame": frame,
            "capture_ms": capture_ms}


def _prepare_robot(services, ctx: JobContext, plan: CylinderPlan, *, label: str):
    """Everything before motion: station items, camera readiness, RUN_ROBOT, link."""
    rdk: RdkIO = services.rdk
    ecfg = services.config.extrusion
    _warn_if_stale(ctx)
    required = measure_station_requirements(rdk, plan, ecfg)
    if not required["ready"]:
        missing = ", ".join(f"{v['type']} {v['name']!r}" for v in required["missing"])
        raise RuntimeError("station is not ready: " + missing)
    if services.live.running:
        services.live.stop()
    try:
        with _camera_hold(services, f"{label}-camera-check"):
            check = services.camera.grab(with_depth=True, timeout=ecfg.grab_timeout_s)
        if check.depth is None:
            raise RuntimeError("RGB-D readiness frame contained no depth")
    except Exception as exc:
        raise RuntimeError(
            "measurement blocked before robot motion: inspection camera is not ready: "
            f"{exc}") from exc
    ctx.log("inspection camera ready: depth frame received before robot motion")
    ctx.check_cancel()
    if rdk.apply_run_mode("run_robot") != "run_robot":
        raise RuntimeError("RoboDK refused RUN_ROBOT mode")
    ensure_real_robot_link(rdk, services.config.robodk, log=ctx.log)
    return rdk.current_joints()


class RingMeasureJob:
    """Measure ONE hand-placed ring: inspect, capture, process, archive, return."""

    def __init__(self, services, plan: CylinderPlan, session: MeasureSession,
                 layer_index: int, *, annotation: dict | None = None,
                 check_collisions: bool = True):
        if not 1 <= layer_index <= len(plan.layers):
            raise ValueError(f"layer_index {layer_index} outside 1..{len(plan.layers)}")
        self.services = services
        self.plan = plan.model_copy(deep=True)
        self.session = session
        self.layer_index = int(layer_index)
        self.annotation = dict(annotation or {})
        self.check_collisions = bool(check_collisions)
        self.result: dict | None = None

    def __call__(self, ctx: JobContext) -> dict:
        services = self.services
        rdk: RdkIO = services.rdk
        ecfg = services.config.extrusion
        layer = self.plan.layers[self.layer_index - 1]
        take = self.session.next_take(self.layer_index)
        name = _program_name(self.plan, self.layer_index, "MEASURE")
        inspection_name = name + "_Inspect"
        archive = ExtrusionArchive(REPO_ROOT / "runs" / "extrusion")
        artifacts: list[str] = []
        current_program: str | None = None
        start_joints = _prepare_robot(services, ctx, self.plan, label="extrusion-measure")
        try:
            with _camera_hold(services, "extrusion-measure"):
                ctx.progress(1, 4, f"layer {self.layer_index} take {take}: moving the camera")
                current_program = inspection_name
                captured = _inspect_and_capture(
                    services, ctx, self.plan, layer, inspection_name=inspection_name,
                    start_joints=start_joints, seed_pose=self.session.last_pose,
                    collisions=self.check_collisions, artifacts=artifacts)
                current_program = None
                inspect, frame = captured["inspect"], captured["frame"]
                T_work_camera, capture_ms = captured["T_work_camera"], captured["capture_ms"]
                ctx.progress(2, 4, "processing the frame")
                nominal = points_array(layer)
                base = dict(
                    trial_id=self.session.trial_id, layer_index=self.layer_index, take=take,
                    mode=MODE, recipe=self.plan.recipe,
                    toolpath_fingerprint=self.plan.fingerprint,
                    color_file="color.png", depth_file="depth.npy",
                    annotation=self.annotation,
                    provenance={**_provenance(services),
                                "work_frame": self.plan.setup.work_frame,
                                "inspection_tool": self.plan.setup.inspection_tool,
                                "inspection_target": inspect["target"],
                                "inspection_pose": inspect["pose"],
                                "T_work_camera": np.asarray(T_work_camera, dtype=float).tolist()})
                floor = self.session.floor_profile(self.layer_index)
                try:
                    processed = process_observation(
                        color=frame.color, depth=frame.depth, T_work_camera=T_work_camera,
                        K=services.config.camera.K, plan=self.plan, layer=layer, config=ecfg,
                        floor_profile=floor)
                except Exception as exc:
                    manifest = LayerManifest(
                        **base, processing={"valid": False, "error": str(exc),
                                            "timings_ms": {"capture_ms": capture_ms}},
                        warnings=[str(exc)])
                    archive.write_layer(manifest, nominal_xyz=nominal, commanded_xyz=nominal,
                                        color=frame.color, depth=frame.depth,
                                        report={"valid": False, "error": str(exc)})
                    self.session.takes[self.layer_index] = take
                    self.session.save()
                    raise RuntimeError(
                        f"layer {self.layer_index} take {take} measurement invalid; "
                        f"raw RGB-D archived: {exc}") from exc
                timings = processed.report["timings_ms"]
                timings["capture_ms"] = capture_ms
                timings["acquisition_to_path_ms"] = capture_ms + timings["total_ms"]
                manifest = LayerManifest(
                    **base, measured_path_file="measured_path.json",
                    pointcloud_file="height-or-pointcloud.npy",
                    metrics=processed.metrics, geometry=processed.geometry,
                    processing=processed.report, warnings=processed.metrics.warnings)
                layer_dir = archive.write_layer(
                    manifest, nominal_xyz=nominal, commanded_xyz=nominal,
                    measured_xyz=processed.measured_xyz,
                    pointcloud_xyz=processed.filtered_xyz,
                    color=frame.color, depth=frame.depth,
                    derived_images={"segmentation.png": processed.segmentation,
                                    "skeleton.png": processed.skeleton,
                                    "comparison.png": processed.comparison},
                    report={**processed.report,
                            "metrics": processed.metrics.model_dump(mode="json")})
                summary = {"layer_index": self.layer_index, "take": take,
                           "layer_dir": str(layer_dir), "annotation": self.annotation,
                           "metrics": processed.metrics.model_dump(mode="json"),
                           "geometry": (processed.geometry.model_dump(mode="json")
                                        if processed.geometry else None),
                           "timings_ms": timings, "valid": processed.metrics.valid,
                           "timestamp": _utcnow()}
                self.session.record_take(layer_index=self.layer_index, take=take,
                                         measured_xyz=processed.measured_xyz,
                                         pose=inspect["pose"], summary=summary)
                self.session.save()
                ctx.log(f"layer {self.layer_index} take {take}: offset "
                        f"{processed.metrics.center_offset_norm_mm:.2f} mm, RMS "
                        f"{processed.metrics.rms_mm:.2f} mm, "
                        f"{timings['acquisition_to_path_ms']:.0f} ms acquisition->path")
            ctx.progress(4, 4, "returning to the start pose")
            self.result = {"kind": "ring_measure", "mode": MODE,
                           "trial_id": self.session.trial_id,
                           "fingerprint": self.plan.fingerprint, **summary}
            return self.result
        finally:
            if current_program:
                try:
                    rdk.stop_program(current_program)
                except Exception:
                    pass
            try:
                rdk.move_j_joints(start_joints)
            except Exception:
                pass
            try:
                rdk.delete_items(list(dict.fromkeys(reversed(artifacts))))
            except Exception:
                pass
```

- [ ] **Step 4: Run targeted tests — green.** Note the `test_measure_blocks_before_motion...` test expects `rdk.events == []`: `_prepare_robot` must raise before `apply_run_mode`.

- [ ] **Step 5: Commit** — `feat(extrusion): MEASURE_ONLY job — inspect, capture, measure, archive a hand-placed ring`

---

### Task 9: `RingCharacterizeJob`

**Files:**
- Modify: `tasni/modules/extrusion/measure.py`
- Test: `tests/test_extrusion_measure.py`

**Interfaces:**
- Produces: `RingCharacterizeJob(services, plan, session, *, check_collisions)` returning `{"kind": "ring_characterize", "characterization": {...summary...}, "capture_dir": str}`; appends the summary to `session.characterizations` and saves.

- [ ] **Step 1: Failing test**

```python
from tasni.modules.extrusion.measure import RingCharacterizeJob
from tasni.modules.extrusion.processing import CharacterizationResult


def fake_characterize(**kwargs):
    fake_characterize.calls.append(kwargs)
    image = np.zeros((12, 12), np.uint8)
    return CharacterizationResult(
        radius_mm=61.2, center_mm=(214.0, 141.0), bead_width_mm=8.3, bead_width_min_mm=7.0,
        bead_width_max_mm=9.5, top_z_mean_mm=6.4, top_z_min_mm=5.1, top_z_max_mm=9.8,
        measured_xyz=np.zeros((10, 3)), segmentation=image, skeleton=image,
        comparison=np.zeros((12, 12, 3), np.uint8),
        report={"coarse": {"radius_mm": 60.0}, "timings_ms": {"total_ms": 12.0}})


fake_characterize.calls = []


def test_characterize_job_measures_the_ring_and_stores_it_in_the_session(tmp_path, monkeypatch):
    svc, rdk, camera = measure_env(tmp_path, monkeypatch)
    monkeypatch.setattr(measure_mod, "characterize_ring", fake_characterize)
    fake_characterize.calls.clear()
    plan = auto_plan()
    root = tmp_path / "runs" / "extrusion"
    session = MeasureSession.create(root, plan)
    out = RingCharacterizeJob(svc, plan, session, check_collisions=True)(Ctx())
    assert out["kind"] == "ring_characterize"
    assert out["characterization"]["radius_mm"] == 61.2
    assert fake_characterize.calls[-1]["search_center_mm"] == (200.0, 150.0)
    assert fake_characterize.calls[-1]["work_frame"] == "Tasni Work Frame"
    assert Path(out["capture_dir"]).name == "characterize-01"
    assert (Path(out["capture_dir"]) / "depth.npy").is_file()
    kinds = [e[0] for e in rdk.events]
    assert "station-program" not in kinds and "create" not in kinds
    assert rdk.events[-1] == ("move-joints", "START")
    assert MeasureSession.load(root, session.trial_id).characterizations[-1]["radius_mm"] == 61.2
```

- [ ] **Step 2: Run, expect ImportError** for `RingCharacterizeJob`.

- [ ] **Step 3: Implement** (append to `measure.py`)

```python
class RingCharacterizeJob:
    """Measure ring 1 with no recipe assumption; the operator applies it to the recipe."""

    def __init__(self, services, plan: CylinderPlan, session: MeasureSession, *,
                 check_collisions: bool = True):
        self.services = services
        self.plan = plan.model_copy(deep=True)
        self.session = session
        self.check_collisions = bool(check_collisions)
        self.result: dict | None = None

    def __call__(self, ctx: JobContext) -> dict:
        services = self.services
        rdk: RdkIO = services.rdk
        ecfg = services.config.extrusion
        layer = self.plan.layers[0]                      # aim at the first layer's top
        inspection_name = _program_name(self.plan, 1, "CHARACTERIZE") + "_Inspect"
        archive = ExtrusionArchive(REPO_ROOT / "runs" / "extrusion")
        artifacts: list[str] = []
        current_program: str | None = None
        start_joints = _prepare_robot(services, ctx, self.plan, label="extrusion-characterize")
        try:
            with _camera_hold(services, "extrusion-characterize"):
                ctx.progress(1, 3, "moving the camera over the ring")
                current_program = inspection_name
                captured = _inspect_and_capture(
                    services, ctx, self.plan, layer, inspection_name=inspection_name,
                    start_joints=start_joints, seed_pose=self.session.last_pose,
                    collisions=self.check_collisions, artifacts=artifacts)
                current_program = None
                frame = captured["frame"]
                ctx.progress(2, 3, "characterizing the ring")
                found = characterize_ring(
                    color=frame.color, depth=frame.depth,
                    T_work_camera=captured["T_work_camera"], K=services.config.camera.K,
                    search_center_mm=(float(self.plan.setup.center_x_mm),
                                      float(self.plan.setup.center_y_mm)),
                    work_frame=self.plan.setup.work_frame, config=ecfg,
                    inspection_tool=self.plan.setup.inspection_tool,
                    print_tool=self.plan.setup.print_tool)
                index = len(self.session.characterizations) + 1
                summary = {**found.summary(), "index": index, "timestamp": _utcnow(),
                           "capture_ms": captured["capture_ms"],
                           "inspection_pose": captured["inspect"]["pose"],
                           "search_center_mm": [self.plan.setup.center_x_mm,
                                                self.plan.setup.center_y_mm]}
                capture_dir = archive.write_characterization(
                    self.session.trial_id, index, color=frame.color, depth=frame.depth,
                    measured_xyz=found.measured_xyz,
                    derived_images={"segmentation.png": found.segmentation,
                                    "skeleton.png": found.skeleton,
                                    "comparison.png": found.comparison},
                    report={**found.report, "summary": summary,
                            "provenance": {**_provenance(services),
                                           "T_work_camera": np.asarray(
                                               captured["T_work_camera"], dtype=float).tolist()}})
                summary["capture_dir"] = str(capture_dir)
                self.session.characterizations.append(summary)
                if captured["inspect"]["pose"]:
                    self.session.last_pose = captured["inspect"]["pose"]
                self.session.save()
                ctx.log(f"ring: radius {found.radius_mm:.1f} mm, bead {found.bead_width_mm:.1f} mm, "
                        f"height {found.top_z_min_mm:.1f}-{found.top_z_max_mm:.1f} mm "
                        f"(mean {found.top_z_mean_mm:.1f}), centre "
                        f"({found.center_mm[0]:.1f}, {found.center_mm[1]:.1f})")
            ctx.progress(3, 3, "returning to the start pose")
            self.result = {"kind": "ring_characterize", "mode": MODE,
                           "trial_id": self.session.trial_id,
                           "characterization": summary, "capture_dir": str(capture_dir)}
            return self.result
        finally:
            if current_program:
                try:
                    rdk.stop_program(current_program)
                except Exception:
                    pass
            try:
                rdk.move_j_joints(start_joints)
            except Exception:
                pass
            try:
                rdk.delete_items(list(dict.fromkeys(reversed(artifacts))))
            except Exception:
                pass
```

- [ ] **Step 4: Run targeted tests — green.**

- [ ] **Step 5: Commit** — `feat(extrusion): characterize job — derive the recipe from the physical ring`

---

### Task 10: API — `/measure/*`, trial counts

**Files:**
- Modify: `tasni/modules/extrusion/module.py`
- Test: `tests/test_extrusion_measure.py`

**Interfaces:**
- Produces endpoints under `/api/modules/extrusion`: `GET /measure/session`, `POST /measure/session/new {note}`, `POST /measure/characterize {confirm_robot_motion, collision_check_enabled}`, `POST /measure/apply-characterization`, `POST /measure/layer {fingerprint, layer_index, annotation, confirm_robot_motion, collision_check_enabled}`; `GET /trials` summary adds `measure_only_trials`, `measure_only_takes` and counts only `LIVE_PRINT` in `total_trials`/`total_layers`; `GET /status` adds `measure_session` (trial id or null).

- [ ] **Step 1: Failing tests**

```python
from fastapi.testclient import TestClient
from tasni.core.config import AppConfig
from tasni.modules.extrusion import module as extrusion_module
from tasni.webapp.server import create_app


def api_plan(client):
    payload = {"recipe": auto_plan().recipe.model_dump(mode="json"),
               "setup": auto_plan().setup.model_dump(mode="json")}
    return client.post("/api/modules/extrusion/generate", json=payload).json()


def test_measure_layer_is_gated_on_fingerprint_confirm_and_connection_only(tmp_path, monkeypatch):
    monkeypatch.setattr(extrusion_module, "REPO_ROOT", tmp_path)
    cfg = AppConfig()
    cfg.extrusion.hardware_io_test_approved = False          # irrelevant to measuring
    client = TestClient(create_app(cfg))
    plan = api_plan(client)
    body = {"fingerprint": "stale", "layer_index": 1, "annotation": {},
            "confirm_robot_motion": True}
    assert client.post("/api/modules/extrusion/measure/layer", json=body).status_code == 409
    body["fingerprint"] = plan["fingerprint"]
    body["confirm_robot_motion"] = False
    assert client.post("/api/modules/extrusion/measure/layer", json=body).status_code == 400
    body["confirm_robot_motion"] = True
    body["layer_index"] = 99
    assert client.post("/api/modules/extrusion/measure/layer", json=body).status_code == 400
    body["layer_index"] = 1
    refused = client.post("/api/modules/extrusion/measure/layer", json=body)
    assert refused.status_code == 409 and "RoboDK" in refused.json()["detail"]
    assert "hardware" not in refused.json()["detail"].lower()


def test_measure_session_is_created_listed_and_excluded_from_print_counts(tmp_path, monkeypatch):
    monkeypatch.setattr(extrusion_module, "REPO_ROOT", tmp_path)
    client = TestClient(create_app(AppConfig()))
    assert client.get("/api/modules/extrusion/measure/session").json()["session"] is None
    assert client.post("/api/modules/extrusion/measure/session/new",
                       json={"note": "x"}).status_code == 409      # needs a generated plan
    api_plan(client)
    created = client.post("/api/modules/extrusion/measure/session/new", json={"note": "rings"}).json()
    trial_id = created["session"]["trial_id"]
    assert (tmp_path / "runs" / "extrusion" / trial_id / "session.json").is_file()
    assert client.get("/api/modules/extrusion/measure/session").json()["session"]["trial_id"] == trial_id
    assert client.get("/api/modules/extrusion/status").json()["measure_session"] == trial_id
    # A LIVE_PRINT trial beside it: only that one is a printed trial.
    live = ExtrusionArchive(tmp_path / "runs" / "extrusion")
    live.create_trial("20990101-000000-live0000", auto_plan())
    trials = client.get("/api/modules/extrusion/trials").json()
    assert trials["summary"]["total_trials"] == 1
    assert trials["summary"]["measure_only_trials"] == 1
    assert {t["trial_id"]: t["mode"] for t in trials["trials"]} == {
        trial_id: "MEASURE_ONLY", "20990101-000000-live0000": "LIVE_PRINT"}


def test_apply_characterization_rewrites_recipe_and_placement(tmp_path, monkeypatch):
    monkeypatch.setattr(extrusion_module, "REPO_ROOT", tmp_path)
    client = TestClient(create_app(AppConfig()))
    before = api_plan(client)
    assert client.post("/api/modules/extrusion/measure/apply-characterization").status_code == 409
    client.post("/api/modules/extrusion/measure/session/new", json={"note": ""})
    session = MeasureSession.latest(tmp_path / "runs" / "extrusion")
    session.characterizations.append({"index": 1, "radius_mm": 61.24, "center_mm": [214.0, 141.0],
                                      "bead_width_mm": 8.31, "top_z_mean_mm": 6.44,
                                      "top_z_min_mm": 5.1, "top_z_max_mm": 9.8})
    session.save()
    after = client.post("/api/modules/extrusion/measure/apply-characterization").json()
    assert after["fingerprint"] != before["fingerprint"]
    assert after["recipe"]["radius_mm"] == 61.2 and after["recipe"]["bead_diameter_mm"] == 8.3
    assert after["recipe"]["layer_height_mm"] == 6.4
    assert after["setup"]["center_x_mm"] == 214.0 and after["setup"]["center_y_mm"] == 141.0
    assert after["setup"]["build_plane_z_mm"] == 0.0
    assert client.get("/api/modules/extrusion/plan").json()["fingerprint"] == after["fingerprint"]
```

- [ ] **Step 2: Run, expect 404s / KeyErrors.**

- [ ] **Step 3: Implement** in `module.py`

Imports: `from .measure import (MODE as MEASURE_MODE, MeasureSession, RingCharacterizeJob, RingMeasureJob, paper_summary)` (`paper_summary` arrives in Task 11 — add it to the import then). Bodies:

```python
class MeasureSessionBody(BaseModel):
    note: str = ""


class MeasureLayerBody(FingerprintBody):
    layer_index: int = Field(ge=1)
    annotation: dict = Field(default_factory=dict)
    confirm_robot_motion: bool = False
    collision_check_enabled: bool = True


class CharacterizeBody(BaseModel):
    confirm_robot_motion: bool = False
    collision_check_enabled: bool = True
```

`__init__`: `self._measure_session: MeasureSession | None = None` and `self._active_measure_job = None`. Helpers on the class:

```python
    def _measure_root(self):
        return REPO_ROOT / "runs" / "extrusion"

    def _session(self, *, create: bool = False) -> MeasureSession | None:
        # Always re-read session.json: the running job holds its OWN MeasureSession
        # object and saves after every take, so the API's view must come from disk,
        # never from a cached copy that predates those saves.
        root = self._measure_root()
        if (self._measure_session is not None
                and (root / self._measure_session.trial_id / "session.json").is_file()):
            self._measure_session = MeasureSession.load(root, self._measure_session.trial_id)
        elif self._measure_session is None:
            self._measure_session = MeasureSession.latest(root)
        if self._measure_session is None and create:
            if self._plan is None:
                raise HTTPException(409, "generate coordinates first; a session records the plan it measures against")
            self._measure_session = MeasureSession.create(
                self._measure_root(), self._plan, note="")
        return self._measure_session

    def _invalidate_checks(self) -> None:
        self._geometry_preflight_fingerprint = None
        self._quick_sim_fingerprint = None
        self._quick_sim_layers.clear()
        self._quick_sim_approves_full_plan = False
        self._dry_run_fingerprint = None
```

(`HTTPException` is imported inside `router()`; move `from fastapi import HTTPException` to a module-level try/except import so the helper can raise it, mirroring how `APIRouter` is handled.)

Endpoints inside `router()`:

```python
        @router.get("/measure/session")
        def measure_session() -> dict:
            session = self._session()
            return {"mode": MEASURE_MODE,
                    "session": None if session is None else session.to_json()}

        @router.post("/measure/session/new")
        def measure_session_new(body: MeasureSessionBody) -> dict:
            if self._plan is None:
                raise HTTPException(409, "generate coordinates first")
            if services.jobs.running:
                raise HTTPException(409, "a job is already running")
            self._measure_session = MeasureSession.create(
                self._measure_root(), self._plan, note=body.note)
            return {"mode": MEASURE_MODE, "session": self._measure_session.to_json()}

        def _require_measure_ready(confirm: bool) -> None:
            if self._plan is None:
                raise HTTPException(409, "generate coordinates first")
            if not confirm:
                raise HTTPException(400, "confirm that the robot may move to the inspection pose")
            if not services.session.is_open:
                raise HTTPException(409, "connect to RoboDK first (the camera move is a real robot motion)")
            if services.jobs.running:
                raise HTTPException(409, "a job is already running")

        @router.post("/measure/characterize")
        def measure_characterize(body: CharacterizeBody) -> dict:
            _require_measure_ready(body.confirm_robot_motion)
            session = self._session(create=True)
            self._active_measure_job = RingCharacterizeJob(
                services, self._plan, session, check_collisions=body.collision_check_enabled)
            try:
                services.jobs.start(self._active_measure_job, name="extrusion-characterize")
            except JobBusy as exc:
                raise HTTPException(409, str(exc))
            return {"status": "started", "mode": MEASURE_MODE, "trial_id": session.trial_id}

        @router.post("/measure/apply-characterization")
        def measure_apply() -> dict:
            session = self._session()
            if session is None or not session.characterizations:
                raise HTTPException(409, "characterize a ring first")
            if services.jobs.running:
                raise HTTPException(409, "wait for the current job to finish")
            found = session.characterizations[-1]
            base_recipe = self._plan.recipe if self._plan else self._default_recipe()
            base_setup = (self._plan.setup.model_dump(mode="json") if self._plan
                          else self._default_setup())
            recipe = base_recipe.model_copy(update={
                "radius_mm": round(float(found["radius_mm"]), 1),
                "bead_diameter_mm": round(max(0.5, float(found["bead_width_mm"])), 1),
                "layer_height_mm": round(max(0.5, float(found["top_z_mean_mm"])), 1)})
            setup = CylinderSetup(**{**base_setup,
                                     "center_x_mm": float(found["center_mm"][0]),
                                     "center_y_mm": float(found["center_mm"][1]),
                                     "build_plane_z_mm": 0.0})
            self._plan = generate_cylinder_plan(recipe, setup)
            self._invalidate_checks()
            return self._plan.model_dump(mode="json")

        @router.post("/measure/layer")
        def measure_layer(body: MeasureLayerBody) -> dict:
            if self._plan is None or body.fingerprint != self._plan.fingerprint:
                raise HTTPException(409, "toolpath changed; generate coordinates again")
            if body.layer_index > len(self._plan.layers):
                raise HTTPException(400, f"layer_index must be 1..{len(self._plan.layers)}")
            _require_measure_ready(body.confirm_robot_motion)
            session = self._session(create=True)
            self._active_measure_job = RingMeasureJob(
                services, self._plan, session, body.layer_index,
                annotation=body.annotation, check_collisions=body.collision_check_enabled)
            try:
                services.jobs.start(self._active_measure_job, name="extrusion-measure")
            except JobBusy as exc:
                raise HTTPException(409, str(exc))
            return {"status": "started", "mode": MEASURE_MODE, "trial_id": session.trial_id,
                    "layer_index": body.layer_index, "take": session.next_take(body.layer_index)}
```

Order matters in `measure_layer`: fingerprint (409) → layer bound (400) → confirm (400) → connection (409), so the test's expectations hold.

`/status`: add `"measure_session": (self._measure_session.trial_id if self._measure_session else None)`.

`/trials`: read `mode = data.get("mode", "LIVE_PRINT")`; count `total_trials`/`total_layers`/`total_recipes` only when `mode == "LIVE_PRINT"`; accumulate `measure_only_trials += 1` and `measure_only_takes += len(layers)` otherwise; include `"mode": mode` on each item and the two new keys in `summary`. The glob `layer-*/manifest.json` already picks up takes; add `"take": manifest.get("take", 1)` and `"annotation": manifest.get("annotation", {})` to each layer entry.

- [ ] **Step 4: Run targeted tests — green** (`tests/test_extrusion.py` API tests included).

- [ ] **Step 5: Commit** — `feat(extrusion): measure-only API — session, characterize, apply, measure layer`

---

### Task 11: Paper summary

**Files:**
- Modify: `tasni/modules/extrusion/measure.py` (`paper_summary`)
- Modify: `tasni/modules/extrusion/module.py` (`GET /trials/{trial_id}/paper-summary`)
- Test: `tests/test_extrusion_measure.py`

**Interfaces:**
- Produces: `paper_summary(root: Path, trial_id: str) -> dict` with keys `trial_id, mode, takes, valid, conditions[], timing_ms{capture_ms,total_ms,acquisition_to_path_ms}, height_mm{}, bead_width_mm{}, characterization, markdown`.

- [ ] **Step 1: Failing test**

```python
from tasni.modules.extrusion.measure import paper_summary


def _write_take(root, trial_id, layer_index, take, *, offset, offset_norm, rms, mean_abs, maximum,
                acq_ms, valid=True):
    manifest = LayerManifest(
        trial_id=trial_id, layer_index=layer_index, take=take, mode="MEASURE_ONLY",
        recipe=auto_plan().recipe, toolpath_fingerprint="f" * 64,
        annotation={"introduced_offset_mm": offset},
        metrics=DeviationMetrics(mean_absolute_mm=mean_abs, rms_mm=rms, maximum_mm=maximum,
                                 measured_center_mm=(0, 0), measured_radius_mm=40,
                                 path_completeness=0.99, maximum_angular_gap_deg=4, valid=valid,
                                 center_offset_norm_mm=offset_norm, shape_rms_mm=0.4),
        geometry=RingGeometry(top_z_mean_mm=6, top_z_min_mm=5, top_z_max_mm=9, top_z_std_mm=1,
                              height_mean_mm=6, height_min_mm=5, height_max_mm=9,
                              height_reference="build_plane", bead_width_mean_mm=8,
                              bead_width_min_mm=7, bead_width_max_mm=9, bead_width_bins=36),
        processing={"timings_ms": {"capture_ms": 40.0, "total_ms": acq_ms - 40.0,
                                   "acquisition_to_path_ms": acq_ms}})
    ExtrusionArchive(root).write_layer(manifest, nominal_xyz=np.zeros((4, 3)),
                                       commanded_xyz=np.zeros((4, 3)))


def test_paper_summary_groups_by_introduced_offset_and_reports_timing(tmp_path):
    root = tmp_path / "runs" / "extrusion"
    session = MeasureSession.create(root, auto_plan(), note="rings")
    t = session.trial_id
    _write_take(root, t, 1, 1, offset=None, offset_norm=0.4, rms=0.5, mean_abs=0.4, maximum=1.1, acq_ms=900)
    _write_take(root, t, 1, 2, offset=[0, 0], offset_norm=0.6, rms=0.6, mean_abs=0.5, maximum=1.3, acq_ms=1100)
    _write_take(root, t, 2, 1, offset=[10, 0], offset_norm=9.8, rms=7.0, mean_abs=6.3, maximum=9.9, acq_ms=1000)
    summary = paper_summary(root, t)
    assert summary["mode"] == "MEASURE_ONLY" and summary["takes"] == 3 and summary["valid"] == 3
    by_name = {c["condition"]: c for c in summary["conditions"]}
    assert by_name["true (no introduced offset)"]["takes"] == 2
    shifted = by_name["introduced offset (10, 0) mm"]
    assert shifted["takes"] == 1 and shifted["center_offset_norm_mm"]["mean"] == 9.8
    assert summary["timing_ms"]["acquisition_to_path_ms"]["mean"] == pytest.approx(1000.0)
    assert summary["timing_ms"]["acquisition_to_path_ms"]["sd"] == pytest.approx(100.0)
    assert summary["height_mm"]["height_max_mm"]["mean"] == 9.0
    assert "10" in summary["markdown"] and "hand-placed" in summary["markdown"]


def test_paper_summary_endpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(extrusion_module, "REPO_ROOT", tmp_path)
    client = TestClient(create_app(AppConfig()))
    api_plan(client)
    trial_id = client.post("/api/modules/extrusion/measure/session/new", json={"note": ""}).json()["session"]["trial_id"]
    _write_take(tmp_path / "runs" / "extrusion", trial_id, 1, 1, offset=None, offset_norm=0.5,
                rms=0.5, mean_abs=0.4, maximum=1.0, acq_ms=950)
    got = client.get(f"/api/modules/extrusion/trials/{trial_id}/paper-summary").json()
    assert got["takes"] == 1 and "markdown" in got
    assert client.get("/api/modules/extrusion/trials/nope/paper-summary").status_code == 404
```

- [ ] **Step 2: Run, expect ImportError.**

- [ ] **Step 3: Implement** (append to `measure.py`)

```python
def _stat(values) -> dict:
    arr = np.array([float(v) for v in values if v is not None], dtype=float)
    return {"n": int(arr.size),
            "mean": float(arr.mean()) if arr.size else None,
            "sd": float(arr.std(ddof=1)) if arr.size > 1 else None,
            "min": float(arr.min()) if arr.size else None,
            "max": float(arr.max()) if arr.size else None}


def _condition_name(manifest: dict) -> str:
    offset = (manifest.get("annotation") or {}).get("introduced_offset_mm")
    if not offset or not any(float(v) for v in offset):
        return "true (no introduced offset)"
    return f"introduced offset ({offset[0]:g}, {offset[1]:g}) mm"


def _fmt(stat: dict, digits: int = 2) -> str:
    if stat["mean"] is None:
        return "–"
    text = f"{stat['mean']:.{digits}f}"
    return text if stat["sd"] is None else f"{text} ± {stat['sd']:.{digits}f}"


def paper_summary(root: Path, trial_id: str) -> dict:
    """Numbers the PFH short paper asks for, grouped by the operator's ground truth."""
    trial_dir = Path(root) / trial_id
    trial_file = trial_dir / "trial.json"
    if not trial_file.is_file():
        raise FileNotFoundError(f"trial does not exist: {trial_id}")
    trial = json.loads(trial_file.read_text(encoding="utf-8"))
    takes = [json.loads(p.read_text(encoding="utf-8"))
             for p in sorted(trial_dir.glob("layer-*/manifest.json"))]
    groups: dict[str, list[dict]] = {}
    for manifest in takes:
        groups.setdefault(_condition_name(manifest), []).append(manifest)
    conditions = []
    for name, items in groups.items():
        metrics = [m["metrics"] for m in items if m.get("metrics")]
        conditions.append({
            "condition": name, "takes": len(items),
            "valid": sum(1 for x in metrics if x.get("valid")),
            "center_offset_norm_mm": _stat(x.get("center_offset_norm_mm") for x in metrics),
            "mean_absolute_mm": _stat(x.get("mean_absolute_mm") for x in metrics),
            "rms_mm": _stat(x.get("rms_mm") for x in metrics),
            "maximum_mm": _stat(x.get("maximum_mm") for x in metrics),
            "shape_rms_mm": _stat(x.get("shape_rms_mm") for x in metrics)})
    timings = [(m.get("processing") or {}).get("timings_ms") or {} for m in takes]
    timing = {key: _stat(t.get(key) for t in timings)
              for key in ("capture_ms", "total_ms", "acquisition_to_path_ms")}
    geometry = [m["geometry"] for m in takes if m.get("geometry")]
    height = {key: _stat(g.get(key) for g in geometry)
              for key in ("height_mean_mm", "height_min_mm", "height_max_mm", "top_z_std_mm")}
    bead = {key: _stat(g.get(key) for g in geometry)
            for key in ("bead_width_mean_mm", "bead_width_min_mm", "bead_width_max_mm")}
    valid = sum(1 for m in takes if (m.get("metrics") or {}).get("valid"))
    session_file = trial_dir / "session.json"
    characterization = None
    if session_file.is_file():
        found = json.loads(session_file.read_text(encoding="utf-8")).get("characterizations") or []
        characterization = found[-1] if found else None

    lines = [f"**Controlled validation of the sensing-and-comparison chain** — trial `{trial_id}`, "
             f"hand-placed dried beads with a known introduced offset (not a printed-cylinder "
             f"deposition deviation). {valid}/{len(takes)} measurements produced a valid, "
             f"branch-free path.", "",
             "| Condition | n | centre offset (mm) | mean abs dev (mm) | RMS (mm) | max (mm) | shape RMS (mm) |",
             "|---|---|---|---|---|---|---|"]
    for c in conditions:
        lines.append(f"| {c['condition']} | {c['takes']} | {_fmt(c['center_offset_norm_mm'])} | "
                     f"{_fmt(c['mean_absolute_mm'])} | {_fmt(c['rms_mm'])} | "
                     f"{_fmt(c['maximum_mm'])} | {_fmt(c['shape_rms_mm'])} |")
    acq = timing["acquisition_to_path_ms"]
    if acq["n"]:
        lines += ["", f"Across {acq['n']} processing cycles, RGB-D acquisition to reconstructed "
                      f"three-dimensional path took {_fmt(acq, 0)} ms "
                      f"(capture {_fmt(timing['capture_ms'], 0)} ms, processing "
                      f"{_fmt(timing['total_ms'], 0)} ms)."]
    if geometry:
        lines += ["", f"Layer height along the ring: mean {_fmt(height['height_mean_mm'], 1)} mm, "
                      f"min {_fmt(height['height_min_mm'], 1)} mm, max "
                      f"{_fmt(height['height_max_mm'], 1)} mm; bead footprint width "
                      f"{_fmt(bead['bead_width_mean_mm'], 1)} mm."]
    if characterization:
        lines += ["", f"Ring characterized from its own scan: radius "
                      f"{characterization['radius_mm']:.1f} mm, bead "
                      f"{characterization['bead_width_mm']:.1f} mm, height "
                      f"{characterization['top_z_min_mm']:.1f}–{characterization['top_z_max_mm']:.1f} mm."]
    return {"trial_id": trial_id, "mode": trial.get("mode", "LIVE_PRINT"),
            "takes": len(takes), "valid": valid, "conditions": conditions,
            "timing_ms": timing, "height_mm": height, "bead_width_mm": bead,
            "characterization": characterization, "markdown": "\n".join(lines)}
```

`module.py`:

```python
        @router.get("/trials/{trial_id}/paper-summary")
        def trial_paper_summary(trial_id: str) -> dict:
            try:
                return paper_summary(self._measure_root(), trial_id)
            except (FileNotFoundError, ValueError) as exc:
                raise HTTPException(404, str(exc)) from exc
```

- [ ] **Step 4: Run targeted tests — green.**

- [ ] **Step 5: Commit** — `feat(extrusion): paper summary — per-condition deviation, timing, height and bead`

---

### Task 12: UI — "Ring stack — measure only" card

**Files:**
- Modify: `tasni/webui/src/pages/Extrusion.tsx`

**Interfaces:**
- Consumes the Task 10/11 endpoints. `api` is `moduleApi("extrusion")` (`api.get(path)`, `api.post(path, body)`); existing state used: `plan`, `status`, `busy`, `connected`, `setMessage`, `refreshStatus`, `cancel`, `setPlan`, `setPreflight`, `setResult`.

- [ ] **Step 1: Types and state** (next to the other interfaces / state hooks)

```tsx
interface MeasureTake {
  layer_index: number; take: number; layer_dir: string; valid: boolean; timestamp: string;
  annotation: { introduced_offset_mm?: [number, number] | null; note?: string };
  metrics: { mean_absolute_mm: number; rms_mm: number; maximum_mm: number;
    center_offset_mm: [number, number]; center_offset_norm_mm: number; shape_rms_mm: number;
    measured_radius_mm: number; path_completeness: number };
  geometry: { height_mean_mm: number; height_min_mm: number; height_max_mm: number;
    bead_width_mean_mm: number } | null;
  timings_ms: { capture_ms: number; total_ms: number; acquisition_to_path_ms: number };
}
interface Characterization {
  index: number; radius_mm: number; center_mm: [number, number]; bead_width_mm: number;
  top_z_mean_mm: number; top_z_min_mm: number; top_z_max_mm: number;
}
interface MeasureSession {
  trial_id: string; takes: Record<string, number>; records: MeasureTake[];
  characterizations: Characterization[];
}

const [measureSession, setMeasureSession] = useState<MeasureSession | null>(null);
const [measureLayer, setMeasureLayer] = useState(1);
const [offsetX, setOffsetX] = useState(0);
const [offsetY, setOffsetY] = useState(0);
const [measureNote, setMeasureNote] = useState("");
const [confirmMotion, setConfirmMotion] = useState(false);
const [paper, setPaper] = useState<string | null>(null);
```

- [ ] **Step 2: Handlers**

```tsx
const refreshMeasure = useCallback(async () => {
  try {
    const data = await api.get<{ session: MeasureSession | null }>("/measure/session");
    setMeasureSession(data.session);
  } catch { /* module unavailable */ }
}, []);
useEffect(() => { refreshMeasure(); }, [refreshMeasure]);
useEffect(() => { if (status && !status.running) refreshMeasure(); }, [status?.running, refreshMeasure]);

const newMeasureSession = async () => {
  setBusy(true);
  try {
    const data = await api.post<{ session: MeasureSession }>("/measure/session/new", { note: measureNote });
    setMeasureSession(data.session); setPaper(null);
    setMessage(`New measure-only session ${data.session.trial_id}.`);
  } catch (e: any) { setMessage(e.message); } finally { setBusy(false); }
};
const characterize = async () => {
  setBusy(true);
  try {
    await api.post("/measure/characterize", { confirm_robot_motion: confirmMotion, collision_check_enabled: true });
    setMessage("CHARACTERIZE started — the robot moves the camera over ring 1, measures it and returns.");
    refreshStatus();
  } catch (e: any) { setBusy(false); setMessage(e.message); }
};
const applyCharacterization = async () => {
  setBusy(true);
  try {
    const next = await api.post<Plan>("/measure/apply-characterization");
    setPlan(next); setRecipe(next.recipe); setSetup(next.setup);
    setPreflight(null); setResult(null); setSelectedLayer(1);
    setMessage(`Recipe and placement set from the measured ring: r ${next.recipe.radius_mm} mm, bead ${next.recipe.bead_diameter_mm} mm, layer ${next.recipe.layer_height_mm} mm.`);
    refreshStatus();
  } catch (e: any) { setMessage(e.message); } finally { setBusy(false); }
};
const measure = async () => {
  if (!plan) return;
  setBusy(true);
  try {
    const shifted = offsetX !== 0 || offsetY !== 0;
    await api.post("/measure/layer", {
      fingerprint: plan.fingerprint, layer_index: measureLayer,
      annotation: { introduced_offset_mm: shifted ? [offsetX, offsetY] : null, note: measureNote },
      confirm_robot_motion: confirmMotion, collision_check_enabled: true,
    });
    setMessage(`MEASURE layer ${measureLayer} started — camera move only; no extrusion, no valve.`);
    refreshStatus();
  } catch (e: any) { setBusy(false); setMessage(e.message); }
};
const showPaper = async () => {
  if (!measureSession) return;
  try {
    const data = await api.get<{ markdown: string }>(`/trials/${measureSession.trial_id}/paper-summary`);
    setPaper(data.markdown);
  } catch (e: any) { setMessage(e.message); }
};
```

(`setRecipe`, `setSetup`, `setSelectedLayer` already exist in the page — confirm the names by reading the state hooks at the top of the component before wiring.)

- [ ] **Step 3: The card** (place after the *Safety workflow* card, before *Measured layers*)

```tsx
<div className="card">
  <h2>Ring stack — measure only (no extrusion)</h2>
  <p className="hint">Hand-placed dried rings. Each press moves ONLY the camera to the derived
    inspection pose (collision-checked), takes one RGB-D frame, measures it and returns.
    No layer program, no valve. Archived as <code>MEASURE_ONLY</code>, never counted as a print.</p>
  <div className="btn-row">
    <input placeholder="session / ring note" value={measureNote} onChange={(e) => setMeasureNote(e.target.value)} />
    <button className="secondary" disabled={!plan || busy || status?.running} onClick={newMeasureSession}>New session</button>
    <span className="hint">{measureSession ? `session ${measureSession.trial_id}` : "no session yet (one is created on first measure)"}</span>
  </div>
  <label><input type="checkbox" checked={confirmMotion} onChange={(e) => setConfirmMotion(e.target.checked)} />
    I confirm the robot may move the camera to the inspection pose (hands clear of the cell).</label>
  <div className="btn-row">
    <button disabled={!plan || !connected || !confirmMotion || busy || status?.running} onClick={characterize}>Characterize ring 1 — ROBOT MOVES</button>
    {measureSession?.characterizations?.length ? (() => {
      const c = measureSession.characterizations[measureSession.characterizations.length - 1];
      return <>
        <span className="hint">measured: r {c.radius_mm.toFixed(1)} mm · bead {c.bead_width_mm.toFixed(1)} mm ·
          height {c.top_z_min_mm.toFixed(1)}–{c.top_z_max_mm.toFixed(1)} (mean {c.top_z_mean_mm.toFixed(1)}) mm ·
          centre ({c.center_mm[0].toFixed(1)}, {c.center_mm[1].toFixed(1)})</span>
        <button className="secondary" disabled={busy || status?.running} onClick={applyCharacterization}>Apply to recipe & placement</button>
      </>;
    })() : null}
  </div>
  <div className="btn-row">
    <label>Layer <select value={measureLayer} onChange={(e) => setMeasureLayer(Number(e.target.value))}>
      {(plan?.layers ?? []).map((l) => <option key={l.layer_index} value={l.layer_index}>{l.layer_index}</option>)}
    </select></label>
    <label>Introduced offset X <input type="number" step={1} value={offsetX} onChange={(e) => setOffsetX(Number(e.target.value))} /> mm</label>
    <label>Y <input type="number" step={1} value={offsetY} onChange={(e) => setOffsetY(Number(e.target.value))} /> mm</label>
    <button disabled={!plan || !connected || !confirmMotion || busy || status?.running} onClick={measure}>Measure layer {measureLayer} — ROBOT MOVES</button>
    {(busy || status?.running) && <button className="secondary" onClick={cancel}>Cancel</button>}
  </div>
  {measureSession?.records?.length ? <table className="metrics">
    <thead><tr><th>Layer</th><th>Take</th><th>Introduced</th><th>Offset dx/dy (|d|)</th><th>Mean |dev|</th><th>RMS</th><th>Max</th><th>Shape RMS</th><th>Height min/mean/max</th><th>Bead</th><th>Acq→path</th><th>Valid</th></tr></thead>
    <tbody>{measureSession.records.map((r) => <tr key={`${r.layer_index}-${r.take}`}>
      <td>{r.layer_index}</td><td>{r.take}</td>
      <td>{r.annotation?.introduced_offset_mm ? `(${r.annotation.introduced_offset_mm.join(", ")}) mm` : "—"}</td>
      <td className="num">{r.metrics.center_offset_mm[0].toFixed(1)} / {r.metrics.center_offset_mm[1].toFixed(1)} ({r.metrics.center_offset_norm_mm.toFixed(2)})</td>
      <td className="num">{r.metrics.mean_absolute_mm.toFixed(2)}</td><td className="num">{r.metrics.rms_mm.toFixed(2)}</td>
      <td className="num">{r.metrics.maximum_mm.toFixed(2)}</td><td className="num">{r.metrics.shape_rms_mm.toFixed(2)}</td>
      <td className="num">{r.geometry ? `${r.geometry.height_min_mm.toFixed(1)} / ${r.geometry.height_mean_mm.toFixed(1)} / ${r.geometry.height_max_mm.toFixed(1)}` : "—"}</td>
      <td className="num">{r.geometry ? r.geometry.bead_width_mean_mm.toFixed(1) : "—"}</td>
      <td className="num">{Math.round(r.timings_ms.acquisition_to_path_ms)} ms</td>
      <td><span className={`badge ${r.valid ? "good" : "bad"}`}>{r.valid ? "VALID" : "INVALID"}</span></td>
    </tr>)}</tbody></table> : null}
  <div className="btn-row">
    <button className="secondary" disabled={!measureSession} onClick={showPaper}>Paper summary</button>
  </div>
  {paper && <pre className="log" style={{ whiteSpace: "pre-wrap" }}>{paper}</pre>}
</div>
```

- [ ] **Step 4: Typecheck and build**

Run: `cd tasni/webui && npm run typecheck && npm run build`
Expected: both pass (the existing chunk-size warning is fine).

- [ ] **Step 5: Commit** — `feat(extrusion ui): ring-stack measure-only card with characterize, measure, paper summary`

---

### Task 13: Docs, merge, brief the operator

**Files:**
- Modify: `docs/extrusion-current-handoff.md` (top "Last updated" block + a new section)
- Modify: `CLAUDE.md` roadmap (one bullet)

- [ ] **Step 1: Add to `docs/extrusion-current-handoff.md`** (new section after "Automatic inspection pose"):

```markdown
## Ring-stack measure-only experiment (paper evidence, 2026-08-27)

Design: `docs/superpowers/specs/2026-08-27-ring-stack-measure-only-design.md`;
plan: `docs/superpowers/plans/2026-08-27-ring-stack-measure-only.md`.

`MEASURE_ONLY` (`tasni/modules/extrusion/measure.py`) measures hand-placed dried
rings: each press moves only the camera to the derived inspection pose, takes one
RGB-D frame, runs `process_observation` (with the previous ring's measured top as
the floor) and archives a take under a `MEASURE_ONLY` trial that `/trials` never
counts as a print. `characterize_ring` derives radius / bead / height / centre from
ring 1 so the recipe comes from the physical ring, not a caliper.
`GET /trials/{id}/paper-summary` groups takes by the operator's introduced offset
and reports deviation, timing (`acquisition_to_path_ms`), height and bead numbers
with a ready-to-paste Markdown block.

Cell protocol: scan surface applied → Center on scanned surface → Generate → place
ring 1 → Characterize → Apply → Generate → Measure L1 ×5 (noise floor) → re-place
×3 → ring 2 true → Measure L2 → ring 3 true → Measure L3 → shift a ring 10 mm
(type it in) → Measure → shift 15 mm → Measure → Paper summary. Keep offsets
≤ 25 mm (radial ROI ±30 mm). Expected for a pure shift d: offset ≈ d, max ≈ d,
mean ≈ 0.64 d, RMS ≈ 0.71 d.

After the first real capture, copy `color.png` + `depth.npy` + `manifest.json` from
the take into `tests/fixtures/extrusion/ring1/` (npz-compressed) and add a
regression test that reprocesses it to the archived metrics.
```

- [ ] **Step 2: `CLAUDE.md` roadmap** — add one bullet under "Roadmap / status": `- ✅ **Ring-stack measure-only experiment** (paper evidence): measure hand-placed rings through the inspection chain, characterize ring 1, paper summary. Spec + plan in docs/superpowers/. Cell run pending.`

- [ ] **Step 3: Full targeted verification**

```
py -3.10 -m pytest tests/test_extrusion_measure.py tests/test_extrusion.py tests/test_extrusion_job.py tests/test_extrusion_processing.py tests/test_rdk_io_extrusion.py -q
cd tasni/webui && npm run typecheck && npm run build
```

- [ ] **Step 4: Commit, merge, push**

```bash
git add docs/extrusion-current-handoff.md CLAUDE.md
git commit -m "docs: ring-stack measure-only experiment — protocol and pointers"
git push -u origin extrusion-ring-stack
# from the main checkout:
git merge --no-ff extrusion-ring-stack -m "Merge extrusion-ring-stack: measure-only ring experiment for the paper"
git push
```

- [ ] **Step 5: Tell the operator** — the pushed merge hash, the test count, that the backend must be **restarted** before the cell run, and the protocol above. The cell run itself, the fixture, and the numbers into the paper are the operator's next session.

---

## Self-review

**Spec coverage:** §4.1 mode/session/job → Tasks 7–8; gates → Task 10; §4.2 floor → Task 4, offset/shape → Task 2, height/bead → Task 5, timing → Task 8 (`capture_ms`, `acquisition_to_path_ms`), characterize → Tasks 6 & 9, filter-chain extraction → Task 3; §4.3 endpoints, config fields, `/trials` counts, paper summary → Tasks 4, 10, 11; §4.4 UI → Task 12; §4.5 renderer + every listed test → Tasks 1–11 (the "reprocess" row is covered by Task 7's take-aware `reprocess_saved_layer` plus the existing live-job reprocess test; the real-capture fixture is deliberately a post-cell step, recorded in Task 13). §5 non-goals respected: `CylinderPrintJob` untouched.

**Placeholder scan:** none — every code step carries its code; tuning guidance in Tasks 3/4/6 names the exact knobs.

**Type consistency:** `floor_profile` (Tasks 4, 5, 8); `RingGeometry` field names identical in Tasks 5, 8, 11; `CharacterizationResult.summary()` keys match what Task 10's `measure_apply` reads (`radius_mm`, `center_mm`, `bead_width_mm`, `top_z_mean_mm`); `MeasureSession.to_json()` keys (`records`, `characterizations`, `takes`, `trial_id`) match the UI's `MeasureSession` interface in Task 12; `_program_name(plan, index, "MEASURE") + "_Inspect"` matches the Task 8 test's expected program name.
