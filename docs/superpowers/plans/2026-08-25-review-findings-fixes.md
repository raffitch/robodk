# Confirmed Review Findings (2026-08-25) Fix Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the 8 findings CONFIRMED by the 2026-08-25 adversarial code review of the scan chain and the cylinder (extrusion) inspection path.

**Architecture:** Small, surgical fixes at the exact defect sites — one commit per finding. No refactors beyond what a finding requires. The scan-audit fix plan (`docs/superpowers/plans/2026-08-14-scan-chain-fixes.md`) continues separately; the one overlap (finding 4 = audit A2) is executed FROM that plan, not re-specced here.

**Tech Stack:** Python 3.10 (`py -3.10`), pytest, numpy, robodk/robolink; Jetson server is plain Python deployed via `tools/jetson_deploy.py`.

**Spec:** The findings table below (the review's verified output — there is no separate spec file).

## The findings this plan implements

| # | File:line | Defect (one line) |
|---|---|---|
| 1 | `tasni/core/rdk_io.py:1329` | `create_inspection_target` hands a WORK-frame pose to `SolveIK`, which does its math in the ROBOT BASE frame → joint target is translated by the full base→work offset (robolink: "pose must be … with respect to the robot base unless you provide the tool and/or reference") |
| 2 | `tasni/modules/scan/service.py:1141` | Vision-vs-depth rectangle corroboration checks sorted side LENGTHS only — a laterally shifted segmentation (equal lengths, moved centre) replaces the work-frame corners ~30–40 mm off under a green lock |
| 3 | `tasni/modules/scan/service.py:611` | Hybrid lock swaps `corners_cam_mm` only; `extent_mm` stays depth-derived, so `plan_scan` tours the smaller depth rectangle and the gate payload contradicts the record |
| 4 | `tasni/core/config.py:375` | `distance_tol_mm` 150 lost the clamp to the accurate depth band; `test_generate_refuses_when_too_far` is red — **= audit A2; execute the OTHER plan's Task 1** |
| 5 | `tasni/modules/scan/service.py:239` | `_plane_rms_mm` returns `float("nan")` → bare NaN kills the browser's `JSON.parse` on `/ws` and 500s FastAPI responses (`allow_nan=False`) |
| 6 | `server/server_unicast_syncronous.py:646` | `RS_LASER_POWER` defaults to 300 vs the 150 the 2026-08-13 depth characterization was measured under — every restart silently invalidates the dated envelope |
| 7 | `server/server_unicast_syncronous.py:645` | Bare `int()`/`float()` on env vars at import; a typo (`RS_VISUAL_PRESET=high_accuracy`) crash-loops the `realsense-camera` service forever (Restart=always, no start limit) |
| 8 | `tasni/modules/extrusion/service.py:187` | The inspection-candidate sweep restarts at straight-down EVERY layer, re-paying a collision-ON `program.Update()` on the 117 MB station per rejected candidate per layer |

## Global Constraints

- Branch: `calibration-improvements`. **Commit + push after every task** (CLAUDE.md working agreement — unpushed work is invisible to the user).
- **Never run the full pytest suite** (user instruction — too slow). Run only the named test files / `-k` selections given in each task.
- Python is invoked as `py -3.10`.
- Any task touching `server/` must end with a Jetson deploy: `py -3.10 tools/jetson_deploy.py deploy`, then check `status`/`logs`, and the task summary must mention the deploy result.
- Line numbers below are as of commit `89d836f`; re-locate by the quoted code if they have drifted.
- Commit messages: end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

## File Structure

- `tasni/core/rdk_io.py` — gains a cached active-frame→base transform, applied inside `_solve_ik` (Task 1).
- `tasni/core/config.py` — one new `ScanConfig` knob `boundary_center_tol_mm` (Task 2).
- `tasni/modules/scan/service.py` — centroid corroboration in `_corners_from_boundary_on_plane` (Task 2); new `_survey_with_vision_boundary` helper + call-site (Task 3); `_plane_rms_mm` → `None`, new `_finite_or_none` (Task 5).
- `tasni/modules/scan/survey_contract.py` — `CaptureRecord.plane_rms_mm: float | None` (Task 5).
- `server/server_unicast_syncronous.py` — `_env_number` parser, laser default → leave-alone (Task 6).
- `tasni/modules/extrusion/inspection.py` — pure `order_candidates_seed_first` (Task 7).
- `tasni/modules/extrusion/service.py` — `seed_pose` threading through `_build_inspection_move` and both jobs (Task 7).
- New tests: `tests/test_rdk_io_frames.py`, `tests/test_server_env.py`; additions to `tests/test_scan_job.py`, `tests/test_extrusion.py`, `tests/test_extrusion_job.py`.
- No frontend changes: `Scan.tsx` `fmtMm` (line 1575) and `SurveyPanel.tsx` `fmt` (line 385) already guard with `Number.isFinite`, so JSON `null` renders as "—".

---

### Task 1: Solve inspection IK in the robot base frame (Finding 1)

The cylinder's "centred by construction" guarantee currently dies at the RoboDK boundary: `pose_from_aim` builds the camera pose in the WORK frame, but `_solve_ik` passes it to `robolink.SolveIK`, whose client-side math is base-frame-only. `use_camera_tool` (calibration) happens to activate the base frame, which is why calibration is unaffected; `use_named_tool_frame` (extrusion inspection + `extrusion_reachability_report`) is not. Fix at `_solve_ik` so every caller inherits it: cache the active frame's pose w.r.t. the robot base at activation time (an RPC per activation, not per solve) and left-multiply before solving. Targets keep storing the WORK-frame pose (their parent is the work frame) — only the IK input converts.

**Files:**
- Modify: `tasni/core/rdk_io.py` (`__init__` ~63, `use_tool_and_frame` ~76–97, `use_camera_tool` ~99–120, `_solve_ik` ~211–228, `use_named_tool_frame` ~1053–1068)
- Create: `tests/test_rdk_io_frames.py`

**Interfaces:**
- Consumes: `RdkIO`, `pose_to_T`, `invert_T` (module-level in `rdk_io.py`).
- Produces: `RdkIO._frame_wrt_base_T: np.ndarray | None` (None ⇒ active frame IS the base) and `RdkIO._frame_pose_wrt_base(frame) -> np.ndarray`. `_solve_ik`'s contract becomes what its docstring already claims: `T` is in the ACTIVE reference frame.

- [x] **Step 1: Write the failing tests**

Create `tests/test_rdk_io_frames.py`:

```python
"""SolveIK must receive poses in the ROBOT BASE frame, whatever frame is active.

robolink's SolveIK does its pose math client-side against the robot base
("pose must be the robot flange with respect to the robot base unless you
provide the tool and/or reference"); it ignores the station's active reference
frame. use_named_tool_frame() activates an arbitrary work frame, so poses built
in that frame (extrusion inspection, reachability sampling) must be converted
before the solve — otherwise the joint target places the camera translated by
the full base->work offset.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import robodk.robomath as robomath
import robolink as rl

from tasni.core.config import RoboDKConfig
from tasni.core.rdk_io import RdkIO, pose_to_T


class FakeTool:
    def Valid(self): return True
    def PoseTool(self): return robomath.xyzrpw_2_pose([0, 0, 150, 0, 0, 0])


class FakeFrame:
    def __init__(self, pose_abs): self._pose = pose_abs
    def Valid(self): return True
    def Type(self): return rl.ITEM_TYPE_FRAME
    def PoseAbs(self): return self._pose


class FakeTarget:
    def setPose(self, pose): self.pose = pose
    def setAsJointTarget(self): self.joint_target = True
    def setJoints(self, joints): self.joints = joints


class FakeMissing:
    def Valid(self): return False


class FakeRobot:
    def __init__(self, base): self._base = base; self.ik_poses = []
    def Parent(self): return self._base
    def setPoseTool(self, tool): pass
    def setPoseFrame(self, frame): pass
    def Joints(self): return robomath.Mat([0.0] * 6)
    def setJoints(self, joints): pass
    def SolveIK(self, pose, joints_approx=None, tool=None, reference=None):
        self.ik_poses.append(pose)
        return robomath.Mat([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])


class FakeRdk:
    def __init__(self, robot, items): self._robot = robot; self._items = items
    def Item(self, name, itemtype=0):
        return self._items.get((name, itemtype),
                               self._items.get(name, FakeMissing()))
    def AddTarget(self, name, frame, robot): return FakeTarget()


def _cell(base_xyz, frame_xyz):
    """Pure-translation frames so the expected pose needs no Euler conventions."""
    base = FakeFrame(robomath.xyzrpw_2_pose(list(base_xyz) + [0, 0, 0]))
    robot = FakeRobot(base)
    frame = FakeFrame(robomath.xyzrpw_2_pose(list(frame_xyz) + [0, 0, 0]))
    cfg = RoboDKConfig()
    items = {cfg.robot_name: robot,
             ("Realsense", rl.ITEM_TYPE_TOOL): FakeTool(),
             ("Work", rl.ITEM_TYPE_FRAME): frame}
    return RdkIO(SimpleNamespace(rdk=FakeRdk(robot, items), config=cfg)), robot


def test_named_frame_pose_reaches_solveik_in_base_coords():
    io, robot = _cell(base_xyz=(0, 0, 500), frame_xyz=(1000, 200, 500))
    io.use_named_tool_frame("Realsense", "Work")
    T_work = np.eye(4); T_work[:3, 3] = [10.0, 20.0, 300.0]

    joints = io.solve_joints_for_pose(T_work)

    assert joints is not None
    got = pose_to_T(robot.ik_poses[0])
    expected = np.eye(4); expected[:3, 3] = [1010.0, 220.0, 300.0]
    np.testing.assert_allclose(got, expected, atol=1e-9)


def test_base_frame_pose_is_passed_through_unchanged():
    io, robot = _cell(base_xyz=(0, 0, 500), frame_xyz=(1000, 200, 500))
    io.use_camera_tool("Realsense")     # adopts the robot's BASE frame
    T_base = np.eye(4); T_base[:3, 3] = [400.0, 0.0, 800.0]
    io.solve_joints_for_pose(T_base)
    np.testing.assert_allclose(pose_to_T(robot.ik_poses[0]), T_base, atol=1e-9)


def test_create_inspection_target_solves_in_base_but_stores_work_pose():
    io, robot = _cell(base_xyz=(0, 0, 0), frame_xyz=(800, -300, 0))
    T_work = np.eye(4); T_work[:3, 3] = [0.0, 0.0, 300.0]
    made = io.create_inspection_target(
        name="T1", T=T_work, inspection_tool="Realsense", work_frame="Work")
    assert made["created"] is True
    solved = pose_to_T(robot.ik_poses[0])
    np.testing.assert_allclose(solved[:3, 3], [800.0, -300.0, 300.0], atol=1e-9)
```

- [x] **Step 2: Run to verify they fail**

Run: `py -3.10 -m pytest tests/test_rdk_io_frames.py -q`
Expected: 2 FAIL (`test_named_frame_pose…` and `test_create_inspection_target…` — the pose reaches SolveIK untranslated), 1 PASS (base-frame pass-through already works).

- [x] **Step 3: Implement the conversion**

In `tasni/core/rdk_io.py` `__init__`, directly under the `self._tool_pose: np.ndarray | None = None` line (~63):

```python
        # Pose of the ACTIVE reference frame w.r.t. the robot base, cached at
        # tool/frame activation. None = the active frame IS the base. SolveIK's
        # pose math is client-side against the robot base and ignores the
        # station's active frame, so _solve_ik multiplies this in first.
        self._frame_wrt_base_T: np.ndarray | None = None
```

Add a private method right below `use_named_tool_frame` (~1068):

```python
    def _frame_pose_wrt_base(self, frame) -> np.ndarray:
        """Pose of ``frame`` w.r.t. the robot's base frame, via station-absolute
        poses — correct wherever either sits in the station tree."""
        import robolink

        base = self.robot().Parent()
        base_T = (pose_to_T(base.PoseAbs())
                  if base.Valid() and base.Type() == robolink.ITEM_TYPE_FRAME
                  else np.eye(4))
        return invert_T(base_T) @ pose_to_T(frame.PoseAbs())
```

Wire the cache at every activation site:
- `use_named_tool_frame` (~1067): after `self._frame = frame`, add `self._frame_wrt_base_T = self._frame_pose_wrt_base(frame)`.
- `use_camera_tool` (~116/118): add `self._frame_wrt_base_T = None` in BOTH branches (base adopted, and the `self._frame = None` fallback).
- `use_tool_and_frame` (~88): after `self._frame = None` add `self._frame_wrt_base_T = None`; inside the frame-adoption branch (~95) after `self._frame = frame` add `self._frame_wrt_base_T = self._frame_pose_wrt_base(frame)`.

In `_solve_ik` (~219), first statement after the docstring:

```python
        if self._frame_wrt_base_T is not None:
            T = self._frame_wrt_base_T @ np.asarray(T, dtype=float)
```

and amend its docstring's first sentence to: *"``T`` is the pose of the camera (the last-activated tool's TCP) in the **active reference frame**; it is converted to robot-base coordinates here (``_frame_wrt_base_T``) because SolveIK's client-side math is base-frame-only."*

- [x] **Step 4: Run the new tests**

Run: `py -3.10 -m pytest tests/test_rdk_io_frames.py -q`
Expected: 3 passed.

- [x] **Step 5: Bounded regression sweep** (RdkIO consumers that build it with fakes)

Run: `py -3.10 -m pytest tests/test_collision_guard.py tests/test_pose_generation.py tests/test_extrusion_job.py -q`
Expected: all pass (calibration/collision paths activate the base frame or never activate one — `_frame_wrt_base_T` stays None).

- [x] **Step 6: Commit + push**

```bash
git add tasni/core/rdk_io.py tests/test_rdk_io_frames.py
git commit -m "Solve IK in the robot base frame whatever reference frame is active"
git push
```

---

### Task 2: Corroborate the vision rectangle's POSITION, not only its lengths (Finding 2)

`_corners_from_boundary_on_plane` accepts the vision rectangle if each sorted side length is within `[-5, +60]` mm of the depth rectangle's. Equal lengths cannot see a lateral shift (gain a 30 mm shadow band on one edge, lose a 30 mm sliver on the opposite one), so shifted corners rebuild the work frame ~30 mm off with `boundary_source="vision"` and no warning. Add an in-plane centroid-offset gate.

**Files:**
- Modify: `tasni/core/config.py` (~423, next to `boundary_shrink_tol_mm`)
- Modify: `tasni/modules/scan/service.py` (`_corners_from_boundary_on_plane`, after the per-axis loop ending ~1151, before `info["boundary_source"] = "vision"`)
- Test: `tests/test_scan_job.py`

**Interfaces:**
- Consumes: existing locals `corners` (vision rectangle, mm), `n` (unit plane normal), `survey.corners_cam_mm`, `info` dict.
- Produces: `ScanConfig.boundary_center_tol_mm: float = 30.0`; `info["center_offset_mm"]` (rounded float) on every corroboration that reaches the centroid check.

- [x] **Step 1: Write the failing tests**

Append to `tests/test_scan_job.py` (it already imports `numpy as np` and `from tasni.modules.scan import service as scan_service`):

```python
def _corroboration_scene(shift_x_mm: float):
    from types import SimpleNamespace
    K = np.array([[600.0, 0.0, 640.0], [0.0, 600.0, 360.0], [0.0, 0.0, 1.0]])
    camera_cfg = SimpleNamespace(K=K, dist=np.zeros(5), size=(1280, 720))
    survey = SimpleNamespace(
        normal_cam=np.array([0.0, 0.0, -1.0]),
        centroid_cam_mm=np.array([0.0, 0.0, 500.0]),
        extent_mm=(300.0, 200.0),
        corners_cam_mm=np.array([[-150.0, -100.0, 500.0], [150.0, -100.0, 500.0],
                                 [150.0, 100.0, 500.0], [-150.0, 100.0, 500.0]]))
    xs = np.array([-150.0, 150.0, 150.0, -150.0]) + shift_x_mm
    ys = np.array([-100.0, -100.0, 100.0, 100.0])
    u = (xs / 500.0 * 600.0 + 640.0) / 1280.0
    v = (ys / 500.0 * 600.0 + 360.0) / 720.0
    return np.column_stack([u, v]), survey, camera_cfg


def test_vision_boundary_rejected_when_laterally_shifted():
    """Review finding: side lengths were the ONLY corroboration, so a shifted
    segmentation (same extent, moved centre) replaced the work-frame corners
    ~40 mm off under a green lock."""
    from tasni.core.config import ScanConfig
    polygon_uv, survey, camera_cfg = _corroboration_scene(shift_x_mm=40.0)
    corners, info = scan_service._corners_from_boundary_on_plane(
        polygon_uv, survey, camera_cfg, ScanConfig())
    assert corners is None
    assert "centre" in info["reason"]
    assert info["boundary_source"] == "depth"


def test_vision_boundary_accepted_when_centred():
    from tasni.core.config import ScanConfig
    polygon_uv, survey, camera_cfg = _corroboration_scene(shift_x_mm=5.0)
    corners, info = scan_service._corners_from_boundary_on_plane(
        polygon_uv, survey, camera_cfg, ScanConfig())
    assert corners is not None
    assert info["boundary_source"] == "vision"
    assert info["center_offset_mm"] <= 6.0
```

- [x] **Step 2: Run to verify the reject case fails**

Run: `py -3.10 -m pytest tests/test_scan_job.py -q -k "vision_boundary"`
Expected: `rejected_when_laterally_shifted` FAILS (corners are returned today), `accepted_when_centred` FAILS on the missing `center_offset_mm` key.

- [x] **Step 3: Add the knob**

`tasni/core/config.py`, directly under `boundary_shrink_tol_mm` (~423):

```python
    boundary_center_tol_mm: float = 30.0      # ...and its centre may shift laterally by at most this
```

- [x] **Step 4: Add the centroid gate**

In `_corners_from_boundary_on_plane`, after the per-axis length loop (the `for axis, (v, d) in enumerate(zip(vis, dep)):` block) and before `info["boundary_source"] = "vision"`:

```python
    center_tol = float(getattr(scfg, "boundary_center_tol_mm", 30.0))
    depth_corners = getattr(survey, "corners_cam_mm", None)
    if depth_corners is not None:
        # Lengths alone cannot see a lateral shift (gain a shadow band on one
        # edge, lose a sliver on the opposite one: same extent, moved corners).
        offset = (np.asarray(corners, dtype=float).mean(axis=0)
                  - np.asarray(depth_corners, dtype=float).mean(axis=0))
        offset -= float(offset @ n) * n          # in-plane component only
        shift = float(np.linalg.norm(offset))
        info["center_offset_mm"] = round(shift, 1)
        if shift > center_tol:
            info["reason"] = (
                f"vision rectangle centre is {shift:.0f} mm from the depth "
                f"rectangle centre (limit {center_tol:.0f}) — the segmentation "
                "drifted sideways, not outward")
            return None, info
```

Also append one sentence to the function docstring's corroboration paragraph: *"The centre is corroborated too (``boundary_center_tol_mm``): equal side lengths cannot see a lateral shift."*

- [x] **Step 5: Run the new tests + the file**

Run: `py -3.10 -m pytest tests/test_scan_job.py -q`
Expected: all pass.

- [x] **Step 6: Commit + push**

```bash
git add tasni/core/config.py tasni/modules/scan/service.py tests/test_scan_job.py
git commit -m "Corroborate the vision boundary's centre, not only its side lengths"
git push
```

---

### Task 3: Propagate the vision extent to the planner and gate payload (Finding 3)

At the hybrid-extent lock (`service.py` ~611) only `survey.corners_cam_mm` is replaced. `plan_scan` sizes the tour from `survey.extent_mm` (`planner.py:101`, the survey stored in `LockedScanSurface` and passed at `service.py:2585`), so it tours the ~20 mm/edge-smaller depth rectangle the hybrid extent exists to escape — and `gate_payload["extent_mm"]` (built at ~459 before the swap) contradicts `record.size_mm` in the same payload. Swap corners AND extent together, and refresh the payload. (The crop path needs nothing: its planner branch uses the declared `crop_size_mm`, not `extent_mm`.)

**Files:**
- Modify: `tasni/modules/scan/service.py` (new helper after `_corners_from_boundary_on_plane` ~1155; call site ~610–611)
- Test: `tests/test_scan_job.py`

**Interfaces:**
- Consumes: `dataclasses.replace` (already imported in `service.py`), `boundary_info["vision_extent_mm"]` — `[longer, shorter]`, the SAME convention as `SurveyMeasurement.extent_mm` ("(longer, shorter)", `survey.py:56`).
- Produces: `_survey_with_vision_boundary(survey, vis_corners, info) -> survey` (module level in `service.py`).

- [x] **Step 1: Write the failing test**

Append to `tests/test_scan_job.py`:

```python
def test_vision_boundary_updates_extent_for_the_planner():
    """Review finding: only corners_cam_mm was replaced at the hybrid lock;
    plan_scan reads survey.extent_mm, so the tour was sized from the smaller
    depth rectangle and the gate payload contradicted the record."""
    from dataclasses import dataclass

    @dataclass
    class _S:
        corners_cam_mm: object
        extent_mm: object

    survey = _S(corners_cam_mm=None, extent_mm=(280.0, 200.0))
    corners = np.zeros((4, 3))
    out = scan_service._survey_with_vision_boundary(
        survey, corners, {"vision_extent_mm": [320.0, 240.0]})
    assert out.extent_mm == (320.0, 240.0)
    assert out.corners_cam_mm is corners
    out2 = scan_service._survey_with_vision_boundary(survey, corners, {})
    assert out2.extent_mm == (280.0, 200.0)
```

- [x] **Step 2: Run to verify it fails**

Run: `py -3.10 -m pytest tests/test_scan_job.py -q -k "updates_extent"`
Expected: FAIL — `_survey_with_vision_boundary` does not exist.

- [x] **Step 3: Add the helper**

In `service.py`, directly after `_corners_from_boundary_on_plane`:

```python
def _survey_with_vision_boundary(survey, vis_corners, info):
    """Adopt the corroborated vision boundary onto the survey WHOLE: corners AND
    extent together. plan_scan sizes the tour from ``survey.extent_mm`` while
    the record/work frame derive from the corners — swapping only the corners
    left the planner touring the ~20 mm/edge-smaller depth rectangle that the
    hybrid extent exists to escape, and the gate payload contradicting the
    record. ``vision_extent_mm`` is (longer, shorter), the same convention as
    ``SurveyMeasurement.extent_mm``."""
    vis_extent = info.get("vision_extent_mm")
    extent = (tuple(float(v) for v in vis_extent)
              if vis_extent is not None else survey.extent_mm)
    return replace(survey, corners_cam_mm=vis_corners, extent_mm=extent)
```

- [x] **Step 4: Use it at the lock**

Replace the two lines at ~610–611:

```python
                if vis_corners is not None:
                    survey = replace(survey, corners_cam_mm=vis_corners)
```

with:

```python
                if vis_corners is not None:
                    survey = _survey_with_vision_boundary(
                        survey, vis_corners, boundary_info)
                    gate_payload["extent_mm"] = (
                        list(survey.extent_mm)
                        if survey.extent_mm is not None else None)
```

- [x] **Step 5: Run the tests**

Run: `py -3.10 -m pytest tests/test_scan_job.py tests/test_scan_planner.py -q`
Expected: all pass.

- [x] **Step 6: Commit + push**

```bash
git add tasni/modules/scan/service.py tests/test_scan_job.py
git commit -m "Hybrid lock: adopt the vision extent for the planner, not only the corners"
git push
```

---

### Task 4: Clamp the standoff gate to the accurate band (Finding 4 = audit A2)

This finding is already fully specced — with code, a currently-red test, and commit steps — as **Task 1 of `docs/superpowers/plans/2026-08-14-scan-chain-fixes.md`** ("Clamp the standoff accept window to the accurate band (A2)").

- [x] **Step 1: Open that plan and execute its Task 1, Steps 1–8, exactly as written** (helper `standoff_accept_window_mm`, both gate sites, the two tests, commit + push). Do not re-derive anything here.

- [x] **Step 2: Confirm the previously red test is green**

Run: `py -3.10 -m pytest tests/test_scan_job.py -q -k "too_far or clamped_to_accurate"`
Expected: 2 passed.

---

### Task 5: `plane_rms_mm` may never be NaN (Finding 5)

`_plane_rms_mm` returns `float("nan")` on sample starvation (reachable: small/distant surface after the 3 %-shrunk-outline mask and post-band floor) or a RANSAC failure. The NaN flows into `CaptureRecord`/`quality` and both serialization paths: starlette's `send_json` uses `json.dumps` without `allow_nan=False` → bare `NaN` that the browser's `JSON.parse` rejects (killing `/ws` gate handling), and FastAPI's HTTP render raises (`allow_nan=False` → 500 on lock/insert/finish). Return `None`, and guard the payload builders. The frontend is already null-safe (`fmtMm`/`fmt` use `Number.isFinite`).

**Files:**
- Modify: `tasni/modules/scan/service.py` (`_plane_rms_mm` ~209–252; `_survey_record_from_lock` ~276 + ~279)
- Modify: `tasni/modules/scan/survey_contract.py` (~129)
- Test: `tests/test_scan_job.py`

**Interfaces:**
- Produces: `_plane_rms_mm(...) -> float | None`; `_finite_or_none(value) -> float | None` (module level in `service.py`); `CaptureRecord.plane_rms_mm: float | None`.

- [x] **Step 1: Write the failing tests**

Append to `tests/test_scan_job.py`:

```python
def test_plane_rms_none_when_starved_never_nan():
    """Review finding: NaN here kills the /ws JSON on the client and 500s the
    lock/insert responses (FastAPI renders with allow_nan=False)."""
    K = np.array([[600.0, 0.0, 32.0], [0.0, 600.0, 32.0], [0.0, 0.0, 1.0]])
    assert scan_service._plane_rms_mm(np.zeros((64, 64)), K) is None


def test_finite_or_none_guards_payload_metrics():
    assert scan_service._finite_or_none(float("nan")) is None
    assert scan_service._finite_or_none(float("inf")) is None
    assert scan_service._finite_or_none(1.25) == 1.25
    assert scan_service._finite_or_none(None) is None
```

- [x] **Step 2: Run to verify they fail**

Run: `py -3.10 -m pytest tests/test_scan_job.py -q -k "never_nan or finite_or_none"`
Expected: FAIL — `_plane_rms_mm` returns `nan` (`is None` fails); `_finite_or_none` does not exist.

- [x] **Step 3: Implement**

In `service.py` `_plane_rms_mm`: change the signature to `-> float | None`, replace all three `return float("nan")` with `return None`, and append to the docstring: *"Returns ``None`` (never NaN) when starved of samples or the fit fails — NaN would poison JSON on both client paths."*

Add above `_survey_record_from_lock`:

```python
def _finite_or_none(value) -> float | None:
    """JSON-safe metric guard: NaN/inf must never reach a payload — the /ws
    send_json path emits bare NaN (the browser's JSON.parse rejects the whole
    event) and FastAPI's HTTP render raises (allow_nan=False -> a 500 on
    lock/insert/finish)."""
    if value is None:
        return None
    v = float(value)
    return v if np.isfinite(v) else None
```

In `_survey_record_from_lock`, change line ~276 `plane_rms_mm=float(plane_rms_mm),` to `plane_rms_mm=_finite_or_none(plane_rms_mm),` and the quality entry at ~279 `"plane_rms_mm": float(plane_rms_mm),` to `"plane_rms_mm": _finite_or_none(plane_rms_mm),`.

In `survey_contract.py` line ~129, change `plane_rms_mm: float` to:

```python
    plane_rms_mm: float | None   # None = too few valid samples to fit (never NaN — must stay JSON-safe)
```

- [x] **Step 4: Run the tests + bounded sweep**

Run: `py -3.10 -m pytest tests/test_scan_job.py tests/test_survey_contract.py tests/test_five_position.py -q`
Expected: all pass (`five_position_capture` at `service.py:1333` also builds `CaptureRecord`s from `_plane_rms_mm` and now stores `None` cleanly).

- [x] **Step 5: Commit + push**

```bash
git add tasni/modules/scan/service.py tasni/modules/scan/survey_contract.py tests/test_scan_job.py
git commit -m "plane_rms_mm: None instead of NaN so /ws and HTTP JSON stay parseable"
git push
```

---

### Task 6: Jetson env parsing must not crash the service; laser power leaves the device alone (Findings 6 + 7)

Two defects at the same site. (a) `RS_VISUAL_PRESET`/`RS_LASER_POWER` are parsed with bare `int()`/`float()` at module import — a typo like `RS_VISUAL_PRESET=high_accuracy` (a name the adjacent comment itself displays) raises before the socket binds, and the unit's `Restart=always` + no start limit crash-loops the camera forever, headless. (b) `RS_LASER_POWER` defaults to 300, silently doubling projector power vs the 150 the 2026-08-13 depth characterization was measured under — the exact invariant commit `9ba798c` cited when it kept the PRESET at leave-alone. Make the laser default leave-alone too, with the same read-back log line.

**Files:**
- Modify: `server/server_unicast_syncronous.py` (~627–646 parse block; `set_high_accuracy_preset` ~669–684)
- Create: `tests/test_server_env.py`

**Interfaces:**
- Produces: `_env_number(name: str, default: float) -> float` (module level, defined ABOVE the two parse lines); `RS_LASER_POWER` default `-1.0` (= leave alone).

- [x] **Step 1: Write the failing tests**

Create `tests/test_server_env.py` (same import scaffolding as `tests/test_scan_telemetry_server.py`):

```python
"""Import-time env parsing on the Jetson server must never crash the service."""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
sys.modules.setdefault("pyrealsense2", SimpleNamespace())
sys.modules.setdefault("turbojpeg", SimpleNamespace())

import server.server_unicast_syncronous as srv  # noqa: E402


def test_env_number_falls_back_on_garbage():
    os.environ["X_TEST_NUM"] = "high_density"
    try:
        assert srv._env_number("X_TEST_NUM", -1.0) == -1.0
    finally:
        del os.environ["X_TEST_NUM"]
    assert srv._env_number("X_TEST_ABSENT", 7.5) == 7.5


def test_typoed_preset_name_does_not_kill_the_import(monkeypatch):
    """The unit is Restart=always with no start limit — an import-time
    ValueError becomes an infinite crash-loop with the camera dark for every
    module (scan, calibration, extrusion inspection)."""
    monkeypatch.setenv("RS_VISUAL_PRESET", "high_accuracy")
    monkeypatch.setenv("RS_LASER_POWER", "please")
    try:
        importlib.reload(srv)
        assert srv.RS_VISUAL_PRESET == -1
        assert srv.RS_LASER_POWER == -1.0
    finally:
        monkeypatch.undo()
        importlib.reload(srv)   # restore module state for other test files


def test_laser_power_defaults_to_leave_alone():
    """Review finding: a default of 300 doubled projector power vs the
    configuration the 2026-08-13 depth characterization was measured under."""
    assert srv.RS_LASER_POWER == -1.0
```

- [x] **Step 2: Run to verify they fail**

Run: `py -3.10 -m pytest tests/test_server_env.py -q`
Expected: `_env_number` AttributeError; the reload test raises ValueError; the default test sees 300.0.

- [x] **Step 3: Implement the parser + defaults**

In `server/server_unicast_syncronous.py`, replace lines ~645–646:

```python
RS_VISUAL_PRESET = int(os.environ.get('RS_VISUAL_PRESET', '-1'))
RS_LASER_POWER = float(os.environ.get('RS_LASER_POWER', '300'))
```

with:

```python
def _env_number(name: str, default: float) -> float:
    """A numeric env override, or ``default`` when unset OR unparsable. A
    typo'd value must never take the service down: the unit is Restart=always
    with no start limit, so an import-time ValueError becomes an infinite
    crash-loop with the camera dark for every module."""
    raw = os.environ.get(name)
    if raw is None:
        return float(default)
    try:
        return float(raw)
    except ValueError:
        print(f"WARNING: {name}={raw!r} is not a number — using {default:g}",
              flush=True)
        return float(default)


RS_VISUAL_PRESET = int(_env_number('RS_VISUAL_PRESET', -1))
# -1 = leave the device's current laser power alone. The 2026-08-13 depth
# characterization was measured at the device's 150; a silent default of 300
# doubled projector power on every restart and invalidated that dated envelope
# (the same invariant that keeps RS_VISUAL_PRESET at leave-alone). Trial higher
# power as an explicit experiment with its own before/after measurement.
RS_LASER_POWER = _env_number('RS_LASER_POWER', -1.0)
```

Also update the block comment above (~636–637): delete the two sentences claiming more projected texture "is the default now" (they described the 300 default) and leave the measurement description.

- [x] **Step 4: Gate the laser write like the preset write**

In `set_high_accuracy_preset`, replace the `wanted = [...]` assignment (~669–674) with:

```python
    wanted = [
        # Every client of this server shares one pipeline, so enabling the
        # emitter here covers scan, calibration, extrusion/cylinder measuring
        # and the live preview alike — there is no per-feature IR state to
        # keep in sync, and nothing anywhere turns it back off.
        ('emitter_enabled', getattr(rs.option, 'emitter_enabled', None), 1.0)]
    if RS_LASER_POWER >= 0:
        wanted.insert(0, ('laser_power', getattr(rs.option, 'laser_power', None),
                          RS_LASER_POWER))
    else:
        try:
            cur = sensor.get_option(rs.option.laser_power)
            print(f"RealSense: laser_power left as-is at {cur:g} "
                  "(set RS_LASER_POWER to change it)", flush=True)
        except Exception:
            pass
```

(The existing `RS_VISUAL_PRESET >= 0` block below it stays exactly as is.)

- [x] **Step 5: Run the tests**

Run: `py -3.10 -m pytest tests/test_server_env.py tests/test_scan_telemetry_server.py -q`
Expected: all pass.

- [x] **Step 6: Commit + push + deploy to the Jetson**

```bash
git add server/server_unicast_syncronous.py tests/test_server_env.py
git commit -m "Jetson server: tolerant env parsing; laser power defaults to leave-alone"
git push
py -3.10 tools/jetson_deploy.py deploy
py -3.10 tools/jetson_deploy.py logs
```

Expected in the logs after restart: `RealSense: laser_power left as-is at …` (and the existing `visual_preset left as-is` line). Mention the deploy result in the task summary.

---

### Task 7: Try last layer's winning inspection pose first (Finding 8)

`_build_inspection_move` restarts the candidate sweep at straight-down every layer, so a rejection that is constant across layers (straight-down collides with the same fixture at every height) re-pays a collision-ON `program.Update()` on the 117 MB station per candidate, per layer, in the live print loop while the bead cools. Reorder only: the previous layer's validated winner goes first; collision validation still gates every layer. (Hoisting the per-candidate `use_named_tool_frame`/`Joints()` RPCs and rewriting the target via `setPose`/`setJoints` instead of delete+rebuild are deliberately NOT in this task — they are a larger `rdk_io` refactor for a fraction of the win.)

**Files:**
- Modify: `tasni/modules/extrusion/inspection.py` (append after `pose_candidates`)
- Modify: `tasni/modules/extrusion/service.py` (`_build_inspection_move` ~149–187; the two call sites ~286 and ~456 and their layer loops)
- Test: `tests/test_extrusion.py`, `tests/test_extrusion_job.py`

**Interfaces:**
- Produces: `order_candidates_seed_first(candidates: list[dict], seed: dict | None) -> list[dict]` in `inspection.py`; `_build_inspection_move(..., seed_pose: dict | None = None)`.
- Consumes: the winner descriptor already returned as `inspect["pose"]` (contains `tilt_deg`/`azimuth_deg`/`roll_deg`).

- [x] **Step 1: Write the failing pure-helper test**

Append to `tests/test_extrusion.py`:

```python
def test_seed_first_reordering_moves_last_winner_to_front():
    from tasni.modules.extrusion.inspection import order_candidates_seed_first
    candidates = [{"tilt_deg": 0.0, "azimuth_deg": 0.0, "roll_deg": r}
                  for r in (0.0, 90.0, 180.0)]
    seed = {"tilt_deg": 0.0, "azimuth_deg": 0.0, "roll_deg": 90.0}
    out = order_candidates_seed_first(candidates, seed)
    assert [c["roll_deg"] for c in out] == [90.0, 0.0, 180.0]
    assert order_candidates_seed_first(candidates, None) == candidates
    unknown = {"tilt_deg": 5.0, "azimuth_deg": 0.0, "roll_deg": 45.0}
    assert order_candidates_seed_first(candidates, unknown) == candidates
```

- [x] **Step 2: Write the failing job-level test**

Append to `tests/test_extrusion_job.py`:

```python
def test_auto_inspection_reuses_the_previous_layers_winner(tmp_path, monkeypatch):
    """Review finding: the candidate sweep restarted at straight-down every
    layer, re-paying a collision-ON program Update per rejected candidate per
    layer. The previous layer's validated winner is tried first now."""
    svc, rdk, camera = services(tmp_path)
    monkeypatch.setattr(service_mod, "new_run_dir",
                        lambda module, stamp: _mkdir(tmp_path / module / stamp))
    rdk.bad_inspections = 1        # layer 1: straight-down fails validation once
    output = CylinderDryRunJob(svc, plan(layers=2, auto_inspection=True))(Ctx())
    first = output["layers"][0]["inspection_pose"]
    second = output["layers"][1]["inspection_pose"]
    assert first["roll_deg"] != 0.0          # layer 1 fell through to a roll
    assert (second["tilt_deg"], second["azimuth_deg"], second["roll_deg"]) == \
        (first["tilt_deg"], first["azimuth_deg"], first["roll_deg"])
    assert second["rejected"] == []          # no wasted candidates on layer 2
```

- [x] **Step 3: Run to verify both fail**

Run: `py -3.10 -m pytest tests/test_extrusion.py -q -k seed_first` and `py -3.10 -m pytest tests/test_extrusion_job.py -q -k reuses_the_previous`
Expected: import error on `order_candidates_seed_first`; the job test fails on the pose-equality assert (layer 2 re-chooses straight-down today).

- [x] **Step 4: Implement the helper**

Append to `tasni/modules/extrusion/inspection.py`:

```python
def order_candidates_seed_first(candidates: list[dict], seed: dict | None) -> list[dict]:
    """Move the candidate matching ``seed``'s (tilt, azimuth, roll) to the front.

    Rejections are usually constant across layers (straight-down collides with
    the same fixture at every height), so the previous layer's winner is by far
    the most likely first pass — trying it first collapses the per-layer search
    to one collision-validated attempt. Validation still gates every layer;
    only the search ORDER changes. Absent/unknown seeds return the list as-is.
    """
    if not seed:
        return candidates
    key = (float(seed.get("tilt_deg", 0.0)), float(seed.get("azimuth_deg", 0.0)),
           float(seed.get("roll_deg", 0.0)))
    for index, candidate in enumerate(candidates):
        if (float(candidate["tilt_deg"]), float(candidate["azimuth_deg"]),
                float(candidate["roll_deg"])) == key:
            return [candidate] + candidates[:index] + candidates[index + 1:]
    return candidates
```

- [x] **Step 5: Thread the seed through the service**

In `tasni/modules/extrusion/service.py`:

1. Extend the existing `from .inspection import …` line with `order_candidates_seed_first`.
2. `_build_inspection_move` signature (~149) gains a trailing keyword `seed_pose: dict | None = None`, and its docstring a final sentence: *"``seed_pose`` (the previous layer's winning descriptor) is tried first — see ``order_candidates_seed_first``."*
3. The loop head (~187) `for candidate in pose_candidates(aim, framing["standoff_mm"], config):` becomes:

```python
    candidates = order_candidates_seed_first(
        pose_candidates(aim, framing["standoff_mm"], config), seed_pose)
    for candidate in candidates:
```

4. In BOTH jobs (`CylinderDryRunJob.__call__` and `CylinderPrintJob`'s layer loop): add `last_inspection_pose: dict | None = None` immediately before the `for … layer …` loop; pass `seed_pose=last_inspection_pose` in the `_build_inspection_move(...)` call (~286 and ~456); and immediately after each call add:

```python
                if inspect["pose"]:
                    last_inspection_pose = inspect["pose"]
```

- [x] **Step 6: Run the tests**

Run: `py -3.10 -m pytest tests/test_extrusion.py tests/test_extrusion_job.py -q`
Expected: all pass (including the existing auto-inspection tests — layer 1 ordering is unchanged because the seed starts as None).

- [x] **Step 7: Commit + push**

```bash
git add tasni/modules/extrusion/inspection.py tasni/modules/extrusion/service.py tests/test_extrusion.py tests/test_extrusion_job.py
git commit -m "Inspection search: try last layer's validated winner first"
git push
```

---

## Self-review notes

- **Coverage:** finding 1 → Task 1; 2 → Task 2; 3 → Task 3; 4 → Task 4 (delegated to the audit plan's Task 1, which is complete there); 5 → Task 5; 6+7 → Task 6; 8 → Task 7. All 8 accounted for.
- **Ordering:** Task 1 first (safety: a wrong-but-validated joint target on the real cell); Tasks 2–5 are host-only scan-lock correctness; Task 6 changes Jetson device behavior (laser back to the characterized 150) and must be deployed; Task 7 is efficiency-only.
- **Interface consistency:** `_survey_with_vision_boundary`, `_finite_or_none`, `_env_number`, `order_candidates_seed_first`, `_frame_pose_wrt_base`/`_frame_wrt_base_T`, and `seed_pose` are each defined once and used with the same names/signatures in their tests and call sites.
- **Deliberately out of scope:** the per-candidate `use_named_tool_frame`/`Joints()` hoist and target-rewrite refactor (noted in Task 7); the crop-path `extent_mm` (its planner branch uses `crop_size_mm`); the remaining 15 tasks of the 2026-08-14 audit plan.

---

## Execution log (2026-08-25) — all 7 tasks DONE, pushed to `calibration-improvements`

| Task | Finding | Commit |
|------|---------|--------|
| 1 | IK solved in the work frame | `ec0b655` |
| 2 | vision boundary centre uncorroborated | `9e073a1` |
| 3 | hybrid lock dropped the vision extent | `df0ea97` |
| 4 | standoff window unclamped (= audit A2) | `74bfbf8` |
| 5 | `plane_rms_mm` NaN poisons JSON | `14ab296` |
| 6+7 | Jetson env parsing / laser default | `e90fc7d` (deployed) |
| 8 | inspection sweep restarts every layer | `dcd2d86` |

### Deviations from the plan as written

1. **Task 4 gates THREE sites, not two.** `stabilize_live_scan_payload`
   (`service.py`, called from `module.py:725` *after* `live_scan_telemetry_payload`)
   re-derived `gates["distance"]` from the symmetric tolerance and handed the clamp
   straight back on the live path. The plan's test would have passed while production
   stayed broken, because it calls the payload builder directly. It now reuses the
   published `distance_window_mm`. Covered by
   `test_stabilize_does_not_hand_back_the_unclamped_window` (verified genuinely red
   against the pre-fix code).

2. **Task 4 exempts reference mode on the FAR edge** (user-confirmed decision). A
   literal clamp made `test_generate_reference_mode_for_oversized_framed_surface`
   fail *by design*: reference mode triggers when `d_fit > accurate_max_mm` and the
   planner pins its ideal AT `accurate_max_mm`, so a clamped top means the operator
   can never stand far enough back to see the surface — `_reference_locate` would
   have become dead code again, undoing `d4e56bd`. `standoff_accept_window_mm` now
   takes `reference_mode`; the near edge clamps in every mode. `_planned_surface_
   standoff_mm` became `_planned_surface_aim` returning `(standoff_mm, mode)`, and
   the live gate detects the same condition from the *unclipped* framing distance.

3. **Task 5 made one frontend change** the plan said would not be needed. Both
   formatters were indeed already null-safe, but `SurveyReport.plane_rms_mm: number`
   became an inaccurate type once the backend can send `null`, which would let a
   future `.toFixed()` through the compiler. Widened to `number | null` (and `fmt`'s
   parameter with it). `npm run build` clean.

4. **Task 6: the Jetson device still READS 300 mW.** The deploy log shows
   `RealSense: laser_power left as-is at 300 (set RS_LASER_POWER to change it)` —
   the fix stops *future* silent changes, but the 300 written by earlier restarts
   persists in the sensor across service restarts. The 2026-08-13 depth envelope was
   measured at 150. **Open item:** decide whether to restore 150 explicitly
   (`RS_LASER_POWER=150` once, or a camera power-cycle) or to re-measure the envelope
   at 300 and re-date it. Until then the characterization and the running device
   disagree.

### Verification

- 184 tests green across `test_rdk_io_frames`, `test_robot_link`, `test_collision_guard`,
  `test_pose_generation`, `test_survey_contract`, `test_five_position`, `test_scan_planner`,
  `test_scan_depth_gate`, `test_scan_overlay`, `test_scan_telemetry_server`,
  `test_server_env`, `test_extrusion`, `test_extrusion_job`.
- `tests/test_scan_job.py` 82 passed (run separately — ~190 s), including the
  previously-red `test_generate_refuses_when_too_far`.
- `npm run build` clean; Jetson service active + listening on 1024 after deploy.
- Not run: the full pytest suite (user instruction — too slow).

### Still open (not in this plan's scope)

- Tasks 2–16 of `2026-08-14-scan-chain-fixes.md` (Task 1 executed from here).
- All of this is bench-verified only; nothing here has been exercised on the real
  KUKA. Task 1 in particular changes what the robot is commanded to do on the
  extrusion inspection path and wants a dry tour before a live print.
