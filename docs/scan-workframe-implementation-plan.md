# Two-Path Workframe Survey Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the two-path workframe survey design in
[scan-workframe-two-path-plan.md](scan-workframe-two-path-plan.md): one immutable
`LockedWorkframeSurvey` contract feeding review/planner/insertion, a compact-eligibility
classifier, a labeled user-specified fast path, a guided five-position survey for large
surfaces, hard safety gates, and a distance-characterization tool.

**Architecture:** New pure-logic modules (`survey_contract.py`, `classifier.py`,
`rect_fit.py`, `corner_evidence.py`, `five_position.py`, `characterize.py`) carry all
geometry and contracts and are fully unit-tested without hardware. Existing
orchestration (`service.py`, `module.py`, `Scan.tsx`) is wired to produce/consume the
contract. The Jetson `server/` is untouched by this plan.

**Tech Stack:** Python 3.10, numpy, OpenCV (`cv2`), pydantic (config), FastAPI
(routes), pytest; React + TypeScript + Vite for `tasni/webui`.

## Global Constraints

- Spec = `docs/scan-workframe-two-path-plan.md` (rev. 2026-08-12). Section numbers below (§N) refer to it.
- Python: run everything with `py -3.10`. Tests: `py -3.10 -m pytest -q` from the repo root (Windows PowerShell). The full suite (244 tests at plan time) must pass at every commit.
- Tests must run **without hardware** — follow the fake-services pattern at the top of `tests/test_scan_job.py`.
- `tests/conftest.py` pre-imports `onnxruntime` before `robolink`/Qt — never reorder that.
- Do NOT touch `Tasni.rdk`, `macros/`, or `server/` in this plan. Host-only.
- New config keys go in `class ScanConfig` (`tasni/core/config.py:351`) — unknown keys raise `KeyError` (`_merge`, `config.py:671`). Never rename or remove existing keys (`tasni.config.json` compat).
- Units: RoboDK base frame is **mm**; TSDF/Open3D code is meters. All new geometry modules work in **mm** and say so in docstrings.
- Frame convention (§2): origin = rectangle center; +Z = surface normal oriented +Z-up in base; +X = long edge; +Y right-handed; corners `C1..C4` clockwise viewed looking along −Z, `C1` nearest robot base.
- Provenance strings are exact (§2, §6, §7): `camera measured - complete boundary`, `camera measured - five-position boundary survey`, `user specified - plane measured, boundary declared`.
- Commit after every task; **push after every commit** (working agreement in CLAUDE.md). End commit bodies with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Web UI check: `cd tasni/webui; npm run build` must succeed for any task touching `webui`.

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `tasni/modules/scan/survey_contract.py` | create | Immutable capture/survey records, provenance, robot-state refresh, corner ordering, frame construction |
| `tasni/modules/scan/classifier.py` | create | Compact-vs-large eligibility (§6 entry conditions) |
| `tasni/modules/scan/rect_fit.py` | create | Global plane, edge-line TLS, constrained rectangle solve (§7) |
| `tasni/modules/scan/corner_evidence.py` | create | Depth+boundary → base-frame corner/edge evidence points |
| `tasni/modules/scan/five_position.py` | create | Guided five-capture state machine (§7) |
| `tasni/core/characterize.py` | create | Distance-trial metrics + `choose_dstar` (§5, Phase 0) |
| `tools/characterize_distance.py` | create | Operator CLI for the distance sweep |
| `tasni/modules/scan/service.py` | modify | Lock builds the contract; provenance flows to targets/run/insert; pose-liveness; lock token |
| `tasni/modules/scan/module.py` | modify | `/surface/region`, `/survey/*` routes, run-token guard, pose-liveness in live loop |
| `tasni/modules/scan/planner.py` | modify | `plan_rect_tour` for large rectangles |
| `tasni/core/config.py` | modify | New `ScanConfig` keys; `collision_filter_hard_fail` default flip |
| `tasni/webui/src/pages/Scan.tsx` | modify | Locked polygon sole source, provenance chip, advisory lamps, region inputs, survey panel |
| `tasni/webui/src/pages/AimHud.tsx` | modify | `GateReading` fields, pose-liveness styling |
| `tests/test_survey_contract.py` etc. | create | One test file per new module (named per task) |

Milestones: **A** (Tasks 1–8, contract + compact + supersessions — shippable alone),
**B** (Tasks 9–14, five-position survey), **C** (Tasks 15–17, characterization + wrap-up).

---

### Task 1: Survey contract module

**Files:**
- Create: `tasni/modules/scan/survey_contract.py`
- Test: `tests/test_survey_contract.py`

**Interfaces:**
- Consumes: nothing (pure module; numpy only).
- Produces (later tasks import these exact names from `tasni.modules.scan.survey_contract`):
  `PROVENANCE_COMPACT`, `PROVENANCE_FIVE_POSITION`, `PROVENANCE_USER_SPECIFIED`,
  `MODE_COMPACT = "compact"`, `MODE_FIVE_POSITION = "five_position"`,
  `MODE_USER_SPECIFIED = "user_specified"`, `PROVENANCE_BY_MODE: dict`,
  `pose_delta(T_a, T_b) -> tuple[float, float]`,
  `RobotStateSnapshot(joints, camera_T, fetched_at, stationary)` with `.camera_T_np()`,
  `refresh_robot_state(rdk, *, settle_s=0.15, joint_tol_deg=0.01, clock=time.monotonic, sleep=time.sleep) -> RobotStateSnapshot`,
  `CaptureRecord(kind, robot, measurement_ts, captured_at, n_frames, standoff_mm, tilt_deg, valid_frac, plane_rms_mm, plane_normal_base, plane_point_base)`,
  `capture_is_fresh(record, *, now, max_age_s) -> bool`,
  `robot_moved_since(snapshot, current_T, *, trans_tol_mm, rot_tol_deg) -> bool`,
  `order_corners_clockwise(corners_base, normal_base) -> np.ndarray`,
  `frame_from_rectangle(corners_base, normal_base) -> np.ndarray`,
  `camera_calibration_id(camera_cfg) -> str`,
  `LockedWorkframeSurvey(mode, boundary_provenance, captures, plane_normal_base, plane_point_base, corners_base, center_base, frame_T_base, size_mm, quality, calibration_id, locked_robot, locked_at)` with `.corners_np()`, `.frame_np()`, `.to_dict()`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_survey_contract.py
import math
import numpy as np
import pytest

from tasni.modules.scan.survey_contract import (
    MODE_COMPACT, PROVENANCE_BY_MODE, PROVENANCE_COMPACT,
    CaptureRecord, LockedWorkframeSurvey, RobotStateSnapshot,
    camera_calibration_id, capture_is_fresh, frame_from_rectangle,
    order_corners_clockwise, pose_delta, refresh_robot_state, robot_moved_since,
)


def _T(x=0.0, y=0.0, z=0.0, rz_deg=0.0):
    T = np.eye(4)
    a = math.radians(rz_deg)
    T[:2, :2] = [[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]
    T[:3, 3] = [x, y, z]
    return T


def test_pose_delta_translation_and_rotation():
    trans, rot = pose_delta(_T(), _T(x=3.0, rz_deg=2.0))
    assert trans == pytest.approx(3.0)
    assert rot == pytest.approx(2.0, abs=1e-6)


class _FakeRdk:
    def __init__(self, joint_seq, T):
        self._seq = list(joint_seq)
        self._T = T
        self.calls = 0

    def current_joints(self):
        self.calls += 1
        return self._seq.pop(0) if len(self._seq) > 1 else self._seq[0]

    def camera_pose_T(self):
        return self._T


def test_refresh_robot_state_stationary_and_moving():
    still = refresh_robot_state(_FakeRdk([[0, 10, 20, 0, 30, 0]] * 2, _T()), sleep=lambda s: None)
    assert still.stationary is True
    assert still.camera_T_np().shape == (4, 4)
    moving = refresh_robot_state(
        _FakeRdk([[0, 10, 20, 0, 30, 0], [0, 10.5, 20, 0, 30, 0]], _T()), sleep=lambda s: None)
    assert moving.stationary is False


def test_refresh_robot_state_requires_pose():
    with pytest.raises(RuntimeError):
        refresh_robot_state(_FakeRdk([[0] * 6] * 2, None), sleep=lambda s: None)


def _snapshot(T=None):
    return RobotStateSnapshot(joints=(0.0,) * 6,
                              camera_T=tuple(map(tuple, (T if T is not None else _T()))),
                              fetched_at=100.0, stationary=True)


def test_robot_moved_since_tolerances_and_fail_open():
    snap = _snapshot()
    assert robot_moved_since(snap, _T(x=0.5), trans_tol_mm=1.2, rot_tol_deg=0.3) is False
    assert robot_moved_since(snap, _T(x=5.0), trans_tol_mm=1.2, rot_tol_deg=0.3) is True
    assert robot_moved_since(snap, None, trans_tol_mm=1.2, rot_tol_deg=0.3) is True


def _record(captured_at=100.0, stationary=True):
    snap = RobotStateSnapshot((0.0,) * 6, tuple(map(tuple, _T())), captured_at, stationary)
    return CaptureRecord(kind="center", robot=snap, measurement_ts=1.0,
                         captured_at=captured_at, n_frames=5, standoff_mm=350.0,
                         tilt_deg=1.0, valid_frac=0.9, plane_rms_mm=0.8,
                         plane_normal_base=(0, 0, 1), plane_point_base=(0, 0, 0))


def test_capture_freshness():
    assert capture_is_fresh(_record(100.0), now=105.0, max_age_s=10.0) is True
    assert capture_is_fresh(_record(100.0), now=120.0, max_age_s=10.0) is False
    assert capture_is_fresh(_record(100.0, stationary=False), now=101.0, max_age_s=10.0) is False


_RECT = np.array([[0, 0, 0], [1200, 0, 0], [1200, 800, 0], [0, 800, 0]], dtype=float)


def test_corner_ordering_is_deterministic_and_starts_near_base():
    ordered = order_corners_clockwise(_RECT, [0, 0, 1])
    shuffled = order_corners_clockwise(_RECT[[2, 0, 3, 1]], [0, 0, 1])
    assert np.allclose(ordered, shuffled)
    assert np.argmin(np.linalg.norm(ordered, axis=1)) == 0  # C1 nearest base
    # clockwise viewed along -Z: signed area in XY must be negative
    x, y = ordered[:, 0], ordered[:, 1]
    area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    assert area < 0


def test_frame_from_rectangle_convention():
    T = frame_from_rectangle(order_corners_clockwise(_RECT, [0, 0, 1]), [0, 0, -1])
    assert np.allclose(T[:3, 3], [600, 400, 0])          # origin = center
    assert T[2, 2] > 0.99                                 # +Z up even if normal fed down
    assert abs(T[:3, 0] @ np.array([0, 0, 1])) < 1e-9     # X in plane
    assert np.allclose(np.cross(T[:3, 0], T[:3, 1]), T[:3, 2], atol=1e-9)  # right-handed


class _Cam:
    K = np.array([[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]])
    dist = np.zeros(5)


def test_calibration_id_stable_and_sensitive():
    a, b = camera_calibration_id(_Cam()), camera_calibration_id(_Cam())
    assert a == b and a.startswith("cam-") and len(a) == 16
    changed = _Cam()
    changed.K = _Cam.K + 1.0
    assert camera_calibration_id(changed) != a


def test_locked_survey_to_dict_roundtrips_geometry():
    snap = _snapshot()
    corners = order_corners_clockwise(_RECT, [0, 0, 1])
    survey = LockedWorkframeSurvey(
        mode=MODE_COMPACT, boundary_provenance=PROVENANCE_BY_MODE[MODE_COMPACT],
        captures=(_record(),), plane_normal_base=(0, 0, 1), plane_point_base=(600, 400, 0),
        corners_base=tuple(map(tuple, corners)), center_base=(600, 400, 0),
        frame_T_base=tuple(map(tuple, frame_from_rectangle(corners, [0, 0, 1]))),
        size_mm=(1200.0, 800.0), quality={"plane_rms_mm": 0.8}, calibration_id="cam-abc",
        locked_robot=snap, locked_at=100.0)
    d = survey.to_dict()
    assert d["boundary_provenance"] == PROVENANCE_COMPACT
    assert np.asarray(d["corners_base"]).shape == (4, 3)
    assert survey.corners_np().shape == (4, 3) and survey.frame_np().shape == (4, 4)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `py -3.10 -m pytest tests\test_survey_contract.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'tasni.modules.scan.survey_contract'`

- [ ] **Step 3: Write the implementation**

```python
# tasni/modules/scan/survey_contract.py
"""Immutable capture and survey contracts (spec §11, Phase 1).

One authoritative ``LockedWorkframeSurvey`` feeds review, planning, and RoboDK
insertion. Everything is stored as frozen dataclasses with nested tuples so no
downstream consumer can mutate locked geometry. All geometry is in **mm**, robot
base frame unless a name says otherwise.
"""
from __future__ import annotations

import hashlib
import math
import time
from dataclasses import asdict, dataclass

import numpy as np

PROVENANCE_COMPACT = "camera measured - complete boundary"
PROVENANCE_FIVE_POSITION = "camera measured - five-position boundary survey"
PROVENANCE_USER_SPECIFIED = "user specified - plane measured, boundary declared"

MODE_COMPACT = "compact"
MODE_FIVE_POSITION = "five_position"
MODE_USER_SPECIFIED = "user_specified"

PROVENANCE_BY_MODE = {
    MODE_COMPACT: PROVENANCE_COMPACT,
    MODE_FIVE_POSITION: PROVENANCE_FIVE_POSITION,
    MODE_USER_SPECIFIED: PROVENANCE_USER_SPECIFIED,
}


def _as_tuple(arr):
    a = np.asarray(arr, dtype=float)
    if a.ndim == 1:
        return tuple(float(v) for v in a)
    return tuple(tuple(float(v) for v in row) for row in a)


def pose_delta(T_a, T_b) -> tuple[float, float]:
    """Translation (mm) and rotation (deg) between two 4x4 poses."""
    Ta = np.asarray(T_a, dtype=float)
    Tb = np.asarray(T_b, dtype=float)
    trans = float(np.linalg.norm(Ta[:3, 3] - Tb[:3, 3]))
    R = Ta[:3, :3].T @ Tb[:3, :3]
    c = (float(np.trace(R)) - 1.0) / 2.0
    rot = math.degrees(math.acos(max(-1.0, min(1.0, c))))
    return trans, rot


@dataclass(frozen=True)
class RobotStateSnapshot:
    joints: tuple[float, ...]
    camera_T: tuple[tuple[float, ...], ...]
    fetched_at: float
    stationary: bool

    def camera_T_np(self) -> np.ndarray:
        return np.asarray(self.camera_T, dtype=float)


def refresh_robot_state(rdk, *, settle_s: float = 0.15, joint_tol_deg: float = 0.01,
                        clock=time.monotonic, sleep=time.sleep) -> RobotStateSnapshot:
    """Explicitly fetch the real robot state twice; stationary iff both agree (§9)."""
    j0 = np.asarray(rdk.current_joints(), dtype=float)
    sleep(settle_s)
    j1 = np.asarray(rdk.current_joints(), dtype=float)
    T = rdk.camera_pose_T()
    if T is None:
        raise RuntimeError("robot pose unavailable - cannot take an authoritative capture")
    stationary = bool(np.max(np.abs(j1 - j0)) <= joint_tol_deg)
    return RobotStateSnapshot(joints=tuple(float(v) for v in j1), camera_T=_as_tuple(T),
                              fetched_at=float(clock()), stationary=stationary)


@dataclass(frozen=True)
class CaptureRecord:
    kind: str  # "compact" | "center" | "corner1".."corner4"
    robot: RobotStateSnapshot
    measurement_ts: float   # camera frame timestamp (server clock)
    captured_at: float      # host monotonic when the frames landed
    n_frames: int
    standoff_mm: float
    tilt_deg: float
    valid_frac: float
    plane_rms_mm: float
    plane_normal_base: tuple[float, float, float]
    plane_point_base: tuple[float, float, float]


def capture_is_fresh(record: CaptureRecord, *, now: float, max_age_s: float) -> bool:
    return (now - record.captured_at) <= max_age_s and record.robot.stationary


def robot_moved_since(snapshot: RobotStateSnapshot, current_T, *,
                      trans_tol_mm: float, rot_tol_deg: float) -> bool:
    if current_T is None:
        return True  # fail-open: an unknown pose counts as moved (§10)
    trans, rot = pose_delta(snapshot.camera_T_np(), current_T)
    return trans > trans_tol_mm or rot > rot_tol_deg


def _up_normal(normal_base) -> np.ndarray:
    n = np.asarray(normal_base, dtype=float)
    n = n / np.linalg.norm(n)
    return -n if n[2] < 0 else n


def order_corners_clockwise(corners_base, normal_base) -> np.ndarray:
    """C1..C4 clockwise viewed looking along -Z (spec §2); C1 nearest robot base."""
    c = np.asarray(corners_base, dtype=float).reshape(4, 3)
    n = _up_normal(normal_base)
    center = c.mean(axis=0)
    u = np.cross(n, [0.0, 0.0, 1.0])
    if np.linalg.norm(u) < 1e-9:
        u = np.array([1.0, 0.0, 0.0])
    u = u / np.linalg.norm(u)
    v = np.cross(n, u)
    ang = [math.atan2(float((p - center) @ v), float((p - center) @ u)) for p in c]
    c = c[np.argsort(ang)[::-1]]  # decreasing angle == clockwise seen from above
    start = int(np.argmin(np.linalg.norm(c, axis=1)))
    return np.roll(c, -start, axis=0)


def frame_from_rectangle(corners_base, normal_base) -> np.ndarray:
    """4x4 workframe: origin = center, +X = long edge, +Z = up-oriented normal (§2)."""
    c = np.asarray(corners_base, dtype=float).reshape(4, 3)
    n = _up_normal(normal_base)
    e1, e2 = c[1] - c[0], c[2] - c[1]
    x = e1 if np.linalg.norm(e1) >= np.linalg.norm(e2) else e2
    x = x - n * float(x @ n)
    x = x / np.linalg.norm(x)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, np.cross(n, x), n, c.mean(axis=0)
    return T


def camera_calibration_id(camera_cfg) -> str:
    """Stable identity of the active intrinsics + distortion (§10, §11)."""
    payload = np.round(np.concatenate([
        np.asarray(camera_cfg.K, dtype=float).ravel(),
        np.asarray(camera_cfg.dist, dtype=float).ravel(),
    ]), 6).tobytes()
    return "cam-" + hashlib.sha1(payload).hexdigest()[:12]


@dataclass(frozen=True)
class LockedWorkframeSurvey:
    mode: str
    boundary_provenance: str
    captures: tuple[CaptureRecord, ...]
    plane_normal_base: tuple[float, float, float]
    plane_point_base: tuple[float, float, float]
    corners_base: tuple[tuple[float, float, float], ...]
    center_base: tuple[float, float, float]
    frame_T_base: tuple[tuple[float, ...], ...]
    size_mm: tuple[float, float]
    quality: dict
    calibration_id: str
    locked_robot: RobotStateSnapshot
    locked_at: float

    def corners_np(self) -> np.ndarray:
        return np.asarray(self.corners_base, dtype=float)

    def frame_np(self) -> np.ndarray:
        return np.asarray(self.frame_T_base, dtype=float)

    def to_dict(self) -> dict:
        return asdict(self)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `py -3.10 -m pytest tests\test_survey_contract.py -q`
Expected: all PASS. Then `py -3.10 -m pytest -q` — no regressions.

- [ ] **Step 5: Commit and push**

```powershell
git add tasni/modules/scan/survey_contract.py tests/test_survey_contract.py
git commit -m "feat(scan): immutable survey contract (LockedWorkframeSurvey, spec Phase 1)"
git push
```

---

### Task 2: Compact classifier

**Files:**
- Create: `tasni/modules/scan/classifier.py`
- Modify: `tasni/core/config.py` (inside `class ScanConfig`, group after `live_frame_margin_uv` ~line 492)
- Test: `tests/test_scan_classifier.py`

**Interfaces:**
- Consumes: `SurveyMeasurement` (`tasni/modules/scan/survey.py:47` — uses `.detected`, `.tilt_deg`).
- Produces: `CompactEligibility(eligible, reasons, guard_ok, boundary_ok, centered_ok, tilt_ok, identity_ok, coverage_ok, predicted_coverage)` frozen dataclass with `.to_dict()`;
  `rectangle_identity_consistent(history_uv, *, tol_uv, min_frames) -> bool`;
  `classify_compact(survey, raw_corners_uv, boundary, scfg, *, predicted_coverage=None, outline_history=None) -> CompactEligibility`.
- New `ScanConfig` keys (exact defaults): `compact_guard_uv: float = 0.06`, `compact_center_tol_uv: float = 0.15`, `compact_identity_frames: int = 5`, `compact_identity_tol_uv: float = 0.04`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_scan_classifier.py
import numpy as np

from tasni.core.config import ScanConfig
from tasni.modules.scan.classifier import (
    CompactEligibility, classify_compact, rectangle_identity_consistent,
)


class _Survey:
    detected = True
    fully_framed = True
    tilt_deg = 2.0


_GOOD_UV = np.array([[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]])
_BOUNDARY = {"overruns": False, "polygon_uv": _GOOD_UV.tolist()}


def _history(n=5, jitter=0.0):
    rng = np.random.default_rng(0)
    return [_GOOD_UV + rng.normal(0, jitter, _GOOD_UV.shape) for _ in range(n)]


def test_compact_all_conditions_pass():
    r = classify_compact(_Survey(), _GOOD_UV, _BOUNDARY, ScanConfig(),
                         outline_history=_history())
    assert isinstance(r, CompactEligibility)
    assert r.eligible is True and r.reasons == ()


def test_guard_band_rejects_edge_hugging_rectangle():
    uv = np.array([[0.01, 0.2], [0.8, 0.2], [0.8, 0.8], [0.01, 0.8]])
    r = classify_compact(_Survey(), uv, _BOUNDARY, ScanConfig(), outline_history=_history())
    assert r.eligible is False and r.guard_ok is False
    assert any("guard" in reason for reason in r.reasons)


def test_boundary_overrun_or_missing_rejects():
    r = classify_compact(_Survey(), _GOOD_UV, {"overruns": True}, ScanConfig(),
                         outline_history=_history())
    assert r.eligible is False and r.boundary_ok is False
    r2 = classify_compact(_Survey(), _GOOD_UV, None, ScanConfig(), outline_history=_history())
    assert r2.boundary_ok is False


def test_off_center_and_tilt_reject():
    off = _GOOD_UV + np.array([0.28, 0.0])
    r = classify_compact(_Survey(), off, _BOUNDARY, ScanConfig(), outline_history=[off] * 5)
    assert r.centered_ok is False
    tilted = _Survey()
    tilted.tilt_deg = 15.0
    r2 = classify_compact(tilted, _GOOD_UV, _BOUNDARY, ScanConfig(), outline_history=_history())
    assert r2.tilt_ok is False and r2.eligible is False


def test_identity_needs_consistent_multiframe_history():
    assert rectangle_identity_consistent(_history(5, jitter=0.001), tol_uv=0.04, min_frames=5)
    assert not rectangle_identity_consistent(_history(3), tol_uv=0.04, min_frames=5)
    wild = _history(4) + [_GOOD_UV + 0.2]
    assert not rectangle_identity_consistent(wild, tol_uv=0.04, min_frames=5)


def test_coverage_gate_deferred_when_none_but_enforced_when_given():
    ok = classify_compact(_Survey(), _GOOD_UV, _BOUNDARY, ScanConfig(),
                          predicted_coverage=None, outline_history=_history())
    assert ok.coverage_ok is True  # deferred to generate_scan_targets' hard gate
    low = classify_compact(_Survey(), _GOOD_UV, _BOUNDARY, ScanConfig(),
                           predicted_coverage=0.5, outline_history=_history())
    assert low.coverage_ok is False and low.eligible is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `py -3.10 -m pytest tests\test_scan_classifier.py -q`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Add config keys, then implement**

In `ScanConfig` (`tasni/core/config.py`):

```python
    # Compact-eligibility classifier (two-path plan §6)
    compact_guard_uv: float = 0.06        # raw corners must sit this far inside the frame
    compact_center_tol_uv: float = 0.15   # outline centroid distance from image center
    compact_identity_frames: int = 5      # consecutive frames the rectangle must agree
    compact_identity_tol_uv: float = 0.04 # max corner drift across those frames
```

```python
# tasni/modules/scan/classifier.py
"""Compact-vs-large eligibility at d* (spec §6).

Pure decision logic — no camera, robot, or RoboDK access. When
``predicted_coverage`` is None the coverage gate defers to
``generate_scan_targets``'s existing hard gate (min_surface_coverage +
surface_coverage_hard_fail).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class CompactEligibility:
    eligible: bool
    reasons: tuple[str, ...]
    guard_ok: bool
    boundary_ok: bool
    centered_ok: bool
    tilt_ok: bool
    identity_ok: bool
    coverage_ok: bool
    predicted_coverage: float | None

    def to_dict(self) -> dict:
        return asdict(self)


def rectangle_identity_consistent(history_uv, *, tol_uv: float, min_frames: int) -> bool:
    """True when the last ``min_frames`` outlines agree corner-for-corner (§6)."""
    if history_uv is None or len(history_uv) < min_frames:
        return False
    recent = [np.asarray(o, dtype=float) for o in history_uv[-min_frames:]]
    ref = recent[0]
    for cur in recent[1:]:
        if cur.shape != ref.shape:
            return False
        if float(np.max(np.linalg.norm(cur - ref, axis=-1))) > tol_uv:
            return False
    return True


def classify_compact(survey, raw_corners_uv, boundary, scfg, *,
                     predicted_coverage: float | None = None,
                     outline_history=None) -> CompactEligibility:
    reasons: list[str] = []

    if not bool(getattr(survey, "detected", False)):
        reasons.append("no surface detected under the reticle")

    guard = float(scfg.compact_guard_uv)
    guard_ok = False
    if raw_corners_uv is not None:
        uv = np.asarray(raw_corners_uv, dtype=float).reshape(-1, 2)
        guard_ok = bool(np.all((uv >= guard) & (uv <= 1.0 - guard)))
    if not guard_ok:
        reasons.append("raw (untrimmed) boundary leaves the guard region")

    boundary_ok = bool(boundary) and not bool(boundary.get("overruns", True))
    if not boundary_ok:
        reasons.append("four physical boundaries not confirmed by segmentation")

    centered_ok = False
    if raw_corners_uv is not None:
        centroid = np.asarray(raw_corners_uv, dtype=float).reshape(-1, 2).mean(axis=0)
        centered_ok = bool(np.linalg.norm(centroid - 0.5) <= float(scfg.compact_center_tol_uv))
    if not centered_ok:
        reasons.append("rectangle is not sufficiently centered")

    tilt_ok = float(getattr(survey, "tilt_deg", 90.0)) <= float(scfg.survey_max_tilt_deg)
    if not tilt_ok:
        reasons.append("plane tilt exceeds the survey tolerance")

    identity_ok = rectangle_identity_consistent(
        outline_history, tol_uv=float(scfg.compact_identity_tol_uv),
        min_frames=int(scfg.compact_identity_frames))
    if not identity_ok:
        reasons.append("rectangle identity not consistent across the multi-frame acquisition")

    coverage_ok = True
    if predicted_coverage is not None:
        coverage_ok = float(predicted_coverage) >= float(scfg.min_surface_coverage)
        if not coverage_ok:
            reasons.append("predicted planned-view coverage below threshold")

    eligible = not reasons
    return CompactEligibility(eligible=eligible, reasons=tuple(reasons), guard_ok=guard_ok,
                              boundary_ok=boundary_ok, centered_ok=centered_ok,
                              tilt_ok=tilt_ok, identity_ok=identity_ok,
                              coverage_ok=coverage_ok, predicted_coverage=predicted_coverage)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `py -3.10 -m pytest tests\test_scan_classifier.py tests\test_scan_config.py -q`
Expected: PASS (config-defaults test stays green — the new keys are additive).

- [ ] **Step 5: Commit and push**

```powershell
git add tasni/modules/scan/classifier.py tasni/core/config.py tests/test_scan_classifier.py
git commit -m "feat(scan): compact-eligibility classifier with guard/identity gates"
git push
```

---

### Task 3: Lock produces the `LockedWorkframeSurvey`

**Files:**
- Modify: `tasni/modules/scan/service.py` — `LockedScanSurface` (`:66`), `lock_scan_surface` (`:184-296`)
- Test: `tests/test_scan_job.py` (extend — reuse its existing fake services/RdkIO/camera fixtures)

**Interfaces:**
- Consumes (Task 1): `refresh_robot_state`, `camera_calibration_id`, `order_corners_clockwise`, `frame_from_rectangle`, `CaptureRecord`, `LockedWorkframeSurvey`, `MODE_COMPACT`, `MODE_USER_SPECIFIED`, `PROVENANCE_BY_MODE`.
- Produces: `LockedScanSurface` gains two fields consumed by Tasks 4/5/8:
  `survey_record: LockedWorkframeSurvey | None = None` and `lock_token: str = ""`;
  `lock_scan_surface` gains keyword `user_region_mm: tuple[float, float] | None = None`;
  new helper `_survey_record_from_lock(survey, seed_T, snapshot, camera_cfg, *, mode, n_frames, measurement_ts, valid_frac, plane_rms_mm) -> LockedWorkframeSurvey | None`;
  new helper `_plane_rms_mm(depth, K, *, stride=8) -> float`.
- The `gate` `JobEvent` published at lock gains keys `"survey"` (the record's `to_dict()`) and `"boundary_provenance"`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_scan_job.py`, using the same fakes `test_lock_then_create_targets_reuses_frozen_surface` (`:247`) already uses)

```python
def test_lock_builds_locked_workframe_survey_compact(scan_services):
    from tasni.modules.scan.service import lock_scan_surface
    from tasni.modules.scan.survey_contract import MODE_COMPACT, PROVENANCE_COMPACT
    locked = lock_scan_surface(scan_services)  # fixture serves a fully-framed surface
    rec = locked.survey_record
    assert rec is not None and rec.mode == MODE_COMPACT
    assert rec.boundary_provenance == PROVENANCE_COMPACT
    assert rec.corners_np().shape == (4, 3)
    assert rec.frame_np()[2, 2] > 0.99            # +Z up
    assert rec.calibration_id.startswith("cam-")
    assert rec.locked_robot.stationary is True
    assert locked.lock_token != ""


def test_lock_crop_is_user_specified_with_declared_size(scan_services):
    from tasni.modules.scan.service import lock_scan_surface
    from tasni.modules.scan.survey_contract import MODE_USER_SPECIFIED, PROVENANCE_USER_SPECIFIED
    locked = lock_scan_surface(scan_services, force_crop=True, user_region_mm=(1200.0, 900.0))
    rec = locked.survey_record
    assert rec.mode == MODE_USER_SPECIFIED
    assert rec.boundary_provenance == PROVENANCE_USER_SPECIFIED
    assert sorted(rec.size_mm, reverse=True) == [1200.0, 900.0]


def test_lock_gate_event_carries_survey_and_provenance(scan_services):
    from tasni.modules.scan.service import lock_scan_surface
    lock_scan_surface(scan_services)
    gate_events = [e for e in scan_services.bus.events if e.kind == "gate"]
    payload = gate_events[-1].data
    assert payload["live"] is False
    assert "outline_uv" in payload            # locked polygon is displayable as-is
    assert payload["boundary_provenance"]
    assert payload["survey"]["mode"] == "compact"
```

If the file has no `scan_services` fixture (it may build services inline per test), follow the construction used by `test_lock_then_create_targets_reuses_frozen_surface` and wrap it as a local fixture; the assertions stay identical. The fake RdkIO must expose `current_joints()` returning a constant list — extend the fake if it lacks it.

- [ ] **Step 2: Run to verify the new tests fail**

Run: `py -3.10 -m pytest tests\test_scan_job.py -q -k "locked_workframe or user_specified or carries_survey"`
Expected: FAIL — `LockedScanSurface` has no `survey_record` / unexpected kwarg.

- [ ] **Step 3: Implement in `service.py`**

3a. Extend the dataclass at `:66`:

```python
@dataclass
class LockedScanSurface:
    frame: object
    reading: object
    survey: object
    gate_payload: dict
    seed_T: object
    seed_joints: object
    locked_at: float
    survey_record: "LockedWorkframeSurvey | None" = None
    lock_token: str = ""
```

(keep existing field names/order; only append the two new ones with defaults).

3b. Add imports and the helpers (place near `_large_surface_crop_mm`):

```python
import uuid
from .survey_contract import (
    MODE_COMPACT, MODE_USER_SPECIFIED, PROVENANCE_BY_MODE,
    CaptureRecord, LockedWorkframeSurvey, camera_calibration_id,
    frame_from_rectangle, order_corners_clockwise, refresh_robot_state,
)
from .plane import fit_plane


def _plane_rms_mm(depth, K, *, stride: int = 8) -> float:
    """Quick plane-fit RMS (mm) of the fused lock depth, for the quality report."""
    d = np.asarray(depth, dtype=float)[::stride, ::stride]
    v, u = np.nonzero(d > 0)
    if len(v) < 50:
        return float("nan")
    z = d[v, u]
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    pts = np.stack([(u * stride - cx) / fx * z, (v * stride - cy) / fy * z, z], axis=1)
    try:
        normal, centroid, _ = fit_plane(pts, distance=6.0)
    except Exception:
        return float("nan")
    res = (pts - centroid) @ np.asarray(normal, dtype=float)
    return float(np.sqrt(np.mean(res ** 2)))


def _survey_record_from_lock(survey, seed_T, snapshot, camera_cfg, *, mode, n_frames,
                             measurement_ts, valid_frac, plane_rms_mm):
    """Build the immutable §11 contract from the authoritative lock acquisition."""
    if survey.corners_cam_mm is None:
        return None
    T = np.asarray(seed_T, dtype=float)
    R, t = T[:3, :3], T[:3, 3]
    corners_base = np.asarray(survey.corners_cam_mm, dtype=float) @ R.T + t
    normal_base = R @ np.asarray(survey.normal_cam, dtype=float)
    if normal_base[2] < 0:
        normal_base = -normal_base
    centroid_base = R @ np.asarray(survey.centroid_cam_mm, dtype=float) + t
    corners_base = order_corners_clockwise(corners_base, normal_base)
    frame_T = frame_from_rectangle(corners_base, normal_base)
    e1 = float(np.linalg.norm(corners_base[1] - corners_base[0]))
    e2 = float(np.linalg.norm(corners_base[2] - corners_base[1]))
    record = CaptureRecord(
        kind="compact" if mode == MODE_COMPACT else "center", robot=snapshot,
        measurement_ts=float(measurement_ts), captured_at=snapshot.fetched_at,
        n_frames=int(n_frames), standoff_mm=float(survey.standoff_mm),
        tilt_deg=float(survey.tilt_deg), valid_frac=float(valid_frac),
        plane_rms_mm=float(plane_rms_mm),
        plane_normal_base=tuple(normal_base), plane_point_base=tuple(centroid_base))
    quality = {
        "plane_rms_mm": float(plane_rms_mm), "standoff_mm": float(survey.standoff_mm),
        "tilt_deg": float(survey.tilt_deg), "valid_frac": float(valid_frac),
        "measure_frames": int(n_frames),
    }
    return LockedWorkframeSurvey(
        mode=mode, boundary_provenance=PROVENANCE_BY_MODE[mode], captures=(record,),
        plane_normal_base=tuple(normal_base), plane_point_base=tuple(centroid_base),
        corners_base=tuple(map(tuple, corners_base)),
        center_base=tuple(corners_base.mean(axis=0)),
        frame_T_base=tuple(map(tuple, frame_T)), size_mm=(max(e1, e2), min(e1, e2)),
        quality=quality, calibration_id=camera_calibration_id(camera_cfg),
        locked_robot=snapshot, locked_at=snapshot.fetched_at)
```

3c. In `lock_scan_surface(services, *, force_crop=False)`:
- change the signature to `lock_scan_surface(services, *, force_crop: bool = False, user_region_mm: tuple[float, float] | None = None)`;
- where `_large_surface_crop_mm(...)` / `_crop_gate_payload(...)` compute the crop dimensions, use `user_region_mm` when given, else the existing `scfg.work_crop_mm`;
- replace the bare `seed_joints = rdk.current_joints()` block (`:288-296`) with an explicit refresh (§9): `snapshot = refresh_robot_state(rdk)`; keep `seed_T = rdk.camera_pose_T()` but assert consistency: `seed_T = snapshot.camera_T_np()`, `seed_joints = list(snapshot.joints)`;
- after gate assembly and BEFORE publishing the gate event, build:

```python
    mode = MODE_USER_SPECIFIED if crop_mode else MODE_COMPACT
    plane_rms = _plane_rms_mm(depth, K)
    record = _survey_record_from_lock(
        survey, snapshot.camera_T_np(), snapshot, services.config.camera,
        mode=mode, n_frames=n_frames, measurement_ts=gate_payload.get("measurement_ts", 0.0),
        valid_frac=gate_payload.get("valid_frac", 0.0), plane_rms_mm=plane_rms)
    if record is not None:
        gate_payload["survey"] = record.to_dict()
        gate_payload["boundary_provenance"] = record.boundary_provenance
```

- construct the return value with `survey_record=record, lock_token=uuid.uuid4().hex`.

Note: in crop mode `survey.corners_cam_mm` holds the reticle square from
`reticle_plane_square` (`survey.py:249-251`), so the same transform path yields the
user-specified rectangle; its `size_mm` therefore equals the declared region.

- [ ] **Step 4: Run the full scan suite**

Run: `py -3.10 -m pytest tests\test_scan_job.py tests\test_survey_contract.py -q` then `py -3.10 -m pytest -q`
Expected: PASS, no regressions (existing lock tests unaffected — new fields have defaults).

- [ ] **Step 5: Commit and push**

```powershell
git add tasni/modules/scan/service.py tests/test_scan_job.py
git commit -m "feat(scan): lock builds the immutable LockedWorkframeSurvey with explicit robot refresh"
git push
```

---

### Task 4: User-specified region route

**Files:**
- Modify: `tasni/modules/scan/module.py` (`ScanModule`), `tasni/modules/scan/service.py` (none beyond Task 3), `tasni/core/config.py` (no new keys — reuses `work_crop_mm` as the persisted default)
- Test: `tests/test_scan_job.py` (module-level test — `ScanModule` methods are plain Python and callable with the fake services)

**Interfaces:**
- Consumes: `lock_scan_surface(..., user_region_mm=...)` (Task 3), `save_overrides` (`tasni/core/config.py:691`).
- Produces: `POST /surface/region` with body `SurfaceRegionBody{width_mm: float, height_mm: float}`; module state `self._user_region_mm: tuple[float, float]` initialised from `cfg.scan.work_crop_mm` and passed into every `lock_scan_surface` call; persisted via `save_overrides({"scan": {"work_crop_mm": [w, h]}})`.

- [ ] **Step 1: Write the failing test**

```python
def test_surface_region_route_updates_lock_dimensions(scan_services, tmp_path, monkeypatch):
    import tasni.modules.scan.module as scan_module
    saved = {}
    monkeypatch.setattr(scan_module, "save_overrides", lambda u: saved.update(u))
    mod = scan_module.ScanModule(scan_services)
    body = scan_module.SurfaceRegionBody(width_mm=1200.0, height_mm=900.0)
    out = mod.surface_region(body)
    assert out["user_region_mm"] == [1200.0, 900.0]
    assert saved == {"scan": {"work_crop_mm": [1200.0, 900.0]}}
    with pytest.raises(Exception):
        mod.surface_region(scan_module.SurfaceRegionBody(width_mm=50.0, height_mm=900.0))
```

- [ ] **Step 2: Run to verify it fails** — `py -3.10 -m pytest tests\test_scan_job.py -q -k surface_region` → FAIL (`SurfaceRegionBody` missing).

- [ ] **Step 3: Implement in `module.py`**

- Add `from tasni.core.config import save_overrides` and the body model next to `SurfaceLockBody` (`:39`):

```python
class SurfaceRegionBody(BaseModel):
    width_mm: float
    height_mm: float
```

- In `__init__` (state block `:50-63`): `self._user_region_mm = tuple(float(v) for v in services.config.scan.work_crop_mm)`.
- New route (register next to `surface_lock` at `:342`):

```python
    def surface_region(self, body: SurfaceRegionBody):
        w, h = float(body.width_mm), float(body.height_mm)
        if not (100.0 <= w <= 4000.0 and 100.0 <= h <= 4000.0):
            raise HTTPException(422, "region dimensions must be 100-4000 mm")
        self._user_region_mm = (w, h)
        save_overrides({"scan": {"work_crop_mm": [w, h]}})
        return {"user_region_mm": [w, h]}
```

(register with the same router mechanism the neighbouring routes use — mirror `surface_lock`'s decorator/registration exactly; `POST /surface/region`.)
- In `surface_lock` (`:342`): pass `user_region_mm=self._user_region_mm` into `lock_scan_surface`.
- Note in code: the Jetson's live crop square (`server/server_unicast_syncronous.py:225`, `WORK_CROP_MM = 1000.0`) remains display-only; the host lock dimensions are authoritative. Server change is deliberately out of scope (§12).

- [ ] **Step 4: Run** — `py -3.10 -m pytest tests\test_scan_job.py -q` then full suite. PASS.

- [ ] **Step 5: Commit and push**

```powershell
git add tasni/modules/scan/module.py tests/test_scan_job.py
git commit -m "feat(scan): user-specified region route with persisted dimensions"
git push
```

---

### Task 5: Provenance and quality flow to targets, run report, and insert

**Files:**
- Modify: `tasni/modules/scan/service.py` — `ScanParams` (`:1354`), `generate_scan_targets` (`:1032`), `_result_report` (`:1373`), `ScanCaptureJob.__call__` (`:1499`), `insert_scan` (`:1816`); `tasni/modules/scan/module.py` — `poses_generate` (`:373`), `run` (`:429`)
- Test: `tests/test_scan_job.py`

**Interfaces:**
- Consumes: `LockedScanSurface.survey_record` (Task 3).
- Produces: `ScanParams` gains `boundary_provenance: str | None = None` and `survey: dict | None = None`;
  `generate_scan_targets`'s return dict gains `"boundary_provenance"`, `"survey"`, `"lock_token"`;
  `_result_report` gains keyword `provenance=None, survey=None` and writes `report["boundary_provenance"]`, `report["survey"]`;
  `insert_scan`'s `runs.write_active("scan", payload)` payload gains `"boundary_provenance"` and `"survey_quality"`.

- [ ] **Step 1: Failing test** — extend `test_generate_run_insert` (`tests/test_scan_job.py:159`) or add:

```python
def test_provenance_flows_lock_to_insert(scan_services):
    from tasni.modules.scan import service as scan_service
    locked = scan_service.lock_scan_surface(scan_services)
    gen = scan_service.generate_scan_targets(scan_services, locked)
    assert gen["boundary_provenance"] == locked.survey_record.boundary_provenance
    assert gen["lock_token"] == locked.lock_token
    params = scan_service.ScanParams(
        boundary_provenance=gen["boundary_provenance"], survey=gen["survey"])
    job = scan_service.ScanCaptureJob(scan_services, params)
    result = job(_fake_ctx())          # reuse the ctx helper the existing run test uses
    assert result["report"]["boundary_provenance"] == gen["boundary_provenance"]
    inserted = scan_service.insert_scan(scan_services, result=job.result)
    payload = scan_services.runs.written["scan"]   # match the fake runs recorder in this file
    assert payload["boundary_provenance"] == gen["boundary_provenance"]
    assert "survey_quality" in payload
```

Adapt the two helper references (`_fake_ctx`, `runs.written`) to the exact fakes already present in this file — `test_generate_run_insert` shows both; do not invent new fixtures.

- [ ] **Step 2: Run to verify failure** — `py -3.10 -m pytest tests\test_scan_job.py -q -k provenance_flows` → FAIL (unexpected kwargs).

- [ ] **Step 3: Implement** (all additive, defaults `None` keep every existing call site working):
- `ScanParams`: append the two fields.
- `generate_scan_targets`: in the returned dict (`:1331-1350`) add
  `"boundary_provenance": locked.survey_record.boundary_provenance if locked and locked.survey_record else None`,
  `"survey": locked.survey_record.to_dict() if locked and locked.survey_record else None`,
  `"lock_token": locked.lock_token if locked else ""`.
- `_result_report(...)`: add `provenance=None, survey=None` keywords; set both keys on the report dict it builds.
- `ScanCaptureJob.__call__`: pass `provenance=self.params.boundary_provenance, survey=self.params.survey` into its `_result_report` call.
- `insert_scan`: read `report.get("boundary_provenance")` / `report.get("survey", {}).get("quality")` from whichever source resolved (result/run_id/job) and add `"boundary_provenance"` + `"survey_quality"` to the `runs.write_active` payload.
- `module.py`: `poses_generate` stores `self._planned_provenance = out.get("boundary_provenance")`, `self._planned_survey = out.get("survey")`, `self._targets_token = out.get("lock_token")`; `run()` builds `ScanParams(..., boundary_provenance=self._planned_provenance, survey=self._planned_survey)`.

- [ ] **Step 4: Run** — `py -3.10 -m pytest tests\test_scan_job.py -q` then full suite. PASS.

- [ ] **Step 5: Commit and push**

```powershell
git add tasni/modules/scan/service.py tasni/modules/scan/module.py tests/test_scan_job.py
git commit -m "feat(scan): boundary provenance flows lock -> targets -> report -> insert"
git push
```

---

### Task 6: Locked polygon is the sole review geometry (web UI)

**Files:**
- Modify: `tasni/webui/src/pages/Scan.tsx` — `lockDisplayGate` (`:72`), lamps array (`:632-639`), lock chip (`:708-712`), `SurfaceModeNotice` (`:938`)
- Modify: `tasni/webui/src/pages/AimHud.tsx` — `interface GateReading` (`:10-51`)

No JS test infra exists; verification is `npm run build` plus the backend test from Task 3 asserting the lock gate payload is self-sufficient (`outline_uv` present).

- [ ] **Step 1: Remove the frontend lock latch (§11, §12).** In `Scan.tsx`, `lockDisplayGate(next, prev)` currently preserves the previous LIVE `outline_uv/points_uv/visible_outline_uv/grid_uv` when the lock snapshot (`live !== true`) arrives. Replace its body so a non-live gate is displayed exactly as sent:

```tsx
function lockDisplayGate(next: GateReading, _prev: GateReading | null): GateReading {
  // Spec §11: no frontend display latch may replace the locked polygon.
  return next;
}
```

Then inline/remove the now-trivial call if the linter complains about the unused parameter.

- [ ] **Step 2: Provenance chip.** Add `boundary_provenance?: string` and `survey?: Record<string, unknown>` to `GateReading` (`AimHud.tsx:10-51`). In the lock-state chip block (`Scan.tsx:708-712`), when the gate is locked (`gate.live === false`) and `gate.boundary_provenance` is set, render it under the chip:

```tsx
{gate?.live === false && gate?.boundary_provenance && (
  <div className="provenance-chip" title="boundary provenance (spec §2)">
    {gate.boundary_provenance}
  </div>
)}
```

Style: small mono text, amber background when the value starts with `user specified`, green otherwise (visually distinct user-specified geometry, §2).

- [ ] **Step 3: Advisory lamps.** In the lamps array (`Scan.tsx:632-639`) tag `CENTER` and `EDGE A` with `advisory: true`; where lamps render, give advisory lamps 55% opacity and `title="advisory — does not block lock"` so mandatory and advisory values are no longer presented identically (§12).

- [ ] **Step 4: Region inputs.** In `SurfaceModeNotice` (`Scan.tsx:938`), when `surface_mode === "crop"`, render two numeric inputs (defaults from `gate.crop_size_mm`) labeled `Region W / H (mm)` plus an `Apply region` button that POSTs `{width_mm, height_mm}` to `/surface/region` via the existing `moduleApi("scan")` helper (`Scan.tsx:10`), then calls the existing `refreshLive()` (`:470`). Rename the toggle copy from `Large crop ON` to `User-specified region`.

- [ ] **Step 5: Build + commit + push**

```powershell
cd tasni\webui; npm run build; cd ..\..
git add tasni/webui/src/pages/Scan.tsx tasni/webui/src/pages/AimHud.tsx
git commit -m "feat(scan-ui): locked polygon sole source, provenance chip, advisory lamps, region inputs"
git push
```

---

### Task 7: Pose-liveness (two-mode guidance, §9)

**Files:**
- Modify: `tasni/modules/scan/service.py` (new helper), `tasni/modules/scan/module.py` (live loop `:184-317`), `tasni/webui/src/pages/Scan.tsx` + `AimHud.tsx`
- Test: `tests/test_scan_depth_gate.py`

**Interfaces:**
- Produces: `annotate_pose_liveness(metrics: dict, *, pose_T, driver_ok: bool) -> dict` in `service.py`, setting `metrics["pose_live"]: bool`; telemetry consumers read `gate.pose_live`.

- [ ] **Step 1: Failing tests** (append to `tests/test_scan_depth_gate.py`):

```python
def test_pose_liveness_flag():
    import numpy as np
    from tasni.modules.scan.service import annotate_pose_liveness
    live = annotate_pose_liveness({}, pose_T=np.eye(4), driver_ok=True)
    assert live["pose_live"] is True
    for pose, ok in ((None, True), (np.eye(4), False), (None, False)):
        assert annotate_pose_liveness({}, pose_T=pose, driver_ok=ok)["pose_live"] is False
```

- [ ] **Step 2: Run to verify failure**, then implement:

```python
def annotate_pose_liveness(metrics: dict, *, pose_T, driver_ok: bool) -> dict:
    """Spec §9: pose-derived guidance is only 'live' when the driver mirrors the arm."""
    metrics["pose_live"] = bool(driver_ok and pose_T is not None)
    return metrics
```

- [ ] **Step 3: Wire the live loop** (`module.py:184-317`): cache `driver_ok` from `services.rdk.robot_connected()[0]` (wrapped in `try/except Exception: driver_ok = False`), refreshed at most every 2 s (store `last_driver_check` monotonic beside `last_ideal_mm`). After `stabilize_live_scan_payload(...)` call `annotate_pose_liveness(metrics, pose_T=pose_T, driver_ok=driver_ok)`.

- [ ] **Step 4: UI.** Add `pose_live?: boolean` to `GateReading`. In `AimHud.tsx`, when `gate.pose_live === false`: render the X/Y readouts and `JogBar` (`:315`) with the existing `PendingReadout` styling (`:239`) and show an amber `POSE: MODEL` chip next to the readouts (title: `driver not monitoring — X/Y guidance is not real-time`). No behavior change when the flag is `true` or absent (older payloads).

- [ ] **Step 5: Verify, commit, push**

```powershell
py -3.10 -m pytest tests\test_scan_depth_gate.py -q
cd tasni\webui; npm run build; cd ..\..
git add tasni/modules/scan/service.py tasni/modules/scan/module.py tasni/webui/src/pages/Scan.tsx tasni/webui/src/pages/AimHud.tsx tests/test_scan_depth_gate.py
git commit -m "feat(scan): pose-liveness flag drives two-mode guidance (spec sec 9)"
git push
```

---

### Task 8: Hard gates — collision hard-fail default + lock-token guard

**Files:**
- Modify: `tasni/core/config.py:585` (`collision_filter_hard_fail`), `tasni/modules/scan/module.py` (`run` `:429`, `surface_unlock` `:368`, `poses_generate` `:373`)
- Test: `tests/test_scan_job.py`, `tests/test_scan_config.py`

**Interfaces:**
- Consumes: `lock_token` on `LockedScanSurface` (Task 3), `_targets_token` (Task 5).
- Produces: `run()` raises before starting a job when targets predate the current lock.

- [ ] **Step 1: Failing tests**

```python
def test_collision_hard_fail_is_default():
    from tasni.core.config import ScanConfig
    assert ScanConfig().collision_filter_hard_fail is True


def test_run_refuses_targets_from_a_previous_lock(scan_services):
    import tasni.modules.scan.module as scan_module
    mod = scan_module.ScanModule(scan_services)
    mod.surface_lock(scan_module.SurfaceLockBody(mode="auto"))
    mod.poses_generate()
    mod.surface_unlock()                       # locked state changed after generation
    with pytest.raises(Exception, match="regenerate"):
        mod.run()
```

- [ ] **Step 2: Verify failure, then implement.**
- Flip the default: `collision_filter_hard_fail: bool = True` (`config.py:585`). Spec §10: a soft collision bypass is not appropriate for production target sets.
- Check the two existing collision tests (`tests/test_scan_job.py:418` bypass, `:442` hard-fail): the bypass test must now set `collision_filter_hard_fail = False` explicitly on its fake config to keep exercising the soft path; the hard-fail test should keep passing unchanged.
- Token guard in `module.py.run()` before job start:

```python
        token = getattr(self, "_targets_token", None)
        if token is not None:
            current = self._locked_surface.lock_token if self._locked_surface else None
            if current != token:
                raise HTTPException(409, "targets predate the current surface lock - regenerate targets")
```

- `surface_unlock()` keeps clearing `self._locked_surface` (token check then fails naturally). `poses_generate` already stores `_targets_token` (Task 5).

- [ ] **Step 3: Run** — `py -3.10 -m pytest tests\test_scan_job.py tests\test_scan_config.py -q` then full suite. PASS.

- [ ] **Step 4: Commit and push**

```powershell
git add tasni/core/config.py tasni/modules/scan/module.py tests/test_scan_job.py tests/test_scan_config.py
git commit -m "feat(scan): collision hard-fail by default; run refuses stale lock tokens"
git push
```

---

### Task 9: Five-position geometry (`rect_fit.py`)

**Files:**
- Create: `tasni/modules/scan/rect_fit.py`
- Test: `tests/test_rect_fit.py`

**Interfaces:**
- Consumes: `fit_plane` (`tasni/modules/scan/plane.py:72`), `_plane_basis` (`plane.py:62`).
- Produces (Task 11 imports these exact names):
  `GlobalPlane(normal, point, rms_mm, max_residual_mm, per_set_rms_mm)`,
  `fit_global_plane(point_sets, *, distance_mm=6.0, n_iterations=1000, seed=0) -> GlobalPlane`,
  `project_points_2d(points, normal, point, u, v) -> np.ndarray`,
  `lift_points_3d(points2d, point, u, v) -> np.ndarray`,
  `EdgeLine(direction, normal, offset, rms_mm, n_points)`,
  `fit_edge_line(points2d) -> EdgeLine`,
  `RectangleSolution(corners2d, size_mm, center2d, angle_deg, edge_rms_mm, parallelism_deg, perpendicularity_deg, discrepancy_mm, corner_agreement_mm)` with `.to_dict()`,
  `solve_constrained_rectangle(edge_points, *, local_corners2d=None) -> RectangleSolution`.
- Edge order convention: `edge_points = [e0, e1, e2, e3]` where `e_i` joins corner `C_{i+1}` to `C_{i+2}` (cyclic), i.e. `e0 = C1→C2`. Corner `C_{i+1}` = intersection of `e_{i-1 mod 4}` and `e_i`. All units mm, 2D coordinates in the plane basis.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_rect_fit.py
import math
import numpy as np
import pytest

from tasni.modules.scan.rect_fit import (
    fit_edge_line, fit_global_plane, lift_points_3d, project_points_2d,
    solve_constrained_rectangle,
)


def _edge_points(a, b, n=60, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0.05, 0.95, n)[:, None]
    pts = np.asarray(a, float) + t * (np.asarray(b, float) - np.asarray(a, float))
    if noise:
        d = np.asarray(b, float) - np.asarray(a, float)
        nrm = np.array([-d[1], d[0]]) / np.linalg.norm(d)
        pts = pts + rng.normal(0, noise, (n, 1)) * nrm
    return pts


# 1600 x 1000 rectangle rotated 17 deg
_ANG = math.radians(17.0)
_R = np.array([[math.cos(_ANG), -math.sin(_ANG)], [math.sin(_ANG), math.cos(_ANG)]])
_CORNERS = (np.array([[0, 0], [1600, 0], [1600, 1000], [0, 1000]], float) - [800, 500]) @ _R.T


def _edges(noise=0.0):
    return [_edge_points(_CORNERS[i], _CORNERS[(i + 1) % 4], noise=noise, seed=i)
            for i in range(4)]


def test_perfect_rectangle_is_recovered_exactly():
    sol = solve_constrained_rectangle(_edges())
    assert sorted(sol.size_mm, reverse=True) == pytest.approx([1600.0, 1000.0], abs=1e-6)
    assert sol.parallelism_deg == pytest.approx(0.0, abs=1e-9)
    assert sol.perpendicularity_deg == pytest.approx(0.0, abs=1e-9)
    assert sol.discrepancy_mm == pytest.approx(0.0, abs=1e-6)
    # corners match the truth up to cyclic order
    diffs = [np.abs(np.roll(sol.corners2d, k, axis=0) - _CORNERS).max() for k in range(4)]
    assert min(diffs) < 1e-6


def test_noisy_rectangle_error_is_bounded():
    sol = solve_constrained_rectangle(_edges(noise=1.0))
    assert abs(max(sol.size_mm) - 1600.0) < 3.0
    assert abs(min(sol.size_mm) - 1000.0) < 3.0
    assert sol.discrepancy_mm < 5.0
    assert max(sol.edge_rms_mm) < 2.5


def test_biased_edge_shows_up_as_discrepancy():
    edges = _edges(noise=0.5)
    d = _CORNERS[1] - _CORNERS[0]
    nrm = np.array([-d[1], d[0]]) / np.linalg.norm(d)
    half = len(edges[0]) // 2
    edges[0][:half] += nrm * 30.0          # corrupt half of edge 0 by 30 mm
    sol = solve_constrained_rectangle(edges)
    assert sol.discrepancy_mm > 5.0 or max(sol.edge_rms_mm) > 5.0


def test_corner_agreement_reported():
    sol = solve_constrained_rectangle(_edges(), local_corners2d=_CORNERS + 2.0)
    assert sol.corner_agreement_mm == pytest.approx(2.0 * math.sqrt(2), abs=0.01)


def test_edge_line_direction_and_rms():
    line = fit_edge_line(_edge_points([0, 0], [100, 0], noise=0.5))
    assert abs(line.direction[1]) < 0.02 and line.rms_mm < 1.0


def test_global_plane_and_projection_roundtrip():
    rng = np.random.default_rng(1)
    sets = []
    for cx in (0.0, 800.0, -800.0):
        xy = rng.uniform(-200, 200, (300, 2)) + [cx, 0.0]
        z = rng.normal(0, 0.4, 300)
        sets.append(np.column_stack([xy, z + 500.0]))
    plane = fit_global_plane(sets)
    assert abs(plane.normal[2]) > 0.999
    assert plane.rms_mm < 1.0 and len(plane.per_set_rms_mm) == 3
    from tasni.modules.scan.plane import _plane_basis
    u, v = _plane_basis(np.asarray(plane.normal))
    p2 = project_points_2d(sets[0], np.asarray(plane.normal), np.asarray(plane.point), u, v)
    p3 = lift_points_3d(p2, np.asarray(plane.point), u, v)
    d = np.abs((p3 - np.asarray(plane.point)) @ np.asarray(plane.normal))
    assert p2.shape == (300, 2) and d.max() < 1e-9
```

- [ ] **Step 2: Run to verify failure** — `py -3.10 -m pytest tests\test_rect_fit.py -q` → module not found.

- [ ] **Step 3: Implement**

```python
# tasni/modules/scan/rect_fit.py
"""Global plane + constrained rectangle fitting for the five-position survey (spec §7).

All inputs/outputs are mm. 2D work happens in a plane basis (u, v) from
``plane._plane_basis``. The constrained solve fits ONE rectangle orientation
theta to all four edges in closed form (total-least-squares on pooled second
moments, with the perpendicular pair rotated 90 deg), then places each edge at
the mean projection of its evidence. The unconstrained-vs-constrained corner
discrepancy is the primary diagnostic for cross-capture registration error
(spec §7 conditioning note).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from .plane import fit_plane


@dataclass(frozen=True)
class GlobalPlane:
    normal: tuple[float, float, float]
    point: tuple[float, float, float]
    rms_mm: float
    max_residual_mm: float
    per_set_rms_mm: tuple[float, ...]


def fit_global_plane(point_sets, *, distance_mm: float = 6.0, n_iterations: int = 1000,
                     seed: int = 0) -> GlobalPlane:
    sets = [np.asarray(p, dtype=float).reshape(-1, 3) for p in point_sets]
    all_pts = np.concatenate(sets, axis=0)
    normal, centroid, _ = fit_plane(all_pts, distance=distance_mm,
                                    n_iterations=n_iterations, seed=seed)
    normal = np.asarray(normal, dtype=float)
    if normal[2] < 0:
        normal = -normal
    res_all = np.abs((all_pts - centroid) @ normal)
    per = tuple(float(np.sqrt(np.mean(((s - centroid) @ normal) ** 2))) for s in sets)
    return GlobalPlane(tuple(normal), tuple(np.asarray(centroid, dtype=float)),
                       float(np.sqrt(np.mean(res_all ** 2))), float(res_all.max()), per)


def project_points_2d(points, normal, point, u, v) -> np.ndarray:
    d = np.asarray(points, dtype=float).reshape(-1, 3) - np.asarray(point, dtype=float)
    return np.column_stack([d @ np.asarray(u, dtype=float), d @ np.asarray(v, dtype=float)])


def lift_points_3d(points2d, point, u, v) -> np.ndarray:
    p2 = np.asarray(points2d, dtype=float).reshape(-1, 2)
    return (np.asarray(point, dtype=float)
            + p2[:, :1] * np.asarray(u, dtype=float)
            + p2[:, 1:2] * np.asarray(v, dtype=float))


@dataclass(frozen=True)
class EdgeLine:
    direction: tuple[float, float]  # unit vector along the edge
    normal: tuple[float, float]     # unit normal; the line is p . normal == offset
    offset: float
    rms_mm: float
    n_points: int


def fit_edge_line(points2d) -> EdgeLine:
    p = np.asarray(points2d, dtype=float).reshape(-1, 2)
    if len(p) < 2:
        raise ValueError("edge line needs at least 2 points")
    c = p.mean(axis=0)
    d = p - c
    w, vecs = np.linalg.eigh(d.T @ d)
    direction = vecs[:, int(np.argmax(w))]
    direction = direction / np.linalg.norm(direction)
    nrm = np.array([-direction[1], direction[0]])
    offset = float(c @ nrm)
    res = p @ nrm - offset
    return EdgeLine(tuple(direction), tuple(nrm), offset,
                    float(np.sqrt(np.mean(res ** 2))), int(len(p)))


@dataclass(frozen=True)
class RectangleSolution:
    corners2d: tuple[tuple[float, float], ...]  # C1..C4, corner i+1 = e_{i-1} x e_i
    size_mm: tuple[float, float]                # (|e1-e3 separation|, |e0-e2 separation|)
    center2d: tuple[float, float]
    angle_deg: float
    edge_rms_mm: tuple[float, float, float, float]
    parallelism_deg: float                      # worst opposite-edge angle mismatch (unconstrained)
    perpendicularity_deg: float                 # worst adjacent-edge deviation from 90 (unconstrained)
    discrepancy_mm: float                       # max |unconstrained corner - constrained corner|
    corner_agreement_mm: float | None           # vs locally detected corners, when given

    def to_dict(self) -> dict:
        return asdict(self)


def _intersect(n1, o1, n2, o2) -> np.ndarray:
    return np.linalg.solve(np.stack([n1, n2]), np.array([o1, o2]))


def _angle_deg(direction) -> float:
    return math.degrees(math.atan2(direction[1], direction[0]))


def _angdiff(a: float, b: float) -> float:
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def solve_constrained_rectangle(edge_points, *, local_corners2d=None) -> RectangleSolution:
    pts = [np.asarray(e, dtype=float).reshape(-1, 2) for e in edge_points]
    if len(pts) != 4:
        raise ValueError("need exactly 4 edge point sets (C1C2, C2C3, C3C4, C4C1)")

    # Unconstrained fit: independent TLS lines -> raw corner intersections + angle checks.
    lines = [fit_edge_line(p) for p in pts]
    un_corners = np.array([
        _intersect(np.asarray(lines[(i - 1) % 4].normal), lines[(i - 1) % 4].offset,
                   np.asarray(lines[i].normal), lines[i].offset)
        for i in range(4)
    ])
    parallelism = max(_angdiff(_angle_deg(lines[0].direction), _angle_deg(lines[2].direction)),
                      _angdiff(_angle_deg(lines[1].direction), _angle_deg(lines[3].direction)))
    perpendicularity = max(
        abs(90.0 - _angdiff(_angle_deg(lines[i].direction),
                            _angle_deg(lines[(i + 1) % 4].direction)))
        for i in range(4))

    # Constrained fit: one theta for all four edges.
    # Cost J(theta) = n0(theta)^T (S0+S2) n0(theta) + n1(theta)^T (S1+S3) n1(theta)
    # with n0 = (-sin t, cos t), n1 = (cos t, sin t) and S_i the centered second
    # moments of edge i. J = const + P cos(2t) + Q sin(2t); minimum at
    # 2t = atan2(-Q, -P).
    S = []
    for p in pts:
        d = p - p.mean(axis=0)
        S.append(d.T @ d)
    A, B = S[0] + S[2], S[1] + S[3]
    P = ((A[1, 1] + B[0, 0]) - (A[0, 0] + B[1, 1])) / 2.0
    Q = B[0, 1] - A[0, 1]
    theta = 0.5 * math.atan2(-Q, -P)

    dir0 = np.array([math.cos(theta), math.sin(theta)])       # edges 0, 2
    dir1 = np.array([-math.sin(theta), math.cos(theta)])      # edges 1, 3
    n0 = np.array([-dir0[1], dir0[0]])
    n1 = np.array([-dir1[1], dir1[0]])
    normals = [n0, n1, n0, n1]
    offsets = [float(p.mean(axis=0) @ normals[i]) for i, p in enumerate(pts)]
    corners = np.array([
        _intersect(normals[(i - 1) % 4], offsets[(i - 1) % 4], normals[i], offsets[i])
        for i in range(4)
    ])
    edge_rms = tuple(float(np.sqrt(np.mean((p @ normals[i] - offsets[i]) ** 2)))
                     for i, p in enumerate(pts))
    size = (abs(offsets[1] - offsets[3]), abs(offsets[0] - offsets[2]))
    discrepancy = float(np.max(np.linalg.norm(un_corners - corners, axis=1)))

    agreement = None
    if local_corners2d is not None:
        local = np.asarray(local_corners2d, dtype=float).reshape(4, 2)
        agreement = float(np.max(np.linalg.norm(local - corners, axis=1)))

    return RectangleSolution(
        corners2d=tuple(map(tuple, corners)), size_mm=size,
        center2d=tuple(corners.mean(axis=0)), angle_deg=math.degrees(theta),
        edge_rms_mm=edge_rms, parallelism_deg=float(parallelism),
        perpendicularity_deg=float(perpendicularity), discrepancy_mm=discrepancy,
        corner_agreement_mm=agreement)
```

- [ ] **Step 4: Run** — `py -3.10 -m pytest tests\test_rect_fit.py -q` → PASS; then full suite.

- [ ] **Step 5: Commit and push**

```powershell
git add tasni/modules/scan/rect_fit.py tests/test_rect_fit.py
git commit -m "feat(scan): global plane + constrained rectangle fit for five-position survey"
git push
```

---

### Task 10: Corner evidence extraction

**Files:**
- Create: `tasni/modules/scan/corner_evidence.py`
- Test: `tests/test_corner_evidence.py`

**Interfaces:**
- Consumes: nothing from other tasks (deprojects mm depth itself — deliberately NOT reusing `_backproject_depth` (`service.py:321`) to avoid its unit ambiguity).
- Produces (Task 11 consumes):
  `CornerEvidence(corner_uv, corner_base_mm, edge_points_base)` — `edge_points_base` is one pooled `(N,3)` mm array of boundary-arm points from BOTH arms; Task 11 assigns points to edges geometrically, so arm labeling is unnecessary;
  `extract_corner_evidence(depth, K, polygon_uv, T_base_cam, *, corner_hint_uv=(0.5, 0.5), arm_frac=0.35, samples_per_arm=40, inset_px=4.0, window_px=2, min_valid_frac=0.3) -> CornerEvidence | None`.
- Depth is a `(H,W)` array in **mm** (RealSense uint16 convention); `polygon_uv` is normalized `(N,2)`; `T_base_cam` is the 4x4 base←camera pose in mm.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_corner_evidence.py
import numpy as np

from tasni.modules.scan.corner_evidence import extract_corner_evidence


def _scene(z_mm=400.0, w=320, h=240):
    """Flat plane at z with an L-shaped boundary meeting at pixel (160, 120)."""
    depth = np.full((h, w), z_mm, dtype=np.float32)
    K = np.array([[300.0, 0, w / 2], [0, 300.0, h / 2], [0, 0, 1]])
    # polygon: corner at image center, arms going +u and +v
    poly = np.array([[0.95, 0.5], [0.5, 0.5], [0.5, 0.95]])
    return depth, K, poly


def test_corner_and_edges_are_extracted_in_base_frame():
    depth, K, poly = _scene()
    T = np.eye(4)
    T[2, 3] = 900.0  # camera 900 mm above base origin, looking along +Z(cam)
    ev = extract_corner_evidence(depth, K, poly, T, corner_hint_uv=(0.5, 0.5))
    assert ev is not None
    assert np.linalg.norm(np.asarray(ev.corner_uv) - 0.5) < 0.02
    pts = ev.edge_points_base
    assert pts.shape[0] >= 20 and pts.shape[1] == 3
    # all evidence lies on the plane z = 900 + 400 (identity rotation)
    assert np.allclose(pts[:, 2], 1300.0, atol=1.0)
    # points split between the two arms: some vary in x, some in y
    assert pts[:, 0].ptp() > 50.0 and pts[:, 1].ptp() > 50.0
    assert ev.corner_base_mm is not None


def test_returns_none_without_depth_support():
    depth, K, poly = _scene()
    depth[:] = 0.0
    assert extract_corner_evidence(depth, K, poly, np.eye(4)) is None


def test_returns_none_for_degenerate_polygon():
    depth, K, _ = _scene()
    tiny = np.array([[0.5, 0.5], [0.501, 0.5]])
    assert extract_corner_evidence(depth, K, tiny, np.eye(4)) is None
```

- [ ] **Step 2: Run to verify failure**, then implement:

```python
# tasni/modules/scan/corner_evidence.py
"""Extract base-frame corner/edge evidence from one corner capture (spec §7).

The boundary polygon (colour/SAM) proposes WHERE the physical edge is; depth +
the calibrated camera pose provide METRIC geometry (spec §11). Samples are
inset a few pixels toward the surface interior so depth is read on the
platform, not in the discontinuity.  All outputs are mm in the robot base.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CornerEvidence:
    corner_uv: tuple[float, float]
    corner_base_mm: tuple[float, float, float] | None
    edge_points_base: np.ndarray  # (N, 3) pooled from both arms


def _median_depth(depth, px, py, window_px: int) -> float:
    h, w = depth.shape
    x0, x1 = max(0, px - window_px), min(w, px + window_px + 1)
    y0, y1 = max(0, py - window_px), min(h, py + window_px + 1)
    patch = np.asarray(depth[y0:y1, x0:x1], dtype=float)
    vals = patch[patch > 0]
    return float(np.median(vals)) if len(vals) else 0.0


def _deproject_base(u_px, v_px, z_mm, K, T_base_cam) -> np.ndarray:
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    p_cam = np.array([(u_px - cx) / fx * z_mm, (v_px - cy) / fy * z_mm, z_mm, 1.0])
    return (np.asarray(T_base_cam, dtype=float) @ p_cam)[:3]


def _walk_arm(poly_px, start_idx, step, arm_len_px, n_samples):
    """Sample points along the polyline from start_idx in direction `step`."""
    out, travelled = [], 0.0
    i = start_idx
    target = np.linspace(arm_len_px / n_samples, arm_len_px, n_samples)
    ti = 0
    while ti < len(target) and 0 <= i + step < len(poly_px):
        a, b = poly_px[i], poly_px[i + step]
        seg = float(np.linalg.norm(b - a))
        while ti < len(target) and travelled + seg >= target[ti]:
            t = (target[ti] - travelled) / max(seg, 1e-9)
            out.append(a + t * (b - a))
            ti += 1
        travelled += seg
        i += step
    return np.asarray(out, dtype=float).reshape(-1, 2)


def extract_corner_evidence(depth, K, polygon_uv, T_base_cam, *,
                            corner_hint_uv=(0.5, 0.5), arm_frac: float = 0.35,
                            samples_per_arm: int = 40, inset_px: float = 4.0,
                            window_px: int = 2, min_valid_frac: float = 0.3):
    depth = np.asarray(depth)
    h, w = depth.shape
    poly = np.asarray(polygon_uv, dtype=float).reshape(-1, 2)
    if len(poly) < 3:
        return None
    poly_px = poly * [w, h]
    hint_px = np.asarray(corner_hint_uv, dtype=float) * [w, h]
    corner_idx = int(np.argmin(np.linalg.norm(poly_px - hint_px, axis=1)))
    corner_px = poly_px[corner_idx]

    arm_len_px = arm_frac * float(np.hypot(w, h))
    arms = [_walk_arm(poly_px, corner_idx, +1, arm_len_px, samples_per_arm),
            _walk_arm(poly_px, corner_idx, -1, arm_len_px, samples_per_arm)]
    interior = poly_px.mean(axis=0)

    pts_base = []
    n_requested = 0
    for arm in arms:
        for p in arm:
            n_requested += 1
            direction = interior - p
            norm = float(np.linalg.norm(direction))
            sample = p + (direction / norm * inset_px if norm > 1e-6 else 0.0)
            px, py = int(round(sample[0])), int(round(sample[1]))
            if not (0 <= px < w and 0 <= py < h):
                continue
            z = _median_depth(depth, px, py, window_px)
            if z <= 0:
                continue
            pts_base.append(_deproject_base(sample[0], sample[1], z, K, T_base_cam))
    if n_requested == 0 or len(pts_base) < max(4, int(min_valid_frac * n_requested)):
        return None

    corner_base = None
    zc = _median_depth(depth, int(round(corner_px[0])), int(round(corner_px[1])),
                       window_px * 3)
    if zc > 0:
        corner_base = tuple(_deproject_base(corner_px[0], corner_px[1], zc, K, T_base_cam))

    return CornerEvidence(corner_uv=(float(corner_px[0] / w), float(corner_px[1] / h)),
                          corner_base_mm=corner_base,
                          edge_points_base=np.asarray(pts_base, dtype=float))
```

- [ ] **Step 3: Run** — `py -3.10 -m pytest tests\test_corner_evidence.py -q` → PASS; full suite.

- [ ] **Step 4: Commit and push**

```powershell
git add tasni/modules/scan/corner_evidence.py tests/test_corner_evidence.py
git commit -m "feat(scan): corner/edge evidence extraction from depth + boundary polygon"
git push
```

---

### Task 11: Five-position survey state machine

**Files:**
- Create: `tasni/modules/scan/five_position.py`
- Modify: `tasni/core/config.py` (`ScanConfig`, six new keys)
- Test: `tests/test_five_position.py`

**Interfaces:**
- Consumes: Task 1 (`CaptureRecord`, `LockedWorkframeSurvey`, `MODE_FIVE_POSITION`, `PROVENANCE_BY_MODE`, `RobotStateSnapshot`, `capture_is_fresh`, `order_corners_clockwise`, `frame_from_rectangle`), Task 9 (`fit_global_plane`, `project_points_2d`, `lift_points_3d`, `solve_constrained_rectangle`), Task 10 (`CornerEvidence`), `_plane_basis` (`plane.py:62`).
- Produces (Task 13 consumes): `SURVEY_STEPS = ("center", "corner1", "corner2", "corner3", "corner4")`;
  `class FivePositionSurvey(scfg, *, clock=time.monotonic)` with
  `.step -> str` (next expected kind, `"review"` when all five accepted),
  `.state() -> dict` (`{"step", "accepted", "corners_base", "warnings"}`),
  `.add_capture(record: CaptureRecord, plane_points_base: np.ndarray, evidence: CornerEvidence | None) -> dict` (raises `RuntimeError`/`ValueError` on gate failure — the §7 acceptance conditions),
  `.recapture(kind: str) -> dict` (drops that capture; `.step` returns to the first missing),
  `.finish(*, calibration_id: str, locked_robot: RobotStateSnapshot) -> LockedWorkframeSurvey`.
- New `ScanConfig` keys (exact defaults):
  `survey_capture_max_age_s: float = 10.0`, `survey_coplanar_warn_mm: float = 3.0`,
  `survey_coplanar_reject_mm: float = 8.0`, `survey_edge_band_mm: float = 25.0`,
  `survey_rect_discrepancy_mm: float = 6.0`, `survey_min_edge_points: int = 20`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_five_position.py
import numpy as np
import pytest

from tasni.core.config import ScanConfig
from tasni.modules.scan.corner_evidence import CornerEvidence
from tasni.modules.scan.five_position import SURVEY_STEPS, FivePositionSurvey
from tasni.modules.scan.survey_contract import (
    MODE_FIVE_POSITION, PROVENANCE_FIVE_POSITION, CaptureRecord, RobotStateSnapshot,
)

_CORNERS = np.array([[200, 300, 0], [1800, 300, 0], [1800, 1300, 0], [200, 1300, 0]], float)
_CLOCK = [100.0]


def _snap():
    return RobotStateSnapshot((0.0,) * 6, tuple(map(tuple, np.eye(4))), _CLOCK[0], True)


def _record(kind, standoff=350.0, tilt=1.0, stationary=True, age=0.0):
    snap = RobotStateSnapshot((0.0,) * 6, tuple(map(tuple, np.eye(4))),
                              _CLOCK[0] - age, stationary)
    return CaptureRecord(kind=kind, robot=snap, measurement_ts=1.0,
                         captured_at=_CLOCK[0] - age, n_frames=5, standoff_mm=standoff,
                         tilt_deg=tilt, valid_frac=0.9, plane_rms_mm=0.6,
                         plane_normal_base=(0, 0, 1), plane_point_base=(1000, 800, 0))


def _plane_points(center, n=300, z=0.0, seed=0):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-150, 150, (n, 2)) + np.asarray(center, float)
    return np.column_stack([xy, np.full(n, z) + rng.normal(0, 0.3, n)])


def _corner_evidence(i, noise=0.4, seed=None):
    c = _CORNERS[i]
    prev_c = _CORNERS[(i - 1) % 4]
    next_c = _CORNERS[(i + 1) % 4]
    rng = np.random.default_rng(seed if seed is not None else i)
    pts = []
    for other in (prev_c, next_c):
        t = np.linspace(0.02, 0.30, 40)[:, None]
        seg = c + t * (other - c)
        seg = seg + np.column_stack([rng.normal(0, noise, 40),
                                     rng.normal(0, noise, 40), np.zeros(40)])
        pts.append(seg)
    return CornerEvidence(corner_uv=(0.5, 0.5), corner_base_mm=tuple(c),
                          edge_points_base=np.concatenate(pts, axis=0))


def _survey(scfg=None):
    return FivePositionSurvey(scfg or ScanConfig(), clock=lambda: _CLOCK[0])


def _run_all(s, z_offsets=(0, 0, 0, 0, 0)):
    s.add_capture(_record("center"), _plane_points([1000, 800], z=z_offsets[0]), None)
    for i in range(4):
        s.add_capture(_record(f"corner{i + 1}"),
                      _plane_points(_CORNERS[i][:2], z=z_offsets[i + 1], seed=i + 1),
                      _corner_evidence(i))
    return s


def test_happy_path_recovers_the_rectangle():
    s = _run_all(_survey())
    assert s.step == "review"
    survey = s.finish(calibration_id="cam-abc", locked_robot=_snap())
    assert survey.mode == MODE_FIVE_POSITION
    assert survey.boundary_provenance == PROVENANCE_FIVE_POSITION
    assert sorted(survey.size_mm, reverse=True) == pytest.approx([1600.0, 1000.0], abs=3.0)
    assert len(survey.captures) == 5
    assert survey.quality["discrepancy_mm"] < 5.0


def test_step_ordering_enforced_and_wrong_kind_rejected():
    s = _survey()
    assert s.step == SURVEY_STEPS[0]
    with pytest.raises(ValueError):
        s.add_capture(_record("corner2"), _plane_points([0, 0]), _corner_evidence(1))


def test_stale_or_moving_capture_rejected():
    s = _survey()
    with pytest.raises(RuntimeError):
        s.add_capture(_record("center", age=60.0), _plane_points([1000, 800]), None)
    with pytest.raises(RuntimeError):
        s.add_capture(_record("center", stationary=False), _plane_points([1000, 800]), None)


def test_corner_without_evidence_rejected():
    s = _survey()
    s.add_capture(_record("center"), _plane_points([1000, 800]), None)
    with pytest.raises(RuntimeError, match="evidence"):
        s.add_capture(_record("corner1"), _plane_points(_CORNERS[0][:2]), None)


def test_recapture_replaces_a_single_corner():
    s = _run_all(_survey())
    s.recapture("corner2")
    assert s.step == "corner2"
    s.add_capture(_record("corner2"), _plane_points(_CORNERS[1][:2], seed=9),
                  _corner_evidence(1, seed=9))
    assert s.step == "review"


def test_noncoplanar_warns_then_rejects():
    warn = _run_all(_survey(), z_offsets=(0, 0, 4.5, 0, 0))
    survey = warn.finish(calibration_id="c", locked_robot=_snap())
    assert "non_flat" in survey.quality.get("flags", [])
    bad = _run_all(_survey(), z_offsets=(0, 0, 30.0, 0, 0))
    with pytest.raises(RuntimeError, match="coplanar"):
        bad.finish(calibration_id="c", locked_robot=_snap())


def test_biased_corner_fails_the_discrepancy_gate():
    s = _survey()
    s.add_capture(_record("center"), _plane_points([1000, 800]), None)
    for i in range(4):
        ev = _corner_evidence(i)
        if i == 2:
            shifted = ev.edge_points_base + np.array([40.0, 0.0, 0.0])
            ev = CornerEvidence(ev.corner_uv,
                                tuple(np.asarray(ev.corner_base_mm) + [40.0, 0.0, 0.0]),
                                shifted)
        s.add_capture(_record(f"corner{i + 1}"),
                      _plane_points(_CORNERS[i][:2], seed=i + 1), ev)
    with pytest.raises(RuntimeError, match="reposition|discrepan"):
        s.finish(calibration_id="c", locked_robot=_snap())
```

- [ ] **Step 2: Run to verify failure**, then add the six config keys to `ScanConfig` (group them after the Task 2 compact keys):

```python
    # Five-position survey (two-path plan §7)
    survey_capture_max_age_s: float = 10.0
    survey_coplanar_warn_mm: float = 3.0
    survey_coplanar_reject_mm: float = 8.0
    survey_edge_band_mm: float = 25.0
    survey_rect_discrepancy_mm: float = 6.0
    survey_min_edge_points: int = 20
```

- [ ] **Step 3: Implement**

```python
# tasni/modules/scan/five_position.py
"""Guided center + four-corner survey (spec §7).

Pure state machine: capture orchestration (camera/robot/RoboDK) lives in
service.py (Task 13). Edge assignment is geometric — all arm points from the
two corners adjacent to an edge are pooled, then filtered to a band around the
corner-to-corner segment — so the extractor never needs to label arms.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .corner_evidence import CornerEvidence
from .plane import _plane_basis
from .rect_fit import (fit_global_plane, lift_points_3d, project_points_2d,
                       solve_constrained_rectangle)
from .survey_contract import (MODE_FIVE_POSITION, PROVENANCE_BY_MODE, CaptureRecord,
                              LockedWorkframeSurvey, RobotStateSnapshot,
                              capture_is_fresh, frame_from_rectangle,
                              order_corners_clockwise)

SURVEY_STEPS = ("center", "corner1", "corner2", "corner3", "corner4")


@dataclass
class _Accepted:
    record: CaptureRecord
    plane_points_base: np.ndarray
    evidence: CornerEvidence | None


class FivePositionSurvey:
    def __init__(self, scfg, *, clock=time.monotonic):
        self._scfg = scfg
        self._clock = clock
        self._accepted: dict[str, _Accepted] = {}
        self.warnings: list[str] = []

    @property
    def step(self) -> str:
        return next((s for s in SURVEY_STEPS if s not in self._accepted), "review")

    def state(self) -> dict:
        corners = [list(a.evidence.corner_base_mm)
                   for k, a in sorted(self._accepted.items())
                   if a.evidence is not None and a.evidence.corner_base_mm is not None]
        return {"step": self.step, "accepted": sorted(self._accepted),
                "corners_base": corners, "warnings": list(self.warnings)}

    def add_capture(self, record: CaptureRecord, plane_points_base, evidence) -> dict:
        expected = self.step
        if expected == "review":
            raise RuntimeError("survey already has all five captures")
        if record.kind != expected:
            raise ValueError(f"expected a {expected!r} capture, got {record.kind!r}")
        if not record.robot.stationary:
            raise RuntimeError("robot was moving during the capture - stop and remeasure")
        if not capture_is_fresh(record, now=self._clock(),
                                max_age_s=float(self._scfg.survey_capture_max_age_s)):
            raise RuntimeError("capture is stale - remeasure")
        if record.valid_frac < float(self._scfg.min_valid_depth_frac):
            raise RuntimeError("not enough valid depth in the capture")
        if record.tilt_deg > float(self._scfg.survey_max_tilt_deg):
            raise RuntimeError("camera tilt exceeds the survey tolerance - level and remeasure")
        if not (float(self._scfg.accurate_min_mm) * 0.8 <= record.standoff_mm
                <= float(self._scfg.accurate_max_mm) * 1.2):
            raise RuntimeError("standoff outside the validated range - adjust distance")
        if expected != "center":
            if evidence is None or len(evidence.edge_points_base) < int(
                    self._scfg.survey_min_edge_points):
                raise RuntimeError(
                    "corner capture has no usable edge evidence - reposition and recapture")
            if evidence.corner_base_mm is None:
                raise RuntimeError("corner depth missing - reposition and recapture")
        pts = np.asarray(plane_points_base, dtype=float).reshape(-1, 3)
        if len(pts) < 50:
            raise RuntimeError("too few plane points in the capture")
        self._accepted[expected] = _Accepted(record, pts, evidence)
        return self.state()

    def recapture(self, kind: str) -> dict:
        self._accepted.pop(kind, None)
        return self.state()

    def finish(self, *, calibration_id: str,
               locked_robot: RobotStateSnapshot) -> LockedWorkframeSurvey:
        if self.step != "review":
            raise RuntimeError(f"survey incomplete - next capture: {self.step}")
        acc = [self._accepted[s] for s in SURVEY_STEPS]

        plane = fit_global_plane([a.plane_points_base for a in acc])
        flags: list[str] = []
        worst = max(plane.per_set_rms_mm)
        if worst > float(self._scfg.survey_coplanar_reject_mm):
            raise RuntimeError(
                f"captures are not coplanar (worst per-position RMS {worst:.1f} mm)")
        if worst > float(self._scfg.survey_coplanar_warn_mm):
            flags.append("non_flat")
            self.warnings.append(
                f"surface labeled non-flat: per-position plane RMS up to {worst:.1f} mm")

        normal = np.asarray(plane.normal)
        point = np.asarray(plane.point)
        u, v = _plane_basis(normal)
        corners_approx = np.array([np.asarray(a.evidence.corner_base_mm, dtype=float)
                                   for a in acc[1:]])
        ordered3d = order_corners_clockwise(corners_approx, normal)
        ordered2d = project_points_2d(ordered3d, normal, point, u, v)
        pooled2d = [project_points_2d(a.evidence.edge_points_base, normal, point, u, v)
                    for a in acc[1:]]
        # map each ordered corner back to its capture's pooled points
        approx2d = project_points_2d(corners_approx, normal, point, u, v)
        pool_for = [pooled2d[int(np.argmin(np.linalg.norm(approx2d - c, axis=1)))]
                    for c in ordered2d]

        band = float(self._scfg.survey_edge_band_mm)
        edge_points = []
        for i in range(4):
            a2, b2 = ordered2d[i], ordered2d[(i + 1) % 4]
            cand = np.concatenate([pool_for[i], pool_for[(i + 1) % 4]], axis=0)
            d = b2 - a2
            length = float(np.linalg.norm(d))
            d = d / length
            n2 = np.array([-d[1], d[0]])
            rel = cand - a2
            along = rel @ d
            dist = np.abs(rel @ n2)
            keep = cand[(dist <= band) & (along >= -0.1 * length) & (along <= 1.1 * length)]
            if len(keep) < int(self._scfg.survey_min_edge_points):
                raise RuntimeError(
                    f"edge C{i + 1}-C{(i + 1) % 4 + 1} has too little evidence - "
                    "reposition and recapture the adjacent corners")
            edge_points.append(keep)

        rect = solve_constrained_rectangle(edge_points, local_corners2d=ordered2d)
        if rect.discrepancy_mm > float(self._scfg.survey_rect_discrepancy_mm):
            raise RuntimeError(
                "rectangle evidence is inconsistent "
                f"(unconstrained-vs-constrained discrepancy {rect.discrepancy_mm:.1f} mm) - "
                "reposition and recapture the weakest corner")

        corners3d = lift_points_3d(np.asarray(rect.corners2d), point, u, v)
        corners3d = order_corners_clockwise(corners3d, normal)
        frame_T = frame_from_rectangle(corners3d, normal)
        quality = {
            "plane_rms_mm": plane.rms_mm, "plane_max_residual_mm": plane.max_residual_mm,
            "per_position_rms_mm": list(plane.per_set_rms_mm),
            "edge_rms_mm": list(rect.edge_rms_mm),
            "parallelism_deg": rect.parallelism_deg,
            "perpendicularity_deg": rect.perpendicularity_deg,
            "discrepancy_mm": rect.discrepancy_mm,
            "corner_agreement_mm": rect.corner_agreement_mm,
            "size_mm": list(rect.size_mm), "flags": flags,
            "warnings": list(self.warnings),
        }
        return LockedWorkframeSurvey(
            mode=MODE_FIVE_POSITION,
            boundary_provenance=PROVENANCE_BY_MODE[MODE_FIVE_POSITION],
            captures=tuple(a.record for a in acc),
            plane_normal_base=tuple(normal), plane_point_base=tuple(point),
            corners_base=tuple(map(tuple, corners3d)),
            center_base=tuple(corners3d.mean(axis=0)),
            frame_T_base=tuple(map(tuple, frame_T)),
            size_mm=(float(max(rect.size_mm)), float(min(rect.size_mm))),
            quality=quality, calibration_id=calibration_id,
            locked_robot=locked_robot, locked_at=locked_robot.fetched_at)
```

- [ ] **Step 4: Run** — `py -3.10 -m pytest tests\test_five_position.py tests\test_scan_config.py -q` → PASS; full suite.

- [ ] **Step 5: Commit and push**

```powershell
git add tasni/modules/scan/five_position.py tasni/core/config.py tests/test_five_position.py
git commit -m "feat(scan): five-position survey state machine with coplanarity and discrepancy gates"
git push
```

---

### Task 12: Large-rectangle scan planning (`plan_rect_tour`)

**Files:**
- Modify: `tasni/modules/scan/planner.py`, `tasni/core/config.py` (two keys)
- Test: `tests/test_scan_planner.py`

**Interfaces:**
- Consumes: `AimPoint` (`planner.py:29`), `ScanPlan` (`planner.py:53`). Before coding, read `plan_scan`'s aim construction (`planner.py:147-172`) and reuse its exact `min_perpendicular_mm` expression and voxel formula (`planner.py:117-120`) so both planners scale identically.
- Produces: `plan_rect_tour(corners_base, normal_base, K, size_px, scan_cfg) -> ScanPlan` with `mode="large_survey"` and one `AimPoint` per tile of an overlap grid covering the rectangle at `accurate_min_mm` standoff.
- New `ScanConfig` keys: `survey_tour_overlap: float = 0.30`, `survey_tour_views_per_tile: int = 2`.

- [ ] **Step 1: Failing tests** (append to `tests/test_scan_planner.py`, reusing its existing `ScanConfig`/K fixtures):

```python
def test_rect_tour_tiles_cover_a_large_rectangle():
    import numpy as np
    from tasni.modules.scan.planner import plan_rect_tour
    from tasni.core.config import ScanConfig
    corners = np.array([[0, 0, 0], [2000, 0, 0], [2000, 1200, 0], [0, 1200, 0]], float)
    K = [[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]]
    plan = plan_rect_tour(corners, [0, 0, 1], K, (1280, 720), ScanConfig())
    assert plan.mode == "large_survey"
    assert len(plan.aims) >= 4                      # 2000x1200 needs a grid at ~350 mm
    pts = np.array([a.point_base_mm for a in plan.aims])
    assert pts[:, 0].min() > 0 and pts[:, 0].max() < 2000    # aims inside the rectangle
    assert np.allclose(pts[:, 2], 0.0, atol=1e-6)            # aims on the plane
    for a in plan.aims:
        assert a.standoff_mm == ScanConfig().accurate_min_mm
        assert np.allclose(a.view_dir_base, [0, 0, -1])


def test_rect_tour_small_rectangle_is_single_tile():
    import numpy as np
    from tasni.modules.scan.planner import plan_rect_tour
    from tasni.core.config import ScanConfig
    corners = np.array([[0, 0, 0], [200, 0, 0], [200, 150, 0], [0, 150, 0]], float)
    K = [[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]]
    plan = plan_rect_tour(corners, [0, 0, 1], K, (1280, 720), ScanConfig())
    assert len(plan.aims) == 1
```

- [ ] **Step 2: Verify failure, add config keys, implement** (in `planner.py`, below `plan_scan`):

```python
def plan_rect_tour(corners_base, normal_base, K, size_px, scan_cfg) -> ScanPlan:
    """Tile a large rectangle with close-range views at accurate_min_mm (spec §7).

    d* stand-in: accurate_min_mm until Phase 0 characterization stores a
    measured value. Footprint per tile from the pinhole model divided by
    frame_margin; tiles overlap by survey_tour_overlap.
    """
    corners = np.asarray(corners_base, dtype=float).reshape(4, 3)
    n = np.asarray(normal_base, dtype=float)
    n = n / np.linalg.norm(n)
    if n[2] < 0:
        n = -n
    fx, fy = float(K[0][0]), float(K[1][1])
    W, H = int(size_px[0]), int(size_px[1])
    d = float(scan_cfg.accurate_min_mm)
    foot_w = d * W / fx / float(scan_cfg.frame_margin)
    foot_h = d * H / fy / float(scan_cfg.frame_margin)
    step = 1.0 - float(scan_cfg.survey_tour_overlap)

    center = corners.mean(axis=0)
    ex, ey = corners[1] - corners[0], corners[3] - corners[0]
    Lx, Ly = float(np.linalg.norm(ex)), float(np.linalg.norm(ey))
    ux, uy = ex / Lx, ey / Ly
    nx = max(1, int(math.ceil(Lx / (foot_w * step))))
    ny = max(1, int(math.ceil(Ly / (foot_h * step))))

    voxel = float(np.clip(d / 1000.0 * scan_cfg.voxel_k,
                          scan_cfg.voxel_min_m, scan_cfg.voxel_max_m))
    aims = []
    for i in range(nx):
        for j in range(ny):
            p = (center + ux * ((i + 0.5) / nx - 0.5) * Lx
                        + uy * ((j + 0.5) / ny - 0.5) * Ly)
            aims.append(AimPoint(
                point_base_mm=tuple(p), view_dir_base=tuple(-n), standoff_mm=d,
                min_perpendicular_mm=d,  # replace with plan_scan's exact expression
                cone_half_angle_deg=float(scan_cfg.flat_cone_deg),
                roll_max_deg=float(scan_cfg.roll_max_deg),
                n_views=int(scan_cfg.survey_tour_views_per_tile)))
    return ScanPlan(mode="large_survey", aims=aims, standoff_mm=d, voxel_size_m=voxel,
                    cone_half_angle_deg=float(scan_cfg.flat_cone_deg), warnings=[])
```

The `min_perpendicular_mm=d` line is a placeholder for whatever expression `plan_scan` uses at `planner.py:147-172` — copy that expression verbatim (this is the one intentional read-the-neighbour step; the value must match `plan_scan`'s convention, not be invented).

- [ ] **Step 3: Wire into `generate_scan_targets`** (`service.py:1032`): where the plan branch is chosen (`:1098-1168`), add a new first branch — when `locked is not None and locked.survey_record is not None and locked.survey_record.mode == MODE_FIVE_POSITION`, call `plan_rect_tour(locked.survey_record.corners_np(), np.asarray(locked.survey_record.plane_normal_base), K, image_size, scfg)` and feed the resulting multi-aim plan through the SAME per-aim candidate generation / reachability / collision / diversity pipeline the single-aim path uses (loop over `plan.aims`, accumulate candidates, name targets `TasniScan_T{tile:02d}_{k}`). Coverage prediction: pass the survey's densified corners (reuse `_densify_quad`, `service.py:334`) to the existing `projected_corner_coverage` call so the §10 coverage hard gate applies across ALL tiles. Add a job test in `tests/test_scan_job.py` asserting a five-position locked surface yields targets from more than one tile for a 2 m rectangle (build the `LockedScanSurface` with a synthetic `LockedWorkframeSurvey` via Task 11's happy-path helper).

- [ ] **Step 4: Run** — `py -3.10 -m pytest tests\test_scan_planner.py tests\test_scan_job.py -q`, then full suite. PASS.

- [ ] **Step 5: Commit and push**

```powershell
git add tasni/modules/scan/planner.py tasni/modules/scan/service.py tasni/core/config.py tests/test_scan_planner.py tests/test_scan_job.py
git commit -m "feat(scan): tiled close-range tour planning for five-position rectangles"
git push
```

---

### Task 13: Survey capture orchestration + REST routes

**Files:**
- Modify: `tasni/modules/scan/service.py` (new `five_position_capture`), `tasni/modules/scan/module.py` (routes + state)
- Test: `tests/test_scan_job.py`

**Interfaces:**
- Consumes: Tasks 1, 10, 11; existing `_camera_hold`, `_combine_depth_frames` (`service.py:1476`), `survey_surface` (`survey.py:136`), `_survey_thresholds` (`service.py:310`), `color_work_boundary` (`color_boundary.py:101`), `sam_work_boundary` (`sam_boundary.py:172`), `refresh_robot_state`.
- Produces: `five_position_capture(services, survey: FivePositionSurvey) -> dict` in `service.py` — performs ONE authoritative step-and-measure acquisition (§7/§9): hold camera → grab `scfg.surface_measure_frames` depth frames + 1 color frame → `refresh_robot_state(rdk)` (raise if not stationary) → fuse depth → `survey_surface` for the local plane → deproject plane inliers to base mm (write a local `_deproject_plane_points_mm(depth, K, T_base_cam, stride=6) -> np.ndarray` — explicit mm, same pattern as `corner_evidence._deproject_base`) → boundary via the configured engine (`scfg.boundary_engine`: inline `color_work_boundary`, or `sam_work_boundary` when `"sam"`/`"sam_then_color"`, falling back to colour) → for corner steps `extract_corner_evidence(...)` → build `CaptureRecord(kind=survey.step, ...)` → `survey.add_capture(...)` → publish a `JobEvent("survey", survey.state())` → return the state dict.
- Module routes (register exactly like the `surface_*` routes):
  `POST /survey/begin` → creates `self._five_survey = FivePositionSurvey(cfg.scan)`, returns state;
  `GET /survey/state` → state or `{"step": null}` when inactive;
  `POST /survey/capture` → `five_position_capture(self.services, self._five_survey)` (stops the live loop the same way `surface_lock` does, restarts it after);
  `POST /survey/recapture` body `SurveyRecaptureBody{kind: str}`;
  `POST /survey/finish` → `record = self._five_survey.finish(calibration_id=camera_calibration_id(cfg.camera), locked_robot=refresh_robot_state(rdk))`, then store `self._locked_surface = LockedScanSurface(frame=None, reading=None, survey=None, gate_payload={"ok": True, "live": False, "surface_mode": "five_position", "boundary_provenance": record.boundary_provenance, "survey": record.to_dict()}, seed_T=record.locked_robot.camera_T_np(), seed_joints=list(record.locked_robot.joints), locked_at=record.locked_at, survey_record=record, lock_token=uuid.uuid4().hex)` and return `{"status": "locked", **record.quality}`;
  `POST /survey/cancel` → clears `self._five_survey`.

- [ ] **Step 1: Failing test** (fake services; the camera fake must serve depth+color like the lock tests):

```python
def test_five_position_capture_uses_fresh_robot_state(scan_services):
    from tasni.core.config import ScanConfig
    from tasni.modules.scan.five_position import FivePositionSurvey
    from tasni.modules.scan.service import five_position_capture
    survey = FivePositionSurvey(scan_services.config.scan)
    state = five_position_capture(scan_services, survey)
    assert state["step"] == "corner1"          # center accepted, machine advanced
    assert survey._accepted["center"].record.robot.stationary is True
    events = [e for e in scan_services.bus.events if e.kind == "survey"]
    assert events and events[-1].data["step"] == "corner1"
```

- [ ] **Step 2: Verify failure, implement `five_position_capture` + routes as specified above.** Keep the function under ~80 lines by reusing the lock flow's helpers; do not duplicate `lock_scan_surface` logic — extract a `_authoritative_acquisition(services) -> tuple[depth, color, n_frames, reading, survey_measurement, snapshot]` helper shared by both if the duplication grows.

- [ ] **Step 3: Run** — `py -3.10 -m pytest tests\test_scan_job.py -q`, full suite. PASS.

- [ ] **Step 4: Commit and push**

```powershell
git add tasni/modules/scan/service.py tasni/modules/scan/module.py tests/test_scan_job.py
git commit -m "feat(scan): five-position survey capture orchestration and REST routes"
git push
```

---

### Task 14: Five-position UI panel

**Files:**
- Modify: `tasni/webui/src/pages/Scan.tsx`
- Create: `tasni/webui/src/pages/SurveyPanel.tsx`

Verification: `npm run build` + manual walkthrough note (no JS test infra).

- [ ] **Step 1: `SurveyPanel.tsx`.** New component `SurveyPanel({api, onFinished})` where `api = moduleApi("scan")`:
  - polls `GET /survey/state` every 1 s while mounted (and refreshes on `survey` websocket events, which `Scan.tsx` forwards as a prop);
  - renders the step strip `CENTER → C1 → C2 → C3 → C4 → REVIEW` with the current step highlighted and accepted steps checked, plus the §7 ordering diagram: a small top-down SVG square with the four corners labeled clockwise and the accumulating polygon drawn from `state.corners_base` (project by dropping Z, autoscale to the SVG viewBox);
  - `Measure` button → `POST /survey/capture`; while pending show `Measuring…` and disable the button (§7: never pretend RoboDK follows the pendant); render the returned error string verbatim when the backend rejects a capture (all gate failures are actionable messages);
  - per-accepted-step `Recapture` buttons → `POST /survey/recapture {kind}`;
  - on REVIEW: show the quality table (every key in `state`/finish response: per-position RMS, edge RMS, parallelism, perpendicularity, discrepancy, size, flags) and a `Accept & lock` button → `POST /survey/finish`, then `onFinished()` (which re-uses the existing `generateTargets()` flow in `Scan.tsx:486`);
  - `Cancel survey` → `POST /survey/cancel`.
- [ ] **Step 2: Entry point in `Scan.tsx`.** Next to the `SurfaceModeNotice` toggle (`:938`), add a `Large surface — guided survey` button (visible when the gate reports `surface_mode === "crop"`), which starts the panel via `POST /survey/begin` and renders `SurveyPanel` in place of the lock controls until finished/cancelled. Forward `survey` websocket events (add a case to the handler at `:297-368`).
- [ ] **Step 3: Build, commit, push**

```powershell
cd tasni\webui; npm run build; cd ..\..
git add tasni/webui/src/pages/SurveyPanel.tsx tasni/webui/src/pages/Scan.tsx
git commit -m "feat(scan-ui): guided five-position survey panel"
git push
```

---

### Task 15: Characterization metrics (`tasni/core/characterize.py`)

**Files:**
- Create: `tasni/core/characterize.py`
- Test: `tests/test_characterize.py`

**Interfaces:**
- Consumes: `fit_plane` (`tasni/modules/scan/plane.py:72`).
- Produces (Task 16 consumes):
  `DistanceTrial(distance_mm, n_captures, plane_rms_mm, plane_max_mm, height_repeat_mm, normal_repeat_deg, length_err_mm, coverage_frac)` frozen dataclass with `.to_dict()`;
  `plane_metrics(point_sets_mm) -> dict` (`{"plane_rms_mm", "plane_max_mm", "height_repeat_mm", "normal_repeat_deg"}` — per-capture plane fits; height repeatability = std of per-capture plane heights along the mean normal; normal repeatability = max angle between any per-capture normal and the mean normal);
  `known_length_error_mm(points_a_mm, points_b_mm, true_mm) -> float` (abs error of the mean a→b distance);
  `summarize_distance_trial(distance_mm, plane_point_sets, length_samples, coverage_frac) -> DistanceTrial` where `length_samples = [(pa, pb, true_mm), ...]`;
  `choose_dstar(trials, *, max_rms_mm, max_height_repeat_mm, max_normal_repeat_deg, max_length_err_mm, min_coverage_frac) -> DistanceTrial | None` (closest passing distance, §5: "closest distance that passes the error budget", not best single capture).

- [ ] **Step 1: Failing tests**

```python
# tests/test_characterize.py
import numpy as np
import pytest

from tasni.core.characterize import (
    choose_dstar, known_length_error_mm, plane_metrics, summarize_distance_trial,
)


def _plane_set(z=400.0, sigma=0.3, n=500, seed=0):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-150, 150, (n, 2))
    return np.column_stack([xy, np.full(n, z) + rng.normal(0, sigma, n)])


def test_plane_metrics_repeatability():
    sets = [_plane_set(z=400.0 + dz, seed=i) for i, dz in enumerate((0.0, 0.1, -0.1))]
    m = plane_metrics(sets)
    assert m["plane_rms_mm"] < 0.5
    assert m["height_repeat_mm"] < 0.3
    assert m["normal_repeat_deg"] < 0.5
    bad = plane_metrics(sets + [_plane_set(z=403.0, seed=9)])
    assert bad["height_repeat_mm"] > 1.0


def test_known_length_error():
    a = np.tile([0.0, 0.0, 400.0], (10, 1))
    b = np.tile([297.4, 0.0, 400.0], (10, 1))
    assert known_length_error_mm(a, b, 297.0) == pytest.approx(0.4, abs=1e-9)


def _trial(d, rms=0.3, length_err=0.3, coverage=0.9):
    sets = [_plane_set(z=d, sigma=rms, seed=i) for i in range(3)]
    a = np.tile([0.0, 0.0, float(d)], (5, 1))
    b = np.tile([297.0 + length_err, 0.0, float(d)], (5, 1))
    return summarize_distance_trial(d, sets, [(a, b, 297.0)], coverage)


def test_choose_dstar_prefers_closest_passing_distance():
    trials = [_trial(300.0, rms=1.5), _trial(400.0), _trial(600.0)]
    best = choose_dstar(trials, max_rms_mm=1.0, max_height_repeat_mm=1.0,
                        max_normal_repeat_deg=1.0, max_length_err_mm=1.0,
                        min_coverage_frac=0.5)
    assert best is not None and best.distance_mm == 400.0
    none = choose_dstar([_trial(300.0, rms=2.0)], max_rms_mm=1.0,
                        max_height_repeat_mm=1.0, max_normal_repeat_deg=1.0,
                        max_length_err_mm=1.0, min_coverage_frac=0.5)
    assert none is None
```

- [ ] **Step 2: Verify failure, implement**

```python
# tasni/core/characterize.py
"""Distance-characterization metrics for selecting d* (spec §5, Phase 0).

Pure geometry, mm units, no hardware access. The CLI in
tools/characterize_distance.py feeds it captures; this module never touches the
camera or robot.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from tasni.modules.scan.plane import fit_plane


@dataclass(frozen=True)
class DistanceTrial:
    distance_mm: float
    n_captures: int
    plane_rms_mm: float
    plane_max_mm: float
    height_repeat_mm: float
    normal_repeat_deg: float
    length_err_mm: float
    coverage_frac: float

    def to_dict(self) -> dict:
        return asdict(self)


def plane_metrics(point_sets_mm) -> dict:
    normals, centroids, rms, max_res = [], [], [], []
    for pts in point_sets_mm:
        p = np.asarray(pts, dtype=float).reshape(-1, 3)
        n, c, _ = fit_plane(p, distance=6.0)
        n = np.asarray(n, dtype=float)
        if n[2] < 0:
            n = -n
        res = (p - c) @ n
        normals.append(n)
        centroids.append(np.asarray(c, dtype=float))
        rms.append(float(np.sqrt(np.mean(res ** 2))))
        max_res.append(float(np.max(np.abs(res))))
    mean_n = np.mean(normals, axis=0)
    mean_n = mean_n / np.linalg.norm(mean_n)
    ang = [math.degrees(math.acos(min(1.0, abs(float(n @ mean_n))))) for n in normals]
    heights_along_mean = [float(c @ mean_n) for c in centroids]
    return {
        "plane_rms_mm": float(np.mean(rms)),
        "plane_max_mm": float(np.max(max_res)),
        "height_repeat_mm": float(np.std(heights_along_mean)),
        "normal_repeat_deg": float(np.max(ang)),
    }


def known_length_error_mm(points_a_mm, points_b_mm, true_mm: float) -> float:
    a = np.asarray(points_a_mm, dtype=float).reshape(-1, 3)
    b = np.asarray(points_b_mm, dtype=float).reshape(-1, 3)
    return abs(float(np.mean(np.linalg.norm(a - b, axis=1))) - float(true_mm))


def summarize_distance_trial(distance_mm, plane_point_sets, length_samples,
                             coverage_frac) -> DistanceTrial:
    m = plane_metrics(plane_point_sets)
    errs = [known_length_error_mm(a, b, t) for a, b, t in length_samples] or [float("nan")]
    return DistanceTrial(distance_mm=float(distance_mm), n_captures=len(plane_point_sets),
                         plane_rms_mm=m["plane_rms_mm"], plane_max_mm=m["plane_max_mm"],
                         height_repeat_mm=m["height_repeat_mm"],
                         normal_repeat_deg=m["normal_repeat_deg"],
                         length_err_mm=float(np.max(errs)),
                         coverage_frac=float(coverage_frac))


def choose_dstar(trials, *, max_rms_mm, max_height_repeat_mm, max_normal_repeat_deg,
                 max_length_err_mm, min_coverage_frac) -> DistanceTrial | None:
    passing = [t for t in trials
               if t.plane_rms_mm <= max_rms_mm
               and t.height_repeat_mm <= max_height_repeat_mm
               and t.normal_repeat_deg <= max_normal_repeat_deg
               and t.length_err_mm <= max_length_err_mm
               and t.coverage_frac >= min_coverage_frac]
    return min(passing, key=lambda t: t.distance_mm) if passing else None
```

- [ ] **Step 3: Run** — `py -3.10 -m pytest tests\test_characterize.py -q` → PASS; full suite.

- [ ] **Step 4: Commit and push**

```powershell
git add tasni/core/characterize.py tests/test_characterize.py
git commit -m "feat(core): distance-characterization metrics and d* selection"
git push
```

---

### Task 16: Characterization CLI + calibration-age gate

**Files:**
- Create: `tools/characterize_distance.py`
- Modify: `tasni/modules/scan/service.py` (lock warning), `tasni/core/config.py` (two keys)
- Test: `tests/test_characterize.py` (pure parts only)

**Interfaces:**
- Consumes: Task 15, `CameraClient` (`tasni/core/camera.py:50` — `grab(with_depth=True)`), `RdkIO`, `camera_calibration_id` (Task 1), `BoardConfig` (`tasni/core/config.py:78` — READ THE CLASS FIRST for exact field names before writing the ChArUco detection; use `cv2.aruco.CharucoBoard` with those fields).
- Produces: `characterization/characterization-YYYYMMDD.json` (git-ignored directory; add `characterization/` to `.gitignore`) containing `{"calibration_id", "date", "trials": [DistanceTrial.to_dict()...], "dstar_mm", "budget": {...}}`;
  `latest_characterization(root: Path) -> dict | None` in `tools/characterize_distance.py` (importable; newest file by name);
  new `ScanConfig` keys `calibration_max_age_days: float = 30.0`, `calibration_expiry_hard_fail: bool = False`.
- CLI flow (operator-driven, step-and-measure): for each `--distances 300,400,500,600,800` entry, prompt the operator to jog to that standoff over the A3 ChArUco board, press Enter, then capture `--frames 5` RGB-D pairs; plane points from depth inliers; length samples from detected ChArUco corner pairs of known separation; oblique captures via a `--tilt` repeat prompt (§5). Writes the JSON and prints the `choose_dstar` verdict for the budget flags (`--max-rms`, `--max-height-repeat`, `--max-normal-repeat`, `--max-length-err`, `--min-coverage`).

- [ ] **Step 1: Failing test** for the importable helper:

```python
def test_latest_characterization_reads_newest(tmp_path):
    import json
    from tools.characterize_distance import latest_characterization
    assert latest_characterization(tmp_path) is None
    (tmp_path / "characterization-20260101.json").write_text(json.dumps({"dstar_mm": 400}))
    (tmp_path / "characterization-20260812.json").write_text(json.dumps({"dstar_mm": 350}))
    assert latest_characterization(tmp_path)["dstar_mm"] == 350
```

- [ ] **Step 2: Implement the CLI** with `latest_characterization` at module top (no hardware import at module scope — import `CameraClient`/`RdkIO` inside `main()` so the test can import the module headlessly). Follow `tools/jetson_probe.py` / `tools/jetson_intrinsics.py` for the repo's CLI conventions.

- [ ] **Step 3: Lock-side gate (§10).** In `lock_scan_surface`, after building the survey record: load `latest_characterization(Path("characterization"))`; if missing or older than `calibration_max_age_days` (parse the `date` field), append `"calibration verification missing or expired"` to `gate_payload.setdefault("warnings", [])` — and raise `RuntimeError` instead when `calibration_expiry_hard_fail` is true. Record `characterization.get("dstar_mm")` into the survey record's `quality` dict as `"dstar_mm"` when present. Add a fake-services test for both the warn and hard-fail branches (monkeypatch `latest_characterization`).

- [ ] **Step 4: Run** full suite; commit and push

```powershell
git add tools/characterize_distance.py tasni/modules/scan/service.py tasni/core/config.py tests/test_characterize.py .gitignore
git commit -m "feat(scan): characterization CLI and calibration-age gate (Phase 0 tool)"
git push
```

---

### Task 17: Docs, roadmap, final verification

**Files:**
- Modify: `docs/agent-debug-map.md` (new modules + routes), `CLAUDE.md` (roadmap bullet), `docs/scan-workframe-two-path-plan.md` (status line → "implemented through Phase N")

- [ ] **Step 1:** Run the complete suite and the web build one final time: `py -3.10 -m pytest -q` and `cd tasni\webui; npm run build`. Both must be green — paste the summary counts into the commit body.
- [ ] **Step 2:** Update `docs/agent-debug-map.md`: add `survey_contract.py`, `classifier.py`, `rect_fit.py`, `corner_evidence.py`, `five_position.py`, `characterize.py`, the `/surface/region` + `/survey/*` routes, and the new config keys to whichever tables it keeps.
- [ ] **Step 3:** Update the roadmap block in `CLAUDE.md` (scan module bullet) with one line: two-path workframe survey implemented (link both docs).
- [ ] **Step 4:** Commit and push:

```powershell
git add docs/agent-debug-map.md CLAUDE.md docs/scan-workframe-two-path-plan.md
git commit -m "docs: record two-path workframe survey implementation"
git push
```

---

## Spec coverage map (self-review)

| Spec item | Task(s) |
|---|---|
| §2 two paths + same result shape | 3, 11, 13 |
| §2 user-specified fast path | 3, 4, 6 |
| §2 frame convention / corner order | 1 |
| §5 d* characterization + in-app tool | 15, 16 |
| §6 compact entry conditions + predict-only coverage | 2, 3, 12 |
| §7 five-position capture sequence + geometry + recovery | 9, 10, 11, 13, 14 |
| §7 non-coplanar warn/reject tiers | 11 |
| §7 discrepancy (registration-floor diagnostic) | 9, 11 |
| §9 two-mode guidance / explicit refresh / step-and-measure | 1, 3, 7, 13, 14 |
| §10 quality report | 3, 5, 11, 16 |
| §10 hard gates (collision, stale pose, lock invalidation, calibration age) | 8, 11, 16 |
| §11 immutable contract, locked polygon sole source | 1, 3, 5, 6 |
| §12 supersessions (crop relabel, lock latch removal, advisory lamps) | 4, 6 |
| §13 phases | Milestones A/B/C mirror Phases 1-2 / 3 / 0+4 |
| §14 unit/synthetic matrix | tests in Tasks 1, 2, 9, 10, 11, 12, 15 |
| §14 mock integration matrix | tests in Tasks 3, 5, 8, 13 |
| §14 cell acceptance | on-cell, out of scope for agents — run per §14 after Milestone B lands |

Known deferred items (explicitly NOT in this plan): server-side live crop-square dimension sync (display-only mismatch, §12 note in Task 4); §13 Phase 5 workframe validation against an independent artifact (needs the cell); auto-driven corner viewpoints (§7 future note); optional edge-midpoint recovery captures for rounded/obstructed corners (§7 — the state machine's `recapture` covers the common weak-corner case; midpoint captures are a follow-up once the cell shows they're needed); the §15 open tolerance decisions — `survey_*` defaults above are engineering guesses to revisit after Phase 0 runs on the cell.
