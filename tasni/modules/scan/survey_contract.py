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

# --- Workflow goal / surface scope (adaptive-scan plan Task 2) ----------------
# Three INDEPENDENT concepts; do not overload ``mode`` above, which records how the
# geometry was acquired (its provenance) and nothing else.
#
#   goal  -- does the user need only the working frame, or also dense surface data?
#   scope -- must every physical boundary be measured, or is a sized ROI acceptable?
#   mode  -- (above) the provenance-bearing measurement path that was actually used.
GOAL_FRAME_ONLY = "frame_only"
GOAL_FULL_SCAN = "full_scan"
WORKFLOW_GOALS = (GOAL_FRAME_ONLY, GOAL_FULL_SCAN)

SCOPE_ENTIRE_PLATFORM = "entire_platform"
SCOPE_DECLARED_REGION = "declared_region"
SURFACE_SCOPES = (SCOPE_ENTIRE_PLATFORM, SCOPE_DECLARED_REGION)

# Legacy ``SurfaceLockBody.mode`` values map onto scope (goal defaults to full_scan,
# which is what every pre-Task-2 client did).
LEGACY_MODE_TO_SCOPE = {"auto": SCOPE_ENTIRE_PLATFORM, "crop": SCOPE_DECLARED_REGION}


def lock_fingerprint(goal: str, scope: str, region_mm=None) -> str:
    """Identity of the *intent* behind a lock (plan Task 2).

    Folded into the lock token so that changing goal or scope invalidates already
    prepared results and already generated scan targets: a token minted under one
    intent can never be mistaken for a token minted under another.
    """
    region = "" if region_mm is None else ",".join(
        f"{float(v):.3f}" for v in tuple(region_mm))
    payload = f"{goal}|{scope}|{region}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:10]


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
    # Task 19 (Defect 1a): rdk.current_joints() is a flat sequence in every fake/
    # test, but on REAL RoboDK it returns a robomath.Mat wrapping an Nx1 COLUMN
    # vector. Mat.__len__ reports the COLUMN count (1 for a column vector), so
    # np.asarray(mat, dtype=float) does not build the flat (N,) array this code
    # assumed -- it builds a (1, N) array (one row holding all N joints) instead.
    # Iterating that row then hands `float()` the WHOLE N-element vector in one
    # call ("only size-1 arrays can be converted to Python scalars") rather than
    # one joint at a time -- reproduced hardware-free in test_survey_contract.py
    # (test_refresh_robot_state_handles_robodk_mat_shaped_joints) using the real
    # robomath.Mat class. .reshape(-1) collapses (N,), (1, N) and (N, 1) alike to
    # the same flat N-vector, so this is correct regardless of which shape a given
    # current_joints() implementation returns.
    j0 = np.asarray(rdk.current_joints(), dtype=float).reshape(-1)
    sleep(settle_s)
    j1 = np.asarray(rdk.current_joints(), dtype=float).reshape(-1)
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


# Relative length difference below which the two edges meeting C1 count as "equal"
# and the length rule is abandoned (see _tie_break_edge). 0.5% of a 500 mm side is
# 2.5 mm -- about depth-noise scale, so a physically square platform lands inside the
# band while a genuinely oblong rectangle (>=0.5% aspect difference) never does.
EDGE_TIE_REL_TOL = 0.005


def _tie_break_edge(e_next, e_prev, l_next, l_prev):
    """Pick +X when both edges meeting C1 are the same length to within tolerance.

    On a square the longer-edge rule carries no signal: measurement noise decides it,
    so the frame flips 90 deg between scans of the same platform. Fall back to the
    edge better aligned with base +X (then base +Y), which noise cannot flip.
    """
    un, up = e_next / l_next, e_prev / l_prev
    for ref in (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])):
        an, ap = abs(float(un @ ref)), abs(float(up @ ref))
        if abs(an - ap) > 1e-9:
            return e_next if an > ap else e_prev
    return e_next  # exactly diagonal to both base axes -- canonical order decides


def workframe_from_rectangle(corners_base, normal_base) -> np.ndarray:
    """THE one workframe convention (adaptive-scan plan Task 1). 4x4 base->frame.

    * origin = ``C1``, the rectangle corner nearest the robot base origin;
    * +X = the LONGER of the two rectangle edges **meeting C1**;
    * +Z = the up-oriented surface normal;
    * +Y = Z x X (right-handed).

    This is the convention already deployed by :mod:`plane` and RoboDK insertion, so
    frames inserted by past runs keep their meaning. It replaces the centre-origin
    frame this module used to build, which was never the one that got inserted.

    Note the "longer edge **meeting C1**" wording: that is not the same as the
    rectangle's global long edge whenever the two rules disagree, which they do on a
    near-square. Corners are canonicalised first, so the result depends only on the
    rectangle's geometry -- never on the order the caller happens to pass corners in.
    That order-independence is what makes every acquisition path agree.
    """
    c = order_corners_clockwise(corners_base, normal_base)
    n = _up_normal(normal_base)
    origin = c[0]
    e_next, e_prev = c[1] - origin, c[3] - origin
    l_next = max(float(np.linalg.norm(e_next)), 1e-12)
    l_prev = max(float(np.linalg.norm(e_prev)), 1e-12)
    if abs(l_next - l_prev) <= EDGE_TIE_REL_TOL * max(l_next, l_prev):
        x = _tie_break_edge(e_next, e_prev, l_next, l_prev)
    else:
        x = e_next if l_next > l_prev else e_prev
    x = x - n * float(x @ n)                 # re-orthogonalise against the normal
    x = x / max(float(np.linalg.norm(x)), 1e-12)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, np.cross(n, x), n, origin
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
