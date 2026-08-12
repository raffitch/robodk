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
