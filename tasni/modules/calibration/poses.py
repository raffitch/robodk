"""Auto-generate calibration viewpoints around a seed camera pose.

The seed is the robot's current (live-gated) pose — the operator jogs until the
aiming HUD locks green, and these poses orbit that view. Why not the old dome
macro: it aligned every pose's Z to a look-at point but added NO roll and swept a
full 360deg azimuth — which (a) under-constrains the hand-eye rotation solve
(rotation axes end up near-coplanar) and (b) orbits the camera to the board's
unprintable back. Here we instead sample a CONE around the seed viewing direction
(board stays visible), with deliberate ROLL about the optical axis and DISTANCE
variation — the diversity hand-eye actually needs.

Pure numpy + reproducible (deterministic spiral, no RNG). Poses are TCP/camera
poses in the same frame as ``seed_T``; the caller filters them by reachability.
"""
from __future__ import annotations

import numpy as np

from ...core.geometry import Rt_to_T, T_to_Rt, invert_T, transform_points

_GOLDEN_ANGLE = np.pi * (3.0 - np.sqrt(5.0))  # ~2.39996 rad, even azimuthal spread


def _basis_from_z(z: np.ndarray, up_hint: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z = z / np.linalg.norm(z)
    if abs(float(np.dot(z, up_hint))) > 0.95:           # up_hint ~parallel to z
        up_hint = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0])
    x = np.cross(up_hint, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return x, y


def _apply_roll(x: np.ndarray, y: np.ndarray, deg: float) -> tuple[np.ndarray, np.ndarray]:
    a = np.deg2rad(deg)
    return np.cos(a) * x + np.sin(a) * y, -np.sin(a) * x + np.cos(a) * y


def generate_calibration_poses(
    seed_T: np.ndarray, *, count: int = 15, look_distance_mm: float = 500.0,
    cone_half_angle_deg: float = 32.0, roll_max_deg: float = 75.0,
    distance_jitter: float = 0.12, oversample: int = 3,
) -> list[np.ndarray]:
    """Candidate camera poses orbiting the point ``seed`` looks at.

    Returns ``count * oversample`` candidates (deterministic), ordered so the
    first is closest to the seed view; the caller keeps the first ``count`` that
    are reachable. ``seed_T`` is the camera (TCP) pose; its +Z points at the board.
    """
    seed_R, seed_pos = T_to_Rt(seed_T)
    fwd, up = seed_R[:, 2], seed_R[:, 1]                 # camera forward, up
    right = seed_R[:, 0]
    center = seed_pos + look_distance_mm * fwd           # ~board location
    half = np.deg2rad(cone_half_angle_deg)

    n = max(count * oversample, count)
    poses: list[np.ndarray] = []
    for i in range(n):
        frac = (i + 0.5) / n
        theta = half * np.sqrt(frac)                     # polar from seed dir (denser center)
        phi = i * _GOLDEN_ANGLE                          # azimuth
        roll = roll_max_deg * (((i * 0.6180339887) % 1.0) * 2 - 1)
        dist = look_distance_mm * (1 + distance_jitter * (((i * 0.3819660113) % 1.0) * 2 - 1))

        # viewpoint direction from the board center, within the cone around -fwd
        d = (-np.cos(theta) * fwd
             + np.sin(theta) * (np.cos(phi) * right + np.sin(phi) * up))
        cam_pos = center + dist * d
        z = center - cam_pos                             # look back at the board
        x, y = _basis_from_z(z, up)
        x, y = _apply_roll(x, y, roll)
        poses.append(Rt_to_T(np.column_stack([x, y, z / np.linalg.norm(z)]), cam_pos))
    return poses


def _rotation_geodesic(Ri: np.ndarray, Rj: np.ndarray) -> float:
    """Geodesic angle (rad) between two rotations: the magnitude of the relative
    rotation ``Ri^T Rj``. This is the metric hand-eye conditioning cares about — it
    counts a difference in viewing tilt AND a difference in roll about the optical
    axis, so spreading on it spreads all three rotation axes (not just +Z)."""
    c = (float(np.trace(Ri.T @ Rj)) - 1.0) / 2.0
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


def select_diverse(poses: list[np.ndarray], count: int, *,
                   seed_fwd: np.ndarray | None = None) -> list[int]:
    """Pick ``count`` poses whose ORIENTATIONS are maximally spread.

    The caller passes the *reachable* candidates (already IK-filtered). Selecting
    the first ``count`` of them is biased: the spiral orders poses centre-outward,
    so the innermost (narrowest-cone) poses win and the kept set clusters near the
    seed — exactly the low rotational diversity that starves the hand-eye solve.
    Instead we run farthest-point sampling on the full camera rotation (geodesic
    angle between rotation matrices), which spreads the kept set across BOTH the
    cone tilt (rotation axes in the camera X-Y plane) AND the roll about the optical
    axis (the third, Z, axis). The earlier version sampled the +Z viewing direction
    only and ignored roll, so the kept set's roll was incidental — yet roll supplies
    the one rotation axis the cone tilt cannot, the axis that lifts the AX=XB
    conditioning toward isotropic.

    The first pick is anchored at the most fronto-parallel pose (closest to
    ``seed_fwd``, else ``poses[0]``) so at least one easy-to-detect view survives.
    Returns indices into ``poses``, sorted ascending for stable target naming.
    """
    n = len(poses)
    if count >= n:
        return list(range(n))
    Rs = [np.asarray(T, float)[:3, :3] for T in poses]
    if seed_fwd is not None:
        sf = np.asarray(seed_fwd, float)
        sf = sf / np.linalg.norm(sf)
        fwd = [R[:, 2] / np.linalg.norm(R[:, 2]) for R in Rs]
        start = int(np.argmax([float(np.dot(f, sf)) for f in fwd]))
    else:
        start = 0
    chosen = [start]
    # geodesic angle from each pose to its nearest already-chosen pose
    d = [_rotation_geodesic(Rs[i], Rs[start]) for i in range(n)]
    while len(chosen) < count:
        nxt = int(np.argmax(d))
        chosen.append(nxt)
        for i in range(n):
            di = _rotation_geodesic(Rs[i], Rs[nxt])
            if di < d[i]:
                d[i] = di
    return sorted(chosen)


def viewing_angle_span(poses: list[np.ndarray], seed_fwd: np.ndarray
                       ) -> tuple[float, float, float]:
    """(min, max, mean) angle in degrees between each pose's +Z and ``seed_fwd``.

    Quantifies how much of the configured cone a pose set actually covers — the
    *effective* cone, which (at a workspace-edge seed where wide poses are
    unreachable) can be far narrower than ``cone_half_angle_deg``.
    """
    sf = np.asarray(seed_fwd, float)
    sf = sf / np.linalg.norm(sf)
    angs = []
    for T in poses:
        f = np.asarray(T, float)[:3, 2]
        f = f / np.linalg.norm(f)
        angs.append(float(np.degrees(np.arccos(np.clip(float(np.dot(f, sf)), -1, 1)))))
    if not angs:
        return (0.0, 0.0, 0.0)
    return (min(angs), max(angs), sum(angs) / len(angs))


def project_pinhole(cam_T: np.ndarray, pts_base: np.ndarray, K: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Project base-frame points into a camera whose pose (in base) is ``cam_T``.

    Pure-numpy pinhole, no lens distortion — adequate for an in/out-of-frame margin
    gate on the low-distortion D4xx RGB lens (edge distortion is far below the safety
    margin). Returns ``(uv (N,2), in_front (N,) bool)`` where ``in_front`` flags
    points ahead of the image plane (z > 0 in the camera frame)."""
    K = np.asarray(K, dtype=float)
    Tcb = invert_T(cam_T)                       # base -> camera
    pc = transform_points(Tcb, pts_base)        # (N,3) in the camera frame
    z = pc[:, 2]
    in_front = z > 1e-6
    zc = np.where(in_front, z, 1.0)             # avoid div-by-zero behind the camera
    u = K[0, 0] * pc[:, 0] / zc + K[0, 2]
    v = K[1, 1] * pc[:, 1] / zc + K[1, 2]
    return np.column_stack([u, v]), in_front


def board_visible_fraction(cam_T: np.ndarray, board_pts_base: np.ndarray,
                           K: np.ndarray, image_size: tuple, *,
                           margin_frac: float = 0.04) -> float:
    """Fraction of board points landing in front of the camera AND inside the frame
    inset by ``margin_frac`` on each side.

    1.0 = the whole board projects safely inside the frame at this pose; a low value
    means the board clips an edge or falls outside the view — a pose the robot could
    reach without ever seeing the board. ``image_size`` is ``(width, height)`` px.
    The caller derives ``board_pts_base`` (board points in the base frame) once from
    the seed detection and reuses it for every candidate."""
    pts = np.asarray(board_pts_base, dtype=float).reshape(-1, 3)
    if pts.shape[0] == 0:
        return 0.0
    w, h = float(image_size[0]), float(image_size[1])
    uv, in_front = project_pinhole(cam_T, pts, K)
    mx, my = margin_frac * w, margin_frac * h
    inside = ((uv[:, 0] >= mx) & (uv[:, 0] <= w - mx)
              & (uv[:, 1] >= my) & (uv[:, 1] <= h - my))
    ok = in_front & inside
    return float(np.count_nonzero(ok)) / pts.shape[0]
