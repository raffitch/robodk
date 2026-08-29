"""Extract base-frame corner/edge evidence from one corner capture (spec §7).

The boundary polygon (colour/SAM) proposes WHERE the physical edge is; depth +
the calibrated camera pose provide METRIC geometry (spec §11). Samples are
inset a few pixels toward the surface interior so depth is read on the
platform, not in the discontinuity at its edge.

Units: ``registered`` (a :class:`~tasni.core.depth_geometry.ColorRegistered`, Task 7)
carries the colour-registered depth points already converted to **millimetres**
through the frame's own ``CameraGeometry.depth_unit_mm``; ``polygon_uv`` is
normalized ``(N, 2)`` image coordinates; ``T_base_cam`` is the 4x4 base<-camera pose
in millimetres. All outputs (``corner_base_mm``, ``edge_points_base``) are
millimetres in the robot base frame.

This module now DOES go through the same ``backproject`` primitive
``service._backproject_depth`` wraps (Task 7's ``core.depth_geometry``) --
indirectly, via the ``ColorRegistered`` the caller builds. The reason it used to
opt out no longer applies: the old raw-depth-is-mm convention was ambiguous by
construction (a caller had to know the frame's units), whereas ``CameraGeometry``
now makes the unit explicit and SHARED by every consumer (this module, the scan
gate, the TSDF fuse). ``median_z_near`` reads the median of the ALREADY-mm points
registered near a colour pixel -- there is nothing left to duplicate.

Interior direction (which side of a boundary segment is "the surface"):
derived PURELY LOCALLY, never from the mean of the whole polygon's vertices.
A production ``polygon_uv`` (from ``color_boundary.py`` / SAM) is a contour
that may be frame-clipped into an arbitrarily non-convex shape, so a global
vertex mean can land outside the surface entirely (e.g. an L-shaped visible
region whose vertex centroid sits in the missing corner) -- insetting toward
it would then sample the background on every point while still returning a
confident, finite result. Instead each boundary segment's inward normal is
derived from the polygon's own consistent winding order (its signed area,
computed once): interior is always to a fixed, consistent side of every
*forward-array-order* directed edge, regardless of whether the sampled
vertex is locally convex or reflex. This generalizes the simpler "bisector
of the two arm directions" idea (which only holds at convex vertices) to
reflex corners too, which real frame-clipped contours can plausibly produce.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CornerEvidence:
    corner_uv: tuple[float, float]
    corner_base_mm: tuple[float, float, float] | None
    edge_points_base: np.ndarray  # (N, 3) pooled from both arms


# A polygon whose bounding-box span is under this many pixels has no usable
# arm length to walk -- treated as degenerate regardless of vertex count.
_MIN_POLY_SPAN_PX = 2.0

# The two arm directions at a corner are unit vectors pointing away from the
# corner along the boundary. For unit vectors, |d_plus + d_minus| = 2*cos(theta/2)
# where theta is the angle between them. A threshold of 0.15 means the guard
# fires once theta exceeds ~171.4 degrees (arms within ~8.6 degrees of exactly
# collinear) -- i.e. not a real corner, just a point on a near-straight run.
_MIN_ARM_BISECTOR_NORM = 0.15


def _deproject_base(u_px, v_px, z_mm, K, T_base_cam) -> np.ndarray:
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    p_cam = np.array([(u_px - cx) / fx * z_mm, (v_px - cy) / fy * z_mm, z_mm, 1.0])
    return (np.asarray(T_base_cam, dtype=float) @ p_cam)[:3]


def _signed_area2(poly_px: np.ndarray) -> float:
    """Twice the signed area of ``poly_px`` (shoelace, closing last->first).

    Sign gives the polygon's winding order; used only to pick a consistent
    "interior side" for edge normals, not to test true geometric closure.
    Meaningful even for an open arc of just a few vertices around one corner
    (the implicit closing edge only needs to be *consistent*, not physical).
    """
    x, y = poly_px[:, 0], poly_px[:, 1]
    return float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _neighbor_index(n: int, idx: int, step: int, closed: bool) -> int | None:
    j = idx + step
    if closed:
        return j % n
    return j if 0 <= j < n else None


def _walk_arm(poly_px, start_idx, step, arm_len_px, n_samples, *,
              closed: bool, orientation_sign: float, max_turn_deg: float):
    """Sample (point, inward-normal) pairs along the polyline from start_idx
    toward `step`.

    - On an OPEN polyline (``closed=False``) the walk stops as soon as the
      next index would leave ``[0, len(poly_px))`` -- never indexes out of
      range, never wraps.
    - On a CLOSED polygon (``closed=True``) indices wrap modulo
      ``len(poly_px)`` instead, bounded by a hard step cap (``2 * n``) so a
      degenerate/duplicate-point loop can't spin forever.
    - Terminates EARLY (keeping points collected so far) once a segment's
      heading turns more than ``max_turn_deg`` away from the arm's own first
      segment direction. This is how a frame-clipped contour that starts
      tracing the image border (a near-90 degree turn) gets excluded rather
      than sampled as if it were real edge evidence.

    Returns ``(points (M, 2), normals (M, 2))`` -- ``normals`` are unit
    vectors, each already oriented to point toward the polygon's interior
    (via ``orientation_sign``, see ``_signed_area2``) for that point's own
    local segment, independent of whether this vertex is convex or reflex.
    """
    n = len(poly_px)
    pts, normals = [], []
    travelled = 0.0
    i = start_idx
    target = np.linspace(arm_len_px / n_samples, arm_len_px, n_samples)
    ti = 0
    initial_tangent = None  # forward-array-order unit tangent of first segment
    steps_taken = 0
    max_steps = 2 * n if closed else n
    cos_limit = float(np.cos(np.radians(max_turn_deg)))

    while ti < len(target) and steps_taken < max_steps:
        j = i + step
        if closed:
            j %= n
        elif not (0 <= j < n):
            break
        a, b = poly_px[i], poly_px[j]
        seg_vec = b - a
        seg_len = float(np.linalg.norm(seg_vec))
        if seg_len > 1e-9:
            forward_unit = (seg_vec if step > 0 else -seg_vec) / seg_len
            if initial_tangent is None:
                initial_tangent = forward_unit
            else:
                cos_turn = float(np.dot(forward_unit, initial_tangent))
                if cos_turn < cos_limit:
                    break  # sharp turn away from this arm's heading: stop
            left = np.array([-forward_unit[1], forward_unit[0]])
            normal = left if orientation_sign > 0 else -left
            while ti < len(target) and travelled + seg_len >= target[ti]:
                t = (target[ti] - travelled) / max(seg_len, 1e-9)
                pts.append(a + t * seg_vec)
                normals.append(normal)
                ti += 1
            travelled += seg_len
        i = j
        steps_taken += 1

    pts_arr = np.asarray(pts, dtype=float).reshape(-1, 2)
    normals_arr = np.asarray(normals, dtype=float).reshape(-1, 2)
    return pts_arr, normals_arr


def extract_corner_evidence(registered, K, polygon_uv, T_base_cam, *,
                             corner_hint_uv=(0.5, 0.5), arm_frac: float = 0.35,
                             samples_per_arm: int = 40, inset_px: float = 4.0,
                             window_px: int = 3, min_valid_frac: float = 0.3,
                             min_points_per_arm: int = 4,
                             max_arm_turn_deg: float = 60.0,
                             closed: bool = False):
    """Extract corner + pooled edge evidence in the robot base frame, or None.

    ``min_points_per_arm`` (new): BOTH arms must independently contribute at
    least this many depth-valid points, or the whole result is discarded.
    Without this, a corner near either end of the vertex array could return
    single-arm evidence pooled and presented as if both arms were sampled --
    a caller reading ``edge_points_base`` has no way to tell.

    ``max_arm_turn_deg`` (new): an arm's walk stops as soon as its local
    heading turns more than this many degrees from its own first segment
    (default 60 deg). A frame-clipped contour that runs out of real surface
    boundary starts tracing the image border instead -- typically a near-90
    degree turn -- and those border-following vertices carry no edge signal;
    60 degrees comfortably tolerates a real edge's natural curvature/noise
    while catching a border turn.

    ``closed`` (new, default False): ``polygon_uv`` is treated as an open
    arc by default (matching this module's existing unit tests, which pass
    small hand-built two-arm fragments, not full contours). Pass
    ``closed=True`` when the polygon is a genuine closed contour (e.g. a
    full SAM/color-boundary segmentation) so a corner near the array's
    start/end still wraps around to find its other arm. This is a caller
    fact, not auto-detected: production contours from
    ``cv2.approxPolyDP(..., closed=True)`` do not duplicate their closing
    vertex, so a position-based "are the endpoints coincident" heuristic
    would almost never fire on real data -- it would look reassuring in
    tests while doing nothing useful in production. ``min_points_per_arm``
    is the actual safety net regardless of this flag: with ``closed=False``
    and a corner genuinely at an array end, the missing arm's evidence is
    zero and the function correctly returns None rather than presenting
    single-arm data as pooled.

    ``registered`` (new): a :class:`~tasni.core.depth_geometry.ColorRegistered` --
    the caller's colour-registered depth points for this frame -- REPLACES the raw
    ``depth`` array. ``window_px`` is therefore a search RADIUS over those points'
    own (float, sub-pixel) colour positions (``ColorRegistered.median_z_near``), not
    a pixel BOX on the depth image; its default moved 2 -> 3 because native depth is
    coarser than colour and sparser once registered, so a slightly wider radius is
    needed to reliably find neighbours near a sample point.
    """
    if registered is None or len(registered) == 0:
        return None
    w, h = registered.color_size
    poly = np.asarray(polygon_uv, dtype=float).reshape(-1, 2)
    if len(poly) < 3 or not np.all(np.isfinite(poly)):
        return None
    poly_px = poly * [w, h]
    if float(np.ptp(poly_px, axis=0).max()) < _MIN_POLY_SPAN_PX:
        return None

    n = len(poly_px)
    hint_px = np.asarray(corner_hint_uv, dtype=float) * [w, h]
    corner_idx = int(np.argmin(np.linalg.norm(poly_px - hint_px, axis=1)))
    corner_px = poly_px[corner_idx]

    nbr_plus = _neighbor_index(n, corner_idx, +1, closed)
    nbr_minus = _neighbor_index(n, corner_idx, -1, closed)
    if nbr_plus is None or nbr_minus is None:
        return None  # corner sits at an array end with no arm on one side

    d_plus_vec = poly_px[nbr_plus] - corner_px
    d_minus_vec = poly_px[nbr_minus] - corner_px
    d_plus_norm = float(np.linalg.norm(d_plus_vec))
    d_minus_norm = float(np.linalg.norm(d_minus_vec))
    if d_plus_norm < 1e-9 or d_minus_norm < 1e-9:
        return None
    d_plus = d_plus_vec / d_plus_norm
    d_minus = d_minus_vec / d_minus_norm
    if float(np.linalg.norm(d_plus + d_minus)) < _MIN_ARM_BISECTOR_NORM:
        return None  # arms are ~collinear: not a real corner

    orientation_sign = 1.0 if _signed_area2(poly_px) >= 0 else -1.0
    arm_len_px = arm_frac * float(np.hypot(w, h))

    arm_pts_plus, arm_normals_plus = _walk_arm(
        poly_px, corner_idx, +1, arm_len_px, samples_per_arm,
        closed=closed, orientation_sign=orientation_sign,
        max_turn_deg=max_arm_turn_deg)
    arm_pts_minus, arm_normals_minus = _walk_arm(
        poly_px, corner_idx, -1, arm_len_px, samples_per_arm,
        closed=closed, orientation_sign=orientation_sign,
        max_turn_deg=max_arm_turn_deg)

    pts_base_per_arm = [[], []]
    n_requested = 0
    for arm_i, (arm_pts, arm_normals) in enumerate(
            ((arm_pts_plus, arm_normals_plus), (arm_pts_minus, arm_normals_minus))):
        for p, normal in zip(arm_pts, arm_normals):
            n_requested += 1
            sample = p + inset_px * normal
            px, py = int(round(sample[0])), int(round(sample[1]))
            if not (0 <= px < w and 0 <= py < h):
                continue
            z = registered.median_z_near(px, py, window_px)
            if not np.isfinite(z) or z <= 0:
                continue
            point = _deproject_base(sample[0], sample[1], z, K, T_base_cam)
            if not np.all(np.isfinite(point)):
                continue
            pts_base_per_arm[arm_i].append(point)

    if n_requested == 0:
        return None
    n_plus_pts, n_minus_pts = len(pts_base_per_arm[0]), len(pts_base_per_arm[1])
    total_pts = n_plus_pts + n_minus_pts
    if total_pts < max(4, int(min_valid_frac * n_requested)):
        return None
    if n_plus_pts < min_points_per_arm or n_minus_pts < min_points_per_arm:
        return None  # single-arm evidence must never be presented as pooled

    # Finding 4: the corner sample gets the SAME protection as edge samples
    # -- inset off the raw boundary intersection (the point most likely to
    # straddle a depth discontinuity) using the same window size, not a
    # larger one. Its inset direction is the corner's own local bisector,
    # built from the two arms' already-oriented first-segment normals (so
    # it is correct at a reflex corner too, not just a convex one).
    normal_plus0 = arm_normals_plus[0] if len(arm_normals_plus) else None
    normal_minus0 = arm_normals_minus[0] if len(arm_normals_minus) else None
    if normal_plus0 is not None and normal_minus0 is not None:
        corner_bisector_raw = normal_plus0 + normal_minus0
        cb_norm = float(np.linalg.norm(corner_bisector_raw))
        corner_bisector = (corner_bisector_raw / cb_norm if cb_norm > 1e-6
                            else normal_plus0)
    else:
        corner_bisector = normal_plus0 if normal_plus0 is not None else normal_minus0

    corner_base = None
    if corner_bisector is not None:
        corner_sample = corner_px + inset_px * corner_bisector
        cpx, cpy = int(round(corner_sample[0])), int(round(corner_sample[1]))
        if 0 <= cpx < w and 0 <= cpy < h:
            zc = registered.median_z_near(cpx, cpy, window_px)
            if np.isfinite(zc) and zc > 0:
                cb = _deproject_base(corner_sample[0], corner_sample[1], zc, K, T_base_cam)
                if np.all(np.isfinite(cb)):
                    corner_base = tuple(float(v) for v in cb)

    edge_points_base = np.asarray(pts_base_per_arm[0] + pts_base_per_arm[1], dtype=float)
    # Invariant, not a guard: every point above already passed a per-point
    # isfinite check before being appended, and the length gate above
    # guarantees at least 2 * min_points_per_arm entries.
    assert edge_points_base.size and np.all(np.isfinite(edge_points_base))

    return CornerEvidence(corner_uv=(float(corner_px[0] / w), float(corner_px[1] / h)),
                           corner_base_mm=corner_base,
                           edge_points_base=edge_points_base)
