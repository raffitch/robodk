"""From a fused table cloud to a working frame + a rectangle — pure numpy.

The scan fuses depth views (already in the robot **base frame**) into one cloud.
This module turns that cloud into the two things the user wants on a work surface:

  * a **reference frame** to program/jog in, and
  * the **oriented rectangle** of the surface, as a visual reference.

Frame convention (geometry-only, no marker — the locked design decision):

  * **Z** = the plane normal, oriented "up" (positive base-Z), i.e. off the surface.
  * the surface extent is an **oriented rectangle** — the minimum-area bounding
    rectangle of the in-plane inliers (convex hull + rotating calipers), whose edges
    align with the table's edges and which (unlike PCA) is correct for a square too.
  * **origin** = the rectangle corner **nearest the robot base origin** (0,0,0) — a
    deterministic, repeatable corner instead of an arbitrary one.
  * **X** = along the **longer** of the two edges meeting that corner; **Y = Z × X**.

This removes the in-plane yaw/corner ambiguity that pure geometry otherwise has, so
re-scanning the same table yields the same frame without a marker.

Unit-agnostic: it works in whatever units the points are in (the scan passes metres);
"nearest the base origin" holds in any units since the origin is the origin. No
RoboDK / open3d / cv2 here, so it is unit-testable on any machine.

NB: "up" assumes the robot base frame's +Z points up (true for this cell). If a cell's
base Z were not up, orient the normal toward the camera viewpoints instead.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...core.geometry import Rt_to_T


@dataclass
class WorkPlane:
    """The surface fit, in the same frame + units as the input points."""

    frame_T: np.ndarray       # 4x4 base->frame (origin + X/Y/Z per the convention)
    corners: np.ndarray       # (4,3) rectangle corners, cyclic order
    size: tuple[float, float]  # (length along frame X, width along frame Y)
    normal: np.ndarray        # unit plane normal == frame +Z
    centroid: np.ndarray      # (3,) inlier centroid
    inlier_count: int
    inlier_frac: float

    def to_dict(self) -> dict:
        return {
            "frame_T": self.frame_T.tolist(),
            "corners": self.corners.tolist(),
            "size": [float(self.size[0]), float(self.size[1])],
            "normal": self.normal.tolist(),
            "centroid": self.centroid.tolist(),
            "inlier_count": int(self.inlier_count),
            "inlier_frac": float(self.inlier_frac),
        }


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two orthonormal in-plane axes for a given unit normal (arbitrary roll)."""
    z = normal / np.linalg.norm(normal)
    a = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(a, z)
    u /= np.linalg.norm(u)
    v = np.cross(z, u)
    return u, v


def fit_plane(points: np.ndarray, *, distance: float = 0.006,
              n_iterations: int = 1000, seed: int = 0
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RANSAC the dominant plane, then least-squares refine on its inliers.

    Returns ``(normal_unit, centroid, inlier_mask)``. The normal is oriented to
    positive base-Z ("up"). Deterministic for a fixed ``seed``.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    n = len(pts)
    if n < 3:
        raise ValueError("need >= 3 points to fit a plane")
    rng = np.random.default_rng(seed)

    best_mask: np.ndarray | None = None
    best_count = -1
    for _ in range(max(1, int(n_iterations))):
        idx = rng.choice(n, 3, replace=False)
        p0, p1, p2 = pts[idx]
        nrm = np.cross(p1 - p0, p2 - p0)
        nn = float(np.linalg.norm(nrm))
        if nn < 1e-12:
            continue
        nrm = nrm / nn
        dist = np.abs((pts - p0) @ nrm)
        mask = dist < distance
        c = int(mask.sum())
        if c > best_count:
            best_count, best_mask = c, mask
    if best_mask is None or best_count < 3:
        raise ValueError("RANSAC failed to find a plane")

    # Least-squares refine: the plane through the inlier centroid whose normal is
    # the smallest-variance SVD direction (more accurate than the 3-point sample).
    inl = pts[best_mask]
    centroid = inl.mean(axis=0)
    _, _, vt = np.linalg.svd(inl - centroid, full_matrices=False)
    normal = vt[2]
    normal = normal / np.linalg.norm(normal)
    if normal[2] < 0:                      # orient "up" (positive base Z)
        normal = -normal
    # Re-select inliers against the refined plane (tighter, symmetric band).
    mask = np.abs((pts - centroid) @ normal) < distance
    return normal, centroid, mask


def _convex_hull_2d(pts: np.ndarray) -> np.ndarray:
    """Andrew's monotone-chain convex hull of ``(N,2)`` points (CCW, no scipy)."""
    p = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    if len(p) <= 2:
        return p

    def half(points):
        out: list = []
        for q in points:
            while len(out) >= 2:
                a, b = out[-2], out[-1]
                if (b[0] - a[0]) * (q[1] - a[1]) - (b[1] - a[1]) * (q[0] - a[0]) <= 0:
                    out.pop()
                else:
                    break
            out.append(q)
        return out

    lower = half(p)
    upper = half(p[::-1])
    return np.array(lower[:-1] + upper[:-1])


def _min_area_rectangle(pts2d: np.ndarray, *,
                        preferred_axis: np.ndarray | None = None,
                        area_tolerance: float = 0.02):
    """Minimum-area bounding rectangle of ``(N,2)`` points (rotating calipers over
    the convex-hull edges). Returns ``(ux, uy, w, h, lo_x, lo_y)``: the rectangle's
    unit axes (2D), its size along each, and the lower corner's projections — so the
    four corners are ``lo + i*w*ux + j*h*uy``. Correct for squares (picks an
    edge-aligned box, not a 45deg one)."""
    hull = _convex_hull_2d(np.asarray(pts2d, float).reshape(-1, 2))
    n = len(hull)
    if n < 3:                              # degenerate — fall back to an AABB
        lo, hi = pts2d.min(0), pts2d.max(0)
        return (np.array([1.0, 0]), np.array([0, 1.0]),
                float(hi[0] - lo[0]), float(hi[1] - lo[1]), float(lo[0]), float(lo[1]))
    candidates = []
    for i in range(n):
        edge = hull[(i + 1) % n] - hull[i]
        L = float(np.linalg.norm(edge))
        if L < 1e-12:
            continue
        ux = edge / L
        uy = np.array([-ux[1], ux[0]])
        px, py = pts2d @ ux, pts2d @ uy
        w, h = float(px.max() - px.min()), float(py.max() - py.min())
        area = w * h
        candidates.append((area, ux, uy, w, h, float(px.min()), float(py.min())))
    min_area = min(c[0] for c in candidates)
    near_best = [c for c in candidates
                 if c[0] <= min_area * (1.0 + max(0.0, float(area_tolerance)))]
    if preferred_axis is not None:
        pref = np.asarray(preferred_axis, float).reshape(2)
        pn = float(np.linalg.norm(pref))
        if pn > 1e-12:
            pref /= pn
            best = max(near_best, key=lambda c: max(abs(float(c[1] @ pref)),
                                                    abs(float(c[2] @ pref))))
        else:
            best = min(candidates, key=lambda c: c[0])
    else:
        best = min(candidates, key=lambda c: c[0])
    _, ux, uy, w, h, lox, loy = best
    return ux, uy, w, h, lox, loy


def _density_extent_1d(values: np.ndarray, lo: float, hi: float, *,
                       core_frac: float = 0.20, max_trim_frac: float = 0.15,
                       min_bin_mm: float = 4.0, min_points: int = 60):
    """Shrink a rectangle side's raw span ``[lo, hi]`` (1D positions along one axis,
    mm) inward to where point density rises to ``core_frac`` of the body's typical
    density.

    A symmetric quantile trim cuts the same small fraction off every edge, so a
    *sparse coplanar halo* just past the real board edge (flying TSDF pixels, depth
    spill — more than the 0.5% the quantile removes) survives and inflates the box,
    making it over-run the physical board. A real edge is a sharp density CLIFF: the
    first populated bin sits right at the board, and the halo bins beyond it are far
    below the body density, so thresholding at a fraction of the body keeps the board
    and drops the halo. Guarded so it cannot eat a real, fully-sampled edge: each side
    trims at most ``max_trim_frac`` of the span, and the whole step is skipped below
    ``min_points`` (too few points to estimate density)."""
    span = float(hi - lo)
    n = int(np.size(values))
    if span <= 0.0 or n < min_points:
        return float(lo), float(hi)
    n_bins = max(8, int(round(span / max(min_bin_mm, span / 80.0))))
    counts, edges = np.histogram(values, bins=n_bins, range=(float(lo), float(hi)))
    occupied = counts[counts > 0]
    if len(occupied) == 0:
        return float(lo), float(hi)
    core = float(np.median(occupied))            # typical density of a populated bin
    thresh = max(1.0, core_frac * core)          # a halo bin sits well below the body
    above = np.where(counts >= thresh)[0]
    if len(above) == 0:
        return float(lo), float(hi)
    new_lo = float(edges[int(above[0])])
    new_hi = float(edges[int(above[-1]) + 1])
    cap = max_trim_frac * span                   # never trim more than this per side
    new_lo = min(new_lo, lo + cap)
    new_hi = max(new_hi, hi - cap)
    return new_lo, new_hi


def _oriented_rectangle(points: np.ndarray, normal: np.ndarray, centroid: np.ndarray):
    """Minimum-area oriented rectangle of ``points`` projected onto the plane.

    Returns ``(corners (4,3) cyclic, ax1, ax2, len1, len2)`` where ``ax1`` is the
    longer edge direction. Corners are ordered so consecutive corners share an edge.
    """
    u, v = _plane_basis(normal)
    rel = np.asarray(points, float).reshape(-1, 3) - centroid
    coords = np.column_stack([rel @ u, rel @ v])     # (N,2) in the (u,v) plane basis
    # TSDF silhouettes contain sparse flying pixels and view-dependent edge
    # fragments. Trim only the extreme fringe so those points cannot rotate the
    # whole work frame diagonally.
    if len(coords) >= 100:
        lo = np.quantile(coords, 0.005, axis=0)
        hi = np.quantile(coords, 0.995, axis=0)
        keep = np.all((coords >= lo) & (coords <= hi), axis=1)
        if int(keep.sum()) >= max(20, int(0.8 * len(coords))):
            coords = coords[keep]

    # A square or partially observed plane can have multiple effectively equal
    # minimum-area boxes. Resolve that ambiguity toward the robot-base axes.
    base_x = np.array([1.0, 0.0, 0.0])
    preferred_2d = np.array([base_x @ u, base_x @ v])
    if np.linalg.norm(preferred_2d) < 1e-6:
        base_y = np.array([0.0, 1.0, 0.0])
        preferred_2d = np.array([base_y @ u, base_y @ v])
    ux, uy, w, h, lox, loy = _min_area_rectangle(
        coords, preferred_axis=preferred_2d)
    # Rectangle axes back in 3D (ux/uy are rotated axes *within* the (u,v) plane).
    ax_a = u * ux[0] + v * ux[1]
    ax_b = u * uy[0] + v * uy[1]

    # Refine the extent per edge: pull each side in to the density cliff, dropping a
    # sparse coplanar halo just past the real board edge that the symmetric quantile
    # trim above leaves in (which makes the box over-run the board). Orientation is
    # already fixed by the min-area rectangle on the fringe-trimmed cloud; this only
    # tightens how far each side reaches, and is guarded so it cannot eat a real edge.
    pa = coords @ ux
    pb = coords @ uy
    lo_a, hi_a = _density_extent_1d(pa, lox, lox + w)
    lo_b, hi_b = _density_extent_1d(pb, loy, loy + h)
    lox, w = lo_a, hi_a - lo_a
    loy, h = lo_b, hi_b - lo_b

    def to3d(coord2d: np.ndarray) -> np.ndarray:
        # coord2d is in the (u,v) plane basis -> back to 3D.
        return centroid + coord2d[0] * u + coord2d[1] * v

    # Corners as positions in the (u,v) plane (lo + offsets along ux/uy), cyclic.
    corners = np.array([
        to3d(lox * ux + loy * uy),
        to3d((lox + w) * ux + loy * uy),
        to3d((lox + w) * ux + (loy + h) * uy),
        to3d(lox * ux + (loy + h) * uy),
    ])
    if w >= h:
        return corners, ax_a, ax_b, w, h
    return corners, ax_b, ax_a, h, w


def work_plane_from_points(points: np.ndarray, *, distance: float = 0.006,
                           n_iterations: int = 1000, seed: int = 0,
                           min_inlier_frac: float = 0.25) -> WorkPlane:
    """Fit the surface and build the working frame + rectangle (the convention above).

    Raises ``ValueError`` if the dominant plane claims fewer than ``min_inlier_frac``
    of the points (no trustworthy single surface — e.g. clutter dominates the cloud).
    """
    pts = np.asarray(points, float).reshape(-1, 3)
    normal, centroid, mask = fit_plane(pts, distance=distance,
                                       n_iterations=n_iterations, seed=seed)
    frac = float(mask.mean())
    if frac < min_inlier_frac:
        raise ValueError(
            f"dominant plane covers only {frac:.0%} of the cloud (< {min_inlier_frac:.0%}) "
            f"— no clear work surface; re-scan with the table filling more of the views")
    inl = pts[mask]
    corners, ax1, ax2, len1, len2 = _oriented_rectangle(inl, normal, centroid)

    # Origin = the corner nearest the base origin (deterministic + repeatable).
    o = int(np.argmin(np.linalg.norm(corners, axis=1)))
    origin = corners[o]
    nxt, prv = corners[(o + 1) % 4], corners[(o - 1) % 4]
    edge_nxt, edge_prv = nxt - origin, prv - origin
    # X = along the LONGER of the two edges meeting the origin corner.
    x_raw = edge_nxt if np.linalg.norm(edge_nxt) >= np.linalg.norm(edge_prv) else edge_prv
    z = normal / np.linalg.norm(normal)
    x = x_raw - (x_raw @ z) * z            # re-orthogonalise against the plane normal
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)                     # right-handed [x, y, z]
    frame_T = Rt_to_T(np.column_stack([x, y, z]), origin)

    return WorkPlane(frame_T=frame_T, corners=corners,
                     size=(max(len1, len2), min(len1, len2)),
                     normal=z, centroid=centroid,
                     inlier_count=int(mask.sum()), inlier_frac=frac)


def bounded_work_plane(wp: WorkPlane, center: np.ndarray,
                       size: tuple[float, float]) -> WorkPlane:
    """Limit an already-fitted plane to a centered rectangular work region.

    Used when the physical plane extends beyond the camera frame (for example a
    large table). The plane inclination remains measured; only the programmable
    footprint is bounded around the surface under the camera reticle.
    """
    z = np.asarray(wp.normal, float)
    z /= np.linalg.norm(z)
    c = np.asarray(center, float).reshape(3)
    c = c - float((c - wp.centroid) @ z) * z
    x = np.asarray(wp.frame_T[:3, 0], float)
    x -= float(x @ z) * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    length, width = float(size[0]), float(size[1])
    if width > length:
        x, y = y, -x
        length, width = width, length
    hx, hy = length / 2.0, width / 2.0
    corners = np.array([
        c - hx * x - hy * y,
        c + hx * x - hy * y,
        c + hx * x + hy * y,
        c - hx * x + hy * y,
    ])
    o = int(np.argmin(np.linalg.norm(corners, axis=1)))
    origin = corners[o]
    nxt, prv = corners[(o + 1) % 4], corners[(o - 1) % 4]
    x_raw = (nxt - origin if np.linalg.norm(nxt - origin) >= np.linalg.norm(prv - origin)
             else prv - origin)
    x_axis = x_raw - float(x_raw @ z) * z
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z, x_axis)
    frame_T = Rt_to_T(np.column_stack([x_axis, y_axis, z]), origin)
    return WorkPlane(
        frame_T=frame_T, corners=corners, size=(length, width),
        normal=z, centroid=c, inlier_count=wp.inlier_count,
        inlier_frac=wp.inlier_frac)
