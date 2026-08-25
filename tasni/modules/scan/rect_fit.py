"""Global plane + constrained rectangle fitting for the five-position survey (spec §7).

All inputs/outputs are millimetres. 2D work happens in a plane basis (u, v) from
``plane._plane_basis``. The constrained solve fits ONE rectangle orientation
theta to all four edges in closed form (total-least-squares on pooled second
moments, with the perpendicular pair rotated 90 deg), then places each edge at
the mean projection of its evidence.

IMPORTANT — what ``RectangleSolution.discrepancy_mm`` can and cannot detect:
it is NOT a general cross-capture registration-error diagnostic. The
constrained model has 5 DOF (theta + 4 offsets) against the unconstrained
model's 8 (4 independent lines x 2 DOF each), and the 3 removed DOF are ALL
angular — the 4 edge offsets are free parameters fitted identically in both
models. So ``discrepancy_mm`` can only ever detect ANGULAR cross-capture
inconsistency (e.g. one capture's edge tilted relative to the rest). A pure
TRANSLATIONAL registration error — the dominant hand-eye/robot-pose error mode
across five separately-registered capture positions — moves it hardly at all,
and this module will return a confident, plausible, badly-mis-sized rectangle
without complaint. The only field that catches translational error is
``corner_agreement_mm``, and only when the caller supplies ``local_corners2d``
(ground truth or an independent corner detector); its value is ``None`` when
that was not supplied, meaning "not checked" — NEVER "checked and fine" — so
callers must treat ``None`` as a failed/skipped check, not a pass (spec §7
conditioning note).
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
    # fit_plane already orients the normal to +Z ("up"); no re-orientation needed here.
    normal = np.asarray(normal, dtype=float)
    res_all = np.abs((all_pts - centroid) @ normal)
    per = tuple(float(np.sqrt(np.mean(((s - centroid) @ normal) ** 2))) for s in sets)
    return GlobalPlane(tuple(normal), tuple(np.asarray(centroid, dtype=float)),
                        float(np.sqrt(np.mean(res_all ** 2))), float(res_all.max()), per)


def project_points_2d(points, normal, point, u, v) -> np.ndarray:
    # `normal` is kept for signature symmetry with the plane-basis contract (it is
    # not otherwise used): once `point` fixes the origin, `u`/`v` alone define the
    # projection, so `normal` does not participate in the computation below.
    d = np.asarray(points, dtype=float).reshape(-1, 3) - np.asarray(point, dtype=float)
    return np.column_stack([d @ np.asarray(u, dtype=float), d @ np.asarray(v, dtype=float)])


def lift_points_3d(points2d, point, u, v) -> np.ndarray:
    p2 = np.asarray(points2d, dtype=float).reshape(-1, 2)
    return (np.asarray(point, dtype=float)
            + p2[:, :1] * np.asarray(u, dtype=float)
            + p2[:, 1:2] * np.asarray(v, dtype=float))


@dataclass(frozen=True)
class EdgeLine:
    direction: tuple[float, float]  # unit vector along the edge; SIGN IS ARBITRARY
                                     # (chosen by eigh) and does NOT encode the
                                     # C_i -> C_{i+1} traversal direction
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
    angle_deg: float                             # orientation of the e0/e2 axis, mod 180 deg
                                                   # ONLY (a line has no intrinsic direction) --
                                                   # do NOT use this for the rectangle's
                                                   # directional sense; corners2d is authoritative
    edge_rms_mm: tuple[float, float, float, float]
    parallelism_deg: float                      # worst opposite-edge angle mismatch (unconstrained)
    perpendicularity_deg: float                 # worst adjacent-edge deviation from 90 (unconstrained)
    discrepancy_mm: float                       # max |unconstrained corner - constrained corner|,
                                                 # ANGULAR cross-capture inconsistency ONLY -- does
                                                 # NOT detect translational registration error (see
                                                 # module docstring); use corner_agreement_mm for that
    corner_agreement_mm: float | None           # vs locally detected/ground-truth corners. None
                                                 # means "not checked" (no local_corners2d given),
                                                 # NEVER "checked and fine" -- callers must treat
                                                 # None as a failed/skipped check, not a pass. This
                                                 # is the field that DOES catch translational error

    def to_dict(self) -> dict:
        return asdict(self)


def _intersect(n1, o1, n2, o2, *, label: str | None = None) -> np.ndarray:
    M = np.stack([n1, n2])
    det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    if abs(det) < 1e-9:
        where = f" ({label})" if label else ""
        raise ValueError(
            f"edge lines are parallel or coincident{where} -- cannot intersect them "
            "into a corner")
    return np.linalg.solve(M, np.array([o1, o2]))


def _angle_deg(direction) -> float:
    return math.degrees(math.atan2(direction[1], direction[0]))


def _angdiff(a: float, b: float) -> float:
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def solve_constrained_rectangle(edge_points, *, local_corners2d=None) -> RectangleSolution:
    pts = [np.asarray(e, dtype=float).reshape(-1, 2) for e in edge_points]
    if len(pts) != 4:
        raise ValueError("need exactly 4 edge point sets (C1C2, C2C3, C3C4, C4C1)")
    for i, p in enumerate(pts):
        if not np.all(np.isfinite(p)):
            raise ValueError(
                f"edge {i} contains non-finite (NaN/Inf) points -- reject or "
                "re-capture this edge before fitting a rectangle to it")

    # Unconstrained fit: independent TLS lines -> angle checks + raw corner
    # intersections. The angle checks are computed BEFORE the intersections
    # (which can raise on parallel/duplicated edges) so a failure can report the
    # diagnostic numbers that explain it, instead of dying silently.
    lines = [fit_edge_line(p) for p in pts]
    parallelism = max(_angdiff(_angle_deg(lines[0].direction), _angle_deg(lines[2].direction)),
                       _angdiff(_angle_deg(lines[1].direction), _angle_deg(lines[3].direction)))
    perpendicularity = max(
        abs(90.0 - _angdiff(_angle_deg(lines[i].direction),
                             _angle_deg(lines[(i + 1) % 4].direction)))
        for i in range(4))
    try:
        un_corners = np.array([
            _intersect(np.asarray(lines[(i - 1) % 4].normal), lines[(i - 1) % 4].offset,
                       np.asarray(lines[i].normal), lines[i].offset,
                       label=f"edges {(i - 1) % 4} and {i}")
            for i in range(4)
        ])
    except ValueError as exc:
        raise ValueError(
            f"{exc} -- check for a duplicated edge capture or a badly mis-assigned "
            f"edge (parallelism_deg={parallelism:.3f}, "
            f"perpendicularity_deg={perpendicularity:.3f})") from exc

    # Constrained fit: one theta for all four edges.
    #
    # For a line with unit normal n and points p, the offset that minimises the
    # residual sum of squares is o = mean(p) . n, and the residual sum of squares
    # at that optimum is n^T S n, where S = sum_i (p_i - mean)(p_i - mean)^T is the
    # edge's own centered scatter matrix. Edges 0 and 2 are the opposite side of
    # the rectangle sharing direction theta (normal n0); edges 1 and 3 share the
    # perpendicular direction theta+90 (normal n1). So the total squared residual
    # over all 4 edges, as a function of theta alone, is
    #   J(theta) = n0(theta)^T (S0+S2) n0(theta) + n1(theta)^T (S1+S3) n1(theta)
    # with n0 = (-sin t, cos t), n1 = (cos t, sin t) (sign of n1 does not matter,
    # it enters only quadratically) and A = S0+S2, B = S1+S3. Expanding both
    # quadratic forms with the half-angle identities
    #   sin^2 t = (1-cos2t)/2, cos^2 t = (1+cos2t)/2, sin t cos t = sin(2t)/2
    # gives J(theta) = const + P*cos(2t) + Q*sin(2t) with
    #   P = ((Ayy + Bxx) - (Axx + Byy)) / 2,  Q = Bxy - Axy.
    # This is minimised where cos(2t - atan2(Q, P)) = -1, i.e. at
    #   2t = atan2(Q, P) + pi == atan2(-Q, -P) (mod 2*pi).
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
