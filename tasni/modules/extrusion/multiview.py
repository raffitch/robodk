"""Merge several camera views of one ring into a single work-frame cloud.

The rings are thin: a top-down frame sees the crest at grazing signal-to-noise
and the flanks hardly at all, and the flanks are what the bead-width and
cross-section numbers are read from. Tilted views add them; this module makes
the tilted views agree with each other well enough to be worth adding.

Two things it deliberately does NOT do.

ICP. The ring is a torus, so sliding one view tangentially around it costs
almost nothing in point-to-point distance and ICP has a nearly free degree of
freedom it will fill with noise. When a shape has a degenerate DOF you stop
registering points and start fitting the model the object actually has: a circle.

Anchor to the top view. Translating each tilted view so its own fitted centre
lands on the top view's makes the merged centre, by construction, the top view's
centre -- so the merged cloud would be the measurement of record while the
headline number was still a single-view answer. The joint solve below fixes the
gauge as ``sum(offsets) == 0`` instead, so no view is privileged and the centre
is a consensus.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from .processing import deposit_floor_mm

MAX_FIT_POINTS = 2000        # per view; the solve is O(points), the fit is not


@dataclass
class ViewCloud:
    """One view's chroma-gated work-frame points, before any merge."""

    name: str
    points: np.ndarray
    chroma_gated: bool
    tilt_deg: float = 0.0
    azimuth_deg: float = 0.0
    T_work_camera: np.ndarray | None = None


@dataclass
class MergeResult:
    points: np.ndarray
    chroma_gated: bool
    used: list[str] = field(default_factory=list)
    dropped: dict[str, str] = field(default_factory=dict)
    consensus_center_mm: tuple[float, float] | None = None
    consensus_radius_mm: float | None = None
    offsets_mm: dict[str, tuple[float, float]] = field(default_factory=dict)
    residual_rms_mm: dict[str, float] = field(default_factory=dict)
    spread_before_mm: float | None = None
    residual_after_mm: float | None = None


def fit_circle(xy) -> tuple[float, float, float]:
    """Least-squares circle through 2-D points (Kasa), returned as (cx, cy, r).

    Algebraic, not geometric: it is a seed for the joint solve, and it is linear,
    so it cannot fail to converge on a partial arc the way an iterative fit can.
    """
    p = np.asarray(xy, dtype=float).reshape(-1, 2)
    A = np.column_stack((2.0 * p[:, 0], 2.0 * p[:, 1], np.ones(len(p))))
    b = (p ** 2).sum(axis=1)
    cx, cy, c = np.linalg.lstsq(A, b, rcond=None)[0]
    return float(cx), float(cy), float(np.sqrt(max(c + cx * cx + cy * cy, 0.0)))


def arc_span_deg(xy, center_xy) -> float:
    """Angular extent actually covered, as the largest gap's complement.

    A view seeing two short opposite arcs covers 360 deg of RANGE but constrains
    a centre about as well as one arc, so span is measured as 360 minus the
    biggest empty wedge.
    """
    p = np.asarray(xy, dtype=float).reshape(-1, 2) - np.asarray(center_xy, dtype=float)
    if len(p) < 3:
        return 0.0
    ang = np.sort(np.degrees(np.arctan2(p[:, 1], p[:, 0])) % 360.0)
    gaps = np.diff(np.concatenate((ang, [ang[0] + 360.0])))
    return float(360.0 - gaps.max())


def level_points(points, *, r_inner_mm: float, r_outer_mm: float, center_xy,
                 config) -> tuple[np.ndarray, dict]:
    """Rotate/translate one view so the surface its ring sits on is z = 0.

    The plane is fitted in an ANNULUS outside the chain's radial ROI band, so it
    sees surface and never deposit. This removes the plane-offset-and-tilt part
    of the systematic warp that grows with incidence -- the cell's own sweep
    measured board length error growing 0.036 -> 0.447 mm from 0 to 20 deg, far
    faster than random noise would.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    centre = np.asarray(center_xy, dtype=float)
    radius = np.linalg.norm(pts[:, :2] - centre, axis=1)
    annulus = pts[(radius >= r_inner_mm) & (radius <= r_outer_mm)]
    if len(annulus) < int(config.multiview_level_min_points):
        raise ValueError(f"levelling annulus held {len(annulus)} points, "
                         f"need {config.multiview_level_min_points}")
    # Plane through the annulus by SVD on the centred points: robust to the
    # annulus being wide, and it needs no RANSAC because the annulus is chosen
    # to exclude the deposit in the first place.
    mean = annulus.mean(axis=0)
    normal = np.linalg.svd(annulus - mean)[2][-1]
    if normal[2] < 0:
        normal = -normal
    level_mm = float(abs(mean[2]))
    if level_mm > float(config.multiview_max_level_mm):
        raise ValueError(f"fitted surface sits {level_mm:.1f} mm from z=0, "
                         f"limit {config.multiview_max_level_mm:.1f} mm")
    # Rotation taking the fitted normal onto +Z (Rodrigues about their cross).
    target = np.array([0.0, 0.0, 1.0])
    axis = np.cross(normal, target)
    s, c = float(np.linalg.norm(axis)), float(np.dot(normal, target))
    if s < 1e-9:
        R = np.eye(3)
    else:
        k = axis / s
        Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        R = np.eye(3) + Kx * s + Kx @ Kx * (1 - c)
    levelled = (R @ (pts - mean).T).T + np.array([mean[0], mean[1], 0.0])
    return levelled, {"level_mm": level_mm,
                      "tilt_removed_deg": float(np.degrees(np.arccos(np.clip(c, -1, 1)))),
                      "annulus_points": int(len(annulus))}


def solve_view_offsets(view_xy: dict[str, np.ndarray], config) -> dict:
    """Per-view lateral offsets that make every view fit ONE shared circle.

    Unknowns: one (dx, dy) per view, plus a shared centre and a shared radius.
    One shared radius, not one per view: a per-view radius would let a view
    carrying residual scale error absorb it silently, where a shared one forces
    it out as a large residual.

    Gauge: sum(offsets) == 0, imposed as two heavily weighted residual rows.
    Without it the problem is translation-degenerate (slide everything together
    and nothing changes). With it, no view is privileged and the recovered
    centre is the consensus of all views rather than the first one's answer.

    Soft-L1 loss so one bad arc cannot drag the solution.
    """
    names = list(view_xy)
    subs, seeds = [], []
    rng = np.random.default_rng(0)
    for name in names:
        p = np.asarray(view_xy[name], dtype=float).reshape(-1, 2)
        if len(p) > MAX_FIT_POINTS:
            p = p[rng.choice(len(p), MAX_FIT_POINTS, replace=False)]
        subs.append(p)
        seeds.append(fit_circle(p))
    seed_centres = np.array([[s[0], s[1]] for s in seeds])
    c0 = seed_centres.mean(axis=0)
    r0 = float(np.mean([s[2] for s in seeds]))
    spread_before = float(np.sqrt(np.mean(
        np.sum((seed_centres - c0) ** 2, axis=1)))) if len(names) > 1 else 0.0

    gauge_weight = 1e3 * max(1.0, float(np.mean([len(p) for p in subs])))

    def residuals(x):
        cx, cy, r = x[0], x[1], x[2]
        d = x[3:].reshape(len(names), 2)
        out = [np.linalg.norm(p + d[i] - [cx, cy], axis=1) - r
               for i, p in enumerate(subs)]
        out.append(np.array([gauge_weight * d[:, 0].sum(),
                             gauge_weight * d[:, 1].sum()]))
        return np.concatenate(out)

    x0 = np.concatenate(([c0[0], c0[1], r0], np.zeros(2 * len(names))))
    sol = least_squares(residuals, x0, loss="soft_l1", f_scale=1.0, max_nfev=200)
    cx, cy, r = float(sol.x[0]), float(sol.x[1]), float(sol.x[2])
    d = sol.x[3:].reshape(len(names), 2)
    per_view, offsets = {}, {}
    for i, name in enumerate(names):
        res = np.linalg.norm(subs[i] + d[i] - [cx, cy], axis=1) - r
        per_view[name] = float(np.sqrt(np.mean(res ** 2)))
        offsets[name] = (float(d[i, 0]), float(d[i, 1]))
    return {"consensus_center_mm": (cx, cy), "consensus_radius_mm": r,
            "offsets_mm": offsets, "residual_rms_mm": per_view,
            "spread_before_mm": spread_before,
            "residual_after_mm": float(np.sqrt(np.mean(
                [v ** 2 for v in per_view.values()])))}


def merge_views(views: list[ViewCloud], *, plan, layer, config) -> MergeResult:
    """Level, register and concatenate. Never raises; degrades to the top view.

    Drops are normal, not errors (spec section 8): a tilted view can miss the
    ring, land badly, or lose its colour gate, and the take must still complete
    on what is left. A view whose CHROMA GATE ABSTAINED is dropped rather than
    contributed, because the deposit floor is a property of the merged cloud --
    letting one abstainer through would drag the whole merge to the 2.5 mm floor,
    which on 2026-08-29 cost a 45 deg sector and made all four takes invalid.
    """
    recipe, setup = plan.recipe, plan.setup
    centre = np.array([float(setup.center_x_mm), float(setup.center_y_mm)])
    r_hi = float(recipe.radius_mm) + float(config.radial_roi_margin_mm)
    r_lo = float(recipe.radius_mm) - float(config.radial_roi_margin_mm)
    max_z = (float(layer.nominal_z_mm) + float(recipe.bead_diameter_mm) / 2
             + float(config.deposit_height_margin_mm))

    top = next((v for v in views if v.name == "top"), views[0] if views else None)
    if top is None:
        raise ValueError("merge_views needs at least the top view")

    levelled: dict[str, np.ndarray] = {}
    fit_xy: dict[str, np.ndarray] = {}
    dropped: dict[str, str] = {}
    for view in views:
        if not view.chroma_gated:
            dropped[view.name] = "colour gate abstained; would move the merged floor"
            continue
        try:
            pts, _ = level_points(view.points, r_inner_mm=r_hi,
                                  r_outer_mm=r_hi + config.multiview_level_annulus_width_mm,
                                  center_xy=centre, config=config)
        except ValueError as exc:
            dropped[view.name] = f"levelling failed: {exc}"
            continue
        # The fit subset only: the chain's own height x radial band, so the
        # circle is fitted to deposit rather than to the whole gated cloud. The
        # offsets it solves are applied to the FULL cloud, so nothing is lost.
        #
        # The lower bound MUST be the chain's own deposit floor, not zero. The
        # surface reads a millimetre or so either side of z = 0 after levelling,
        # so a zero floor pulls a ring-shaped slab of board into the circle fit
        # and biases the centre it is trying to measure.
        radius = np.linalg.norm(pts[:, :2] - centre, axis=1)
        min_z = deposit_floor_mm(config, True)      # gated: abstainers already dropped
        band = pts[(radius >= r_lo) & (radius <= r_hi)
                   & (pts[:, 2] >= min_z) & (pts[:, 2] <= max_z)]
        if len(band) < int(config.multiview_min_view_points):
            dropped[view.name] = f"only {len(band)} ring points, need {config.multiview_min_view_points}"
            continue
        seed = fit_circle(band[:, :2])
        span = arc_span_deg(band[:, :2], seed[:2])
        if span < float(config.multiview_min_arc_deg):
            dropped[view.name] = f"sees {span:.0f} deg of arc, need {config.multiview_min_arc_deg:.0f}"
            continue
        levelled[view.name] = pts
        fit_xy[view.name] = band[:, :2]

    def top_only(reason: str | None = None) -> MergeResult:
        if reason:
            dropped.setdefault("__merge__", reason)
        return MergeResult(points=np.asarray(top.points, dtype=float),
                           chroma_gated=bool(top.chroma_gated),
                           used=["top"], dropped=dropped)

    if len(fit_xy) < int(config.multiview_min_views):
        return top_only(f"only {len(fit_xy)} view(s) usable, "
                        f"need {config.multiview_min_views}")

    solved = solve_view_offsets(fit_xy, config)
    while True:
        far = [n for n, d in solved["offsets_mm"].items()
               if float(np.hypot(*d)) > float(config.multiview_max_offset_mm)]
        if not far:
            break
        # The gauge (sum(offsets) == 0) means ONE badly misregistered view
        # drags every other view's offset with it -- flagging everyone over
        # threshold in a single pass would drop the whole take (verified: the
        # three good views alone solve to <0.1 mm, but combined with the bad
        # one they read ~7.5 mm each). So isolate a genuine outlier by
        # dropping only the worst offender per round and re-solving, rather
        # than punishing the rest for one view's fault.
        worst = max(far, key=lambda n: float(np.hypot(*solved["offsets_mm"][n])))
        dropped[worst] = (f"solved offset {np.hypot(*solved['offsets_mm'][worst]):.1f} mm "
                          f"exceeds {config.multiview_max_offset_mm:.1f} mm")
        fit_xy.pop(worst, None)
        levelled.pop(worst, None)
        if len(fit_xy) < int(config.multiview_min_views):
            return top_only("registration rejected too many views")
        solved = solve_view_offsets(fit_xy, config)

    merged = []
    for name, pts in levelled.items():
        dx, dy = solved["offsets_mm"][name]
        merged.append(pts + np.array([dx, dy, 0.0]))
    return MergeResult(
        points=np.vstack(merged), chroma_gated=True,
        used=list(levelled), dropped=dropped,
        consensus_center_mm=solved["consensus_center_mm"],
        consensus_radius_mm=solved["consensus_radius_mm"],
        offsets_mm=solved["offsets_mm"], residual_rms_mm=solved["residual_rms_mm"],
        spread_before_mm=solved["spread_before_mm"],
        residual_after_mm=solved["residual_after_mm"])
