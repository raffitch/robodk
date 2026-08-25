"""Guided center + four-corner survey (spec §7).

Pure state machine: capture orchestration (camera/robot/RoboDK) lives in
service.py (Task 13). Edge assignment is geometric -- all arm points from the
two corners adjacent to an edge are pooled, then filtered to a band around the
corner-to-corner segment -- so the extractor (``corner_evidence.py``, Task 10)
never needs to label which arm a point came from.

Two findings from earlier review of this plan matter here (see
``docs/agent-debug-map.md`` / the Task 11 handoff for the full writeup):

  (A) ``rect_fit.RectangleSolution.discrepancy_mm`` cannot detect a pure
      TRANSLATIONAL registration error -- the constrained (5 DOF) and
      unconstrained (8 DOF) rectangle models fit every edge's OFFSET freely
      and identically; only the 3 removed DOF are angular. So a capture whose
      evidence is rigidly mis-registered (the dominant hand-eye/robot-pose
      error mode across five separately-registered positions) can produce a
      confident, badly-wrong rectangle while ``discrepancy_mm`` barely moves.
      ``corner_agreement_mm`` (fitted rectangle vs the surveyed corners) is
      what catches that, so ``finish()`` gates on BOTH, and treats
      ``corner_agreement_mm is None`` as a FAILED check -- never a pass --
      since this module always supplies ``local_corners2d``, so ``None``
      coming back means something is wrong with the pipeline, not the survey.

  (B) ``corner_evidence.extract_corner_evidence`` takes a ``closed`` flag
      (default False) governing whether its polygon walk wraps around a
      closed contour. This module never calls that function directly (Task
      13's capture orchestration does) and only ever *consumes* the
      ``CornerEvidence`` it returns, so this file has no ``closed`` decision
      to make -- flagged here for whoever wires Task 13 up.
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
                              capture_is_fresh, order_corners_clockwise,
                              workframe_from_rectangle)

SURVEY_STEPS = ("center", "corner1", "corner2", "corner3", "corner4")


@dataclass
class _Accepted:
    record: CaptureRecord
    plane_points_base: np.ndarray
    evidence: CornerEvidence | None


def _match_ordered_corners(ordered3d: np.ndarray, corners_approx: np.ndarray) -> list[int]:
    """Map each row of ``ordered3d`` back to the index (into ``corners_approx``,
    i.e. capture order corner1..corner4) it came from.

    ``order_corners_clockwise`` only permutes/rolls its input array -- it never
    edits a coordinate -- so every output row is an EXACT copy of one input
    row. Matching is therefore exact, not merely approximate-nearest, even
    when two surveyed corners sit close together in projection (e.g. the
    short edge of a long, narrow platform: see
    ``test_match_ordered_corners_maps_exactly_even_when_two_corners_are_close``).
    A duplicate-use guard stops two ordered corners being matched to the same
    capture, and an exact-distance check turns any genuine mismatch (a bug,
    not an operator error) into a loud internal error instead of a silent
    mis-pairing that would scramble which capture's edge evidence feeds which
    edge.
    """
    corners_approx = np.asarray(corners_approx, dtype=float)
    used: set[int] = set()
    order: list[int] = []
    for row in np.asarray(ordered3d, dtype=float):
        d = np.linalg.norm(corners_approx - row, axis=1)
        if used:
            d[list(used)] = np.inf
        j = int(np.argmin(d))
        if d[j] > 1e-6:
            raise RuntimeError(
                "internal error: could not match a surveyed corner back to its "
                "capture (this indicates a bug in the survey pipeline, not an "
                "operator error) - do not trust this rectangle")
        used.add(j)
        order.append(j)
    return order


def _assign_edge_evidence(ordered2d: np.ndarray, pool_for: list[np.ndarray],
                          band_mm: float) -> list[np.ndarray]:
    """Pool each ordered corner's evidence onto its two adjacent edges, then
    trim to a perpendicular band + along-range around the corner-to-corner
    segment.

    Each corner's pooled evidence spans BOTH edges meeting there (Task 10
    pools both arms without labelling them), so a point near a corner is
    geometrically ambiguous between the two edges from a same-edge
    distance-band test alone: a point a few mm along the WRONG edge from the
    corner can sit well inside a generous band around the RIGHT edge's line
    too (this is exactly the case for a corner whose two edges are very
    different lengths -- the short edge's near-corner points comfortably
    clear the long edge's band). So every candidate point is first assigned
    to whichever of its two CANDIDATE edges' APPROXIMATE lines (through the
    surveyed corners, before any line is fitted from pooled data) it is
    nearer to, and only THEN trimmed to the band/along window. This is what
    stops a point genuinely on edge A from being fitted into edge B's line.
    """
    n = 4
    ordered2d = np.asarray(ordered2d, dtype=float).reshape(n, 2)
    edge_dir = []
    edge_normal = []
    edge_len = []
    for i in range(n):
        d = ordered2d[(i + 1) % n] - ordered2d[i]
        length = float(np.linalg.norm(d))
        if length < 1e-9:
            raise RuntimeError(
                "two adjacent surveyed corners coincide - recapture the survey")
        d = d / length
        edge_dir.append(d)
        edge_normal.append(np.array([-d[1], d[0]]))
        edge_len.append(length)

    buckets: list[list[np.ndarray]] = [[] for _ in range(n)]
    for k in range(n):
        pts = np.asarray(pool_for[k], dtype=float).reshape(-1, 2)
        prev_i, next_i = (k - 1) % n, k
        rel_prev = pts - ordered2d[prev_i]
        dist_prev = np.abs(rel_prev @ edge_normal[prev_i])
        rel_next = pts - ordered2d[next_i]
        dist_next = np.abs(rel_next @ edge_normal[next_i])
        closer_to_next = dist_next <= dist_prev
        buckets[next_i].append(pts[closer_to_next])
        buckets[prev_i].append(pts[~closer_to_next])

    edge_points = []
    for i in range(n):
        cand = np.concatenate(buckets[i], axis=0)
        a2, d, n2, length = ordered2d[i], edge_dir[i], edge_normal[i], edge_len[i]
        rel = cand - a2
        along = rel @ d
        dist = np.abs(rel @ n2)
        keep = cand[(dist <= band_mm) & (along >= -0.1 * length) & (along <= 1.1 * length)]
        edge_points.append(keep)
    return edge_points


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
        # NaN-safe by construction: `x < lo` / `x > hi` silently PASS a NaN
        # (every comparison against NaN is False, including `nan < lo` and
        # `nan > hi`), so these gates are written as `not (x satisfies the
        # bound)` -- the pattern the standoff check just below already used
        # correctly (`not (lo <= x <= hi)` correctly rejects a NaN standoff;
        # `x < lo or x > hi` would not have).
        #
        # THRESHOLD is step-aware, not just the metric feeding it (Task 13 review,
        # remedy ii, round 3): five_position_capture already computes a DIFFERENT
        # valid_frac per step -- the CENTRE step's coarse centre-patch fraction
        # (the reticle really does sit on the surface there, so it routinely reads
        # close to 1.0) vs. a CORNER step's plane-inlier PURITY (of the depth
        # actually measured, how much lies on the work plane). Those two metrics
        # have structurally different achievable ranges: centring the reticle on a
        # 90-degree corner means at most one quadrant of the frame can ever be the
        # work surface, capping a mathematically PERFECT corner aim's purity at
        # ~25% (see ScanConfig.survey_corner_min_plane_coverage_frac's own
        # comment) -- comfortably below min_valid_depth_frac's default (0.5,
        # documented as "this fraction of the PATCH", tuned for the centre-patch
        # metric and never redesigned for a whole-frame corner one). Applying that
        # single threshold to both meant EVERY corner capture against a real
        # (non-silent) background was rejected regardless of aim quality -- not an
        # edge case on a physical cell with a floor/fixtures inside D435i range,
        # but the common one. Corner steps are instead gated against
        # survey_corner_min_plane_coverage_frac -- reused rather than adding a
        # near-duplicate key: at the ceiling, purity and coverage are the SAME
        # quantity (their denominators coincide whenever the background returns
        # real depth), so the "admit the ~25% ceiling with margin" reasoning
        # documented on that key applies equally to both metrics.
        if expected == "center":
            if not (record.valid_frac >= float(self._scfg.min_valid_depth_frac)):
                raise RuntimeError("not enough valid depth in the capture")
        else:
            corner_min_frac = float(self._scfg.survey_corner_min_plane_coverage_frac)
            if not (record.valid_frac >= corner_min_frac):
                raise RuntimeError(
                    f"{expected} capture: only {record.valid_frac:.0%} of the depth "
                    f"measured is trustworthy work-plane data (< {corner_min_frac:.0%} "
                    "minimum) - too little of the table is in view; move closer to "
                    "the corner or reposition and recapture")
        if not (record.tilt_deg <= float(self._scfg.survey_max_tilt_deg)):
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
            # Same hazard as the plane-points check below, through a different
            # intake field: corner_agreement_mm downstream is computed from
            # this value (via order_corners_clockwise -> project_points_2d ->
            # local_corners2d), and nothing else on that path checks
            # finiteness -- solve_constrained_rectangle only validates its
            # edge-point sets, not local_corners2d. Reject here, at the gate
            # layer, rather than trust the extractor's own invariant.
            if not np.all(np.isfinite(np.asarray(evidence.corner_base_mm, dtype=float))):
                raise RuntimeError(
                    f"{expected} capture has a non-finite corner depth (NaN/Inf) - "
                    "reposition and recapture")
        pts = np.asarray(plane_points_base, dtype=float).reshape(-1, 3)
        # A single non-finite plane point silently defeats the coplanarity
        # gates downstream (fit_global_plane's per-position RMS becomes NaN,
        # and NaN compares False against every threshold -- see finish()'s
        # own defensive check for the full story), so this is the gate layer
        # and it is rejected here at intake, not left for finish() to trust.
        if not np.all(np.isfinite(pts)):
            raise RuntimeError(
                f"{expected} capture has non-finite plane points (NaN/Inf) - "
                "reposition and recapture")
        if len(pts) < 50:
            raise RuntimeError("too few plane points in the capture - reposition and recapture")
        self._accepted[expected] = _Accepted(record, pts, evidence)
        return self.state()

    def recapture(self, kind: str) -> dict:
        if kind not in SURVEY_STEPS:
            raise ValueError(f"unknown capture kind {kind!r} - expected one of {SURVEY_STEPS}")
        self._accepted.pop(kind, None)
        return self.state()

    def finish(self, *, calibration_id: str,
               locked_robot: RobotStateSnapshot) -> LockedWorkframeSurvey:
        if self.step != "review":
            raise RuntimeError(f"survey incomplete - next capture: {self.step}")
        acc = [self._accepted[s] for s in SURVEY_STEPS]

        # -- coplanarity: two-tier warn/reject (ambiguity resolution #3) ----
        plane = fit_global_plane([a.plane_points_base for a in acc])
        flags: list[str] = []
        local_warnings: list[str] = []
        per_rms = np.asarray(plane.per_set_rms_mm, dtype=float)
        # Defense in depth (add_capture already blocks non-finite plane points
        # at intake, but this must not silently trust its own inputs either):
        # Python's/numpy's max() returns NaN whenever any element is NaN, and
        # `nan > threshold` is False -- so an unguarded `max()` + `>` pipeline
        # would silently PASS both coplanarity tiers on exactly the surface
        # they exist to reject. Refuse loudly instead of computing a worst
        # value from tainted data.
        if not np.all(np.isfinite(per_rms)):
            bad = ", ".join(SURVEY_STEPS[i] for i in range(len(per_rms))
                            if not np.isfinite(per_rms[i]))
            raise RuntimeError(
                f"capture(s) {bad} produced non-finite plane geometry (NaN/Inf) - "
                "reposition and recapture; do not trust this survey")
        worst_idx = int(np.argmax(per_rms))
        worst = float(per_rms[worst_idx])
        worst_capture = SURVEY_STEPS[worst_idx]
        if worst > float(self._scfg.survey_coplanar_reject_mm):
            raise RuntimeError(
                f"captures are not coplanar (worst offender: {worst_capture}, "
                f"per-position plane RMS {worst:.1f} mm > "
                f"{float(self._scfg.survey_coplanar_reject_mm):.1f} mm) - re-survey the "
                f"surface starting with {worst_capture}; if this is a genuinely non-flat "
                "table, loosen survey_coplanar_reject_mm in the config")
        if worst > float(self._scfg.survey_coplanar_warn_mm):
            flags.append("non_flat")
            local_warnings.append(
                f"surface labeled non-flat: worst offender {worst_capture}, "
                f"per-position plane RMS {worst:.1f} mm")
        # Assign (never append/accumulate): finish() can legitimately be
        # called more than once on the same survey (e.g. after a recapture),
        # and self.warnings must reflect only the MOST RECENT run, not carry
        # a stale warning describing a capture that has since been replaced.
        self.warnings = local_warnings

        normal = np.asarray(plane.normal)
        point = np.asarray(plane.point)
        u, v = _plane_basis(normal)

        # -- match each ordered (clockwise) corner back to its capture ------
        corner_accs = acc[1:]  # corner1..corner4, in capture order
        corners_approx = np.array([np.asarray(a.evidence.corner_base_mm, dtype=float)
                                   for a in corner_accs])
        ordered3d = order_corners_clockwise(corners_approx, normal)
        capture_order = _match_ordered_corners(ordered3d, corners_approx)
        ordered2d = project_points_2d(ordered3d, normal, point, u, v)
        pool_for = [project_points_2d(corner_accs[j].evidence.edge_points_base,
                                      normal, point, u, v) for j in capture_order]

        # -- geometric edge assignment (ambiguity resolution #2) -----------
        band = float(self._scfg.survey_edge_band_mm)
        edge_candidates = _assign_edge_evidence(ordered2d, pool_for, band)
        edge_points = []
        for i, keep in enumerate(edge_candidates):
            if len(keep) < int(self._scfg.survey_min_edge_points):
                raise RuntimeError(
                    f"edge C{i + 1}-C{(i + 1) % 4 + 1} has too little evidence "
                    f"({len(keep)} < {int(self._scfg.survey_min_edge_points)} pts) - "
                    "reposition and recapture the adjacent corners")
            edge_points.append(keep)

        rect = solve_constrained_rectangle(edge_points, local_corners2d=ordered2d)

        # -- Finding (A): gate on corner_agreement_mm IN ADDITION to
        # discrepancy_mm. corner_agreement_mm is the ONLY field that catches a
        # translational registration error; None means "not checked", never
        # "checked and fine" -- this module always passes local_corners2d, so
        # None here would mean a bug, and we must fail loudly rather than
        # silently accept an unverified rectangle.
        if rect.corner_agreement_mm is None:
            raise RuntimeError(
                "internal error: corner agreement was not checked - refusing to "
                "trust an unverified rectangle (this indicates a bug, not an "
                "operator error)")
        # NaN-safe (same class as the add_capture rewrite above): `value >
        # threshold` silently PASSES a NaN, so these are written as
        # `not (value satisfies the bound)`. Not reachable through the sole
        # production path today (add_capture now blocks a non-finite
        # corner_base_mm at intake, and solve_constrained_rectangle itself
        # rejects non-finite edge points before computing discrepancy_mm) --
        # but this is the gate layer, and a gate must be safe on its own
        # terms, not merely because its current caller happens to be.
        if not (rect.corner_agreement_mm <= float(self._scfg.survey_corner_agreement_mm)):
            if not np.isfinite(rect.corner_agreement_mm):
                raise RuntimeError(
                    "internal error: corner agreement is non-finite (NaN/Inf) - "
                    "refusing to trust this rectangle; reposition and recapture "
                    "the survey")
            corner_deltas = np.linalg.norm(ordered2d - np.asarray(rect.corners2d), axis=1)
            worst_corner = f"C{int(np.argmax(corner_deltas)) + 1}"
            raise RuntimeError(
                "surveyed corners disagree with the fitted rectangle (corner "
                f"agreement {rect.corner_agreement_mm:.1f} mm > "
                f"{float(self._scfg.survey_corner_agreement_mm):.1f} mm, worst at "
                f"{worst_corner}) - {worst_corner} is likely mis-registered; reposition "
                f"and recapture it")
        if not (rect.discrepancy_mm <= float(self._scfg.survey_rect_discrepancy_mm)):
            if not np.isfinite(rect.discrepancy_mm):
                raise RuntimeError(
                    "internal error: rectangle discrepancy is non-finite (NaN/Inf) - "
                    "refusing to trust this rectangle; reposition and recapture "
                    "the survey")
            i = int(np.argmax(np.asarray(rect.edge_rms_mm)))
            worst_edge = f"C{i + 1}-C{(i + 1) % 4 + 1}"
            raise RuntimeError(
                "rectangle evidence is inconsistent (unconstrained-vs-constrained "
                f"discrepancy {rect.discrepancy_mm:.1f} mm > "
                f"{float(self._scfg.survey_rect_discrepancy_mm):.1f} mm, worst edge "
                f"{worst_edge}) - reposition and recapture around {worst_edge}")

        corners3d = lift_points_3d(np.asarray(rect.corners2d), point, u, v)
        corners3d = order_corners_clockwise(corners3d, normal)
        frame_T = workframe_from_rectangle(corners3d, normal)
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
