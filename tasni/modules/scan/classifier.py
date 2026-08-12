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
