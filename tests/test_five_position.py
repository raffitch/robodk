"""Five-position (center + four-corner) survey state machine (spec §7, Task 11).

Pure numpy: no RoboDK/camera/hardware. Capture orchestration is Task 13's job;
this only tests the state machine's gates and the final rectangle fit.

Two findings from earlier review (see task-11-brief.md / task-11-report.md)
shape these tests:

  (A) ``discrepancy_mm`` (rect_fit.RectangleSolution) cannot detect a pure
      TRANSLATIONAL registration error -- only ``corner_agreement_mm`` can
      (see the rect_fit module docstring + tests/test_rect_fit.py). So
      ``finish()`` must gate on BOTH, and a translationally-biased corner is
      rejected via the corner-agreement gate, not the discrepancy gate.

  (B) ``extract_corner_evidence`` (Task 10) takes a ``closed`` flag
      (default False) that this module never calls directly (Task 13's job) --
      not exercised here, noted for the record.

    py -3.10 -m pytest tests\\test_five_position.py -q
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from tasni.core.config import ScanConfig
from tasni.modules.scan.corner_evidence import CornerEvidence
from tasni.modules.scan.five_position import (
    SURVEY_STEPS, FivePositionSurvey, _assign_edge_evidence, _match_ordered_corners,
)
from tasni.modules.scan.survey_contract import (
    MODE_FIVE_POSITION, PROVENANCE_FIVE_POSITION, CaptureRecord, RobotStateSnapshot,
    order_corners_clockwise,
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
    assert survey.quality["corner_agreement_mm"] < 5.0


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
    # NOTE on the warn-tier offset magnitude: fit_global_plane's RANSAC + SVD
    # refine can partially "absorb" a single spatially-localized offset by
    # tilting the whole plane toward it (measured: a 4.5 mm offset on one
    # corner's 300 points -- as the brief originally specified -- yields a
    # per-position RMS of only ~1.5 mm, well under the 3.0 mm warn
    # threshold, because a tiny tilt cheaply explains most of it away).
    # Measured sweep: 4.5->1.46mm, 8->2.55mm, 10->3.19mm, 12->3.78mm,
    # 30->30.0mm (RANSAC then treats the position as a clear outlier and
    # fits the plane through the other four instead). 12 mm lands with
    # comfortable margin inside the (3.0, 8.0] warn band.
    warn = _run_all(_survey(), z_offsets=(0, 0, 12.0, 0, 0))
    survey = warn.finish(calibration_id="c", locked_robot=_snap())
    assert "non_flat" in survey.quality.get("flags", [])
    bad = _run_all(_survey(), z_offsets=(0, 0, 30.0, 0, 0))
    with pytest.raises(RuntimeError, match="coplanar"):
        bad.finish(calibration_id="c", locked_robot=_snap())


def test_biased_corner_is_rejected_by_the_corner_agreement_gate():
    """Finding (A): a TRANSLATIONAL bias of one corner's evidence moves
    ``discrepancy_mm`` hardly at all (it only sees ANGULAR inconsistency --
    see rect_fit's module docstring and test_discrepancy_is_blind_to_pure_
    translation_but_corner_agreement_catches_it). The brief's original test
    asserted this would fail the DISCREPANCY gate, which is not something the
    metric can do; reworked here to assert the survey is still REJECTED (via
    whichever gate the implementation actually fires -- measured to be the
    corner-agreement gate) rather than silently accepting a 40 mm-wrong
    rectangle.
    """
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
    with pytest.raises(RuntimeError, match="agreement|reposition"):
        s.finish(calibration_id="c", locked_robot=_snap())


def test_none_corner_agreement_is_treated_as_a_failed_check(monkeypatch):
    """Finding (A): ``corner_agreement_mm is None`` means "not checked", never
    "checked and fine". The survey always passes ``local_corners2d``, so a
    ``None`` coming back would indicate a bug -- ``finish()`` must fail loudly
    rather than silently proceeding. Simulated here by monkeypatching
    ``solve_constrained_rectangle`` to strip the field, since the real
    function can never return None when given ``local_corners2d``.
    """
    import tasni.modules.scan.five_position as five_position

    real_solve = five_position.solve_constrained_rectangle

    def _strip_agreement(edge_points, *, local_corners2d=None):
        sol = real_solve(edge_points, local_corners2d=local_corners2d)
        return dataclasses.replace(sol, corner_agreement_mm=None)

    monkeypatch.setattr(five_position, "solve_constrained_rectangle", _strip_agreement)
    s = _run_all(_survey())
    with pytest.raises(RuntimeError, match="agreement"):
        s.finish(calibration_id="c", locked_robot=_snap())


def test_edge_points_are_assigned_to_the_correct_edge_not_the_neighbor():
    """Ambiguity resolution #2: pooled corner evidence spans BOTH edges meeting
    at that corner (Task 10 pools both arms without labelling them). Near a
    corner, a point genuinely on the WRONG (adjacent) edge can sit well inside
    a same-edge distance-band test around the RIGHT edge's line -- verified
    here by construction with a thin rectangle (long edge 1600 mm, short edge
    60 mm) where this is easy to trigger: a point 10 mm along the SHORT edge
    from corner 0 is well within the 25 mm band of the LONG edge's line too.
    """
    ordered2d = np.array([[0, 0], [1600, 0], [1600, 60], [0, 60]], float)
    band = 25.0
    genuine_edge0_pt = np.array([10.0, 0.5])     # on edge0 (C0->C1), near C0
    foreign_pt = np.array([0.5, 10.0])           # actually on edge3 (C3->C0), near C0
    pool_for = [
        np.array([genuine_edge0_pt, foreign_pt]),  # corner 0's pooled evidence
        np.empty((0, 2)),
        np.empty((0, 2)),
        np.empty((0, 2)),
    ]
    edges = _assign_edge_evidence(ordered2d, pool_for, band)
    e0, e3 = edges[0], edges[3]
    assert any(np.allclose(p, genuine_edge0_pt) for p in e0)
    assert not any(np.allclose(p, foreign_pt) for p in e0), (
        "a point genuinely on edge3 leaked into edge0's fit")
    assert any(np.allclose(p, foreign_pt) for p in e3), (
        "the foreign point should have landed on the edge it actually belongs to")


def test_match_ordered_corners_maps_exactly_even_when_two_corners_are_close():
    """Ambiguity resolution #2: ``order_corners_clockwise`` only reorders its
    input -- it never recomputes a coordinate -- so matching an ordered corner
    back to its originating capture must be EXACT, even when two surveyed
    corners sit close together in projection (e.g. the short edge of a long,
    narrow platform), not just approximately-nearest.
    """
    thin_corners = np.array([[0, 0, 0], [1600, 0, 0], [1600, 40, 0], [0, 40, 0]], float)
    normal = np.array([0.0, 0.0, 1.0])
    ordered = order_corners_clockwise(thin_corners, normal)
    order = _match_ordered_corners(ordered, thin_corners)
    assert sorted(order) == [0, 1, 2, 3]
    for i, j in enumerate(order):
        assert np.allclose(ordered[i], thin_corners[j])


def test_match_ordered_corners_rejects_unmatchable_input():
    """A defensive internal-error path: if the ordered corners cannot be
    exactly matched back to the originals (a genuine bug, not an operator
    mistake), fail loudly instead of silently mis-pairing captures.
    """
    from tasni.modules.scan.five_position import _match_ordered_corners
    corners = np.array([[0, 0, 0], [1600, 0, 0], [1600, 1000, 0], [0, 1000, 0]], float)
    bogus_ordered = corners.copy()
    bogus_ordered[0] = [999.0, 999.0, 0.0]  # not present in `corners`
    with pytest.raises(RuntimeError):
        _match_ordered_corners(bogus_ordered, corners)
