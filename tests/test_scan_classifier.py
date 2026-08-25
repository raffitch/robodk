import numpy as np

from tasni.core.config import ScanConfig
from tasni.modules.scan.classifier import (
    CompactEligibility, classify_compact, rectangle_identity_consistent,
)


class _Survey:
    detected = True
    fully_framed = True
    tilt_deg = 2.0


_GOOD_UV = np.array([[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]])
_BOUNDARY = {"overruns": False, "polygon_uv": _GOOD_UV.tolist()}


def _history(n=5, jitter=0.0):
    rng = np.random.default_rng(0)
    return [_GOOD_UV + rng.normal(0, jitter, _GOOD_UV.shape) for _ in range(n)]


def test_compact_all_conditions_pass():
    r = classify_compact(_Survey(), _GOOD_UV, _BOUNDARY, ScanConfig(),
                         outline_history=_history())
    assert isinstance(r, CompactEligibility)
    assert r.eligible is True and r.reasons == ()


def test_guard_band_rejects_edge_hugging_rectangle():
    uv = np.array([[0.01, 0.2], [0.8, 0.2], [0.8, 0.8], [0.01, 0.8]])
    r = classify_compact(_Survey(), uv, _BOUNDARY, ScanConfig(), outline_history=_history())
    assert r.eligible is False and r.guard_ok is False
    assert any("guard" in reason for reason in r.reasons)


def test_boundary_overrun_or_missing_rejects():
    r = classify_compact(_Survey(), _GOOD_UV, {"overruns": True}, ScanConfig(),
                         outline_history=_history())
    assert r.eligible is False and r.boundary_ok is False
    r2 = classify_compact(_Survey(), _GOOD_UV, None, ScanConfig(), outline_history=_history())
    assert r2.boundary_ok is False


def test_off_center_and_tilt_reject():
    off = _GOOD_UV + np.array([0.28, 0.0])
    r = classify_compact(_Survey(), off, _BOUNDARY, ScanConfig(), outline_history=[off] * 5)
    assert r.centered_ok is False
    tilted = _Survey()
    tilted.tilt_deg = 15.0
    r2 = classify_compact(tilted, _GOOD_UV, _BOUNDARY, ScanConfig(), outline_history=_history())
    assert r2.tilt_ok is False and r2.eligible is False


def test_identity_needs_consistent_multiframe_history():
    assert rectangle_identity_consistent(_history(5, jitter=0.001), tol_uv=0.04, min_frames=5)
    assert not rectangle_identity_consistent(_history(3), tol_uv=0.04, min_frames=5)
    wild = _history(4) + [_GOOD_UV + 0.2]
    assert not rectangle_identity_consistent(wild, tol_uv=0.04, min_frames=5)


def test_coverage_gate_deferred_when_none_but_enforced_when_given():
    ok = classify_compact(_Survey(), _GOOD_UV, _BOUNDARY, ScanConfig(),
                          predicted_coverage=None, outline_history=_history())
    assert ok.coverage_ok is True  # deferred to generate_scan_targets' hard gate
    low = classify_compact(_Survey(), _GOOD_UV, _BOUNDARY, ScanConfig(),
                           predicted_coverage=0.5, outline_history=_history())
    assert low.coverage_ok is False and low.eligible is False


# --- Task 18 review, Important 3: tilt_deg is None (not absent) on an
# undetected survey -- getattr(survey, "tilt_deg", 90.0)'s default only fires
# when the attribute is MISSING, never when it exists and is None, so an
# undetected survey (survey_surface's own _not_detected() always sets
# tilt_deg=None) used to crash float(None) here before "no surface detected"
# was ever reached.

class _UndetectedSurvey:
    detected = False
    fully_framed = False
    tilt_deg = None


class _LevelSurvey:
    detected = True
    fully_framed = True
    tilt_deg = 0.0            # perfectly fronto-parallel -- falsy, not "unknown"


def test_tilt_deg_none_fails_tilt_gate_without_crashing():
    r = classify_compact(_UndetectedSurvey(), None, None, ScanConfig(), outline_history=[])
    assert r.eligible is False
    assert r.tilt_ok is False
    assert any("no surface detected" in reason for reason in r.reasons)
    print("[tilt None] undetected survey -> tilt_ok False, no crash:", r.reasons)


def test_tilt_deg_zero_is_not_mistaken_for_unmeasured():
    """A perfectly level plane reads tilt_deg=0.0 -- falsy in Python, so a naive
    `getattr(...) or 90.0` fallback (rejected in review) would wrongly treat it
    as "unknown" and fail tilt_ok. The real fix must tell "attribute is None"
    apart from "attribute is 0.0"."""
    r = classify_compact(_LevelSurvey(), _GOOD_UV, _BOUNDARY, ScanConfig(),
                         outline_history=_history())
    assert r.tilt_ok is True, r
    print("[tilt zero] tilt_deg=0.0 (level) -> tilt_ok True, not misread as unmeasured")
