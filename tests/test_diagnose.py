"""Unit test for quality.diagnose() — the metrics->verdict mapping (Phase 3 #10).

Pure function of a CalibrationReport's numbers, so no robot/camera. Checks the
pass/borderline/fail headline and that the cause attribution distinguishes the
camera-model fault (high reprojection + tight spread) from the geometry fault
(large board-consistency spread).

    py -3.10 tests/test_diagnose.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tasni.modules.calibration.quality import (  # noqa: E402
    CalibrationReport, SplitMetrics, diagnose)

_I4 = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]


def _report(train_px, val_px, bc_rms, *, well=True, intr_warn=False):
    return CalibrationReport(
        refined=True, method="TSAI", X_cam2gripper=_I4, T_base_target=_I4,
        train=SplitMetrics(12, train_px, train_px * 2),
        validation=(SplitMetrics(3, val_px, val_px * 2) if val_px is not None else None),
        board_consistency_mm={"rms": bc_rms, "max": bc_rms * 2},
        motion_diversity={"well_conditioned": well, "axis_spread": 0.2,
                          "min_pair_deg": 10.0, "max_pair_deg": 120.0, "n_pairs": 60},
        intrinsics_check=({"warn": True, "note": "fx off by 4%"} if intr_warn else None),
    )


def _has(causes, *needles):
    return any(all(n in c for n in needles) for c in causes)


def test_pass_when_all_tight():
    d = diagnose(_report(0.3, 0.4, 0.2))
    assert d["verdict"] == "pass", d
    print("[pass]", d["headline"])


def test_intrinsics_pattern_high_reproj_tight_spread():
    # mid reprojection (1-3 px) but a tight mm spread -> blame the camera model.
    d = diagnose(_report(1.8, 2.0, 0.3))
    assert d["verdict"] == "borderline", d
    assert _has(d["causes"], "intrinsics/distortion"), d["causes"]
    assert not _has(d["causes"], "robot-pose"), d["causes"]
    print("[intrinsics]", d["causes"][0][:60])


def test_geometry_pattern_large_spread_fails():
    # tight reprojection but a large mm spread -> blame geometry; spread >5 mm = fail.
    d = diagnose(_report(0.6, 0.8, 6.0))
    assert d["verdict"] == "fail", d
    assert _has(d["causes"], "robot-pose"), d["causes"]
    assert not _has(d["causes"], "intrinsics/distortion"), d["causes"]
    print("[geometry]", d["causes"][0][:60])


def test_high_reproj_fails():
    d = diagnose(_report(4.0, 5.0, 0.4))
    assert d["verdict"] == "fail", d
    print("[high reproj]", d["headline"])


def test_weak_diversity_is_borderline():
    d = diagnose(_report(0.4, 0.4, 0.2, well=False))
    assert d["verdict"] == "borderline", d
    assert _has(d["causes"], "motion diversity"), d["causes"]
    print("[weak diversity]", d["causes"][0][:60])


def test_intrinsics_warn_flagged():
    d = diagnose(_report(0.5, 0.6, 0.3, intr_warn=True))
    assert d["verdict"] == "borderline", d
    assert _has(d["causes"], "intrinsics self-check"), d["causes"]
    print("[intrinsics warn]", d["causes"][0][:60])


def test_overfit_flagged():
    # train tight, val much larger (and mid) -> overfit note.
    d = diagnose(_report(0.5, 1.5, 0.3))
    assert _has(d["causes"], "overfit"), d["causes"]
    print("[overfit]", d["causes"][-1][:60])


if __name__ == "__main__":
    test_pass_when_all_tight()
    test_intrinsics_pattern_high_reproj_tight_spread()
    test_geometry_pattern_large_spread_fails()
    test_high_reproj_fails()
    test_weak_diversity_is_borderline()
    test_intrinsics_warn_flagged()
    test_overfit_flagged()
    print("\nDiagnose verdict tests passed.")
