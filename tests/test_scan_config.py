"""ScanConfig defaults + JSON-override merge + unknown-key guard.

Pure config — no RoboDK, no camera. Mirrors the layered-config semantics the
calibration module relies on (validate-on-assignment, deep-merge, forbid extras).

    py -3.10 tests/test_scan_config.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tasni.core.config import AppConfig, ScanConfig, load_config  # noqa: E402


def test_defaults_present_and_sane():
    cfg = AppConfig()
    s = cfg.scan
    assert isinstance(s, ScanConfig)
    assert s.target_prefix == "TasniScan_"          # never reuse calibration targets
    assert s.target_prefix != cfg.robodk.target_prefix
    # standoff band is a real +/- window around the ideal distance
    assert s.distance_tol_mm > 0 and s.ideal_distance_mm > s.distance_tol_mm
    # TSDF knobs in metres, ordered sensibly
    assert 0 < s.voxel_size_m < s.sdf_trunc_m
    assert 0 < s.depth_min_m < s.depth_max_m
    # pose generation reuses the cone+roll generator's parameter names
    for k in ("pose_count", "cone_half_angle_deg", "roll_max_deg",
              "distance_jitter", "look_distance_mm"):
        assert hasattr(s, k)
    assert s.collision_filter_hard_fail is False
    print("[defaults] scan present; prefix", s.target_prefix,
          "voxel", s.voxel_size_m, "ideal", s.ideal_distance_mm)


def test_json_override_merges_only_targeted_fields():
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "tasni.config.json"
        p.write_text(json.dumps(
            {"scan": {"voxel_size_m": 0.002, "pose_count": 20}}), encoding="utf-8")
        cfg = load_config(p)
        assert cfg.scan.voxel_size_m == 0.002
        assert cfg.scan.pose_count == 20
        # untouched fields keep their defaults; other sections unaffected
        assert cfg.scan.target_prefix == "TasniScan_"
        assert cfg.calibration.pose_count == AppConfig().calibration.pose_count
    print("[merge] scan overrides applied; rest defaulted")


def test_unknown_scan_key_rejected():
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "tasni.config.json"
        p.write_text(json.dumps({"scan": {"voxel_size_mm": 4}}), encoding="utf-8")
        try:
            load_config(p)
            raise AssertionError("expected an unknown-key error")
        except KeyError as e:
            assert "voxel_size_mm" in str(e)
    print("[guard] unknown scan key -> KeyError (typo is an error, not a no-op)")


if __name__ == "__main__":
    test_defaults_present_and_sane()
    test_json_override_merges_only_targeted_fields()
    test_unknown_scan_key_rejected()
    print("\nScanConfig tests passed.")
