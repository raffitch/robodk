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
    # §10: a soft collision bypass is not appropriate for a production target
    # set, so scan defaults to the STRICT collision gate (unlike calibration's
    # collision_filter_hard_fail, which stays soft — this is scan-only).
    assert s.collision_filter_hard_fail is True
    print("[defaults] scan present; prefix", s.target_prefix,
          "voxel", s.voxel_size_m, "ideal", s.ideal_distance_mm)


def test_collision_hard_fail_is_default():
    """Dedicated regression for the Task 8 default flip: a fresh ScanConfig()
    must refuse (not silently bypass) a noisy collision map unless the operator
    explicitly opts back into the soft path."""
    assert ScanConfig().collision_filter_hard_fail is True
    print("[defaults] scan collision_filter_hard_fail defaults to True (hard fail)")


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


def test_compact_guard_uv_is_satisfiable_at_the_recommended_standoff():
    """Task 18 review, Critical 2: planner.py's plan_scan frames the surface at
    a standoff where it spans dim/frame_margin of each image axis (the fit-to-
    frame standoff times frame_margin's comfortable border), leaving a PER-SIDE
    normalized margin of (1 - 1/frame_margin) / 2 -- split between the two
    edges. classify_compact's compact_guard_uv gate then demands every raw
    corner sit at least compact_guard_uv inside the frame on EVERY side. If
    compact_guard_uv exceeds that per-side margin, EVERY framed-limited compact
    lock taken at the system's own recommended ideal_distance_mm fails
    guard_ok -- not because the operator aimed badly, but because the two
    constants are mutually unsatisfiable. And silently: the distance/framed
    lamps still read green, nothing tells the operator to back off.

    Required invariant, derived directly from that per-side margin:

        (1 - 1/frame_margin) / 2 >= compact_guard_uv
        1 - 1/frame_margin       >= 2 * compact_guard_uv
        1/frame_margin           <= 1 - 2 * compact_guard_uv
        frame_margin              >= 1 / (1 - 2 * compact_guard_uv)

    Asserted directly from the CONFIG DEFAULTS (not a synthetic scene) so a
    future change to either constant fails this test loudly instead of
    silently making every compact lock unsatisfiable -- which is exactly what
    happened: compact_guard_uv was 0.06 against frame_margin's 1.12, requiring
    frame_margin >= 1 / (1 - 0.12) = 1.13636..., and 1.12 < 1.13636 -- failed
    by ~1.5% of a percentage point on every single surface.
    """
    s = ScanConfig()
    required_frame_margin = 1.0 / (1.0 - 2.0 * s.compact_guard_uv)
    assert s.frame_margin >= required_frame_margin, (
        f"frame_margin={s.frame_margin} must be >= {required_frame_margin:.6f} "
        f"(1 / (1 - 2*compact_guard_uv), compact_guard_uv={s.compact_guard_uv}) "
        "or every framed-limited compact lock at the recommended standoff fails guard_ok")
    print(f"[guard invariant] frame_margin={s.frame_margin} >= required "
          f"{required_frame_margin:.4f} (compact_guard_uv={s.compact_guard_uv})")


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


# -- protocol 2: depth_scale removed, 1080p default, K migration ------------
def test_depth_scale_is_gone_and_a_stale_override_is_dropped_with_a_warning(tmp_path, capsys):
    from tasni.core.config import ScanConfig, load_config
    assert "depth_scale" not in ScanConfig.model_fields
    p = tmp_path / "tasni.config.json"
    p.write_text('{"scan": {"depth_scale": 1000.0, "voxel_size_m": 0.002}}', encoding="utf-8")
    cfg = load_config(p)
    assert cfg.scan.voxel_size_m == 0.002
    assert "scan.depth_scale" in capsys.readouterr().out


def test_calibrated_720p_intrinsics_migrate_to_1080p_by_exact_scale():
    from tasni.core.config import CameraConfig, migrate_camera_intrinsics, _DEFAULT_INTRINSICS
    cam = CameraConfig()
    assert cam.resolution == "1920x1080"
    cal = [[889.8742, 0.0, 648.9804], [0.0, 890.8099, 362.0046], [0.0, 0.0, 1.0]]
    cam.intrinsics = {**cam.intrinsics, "1280x720": cal}
    assert migrate_camera_intrinsics(cam) is True
    K = cam.K
    assert abs(K[0, 0] - 889.8742 * 1.5) < 1e-6 and abs(K[1, 2] - 362.0046 * 1.5) < 1e-6
    assert migrate_camera_intrinsics(cam) is False                      # idempotent
    fresh = CameraConfig()
    assert migrate_camera_intrinsics(fresh) is False                    # factory 720p: nothing to carry
    assert fresh.intrinsics["1920x1080"] == _DEFAULT_INTRINSICS["1920x1080"]


if __name__ == "__main__":
    test_defaults_present_and_sane()
    test_collision_hard_fail_is_default()
    test_json_override_merges_only_targeted_fields()
    test_compact_guard_uv_is_satisfiable_at_the_recommended_standoff()
    test_unknown_scan_key_rejected()
    test_calibrated_720p_intrinsics_migrate_to_1080p_by_exact_scale()
    print("\nScanConfig tests passed.")
