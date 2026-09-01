"""The greeting's extrinsic must be row-major and PROVEN against the SDK's own
transform; the server used to re-derive this empirically every telemetry frame."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))

from server import rs_geometry  # noqa: E402


def _rot_z(deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


R_TRUE = _rot_z(3.0) @ np.array([[1, 0, 0], [0, 0.999, -0.0447], [0, 0.0447, 0.999]])
T_TRUE_M = np.array([0.0147, -0.0002, 0.0003])


def _fake_rs(column_major: bool):
    """A pyrealsense2 stand-in whose rs2_transform_point_to_point is the ground
    truth, and whose .rotation layout is column-major (correct) or not (bug)."""
    rot = (R_TRUE.T if column_major else R_TRUE).reshape(-1).tolist()
    ext = SimpleNamespace(rotation=rot, translation=T_TRUE_M.tolist())
    def transform(e, p):
        return (R_TRUE @ np.asarray(p, float) + T_TRUE_M).tolist()
    return ext, SimpleNamespace(rs2_transform_point_to_point=transform)


def _profiles(ext):
    intr_d = SimpleNamespace(width=1280, height=720, fx=640.0, fy=640.0, ppx=640.0, ppy=360.0,
                             model=SimpleNamespace(name="brown_conrady"), coeffs=[0, 0, 0, 0, 0])
    intr_c = SimpleNamespace(width=1920, height=1080, fx=1362.0, fy=1362.0, ppx=975.0, ppy=550.0,
                             model=SimpleNamespace(name="brown_conrady"), coeffs=[0, 0, 0, 0, 0])
    depth = SimpleNamespace(intrinsics=intr_d, get_extrinsics_to=lambda other: ext)
    color = SimpleNamespace(intrinsics=intr_c)
    return depth, color


def test_column_major_rotation_is_transposed_into_row_major_and_verified():
    ext, rs = _fake_rs(column_major=True)
    depth, color = _profiles(ext)
    R, t_mm = rs_geometry.extrinsic_row_major(depth, color, rs)
    np.testing.assert_allclose(R, R_TRUE, atol=1e-12)
    np.testing.assert_allclose(t_mm, T_TRUE_M * 1000.0, atol=1e-9)


def test_wrong_layout_is_refused_not_guessed():
    ext, rs = _fake_rs(column_major=False)
    depth, color = _profiles(ext)
    try:
        rs_geometry.extrinsic_row_major(depth, color, rs)
    except RuntimeError as e:
        assert "extrinsic" in str(e)
    else:
        raise AssertionError("a mismatching rotation layout must raise")


def test_greeting_is_one_json_line_with_protocol_2():
    ext, rs = _fake_rs(column_major=True)
    depth, color = _profiles(ext)
    static = rs_geometry.StaticGeometry(
        depth=rs_geometry.intrinsics_dict(depth.intrinsics),
        color=rs_geometry.intrinsics_dict(color.intrinsics),
        R_dc=R_TRUE, t_dc_mm=T_TRUE_M * 1000.0, depth_size=(1280, 720), color_size=(1920, 1080))
    g = rs_geometry.build_greeting(
        static, depth_unit_mm=0.1, filters=["threshold", "disparity", "spatial", "temporal",
                                            "disparity_inv"],
        temps={"asic_c": 41.5, "projector_c": 38.0}, global_time_enabled=True,
        achieved={"visual_preset": 0.0, "laser_power": 150.0},
        spatial_smooth_delta=20.0,
        device={"serial": "S1", "fw": "5.16.00.01", "librealsense": "2.55.1"})
    line = rs_geometry.greeting_line(g)
    assert line.endswith(b"\n") and line.count(b"\n") == 1
    back = json.loads(line.decode("utf-8"))
    assert back["protocol"] == 2 and back["aligned"] is False
    assert back["depth_unit_mm"] == 0.1
    assert back["depth"]["width"] == 1280 and back["color"]["width"] == 1920
    np.testing.assert_allclose(back["depth_to_color"]["rotation_row_major"], R_TRUE, atol=1e-12)
    assert back["depth_to_color"]["translation_mm"][0] == 14.7
    assert back["device"]["visual_preset"] == 0.0 and back["temps"]["asic_c"] == 41.5


def _static():
    ext, rs = _fake_rs(column_major=True)
    depth, color = _profiles(ext)
    return rs_geometry.StaticGeometry(
        depth=rs_geometry.intrinsics_dict(depth.intrinsics),
        color=rs_geometry.intrinsics_dict(color.intrinsics),
        R_dc=R_TRUE, t_dc_mm=T_TRUE_M * 1000.0,
        depth_size=(1280, 720), color_size=(1920, 1080))


def _greeting(filters=None, **kw):
    return rs_geometry.build_greeting(
        _static(), depth_unit_mm=0.1,
        filters=filters or ["threshold", "disparity", "spatial", "temporal",
                            "disparity_inv"],
        temps={"asic_c": 41.5}, global_time_enabled=True,
        achieved={"visual_preset": 0.0, "laser_power": 150.0},
        device={"serial": "S1"}, **kw)


def test_the_greeting_carries_the_spatial_filters_achieved_smooth_delta():
    """The filter NAMES say whether the spatial filter ran; only this says what it
    ran AT. Without it the two arms of a smooth_delta A/B archive identical
    provenance (docs/inspection-roll-probe-handoff.md 3.1)."""
    g = _greeting(spatial_smooth_delta=4.0)
    assert g["filter_options"]["spatial_smooth_delta"] == 4.0
    assert json.loads(rs_geometry.greeting_line(g).decode("utf-8")
                      )["filter_options"]["spatial_smooth_delta"] == 4.0


def test_no_spatial_filter_records_null_not_a_number():
    """``None`` is the control arm (no spatial filter at all), and it must be JSON
    ``null`` -- distinct from any delta a running filter could report."""
    g = _greeting(spatial_smooth_delta=None,
                  filters=["threshold", "disparity", "temporal", "disparity_inv"])
    assert g["filter_options"]["spatial_smooth_delta"] is None
    assert b'"spatial_smooth_delta":null' in rs_geometry.greeting_line(g)
