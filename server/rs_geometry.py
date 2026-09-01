"""Depth/colour camera models and the depth->colour extrinsic, for the greeting.

Takes the ``rs`` module as a parameter (host tests stub pyrealsense2). The
extrinsic is transposed from librealsense's column-major layout to row-major and
CHECKED against ``rs2_transform_point_to_point`` on a test point; a mismatch
raises. The server used to resolve the layout empirically on every telemetry
frame by projecting eight sample points both ways (see git history of
``stream_h264``); the host must never inherit that guess.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np


def intrinsics_dict(intr) -> dict:
    model = getattr(intr, "model", None)
    return {
        "width": int(intr.width), "height": int(intr.height),
        "fx": float(intr.fx), "fy": float(intr.fy),
        "ppx": float(intr.ppx), "ppy": float(intr.ppy),
        "model": str(getattr(model, "name", model)) if model is not None else "none",
        "coeffs": [float(c) for c in (getattr(intr, "coeffs", None) or [0, 0, 0, 0, 0])],
    }


def extrinsic_row_major(depth_profile, color_profile, rs) -> tuple[np.ndarray, np.ndarray]:
    ext = depth_profile.get_extrinsics_to(color_profile)
    R = np.asarray(ext.rotation, dtype=float).reshape(3, 3).T      # column-major -> row-major
    t_m = np.asarray(ext.translation, dtype=float)
    probe = [0.12, -0.05, 0.45]                                       # metres, in front of the camera
    expected = np.asarray(rs.rs2_transform_point_to_point(ext, probe), dtype=float)
    got = R @ np.asarray(probe) + t_m
    if not np.allclose(got, expected, atol=1e-6):
        raise RuntimeError(
            f"depth->colour extrinsic layout check failed: transposed rotation maps the "
            f"probe to {got.tolist()} but the SDK says {expected.tolist()}; refusing to "
            f"serve a geometry that would be wrong for every client")
    return R, t_m * 1000.0


@dataclass(frozen=True)
class StaticGeometry:
    depth: dict
    color: dict
    R_dc: np.ndarray
    t_dc_mm: np.ndarray
    depth_size: tuple[int, int]
    color_size: tuple[int, int]


def static_geometry(profile, rs) -> StaticGeometry:
    depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    R, t_mm = extrinsic_row_major(depth_profile, color_profile, rs)
    d, c = intrinsics_dict(depth_profile.intrinsics), intrinsics_dict(color_profile.intrinsics)
    return StaticGeometry(depth=d, color=c, R_dc=R, t_dc_mm=t_mm,
                          depth_size=(d["width"], d["height"]),
                          color_size=(c["width"], c["height"]))


def build_greeting(static: StaticGeometry, *, depth_unit_mm: float, filters: list,
                   temps: dict, global_time_enabled, achieved: dict, device: dict,
                   spatial_smooth_delta) -> dict:
    """``filters`` names the chain that ran; ``filter_options`` says what it ran AT.

    ``spatial_smooth_delta`` is the ACHIEVED value read back off the spatial filter,
    ``None`` when no spatial filter is in the chain (or when the SDK would not report
    it). It is REQUIRED rather than defaulted: it is the only record of which arm of a
    smooth_delta A/B a take came from, and a greeting path that forgot it would archive
    two indistinguishable arms with no error (docs/inspection-roll-probe-handoff.md 3.1).
    """
    return {
        "protocol": 2,
        "aligned": False,
        "depth_unit_mm": float(depth_unit_mm),
        "depth": dict(static.depth),
        "color": dict(static.color),
        "depth_to_color": {
            "rotation_row_major": np.asarray(static.R_dc, float).round(12).tolist(),
            "translation_mm": np.asarray(static.t_dc_mm, float).round(6).tolist(),
        },
        "filters": list(filters),
        "filter_options": {
            "spatial_smooth_delta": (None if spatial_smooth_delta is None
                                     else float(spatial_smooth_delta)),
        },
        "device": {**dict(device),
                   "visual_preset": achieved.get("visual_preset"),
                   "laser_power": achieved.get("laser_power")},
        "temps": dict(temps),
        "global_time_enabled": global_time_enabled,
    }


def greeting_line(greeting: dict) -> bytes:
    return json.dumps(greeting, separators=(",", ":")).encode("utf-8") + b"\n"
