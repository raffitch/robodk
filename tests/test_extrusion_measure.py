"""Ring-stack measure-only experiment: synthetic proof, processing, jobs, API."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import extrusion_synthetic as syn
from tasni.modules.extrusion.processing import depth_to_work_points


def test_renderer_puts_a_ring_where_it_says_at_the_height_it_says():
    center = (200.0, 150.0)
    T = syn.inspection_camera_T([center[0], center[1], 6.0], 300.0)
    depth = syn.render_scene([syn.RingSpec(60.0, 8.0, center, height_fn=syn.flat(6.0))], T,
                             plane_center_xy_mm=center, noise_mm=0.0)
    assert depth.dtype == np.uint16 and depth.shape == (720, 1280)
    points, raw = depth_to_work_points(depth, syn.K_720P, T)
    assert raw > 100_000                                  # plane + ring both rendered
    ring = points[points[:, 2] > 3.0]
    radii = np.linalg.norm(ring[:, :2] - np.array(center), axis=1)
    assert 55.0 < radii.min() and radii.max() < 65.0     # 60 +/- bead/2 (+ rounding)
    assert 5.0 < ring[:, 2].max() < 7.0                   # crest at 6 mm
    plane = points[points[:, 2] <= 1.0]
    assert len(plane) > 50_000


# ---------------------------------------------------------------- circle metrics

from tasni.modules.extrusion.comparison import compare_circle


def test_shifted_circle_reports_its_offset_and_zero_shape_error():
    theta = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    shifted = np.column_stack((40 * np.cos(theta) + 10, 40 * np.sin(theta), np.full(360, 5.0)))
    m = compare_circle(shifted, 40.0, nominal_center_mm=(0.0, 0.0))
    assert m.center_offset_mm == pytest.approx((10.0, 0.0), abs=1e-6)
    assert m.center_offset_norm_mm == pytest.approx(10.0, abs=1e-6)
    assert m.shape_rms_mm < 1e-6 and m.shape_max_mm < 1e-6
    # Deviation is still measured from the NOMINAL centre (the paper's number).
    assert m.mean_absolute_mm == pytest.approx(6.35, abs=0.05)
    assert m.rms_mm == pytest.approx(7.06, abs=0.05)
    assert m.maximum_mm == pytest.approx(10.0, abs=0.05)


def test_old_metrics_payload_without_offset_fields_still_validates():
    from tasni.modules.extrusion.models import DeviationMetrics
    old = DeviationMetrics(mean_absolute_mm=1, rms_mm=1, maximum_mm=1,
                           measured_center_mm=(0, 0), measured_radius_mm=40,
                           path_completeness=1, maximum_angular_gap_deg=2, valid=True)
    assert old.center_offset_norm_mm == 0.0 and old.shape_rms_mm == 0.0
