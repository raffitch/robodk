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
