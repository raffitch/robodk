"""Centring the cylinder on the middle of the platform, resolved from the STATION.

The old path read the platform only from ``runs/scan/active.json``, so deleting a
session orphaned a placement whose geometry was still sitting in RoboDK. These tests
pin the replacement: the centre comes from the station first (a scan-inserted centre
frame, else the work-surface object's own mesh), and the run pointer on disk is only
the last resort. The centre is then expressed in *whichever* frame the operator picked
in the Work frame dropdown.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tasni.modules.extrusion.surface import (CENTER_FRAME_NAME, SURFACE_OBJECT_NAME,
                                             mesh_corners, platform_from_corners,
                                             resolve_platform_center)

# A 400 x 300 rectangle whose corner sits at the origin of "Tasni Work Frame".
CORNERS = np.array([[0.0, 0.0, 0.0], [400.0, 0.0, 0.0],
                    [400.0, 300.0, 0.0], [0.0, 300.0, 0.0]])


def active_payload(**updates) -> dict:
    payload = {"module": "scan", "run_id": "20260812-101500",
               "frame": "Tasni Work Frame", "rectangle": SURFACE_OBJECT_NAME,
               "size_mm": [400.0, 300.0],
               "rectangle_corners_frame_mm": CORNERS.tolist()}
    payload.update(updates)
    return payload


# --------------------------------------------------------------------------- mesh


def test_mesh_corners_dedupes_the_triangle_soup_roboDK_returns():
    """add_rectangle writes 4 triangles, so GetPoints yields 12 XYZijk rows."""
    tri = [0, 1, 2, 0, 2, 3, 0, 2, 1, 0, 3, 2]
    points = [list(CORNERS[i]) + [0.0, 0.0, 1.0] for i in tri]
    corners = mesh_corners(points)
    assert corners.shape == (4, 3)
    np.testing.assert_allclose(np.sort(corners, axis=0), np.sort(CORNERS, axis=0))


def test_mesh_corners_refuses_degenerate_input():
    assert mesh_corners([]) is None
    assert mesh_corners([[0, 0, 0], [1, 1, 1]]) is None          # < 3 distinct
    assert mesh_corners([[0, 0, np.nan], [1, 0, 0], [0, 1, 0]]) is None


# ----------------------------------------------------------------- corners -> box


def test_platform_from_corners_gives_the_middle_not_the_origin():
    box = platform_from_corners(CORNERS)
    assert box["center_mm"] == [200.0, 150.0]
    assert box["extents_known"] is True
    assert box["size_mm"] == [400.0, 300.0]
    assert box["bounds_mm"]["x_max"] == 400.0


def test_platform_centre_is_exact_even_when_the_frame_is_rotated():
    """Expressed in a rotated frame the rectangle is no longer axis aligned.

    The centre (mean of the corners) stays exact under any rotation, but the
    axis-aligned bounds would overstate the surface, so extents must be withheld
    rather than reported wrong.
    """
    th = np.deg2rad(30.0)
    R = np.array([[np.cos(th), -np.sin(th), 0.0], [np.sin(th), np.cos(th), 0.0],
                  [0.0, 0.0, 1.0]])
    box = platform_from_corners(CORNERS @ R.T)
    np.testing.assert_allclose(box["center_mm"], (R @ [200.0, 150.0, 0.0])[:2], atol=1e-9)
    assert box["extents_known"] is False
    assert box["bounds_mm"] is None and box["size_mm"] is None


def test_platform_from_corners_handles_a_frame_whose_y_points_off_the_rectangle():
    box = platform_from_corners(np.array([[0.0, 0.0, 0.0], [400.0, 0.0, 0.0],
                                          [400.0, -300.0, 0.0], [0.0, -300.0, 0.0]]))
    assert box["center_mm"] == [200.0, -150.0]
    assert box["size_mm"] == [400.0, 300.0]


# ------------------------------------------------------------------- resolution


def test_centre_frame_wins_and_needs_no_extents():
    """Tasni Work Center's origin IS the middle, so a bare origin is enough."""
    got = resolve_platform_center("Tasni Work Frame",
                                  center_frame_xyz=[200.0, 150.0, 0.0],
                                  surface_corners=CORNERS, active=active_payload())
    assert got["source"] == "center_frame"
    assert got["center_mm"] == [200.0, 150.0]
    assert got["frame"] == "Tasni Work Frame"
    # Extents come from the surface object when it is there alongside the frame.
    assert got["extents_known"] is True and got["size_mm"] == [400.0, 300.0]


def test_surface_object_is_used_when_the_centre_frame_is_absent():
    got = resolve_platform_center("Tasni Work Frame", surface_corners=CORNERS,
                                  active=active_payload())
    assert got["source"] == "surface_object"
    assert got["center_mm"] == [200.0, 150.0]
    assert got["run_id"] is None          # station-sourced: no run to claim


def test_disk_pointer_is_the_last_resort_and_carries_its_run_id():
    got = resolve_platform_center("Tasni Work Frame", active=active_payload())
    assert got["source"] == "scan_run"
    assert got["center_mm"] == [200.0, 150.0]
    assert got["run_id"] == "20260812-101500"


def test_disk_pointer_is_refused_for_a_different_work_frame():
    """active.json's corners are in ITS frame; they say nothing about another one."""
    assert resolve_platform_center("World", active=active_payload()) is None


def test_station_sources_work_in_any_selected_frame():
    """The dropdown picks the coordinate system, not the location."""
    got = resolve_platform_center("World", center_frame_xyz=[1200.0, -340.0, 25.0])
    assert got["center_mm"] == [1200.0, -340.0]
    assert got["frame"] == "World"
    assert got["extents_known"] is False


def test_build_plane_z_comes_from_the_platform_not_from_zero():
    """Zero is only the build plane while the selected frame sits ON the surface.

    Picking World (or any frame off the table) must still put the first bead on the
    platform, so the resolved Z travels with the centre.
    """
    assert resolve_platform_center(
        "World", center_frame_xyz=[1200.0, -340.0, 25.0])["center_z_mm"] == 25.0
    raised = CORNERS + np.array([0.0, 0.0, 25.0])
    assert resolve_platform_center(
        "World", surface_corners=raised)["center_z_mm"] == 25.0
    assert resolve_platform_center(
        "Tasni Work Frame", active=active_payload())["center_z_mm"] == 0.0


def test_nothing_resolvable_returns_none():
    assert resolve_platform_center("Tasni Work Frame") is None
    assert resolve_platform_center("Tasni Work Frame", active={"frame": "Tasni Work Frame"}) is None


def test_names_match_what_the_scan_module_inserts():
    from tasni.modules.scan.service import RECT_NAME

    assert SURFACE_OBJECT_NAME == RECT_NAME
    assert CENTER_FRAME_NAME == "Tasni Work Center"


# ------------------------------------------------------- fit survives no run id


def _recipe(**updates):
    from tasni.modules.extrusion.models import CylinderRecipe

    values = dict(radius_mm=40, layer_count=3, layer_height_mm=5, bead_diameter_mm=6,
                  robot_speed_mm_s=75, extrusion_rate_pct=30, points_per_circle=72)
    values.update(updates)
    return CylinderRecipe(**values)


def _setup(**updates):
    from tasni.modules.extrusion.models import CylinderSetup

    values = dict(print_tool="Nozzle", work_frame="Tasni Work Frame",
                  inspection_tool="Camera", inspection_target="Inspect",
                  center_x_mm=200, center_y_mm=150, orientation_rpy_deg=(180, 0, 90))
    values.update(updates)
    return CylinderSetup(**values)


def _surface(**updates) -> dict:
    values = dict(frame="Tasni Work Frame", run_id=None, applied_at=None,
                  size_mm=[400.0, 300.0], available=True, center_mm=[200.0, 150.0],
                  corners_frame_mm=CORNERS.tolist(), note="",
                  bounds_mm={"x_min": 0.0, "x_max": 400.0,
                             "y_min": 0.0, "y_max": 300.0})
    values.update(updates)
    return values


def test_station_placed_wall_that_overhangs_is_still_rejected():
    """A station-sourced centre has no run id, so it reads as 'manual'.

    Without this the fit check would silently stop guarding: the platform is the
    table, and a wall hanging off it is wrong however the centre was obtained.
    """
    from tasni.modules.extrusion.surface import surface_check

    check = surface_check(_setup(), _recipe(radius_mm=170), _surface())
    assert check["ok"] is False
    assert check["fit"]["checked"] is True
    assert "overhangs" in check["problem"]


def test_station_placed_wall_that_fits_passes_with_the_margin_reported():
    from tasni.modules.extrusion.surface import surface_check

    check = surface_check(_setup(), _recipe(), _surface())
    assert check["ok"] is True
    assert check["fit"]["minimum_margin_mm"] == 107.0


def test_a_plan_in_another_frame_is_not_measured_against_this_platform():
    """Only an advisory: the rectangle's bounds mean nothing in a frame it is not in."""
    from tasni.modules.extrusion.surface import surface_check

    check = surface_check(_setup(work_frame="World", center_x_mm=0, center_y_mm=0),
                          _recipe(radius_mm=170), _surface())
    assert check["ok"] is True and check["placement"] == "manual"
    assert "placed manually" in check["advisory"]
    assert check.get("fit") is None


def test_extents_unknown_leaves_an_advisory_rather_than_a_false_pass():
    from tasni.modules.extrusion.surface import surface_check

    check = surface_check(_setup(), _recipe(radius_mm=170),
                          _surface(bounds_mm=None, size_mm=None))
    assert check["ok"] is True
    assert "extents" in check["advisory"]
