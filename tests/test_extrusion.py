"""Cylinder planning, comparison, correction, archive, and API foundations."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from tasni.core import runs
from tasni.core.config import AppConfig, ExtrusionConfig
from tasni.modules.extrusion.archive import ExtrusionArchive
from tasni.modules.extrusion.comparison import compare_circle, corrected_circle
from tasni.modules.extrusion.inspection import (aim_point_mm, cylinder_diameter_mm,
                                                framing_standoff, inspection_plan,
                                                pose_candidates)
from tasni.modules.extrusion.models import CylinderRecipe, CylinderSetup, LayerManifest
from tasni.modules.extrusion.service import geometry_preflight, station_requirements
from tasni.modules.extrusion.surface import (active_scan_surface, surface_check,
                                             surface_fit)
from tasni.modules.extrusion.toolpath import generate_cylinder_plan, points_array
from tasni.webapp.server import create_app
from tools.setup_extrusion_station import LICENSED_ISOLATED_ARGS, expected_instructions


def recipe(**updates) -> CylinderRecipe:
    values = dict(radius_mm=40, layer_count=3, layer_height_mm=5,
                  bead_diameter_mm=6, robot_speed_mm_s=75,
                  extrusion_rate_pct=30, points_per_circle=72)
    values.update(updates)
    return CylinderRecipe(**values)


def setup(**updates) -> CylinderSetup:
    values = dict(print_tool="Nozzle", work_frame="BuildFrame",
                  inspection_tool="Camera", inspection_target="Inspect",
                  center_x_mm=10, center_y_mm=-5,
                  orientation_rpy_deg=(180, 0, 90),
                  approach_clearance_mm=30, retreat_clearance_mm=50)
    values.update(updates)
    return CylinderSetup(**values)


def generate_payload(**recipe_updates) -> dict:
    return {"recipe": recipe(**recipe_updates).model_dump(mode="json"),
            "setup": setup().model_dump(mode="json")}


def test_config_keeps_verified_mapping_separate_from_hardware_approval():
    config = ExtrusionConfig()
    assert config.valve_outputs == ["IO_508", "IO_601"]
    assert config.valve_mapping_verified is True
    assert config.hardware_io_test_approved is False
    assert config.default_print_tool == "" and config.default_work_frame == ""
    assert expected_instructions(config.valve_outputs, 1) == ["Set IO_508=1", "Set IO_601=1"]
    assert expected_instructions(config.valve_outputs, 0) == ["Set IO_508=0", "Set IO_601=0"]
    assert "-NEWINSTANCE" in LICENSED_ISOLATED_ARGS
    assert "-NOUI" in LICENSED_ISOLATED_ARGS
    assert "-SKIPINI" not in LICENSED_ISOLATED_ARGS


def test_plan_is_closed_layered_and_fingerprinted():
    plan = generate_cylinder_plan(recipe(), setup())
    assert len(plan.layers) == 3
    assert [layer.nominal_z_mm for layer in plan.layers] == [3, 8, 13]
    for layer in plan.layers:
        pts = points_array(layer)
        assert pts.shape == (73, 3)
        np.testing.assert_allclose(pts[0], pts[-1], atol=0)
    assert math.isclose(plan.total_path_length_mm, 3 * 2 * math.pi * 40)
    changed = generate_cylinder_plan(recipe(radius_mm=41), setup())
    assert changed.fingerprint != plan.fingerprint
    changed_setup = generate_cylinder_plan(recipe(), setup(print_tool="OtherNozzle"))
    assert changed_setup.fingerprint != plan.fingerprint
    changed_motion = generate_cylinder_plan(
        recipe(travel_speed_mm_s=300, path_rounding_mm=2), setup())
    assert changed_motion.fingerprint != plan.fingerprint


def test_build_plane_z_offsets_every_layer_and_changes_fingerprint():
    baseline = generate_cylinder_plan(recipe(), setup())
    raised = generate_cylinder_plan(recipe(), setup(build_plane_z_mm=125))
    assert [layer.nominal_z_mm for layer in raised.layers] == [128, 133, 138]
    assert raised.fingerprint != baseline.fingerprint


def test_geometry_preflight_does_not_claim_robodk_dry_run():
    result = geometry_preflight(generate_cylinder_plan(recipe(), setup()))
    assert result["all_ok"] is True
    assert result["dry_run_passed"] is False
    assert all(event["physical_output_blocked"] for event in result["simulated_valve_events"])


def test_circle_metrics_and_bounded_correction():
    theta = np.linspace(0, 2 * math.pi, 181)
    measured = np.column_stack((42 * np.cos(theta) + 3,
                                42 * np.sin(theta) - 4,
                                np.full_like(theta, 8)))
    metrics = compare_circle(measured, 40, nominal_center_mm=(3, -4))
    assert metrics.valid
    assert math.isclose(metrics.measured_radius_mm, 42, abs_tol=1e-9)
    assert math.isclose(metrics.rms_mm, 2, abs_tol=1e-9)
    corrected = corrected_circle(measured, 40, 8, nominal_center_mm=(3, -4), point_count=72,
                                 gain=1, smoothing_points=9, max_correction_mm=5)
    radii = np.linalg.norm(corrected[:, :2] - np.array([3, -4]), axis=1)
    np.testing.assert_allclose(radii, 38, atol=1e-8)
    np.testing.assert_allclose(corrected[0], corrected[-1])


def test_archive_writes_reprocessable_layer(tmp_path: Path):
    plan = generate_cylinder_plan(recipe(layer_count=1), setup())
    archive = ExtrusionArchive(tmp_path)
    trial = archive.create_trial("trial-001", plan, provenance={"git_commit": "abc"})
    pts = points_array(plan.layers[0])
    manifest = LayerManifest(trial_id="trial-001", layer_index=1,
                             recipe=plan.recipe, toolpath_fingerprint=plan.fingerprint,
                             color_file="color.png", depth_file="depth.npy")
    color = np.zeros((8, 10, 3), dtype=np.uint8)
    depth = np.full((8, 10), 500, dtype=np.uint16)
    layer = archive.write_layer(manifest, nominal_xyz=pts, commanded_xyz=pts,
                                color=color, depth=depth)
    assert (trial / "trial.json").is_file()
    assert (layer / "manifest.json").is_file()
    assert np.array_equal(np.load(layer / "depth.npy"), depth)
    path_data = json.loads((layer / "nominal_path.json").read_text(encoding="utf-8"))
    assert path_data["frame"] == "work" and path_data["units"] == "mm"


def scan_surface(**updates) -> dict:
    """A 400 x 300 mm measured rectangle whose frame origin is its corner."""
    corners = [[0, 0, 0], [400, 0, 0], [400, 300, 0], [0, 300, 0]]
    values = dict(frame="Tasni Work Frame", run_id="20260812-101500",
                  applied_at="2026-08-12T10:16:00", size_mm=[400.0, 300.0],
                  available=True, center_mm=[200.0, 150.0], corners_frame_mm=corners,
                  bounds_mm={"x_min": 0.0, "x_max": 400.0, "y_min": 0.0, "y_max": 300.0},
                  note="")
    values.update(updates)
    return values


def write_active_scan(root: Path, *, corners=None, run_id="20260812-101500") -> None:
    (root / "runs" / "scan").mkdir(parents=True, exist_ok=True)
    payload = {"module": "scan", "run_id": run_id, "frame": "Tasni Work Frame",
               "rectangle": "Tasni Work Surface", "applied_at": "2026-08-12T10:16:00",
               "size_mm": [400.0, 300.0],
               "rectangle_center_frame_mm": [200.0, 150.0],
               "rectangle_corners_frame_mm": corners if corners is not None else
               [[0, 0, 0], [400, 0, 0], [400, 300, 0], [0, 300, 0]]}
    (root / "runs" / "scan" / "active.json").write_text(json.dumps(payload), encoding="utf-8")


def test_scan_frame_origin_is_a_corner_so_centring_needs_the_rectangle(tmp_path):
    """The whole point of the handoff: frame zero is NOT the middle of the surface."""
    write_active_scan(tmp_path)
    surface = active_scan_surface(root=tmp_path)
    assert surface["available"] is True
    assert surface["center_mm"] == [200.0, 150.0]      # not [0, 0]
    assert surface["frame"] == "Tasni Work Frame"
    assert surface["run_id"] == "20260812-101500"


def test_active_scan_surface_recovers_geometry_from_the_run_report(tmp_path):
    """A payload written before the frame-local fields existed still centres."""
    run = tmp_path / "runs" / "scan" / "20260812-101500"
    run.mkdir(parents=True)
    # Frame sits at base (1000, 500, 300) rotated 90 deg about Z; corners follow.
    frame_T = [[0, -1, 0, 1000], [1, 0, 0, 500], [0, 0, 1, 300], [0, 0, 0, 1]]
    corners = [[1000, 500, 300], [1000, 900, 300], [700, 900, 300], [700, 500, 300]]
    (run / "report.json").write_text(json.dumps(
        {"plane": {"frame_T_mm": frame_T, "corners_mm": corners,
                   "size_mm": [400.0, 300.0]}}), encoding="utf-8")
    (tmp_path / "runs" / "scan" / "active.json").write_text(json.dumps(
        {"module": "scan", "run_id": "20260812-101500", "frame": "Tasni Work Frame",
         "size_mm": [400.0, 300.0]}), encoding="utf-8")
    surface = active_scan_surface(root=tmp_path)
    assert surface["available"] is True
    np.testing.assert_allclose(surface["center_mm"], [200.0, 150.0], atol=1e-9)


def test_unrecoverable_surface_refuses_to_guess_a_centre(tmp_path):
    (tmp_path / "runs" / "scan").mkdir(parents=True)
    (tmp_path / "runs" / "scan" / "active.json").write_text(json.dumps(
        {"module": "scan", "run_id": "missing-run", "frame": "Tasni Work Frame",
         "size_mm": [400.0, 300.0]}), encoding="utf-8")
    surface = active_scan_surface(root=tmp_path)
    assert surface["available"] is False and surface["center_mm"] is None
    assert "Re-insert the scan" in surface["note"]
    assert active_scan_surface(root=tmp_path / "empty") is None


def test_surface_fit_names_the_overhanging_edge():
    fits = surface_fit(scan_surface(), center_x_mm=200, center_y_mm=150,
                       outer_radius_mm=43)
    assert fits["inside"] is True and fits["minimum_margin_mm"] == 107.0
    # A wall wider than the short axis overhangs both Y edges, not the X ones.
    over = surface_fit(scan_surface(), center_x_mm=200, center_y_mm=150,
                       outer_radius_mm=170)
    assert over["inside"] is False
    assert over["margins_mm"]["y_min"] == -20.0 and over["margins_mm"]["y_max"] == -20.0
    assert over["margins_mm"]["x_min"] == 30.0


def test_centring_handles_a_frame_whose_y_points_off_the_rectangle(tmp_path):
    """Real cells hit this: frame +Y is Z x X, which can point away from the surface.

    The rectangle then spans negative Y, so half the recorded extent is the wrong
    centre and only the corners give the sign.
    """
    write_active_scan(tmp_path, corners=[[0, 0, 0], [400, 0, 0],
                                         [400, -300, 0], [0, -300, 0]])
    surface = active_scan_surface(root=tmp_path)
    assert surface["center_mm"] == [200.0, -150.0]
    fit = surface_fit(surface, center_x_mm=200, center_y_mm=-150, outer_radius_mm=43)
    assert fit["inside"] is True and fit["minimum_margin_mm"] == 107.0


def test_manual_placement_passes_with_an_advisory_when_a_surface_is_applied():
    check = surface_check(setup(), recipe(), scan_surface())
    assert check["ok"] is True and check["placement"] == "manual"
    assert "placed manually" in check["advisory"]
    assert surface_check(setup(), recipe(), None)["advisory"] == ""


def test_surface_placed_plan_is_invalidated_by_a_rescan():
    placed = setup(work_frame="Tasni Work Frame", center_x_mm=200, center_y_mm=150,
                   scan_run_id="20260812-101500")
    assert surface_check(placed, recipe(), scan_surface())["ok"] is True
    rescanned = surface_check(placed, recipe(), scan_surface(run_id="20260812-180000"))
    assert rescanned["ok"] is False and "re-scanned" in rescanned["problem"]
    removed = surface_check(placed, recipe(), None)
    assert removed["ok"] is False and "no scan is applied" in removed["problem"]
    moved = surface_check(setup(work_frame="World", scan_run_id="20260812-101500"),
                          recipe(), scan_surface())
    assert moved["ok"] is False and "is not the scanned surface frame" in moved["problem"]


def test_preflight_rejects_a_wall_that_overhangs_the_measured_surface():
    placed = setup(work_frame="Tasni Work Frame", center_x_mm=200, center_y_mm=150,
                   scan_run_id="20260812-101500")
    plan = generate_cylinder_plan(recipe(radius_mm=170, bead_diameter_mm=6), placed)
    result = geometry_preflight(plan, surface=scan_surface())
    assert result["all_ok"] is False
    assert all(layer["ok"] for layer in result["layers"])   # geometry itself is fine
    assert "overhangs the measured surface" in result["surface"]["problem"]
    assert "y_min by 23.0 mm" in result["surface"]["problem"]


def test_surface_placement_is_fingerprinted():
    placed = setup(work_frame="Tasni Work Frame", scan_run_id="20260812-101500")
    rescanned = setup(work_frame="Tasni Work Frame", scan_run_id="20260812-180000")
    assert (generate_cylinder_plan(recipe(), placed).fingerprint
            != generate_cylinder_plan(recipe(), rescanned).fingerprint)


def test_api_centres_on_the_scanned_surface_and_gates_when_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(runs, "REPO_ROOT", tmp_path)
    client = TestClient(create_app(AppConfig()))
    empty = client.get("/api/modules/extrusion/scan-surface").json()
    assert empty["applied"] is False and empty["available"] is False
    assert client.post("/api/modules/extrusion/center-on-surface",
                       json={"radius_mm": 40, "bead_diameter_mm": 6}).status_code == 409

    write_active_scan(tmp_path)
    applied = client.get("/api/modules/extrusion/scan-surface").json()
    assert applied["applied"] is True and applied["available"] is True
    centred = client.post("/api/modules/extrusion/center-on-surface",
                          json={"radius_mm": 40, "bead_diameter_mm": 6}).json()
    assert centred["setup"] == {"work_frame": "Tasni Work Frame", "center_x_mm": 200.0,
                                "center_y_mm": 150.0, "build_plane_z_mm": 0.0,
                                "scan_run_id": "20260812-101500"}
    assert centred["fit"]["inside"] is True

    # That placement flows through generate -> preflight as surface-placed.
    payload = generate_payload()
    payload["setup"].update(centred["setup"])
    plan = client.post("/api/modules/extrusion/generate", json=payload).json()
    preflight = client.post("/api/modules/extrusion/preflight",
                            json={"fingerprint": plan["fingerprint"]}).json()
    assert preflight["surface"]["placement"] == "scan_surface"
    assert preflight["surface"]["ok"] is True

    # Re-scanning the table invalidates the placement, not just the recipe.
    write_active_scan(tmp_path, run_id="20260812-180000")
    stale = client.post("/api/modules/extrusion/preflight",
                        json={"fingerprint": plan["fingerprint"]}).json()
    assert stale["all_ok"] is False and stale["surface"]["ok"] is False


def test_api_registers_module_and_invalidates_preflight_on_regeneration():
    client = TestClient(create_app(AppConfig()))
    modules = client.get("/api/modules").json()["modules"]
    assert "extrusion" in {module["id"] for module in modules}
    payload = generate_payload()
    first = client.post("/api/modules/extrusion/generate", json=payload).json()
    ok = client.post("/api/modules/extrusion/preflight",
                     json={"fingerprint": first["fingerprint"]})
    assert ok.status_code == 200 and ok.json()["all_ok"]
    payload["recipe"]["radius_mm"] = 41
    client.post("/api/modules/extrusion/generate", json=payload)
    stale = client.post("/api/modules/extrusion/preflight",
                        json={"fingerprint": first["fingerprint"]})
    assert stale.status_code == 409


def test_api_live_print_is_fail_closed():
    client = TestClient(create_app(AppConfig()))
    plan = client.post("/api/modules/extrusion/generate", json=generate_payload()).json()
    response = client.post("/api/modules/extrusion/print",
                           json={"fingerprint": plan["fingerprint"]})
    assert response.status_code == 409
    assert "dry run" in response.json()["detail"]


# -- automatic inspection pose ---------------------------------------------

CAMERA = AppConfig().camera            # D435i colour, 1280x720, fx=fy~908 px


def framing(width_mm, height_mm, **updates):
    config = ExtrusionConfig(**updates)
    return framing_standoff(width_mm=width_mm, height_mm=height_mm, K=CAMERA.K,
                            size_px=CAMERA.size, frame_margin=config.inspection_frame_margin,
                            near_mm=config.inspection_min_mm, far_mm=config.inspection_max_mm)


def test_a3_sheet_reproduces_the_distance_measured_on_the_cell():
    """The operator measured an A3 sheet filling the frame at ~380-400 mm.

    Nothing is hardcoded to that: it falls out of the camera's own intrinsics via
    the same pinhole rule the scan planner uses, with the SHORT side binding
    (297 mm needs 375 mm; the 420 mm side would only need 298 mm). This test is
    the anchor that keeps the model honest against the real cell.
    """
    bare = framing(420, 297, inspection_frame_margin=1.0)
    assert 374.0 < bare["d_fit_mm"] < 376.0
    assert bare["binding_axis"] == "height"
    assert 385.0 < bare["d_fit_mm"] * 1.05 < 400.0        # inside the measured band
    assert bare["clamped_to"] is None and bare["fits"] is True


def test_small_cylinder_is_held_at_the_accurate_near_limit_not_pushed_closer():
    """A 95 mm cylinder wants 134 mm to fill the frame — inside the D435i's blind
    zone. Framing is not the binding constraint at cylinder scale; the depth band
    is, so the standoff clamps up and the UI is told the object stays small."""
    fit = framing(95, 95)
    assert 130.0 < fit["d_fit_mm"] < 140.0
    assert fit["standoff_mm"] == 300.0 and fit["clamped_to"] == "near"
    assert fit["fits"] is True
    assert 0.38 < fit["fill_fraction"]["height"] < 0.42   # ~40% of frame height
    assert any("near" in warning for warning in fit["warnings"])


def test_object_too_big_for_the_accurate_band_is_refused_not_backed_away_from():
    fit = framing(900, 900)
    assert fit["fits"] is False and fit["clamped_to"] == "far"
    assert fit["standoff_mm"] == 800.0
    assert any("accurate depth band" in warning for warning in fit["warnings"])


def test_cylinder_diameter_is_the_outside_of_the_deposited_wall():
    assert cylinder_diameter_mm(recipe(radius_mm=40, bead_diameter_mm=6)) == 86.0


def test_aim_point_tracks_the_top_of_the_layer_just_printed():
    made = recipe(bead_diameter_mm=6, layer_height_mm=5)
    placed = setup(center_x_mm=10, center_y_mm=-5, build_plane_z_mm=12)
    first = aim_point_mm(made, placed, 1)
    second = aim_point_mm(made, placed, 2)
    # centre-line = build + bead/2 + i*height; the surface measured is a bead-radius
    # above that, so layer 1's top is one full bead above the build plane.
    assert first.tolist() == [10.0, -5.0, 18.0]
    assert second[2] - first[2] == 5.0


def test_the_cylinder_axis_is_the_optical_axis_for_every_candidate():
    """'Centred in frame' is a geometric guarantee, not a tolerance: the aim point
    sits on the camera's +Z axis at exactly the standoff, so it projects onto the
    principal point whatever roll/tilt the reachability search ends up choosing."""
    aim = np.array([10.0, -5.0, 18.0])
    for candidate in pose_candidates(aim, 300.0, ExtrusionConfig()):
        T = candidate["T"]
        in_camera = np.linalg.inv(T) @ np.append(aim, 1.0)
        assert np.allclose(in_camera[:3], [0.0, 0.0, 300.0], atol=1e-9)
        assert np.isclose(np.linalg.norm(T[:3, 3] - aim), 300.0)


def test_the_first_candidate_looks_straight_down_because_incidence_costs_most():
    """Cell characterization (2026-08-13): 29 deg of incidence multiplies plane RMS
    by 11, while tripling the distance only triples it. So tilt is a last resort
    and the search must start fronto-parallel."""
    candidates = pose_candidates(np.array([0.0, 0.0, 0.0]), 300.0, ExtrusionConfig())
    first = candidates[0]
    assert first["tilt_deg"] == 0.0 and first["roll_deg"] == 0.0
    assert np.allclose(first["T"][:3, 3], [0.0, 0.0, 300.0])
    assert np.allclose(first["T"][:3, 2], [0.0, 0.0, -1.0])    # camera +Z looks down
    assert max(c["tilt_deg"] for c in candidates) <= 10.0
    assert len({(c["tilt_deg"], c["azimuth_deg"], c["roll_deg"]) for c in candidates}) == len(candidates)


def test_inspection_plan_lifts_the_camera_one_layer_height_per_layer():
    made = recipe(layer_count=3, layer_height_mm=5, bead_diameter_mm=6)
    result = inspection_plan(made, setup(), K=CAMERA.K, size_px=CAMERA.size,
                             config=ExtrusionConfig())
    heights = [layer["camera_z_mm"] for layer in result["layers"]]
    assert len(heights) == 3
    assert np.allclose(np.diff(heights), 5.0)
    assert result["standoff_mm"] == 300.0
    assert heights[0] == result["layers"][0]["top_z_mm"] + 300.0


def test_auto_mode_allows_an_empty_inspection_target_but_manual_still_requires_one():
    assert setup(inspection_auto=True, inspection_target="").inspection_auto is True
    try:
        setup(inspection_target="")
    except ValueError as exc:
        assert "inspection_target" in str(exc)
    else:
        raise AssertionError("manual inspection must still name a taught target")


def test_switching_to_the_automatic_pose_changes_the_fingerprint():
    manual = generate_cylinder_plan(recipe(), setup())
    auto = generate_cylinder_plan(recipe(), setup(inspection_auto=True))
    assert manual.fingerprint != auto.fingerprint


def test_station_requirements_stops_demanding_a_taught_target_in_auto_mode():
    class Station:
        def item_exists_as(self, name, kind): return bool(name)
        def program_instructions(self, name):
            return (["Set IO_508=1", "Set IO_601=1"] if name == "AirOn"
                    else ["Set IO_508=0", "Set IO_601=0"])

    config = ExtrusionConfig()
    auto = generate_cylinder_plan(recipe(), setup(inspection_auto=True, inspection_target=""))
    report = station_requirements(Station(), auto, config)
    assert report["ready"] is True
    assert not any(item["role"] == "inspection_target" for item in report["items"])
    manual = generate_cylinder_plan(recipe(), setup())
    assert any(item["role"] == "inspection_target"
               for item in station_requirements(Station(), manual, config)["items"])


def test_preflight_reports_the_derived_inspection_geometry():
    plan = generate_cylinder_plan(recipe(), setup(inspection_auto=True, inspection_target=""))
    result = geometry_preflight(plan, camera=CAMERA, config=ExtrusionConfig())
    assert result["inspection"]["auto"] is True
    assert result["inspection"]["standoff_mm"] == 300.0
    assert result["all_ok"] is True


def test_api_previews_the_automatic_inspection_pose_without_touching_the_station():
    client = TestClient(create_app(AppConfig()))
    payload = generate_payload()
    payload["setup"]["inspection_auto"] = True
    payload["setup"]["inspection_target"] = ""
    plan = client.post("/api/modules/extrusion/generate", json=payload).json()
    preview = client.post("/api/modules/extrusion/inspection-pose",
                          json={"fingerprint": plan["fingerprint"]})
    assert preview.status_code == 200
    body = preview.json()
    assert body["standoff_mm"] == 300.0
    assert body["object_diameter_mm"] == 86.0
    assert len(body["layers"]) == 3
    assert body["framing"]["clamped_to"] == "near"
    assert "T" not in body["layers"][0]["candidates"][0]      # JSON-safe descriptors only


def test_api_reset_invalidates_generated_coordinates_without_a_station():
    client = TestClient(create_app(AppConfig()))
    plan = client.post("/api/modules/extrusion/generate", json=generate_payload()).json()
    reset = client.post("/api/modules/extrusion/reset")
    assert reset.status_code == 200 and reset.json()["removed"] == []
    stale = client.post("/api/modules/extrusion/preflight",
                        json={"fingerprint": plan["fingerprint"]})
    assert stale.status_code == 409


def test_seed_first_reordering_moves_last_winner_to_front():
    from tasni.modules.extrusion.inspection import order_candidates_seed_first
    candidates = [{"tilt_deg": 0.0, "azimuth_deg": 0.0, "roll_deg": r}
                  for r in (0.0, 90.0, 180.0)]
    seed = {"tilt_deg": 0.0, "azimuth_deg": 0.0, "roll_deg": 90.0}
    out = order_candidates_seed_first(candidates, seed)
    assert [c["roll_deg"] for c in out] == [90.0, 0.0, 180.0]
    assert order_candidates_seed_first(candidates, None) == candidates
    unknown = {"tilt_deg": 5.0, "azimuth_deg": 0.0, "roll_deg": 45.0}
    assert order_candidates_seed_first(candidates, unknown) == candidates
