"""Cylinder planning, comparison, correction, archive, and API foundations."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from tasni.core.config import AppConfig, ExtrusionConfig
from tasni.modules.extrusion.archive import ExtrusionArchive
from tasni.modules.extrusion.comparison import compare_circle, corrected_circle
from tasni.modules.extrusion.models import CylinderRecipe, LayerManifest
from tasni.modules.extrusion.service import geometry_preflight
from tasni.modules.extrusion.toolpath import generate_cylinder_plan, points_array
from tasni.webapp.server import create_app
from tools.setup_extrusion_station import LICENSED_ISOLATED_ARGS, expected_instructions


def recipe(**updates) -> CylinderRecipe:
    values = dict(radius_mm=40, layer_count=3, layer_height_mm=5,
                  bead_diameter_mm=6, robot_speed_mm_s=75,
                  extrusion_rate_pct=30, points_per_circle=72)
    values.update(updates)
    return CylinderRecipe(**values)


def test_config_keeps_verified_mapping_separate_from_hardware_approval():
    config = ExtrusionConfig()
    assert config.valve_outputs == ["IO_508", "IO_601"]
    assert config.valve_mapping_verified is True
    assert config.hardware_io_test_approved is False
    assert expected_instructions(config.valve_outputs, 1) == ["Set IO_508=1", "Set IO_601=1"]
    assert expected_instructions(config.valve_outputs, 0) == ["Set IO_508=0", "Set IO_601=0"]
    assert "-NEWINSTANCE" in LICENSED_ISOLATED_ARGS
    assert "-NOUI" in LICENSED_ISOLATED_ARGS
    assert "-SKIPINI" not in LICENSED_ISOLATED_ARGS


def test_plan_is_closed_layered_and_fingerprinted():
    plan = generate_cylinder_plan(recipe())
    assert len(plan.layers) == 3
    assert [layer.nominal_z_mm for layer in plan.layers] == [3, 8, 13]
    for layer in plan.layers:
        pts = points_array(layer)
        assert pts.shape == (73, 3)
        np.testing.assert_allclose(pts[0], pts[-1], atol=0)
    assert math.isclose(plan.total_path_length_mm, 3 * 2 * math.pi * 40)
    changed = generate_cylinder_plan(recipe(radius_mm=41))
    assert changed.fingerprint != plan.fingerprint


def test_geometry_preflight_does_not_claim_robodk_dry_run():
    result = geometry_preflight(generate_cylinder_plan(recipe()))
    assert result["all_ok"] is True
    assert result["dry_run_passed"] is False
    assert all(event["physical_output_blocked"] for event in result["simulated_valve_events"])


def test_circle_metrics_and_bounded_correction():
    theta = np.linspace(0, 2 * math.pi, 181)
    measured = np.column_stack((42 * np.cos(theta) + 3,
                                42 * np.sin(theta) - 4,
                                np.full_like(theta, 8)))
    metrics = compare_circle(measured, 40)
    assert metrics.valid
    assert math.isclose(metrics.measured_radius_mm, 42, abs_tol=1e-9)
    assert math.isclose(metrics.rms_mm, 2, abs_tol=1e-9)
    corrected = corrected_circle(measured, 40, 8, point_count=72,
                                 gain=1, smoothing_points=9, max_correction_mm=5)
    radii = np.linalg.norm(corrected[:, :2], axis=1)
    np.testing.assert_allclose(radii, 38, atol=1e-8)
    np.testing.assert_allclose(corrected[0], corrected[-1])


def test_archive_writes_reprocessable_layer(tmp_path: Path):
    plan = generate_cylinder_plan(recipe(layer_count=1))
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


def test_api_registers_module_and_invalidates_preflight_on_regeneration():
    client = TestClient(create_app(AppConfig()))
    modules = client.get("/api/modules").json()["modules"]
    assert "extrusion" in {module["id"] for module in modules}
    payload = recipe().model_dump(mode="json")
    first = client.post("/api/modules/extrusion/generate", json=payload).json()
    ok = client.post("/api/modules/extrusion/preflight",
                     json={"fingerprint": first["fingerprint"]})
    assert ok.status_code == 200 and ok.json()["all_ok"]
    payload["radius_mm"] = 41
    client.post("/api/modules/extrusion/generate", json=payload)
    stale = client.post("/api/modules/extrusion/preflight",
                        json={"fingerprint": first["fingerprint"]})
    assert stale.status_code == 409


def test_api_live_print_is_fail_closed():
    client = TestClient(create_app(AppConfig()))
    plan = client.post("/api/modules/extrusion/generate",
                       json=recipe().model_dump(mode="json")).json()
    response = client.post("/api/modules/extrusion/print",
                           json={"fingerprint": plan["fingerprint"]})
    assert response.status_code == 409
    assert "dry run" in response.json()["detail"]
