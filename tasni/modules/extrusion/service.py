"""Pure planning/preflight services for the cylinder workflow."""
from __future__ import annotations

import math
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ...core import runs
from ...core.config import ExtrusionConfig
from ...core.jobrunner import JobContext
from ...core.logging import REPO_ROOT, new_run_dir
from ...core.rdk_io import RdkIO
from ..calibration.service import _camera_hold, ensure_real_robot_link
from .archive import ExtrusionArchive
from .models import CylinderPlan, CylinderRecipe, CylinderSetup, LayerManifest
from .processing import process_observation
from .toolpath import generate_cylinder_plan, points_array


def geometry_preflight(plan: CylinderPlan) -> dict:
    """Validate the generated geometry without claiming a RoboDK dry-run pass."""
    layers: list[dict] = []
    all_ok = True
    for layer in plan.layers:
        pts = points_array(layer)
        gaps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        expected = 2.0 * math.pi * plan.recipe.radius_mm
        actual = float(gaps.sum())
        closed = bool(np.allclose(pts[0], pts[-1], atol=1e-9))
        finite = bool(np.isfinite(pts).all())
        length_ok = abs(actual - expected) <= max(0.1, expected * 0.001)
        ok = closed and finite and length_ok and len(pts) == plan.recipe.points_per_circle + 1
        all_ok &= ok
        layers.append({
            "layer_index": layer.layer_index, "point_count": len(pts),
            "closed": closed, "finite": finite, "length_mm": actual,
            "maximum_segment_mm": float(gaps.max()), "ok": ok,
        })
    return {
        "kind": "geometry_preflight", "fingerprint": plan.fingerprint,
        "all_ok": all_ok, "layers": layers,
        "dry_run_passed": False,
        "note": "Geometry is valid; RoboDK reachability/collision/program execution is still required.",
        "simulated_valve_events": [
            {"event": "AirOn", "physical_output_blocked": True},
            {"event": "AirOff", "physical_output_blocked": True},
        ],
    }


def station_requirements(rdk: RdkIO, plan: CylinderPlan, config) -> dict:
    """Validate the exact selected station items and fail-safe valve programs."""
    selected = plan.setup
    checks = [
        ("print_tool", selected.print_tool, "tool"),
        ("work_frame", selected.work_frame, "frame"),
        ("inspection_tool", selected.inspection_tool, "tool"),
        ("inspection_target", selected.inspection_target, "target"),
        ("air_on_program", config.air_on_program, "program"),
        ("air_off_program", config.air_off_program, "program"),
    ]
    items = [{"role": role, "name": name, "type": kind,
              "present": rdk.item_exists_as(name, kind)}
             for role, name, kind in checks]
    expected_on = [f"Set {output}={config.valve_active_value}"
                   for output in config.valve_outputs]
    expected_off = [f"Set {output}={config.valve_inactive_value}"
                    for output in config.valve_outputs]
    actual_on = (rdk.program_instructions(config.air_on_program)
                 if rdk.item_exists_as(config.air_on_program, "program") else [])
    actual_off = (rdk.program_instructions(config.air_off_program)
                  if rdk.item_exists_as(config.air_off_program, "program") else [])
    mapping_ok = actual_on == expected_on and actual_off == expected_off
    if not mapping_ok:
        items.append({"role": "valve_instruction_mapping", "name": "AirOn/AirOff",
                      "type": "program instructions", "present": False,
                      "expected": {"on": expected_on, "off": expected_off},
                      "actual": {"on": actual_on, "off": actual_off}})
    return {"ready": all(item["present"] for item in items), "items": items,
            "valve_mapping_verified": mapping_ok,
            "missing": [item for item in items if not item["present"]]}


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _program_name(plan: CylinderPlan, layer_index: int, mode: str) -> str:
    return f"TasniCylinder_{mode}_{plan.fingerprint[:10]}_L{layer_index:03d}"


def _wait_program(ctx: JobContext, rdk: RdkIO, name: str) -> None:
    while rdk.program_busy(name):
        if ctx.cancelled:
            rdk.stop_program(name)
            ctx.check_cancel()
        time.sleep(0.05)


def _require_program_valid(report: dict, layer_index: int) -> None:
    if report["percent_ok"] < 99.999:
        raise RuntimeError(
            f"layer {layer_index} RoboDK validation failed at "
            f"{report['percent_ok']:.1f}%: {report['problems'] or 'unspecified path problem'}")


class CylinderDryRunJob:
    """Execute the complete cylinder in RoboDK SIMULATE with mock-only valve calls."""

    def __init__(self, services, plan: CylinderPlan, *, on_pass=None):
        self.services = services
        self.plan = plan.model_copy(deep=True)
        self.on_pass = on_pass
        self.result: dict | None = None

    def __call__(self, ctx: JobContext) -> dict:
        rdk: RdkIO = self.services.rdk
        ecfg = self.services.config.extrusion
        required = station_requirements(rdk, self.plan, ecfg)
        if not required["ready"]:
            missing = ", ".join(f"{v['type']} {v['name']!r}" for v in required["missing"])
            raise RuntimeError("station is not ready: " + missing)
        prior_mode = rdk.current_run_mode()
        start_joints = rdk.current_joints()
        current_program = None
        artifacts: list[str] = []
        reports: list[dict] = []
        valve_events: list[dict] = []
        stamp = time.strftime("%Y%m%d-%H%M%S")
        run_dir = new_run_dir("extrusion-dry-run", stamp)
        try:
            rdk.apply_run_mode("simulate")
            mock_on, mock_off = rdk.ensure_mock_valve_programs(
                f"TasniDry_{self.plan.fingerprint[:8]}_")
            artifacts.extend((mock_on, mock_off))
            ctx.log("DRY_RUN: SIMULATE mode; physical valve/rate outputs blocked")
            total = len(self.plan.layers)
            for index, layer in enumerate(self.plan.layers, start=1):
                ctx.check_cancel()
                ctx.progress(index, total, f"dry-running layer {layer.layer_index}")
                name = _program_name(self.plan, layer.layer_index, "DRY")
                built = rdk.create_extrusion_layer_program(
                    name=name, points_xyz=points_array(layer),
                    orientation_rpy_deg=self.plan.setup.orientation_rpy_deg,
                    print_tool=self.plan.setup.print_tool,
                    work_frame=self.plan.setup.work_frame,
                    speed_mm_s=self.plan.recipe.robot_speed_mm_s,
                    approach_clearance_mm=self.plan.setup.approach_clearance_mm,
                    retreat_clearance_mm=self.plan.setup.retreat_clearance_mm,
                    air_on_program=mock_on, air_off_program=mock_off)
                artifacts.extend([built["program"], *built["targets"]])
                current_program = name
                validation = rdk.update_program(name, collisions=True)
                _require_program_valid(validation, layer.layer_index)
                valve_events.extend([
                    {"layer_index": layer.layer_index, "event": "AirOn",
                     "mode": "MOCK", "physical_output_blocked": True},
                    {"layer_index": layer.layer_index, "event": "AirOff",
                     "mode": "MOCK", "physical_output_blocked": True},
                ])
                started = rdk.start_program(name, real_robot=False)
                if started < 0:
                    raise RuntimeError(f"layer {layer.layer_index} simulation could not start")
                _wait_program(ctx, rdk, name)
                current_program = None
                inspection_name = name + "_Inspect"
                inspect = rdk.create_inspection_program(
                    name=inspection_name,
                    inspection_tool=self.plan.setup.inspection_tool,
                    inspection_target=self.plan.setup.inspection_target,
                    speed_mm_s=self.plan.recipe.robot_speed_mm_s)
                artifacts.append(inspect["program"])
                inspection_validation = rdk.update_program(inspection_name, collisions=True)
                _require_program_valid(inspection_validation, layer.layer_index)
                current_program = inspection_name
                started = rdk.start_program(inspection_name, real_robot=False)
                if started < 0:
                    raise RuntimeError(
                        f"layer {layer.layer_index} inspection simulation could not start")
                _wait_program(ctx, rdk, inspection_name)
                current_program = None
                reports.append({"layer_index": layer.layer_index,
                                "path": validation, "inspection": inspection_validation})
                ctx.log(f"layer {layer.layer_index}: path + inspection motion simulated; "
                        f"valve ON/OFF shown as mock events")
            rdk.move_j_joints(start_joints)
            report = {
                "kind": "cylinder_dry_run", "mode": "DRY_RUN",
                "fingerprint": self.plan.fingerprint, "all_ok": True,
                "returned_to_start": True, "physical_outputs_blocked": True,
                "layers": reports, "valve_events": valve_events,
                "setup": self.plan.setup.model_dump(mode="json"),
                "recipe": self.plan.recipe.model_dump(mode="json"),
                "run_dir": str(run_dir), "git_commit": _git_commit(),
            }
            (run_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            self.result = report
            if self.on_pass is not None:
                self.on_pass(self.plan.fingerprint)
            ctx.log("dry run PASS: complete path, collisions, valve mocks, inspection, return-to-start")
            return report
        finally:
            if current_program:
                try:
                    rdk.stop_program(current_program)
                except Exception:
                    pass
            try:
                rdk.move_j_joints(start_joints)
            except Exception:
                pass
            try:
                rdk.delete_items(list(dict.fromkeys(reversed(artifacts))))
            except Exception:
                pass
            rdk.set_run_mode_raw(prior_mode)


class CylinderPrintJob:
    """Print, capture one RGB-D frame, process, and archive each cylinder layer."""

    def __init__(self, services, plan: CylinderPlan):
        self.services = services
        self.plan = plan.model_copy(deep=True)
        self.result: dict | None = None
        self.corrected_reference_xyz: np.ndarray | None = None

    def __call__(self, ctx: JobContext) -> dict:
        services = self.services
        rdk: RdkIO = services.rdk
        ecfg = services.config.extrusion
        required = station_requirements(rdk, self.plan, ecfg)
        if not required["ready"]:
            missing = ", ".join(f"{v['type']} {v['name']!r}" for v in required["missing"])
            raise RuntimeError("station is not ready: " + missing)
        if not ecfg.hardware_io_test_approved:
            raise RuntimeError("hardware I/O test is not approved")
        if services.live.running:
            services.live.stop()
        applied = rdk.apply_run_mode("run_robot")
        if applied != "run_robot":
            raise RuntimeError("RoboDK refused RUN_ROBOT mode")
        ensure_real_robot_link(rdk, services.config.robodk, log=ctx.log)
        start_joints = rdk.current_joints()
        current_program = None
        artifacts: list[str] = []
        valve: list[dict] = []
        stamp = time.strftime("%Y%m%d-%H%M%S")
        trial_id = f"{stamp}-{self.plan.fingerprint[:8]}"
        archive = ExtrusionArchive(REPO_ROOT / "runs" / "extrusion")
        calibration = runs.read_active("calibration")
        provenance = {
            "git_commit": _git_commit(), "calibration": calibration,
            "camera_resolution": services.config.camera.resolution,
            "camera_intrinsics": {
                "K": np.asarray(services.config.camera.K, dtype=float).tolist(),
                "dist_coeffs": list(services.config.camera.dist_coeffs),
            },
            "processing_config": ecfg.model_dump(mode="json"),
        }
        trial_dir = archive.create_trial(trial_id, self.plan, provenance=provenance)
        nominal_plan = generate_cylinder_plan(self.plan.recipe, self.plan.setup)

        def valve_off(reason: str, *, required: bool = False) -> bool:
            event = {"timestamp": _utcnow(), "requested_state": "OFF",
                     "program": ecfg.air_off_program, "reason": reason,
                     "confirmed": None}
            valve.append(event)
            try:
                rdk.run_station_program(ecfg.air_off_program, real_robot=True)
                event["command_completed"] = True
                ctx.log(f"valve OFF: {reason}")
                return True
            except Exception as exc:
                event["command_completed"] = False
                event["fault"] = str(exc)
                ctx.log(f"WARNING: valve OFF command failed ({reason}): {exc}")
                if required:
                    raise RuntimeError(
                        f"cannot establish fail-safe valve OFF state ({reason}): {exc}") from exc
                return False

        summaries: list[dict] = []
        try:
            valve_off("job startup before motion", required=True)
            total = len(self.plan.layers)
            with _camera_hold(services, "extrusion-run"):
                for index, layer in enumerate(self.plan.layers, start=1):
                    ctx.check_cancel()
                    ctx.progress(index, total, f"printing layer {layer.layer_index}")
                    valve_off(f"before layer {layer.layer_index} approach", required=True)
                    name = _program_name(self.plan, layer.layer_index, "LIVE")
                    built = rdk.create_extrusion_layer_program(
                        name=name, points_xyz=points_array(layer),
                        orientation_rpy_deg=self.plan.setup.orientation_rpy_deg,
                        print_tool=self.plan.setup.print_tool,
                        work_frame=self.plan.setup.work_frame,
                        speed_mm_s=self.plan.recipe.robot_speed_mm_s,
                        approach_clearance_mm=self.plan.setup.approach_clearance_mm,
                        retreat_clearance_mm=self.plan.setup.retreat_clearance_mm,
                        air_on_program=ecfg.air_on_program,
                        air_off_program=ecfg.air_off_program)
                    artifacts.extend([built["program"], *built["targets"]])
                    validation = rdk.update_program(name, collisions=True)
                    _require_program_valid(validation, layer.layer_index)
                    current_program = name
                    scheduled_at = _utcnow()
                    valve.extend([
                        {"timestamp": scheduled_at, "layer_index": layer.layer_index,
                         "requested_state": "ON", "program": ecfg.air_on_program,
                         "source": "path-start program event", "confirmed": None},
                        {"timestamp": scheduled_at, "layer_index": layer.layer_index,
                         "requested_state": "OFF", "program": ecfg.air_off_program,
                         "source": "path-finish program event", "confirmed": None},
                    ])
                    started = rdk.start_program(name, real_robot=True)
                    if started < 0:
                        raise RuntimeError(f"layer {layer.layer_index} live program could not start")
                    _wait_program(ctx, rdk, name)
                    current_program = None
                    valve_off(f"layer {layer.layer_index} completion/inspection confirmation",
                              required=True)
                    inspection_name = name + "_Inspect"
                    inspect = rdk.create_inspection_program(
                        name=inspection_name,
                        inspection_tool=self.plan.setup.inspection_tool,
                        inspection_target=self.plan.setup.inspection_target,
                        speed_mm_s=self.plan.recipe.robot_speed_mm_s)
                    artifacts.append(inspect["program"])
                    inspection_validation = rdk.update_program(inspection_name, collisions=True)
                    _require_program_valid(inspection_validation, layer.layer_index)
                    current_program = inspection_name
                    started = rdk.start_program(inspection_name, real_robot=True)
                    if started < 0:
                        raise RuntimeError(
                            f"layer {layer.layer_index} inspection program could not start")
                    _wait_program(ctx, rdk, inspection_name)
                    current_program = None
                    time.sleep(ecfg.settle_s)
                    # Re-select the inspection TCP and chosen work frame in the API
                    # before reading the camera pose; the generated program's tool
                    # instructions do not update RdkIO's cached tool transform.
                    rdk.use_named_tool_frame(self.plan.setup.inspection_tool,
                                             self.plan.setup.work_frame)
                    T_work_camera = rdk.camera_pose_T()
                    frame = services.camera.grab(with_depth=True, timeout=ecfg.grab_timeout_s)
                    if frame.depth is None:
                        raise RuntimeError(f"layer {layer.layer_index}: RGB-D capture returned no depth")
                    ok, jpeg = __import__("cv2").imencode(".jpg", frame.color)
                    if ok:
                        ctx.frame(jpeg.tobytes())
                    nominal = points_array(nominal_plan.layers[layer.layer_index - 1])
                    commanded = points_array(layer)
                    base_manifest = dict(
                        trial_id=trial_id, layer_index=layer.layer_index,
                        recipe=self.plan.recipe, toolpath_fingerprint=self.plan.fingerprint,
                        color_file="color.png", depth_file="depth.npy",
                        provenance={**provenance, "work_frame": self.plan.setup.work_frame,
                                    "print_tool": self.plan.setup.print_tool,
                                    "inspection_tool": self.plan.setup.inspection_tool,
                                    "inspection_target": self.plan.setup.inspection_target,
                                    "T_work_camera": np.asarray(
                                        T_work_camera, dtype=float).tolist()},
                        valve_transitions=[v for v in valve if v.get("layer_index") in
                                           (None, layer.layer_index)])
                    try:
                        processed = process_observation(
                            color=frame.color, depth=frame.depth,
                            T_work_camera=T_work_camera, K=services.config.camera.K,
                            plan=self.plan, layer=layer, config=ecfg)
                    except Exception as exc:
                        manifest = LayerManifest(**base_manifest,
                                                 processing={"valid": False, "error": str(exc)},
                                                 warnings=[str(exc)])
                        archive.write_layer(manifest, nominal_xyz=nominal,
                                            commanded_xyz=commanded, color=frame.color,
                                            depth=frame.depth,
                                            report={"valid": False, "error": str(exc)})
                        raise RuntimeError(
                            f"layer {layer.layer_index} measurement invalid; raw RGB-D archived: {exc}") from exc
                    if processed.corrected_xyz is not None:
                        self.corrected_reference_xyz = processed.corrected_xyz.copy()
                    manifest = LayerManifest(
                        **base_manifest, measured_path_file="measured_path.json",
                        corrected_path_file=("corrected_path.json"
                                             if processed.corrected_xyz is not None else None),
                        pointcloud_file=("height-or-pointcloud.npy"
                                         if processed.filtered_xyz is not None else None),
                        metrics=processed.metrics, processing=processed.report,
                        warnings=processed.metrics.warnings)
                    layer_dir = archive.write_layer(
                        manifest, nominal_xyz=nominal, commanded_xyz=commanded,
                        measured_xyz=processed.measured_xyz,
                        corrected_xyz=processed.corrected_xyz,
                        pointcloud_xyz=processed.filtered_xyz,
                        color=frame.color, depth=frame.depth,
                        derived_images={"segmentation.png": processed.segmentation,
                                        "skeleton.png": processed.skeleton,
                                        "comparison.png": processed.comparison},
                        report={**processed.report,
                                "metrics": processed.metrics.model_dump(mode="json")})
                    summary = {"layer_index": layer.layer_index,
                               "metrics": processed.metrics.model_dump(mode="json"),
                               "run_dir": str(layer_dir),
                               "correction_calculated": processed.corrected_xyz is not None,
                               "correction_executed": False}
                    summaries.append(summary)
                    ctx.log(f"layer {layer.layer_index}: captured one RGB-D frame; "
                            f"RMS deviation {processed.metrics.rms_mm:.2f} mm; archived")
            valve_off("normal completion", required=True)
            rdk.move_j_joints(start_joints)
            result = {"kind": "cylinder_print", "mode": "LIVE_PRINT",
                      "fingerprint": self.plan.fingerprint, "trial_id": trial_id,
                      "trial_dir": str(trial_dir), "layers": summaries,
                      "correction_available": self.corrected_reference_xyz is not None,
                      "correction_executed": False}
            (trial_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            self.result = result
            return result
        finally:
            if current_program:
                try:
                    rdk.stop_program(current_program)
                except Exception:
                    pass
            safe_to_move = valve_off("job exit/cancellation/fault")
            if safe_to_move:
                try:
                    rdk.move_j_joints(start_joints)
                except Exception:
                    pass
            else:
                ctx.log("FAULT: valve OFF could not be established; return motion inhibited")
            try:
                rdk.delete_items(list(dict.fromkeys(reversed(artifacts))))
            except Exception:
                pass


def reprocess_saved_layer(root: str | Path, trial_id: str, layer_index: int) -> dict:
    """Rebuild only derived artifacts from one archived raw RGB-D observation."""
    archive = ExtrusionArchive(root)
    layer_dir = archive.layer_dir(trial_id, layer_index)
    trial_dir = layer_dir.parent
    trial = json.loads((trial_dir / "trial.json").read_text(encoding="utf-8"))
    manifest = LayerManifest.model_validate_json(
        (layer_dir / "manifest.json").read_text(encoding="utf-8"))
    provenance = manifest.provenance
    processing_payload = trial.get("provenance", {}).get("processing_config")
    intrinsics = trial.get("provenance", {}).get("camera_intrinsics", {})
    transform = provenance.get("T_work_camera")
    if not processing_payload or "K" not in intrinsics or transform is None:
        raise RuntimeError(
            "archive predates reproducible reprocessing provenance "
            "(processing config, intrinsics, or camera pose is missing)")
    color_path, depth_path = layer_dir / "color.png", layer_dir / "depth.npy"
    if not color_path.is_file() or not depth_path.is_file():
        raise RuntimeError("archived raw color/depth observation is incomplete")
    import cv2
    color = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
    if color is None:
        raise RuntimeError("archived color image could not be decoded")
    depth = np.load(depth_path, allow_pickle=False)
    recipe = CylinderRecipe.model_validate(trial["recipe"])
    setup = CylinderSetup.model_validate(trial["setup"])
    plan = generate_cylinder_plan(recipe, setup)
    if layer_index > len(plan.layers):
        raise RuntimeError("archived layer index exceeds the stored recipe")
    processed = process_observation(
        color=color, depth=depth,
        T_work_camera=np.asarray(transform, dtype=float),
        K=np.asarray(intrinsics["K"], dtype=float),
        plan=plan, layer=plan.layers[layer_index - 1],
        config=ExtrusionConfig.model_validate(processing_payload))
    reprocessed_at = _utcnow()
    report = {
        **processed.report,
        "metrics": processed.metrics.model_dump(mode="json"),
        "offline_reprocess": True,
        "reprocessed_at": reprocessed_at,
    }
    next_manifest = manifest.model_copy(update={
        "measured_path_file": "measured_path.json",
        "corrected_path_file": ("corrected_path.json"
                                if processed.corrected_xyz is not None else None),
        "pointcloud_file": ("height-or-pointcloud.npy"
                            if processed.filtered_xyz is not None else None),
        "metrics": processed.metrics,
        "processing": report,
        "warnings": processed.metrics.warnings,
        "provenance": {**provenance, "last_reprocessed_at": reprocessed_at},
    })
    archive.rewrite_processing(
        next_manifest, measured_xyz=processed.measured_xyz,
        corrected_xyz=processed.corrected_xyz,
        pointcloud_xyz=processed.filtered_xyz,
        derived_images={"segmentation.png": processed.segmentation,
                        "skeleton.png": processed.skeleton,
                        "comparison.png": processed.comparison},
        report=report)
    return {
        "trial_id": trial_id, "layer_index": layer_index,
        "reprocessed_at": reprocessed_at,
        "metrics": processed.metrics.model_dump(mode="json"),
        "run_dir": str(layer_dir),
    }
