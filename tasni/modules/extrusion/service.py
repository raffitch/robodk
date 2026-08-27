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
from ...core.build_info import build_info, staleness_warning
from ...core.config import ExtrusionConfig
from ...core.jobrunner import JobContext
from ...core.logging import REPO_ROOT, new_run_dir
from ...core.rdk_io import RdkIO
from ..calibration.service import _camera_hold, ensure_real_robot_link
from .archive import ExtrusionArchive
from .inspection import (aim_point_mm, cylinder_diameter_mm, framing_standoff,
                         inspection_plan, order_candidates_seed_first,
                         pose_candidates)
from .models import CylinderPlan, CylinderRecipe, CylinderSetup, LayerManifest
from .processing import process_observation
from .surface import surface_check
from .toolpath import generate_cylinder_plan, points_array


def geometry_preflight(plan: CylinderPlan, *, surface: dict | None = None,
                       camera=None, config=None) -> dict:
    """Validate the generated geometry without claiming a RoboDK dry-run pass.

    ``surface`` is the currently applied scan surface (``None`` = none applied). A
    surface-placed plan fails here if that surface changed or if the wall overhangs
    it, before any robot motion is offered. ``camera``/``config`` add the derived
    inspection geometry (pure — no station), so a cylinder too big to frame within
    the accurate depth band is caught here rather than mid-dry-run.
    """
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
    placement = surface_check(plan.setup, plan.recipe, surface)
    all_ok &= bool(placement["ok"])
    if camera is None or config is None:
        inspection = {"auto": bool(plan.setup.inspection_auto), "checked": False}
    else:
        inspection = {**inspection_plan(plan.recipe, plan.setup, K=camera.K,
                                        size_px=camera.size, config=config),
                      "checked": True}
        if plan.setup.inspection_auto:
            all_ok &= bool(inspection["ok"])
    return {
        "kind": "geometry_preflight", "fingerprint": plan.fingerprint,
        "all_ok": all_ok, "layers": layers, "surface": placement,
        "inspection": inspection, "dry_run_passed": False,
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
        ("air_on_program", config.air_on_program, "program"),
        ("air_off_program", config.air_off_program, "program"),
    ]
    if not selected.inspection_auto:
        # In automatic mode there is nothing taught to check: the target is derived
        # per layer and created (then collision-validated) during the run.
        checks.insert(3, ("inspection_target", selected.inspection_target, "target"))
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
    """The CHECKED-OUT commit. Kept for continuity in archived reports.

    This is not necessarily the code that ran: the app caches imported modules,
    so a report could name a commit the process never loaded. ``build_info()``
    records the running build, which is the value to trust.
    """
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _warn_if_stale(ctx) -> None:
    """Say so, loudly and early, when the process is running stale code."""
    warning = staleness_warning()
    if warning:
        ctx.log("STALE CODE: " + warning)


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
    # A cancellation can arrive after the final busy poll. Do not let the
    # caller continue into inspection, capture, or the next layer in that race.
    ctx.check_cancel()


def _program_valid(report: dict) -> bool:
    return report["percent_ok"] >= 99.999


def _require_program_valid(report: dict, layer_index: int) -> None:
    if not _program_valid(report):
        raise RuntimeError(
            f"layer {layer_index} RoboDK validation failed at "
            f"{report['percent_ok']:.1f}%: {report['problems'] or 'unspecified path problem'}")


def _build_inspection_move(rdk: RdkIO, plan: CylinderPlan, layer, *,
                           inspection_name: str, config, camera,
                           seed_pose: dict | None = None,
                           collisions: bool = True) -> dict:
    """Create the inspection program for one layer and return its validation.

    Manual mode moves to the taught target, exactly as before. Automatic mode
    derives the viewpoint from this layer's own geometry (see
    ``modules/extrusion/inspection.py``) and walks the ordered candidate list —
    fronto-parallel first — accepting the first that both has an IK solution and
    passes RoboDK's collision-enabled validation. That validation is the same
    authoritative gate the taught path already used, so nothing new decides
    safety here; the candidates only decide which viewpoint gets offered to it.

    Fails loudly with every rejection when none survives. Backing off, tilting
    past the configured cone, or dropping collision checking to obtain a pass are
    all deliberately absent: straight down at ~300 mm over a fresh print is the
    tightest clearance in this workflow, and the print tool shares the flange with
    the camera.
    """
    speed_mm_s = plan.recipe.travel_speed_mm_s
    if not plan.setup.inspection_auto:
        created = rdk.create_inspection_program(
            name=inspection_name, inspection_tool=plan.setup.inspection_tool,
            inspection_target=plan.setup.inspection_target, speed_mm_s=speed_mm_s)
        validation = rdk.update_program(inspection_name, collisions=collisions)
        _require_program_valid(validation, layer.layer_index)
        return {"artifacts": [created["program"]], "validation": validation,
                "target": plan.setup.inspection_target, "pose": None}

    framing = framing_standoff(
        width_mm=cylinder_diameter_mm(plan.recipe),
        height_mm=cylinder_diameter_mm(plan.recipe), K=camera.K, size_px=camera.size,
        frame_margin=config.inspection_frame_margin,
        near_mm=config.inspection_min_mm, far_mm=config.inspection_max_mm)
    if not framing["fits"]:
        raise RuntimeError("; ".join(framing["warnings"]))
    aim = aim_point_mm(plan.recipe, plan.setup, layer.layer_index)
    target_name = inspection_name + "_Target"
    rejected: list[dict] = []
    candidates = order_candidates_seed_first(
        pose_candidates(aim, framing["standoff_mm"], config), seed_pose)
    for candidate in candidates:
        descriptor = {k: v for k, v in candidate.items() if k != "T"}
        made = rdk.create_inspection_target(
            name=target_name, T=candidate["T"],
            inspection_tool=plan.setup.inspection_tool,
            work_frame=plan.setup.work_frame)
        if not made["created"]:
            rejected.append({**descriptor, "reason": made["reason"]})
            continue
        created = rdk.create_inspection_program(
            name=inspection_name, inspection_tool=plan.setup.inspection_tool,
            inspection_target=target_name, speed_mm_s=speed_mm_s)
        validation = rdk.update_program(inspection_name, collisions=collisions)
        if _program_valid(validation):
            return {
                "artifacts": [target_name, created["program"]],
                "validation": validation, "target": target_name,
                "pose": {**descriptor, "standoff_mm": framing["standoff_mm"],
                         "aim_mm": [float(v) for v in aim],
                         "clamped_to": framing["clamped_to"],
                         "fill_fraction": framing["fill_fraction"],
                         "rejected": rejected},
            }
        rejected.append({**descriptor,
                         "reason": (validation["problems"]
                                    or f"validated at only {validation['percent_ok']:.1f}%")})
    tried = ", ".join(f"tilt {r['tilt_deg']:.0f}/azimuth {r['azimuth_deg']:.0f}/"
                      f"roll {r['roll_deg']:.0f} deg: {r['reason']}" for r in rejected)
    qualification = "reachable, collision-free" if collisions else "reachable, feasible"
    raise RuntimeError(
        f"layer {layer.layer_index}: no {qualification} inspection pose at "
        f"{framing['standoff_mm']:.0f} mm above "
        f"[{aim[0]:.1f}, {aim[1]:.1f}, {aim[2]:.1f}] in {plan.setup.work_frame!r}. "
        f"Tried {len(rejected)} viewpoint(s) — {tried}")


class CylinderDryRunJob:
    """Execute a cylinder simulation with mock-only valve calls.

    ``check_collisions=False`` is an explicitly visual preview. The caller decides
    whether its selected-layer coverage is sufficient to unlock a physical run; the
    report always records exactly which layers were actually simulated.
    """

    def __init__(self, services, plan: CylinderPlan, *, on_pass=None,
                 on_preview_pass=None, check_collisions: bool = True,
                 layer_indices: list[int] | None = None,
                 approve_full_plan: bool = False):
        self.services = services
        self.plan = plan.model_copy(deep=True)
        self.on_pass = on_pass
        self.on_preview_pass = on_preview_pass
        self.check_collisions = bool(check_collisions)
        self.approve_full_plan = bool(approve_full_plan)
        available = {layer.layer_index for layer in self.plan.layers}
        requested = (list(layer_indices) if layer_indices is not None
                     else sorted(available))
        if not requested:
            raise ValueError("select at least one layer to simulate")
        invalid = sorted(set(requested) - available)
        if invalid:
            raise ValueError(f"layer indices are outside this plan: {invalid}")
        self.layer_indices = sorted(set(requested))
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
        mock_artifacts: list[str] = []
        path_artifacts: list[str] = []
        # Branch-locked waypoint targets are bulk, not evidence: they are removed
        # even when a run fails, so a failure leaves only the curve, project and
        # program to inspect instead of dozens of targets to hand-delete.
        path_targets: list[str] = []
        completed = False
        reports: list[dict] = []
        valve_events: list[dict] = []
        stamp = time.strftime("%Y%m%d-%H%M%S")
        run_kind = ("extrusion-dry-run" if self.check_collisions
                    else "extrusion-quick-simulation")
        run_dir = new_run_dir(run_kind, stamp)
        collision_map_optimization = None
        prior_simulation_speed = None
        try:
            rdk.apply_run_mode("simulate")
            # Program.Update(COLLISION_ON/OFF) below is the authoritative validation.
            # RoboDK's global collision toolbar is a separate per-animation check; if
            # left on it repeats expensive dense-mesh work during playback. Keep it off
            # for both modes so checked runs validate once, not twice.
            rdk.set_collision_checking(False)
            if not self.check_collisions:
                prior_simulation_speed = rdk.simulation_speed()
                rdk.set_simulation_speed(ecfg.quick_simulation_speed_ratio)
            if self.check_collisions:
                ignored = list(ecfg.collision_visual_ignore_objects)
                proxy = ecfg.collision_surface_proxy_object
                if ignored and rdk.item_exists_as(proxy, "object"):
                    collision_map_optimization = rdk.disable_object_collision_pairs(ignored)
                    collision_map_optimization["surface_proxy"] = proxy
                    if collision_map_optimization["objects"]:
                        ctx.log(
                            "collision validation: excluded redundant visual mesh(es) "
                            + ", ".join(collision_map_optimization["objects"])
                            + f"; {proxy} and all other station/robot/tool checks "
                              "remain active")
                elif ignored:
                    collision_map_optimization = {
                        "objects": [], "pairs_disabled": 0, "pairs_failed": 0,
                        "surface_proxy": proxy, "skipped": True,
                        "reason": "collision surface proxy is absent",
                    }
                    ctx.log(
                        f"collision optimization skipped: required proxy {proxy!r} "
                        "is absent; all geometry remains collision-active")
            mock_on, mock_off = rdk.ensure_mock_valve_programs(
                f"Tasni{'Dry' if self.check_collisions else 'Quick'}_"
                f"{self.plan.fingerprint[:8]}_")
            mock_artifacts.extend((mock_on, mock_off))
            if self.check_collisions:
                ctx.log("DRY_RUN: SIMULATE mode; collision validation ON; "
                        "physical valve/rate outputs blocked")
            else:
                ctx.log("QUICK_SIMULATION: collision validation OFF; advisory visual "
                        "preview only; physical valve/rate outputs blocked")
            _warn_if_stale(ctx)
            selected_layers = [layer for layer in self.plan.layers
                               if layer.layer_index in self.layer_indices]
            total = len(selected_layers)
            last_inspection_pose: dict | None = None
            for index, layer in enumerate(selected_layers, start=1):
                ctx.check_cancel()
                action = "dry-running" if self.check_collisions else "quick-simulating"
                ctx.progress(index, total, f"{action} layer {layer.layer_index}")
                program_mode = "DRY" if self.check_collisions else "QUICK"
                name = _program_name(self.plan, layer.layer_index, program_mode)
                built = rdk.create_extrusion_layer_program(
                    name=name, points_xyz=points_array(layer),
                    orientation_rpy_deg=self.plan.setup.orientation_rpy_deg,
                    print_tool=self.plan.setup.print_tool,
                    work_frame=self.plan.setup.work_frame,
                    speed_mm_s=self.plan.recipe.robot_speed_mm_s,
                    travel_speed_mm_s=self.plan.recipe.travel_speed_mm_s,
                    rounding_mm=self.plan.recipe.path_rounding_mm,
                    approach_clearance_mm=self.plan.setup.approach_clearance_mm,
                    retreat_clearance_mm=self.plan.setup.retreat_clearance_mm,
                    air_on_program=mock_on, air_off_program=mock_off,
                    maximum_tool_axis_spin_deg=(
                        self.plan.setup.maximum_tool_axis_spin_deg),
                    maximum_path_targets=ecfg.max_path_targets_per_layer,
                    check_cancel=ctx.check_cancel)
                path_artifacts.extend(built["artifacts"])
                path_targets.extend(built["targets"])
                current_program = name
                # Generation and Update are synchronous RoboDK calls. A cancel
                # received during either call is acted on as soon as it returns,
                # before any simulated program can be started.
                ctx.check_cancel()
                validation = rdk.update_program(
                    name, collisions=self.check_collisions)
                ctx.check_cancel()
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
                inspect = _build_inspection_move(
                    rdk, self.plan, layer, inspection_name=inspection_name,
                    config=ecfg, camera=self.services.config.camera,
                    seed_pose=last_inspection_pose,
                    collisions=self.check_collisions)
                ctx.check_cancel()
                if inspect["pose"]:
                    last_inspection_pose = inspect["pose"]
                path_artifacts.extend(inspect["artifacts"])
                inspection_validation = inspect["validation"]
                current_program = inspection_name
                started = rdk.start_program(inspection_name, real_robot=False)
                if started < 0:
                    raise RuntimeError(
                        f"layer {layer.layer_index} inspection simulation could not start")
                _wait_program(ctx, rdk, inspection_name)
                current_program = None
                reports.append({"layer_index": layer.layer_index,
                                "path": validation, "inspection": inspection_validation,
                                "inspection_target": inspect["target"],
                                "inspection_pose": inspect["pose"]})
                if inspect["pose"]:
                    pose = inspect["pose"]
                    ctx.log(f"layer {layer.layer_index}: inspection pose derived — "
                            f"{pose['standoff_mm']:.0f} mm above the layer top, "
                            f"tilt {pose['tilt_deg']:.0f}deg / roll {pose['roll_deg']:.0f}deg"
                            + (f" (after {len(pose['rejected'])} rejected viewpoint(s))"
                               if pose["rejected"] else ""))
                ctx.log(f"layer {layer.layer_index}: path + inspection motion simulated; "
                        f"valve ON/OFF shown as mock events")
            rdk.move_j_joints(start_joints)
            report = {
                "kind": ("cylinder_dry_run" if self.check_collisions
                         else "cylinder_quick_simulation"),
                "mode": ("DRY_RUN" if self.check_collisions
                         else "QUICK_SIMULATION"),
                "fingerprint": self.plan.fingerprint, "all_ok": True,
                "returned_to_start": True, "physical_outputs_blocked": True,
                "collision_check_enabled": self.check_collisions,
                "simulated_layer_indices": self.layer_indices,
                "full_plan_simulated": len(self.layer_indices) == len(self.plan.layers),
                "representative_layers_approve_full_plan": self.approve_full_plan,
                "live_print_approved": bool(
                    not self.check_collisions and
                    (self.approve_full_plan
                     or len(self.layer_indices) == len(self.plan.layers))),
                "collision_map_optimization": collision_map_optimization,
                "layers": reports, "valve_events": valve_events,
                "setup": self.plan.setup.model_dump(mode="json"),
                "recipe": self.plan.recipe.model_dump(mode="json"),
                "run_dir": str(run_dir), "git_commit": _git_commit(),
                "build": build_info(),
            }
            (run_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            self.result = report
            if self.on_pass is not None and self.check_collisions:
                self.on_pass(self.plan.fingerprint)
            if self.on_preview_pass is not None and not self.check_collisions:
                self.on_preview_pass(
                    self.plan.fingerprint, self.layer_indices,
                    approve_full_plan=self.approve_full_plan)
            if self.check_collisions:
                ctx.log("dry run PASS: complete path, collisions, valve mocks, "
                        "inspection, return-to-start")
            else:
                ctx.log("quick visual simulation PASS: collision checks were skipped; "
                        f"simulated layer(s) {self.layer_indices}")
            completed = True
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
                cleanup = (mock_artifacts + path_targets
                           + (path_artifacts if completed else []))
                rdk.delete_items(list(dict.fromkeys(reversed(cleanup))))
            except Exception:
                pass
            if prior_simulation_speed is not None:
                try:
                    rdk.set_simulation_speed(prior_simulation_speed)
                except Exception:
                    pass
            if not completed and path_artifacts:
                ctx.log("run failed: curve/project/program kept in RoboDK for "
                        "inspection (waypoint targets were removed); "
                        "Reset / clean RoboDK path removes them")
            rdk.set_run_mode_raw(prior_mode)


class CylinderPrintJob:
    """Print, capture one RGB-D frame, process, and archive each cylinder layer."""

    def __init__(self, services, plan: CylinderPlan, *, check_collisions: bool = True):
        self.services = services
        self.plan = plan.model_copy(deep=True)
        self.check_collisions = bool(check_collisions)
        self.result: dict | None = None
        self.corrected_reference_xyz: np.ndarray | None = None

    def __call__(self, ctx: JobContext) -> dict:
        services = self.services
        rdk: RdkIO = services.rdk
        ecfg = services.config.extrusion
        _warn_if_stale(ctx)
        required = station_requirements(rdk, self.plan, ecfg)
        if not required["ready"]:
            missing = ", ".join(f"{v['type']} {v['name']!r}" for v in required["missing"])
            raise RuntimeError("station is not ready: " + missing)
        if not ecfg.hardware_io_test_approved:
            raise RuntimeError("hardware I/O test is not approved")
        if services.live.running:
            services.live.stop()
        # Prove that the inspection dependency is available before switching to
        # RUN_ROBOT or issuing even the fail-safe valve program. A dead Jetson
        # must block the run here, not after material has already been deposited.
        try:
            with _camera_hold(services, "extrusion-startup-camera-check"):
                camera_check = services.camera.grab(
                    with_depth=True, timeout=ecfg.grab_timeout_s)
            if camera_check.depth is None:
                raise RuntimeError("RGB-D readiness frame contained no depth")
        except Exception as exc:
            raise RuntimeError(
                "live print blocked before robot motion: inspection camera is not ready: "
                f"{exc}") from exc
        ctx.log("inspection camera ready: depth frame received before robot motion")
        ctx.check_cancel()
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
            "git_commit": _git_commit(), "build": build_info(),
            "calibration": calibration,
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
            last_inspection_pose: dict | None = None
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
                        travel_speed_mm_s=self.plan.recipe.travel_speed_mm_s,
                        rounding_mm=self.plan.recipe.path_rounding_mm,
                        approach_clearance_mm=self.plan.setup.approach_clearance_mm,
                        retreat_clearance_mm=self.plan.setup.retreat_clearance_mm,
                        air_on_program=ecfg.air_on_program,
                        air_off_program=ecfg.air_off_program,
                        maximum_tool_axis_spin_deg=(
                            self.plan.setup.maximum_tool_axis_spin_deg),
                        maximum_path_targets=ecfg.max_path_targets_per_layer,
                        check_cancel=ctx.check_cancel)
                    artifacts.extend(built["artifacts"])
                    # Never start real motion if cancellation arrived during a
                    # blocking RoboDK generation or collision-validation call.
                    ctx.check_cancel()
                    validation = rdk.update_program(
                        name, collisions=self.check_collisions)
                    ctx.check_cancel()
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
                    inspect = _build_inspection_move(
                        rdk, self.plan, layer, inspection_name=inspection_name,
                        config=ecfg, camera=services.config.camera,
                        seed_pose=last_inspection_pose,
                        collisions=self.check_collisions)
                    ctx.check_cancel()
                    if inspect["pose"]:
                        last_inspection_pose = inspect["pose"]
                    artifacts.extend(inspect["artifacts"])
                    inspection_validation = inspect["validation"]
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
                                    "inspection_target": inspect["target"],
                                    "inspection_pose": inspect["pose"],
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
                      "collision_check_enabled": self.check_collisions,
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
