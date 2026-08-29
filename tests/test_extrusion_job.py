"""Extrusion dry/live job safety with fake RoboDK and camera services."""
from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tasni.core.camera import CameraError, Frame
from tasni.core.camera_lease import CameraLease
from tasni.core.config import AppConfig
from tasni.modules.extrusion.models import (CylinderRecipe, CylinderSetup,
                                            DeviationMetrics)
from tasni.modules.extrusion.processing import ProcessingResult
from tasni.modules.extrusion import service as service_mod
from tasni.modules.extrusion.service import (CylinderDryRunJob, CylinderPrintJob,
                                             reprocess_saved_layer)
from tasni.modules.extrusion.toolpath import generate_cylinder_plan


# The fake robot's joint vector. Real joints (not a string sentinel) because the
# job now takes a SETTLED pose reading, which parses them numerically.
START_JOINTS = (11.0, 22.0, 33.0, 44.0, 55.0, 66.0)
SIDE_APPROACH_JOINTS = (10.0, -70.0, 120.0, 0.0, -40.0, 0.0)
SIDE_JOINTS = (10.0, -55.0, 100.0, 0.0, -25.0, 0.0)

# Camera 500 mm above the layer-1 aim point (which sits at z=6 mm), so the
# fake pose and the fake 500 mm depth frame describe the same geometry.
FAKE_CAMERA_T = np.eye(4)
FAKE_CAMERA_T[:3, 3] = [0.0, 0.0, 506.0]


class Ctx:
    def __init__(self): self.logs = []; self.progresses = []; self.frames = []; self._cancelled = False
    def progress(self, *args): self.progresses.append(args)
    def log(self, message): self.logs.append(message)
    def frame(self, payload): self.frames.append(payload)
    @property
    def cancelled(self): return self._cancelled
    def check_cancel(self):
        if self._cancelled: raise RuntimeError("cancelled")


class FakeRdk:
    def __init__(self):
        self.mode = 6; self.absent: set[str] = set(); self.events = []; self.created = []; self.deleted = []
        self.station_calls = 0; self.fail_station_call = None
        self.targets = []; self.unreachable_targets = 0; self.bad_inspections = 0
        self.flipped_inspections = 0
        # Taught side-photo targets carry stored joints, as GUI-taught targets do.
        # Set an entry to None to model a cartesian-only target.
        self.taught_joints = {"TowardsSideCapture": SIDE_APPROACH_JOINTS,
                              "SideCapture": SIDE_JOINTS}
    def item_exists_as(self, name, kind): return name not in self.absent and bool(name)
    def target_joints(self, name): return self.taught_joints.get(name)
    def move_j(self, name): self.events.append(("move-target", name))
    def program_instructions(self, name):
        return (["Set IO_508=1", "Set IO_601=1"] if name == "AirOn"
                else ["Set IO_508=0", "Set IO_601=0"] if name == "AirOff" else [])
    def current_run_mode(self): return self.mode
    def set_run_mode_raw(self, value): self.mode = value; self.events.append(("restore-mode", value))
    def apply_run_mode(self, mode): self.mode = 1 if mode == "simulate" else 6; self.events.append(("mode", mode)); return mode
    def current_joints(self): return START_JOINTS
    def move_j_joints(self, joints): self.events.append(("move-joints", joints))
    def set_collision_checking(self, active):
        self.events.append(("global-collisions", active)); return True
    def simulation_speed(self): return 1.0
    def set_simulation_speed(self, ratio): self.events.append(("sim-speed", ratio))
    def disable_object_collision_pairs(self, names):
        self.events.append(("collision-ignore", tuple(names)))
        return {"objects": list(names), "pairs_disabled": len(names), "pairs_failed": 0}
    def ensure_mock_valve_programs(self, prefix):
        names = (prefix + "AirOn", prefix + "AirOff"); self.events.append(("mock", names)); return names
    def create_extrusion_layer_program(self, **kwargs):
        self.created.append(kwargs); self.events.append(("create", kwargs["name"]))
        return {"program": kwargs["name"], "project": kwargs["name"] + "_Settings",
                "artifacts": [kwargs["name"], kwargs["name"] + "_Settings",
                              kwargs["name"] + "_Curve"], "targets": []}
    def create_inspection_program(self, **kwargs):
        self.events.append(("create-inspection", kwargs["name"],
                            kwargs["inspection_tool"], kwargs["inspection_target"]))
        return {"program": kwargs["name"], "targets": []}
    def create_inspection_target(self, **kwargs):
        self.targets.append(kwargs)
        xyz = np.asarray(kwargs["T"], dtype=float)[:3, 3]
        self.events.append(("create-target", kwargs["name"], tuple(np.round(xyz, 3))))
        if len(self.targets) <= self.unreachable_targets:
            return {"created": False, "target": kwargs["name"],
                    "reason": "no IK solution on the neutral wrist branch within +/-90 deg"}
        return {"created": True, "target": kwargs["name"], "reason": "",
                "joints": [89.8, -62.5, 147.8, 0.9, -54.1, -0.2],
                "axis_4_rotation_deg": 0.69, "axis_5_rotation_deg": -11.58,
                "axis_6_rotation_deg": -0.83}

    def camera_axes_in_frame(self, tool, frame, joints):
        """Measured on the cell: the camera's +X reads [-1, 0, 0] in the work frame."""
        self.events.append(("camera-axes", tool, frame, joints))
        T = np.eye(4)
        T[:3, 0], T[:3, 1], T[:3, 2] = [-1, 0, 0], [0, 1, 0], [0, 0, -1]
        return T

    def program_neutral_wrist_report(self, name, neutral_joints, limit=90.0):
        self.events.append(("wrist", name, limit))
        if self.flipped_inspections:
            self.flipped_inspections -= 1
            raise RuntimeError(
                "generated path sample 3 turns axis 4 178.1 deg from neutral; "
                "limit is +/-90.0 deg. The wrist-flipped path was blocked.")
        return {"sample_count": 12, "maximum_axis_4_rotation_seen_deg": 0.7,
                "maximum_axis_5_rotation_seen_deg": 11.6,
                "maximum_axis_6_rotation_seen_deg": 0.9}
    def update_program(self, name, collisions=True):
        self.events.append(("update", name, collisions))
        if name.endswith("_Inspect") and self.bad_inspections:
            self.bad_inspections -= 1
            return {"instructions_ok": 1, "time_s": 0, "distance_mm": 0,
                    "percent_ok": 41.0, "problems": "Collision detected: spindle/Table"}
        return {"instructions_ok": 100, "time_s": 0.5, "distance_mm": 100,
                "percent_ok": 100.0, "problems": ""}
    # A healthy dispatch really makes the DRIVER work: READY -> WORKING -> READY.
    # A fake that sat on READY would model the very fault under investigation and
    # call it a successful print.
    driver_polls = 0
    def driver_state(self):
        self.driver_polls += 1
        code = 1 if self.driver_polls == 2 else 0
        return {"code": code, "name": "WORKING" if code else "READY", "message": ""}
    def _dispatch_report(self, name):
        # Mirror the real RdkIO: a healthy dispatch clears every instruction and
        # the station really is in RUN_ROBOT. A fake that returned None here let
        # the live job's dispatch logging pass tests while breaking on the cell.
        return {"started": True, "start_method": "item_start",
                "start_result": "OK", "run_code": None,
                "instruction_count": 12, "run_mode": 6,
                "run_mode_expected": 6}
    def dispatch_program(self, name, real_robot):
        self.events.append(("start", name, real_robot))
        return self._dispatch_report(name)
    def start_program(self, name, real_robot):
        return 0 if self.dispatch_program(name, real_robot)["started"] else -1
    def program_busy(self, name): return False
    def stop_program(self, name): self.events.append(("stop", name))
    def delete_items(self, names): self.deleted.extend(names)
    def run_station_program(self, name, real_robot):
        self.station_calls += 1
        self.events.append(("station-program", name, real_robot))
        if self.station_calls == self.fail_station_call:
            raise RuntimeError("controller did not confirm output OFF")
        return self._dispatch_report(name)
    def use_named_tool_frame(self, tool, frame): self.events.append(("select", tool, frame)); return np.eye(4)
    def camera_pose_T(self): return FAKE_CAMERA_T.copy()


class FakeCamera:
    def __init__(self): self.grabs = 0; self.depth_grabs = 0; self.witness_grabs = 0
    def grab(self, **kwargs):
        self.grabs += 1
        # Colour-only grabs are the flange-camera motion witness taken either side
        # of a layer program; the depth grab is the ONE authoritative measurement
        # capture per layer. They are counted apart so the unicast-camera
        # discipline stays asserted on the capture that matters.
        if kwargs.get("color_only"):
            self.witness_grabs += 1
        else:
            self.depth_grabs += 1
        return Frame(color=np.zeros((16, 16, 3), np.uint8),
                     depth=np.full((16, 16), 500, np.uint16), timestamp=1.0)


def plan(*, layers=2, correction=True, auto_inspection=False):
    return generate_cylinder_plan(
        CylinderRecipe(radius_mm=40, layer_count=layers, layer_height_mm=5,
                       bead_diameter_mm=6, robot_speed_mm_s=75,
                       extrusion_rate_pct=30, points_per_circle=24,
                       correction_enabled=correction),
        CylinderSetup(print_tool="SelectedNozzle", work_frame="SelectedFrame",
                      inspection_tool="SelectedCamera",
                      inspection_target="" if auto_inspection else "SelectedInspect",
                      inspection_auto=auto_inspection,
                      orientation_rpy_deg=(180, 0, 30), approach_clearance_mm=25,
                      retreat_clearance_mm=35))


def services(tmp_path):
    cfg = AppConfig()
    cfg.extrusion.hardware_io_test_approved = True
    # FakeRdk.program_busy is always False, so the real grace would be paid
    # in full for every program. The grace itself is covered by
    # tests/test_extrusion_wait.py against a scripted busy sequence.
    cfg.extrusion.program_start_grace_s = 0.0
    cfg.extrusion.inspection_arrival_retry_s = 0.0
    cfg.robodk.connect_robot_on_connect = False
    rdk = FakeRdk(); camera = FakeCamera()
    return SimpleNamespace(config=cfg, rdk=rdk, camera=camera,
                           camera_lease=CameraLease(),
                           live=SimpleNamespace(running=False, stop=lambda: None)), rdk, camera


def fake_processing(**kwargs):
    layer = kwargs["layer"]
    pts = np.array([[p.x_mm, p.y_mm, p.z_mm] for p in layer.points])
    metrics = DeviationMetrics(mean_absolute_mm=1, rms_mm=1.2, maximum_mm=2,
                               measured_center_mm=(0, 0), measured_radius_mm=41,
                               path_completeness=.99, maximum_angular_gap_deg=5,
                               valid=True)
    image = np.zeros((12, 12), np.uint8)
    return ProcessingResult(pts, pts.copy(), metrics, image, image,
                            np.zeros((12, 12, 3), np.uint8),
                            {"counts": {"raw_depth_pixels": 256},
                             "timings_ms": {"total_ms": 10},
                             "branch_guard_attempts": [{"attempt": 1}]},
                            filtered_xyz=pts.copy())


def test_dry_run_uses_mock_outputs_and_restores_mode(tmp_path, monkeypatch):
    svc, rdk, camera = services(tmp_path)
    monkeypatch.setattr(service_mod, "new_run_dir",
                        lambda module, stamp: _mkdir(tmp_path / module / stamp))
    passed = []
    output = CylinderDryRunJob(svc, plan(), on_pass=passed.append)(Ctx())
    assert output["all_ok"] and output["physical_outputs_blocked"]
    assert passed == [output["fingerprint"]]
    assert camera.grabs == 0
    assert all(call["air_on_program"].startswith("TasniDry_") for call in rdk.created)
    assert all(call["travel_speed_mm_s"] == 200 and
               call["rounding_mm"] == 1 for call in rdk.created)
    assert not any(event[0] == "station-program" for event in rdk.events)
    assert ("collision-ignore", ("Tasni Scan Mesh",)) in rdk.events
    assert ("global-collisions", False) in rdk.events
    assert all(event[2] is True for event in rdk.events if event[0] == "update")
    assert output["collision_check_enabled"] is True
    assert output["full_plan_simulated"] is True
    assert rdk.mode == 6


def test_quick_visual_simulation_skips_collisions_and_never_grants_live_approval(
        tmp_path, monkeypatch):
    svc, rdk, camera = services(tmp_path)
    monkeypatch.setattr(service_mod, "new_run_dir",
                        lambda module, stamp: _mkdir(tmp_path / module / stamp))
    approved = []
    previewed = []

    output = CylinderDryRunJob(
        svc, plan(), on_pass=approved.append,
        on_preview_pass=lambda fingerprint, layers, **_: previewed.append((fingerprint, layers)),
        check_collisions=False)(Ctx())

    assert output["kind"] == "cylinder_quick_simulation"
    assert output["mode"] == "QUICK_SIMULATION"
    assert output["collision_check_enabled"] is False
    assert output["simulated_layer_indices"] == [1, 2]
    assert output["full_plan_simulated"] is True
    assert approved == []
    assert previewed == [(output["fingerprint"], [1, 2])]
    assert camera.grabs == 0
    assert not any(event[0] == "collision-ignore" for event in rdk.events)
    assert ("global-collisions", False) in rdk.events
    assert ("sim-speed", 5.0) in rdk.events
    assert ("sim-speed", 1.0) in rdk.events
    assert all(event[2] is False for event in rdk.events if event[0] == "update")
    assert all(call["name"].startswith("TasniCylinder_QUICK_")
               for call in rdk.created)


def test_quick_visual_simulation_can_run_only_selected_layers(tmp_path, monkeypatch):
    svc, rdk, _ = services(tmp_path)
    monkeypatch.setattr(service_mod, "new_run_dir",
                        lambda module, stamp: _mkdir(tmp_path / module / stamp))
    previewed = []

    output = CylinderDryRunJob(
        svc, plan(layers=3), check_collisions=False, layer_indices=[2],
        approve_full_plan=True,
        on_preview_pass=lambda fingerprint, layers, **_: previewed.append((fingerprint, layers)),
    )(Ctx())

    assert output["simulated_layer_indices"] == [2]
    assert output["full_plan_simulated"] is False
    assert output["representative_layers_approve_full_plan"] is True
    assert output["live_print_approved"] is True
    assert previewed == [(output["fingerprint"], [2])]
    assert len(rdk.created) == 1 and "_L002" in rdk.created[0]["name"]
    assert not any("_L001" in str(event) or "_L003" in str(event)
                   for event in rdk.events)


def test_failed_dry_run_keeps_path_for_inspection_but_deletes_mock_io(tmp_path, monkeypatch):
    svc, rdk, _ = services(tmp_path)
    monkeypatch.setattr(service_mod, "new_run_dir",
                        lambda module, stamp: _mkdir(tmp_path / module / stamp))
    rdk.update_program = lambda name, collisions=True: {
        "instructions_ok": 1, "time_s": 0, "distance_mm": 0,
        "percent_ok": 2.6, "problems": "Collision detected at MoveJ 1",
    }
    ctx = Ctx()
    with pytest.raises(RuntimeError, match="Collision detected"):
        CylinderDryRunJob(svc, plan(layers=1))(ctx)
    assert any(name.startswith("TasniDry_") for name in rdk.deleted)
    assert not any(name.startswith("TasniCylinder_") for name in rdk.deleted)
    assert any("kept in RoboDK" in message for message in ctx.logs)


def test_cancel_during_blocking_validation_never_starts_program(tmp_path, monkeypatch):
    """A cancel received inside RoboDK Update must win before playback starts."""
    svc, rdk, _ = services(tmp_path)
    monkeypatch.setattr(service_mod, "new_run_dir",
                        lambda module, stamp: _mkdir(tmp_path / module / stamp))
    ctx = Ctx()
    original_update = rdk.update_program

    def update_then_cancel(name, collisions=True):
        report = original_update(name, collisions=collisions)
        ctx._cancelled = True
        return report

    rdk.update_program = update_then_cancel
    with pytest.raises(RuntimeError, match="cancelled"):
        CylinderDryRunJob(svc, plan(layers=1))(ctx)

    assert not any(event[0] == "start" for event in rdk.events)


def _mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_live_print_forces_off_captures_once_and_archives(tmp_path, monkeypatch):
    svc, rdk, camera = services(tmp_path)
    monkeypatch.setattr(service_mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(service_mod, "process_observation", fake_processing)
    monkeypatch.setattr(service_mod, "_git_commit", lambda: "abc123")
    monkeypatch.setattr(service_mod.time, "sleep", lambda _: None)
    monkeypatch.setattr(service_mod.runs, "read_active", lambda module: {"run_id": "cal-1"})
    output = CylinderPrintJob(svc, plan(), check_collisions=False)(Ctx())
    assert output["kind"] == "cylinder_print"
    assert output["collision_check_enabled"] is False
    assert output["correction_available"] and not output["correction_executed"]
    # One readiness frame before any robot command, then one measurement/layer.
    assert camera.depth_grabs == 3          # exactly one measurement per layer
    assert camera.witness_grabs == 4        # 2 layers x witness either side of the program
    # First real-cell action after mode/link setup is fail-safe AirOff, before create/motion.
    first_off = next(i for i, e in enumerate(rdk.events) if e[0] == "station-program")
    first_create = next(i for i, e in enumerate(rdk.events) if e[0] == "create")
    assert first_off < first_create
    assert rdk.events[first_off][1] == svc.config.extrusion.air_off_program
    assert all(call["print_tool"] == "SelectedNozzle" and
               call["work_frame"] == "SelectedFrame" for call in rdk.created)
    assert all(call["air_on_program"] == "AirOn" and
               call["air_off_program"] == "AirOff" for call in rdk.created)
    assert all(event[2] is False for event in rdk.events if event[0] == "update")
    for index in (1, 2):
        layer = Path(output["trial_dir"]) / f"layer-{index:03d}"
        for name in ("manifest.json", "color.png", "depth.npy", "segmentation.png",
                     "skeleton.png", "comparison.png", "nominal_path.json",
                     "measured_path.json", "corrected_path.json", "report.json",
                     "height-or-pointcloud.npy"):
            assert (layer / name).is_file(), name
    reprocessed = reprocess_saved_layer(
        tmp_path / "runs" / "extrusion", output["trial_id"], 1)
    assert reprocessed["layer_index"] == 1
    manifest = (Path(output["trial_dir"]) / "layer-001" / "manifest.json").read_text()
    assert "last_reprocessed_at" in manifest


def test_live_print_forces_off_after_processing_fault(tmp_path, monkeypatch):
    svc, rdk, _ = services(tmp_path)
    monkeypatch.setattr(service_mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(service_mod, "_git_commit", lambda: "abc123")
    monkeypatch.setattr(service_mod.time, "sleep", lambda _: None)
    monkeypatch.setattr(service_mod.runs, "read_active", lambda module: None)
    monkeypatch.setattr(service_mod, "process_observation",
                        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("bad skeleton")))
    with pytest.raises(RuntimeError, match="raw RGB-D archived"):
        CylinderPrintJob(svc, plan(layers=1))(Ctx())
    station_calls = [event for event in rdk.events if event[0] == "station-program"]
    assert station_calls[-1][1] == "AirOff"
    trial = next((tmp_path / "runs" / "extrusion").iterdir())
    assert (trial / "layer-001" / "depth.npy").is_file()
    assert "bad skeleton" in (trial / "layer-001" / "report.json").read_text()


def test_live_print_blocks_before_any_robot_command_when_camera_is_offline(tmp_path):
    svc, rdk, _ = services(tmp_path)

    class OfflineCamera:
        def grab(self, **kwargs):
            raise CameraError("camera timeout (10.12.171.70:1024)")

    svc.camera = OfflineCamera()
    with pytest.raises(RuntimeError, match="blocked before robot motion"):
        CylinderPrintJob(svc, plan(layers=1))(Ctx())

    assert not any(event[0] in {"mode", "station-program", "create", "start"}
                   for event in rdk.events)


def test_failed_final_valve_off_inhibits_return_motion(tmp_path, monkeypatch):
    svc, rdk, _ = services(tmp_path)
    rdk.fail_station_call = 4  # startup, pre-layer, post-path pass; fault-exit OFF fails
    monkeypatch.setattr(service_mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(service_mod, "_git_commit", lambda: "abc123")
    monkeypatch.setattr(service_mod.time, "sleep", lambda _: None)
    monkeypatch.setattr(service_mod.runs, "read_active", lambda module: None)
    monkeypatch.setattr(service_mod, "process_observation",
                        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("bad skeleton")))
    ctx = Ctx()
    with pytest.raises(RuntimeError, match="raw RGB-D archived"):
        CylinderPrintJob(svc, plan(layers=1))(ctx)
    assert not any(event[0] == "move-joints" for event in rdk.events)
    assert any("return motion inhibited" in message for message in ctx.logs)


# -- automatic inspection pose ---------------------------------------------

def test_auto_inspection_creates_one_target_per_layer_at_the_planned_standoff(
        tmp_path, monkeypatch):
    svc, rdk, _ = services(tmp_path)
    monkeypatch.setattr(service_mod, "new_run_dir",
                        lambda module, stamp: _mkdir(tmp_path / module / stamp))
    output = CylinderDryRunJob(svc, plan(layers=2, auto_inspection=True))(Ctx())
    assert output["all_ok"]
    created = [event for event in rdk.events if event[0] == "create-target"]
    assert len(created) == 2, "one derived pose per layer, not one for the trial"
    # The camera rides one layer height up, holding the standoff to the fresh top.
    assert created[1][2][2] - created[0][2][2] == 5.0
    assert created[0][2][2] == 6.0 + 300.0        # first layer top + standoff
    assert all(name.startswith("TasniCylinder_") and name.endswith("_Target")
               for _, name, _ in created)
    assert "L001" in created[0][1] and "L002" in created[1][1]
    # The derived target is what the inspection program actually moves to.
    inspections = [event for event in rdk.events if event[0] == "create-inspection"]
    assert [event[3] for event in inspections] == [name for _, name, _ in created]
    assert all(name in rdk.deleted for _, name, _ in created)
    assert output["layers"][0]["inspection_pose"]["tilt_deg"] == 0.0


def test_auto_inspection_moves_to_the_next_candidate_when_one_collides(
        tmp_path, monkeypatch):
    svc, rdk, _ = services(tmp_path)
    monkeypatch.setattr(service_mod, "new_run_dir",
                        lambda module, stamp: _mkdir(tmp_path / module / stamp))
    rdk.unreachable_targets = 1     # straight down has no IK here
    rdk.bad_inspections = 1         # ...and the next one collides
    output = CylinderDryRunJob(svc, plan(layers=1, auto_inspection=True))(Ctx())
    assert output["all_ok"]
    chosen = output["layers"][0]["inspection_pose"]
    assert len(rdk.targets) == 3, "tried straight down, then the rejected one, then a third"
    assert "neutral wrist branch" in chosen["rejected"][0]["reason"]
    assert "Collision detected" in chosen["rejected"][1]["reason"]
    assert (chosen["tilt_deg"], chosen["roll_deg"]) != (0.0, 0.0)


def test_auto_inspection_fails_loudly_instead_of_inspecting_from_nowhere(
        tmp_path, monkeypatch):
    svc, rdk, _ = services(tmp_path)
    monkeypatch.setattr(service_mod, "new_run_dir",
                        lambda module, stamp: _mkdir(tmp_path / module / stamp))
    rdk.unreachable_targets = 99
    with pytest.raises(RuntimeError, match="no reachable, collision-free inspection pose"):
        CylinderDryRunJob(svc, plan(layers=1, auto_inspection=True))(Ctx())


def test_live_print_archives_the_derived_viewpoint_it_actually_measured_from(
        tmp_path, monkeypatch):
    """The measurement is only reproducible if the record says where it was taken
    from — in auto mode that is a per-layer derived pose, not a taught name."""
    svc, rdk, _ = services(tmp_path)
    monkeypatch.setattr(service_mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(service_mod, "_git_commit", lambda: "abc123")
    monkeypatch.setattr(service_mod.time, "sleep", lambda _: None)
    monkeypatch.setattr(service_mod.runs, "read_active", lambda module: None)
    monkeypatch.setattr(service_mod, "process_observation", fake_processing)
    monkeypatch.setattr(service_mod, "ensure_real_robot_link", lambda *a, **k: None)
    output = CylinderPrintJob(svc, plan(layers=1, auto_inspection=True))(Ctx())
    manifest = json.loads((Path(output["trial_dir"]) / "layer-001" / "manifest.json")
                          .read_text(encoding="utf-8"))
    provenance = manifest["provenance"]
    assert provenance["inspection_target"].endswith("_Target")
    assert provenance["inspection_pose"]["standoff_mm"] == 300.0
    assert provenance["inspection_pose"]["aim_mm"][2] == 6.0     # top of layer 1
    assert provenance["inspection_pose"]["tilt_deg"] == 0.0


def test_auto_inspection_reuses_the_previous_layers_winner(tmp_path, monkeypatch):
    """Review finding: the candidate sweep restarted at straight-down every
    layer, re-paying a collision-ON program Update per rejected candidate per
    layer. The previous layer's validated winner is tried first now."""
    svc, rdk, camera = services(tmp_path)
    monkeypatch.setattr(service_mod, "new_run_dir",
                        lambda module, stamp: _mkdir(tmp_path / module / stamp))
    rdk.bad_inspections = 1        # layer 1: straight-down fails validation once
    output = CylinderDryRunJob(svc, plan(layers=2, auto_inspection=True))(Ctx())
    first = output["layers"][0]["inspection_pose"]
    second = output["layers"][1]["inspection_pose"]
    assert first["roll_deg"] != 0.0          # layer 1 fell through to a roll
    assert (second["tilt_deg"], second["azimuth_deg"], second["roll_deg"]) == \
        (first["tilt_deg"], first["azimuth_deg"], first["roll_deg"])
    assert second["rejected"] == []          # no wasted candidates on layer 2


def test_auto_inspection_rolls_relative_to_the_camera_not_the_work_frame(
        tmp_path, monkeypatch):
    """Roll zero must mean the operator's parked camera orientation.

    Measured on the cell: the camera's +X reads [-1, 0, 0] in the work frame, so
    a frame-referenced roll zero is 180 deg away and RoboDK can only reach it
    through a wrist flip.
    """
    svc, rdk, _ = services(tmp_path)
    monkeypatch.setattr(service_mod, "new_run_dir",
                        lambda module, stamp: _mkdir(tmp_path / module / stamp))

    output = CylinderDryRunJob(svc, plan(layers=1, auto_inspection=True))(Ctx())

    pose = output["layers"][0]["inspection_pose"]
    assert pose["roll_reference"] == "camera_at_start"
    assert pose["roll_reference_x"] == [-1.0, 0.0, 0.0]
    chosen = rdk.targets[0]["T"]
    np.testing.assert_allclose(np.asarray(chosen)[:3, 0], [-1.0, 0.0, 0.0], atol=1e-9)
    assert ("camera-axes", "SelectedCamera", "SelectedFrame", START_JOINTS) in rdk.events


def test_auto_inspection_records_the_joints_it_actually_locked(tmp_path, monkeypatch):
    svc, rdk, _ = services(tmp_path)
    monkeypatch.setattr(service_mod, "new_run_dir",
                        lambda module, stamp: _mkdir(tmp_path / module / stamp))

    output = CylinderDryRunJob(svc, plan(layers=1, auto_inspection=True))(Ctx())

    pose = output["layers"][0]["inspection_pose"]
    assert pose["joints"] == [89.8, -62.5, 147.8, 0.9, -54.1, -0.2]
    assert pose["axis_4_rotation_deg"] == 0.69
    assert pose["axis_5_rotation_deg"] == -11.58
    assert pose["wrist"]["maximum_axis_4_rotation_seen_deg"] == 0.7


def test_an_inspection_path_that_flips_mid_move_rejects_that_candidate(
        tmp_path, monkeypatch):
    """The endpoint can be neutral while the path RoboDK interpolates is not.

    That is a rejection like a collision, not a run failure -- the walk moves on.
    """
    svc, rdk, _ = services(tmp_path)
    monkeypatch.setattr(service_mod, "new_run_dir",
                        lambda module, stamp: _mkdir(tmp_path / module / stamp))
    rdk.flipped_inspections = 1

    output = CylinderDryRunJob(svc, plan(layers=1, auto_inspection=True))(Ctx())

    assert output["all_ok"]
    chosen = output["layers"][0]["inspection_pose"]
    assert len(chosen["rejected"]) == 1
    assert "turns axis 4" in chosen["rejected"][0]["reason"]
    assert (chosen["tilt_deg"], chosen["azimuth_deg"], chosen["roll_deg"]) != (0.0, 0.0, 0.0)


def test_live_print_deletes_its_robodk_artifacts_by_default(tmp_path, monkeypatch):
    svc, rdk, camera = services(tmp_path)
    monkeypatch.setattr(service_mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(service_mod, "process_observation", fake_processing)
    monkeypatch.setattr(service_mod, "_git_commit", lambda: "abc123")
    monkeypatch.setattr(service_mod.time, "sleep", lambda _: None)
    monkeypatch.setattr(service_mod.runs, "read_active", lambda module: {"run_id": "cal-1"})
    CylinderPrintJob(svc, plan(), check_collisions=False)(Ctx())
    assert rdk.deleted, "generated programs/targets should be cleaned up by default"


def test_live_print_can_keep_its_robodk_artifacts(tmp_path, monkeypatch):
    """The generated programs are the only record of what the robot was actually
    told to do. Deleting them unconditionally makes a failed run unexaminable --
    the operator must be able to keep them and inspect the station afterwards.
    """
    svc, rdk, camera = services(tmp_path)
    monkeypatch.setattr(service_mod, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(service_mod, "process_observation", fake_processing)
    monkeypatch.setattr(service_mod, "_git_commit", lambda: "abc123")
    monkeypatch.setattr(service_mod.time, "sleep", lambda _: None)
    monkeypatch.setattr(service_mod.runs, "read_active", lambda module: {"run_id": "cal-1"})
    output = CylinderPrintJob(svc, plan(), check_collisions=False,
                              keep_artifacts=True)(Ctx())
    assert rdk.deleted == [], "artifacts must survive when the operator asked to keep them"
    assert output["artifacts_kept"] is True
    assert output["artifacts"], "the run must name what it left in the station"
