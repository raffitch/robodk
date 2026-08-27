"""Ring-stack measure-only experiment: inspect -> capture -> process -> archive.

Nothing here prints. The operator places dried rings by hand; each press moves
ONLY the camera (the same derived, collision-validated, wrist-gated inspection
move the live print uses), takes one RGB-D frame, measures it and returns to
the start pose. No layer program, no AirOn/AirOff, no hardware-I/O gate.
Trials are archived with ``mode = "MEASURE_ONLY"`` and never counted as prints.

The inspect-and-capture sequence is deliberately DUPLICATED from
``service.CylinderPrintJob`` rather than factored out of it: that loop was
cell-validated on 2026-08-27, and refactoring it to serve an experiment would
put the live print at risk for no gain.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from ...core import runs
from ...core.build_info import build_info
from ...core.jobrunner import JobContext
from ...core.logging import REPO_ROOT
from ...core.rdk_io import RdkIO
from ..calibration.service import _camera_hold, ensure_real_robot_link
from .archive import ExtrusionArchive
from .models import CylinderPlan, LayerManifest
from .processing import characterize_ring, process_observation
from .service import (_build_inspection_move, _git_commit, _program_name, _utcnow,
                      _wait_program, _warn_if_stale)
from .toolpath import points_array

MODE = "MEASURE_ONLY"


def measure_station_requirements(rdk: RdkIO, plan: CylinderPlan, config) -> dict:
    """Only what the camera move needs: no print tool, no valve programs."""
    selected = plan.setup
    checks = [("work_frame", selected.work_frame, "frame"),
              ("inspection_tool", selected.inspection_tool, "tool")]
    if not selected.inspection_auto:
        checks.append(("inspection_target", selected.inspection_target, "target"))
    items = [{"role": role, "name": name, "type": kind,
              "present": rdk.item_exists_as(name, kind)} for role, name, kind in checks]
    return {"ready": all(item["present"] for item in items), "items": items,
            "missing": [item for item in items if not item["present"]]}


def _provenance(services) -> dict:
    return {"git_commit": _git_commit(), "build": build_info(),
            "calibration": runs.read_active("calibration"),
            "camera_resolution": services.config.camera.resolution,
            "camera_intrinsics": {
                "K": np.asarray(services.config.camera.K, dtype=float).tolist(),
                "dist_coeffs": list(services.config.camera.dist_coeffs)},
            "processing_config": services.config.extrusion.model_dump(mode="json")}


class MeasureSession:
    """One MEASURE_ONLY trial and everything measured in it (persisted as session.json)."""

    def __init__(self, root: Path, trial_id: str):
        self.root = Path(root)
        self.trial_id = trial_id
        self.trial_dir = self.root / trial_id
        self.takes: dict[int, int] = {}
        self.tops: dict[int, list[list[float]]] = {}      # layer -> latest measured_xyz
        self.last_pose: dict | None = None
        self.characterizations: list[dict] = []
        self.records: list[dict] = []

    # -- persistence --------------------------------------------------------
    @classmethod
    def create(cls, root: Path, plan: CylinderPlan, *, note: str = "",
               provenance: dict | None = None) -> "MeasureSession":
        trial_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{plan.fingerprint[:8]}"
        ExtrusionArchive(root).create_trial(
            trial_id, plan, provenance=provenance or {}, mode=MODE,
            experiment={"note": note, "kind": "hand-placed dried rings",
                        "created_at": _utcnow()})
        session = cls(root, trial_id)
        session.save()
        return session

    @classmethod
    def load(cls, root: Path, trial_id: str) -> "MeasureSession":
        session = cls(root, trial_id)
        data = json.loads((session.trial_dir / "session.json").read_text(encoding="utf-8"))
        session.takes = {int(k): int(v) for k, v in data.get("takes", {}).items()}
        session.tops = {int(k): v for k, v in data.get("tops", {}).items()}
        session.last_pose = data.get("last_pose")
        session.characterizations = list(data.get("characterizations", []))
        session.records = list(data.get("records", []))
        return session

    @classmethod
    def latest(cls, root: Path) -> "MeasureSession | None":
        root = Path(root)
        if not root.is_dir():
            return None
        for path in sorted(root.iterdir(), reverse=True):
            trial_file, session_file = path / "trial.json", path / "session.json"
            if not (trial_file.is_file() and session_file.is_file()):
                continue
            if json.loads(trial_file.read_text(encoding="utf-8")).get("mode") == MODE:
                return cls.load(root, path.name)
        return None

    def save(self) -> None:
        (self.trial_dir / "session.json").write_text(
            json.dumps(self.to_json(), indent=2), encoding="utf-8")

    def to_json(self) -> dict:
        return {"trial_id": self.trial_id, "mode": MODE, "takes": self.takes,
                "tops": self.tops, "last_pose": self.last_pose,
                "characterizations": self.characterizations, "records": self.records}

    # -- experiment state ---------------------------------------------------
    def next_take(self, layer_index: int) -> int:
        return self.takes.get(layer_index, 0) + 1

    def floor_profile(self, layer_index: int) -> np.ndarray | None:
        """The ring BELOW this one, as MEASURED -- not as planned."""
        below = self.tops.get(layer_index - 1)
        return None if below is None else np.asarray(below, dtype=float)

    def record_take(self, *, layer_index: int, take: int, measured_xyz, pose: dict | None,
                    summary: dict) -> None:
        self.takes[layer_index] = take
        if measured_xyz is not None:
            self.tops[layer_index] = np.asarray(measured_xyz, dtype=float).tolist()
        if pose:
            self.last_pose = pose
        self.records.append(summary)


def _inspect_and_capture(services, ctx: JobContext, plan: CylinderPlan, layer, *,
                         inspection_name: str, start_joints, seed_pose, collisions: bool,
                         artifacts: list[str]) -> dict:
    """Move the camera to the derived pose, settle, read the pose, grab ONE frame."""
    rdk: RdkIO = services.rdk
    ecfg = services.config.extrusion
    inspect = _build_inspection_move(
        rdk, plan, layer, inspection_name=inspection_name, config=ecfg,
        camera=services.config.camera, start_joints=start_joints,
        seed_pose=seed_pose, collisions=collisions)
    ctx.check_cancel()
    artifacts.extend(inspect["artifacts"])
    if rdk.start_program(inspection_name, real_robot=True) < 0:
        raise RuntimeError(f"inspection program {inspection_name} could not start")
    _wait_program(ctx, rdk, inspection_name)
    time.sleep(ecfg.settle_s)
    # Re-select the inspection TCP and chosen work frame before reading the
    # camera pose: the generated program's tool instruction does not update
    # RdkIO's cached tool transform.
    rdk.use_named_tool_frame(plan.setup.inspection_tool, plan.setup.work_frame)
    T_work_camera = rdk.camera_pose_T()
    started = time.perf_counter()
    frame = services.camera.grab(with_depth=True, timeout=ecfg.grab_timeout_s)
    capture_ms = (time.perf_counter() - started) * 1000.0
    if frame.depth is None:
        raise RuntimeError("RGB-D capture returned no depth")
    ok, jpeg = cv2.imencode(".jpg", frame.color)
    if ok:
        ctx.frame(jpeg.tobytes())
    return {"inspect": inspect, "T_work_camera": T_work_camera, "frame": frame,
            "capture_ms": capture_ms}


def _prepare_robot(services, ctx: JobContext, plan: CylinderPlan, *, label: str):
    """Everything before motion: station items, camera readiness, RUN_ROBOT, link.

    The camera is proven alive BEFORE RUN_ROBOT is applied, so a dead Jetson
    stops the measurement with the arm still parked rather than after it moved.
    """
    rdk: RdkIO = services.rdk
    ecfg = services.config.extrusion
    _warn_if_stale(ctx)
    required = measure_station_requirements(rdk, plan, ecfg)
    if not required["ready"]:
        missing = ", ".join(f"{v['type']} {v['name']!r}" for v in required["missing"])
        raise RuntimeError("station is not ready: " + missing)
    if services.live.running:
        services.live.stop()
    try:
        with _camera_hold(services, f"{label}-camera-check"):
            check = services.camera.grab(with_depth=True, timeout=ecfg.grab_timeout_s)
        if check.depth is None:
            raise RuntimeError("RGB-D readiness frame contained no depth")
    except Exception as exc:
        raise RuntimeError(
            "measurement blocked before robot motion: inspection camera is not ready: "
            f"{exc}") from exc
    ctx.log("inspection camera ready: depth frame received before robot motion")
    ctx.check_cancel()
    if rdk.apply_run_mode("run_robot") != "run_robot":
        raise RuntimeError("RoboDK refused RUN_ROBOT mode")
    ensure_real_robot_link(rdk, services.config.robodk, log=ctx.log)
    return rdk.current_joints()


class RingMeasureJob:
    """Measure ONE hand-placed ring: inspect, capture, process, archive, return."""

    def __init__(self, services, plan: CylinderPlan, session: MeasureSession,
                 layer_index: int, *, annotation: dict | None = None,
                 check_collisions: bool = True):
        if not 1 <= layer_index <= len(plan.layers):
            raise ValueError(f"layer_index {layer_index} outside 1..{len(plan.layers)}")
        self.services = services
        self.plan = plan.model_copy(deep=True)
        self.session = session
        self.layer_index = int(layer_index)
        self.annotation = dict(annotation or {})
        self.check_collisions = bool(check_collisions)
        self.result: dict | None = None

    def __call__(self, ctx: JobContext) -> dict:
        services = self.services
        rdk: RdkIO = services.rdk
        ecfg = services.config.extrusion
        layer = self.plan.layers[self.layer_index - 1]
        take = self.session.next_take(self.layer_index)
        name = _program_name(self.plan, self.layer_index, "MEASURE")
        inspection_name = name + "_Inspect"
        archive = ExtrusionArchive(REPO_ROOT / "runs" / "extrusion")
        artifacts: list[str] = []
        current_program: str | None = None
        start_joints = _prepare_robot(services, ctx, self.plan, label="extrusion-measure")
        try:
            with _camera_hold(services, "extrusion-measure"):
                ctx.progress(1, 4, f"layer {self.layer_index} take {take}: moving the camera")
                current_program = inspection_name
                captured = _inspect_and_capture(
                    services, ctx, self.plan, layer, inspection_name=inspection_name,
                    start_joints=start_joints, seed_pose=self.session.last_pose,
                    collisions=self.check_collisions, artifacts=artifacts)
                current_program = None
                inspect, frame = captured["inspect"], captured["frame"]
                T_work_camera, capture_ms = captured["T_work_camera"], captured["capture_ms"]
                ctx.progress(2, 4, "processing the frame")
                nominal = points_array(layer)
                base = dict(
                    trial_id=self.session.trial_id, layer_index=self.layer_index, take=take,
                    mode=MODE, recipe=self.plan.recipe,
                    toolpath_fingerprint=self.plan.fingerprint,
                    color_file="color.png", depth_file="depth.npy",
                    annotation=self.annotation,
                    provenance={**_provenance(services),
                                "work_frame": self.plan.setup.work_frame,
                                "inspection_tool": self.plan.setup.inspection_tool,
                                "inspection_target": inspect["target"],
                                "inspection_pose": inspect["pose"],
                                "T_work_camera": np.asarray(T_work_camera, dtype=float).tolist()})
                floor = self.session.floor_profile(self.layer_index)
                try:
                    processed = process_observation(
                        color=frame.color, depth=frame.depth, T_work_camera=T_work_camera,
                        K=services.config.camera.K, plan=self.plan, layer=layer, config=ecfg,
                        floor_profile=floor)
                except Exception as exc:
                    # A failed measurement still archives its raw RGB-D: the
                    # operator cannot re-place the ring exactly, so the frame is
                    # the only thing that can be reprocessed later.
                    manifest = LayerManifest(
                        **base, processing={"valid": False, "error": str(exc),
                                            "timings_ms": {"capture_ms": capture_ms}},
                        warnings=[str(exc)])
                    archive.write_layer(manifest, nominal_xyz=nominal, commanded_xyz=nominal,
                                        color=frame.color, depth=frame.depth,
                                        report={"valid": False, "error": str(exc)})
                    self.session.takes[self.layer_index] = take
                    self.session.save()
                    raise RuntimeError(
                        f"layer {self.layer_index} take {take} measurement invalid; "
                        f"raw RGB-D archived: {exc}") from exc
                timings = processed.report["timings_ms"]
                timings["capture_ms"] = capture_ms
                timings["acquisition_to_path_ms"] = capture_ms + timings["total_ms"]
                manifest = LayerManifest(
                    **base, measured_path_file="measured_path.json",
                    pointcloud_file="height-or-pointcloud.npy",
                    metrics=processed.metrics, geometry=processed.geometry,
                    processing=processed.report, warnings=processed.metrics.warnings)
                layer_dir = archive.write_layer(
                    manifest, nominal_xyz=nominal, commanded_xyz=nominal,
                    measured_xyz=processed.measured_xyz,
                    pointcloud_xyz=processed.filtered_xyz,
                    color=frame.color, depth=frame.depth,
                    derived_images={"segmentation.png": processed.segmentation,
                                    "skeleton.png": processed.skeleton,
                                    "comparison.png": processed.comparison},
                    report={**processed.report,
                            "metrics": processed.metrics.model_dump(mode="json")})
                summary = {"layer_index": self.layer_index, "take": take,
                           "layer_dir": str(layer_dir), "annotation": self.annotation,
                           "metrics": processed.metrics.model_dump(mode="json"),
                           "geometry": (processed.geometry.model_dump(mode="json")
                                        if processed.geometry else None),
                           "timings_ms": timings, "valid": processed.metrics.valid,
                           "timestamp": _utcnow()}
                self.session.record_take(layer_index=self.layer_index, take=take,
                                         measured_xyz=processed.measured_xyz,
                                         pose=inspect["pose"], summary=summary)
                self.session.save()
                ctx.log(f"layer {self.layer_index} take {take}: offset "
                        f"{processed.metrics.center_offset_norm_mm:.2f} mm, RMS "
                        f"{processed.metrics.rms_mm:.2f} mm, "
                        f"{timings['acquisition_to_path_ms']:.0f} ms acquisition->path")
            ctx.progress(4, 4, "returning to the start pose")
            self.result = {"kind": "ring_measure", "mode": MODE,
                           "trial_id": self.session.trial_id,
                           "fingerprint": self.plan.fingerprint, **summary}
            return self.result
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
