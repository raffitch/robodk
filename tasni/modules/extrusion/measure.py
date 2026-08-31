"""Ring-stack measure-only experiment: inspect -> capture -> process -> archive.

Nothing here prints. The operator places dried rings by hand; each press moves
ONLY the camera (the same derived, collision-validated, wrist-gated inspection
move the live print uses), fuses a short RGB-D burst, measures it and returns to
the start pose. No layer program, no AirOn/AirOff, no hardware-I/O gate.
Trials are archived with ``mode = "MEASURE_ONLY"`` and never counted as prints.

The inspect-and-capture sequence is deliberately DUPLICATED from
``service.CylinderPrintJob`` rather than factored out of it: that loop was
cell-validated on 2026-08-27, and refactoring it to serve an experiment would
put the live print at risk for no gain.
"""
from __future__ import annotations

import json
import math
import time
import warnings
from dataclasses import replace
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
from .processing import characterize_ring, measure_take
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
        # The plan the operator APPLIED from a characterization: the only plan
        # this session's takes may be scored against, and the one to rebuild
        # after a backend restart.
        self.applied: dict | None = None

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
        session.applied = data.get("applied")
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
                "characterizations": self.characterizations, "records": self.records,
                "applied": self.applied}

    # -- experiment state ---------------------------------------------------
    def next_take(self, layer_index: int) -> int:
        return self.takes.get(layer_index, 0) + 1

    def floor_profile(self, layer_index: int) -> np.ndarray | None:
        """The ring BELOW this one, as MEASURED -- not as planned."""
        below = self.tops.get(layer_index - 1)
        return None if below is None else np.asarray(below, dtype=float)

    def record_take(self, *, layer_index: int, take: int, measured_xyz, pose: dict | None,
                    summary: dict) -> None:
        """Record one take. ``measured_xyz`` None (a failure) leaves the floor alone.

        A ring that could not be measured is not a surface to stack the next
        layer's ROI on, so a failed take must never become ``tops[layer]``.
        """
        self.takes[layer_index] = max(take, self.takes.get(layer_index, 0))
        if measured_xyz is not None:
            self.tops[layer_index] = np.asarray(measured_xyz, dtype=float).tolist()
        if pose:
            self.last_pose = pose
        self.upsert_record(summary)

    def upsert_record(self, summary: dict) -> None:
        """Replace the record for this (layer, take) or append it, keeping order.

        Reprocessing an archived take rewrites a record that already exists;
        appending a second one would double-count it in the paper statistics.
        """
        key = (summary.get("layer_index"), summary.get("take"))
        for index, existing in enumerate(self.records):
            if (existing.get("layer_index"), existing.get("take")) == key:
                self.records[index] = summary
                return
        self.records.append(summary)

    def sync_take_from_archive(self, layer_dir: Path, manifest: dict, measured_xyz) -> dict:
        """Fold an offline-reprocessed take back into the session.

        ``reprocess_saved_layer`` rewrites a take's manifest; without this the
        session that the next layer's floor and the operator's table are read
        from would still hold the failed take (cell, 2026-08-28: layer-001 was
        rescued to a valid measurement while session.json stayed empty).
        """
        summary = take_summary(layer_dir, manifest, reprocessed=True)
        self.record_take(layer_index=summary["layer_index"], take=summary["take"],
                         measured_xyz=measured_xyz if summary["valid"] else None,
                         pose=None, summary=summary)
        self.save()
        return summary


def depth_plane_check(depth, T_work_camera, config, *, unit_mm: float = 1.0) -> dict:
    """Does this depth frame describe the pose it was taken at?

    Looking straight down at the work plane, the median depth over the frame is
    the camera's own height above that plane: the board dominates the view and
    the deposit stands a few millimetres proud of it. So the median must fall
    between "the camera height, less the tallest deposit we would believe" and
    "the camera height, plus a little for the plane's own error".

    On the cell (2026-08-29) a capture came back with a correct colour frame of
    the board at 312 mm and a depth frame reading 447 mm -- the height the arm
    had been parked at. The depth stream had STALLED: every grab returned a
    byte-identical depth buffer for over an hour while colour stayed live, so
    the camera kept reporting the distance it had last managed to measure.
    Restarting the camera service cleared it. That frame failed loudly because
    everything landed outside the search region; a smaller residual would have
    passed every gate and quietly moved a paper number instead.

    ``depth`` is raw camera WORDS, not millimetres -- ``unit_mm`` converts
    (protocol 2's native depth is 0.1 mm/word; the caller passes
    ``frame.geometry.depth_unit_mm``). Left at the 1.0 mm/word default this
    reads a 0.1 mm-unit frame's true ~300 mm standoff as ~3000 mm, which fails
    loudly but points at the work frame or a frozen camera -- the wrong cause.

    **Off-axis.** The paragraph above holds looking straight DOWN, where the
    frame's median depth IS the camera's height above the plane. Tilt separates
    them: the camera drops to ``aim_z + standoff*cos(t)`` while the median stays
    at roughly the standoff, so the median runs ABOVE camera_z by
    ``standoff*(1 - cos t)`` -- and the high side has only
    ``depth_plane_slack_mm`` (15 mm) of budget. Computed against the real
    constants that fails above ~18 deg at a 300 mm standoff and above ~14 deg at
    500 mm, and even where it passes it spends 5-12 mm of that 15 mm budget on
    geometry, leaving almost nothing to catch the fault this gate exists for.

    So the expectation is scaled by the incidence, read from the pose itself:
    ``pose_from_aim`` sets ``z_axis = -away`` with ``away_z = cos(tilt)``, so
    ``-T[2, 2]`` IS ``cos(tilt)`` -- no convention to get wrong, and nothing to
    pass in that could disagree with where the arm actually went. At tilt 0
    every expression below collapses to the height-based form exactly, which is
    what keeps the cell-validated single-view path unmoved. Swept over
    300-800 mm x 0-30 deg the residual holds at +5.0..+5.8 mm (the ``aim_z``
    term the tilt-0 gate already carries), so sensitivity stays flat instead of
    decaying with tilt.
    """
    T = np.asarray(T_work_camera, dtype=float)
    camera_z = float(T[2, 3])
    values = np.asarray(depth)
    valid = values[values > 0]
    observed = float(np.median(valid)) * float(unit_mm) if valid.size else float("nan")
    ceiling = float(getattr(config, "characterize_max_height_mm", 40.0))
    slack = float(getattr(config, "depth_plane_slack_mm", 15.0))
    # -T[2,2] is cos(tilt) for every pose_from_aim pose; clamp guards a
    # hand-built or degenerate transform rather than trusting the caller.
    cos_incidence = float(np.clip(-T[2, 2], -1.0, 1.0))
    floor_cos = float(getattr(config, "multiview_min_cos_incidence", 0.5))
    base = {"camera_z_mm": camera_z, "observed_depth_mm": observed,
            "valid_pixels": int(valid.size), "cos_incidence": cos_incidence}
    if cos_incidence < floor_cos:
        # Dividing by this would manufacture an expectation from nothing. A pose
        # this far off-axis is a bug upstream, not a view worth gating.
        if cos_incidence <= 0.0:
            # A genuinely oblique-but-real view still has cos_incidence > 0; at
            # or below zero the lens is pointed AWAY from the work plane, which
            # a real hand-eye pose never produces. Seen once already (this
            # task): an identity-rotation test double stood in for a camera
            # pose because the old gate never read rotation at all.
            refused = (
                f"camera pose has the lens facing AWAY from the work plane "
                f"(cos_incidence={cos_incidence:.2f}) -- this usually means the pose was "
                f"built from an identity or otherwise non-camera rotation rather than a "
                f"real hand-eye pose, not that the view is genuinely oblique")
        else:
            refused = (f"camera incidence {np.degrees(np.arccos(cos_incidence)):.0f} deg "
                       f"exceeds the {np.degrees(np.arccos(floor_cos)):.0f} deg limit")
        return {**base, "expected_depth_mm": float("nan"),
                "accepted_range_mm": [float("nan"), float("nan")], "agrees": False,
                "refused": refused}
    expected = camera_z / cos_incidence
    low, high = expected - ceiling / cos_incidence, expected + slack / cos_incidence
    return {**base, "expected_depth_mm": expected,
            "accepted_range_mm": [round(low, 1), round(high, 1)],
            "agrees": bool(valid.size and low <= observed <= high)}


def take_summary(layer_dir: Path, manifest: dict, *, reprocessed: bool = False) -> dict:
    """One row of the session table, built from what the archive actually holds."""
    processing = manifest.get("processing") or {}
    metrics = manifest.get("metrics")
    return {"layer_index": int(manifest["layer_index"]), "take": int(manifest.get("take", 1)),
            "layer_dir": str(layer_dir), "layer_name": Path(layer_dir).name,
            "annotation": manifest.get("annotation") or {},
            "metrics": metrics, "geometry": manifest.get("geometry"),
            "timings_ms": processing.get("timings_ms") or {},
            "valid": bool((metrics or {}).get("valid", False)),
            "error": processing.get("error"),
            "reprocessed": bool(reprocessed or processing.get("offline_reprocess")),
            "timestamp": _utcnow()}


def _move_to_inspection(services, ctx: JobContext, plan: CylinderPlan, layer, *,
                        inspection_name: str, start_joints, seed_pose, collisions: bool,
                        artifacts: list[str], near_mm: float | None = None) -> dict:
    """Take the camera out to the derived pose, settle, and read where it landed.

    Split out of the capture so ONE excursion can serve several frames: with the
    arm parked at the pose, repeated grabs measure the sensing chain alone,
    without the robot's re-approach folded into every number (and without the
    minute of travel each extra trip costs at the cell).
    """
    rdk: RdkIO = services.rdk
    ecfg = services.config.extrusion
    inspect = _build_inspection_move(
        rdk, plan, layer, inspection_name=inspection_name, config=ecfg,
        camera=services.config.camera, start_joints=start_joints,
        seed_pose=seed_pose, collisions=collisions, near_mm=near_mm)
    ctx.check_cancel()
    artifacts.extend(inspect["artifacts"])
    # What the excursion costs the print starts here: the arm leaving the path.
    departed = time.perf_counter()
    if rdk.start_program(inspection_name, real_robot=True) < 0:
        raise RuntimeError(f"inspection program {inspection_name} could not start")
    _wait_program(ctx, rdk, inspection_name)
    move_ms = (time.perf_counter() - departed) * 1000.0
    time.sleep(ecfg.settle_s)
    # Re-select the inspection TCP and chosen work frame before reading the
    # camera pose: the generated program's tool instruction does not update
    # RdkIO's cached tool transform.
    rdk.use_named_tool_frame(plan.setup.inspection_tool, plan.setup.work_frame)
    return {"inspect": inspect, "T_work_camera": rdk.camera_pose_T(), "move_ms": move_ms,
            # The dwell is commanded, not measured: reporting the configured
            # value keeps the cycle total honest when a test stubs out sleep.
            "settle_ms": float(ecfg.settle_s) * 1000.0}


def _nonzero_median_depth(depth_frames) -> np.ndarray:
    """Per-pixel median of valid depth words; zero remains no measurement.

    The D435 uses zero for an invalid depth sample. Including that sentinel in a
    numeric median pulls thin or reflective surfaces toward the camera whenever
    only part of a burst sees them, so invalid samples must be ignored rather
    than averaged.
    """
    stack = np.asarray(depth_frames)
    if stack.ndim != 3 or stack.shape[0] < 1:
        raise ValueError("depth fusion needs an NxHxW burst")
    if stack.shape[0] == 1:
        return stack[0].copy()
    samples = np.where(stack > 0, stack.astype(np.float32), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        fused = np.nanmedian(samples, axis=0)
    return np.rint(np.nan_to_num(fused, nan=0.0)).astype(stack.dtype)


def _same_reconstruction_geometry(first, candidate) -> bool:
    """Whether two greetings describe the same depth-to-colour reconstruction.

    A protocol-2 greeting also carries live device temperature and achieved
    settings. Those are provenance, not geometry, and can legitimately change
    between connections. Comparing ``to_dict()`` therefore produced intermittent
    false failures during fusion. Keep the guard on only the values that can move
    a reconstructed point or change the frame layout.
    """
    if first is None or candidate is None:
        return first is candidate
    scalars_match = (
        first.protocol == candidate.protocol
        and first.legacy == candidate.legacy
        and first.depth_size == candidate.depth_size
        and first.color_size == candidate.color_size
        and np.isclose(first.depth_unit_mm, candidate.depth_unit_mm, rtol=0, atol=1e-12)
    )
    def same_array(a, b) -> bool:
        left, right = np.asarray(a), np.asarray(b)
        return left.shape == right.shape and np.allclose(left, right, rtol=0, atol=1e-10)

    arrays_match = all(same_array(a, b) for a, b in (
        (first.depth_K, candidate.depth_K),
        (first.depth_dist, candidate.depth_dist),
        (first.color_K_factory, candidate.color_K_factory),
        (first.T_color_depth, candidate.T_color_depth),
    ))
    return bool(scalars_match and arrays_match)


def _capture_at_pose(services, ctx: JobContext, T_work_camera) -> dict:
    """Grab and median-fuse validated RGB-D frames at one stationary pose."""
    ecfg = services.config.extrusion
    started = time.perf_counter()
    wanted = int(getattr(ecfg, "measure_depth_fusion_frames", 1))
    attempts = int(getattr(ecfg, "depth_stale_retries", 2))
    frames = []
    retries, frozen = 0, False
    previous = None
    check = None
    # One connection means one protocol greeting and no TCP handshake/slow-start
    # per frame. It also makes the five samples a genuinely consecutive burst.
    with services.camera.stream(timeout=ecfg.grab_timeout_s) as stream:
        while len(frames) < wanted:
            ctx.check_cancel()
            frame = stream.read(with_depth=True)
            if frame.depth is None:
                raise RuntimeError("RGB-D capture returned no depth")
            # Checked before any depth word is interpreted as millimetres: a missing
            # greeting on a 0.1 mm-unit frame would otherwise look ten times too far.
            if frame.geometry is None:
                raise RuntimeError("depth frame arrived without a protocol-2 greeting")
            if frames:
                first = frames[0]
                if (frame.depth.shape != first.depth.shape
                        or frame.color.shape != first.color.shape
                        or not _same_reconstruction_geometry(first.geometry, frame.geometry)):
                    raise RuntimeError(
                        "camera reconstruction geometry changed inside one depth-fusion burst")
            candidate = depth_plane_check(frame.depth, T_work_camera, ecfg,
                                          unit_mm=frame.geometry.depth_unit_mm)
            if not candidate["agrees"]:
                if previous is not None and np.array_equal(previous, frame.depth):
                    frozen = True
                previous = frame.depth
                if retries >= attempts:
                    check = candidate
                    detail = (
                        "every grab returned a byte-identical depth buffer, so the camera's "
                        "depth stream is FROZEN - restart it with `py -3.10 "
                        "tools/jetson_deploy.py restart`, then measure again."
                        if frozen else
                        "the depth is changing but does not match this pose. Check that the "
                        "work frame still sits on the physical surface.")
                    raise RuntimeError(
                        f"the depth frame does not describe this pose: median depth "
                        f"{check['observed_depth_mm']:.0f} mm with the camera "
                        f"{check['camera_z_mm']:.0f} mm above the work plane (expected "
                        f"{check['accepted_range_mm'][0]:.0f}-"
                        f"{check['accepted_range_mm'][1]:.0f} mm), after {retries} retry(s). "
                        + detail)
                retries += 1
                ctx.log(f"depth said {candidate['observed_depth_mm']:.0f} mm with the camera "
                        f"{candidate['camera_z_mm']:.0f} mm above the work plane - discarding "
                        f"it and grabbing again ({retries}/{attempts})")
                continue
            frames.append(frame)
            previous = frame.depth

    raw_depths = np.stack([np.asarray(item.depth) for item in frames])
    fused_depth = _nonzero_median_depth(raw_depths)
    representative = frames[len(frames) // 2]
    frame = replace(representative, depth=fused_depth)
    check = depth_plane_check(frame.depth, T_work_camera, ecfg,
                              unit_mm=frame.geometry.depth_unit_mm)
    check.update({"retries": retries, "frozen_stream": frozen,
                  "fusion_frames": len(frames)})
    capture_ms = (time.perf_counter() - started) * 1000.0
    ok, jpeg = cv2.imencode(".jpg", frame.color)
    if ok:
        ctx.frame(jpeg.tobytes())
    fusion = {"method": "per-pixel nonzero median", "requested_frames": wanted,
              "captured_frames": len(frames), "raw_file": "depth-frames.npy",
              "timestamps": [float(item.timestamp) for item in frames]}
    return {"frame": frame, "depth_frames": raw_depths, "depth_fusion": fusion,
            "capture_ms": capture_ms, "depth_plane_check": check}


def _inspect_and_capture(services, ctx: JobContext, plan: CylinderPlan, layer, *,
                         inspection_name: str, start_joints, seed_pose, collisions: bool,
                         artifacts: list[str], near_mm: float | None = None) -> dict:
    """Move the camera to the derived pose, settle, read the pose, grab ONE frame."""
    moved = _move_to_inspection(
        services, ctx, plan, layer, inspection_name=inspection_name,
        start_joints=start_joints, seed_pose=seed_pose, collisions=collisions,
        artifacts=artifacts, near_mm=near_mm)
    return {**moved, **_capture_at_pose(services, ctx, moved["T_work_camera"])}


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


def add_return_timing(layer_dir, return_ms: float, *, whole_excursion: bool = True) -> dict | None:
    """Fold the trip back into the take's timings, once the arm is home.

    The archive is written before the return move -- a measurement must survive
    a failure on the way back -- so the closing half of the excursion is patched
    into the manifest afterwards. Nothing derived is touched.

    ``whole_excursion`` is False when several takes SHARED one trip out: the
    return is still a real return and stands as a sample of one, but no single
    take of that group cost a whole excursion, so none of them may claim an
    ``inspection_cycle_ms``. The paper's cycle figure is the price of leaving
    the path for one measurement, and averaging a shared trip into it would
    quietly divide that price by the number of frames taken while parked.
    """
    manifest_file = Path(layer_dir) / "manifest.json"
    if not manifest_file.is_file():
        return None
    payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    timings = (payload.setdefault("processing", {}).setdefault("timings_ms", {}))
    timings["return_ms"] = float(return_ms)
    if whole_excursion:
        timings["inspection_cycle_ms"] = float(sum(
            float(timings.get(key) or 0.0) for key in
            ("move_to_pose_ms", "settle_ms", "capture_ms", "total_ms", "return_ms")))
    manifest_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return timings


def side_capture_requirements(rdk: RdkIO, config) -> dict:
    """Are the two taught side-photo targets in the station?

    Reported, never enforced: the photo is a figure for the paper and the
    measurement is irreplaceable, so a station without these targets must still
    be able to measure.
    """
    names = [("side_capture_target", config.side_capture_target),
             ("side_capture_approach_target", config.side_capture_approach_target)]
    items = [{"role": role, "name": name, "type": "target",
              "present": bool(name) and rdk.item_exists_as(name, "target")}
             for role, name in names]
    return {"ready": all(item["present"] for item in items), "items": items,
            "missing": [item for item in items if not item["present"]]}


def capture_side_photo(services, ctx: JobContext, *, start_joints) -> dict:
    """One RGB photo of the stack from the side, via the taught approach target.

    The route is neutral -> approach -> side, and back side -> approach ->
    neutral. The approach target is not a nicety: the operator taught it because
    the direct joint move between the neutral pose and the side pose sweeps the
    arm through the things standing around the cell, and nothing in the station
    model knows they are there. So it is used in BOTH directions, and the return
    leg runs even when the capture failed -- an arm left out at the side pose is
    the worst outcome available here.

    Never raises. A missing target, a refused move or a dead camera returns a
    record saying so; the measurement it belongs to is already on disk and is
    not put at risk for a figure.
    """
    rdk: RdkIO = services.rdk
    ecfg = services.config.extrusion
    side, approach = ecfg.side_capture_target, ecfg.side_capture_approach_target
    record = {"captured": False, "target": side, "approach_target": approach,
              "excursion_ms": None, "error": None, "image_file": None}
    required = side_capture_requirements(rdk, ecfg)
    if not required["ready"]:
        missing = ", ".join(f"target {item['name']!r}" for item in required["missing"])
        record["error"] = f"side photo skipped: {missing} is not in the station"
        ctx.log(record["error"])
        return record
    def go(name: str) -> None:
        """Move to a taught target by its STORED JOINTS whenever it has them.

        ``move_j(name)`` is ``MoveJ(target_item)``, and for a CARTESIAN target
        RoboDK resolves that pose against the tool and frame active right now --
        not the ones it was taught with. By the time the side photo runs, the
        last take left ``Realsense`` + the work frame selected
        (``_move_to_inspection``), so a target taught under any other selection
        resolves somewhere else entirely and RoboDK reaches it on whatever IK
        branch that implies. Cell 2026-08-29: the excursion took 137.8 s against
        2.7 s for an inspection move, and the arm visibly went somewhere wrong.

        A stored joint vector has no such dependency -- it is the pose, on the
        branch it was taught on. Fall back to the item move only when the target
        carries no joints, and say so, because that move is the ambiguous one.
        """
        joints = rdk.target_joints(name)
        if joints is not None:
            rdk.move_j_joints(joints)
            return
        ctx.log(f"WARNING side photo: target {name!r} stores no joints; moving to it as a "
                "cartesian target, which depends on the active tool and frame")
        rdk.move_j(name)

    departed = time.perf_counter()
    reached = False
    try:
        ctx.log(f"side photo: {approach} -> {side}")
        go(approach)
        go(side)
        reached = True
        time.sleep(ecfg.side_capture_settle_s)
        with _camera_hold(services, "extrusion-side-photo"):
            frame = services.camera.grab(color_only=True, timeout=ecfg.grab_timeout_s)
        record["color"] = frame.color
        record["captured"] = True
        ok, jpeg = cv2.imencode(".jpg", frame.color)
        if ok:
            ctx.frame(jpeg.tobytes())
    except Exception as exc:
        record["error"] = f"side photo failed: {exc}"
        ctx.log(record["error"])
    finally:
        # Back the way we came, whatever happened on the way out.
        try:
            if reached:
                go(approach)
            rdk.move_j_joints(start_joints)
        except Exception as exc:
            ctx.log(f"WARNING side photo: could not retrace to the neutral pose: {exc}")
        record["excursion_ms"] = (time.perf_counter() - departed) * 1000.0
    return record


class RingMeasureJob:
    """Measure a hand-placed ring: inspect, capture, process, archive, return.

    Two counts shape what one press does, and they answer different questions.

    ``repeats``     frames taken with the arm PARKED at the inspection pose.
                    Their spread is the sensing chain's own repeatability, with
                    the robot's re-approach excluded by construction -- one trip
                    out, N frames, seconds apart.
    ``excursions``  complete trips out and back, each re-deriving and re-running
                    the inspection move. Their spread additionally contains the
                    arm's re-approach. This is what the noise floor measures,
                    and it is the expensive axis: a whole excursion per frame.

    Splitting them is what lets the run ask for repeatability without paying for
    it everywhere: the noise floor buys the expensive kind once (unattended),
    and every later condition takes the cheap kind, with the arm still.

    A batch banks each take as it lands, so a failure part-way through keeps
    what it already measured -- pressing again continues from there.
    """

    def __init__(self, services, plan: CylinderPlan, session: MeasureSession,
                 layer_index: int, *, annotation: dict | None = None,
                 check_collisions: bool = True,
                 close_range_tool_clear: bool = False,
                 repeats: int = 1, excursions: int = 1,
                 side_photo: bool | None = None):
        if not 1 <= layer_index <= len(plan.layers):
            raise ValueError(f"layer_index {layer_index} outside 1..{len(plan.layers)}")
        self.services = services
        self.plan = plan.model_copy(deep=True)
        self.session = session
        self.layer_index = int(layer_index)
        self.annotation = dict(annotation or {})
        self.check_collisions = bool(check_collisions)
        self.close_range_tool_clear = bool(close_range_tool_clear)
        self.repeats = max(1, int(repeats))
        self.excursions = max(1, int(excursions))
        self.side_photo = (services.config.extrusion.side_capture_enabled
                           if side_photo is None else bool(side_photo))
        self.results: list[dict] = []
        self.result: dict | None = None

    def __call__(self, ctx: JobContext) -> dict:
        services = self.services
        layer = self.plan.layers[self.layer_index - 1]
        inspection_name = _program_name(self.plan, self.layer_index, "MEASURE") + "_Inspect"
        archive = ExtrusionArchive(REPO_ROOT / "runs" / "extrusion")
        start_joints = _prepare_robot(services, ctx, self.plan, label="extrusion-measure")
        total = self.excursions * self.repeats
        if total > 1:
            ctx.log(f"layer {self.layer_index}: {self.excursions} excursion(s) x "
                    f"{self.repeats} frame(s) at the pose = {total} takes, unattended")
        for excursion in range(1, self.excursions + 1):
            ctx.check_cancel()
            self._one_excursion(ctx, layer=layer, inspection_name=inspection_name,
                                archive=archive, start_joints=start_joints,
                                excursion=excursion, total=total)
            # An invalid observation is useful evidence and is already archived,
            # but five more robot trips cannot make that same processing result
            # valid. Stop an unattended batch after the first failed gate so the
            # operator can inspect/reprocess it instead of spending cell time on
            # four more frames they cannot cite.
            if self.results and not self.results[-1].get("valid", False):
                remaining = self.excursions - excursion
                if remaining:
                    ctx.log(f"take {self.results[-1]['take']} is invalid; stopped before "
                            f"{remaining} remaining robot trip(s)")
                break
        stopped_early = len(self.results) < total
        invalid_batch = any(not item.get("valid", False) for item in self.results)
        # An invalid measurement is not a paper take. Do not add another robot
        # excursion merely to photograph a condition the validity gate rejected.
        side = (None if invalid_batch else
                self._side_photo(ctx, archive=archive, start_joints=start_joints))
        self.result = {"kind": "ring_measure", "mode": MODE,
                       "trial_id": self.session.trial_id,
                       "fingerprint": self.plan.fingerprint,
                       **self.results[-1],
                       # The batch as a whole, so a caller that asked for five
                       # takes can tell five happened from the result alone.
                       "takes_recorded": [r["take"] for r in self.results],
                       "excursions": self.excursions, "repeats": self.repeats,
                       "takes_requested": total, "stopped_early": stopped_early,
                       "invalid_batch": invalid_batch,
                       "side_view": side}
        return self.result

    # -- the paper's photo, once the layer's capture is complete -------------
    def _side_photo(self, ctx: JobContext, *, archive: ExtrusionArchive,
                    start_joints) -> dict | None:
        """One side-on RGB photo of the stack, attached to this press's last take.

        Once per press, not once per take: the ring has not moved between the
        frames of a capture, so a photo per frame would be the same photo. It
        runs LAST, after every measurement is archived and the arm is home, and
        it cannot fail the job -- capture_side_photo swallows its own errors and
        the archive write is guarded here.

        Its cost is deliberately kept out of the take's timings: the paper's
        "what one inspection costs" figure is the price of measuring a layer,
        and a photo for a figure is not part of that.
        """
        if not self.side_photo or not self.results:
            return None
        ctx.progress(len(self.results), len(self.results),
                     "side photo of the stack for the paper")
        record = capture_side_photo(self.services, ctx, start_joints=start_joints)
        color = record.pop("color", None)
        last = self.results[-1]
        try:
            entry = archive.write_side_view(Path(last["layer_dir"]),
                                            color=color, record=record)
        except Exception as exc:
            ctx.log(f"side photo not archived (the measurement stands): {exc}")
            return record
        # The operator's table reads the session, not the manifest.
        for row in (last, *(r for r in self.session.records
                            if r.get("layer_name") == last.get("layer_name"))):
            row["side_view"] = entry
        self.session.save()
        if entry.get("captured"):
            ctx.log(f"side photo archived beside take {last['take']} "
                    f"({record['excursion_ms']:.0f} ms, not counted in the cycle)")
        return entry

    # -- one trip out and back ---------------------------------------------
    def _one_excursion(self, ctx: JobContext, *, layer, inspection_name: str,
                       archive: ExtrusionArchive, start_joints, excursion: int,
                       total: int) -> None:
        """Go out, take ``repeats`` frames without moving, come home.

        The return home and the artifact cleanup are per-trip: a batch that dies
        on trip three must not leave the arm parked over the ring, and must not
        leave three generations of inspection target in the station.
        """
        services = self.services
        rdk: RdkIO = services.rdk
        ecfg = services.config.extrusion
        artifacts: list[str] = []
        current_program: str | None = None
        last_dir: Path | None = None
        # Several takes sharing one trip means no single take of them cost a
        # whole excursion -- see add_return_timing.
        shared = self.repeats > 1
        try:
            with _camera_hold(services, "extrusion-measure"):
                trip = (f" (trip {excursion} of {self.excursions})"
                        if self.excursions > 1 else "")
                ctx.progress(len(self.results), total,
                             f"layer {self.layer_index}: moving the camera{trip}")
                current_program = inspection_name
                moved = _move_to_inspection(
                    services, ctx, self.plan, layer, inspection_name=inspection_name,
                    start_joints=start_joints, seed_pose=self.session.last_pose,
                    collisions=self.check_collisions, artifacts=artifacts,
                    near_mm=(ecfg.measure_close_range_min_mm
                             if self.close_range_tool_clear else None))
                current_program = None
                for repeat in range(1, self.repeats + 1):
                    ctx.check_cancel()
                    last_dir = self._one_take(ctx, layer=layer, archive=archive,
                                              moved=moved, excursion=excursion,
                                              repeat=repeat, total=total)
            # Publication figures are deliberately NOT rendered in the live job.
            # The existing figure endpoint renders a requested PNG/PDF from this
            # archive on demand. Matplotlib took 90-108 s per take on the cell and,
            # when called here, kept the robot parked over the ring for all of it.
            ctx.progress(len(self.results), total, "returning to the start pose")
        finally:
            if current_program:
                try:
                    rdk.stop_program(current_program)
                except Exception:
                    pass
            came_home = time.perf_counter()
            try:
                rdk.move_j_joints(start_joints)
            except Exception:
                pass
            return_ms = (time.perf_counter() - came_home) * 1000.0
            if last_dir is not None:
                try:
                    complete = add_return_timing(last_dir, return_ms,
                                                 whole_excursion=not shared)
                    if complete is not None:
                        # The operator's table and the paper both read the cycle,
                        # so the session row cannot lag the manifest.
                        for record in self.session.records:
                            if record.get("layer_name") == Path(last_dir).name:
                                record["timings_ms"] = complete
                        self.session.save()
                        for record in self.results:
                            if record.get("layer_name") == Path(last_dir).name:
                                record["timings_ms"] = complete
                        cycle = complete.get("inspection_cycle_ms")
                        ctx.log(f"inspection excursion {cycle:.0f} ms "
                                f"(out {complete.get('move_to_pose_ms', 0):.0f}, back "
                                f"{return_ms:.0f})" if cycle is not None else
                                f"trip home {return_ms:.0f} ms; {self.repeats} frames "
                                "shared this excursion, so none of them is priced as one")
                except Exception as exc:
                    ctx.log(f"return timing not recorded (the measurement stands): {exc}")
            try:
                rdk.delete_items(list(dict.fromkeys(reversed(artifacts))))
            except Exception:
                pass

    # -- one fused observation, processed and archived -----------------------
    def _one_take(self, ctx: JobContext, *, layer, archive: ExtrusionArchive,
                  moved: dict, excursion: int, repeat: int, total: int) -> Path:
        services = self.services
        ecfg = services.config.extrusion
        take = self.session.next_take(self.layer_index)
        inspect = moved["inspect"]
        T_work_camera = moved["T_work_camera"]
        parked = (f" (frame {repeat} of {self.repeats}, arm parked)"
                  if self.repeats > 1 else "")
        ctx.progress(len(self.results), total,
                     f"layer {self.layer_index} take {take}: capturing{parked}")
        captured = _capture_at_pose(services, ctx, T_work_camera)
        frame, capture_ms = captured["frame"], captured["capture_ms"]
        ctx.progress(len(self.results), total, f"take {take}: processing the frame")
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
                        # Which trip out this frame belongs to, and where in it.
                        # Without this a reader cannot tell three frames of one
                        # excursion from three separate re-approaches -- and the
                        # difference between those two spreads is the finding.
                        "excursion_index": excursion,
                        "repeat_index": repeat,
                        "repeats_in_excursion": self.repeats,
                        "depth_fusion": captured["depth_fusion"],
                        "T_work_camera": np.asarray(T_work_camera, dtype=float).tolist(),
                        # The frame's own greeting: native depth intrinsics and
                        # the depth->colour extrinsic. Without it a reprocess or
                        # a figure has no way to know this take was captured
                        # unaligned, 0.1 mm (protocol 2) rather than the legacy
                        # aligned 1 mm convention -- see figures.geometry_for_take.
                        "camera_geometry": frame.geometry.to_dict()})
        floor = self.session.floor_profile(self.layer_index)
        camera_cfg = services.config.camera
        try:
            processed = measure_take(
                color=frame.color, depth=frame.depth, geometry=frame.geometry,
                T_work_camera=T_work_camera, K=camera_cfg.K, dist=camera_cfg.dist,
                plan=self.plan, layer=layer, config=ecfg, floor_profile=floor)
        except Exception as exc:
            # A failed measurement still archives its raw RGB-D: the operator
            # cannot re-place the ring exactly, so the frame is the only thing
            # that can be reprocessed later.
            manifest = LayerManifest(
                **base, processing={"valid": False, "error": str(exc),
                                    "timings_ms": {"capture_ms": capture_ms}},
                warnings=[str(exc)])
            failed_dir = archive.write_layer(
                manifest, nominal_xyz=nominal, commanded_xyz=nominal,
                color=frame.color, depth=frame.depth,
                depth_frames=captured["depth_frames"],
                report={"valid": False, "error": str(exc)})
            # A failure the operator cannot see is a failure they cannot
            # reprocess -- and the raw frame that would rescue it is already
            # on disk.
            self.session.record_take(
                layer_index=self.layer_index, take=take, measured_xyz=None,
                pose=inspect["pose"],
                summary=take_summary(failed_dir, manifest.model_dump(mode="json")))
            self.session.save()
            ctx.checkpoint("extrusion_take", {
                "trial_id": self.session.trial_id, "layer_index": self.layer_index,
                "take": take, "valid": False, "layer_name": failed_dir.name})
            raise RuntimeError(
                f"layer {self.layer_index} take {take} measurement invalid; "
                f"raw RGB-D archived: {exc}") from exc
        timings = processed.report["timings_ms"]
        timings["capture_ms"] = capture_ms
        timings["acquisition_to_path_ms"] = capture_ms + timings["total_ms"]
        # Only the frame that actually followed the move may claim the move. The
        # frames after it were taken with the arm already still, and charging
        # them a travel time nothing travelled is how a shared trip would make
        # the excursion look cheaper than it is.
        if repeat == 1:
            timings["move_to_pose_ms"] = moved["move_ms"]
            timings["settle_ms"] = moved["settle_ms"]
        processed.report["depth_plane_check"] = captured["depth_plane_check"]
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
            depth_frames=captured["depth_frames"],
            derived_images={"segmentation.png": processed.segmentation,
                            "skeleton.png": processed.skeleton,
                            "comparison.png": processed.comparison},
            report={**processed.report,
                    "metrics": processed.metrics.model_dump(mode="json")})
        summary = {"layer_index": self.layer_index, "take": take,
                   "layer_dir": str(layer_dir),
                   # The directory NAME as well as the path: the browser
                   # addresses figures by it and cannot use an absolute
                   # server path.
                   "layer_name": layer_dir.name, "annotation": self.annotation,
                   "metrics": processed.metrics.model_dump(mode="json"),
                   "geometry": (processed.geometry.model_dump(mode="json")
                                if processed.geometry else None),
                   "timings_ms": timings, "valid": processed.metrics.valid,
                   "timestamp": _utcnow()}
        self.session.record_take(layer_index=self.layer_index, take=take,
                                 measured_xyz=processed.measured_xyz,
                                 pose=inspect["pose"], summary=summary)
        self.session.save()
        self.results.append(summary)
        ctx.checkpoint("extrusion_take", {
            "trial_id": self.session.trial_id, "layer_index": self.layer_index,
            "take": take, "valid": bool(processed.metrics.valid),
            "layer_name": layer_dir.name})
        ctx.log(f"layer {self.layer_index} take {take}: offset "
                f"{processed.metrics.center_offset_norm_mm:.2f} mm, RMS "
                f"{processed.metrics.rms_mm:.2f} mm, "
                f"{timings['acquisition_to_path_ms']:.0f} ms acquisition->path")
        return layer_dir


class RingCharacterizeJob:
    """Measure ring 1 with no recipe assumption; the operator applies it to the recipe.

    The recipe has to come from the physical ring -- the operator has dried
    beads and no calipers -- so this runs before any measurement of a stack and
    its result seeds radius, bead, layer height and the cylinder centre.
    """

    def __init__(self, services, plan: CylinderPlan, session: MeasureSession, *,
                 check_collisions: bool = True,
                 close_range_tool_clear: bool = False):
        self.services = services
        self.plan = plan.model_copy(deep=True)
        self.session = session
        self.check_collisions = bool(check_collisions)
        self.close_range_tool_clear = bool(close_range_tool_clear)
        self.result: dict | None = None

    def __call__(self, ctx: JobContext) -> dict:
        services = self.services
        rdk: RdkIO = services.rdk
        ecfg = services.config.extrusion
        layer = self.plan.layers[0]                      # aim at the first layer's top
        inspection_name = _program_name(self.plan, 1, "CHARACTERIZE") + "_Inspect"
        archive = ExtrusionArchive(REPO_ROOT / "runs" / "extrusion")
        artifacts: list[str] = []
        current_program: str | None = None
        start_joints = _prepare_robot(services, ctx, self.plan, label="extrusion-characterize")
        try:
            with _camera_hold(services, "extrusion-characterize"):
                ctx.progress(1, 3, "moving the camera over the ring")
                current_program = inspection_name
                captured = _inspect_and_capture(
                    services, ctx, self.plan, layer, inspection_name=inspection_name,
                    start_joints=start_joints, seed_pose=self.session.last_pose,
                    collisions=self.check_collisions, artifacts=artifacts,
                    near_mm=(ecfg.measure_close_range_min_mm
                             if self.close_range_tool_clear else None))
                current_program = None
                frame = captured["frame"]
                ctx.progress(2, 3, "characterizing the ring")
                index = archive.next_characterization_index(self.session.trial_id)
                provenance = {**_provenance(services),
                              "T_work_camera": np.asarray(
                                  captured["T_work_camera"], dtype=float).tolist(),
                              # Without this a characterization directory -- exactly
                              # where the archived ring1_*.npz fixtures came from --
                              # records native, unaligned, 0.1 mm depth with nothing
                              # saying so (Task 9 review, Important 5).
                              "camera_geometry": frame.geometry.to_dict(),
                              "depth_fusion": captured["depth_fusion"]}
                camera_cfg = services.config.camera
                try:
                    found = characterize_ring(
                        color=frame.color, depth=frame.depth, geometry=frame.geometry,
                        T_work_camera=captured["T_work_camera"], K=camera_cfg.K,
                        dist=camera_cfg.dist,
                        search_center_mm=(float(self.plan.setup.center_x_mm),
                                          float(self.plan.setup.center_y_mm)),
                        work_frame=self.plan.setup.work_frame, config=ecfg,
                        inspection_tool=self.plan.setup.inspection_tool,
                        print_tool=self.plan.setup.print_tool)
                except Exception as exc:
                    failed_report = {
                        "kind": "characterization", "valid": False, "error": str(exc),
                        "capture_ms": captured["capture_ms"],
                        "inspection_pose": captured["inspect"]["pose"],
                        "search_center_mm": [self.plan.setup.center_x_mm,
                                             self.plan.setup.center_y_mm],
                        "depth_shape": list(np.asarray(frame.depth).shape),
                        "provenance": provenance,
                    }
                    capture_dir = archive.write_characterization(
                        self.session.trial_id, index, color=frame.color, depth=frame.depth,
                        depth_frames=captured["depth_frames"],
                        measured_xyz=np.empty((0, 3)), derived_images={},
                        report=failed_report)
                    if captured["inspect"]["pose"]:
                        self.session.last_pose = captured["inspect"]["pose"]
                    self.session.save()
                    raise RuntimeError(
                        f"ring characterization invalid; raw RGB-D archived: {capture_dir}: "
                        f"{exc}") from exc
                summary = {**found.summary(), "index": index, "timestamp": _utcnow(),
                           "capture_ms": captured["capture_ms"],
                           "inspection_pose": captured["inspect"]["pose"],
                           "search_center_mm": [self.plan.setup.center_x_mm,
                                                self.plan.setup.center_y_mm]}
                capture_dir = archive.write_characterization(
                    self.session.trial_id, index, color=frame.color, depth=frame.depth,
                    depth_frames=captured["depth_frames"],
                    measured_xyz=found.measured_xyz,
                    derived_images={"segmentation.png": found.segmentation,
                                    "skeleton.png": found.skeleton,
                                    "comparison.png": found.comparison},
                    report={**found.report, "summary": summary,
                            "provenance": provenance})
                summary["capture_dir"] = str(capture_dir)
                self.session.characterizations.append(summary)
                if captured["inspect"]["pose"]:
                    self.session.last_pose = captured["inspect"]["pose"]
                self.session.save()
                ctx.log(f"ring: radius {found.radius_mm:.1f} mm, bead {found.bead_width_mm:.1f} mm, "
                        f"height {found.top_z_min_mm:.1f}-{found.top_z_max_mm:.1f} mm "
                        f"(mean {found.top_z_mean_mm:.1f}), centre "
                        f"({found.center_mm[0]:.1f}, {found.center_mm[1]:.1f})")
            ctx.progress(3, 3, "returning to the start pose")
            self.result = {"kind": "ring_characterize", "mode": MODE,
                           "trial_id": self.session.trial_id,
                           "characterization": summary, "capture_dir": str(capture_dir)}
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


def _stat(values) -> dict:
    arr = np.array([float(v) for v in values if v is not None], dtype=float)
    return {"n": int(arr.size),
            "mean": float(arr.mean()) if arr.size else None,
            "sd": float(arr.std(ddof=1)) if arr.size > 1 else None,
            "min": float(arr.min()) if arr.size else None,
            "max": float(arr.max()) if arr.size else None}


# The relation a bodily translation MUST satisfy, and the band it is checked in.
# A ring of radius R shifted by d << R from the nominal centre has radial
# deviation d*cos(theta) about that centre, so max = d, mean|dev| = 2d/pi and
# RMS = d/sqrt(2). The handoff calls this the built-in sanity check: readings
# that do not obey it mean the chain measured something other than the shift.
#
# The tolerance is this cell's own error floor -- hand-eye board consistency
# 1.26 mm, work-plane RMS 1.39 mm -- rounded to 1.5 mm, plus a proportional term
# so a larger introduced offset is not held to an absolute band it never claimed.
SHIFT_CHECK_FLOOR_MM = 1.5
SHIFT_CHECK_FRACTION = 0.15


def pure_shift_expectation(introduced_norm_mm: float) -> dict:
    """Deviation triple a pure translation of ``d`` produces (first order in d/R)."""
    d = float(introduced_norm_mm)
    return {"mean_absolute_mm": 2.0 * d / math.pi,
            "rms_mm": d / math.sqrt(2.0),
            "maximum_mm": d}


CAPTURE_LABEL = {"parked": "arm parked", "re-approach": "re-approached",
                 "single": "one take"}


def capture_style(manifests: list[dict]) -> str:
    """Did these takes share one trip out, or did the arm re-approach for each?

    The distinction is the whole point of splitting the run's two repeat counts:
    frames taken with the arm PARKED measure the sensing chain alone, while
    takes that each had their own excursion also contain the robot's
    re-approach. Takes archived before that split carry no stamp, and each of
    them WAS its own press and so its own excursion -- which is what the default
    reads as.
    """
    if len(manifests) < 2:
        return "single"
    provenance = [m.get("provenance") or {} for m in manifests]
    parked = max((int(p.get("repeats_in_excursion") or 1) for p in provenance), default=1)
    trips = {int(p.get("excursion_index") or 0) for p in provenance}
    return "parked" if parked > 1 and len(trips) == 1 else "re-approach"


def centre_spread(manifests: list[dict]) -> dict:
    """How far the FITTED centres of one condition's takes scatter about their own mean.

    This is repeatability proper -- the chain asked the same question several
    times over an unchanged ring -- as opposed to centre_offset, which measures
    where the ring sits relative to the plan. Reported as an unbiased 2-D
    spread: sqrt(sum d^2 / (n-1)), the radial standard deviation.
    """
    centres = [m["metrics"]["measured_center_mm"] for m in manifests
               if (m.get("metrics") or {}).get("valid")
               and (m.get("metrics") or {}).get("measured_center_mm")]
    if len(centres) < 2:
        return {"n": len(centres), "rms_mm": None, "max_mm": None}
    arr = np.asarray(centres, dtype=float)
    deviations = np.linalg.norm(arr - arr.mean(axis=0), axis=1)
    n = len(centres)
    return {"n": n,
            "rms_mm": float(np.sqrt(float((deviations ** 2).sum()) / (n - 1))),
            "max_mm": float(deviations.max())}


def _pooled_spread(groups: list[list[dict]]) -> dict:
    """Pool several conditions' centre spreads into one estimate.

    Each condition scatters about its OWN mean -- the ring is somewhere
    different in each -- so the deviations pool, not the positions: the standard
    (n-1)-weighted pooled variance.
    """
    weight = 0.0
    total = 0.0
    n_takes = 0
    worst = None
    for items in groups:
        spread = centre_spread(items)
        if spread["rms_mm"] is None:
            continue
        degrees = spread["n"] - 1
        total += (spread["rms_mm"] ** 2) * degrees
        weight += degrees
        n_takes += spread["n"]
        worst = spread["max_mm"] if worst is None else max(worst, spread["max_mm"])
    return {"conditions": sum(1 for g in groups if centre_spread(g)["rms_mm"] is not None),
            "takes": n_takes,
            "rms_mm": float(np.sqrt(total / weight)) if weight else None,
            "max_mm": worst}


def shift_consistency(observed: dict, introduced_norm_mm: float, *,
                      floor_mm: float = SHIFT_CHECK_FLOOR_MM,
                      fraction: float = SHIFT_CHECK_FRACTION) -> "dict | None":
    """Do these deviations look like the translation the operator introduced?

    ``None`` when no offset was introduced -- there is no relation to check, and
    a zero-offset group must not be reported as passing one. Otherwise every
    disagreeing statistic is named, so a failure says WHICH number is wrong
    rather than only that something is.
    """
    d = float(introduced_norm_mm or 0.0)
    if d <= 0.0:
        return None
    expected = pure_shift_expectation(d)
    tolerance = max(float(floor_mm), float(fraction) * d)
    delta: dict[str, float] = {}
    disagreements: list[str] = []
    for key, want in expected.items():
        got = observed.get(key)
        if got is None:
            continue
        delta[key] = float(got) - want
        if abs(delta[key]) > tolerance:
            disagreements.append(key)
    return {"expected_mm": expected,
            "observed_mm": {key: (None if observed.get(key) is None else float(observed[key]))
                            for key in expected},
            "delta_mm": delta, "tolerance_mm": tolerance,
            "consistent": not disagreements, "disagreements": disagreements}


def introduced_offset_mm(manifest: dict) -> "tuple[float, float]":
    """The ground truth the operator typed before pressing Measure; absent = none."""
    offset = (manifest.get("annotation") or {}).get("introduced_offset_mm")
    if not offset or len(offset) != 2:
        return (0.0, 0.0)
    return (float(offset[0]), float(offset[1]))


def detection_error_mm(manifest: dict) -> "float | None":
    """How far the MEASURED centre offset lands from the one the operator typed.

    This is the number the paper actually claims: not where the ring sat, but
    how well the sensing-and-comparison chain recovered a displacement it was
    told about. ``None`` when the take predates the measured offset VECTOR --
    missing is not zero, and averaging it in would read as a perfect measurement.
    """
    measured = (manifest.get("metrics") or {}).get("center_offset_mm")
    if measured is None or len(measured) != 2:
        return None
    truth = introduced_offset_mm(manifest)
    return float(math.hypot(float(measured[0]) - truth[0], float(measured[1]) - truth[1]))


# The Run guide's "which way is +X" take: the ring deliberately moved an amount
# nobody typed. It is a zero-offset take on paper and a displaced ring in fact,
# so it can never stand in for where the ring sat before a shift.
AXIS_CHECK_PHASE = "axis check"


def _undisplaced_takes(manifests: list[dict], layer: int, *,
                       before_take: int | None = None) -> list[dict]:
    """Valid, zero-offset, centre-bearing takes of one layer, axis check excluded."""
    found = []
    for other in manifests:
        metrics = other.get("metrics") or {}
        phase = str((other.get("annotation") or {}).get("phase") or "").strip()
        if (int(other.get("layer_index") or 0) == layer
                and (before_take is None or int(other.get("take") or 1) < before_take)
                and introduced_offset_mm(other) == (0.0, 0.0)
                and phase != AXIS_CHECK_PHASE
                and metrics.get("valid") and metrics.get("measured_center_mm")):
            found.append(other)
    return found


def pre_shift_reference(manifest: dict, manifests: list[dict]) -> "dict | None":
    """The measurement THIS take's displacement is measured from.

    First choice, and the only one until 2026-08-29: the same ring's last valid
    zero-offset take before this one. Same layer, a lower take number, not the
    axis-check throwaway (a ring moved an untyped amount). A zero-offset take
    AFTER the shift is the ring put back, not where it was moved from, so only
    earlier takes qualify.

    Fallback, for a ring that arrives already displaced: the latest valid
    zero-offset take of the layer BENEATH it. The protocol's top ring is now
    PLACED off-centre rather than slid, so it has no undisplaced measurement of
    its own to pair with -- and it never will. Physically that is the right
    reference anyway: the rule measures how far this ring sits from the one it
    was stacked on, and centre(this) - centre(below) is exactly that. Scored
    against the plan centre instead, the whole stack's placement error would be
    charged to the chain.
    """
    layer = int(manifest.get("layer_index") or 0)
    take = int(manifest.get("take") or 1)
    same = _undisplaced_takes(manifests, layer, before_take=take)
    if same:
        return max(same, key=lambda m: int(m.get("take") or 1))
    below = _undisplaced_takes(manifests, layer - 1) if layer > 1 else []
    return max(below, key=lambda m: int(m.get("take") or 1), default=None)


def paired_detection(manifest: dict, manifests: list[dict]) -> "dict | None":
    """The introduced shift scored against the ring's OWN position before it moved.

    The steel rule measures the displacement from where the ring sat, so the
    chain is scored the same way: fitted centre after the shift minus the fitted
    centre of the last zero-offset take of the same layer, against the vector
    the operator typed. Scored against the plan centre instead, a top ring
    "placed true" by eye carries its placement error into what the paper calls
    the chain's error. ``None`` when nothing was introduced, the take is
    invalid, or no earlier zero-offset take of this layer exists to pair with.
    """
    truth = introduced_offset_mm(manifest)
    if not any(truth):
        return None
    metrics = manifest.get("metrics") or {}
    after = metrics.get("measured_center_mm")
    if not metrics.get("valid") or not after or len(after) != 2:
        return None
    reference = pre_shift_reference(manifest, manifests)
    if reference is None:
        return None
    before = reference["metrics"]["measured_center_mm"]
    shift = (float(after[0]) - float(before[0]), float(after[1]) - float(before[1]))
    reference_layer = int(reference.get("layer_index") or 0)
    return {"reference_take": int(reference.get("take") or 1),
            "reference_layer": reference_layer,
            # True when the ring arrived displaced and is measured against the
            # one it was stacked on, rather than against its own earlier self.
            "relative_to_layer_below": reference_layer != int(manifest.get("layer_index") or 0),
            "measured_shift_mm": [shift[0], shift[1]],
            "measured_shift_norm_mm": float(math.hypot(*shift)),
            "detection_error_mm": float(math.hypot(shift[0] - truth[0],
                                                   shift[1] - truth[1]))}


def _condition_name(manifest: dict) -> str:
    """Layer + the phase the operator recorded + the ground truth they typed.

    Five untouched takes (sensing noise floor) and three re-placed takes
    (placement repeatability) are different experiments that share an empty
    offset; pooling them hides one inside the other. The layer matters too --
    per-layer numbers are the evidence that measuring climbs with the stack.
    """
    annotation = manifest.get("annotation") or {}
    parts = [f"layer {int(manifest.get('layer_index', 0))}"]
    phase = str(annotation.get("phase") or "").strip()
    if phase:
        parts.append(phase)
    offset = annotation.get("introduced_offset_mm")
    if offset and any(float(v) for v in offset):
        parts.append(f"introduced offset ({offset[0]:g}, {offset[1]:g}) mm")
    elif not phase:
        parts.append("no introduced offset")
    return " - ".join(parts)


def live_timings(manifest: dict) -> dict:
    """The timings measured ON THE CELL for this take, if any.

    An offline reprocess produces a processing time for a run that never
    happened; requirement 3 is scan-to-feedback time on the real cell, so an
    offline take contributes only what stayed true -- the capture it was
    reprocessed from -- unless its original live timings were preserved.
    """
    processing = manifest.get("processing") or {}
    timings = dict(processing.get("timings_ms") or {})
    if not processing.get("offline_reprocess"):
        return timings
    preserved = dict(processing.get("live_timings_ms") or {})
    if preserved:
        return preserved
    return {"capture_ms": timings.get("capture_ms")}


def _fmt(stat: dict, digits: int = 2) -> str:
    if stat["mean"] is None:
        return "-"
    text = f"{stat['mean']:.{digits}f}"
    return text if stat["sd"] is None else f"{text} +/- {stat['sd']:.{digits}f}"


def paper_summary(root: Path, trial_id: str) -> dict:
    """Numbers the PFH short paper asks for, grouped by the operator's ground truth.

    The wording of the Markdown block matters: this is a controlled validation
    of the sensing-and-comparison chain against a KNOWN introduced offset on
    hand-placed dried beads. It is not the deposition deviation of a printed
    cylinder, and the summary must not let it be read as one.
    """
    trial_dir = Path(root) / trial_id
    trial_file = trial_dir / "trial.json"
    if not trial_file.is_file():
        raise FileNotFoundError(f"trial does not exist: {trial_id}")
    trial = json.loads(trial_file.read_text(encoding="utf-8"))
    takes = [json.loads(p.read_text(encoding="utf-8"))
             for p in sorted(trial_dir.glob("layer-*/manifest.json"))]
    groups: dict[str, list[dict]] = {}
    for manifest in takes:
        groups.setdefault(_condition_name(manifest), []).append(manifest)
    conditions = []
    for name, items in groups.items():
        # Only a take that produced a branch-free path measured anything. An
        # invalid take is still reported (takes vs valid) but never averaged:
        # its deviation numbers describe a reconstruction that failed.
        measured = [m for m in items if (m.get("metrics") or {}).get("valid")]
        metrics = [m["metrics"] for m in measured]
        introduced = introduced_offset_mm(items[0])
        introduced_norm = float(math.hypot(*introduced))
        deviation = {"mean_absolute_mm": _stat(x.get("mean_absolute_mm") for x in metrics),
                     "rms_mm": _stat(x.get("rms_mm") for x in metrics),
                     "maximum_mm": _stat(x.get("maximum_mm") for x in metrics)}
        paired = [p for p in (paired_detection(m, takes) for m in measured) if p]
        conditions.append({
            "condition": name, "takes": len(items),
            "valid": sum(1 for x in metrics if x.get("valid")),
            "introduced_offset_mm": list(introduced),
            "introduced_norm_mm": introduced_norm,
            "layer_index": int(items[0].get("layer_index", 0)),
            "phase": str((items[0].get("annotation") or {}).get("phase") or "") or None,
            "center_offset_norm_mm": _stat(x.get("center_offset_norm_mm") for x in metrics),
            "detection_error_mm": _stat(detection_error_mm(m) for m in measured),
            # The claim proper: the shift as the ring itself moved, not as it
            # sits relative to a plan centre it was never exactly placed on.
            "paired_shift_norm_mm": _stat(p["measured_shift_norm_mm"] for p in paired),
            "paired_detection_error_mm": _stat(p["detection_error_mm"] for p in paired),
            "paired_reference_takes": sorted({p["reference_take"] for p in paired}),
            "paired_reference_layers": sorted({p["reference_layer"] for p in paired}),
            # A ring that arrived displaced is scored against the ring beneath
            # it, not against an earlier self it never had.
            "paired_against_layer_below": bool(
                paired and all(p["relative_to_layer_below"] for p in paired)),
            # How this condition's takes were bought, and how tightly they
            # agreed with each other. Together across conditions these separate
            # the camera's repeatability from the robot's -- see `repeatability`.
            "capture": capture_style(items),
            "centre_spread": centre_spread(items),
            **deviation,
            "shape_rms_mm": _stat(x.get("shape_rms_mm") for x in metrics),
            "shift_consistency": shift_consistency(
                {key: stat["mean"] for key, stat in deviation.items()}, introduced_norm)})
    timings = [live_timings(m) for m in takes]
    timing = {key: _stat(t.get(key) for t in timings)
              for key in ("capture_ms", "total_ms", "acquisition_to_path_ms",
                          "move_to_pose_ms", "settle_ms", "return_ms",
                          "inspection_cycle_ms")}
    timing["offline_reprocessed_takes"] = sum(
        1 for m in takes if (m.get("processing") or {}).get("offline_reprocess"))
    geometry = [m["geometry"] for m in takes if m.get("geometry")]
    height = {key: _stat(g.get(key) for g in geometry)
              for key in ("height_mean_mm", "height_min_mm", "height_max_mm", "top_z_std_mm")}
    bead = {key: _stat(g.get(key) for g in geometry)
            for key in ("bead_width_mean_mm", "bead_width_min_mm", "bead_width_max_mm")}
    valid = sum(1 for m in takes if (m.get("metrics") or {}).get("valid"))
    # The two repeatabilities, kept apart. A condition measured with the arm
    # parked scatters by the camera alone; one where the arm went away and came
    # back for every take scatters by the camera AND the re-approach. Quoting a
    # single "repeatability" over both would attribute the robot's contribution
    # to the sensing chain, or hide it entirely, depending on which conditions
    # happened to dominate the pool.
    by_style: dict[str, list[list[dict]]] = {}
    for name, items in groups.items():
        by_style.setdefault(capture_style(items), []).append(items)
    repeatability = {"sensing": _pooled_spread(by_style.get("parked", [])),
                     "re_approach": _pooled_spread(by_style.get("re-approach", []))}
    session_file = trial_dir / "session.json"
    characterization = None
    if session_file.is_file():
        found = json.loads(session_file.read_text(encoding="utf-8")).get("characterizations") or []
        characterization = found[-1] if found else None

    headline = (f"Controlled validation of the sensing-and-comparison chain - trial "
                f"{trial_id}, hand-placed dried beads with a known introduced offset (not a "
                f"printed-cylinder deposition deviation). {valid}/{len(takes)} measurements "
                f"produced a valid, branch-free path.")
    # The claim itself, in words: not where the ring sat, but how well a
    # displacement the chain was TOLD about was recovered. Built once, as data,
    # so the Markdown block and the Word draft can never word it differently.
    prose: list[str] = []
    sensing, reapproach = repeatability["sensing"], repeatability["re_approach"]
    if sensing["rms_mm"] is not None:
        prose.append(
            f"Sensing repeatability, with the arm held at the inspection pose between frames: "
            f"{sensing['rms_mm']:.2f} mm RMS about each condition's own mean centre "
            f"(worst {sensing['max_mm']:.2f} mm) over {sensing['takes']} takes in "
            f"{sensing['conditions']} condition(s). This is the chain's own scatter, with the "
            "robot's re-approach excluded by construction.")
    if reapproach["rms_mm"] is not None:
        prose.append(
            f"Re-approach repeatability, the arm leaving and returning to the pose for every "
            f"take: {reapproach['rms_mm']:.2f} mm RMS (worst {reapproach['max_mm']:.2f} mm) "
            f"over {reapproach['takes']} takes in {reapproach['conditions']} condition(s). "
            "Measured the same way as the line above, so the difference between the two is "
            "what re-approaching the ring costs.")
    for c in conditions:
        if c["introduced_norm_mm"] > 0 and c["paired_detection_error_mm"]["n"]:
            refs = c["paired_reference_takes"]
            ref_layers = c["paired_reference_layers"]
            ref_text = (f"take {refs[0]}" if len(refs) == 1
                        else "takes " + ", ".join(str(r) for r in refs))
            ref_layer = (ref_layers[0] if len(ref_layers) == 1 else c["layer_index"])
            against = (f"the measured centre of the ring it was stacked on (layer "
                       f"{ref_layer} {ref_text})"
                       if c["paired_against_layer_below"] else
                       f"the ring's own last measured position before it was moved (layer "
                       f"{ref_layer} {ref_text})")
            prose.append(f"A {c['introduced_norm_mm']:.1f} mm introduced offset was recovered "
                         f"as {_fmt(c['paired_shift_norm_mm'])} mm over "
                         f"{c['paired_detection_error_mm']['n']} take(s), measured against "
                         f"{against}; detection error "
                         f"{_fmt(c['paired_detection_error_mm'])} mm (against the plan centre: "
                         f"{_fmt(c['detection_error_mm'])} mm).")
        elif c["introduced_norm_mm"] > 0:
            prose.append(f"A {c['introduced_norm_mm']:.1f} mm introduced offset was recovered "
                         f"as {_fmt(c['center_offset_norm_mm'])} mm over {c['takes']} take(s); "
                         f"detection error {_fmt(c['detection_error_mm'])} mm, scored against "
                         f"the plan centre only - no zero-offset take of layer "
                         f"{c['layer_index']} precedes it to pair with, so this includes the "
                         "ring's placement error.")
        else:
            prose.append(f"With no offset introduced the chain read a centre offset of "
                         f"{_fmt(c['center_offset_norm_mm'])} mm over {c['takes']} take(s) - "
                         f"the baseline this comparison is read against.")
        check = c["shift_consistency"]
        if check and not check["consistent"]:
            named = ", ".join(f"{key} {check['observed_mm'][key]:.2f} vs "
                              f"{check['expected_mm'][key]:.2f} expected"
                              for key in check["disagreements"])
            prose.append(f"WARNING - {c['condition']}: the deviation profile does not match a "
                         f"pure translation of {c['introduced_norm_mm']:.1f} mm "
                         f"(tolerance {check['tolerance_mm']:.2f} mm): {named}. "
                         "Investigate before citing this condition.")
    acq = timing["acquisition_to_path_ms"]
    if acq["n"]:
        prose.append(f"Across {acq['n']} processing cycles, RGB-D acquisition to reconstructed "
                     f"three-dimensional path took {_fmt(acq, 0)} ms "
                     f"(capture {_fmt(timing['capture_ms'], 0)} ms, processing "
                     f"{_fmt(timing['total_ms'], 0)} ms).")
    cycle = timing["inspection_cycle_ms"]
    if cycle["n"]:
        prose.append(f"Inspecting one layer cost {_fmt(cycle, 0)} ms of machine time end to end "
                     f"over {cycle['n']} excursion(s) - leaving the path, settling, capturing, "
                     f"reconstructing and returning (move out {_fmt(timing['move_to_pose_ms'], 0)} "
                     f"ms, return {_fmt(timing['return_ms'], 0)} ms). This is the cost of "
                     "inspecting between layers rather than during deposition.")
    if timing["offline_reprocessed_takes"]:
        prose.append(f"{timing['offline_reprocessed_takes']} take(s) were reprocessed offline "
                     "from their archived RGB-D frame; their geometry counts, and their "
                     "processing time is excluded from the cycle statistic above unless it "
                     "was measured live.")
    if geometry:
        prose.append(f"Layer height along the ring: mean {_fmt(height['height_mean_mm'], 1)} mm, "
                     f"min {_fmt(height['height_min_mm'], 1)} mm, max "
                     f"{_fmt(height['height_max_mm'], 1)} mm; bead footprint width "
                     f"{_fmt(bead['bead_width_mean_mm'], 1)} mm.")
    if characterization:
        prose.append(f"Ring characterized from its own scan: radius "
                     f"{characterization['radius_mm']:.1f} mm, bead "
                     f"{characterization['bead_width_mm']:.1f} mm, height "
                     f"{characterization['top_z_min_mm']:.1f}-{characterization['top_z_max_mm']:.1f} mm.")

    lines = [f"**{headline}**", "",
             "| Condition | n | how | centre spread (mm) | centre offset (mm) | "
             "detection error (mm) | paired detection error (mm) | "
             "mean abs dev (mm) | RMS (mm) | max (mm) | shape RMS (mm) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in conditions:
        spread = c["centre_spread"]["rms_mm"]
        lines.append(f"| {c['condition']} | {c['takes']} | {CAPTURE_LABEL[c['capture']]} | "
                     f"{'-' if spread is None else f'{spread:.2f}'} | "
                     f"{_fmt(c['center_offset_norm_mm'])} | "
                     f"{_fmt(c['detection_error_mm'])} | "
                     f"{_fmt(c['paired_detection_error_mm'])} | "
                     f"{_fmt(c['mean_absolute_mm'])} | {_fmt(c['rms_mm'])} | "
                     f"{_fmt(c['maximum_mm'])} | {_fmt(c['shape_rms_mm'])} |")
    for paragraph in prose:
        lines += ["", paragraph]
    return {"trial_id": trial_id, "mode": trial.get("mode", "LIVE_PRINT"),
            "takes": len(takes), "valid": valid, "conditions": conditions,
            "repeatability_mm": repeatability,
            "timing_ms": timing, "height_mm": height, "bead_width_mm": bead,
            "characterization": characterization, "headline": headline,
            "prose": prose, "manifests": takes, "markdown": "\n".join(lines)}
