"""The calibration *job* + the gate-gated target generation — the orchestration
that ties the core services to the pure library. Real-robot, RealSense-only.

Flow (no taught NEUTRAL):

  1. Live aiming gate (LivePreview): the operator jogs the robot until the board
     sits at the ideal distance + angle and all HUD lamps go green.
  2. ``generate_calibration_targets`` (gated): one authoritative grab confirms the
     gate, then reachable viewpoints are generated around the robot's *current*
     pose and left in the station as ``TasniCalib_*`` to inspect in RoboDK.
  3. ``CalibrationJob`` visits those targets on the real robot, detects ChArUco,
     records the true flange pose, solves TSAI (+refine) and reports quality.

The job never auto-applies (the user applies after reviewing the metrics) and
returns the robot to its starting joints when it finishes.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import base64

import cv2
import numpy as np

from ...core.aiming import GateThresholds, evaluate_gate
from ...core.events import JobEvent
from ...core.geometry import invert_T
from ...core.jobrunner import JobContext
from ...core.logging import get_logger, new_run_dir
from ...core.rdk_io import RdkIO
from .charuco import CharucoTarget
from .handeye import (CalibrationView, cross_validate, estimate_board_in_base,
                      refine, solve_best, solve_handeye)
from .intrinsics import verify_intrinsics
from .poses import generate_calibration_poses
from .quality import evaluate

log = get_logger("tasni.calibration")

TARGET_PREFIX = "TasniCalib_"


def gate_thresholds(ccfg) -> GateThresholds:
    """Build the gate thresholds from a CalibrationConfig (one source of truth so
    the live preview and the authoritative generate-grab gate identically)."""
    return GateThresholds(min_corners=ccfg.min_charuco_corners,
                          ideal_distance_mm=ccfg.ideal_distance_mm,
                          distance_tol_mm=ccfg.distance_tol_mm,
                          max_tilt_deg=ccfg.max_tilt_deg,
                          center_tol_mm=ccfg.center_tol_mm,
                          invert_x=ccfg.jog_invert_x,
                          invert_y=ccfg.jog_invert_y,
                          invert_z=ccfg.jog_invert_z)


def generate_calibration_targets(services) -> dict:
    """Gate-gated target creation (synchronous, no robot motion).

    Stops the live preview, takes one authoritative frame, and refuses unless the
    board is detected at the ideal distance + angle. On success the robot's
    current pose is the seed: reachable cone+roll poses are generated and written
    into the station as ``TasniCalib_*`` (prior ones cleared first). Returns the
    count created and the gate reading. Raises ``RuntimeError`` if not ready.
    """
    cfg = services.config
    ccfg = cfg.calibration
    rdk: RdkIO = services.rdk
    cam = services.camera
    tool_name = cfg.robodk.camera_tool
    K, dist = cfg.camera.K, cfg.camera.dist

    if not rdk.item_exists(tool_name):
        raise RuntimeError(
            f"tool {tool_name!r} not found — mount the RealSense camera "
            f"(3D model + a tool named {tool_name!r}) on the flange in RoboDK")

    # Free the camera (unicast) so our authoritative grab gets the frame.
    if services.live.running:
        services.live.stop()

    tool_pose = rdk.use_camera_tool(tool_name)
    board = CharucoTarget(cfg.board)
    frame = cam.grab(color_only=True)
    det = board.detect(frame.color, K, dist, min_corners=ccfg.min_charuco_corners)
    reading = evaluate_gate(det, K, frame.color.shape, gate_thresholds(ccfg),
                            board_center_mm=board.board_center)

    # Show the operator exactly the frame we judged.
    img = (board.annotate(frame.color, det, K, dist, "TARGET CREATION")
           if det is not None else frame.color)
    ok, jpeg = cv2.imencode(".jpg", img)
    if ok:
        services.bus.publish(JobEvent("frame",
            {"jpeg_b64": base64.b64encode(jpeg.tobytes()).decode("ascii")}))
    services.bus.publish(JobEvent("gate", {**reading.to_dict(), "live": False}))

    if not reading.ok:
        bad = [name for name, good in reading.gates.items() if not good]
        raise RuntimeError(
            "board not in the ideal band — fix " + ", ".join(bad)
            + f" (distance {reading.distance_mm and round(reading.distance_mm)} mm, "
            + f"tilt {reading.tilt_deg and round(reading.tilt_deg, 1)}°). "
            + "Jog the robot until all HUD lamps are green, then create targets.")

    seed_T = rdk.tcp_pose_T()
    look = float(reading.distance_mm)

    prior = rdk.list_targets(TARGET_PREFIX)
    if prior:
        rdk.delete_items(prior)

    candidates = generate_calibration_poses(
        seed_T, count=ccfg.pose_count, look_distance_mm=look,
        cone_half_angle_deg=ccfg.cone_half_angle_deg,
        roll_max_deg=ccfg.roll_max_deg, distance_jitter=ccfg.distance_jitter)
    created: list[str] = []
    for T in candidates:
        if len(created) >= ccfg.pose_count:
            break
        if rdk.is_reachable(T):
            name = f"{TARGET_PREFIX}{len(created) + 1:02d}"
            rdk.add_target(name, T)
            created.append(name)
    if len(created) < 6:
        raise RuntimeError(
            f"only {len(created)} reachable poses around this view — jog to a more "
            f"open part of the workspace (still framing the board) and retry")

    services.bus.publish(JobEvent("log",
        {"message": f"created {len(created)} calibration targets "
                    f"(working distance ~{look:.0f} mm) — inspect them in RoboDK"}))
    _ = tool_pose  # (kept active on the robot for the upcoming run)
    return {"created": len(created), "targets": created,
            "look_distance_mm": look, "gate": reading.to_dict()}


def _split_views(views: list, holdout: int, strategy: str, seed: int):
    """Train/validation split. 'shuffle' (seeded, unbiased) avoids the bias of
    'tail', where the deterministic-spiral pose order makes the last poses
    systematically the widest-angle views."""
    if not holdout:
        return list(views), []
    if strategy == "tail":
        return views[:-holdout], views[-holdout:]
    idx = list(range(len(views)))
    np.random.default_rng(seed).shuffle(idx)
    val_i = set(idx[:holdout])
    return ([v for i, v in enumerate(views) if i not in val_i],
            [views[i] for i in idx[:holdout]])


@dataclass
class TourPoseResult:
    """One pose's verdict on the simulated dry tour."""
    name: str
    reachable: bool
    collision: bool | None        # None = collisions not checked on this build
    ok: bool
    error: str | None = None

    def to_dict(self) -> dict:
        return {"name": self.name, "reachable": self.reachable,
                "collision": self.collision, "ok": self.ok, "error": self.error}


class SimTourJob:
    """Dry-run the generated ``TasniCalib_*`` targets in RoboDK **simulate** mode.

    Visits each target with the robot in simulation (no hardware motion), recording
    per-pose reachability + (if the build supports it) collision state, then returns
    to the start joints — the same return-to-start guarantee the real run makes. It
    is a **soft gate**: failures warn prominently but never block Run (the operator
    may still proceed once the cell is clear). Runs through the JobRunner, so it can
    never overlap a real calibration run. Restores the prior run mode on the way out
    so a dry tour can't leave the station silently in RUN_ROBOT.
    """

    def __init__(self, services):
        self.services = services
        self.tool_name: str = services.config.robodk.camera_tool

    def __call__(self, ctx: JobContext) -> dict:
        rdk: RdkIO = self.services.rdk

        if not rdk.item_exists(self.tool_name):
            raise RuntimeError(
                f"tool {self.tool_name!r} not found — mount the RealSense camera "
                f"(3D model + a tool named {self.tool_name!r}) on the flange in RoboDK")

        targets = rdk.list_targets(TARGET_PREFIX)
        if not targets:
            raise RuntimeError(
                "no TasniCalib_* targets to simulate — aim the camera until the "
                "gate is green and click Create targets first")

        prior_mode = rdk.current_run_mode()
        rdk.apply_run_mode("simulate")
        ctx.log(f"dry run (SIMULATE) — visiting {len(targets)} targets, no hardware motion")
        rdk.use_camera_tool(self.tool_name)
        collisions_on = rdk.set_collision_checking(True)
        try:
            start_joints = rdk.current_joints()
        except Exception:
            start_joints = None

        results: list[TourPoseResult] = []
        total = len(targets)
        try:
            for i, name in enumerate(targets):
                ctx.check_cancel()
                ctx.progress(i + 1, total, f"checking {name}")
                reachable = rdk.is_reachable(rdk.target_pose_T(name))
                collision: bool | None = None
                err: str | None = None
                if reachable:
                    try:
                        rdk.move_j(name)
                    except Exception as e:   # noqa: BLE001 - a sim move failure is a fail, not a crash
                        reachable, err = False, str(e)
                    if reachable and collisions_on:
                        n_col = rdk.collisions()
                        collision = None if n_col is None else bool(n_col)
                ok = reachable and not bool(collision)
                results.append(TourPoseResult(name, reachable, collision, ok, err))
                flag = "OK" if ok else ("UNREACHABLE" if not reachable else "COLLISION")
                ctx.log(f"{name}: {flag}")

            # Return-to-start (the guarantee the real run makes; verify it here too).
            returned = False
            if start_joints is not None:
                try:
                    rdk.move_j_joints(start_joints)
                    returned = True
                except Exception:
                    returned = False

            n_pass = sum(1 for r in results if r.ok)
            n_unreachable = sum(1 for r in results if not r.reachable)
            n_collision = sum(1 for r in results if r.collision)
            all_ok = n_pass == total and returned
            ctx.log(f"dry run complete: {n_pass}/{total} poses OK; "
                    f"return-to-start {'ok' if returned else 'FAILED'}"
                    + ("" if collisions_on else "; collisions not checked on this build"))
            return {
                "kind": "sim_tour",
                "total": total,
                "passed": n_pass,
                "unreachable": n_unreachable,
                "collisions": n_collision,
                "collisions_checked": collisions_on,
                "returned_to_start": returned,
                "all_ok": all_ok,
                "poses": [r.to_dict() for r in results],
            }
        finally:
            rdk.set_collision_checking(False)
            rdk.set_run_mode_raw(prior_mode)
            if start_joints is not None:
                try:
                    rdk.move_j_joints(start_joints)
                except Exception:
                    pass


@dataclass
class CalibrationParams:
    holdout_count: int | None = None    # override config.calibration.holdout_count
    refine: bool | None = None          # override config.calibration.refine
    save_frames: bool = True


@dataclass
class CalibrationResult:
    report: dict
    summary: str
    run_dir: str
    tool_name: str
    n_captured: int
    n_skipped: list[str] = field(default_factory=list)


class CalibrationJob:
    """Callable run by the JobRunner. Visits the pre-generated ``TasniCalib_*``
    targets, solves, and holds the result for the separate apply step (so writing
    the tool pose is an explicit user action)."""

    def __init__(self, services, params: CalibrationParams):
        self.services = services
        self.params = params
        self.solved_X: np.ndarray | None = None   # T_flange_cam (cam2flange)
        self.tool_name: str = services.config.robodk.camera_tool
        self.result: CalibrationResult | None = None

    def __call__(self, ctx: JobContext) -> dict:
        cfg = self.services.config
        rdk: RdkIO = self.services.rdk
        cam = self.services.camera
        ccfg = cfg.calibration
        K, dist = cfg.camera.K, cfg.camera.dist
        board = CharucoTarget(cfg.board)

        if not rdk.item_exists(self.tool_name):
            raise RuntimeError(
                f"tool {self.tool_name!r} not found — mount the RealSense camera "
                f"(3D model + a tool named {self.tool_name!r}) on the flange in RoboDK")

        targets = rdk.list_targets(TARGET_PREFIX)
        holdout = (self.params.holdout_count if self.params.holdout_count is not None
                   else ccfg.holdout_count)
        if len(targets) < holdout + 3:
            raise RuntimeError(
                f"only {len(targets)} {TARGET_PREFIX}* targets; need >= {holdout + 3}. "
                f"Aim the camera until the gate is green and click Create targets first.")

        # The camera (unicast) must be ours for the capture grabs.
        if self.services.live.running:
            self.services.live.stop()

        applied_mode = rdk.apply_run_mode("run_robot")
        ctx.log(f"run mode: {applied_mode} (REAL ROBOT)")
        tool_pose = rdk.use_camera_tool(self.tool_name)
        ctx.log(f"tool: {self.tool_name}; {len(targets)} targets to visit")

        try:
            start_joints = rdk.current_joints()
        except Exception:
            start_joints = None

        stamp = time.strftime("%Y%m%d-%H%M%S")
        run_dir = new_run_dir("calibration", stamp)

        try:
            views, skipped = self._capture(ctx, rdk, cam, board, K, dist,
                                           tool_pose, targets, run_dir)
            do_refine = (self.params.refine if self.params.refine is not None
                         else ccfg.refine)
            if len(views) < holdout + 3:
                raise RuntimeError(
                    f"only {len(views)} usable views; need >= {holdout + 3}. "
                    f"Skipped (no board): {skipped}")

            train, val = _split_views(views, holdout, ccfg.holdout_strategy,
                                      ccfg.split_seed)
            ctx.progress(len(targets), len(targets), "solving")
            if ccfg.solver_method == "best":
                X, method, ranking = solve_best(train, K, dist)
            else:
                method, ranking = ccfg.solver_method, None
                X = solve_handeye(train, method)
            T_bt = estimate_board_in_base(train, X)
            if do_refine:
                X, T_bt = refine(train, X, T_bt, K, dist)
            ctx.log(f"solver: {method}{' (+refine)' if do_refine else ''}"
                    + (f"; ranking " + ", ".join(f"{m} {r:.2f}px" for m, r in ranking)
                       if ranking else ""))
            xcheck = (verify_intrinsics(train, K, dist, cfg.camera.size)
                      if ccfg.verify_intrinsics else None)
            cv_rms = cross_validate(train, method, K, dist, ccfg.cross_val_folds)
            report = evaluate(train, val, X, T_bt, K, dist, refined=do_refine,
                              method=method, method_ranking=ranking,
                              intrinsics_check=xcheck, cross_val_rms_px=cv_rms)
            self.solved_X = X

            summary = report.summary()
            for line in summary.splitlines():
                ctx.log(line)
            (run_dir / "report.json").write_text(json.dumps(report.to_dict(), indent=2),
                                                 encoding="utf-8")
            (run_dir / "summary.txt").write_text(summary, encoding="utf-8")

            self.result = CalibrationResult(
                report=report.to_dict(), summary=summary, run_dir=str(run_dir),
                tool_name=self.tool_name, n_captured=len(views), n_skipped=skipped)
            return {
                "summary": summary, "report": report.to_dict(), "run_dir": str(run_dir),
                "tool_name": self.tool_name, "n_captured": len(views),
                "n_skipped": skipped, "can_apply": True,
            }
        finally:
            # Return to where the run started. The generated targets are left in
            # the station (the user created them deliberately); Clear poses removes
            # them.
            if start_joints is not None:
                try:
                    ctx.log("returning to start pose")
                    rdk.move_j_joints(start_joints)
                except Exception:
                    pass

    # -- helpers ------------------------------------------------------------
    def _grab_frames(self, cam, n: int):
        """Grab ``n`` frames as fresh as possible for median detection. Uses a
        held stream when the client supports it (no per-frame reconnect) and falls
        back to one-shot grabs; ``n == 1`` is a single grab (the old behaviour)."""
        if n <= 1:
            return [cam.grab(color_only=True)]
        stream = getattr(cam, "stream", None)
        if stream is not None:
            try:
                out = []
                with stream(color_only=True) as s:
                    for k in range(n):
                        out.append(s.read(drain=(k == 0)))
                return out
            except Exception as e:   # noqa: BLE001 - capture must go on
                log.warning("stream burst failed (%s); using one-shot grabs", e)
        return [cam.grab(color_only=True) for _ in range(n)]

    def _capture(self, ctx, rdk, cam, board, K, dist, tool_pose, targets, run_dir):
        views, skipped = [], []
        total = len(targets)
        ccfg = self.services.config.calibration
        for i, name in enumerate(targets):
            ctx.check_cancel()
            ctx.progress(i + 1, total, f"capturing {name}")
            rdk.move_j(name)
            time.sleep(ccfg.settle_s)
            images = [f.color for f in self._grab_frames(cam, ccfg.frames_per_pose)]
            det = board.detect_median(images, K, dist,
                                      min_corners=ccfg.min_charuco_corners)
            if det is None:
                ctx.log(f"{name}: no board / too few corners — skipped")
                skipped.append(name)
                continue
            rep = images[len(images) // 2]                 # a representative frame
            self._emit_frame(ctx, board.annotate(rep, det, K, dist, name),
                             run_dir / f"{name}.jpg")
            flange = rdk.tcp_pose_T() @ invert_T(tool_pose)   # true flange in frame
            views.append(CalibrationView(name, flange, det.R_target2cam,
                                         det.t_target2cam, det.corners, det.obj_points))
            n_frames = len(images)
            ctx.log(f"{name}: {det.n_corners} corners"
                    + (f" (median of {n_frames} frames)" if n_frames > 1 else ""))
        return views, skipped

    def _emit_frame(self, ctx, img, save_path=None) -> None:
        ok, jpeg = cv2.imencode(".jpg", img)
        if ok:
            ctx.frame(jpeg.tobytes())
            if save_path is not None and self.params.save_frames:
                cv2.imwrite(str(save_path), img)

    def apply_to_tool(self) -> str:
        """Write the solved camera pose into the Realsense tool (explicit step)."""
        if self.solved_X is None:
            raise RuntimeError("nothing solved yet")
        self.services.rdk.set_tool_pose(self.tool_name, self.solved_X)
        return self.tool_name
