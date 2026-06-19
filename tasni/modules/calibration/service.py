"""The calibration *job* — orchestration that ties the core services to the
pure library. Real-robot, RealSense-only, self-contained:

  Realsense tool (forced) -> move to NEUTRAL (must frame the board) -> auto-
  generate reachable viewpoints around that view -> visit each, detect ChArUco,
  record the true flange pose -> TSAI (+refine) -> quality report. The temp
  targets it creates are deleted afterwards (even on error/cancel), and it never
  auto-applies — the user applies the result after reviewing the metrics.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from ...core.geometry import invert_T
from ...core.jobrunner import JobContext
from ...core.logging import get_logger, new_run_dir
from ...core.rdk_io import RdkIO
from .charuco import CharucoTarget
from .handeye import CalibrationView, estimate_board_in_base, refine, solve_tsai
from .poses import generate_calibration_poses
from .quality import evaluate

log = get_logger("tasni.calibration")


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
    """Callable run by the JobRunner. Holds the solved transform for the
    separate apply step (so writing the tool pose is an explicit user action)."""

    def __init__(self, services, params: CalibrationParams):
        self.services = services
        self.params = params
        self.solved_X: np.ndarray | None = None   # T_flange_cam (cam2flange)
        self.tool_name: str = services.config.robodk.camera_tool
        self.result: CalibrationResult | None = None
        self._temp_targets: list[str] = []

    def __call__(self, ctx: JobContext) -> dict:
        cfg = self.services.config
        rdk: RdkIO = self.services.rdk
        cam = self.services.camera
        ccfg = cfg.calibration
        K, dist = cfg.camera.K, cfg.camera.dist
        board = CharucoTarget(cfg.board)
        neutral = cfg.robodk.neutral_target

        # --- validate the cell is set up the way calibration requires --------
        if not rdk.item_exists(self.tool_name):
            raise RuntimeError(
                f"tool {self.tool_name!r} not found — mount the RealSense camera "
                f"(3D model + a tool named {self.tool_name!r}) on the flange in RoboDK")
        if not rdk.item_exists(neutral):
            raise RuntimeError(f"target {neutral!r} not found — add a pose named "
                               f"{neutral!r} that frames the calibration board")

        applied_mode = rdk.apply_run_mode("run_robot")
        ctx.log(f"run mode: {applied_mode} (REAL ROBOT)")
        tool_pose = rdk.use_tool_and_frame(self.tool_name, neutral)
        ctx.log(f"tool: {self.tool_name}")

        stamp = time.strftime("%Y%m%d-%H%M%S")
        run_dir = new_run_dir("calibration", stamp)

        try:
            # --- NEUTRAL: frame the board, derive the working distance -------
            ctx.progress(0, ccfg.pose_count, f"moving to {neutral}")
            rdk.move_j(neutral)
            time.sleep(ccfg.settle_s)
            seed_frame = cam.grab()
            seed_det = board.detect(seed_frame.color, K, dist,
                                    min_corners=ccfg.min_charuco_corners)
            if seed_det is None:
                raise RuntimeError(
                    f"ChArUco board not visible at {neutral} — reposition the board "
                    f"or the {neutral} pose so the camera sees it, then retry")
            self._emit_frame(ctx, board.annotate(seed_frame.color, seed_det, K, dist,
                                                 f"{neutral} (board OK)"))
            look = float(np.linalg.norm(seed_det.t_target2cam))
            ctx.log(f"board visible at {neutral}; working distance ~{look:.0f} mm")
            seed_T = rdk.tcp_pose_T()

            # --- plan reachable viewpoints + create temp targets -------------
            poses = self._plan_poses(ctx, rdk, seed_T, look)

            # --- capture -----------------------------------------------------
            views, skipped = self._capture(ctx, rdk, cam, board, K, dist,
                                           tool_pose, poses, run_dir)

            holdout = (self.params.holdout_count if self.params.holdout_count is not None
                       else ccfg.holdout_count)
            do_refine = (self.params.refine if self.params.refine is not None
                         else ccfg.refine)
            if len(views) < holdout + 3:
                raise RuntimeError(
                    f"only {len(views)} usable views; need >= {holdout + 3}. "
                    f"Skipped (no board): {skipped}")

            train = views[:-holdout] if holdout else views
            val = views[-holdout:] if holdout else []
            ctx.progress(len(poses), len(poses),
                         f"solving (TSAI{', refine' if do_refine else ''})")
            X = solve_tsai(train)
            T_bt = estimate_board_in_base(train, X)
            if do_refine:
                X, T_bt = refine(train, X, T_bt, K, dist)
            report = evaluate(train, val, X, T_bt, K, dist, refined=do_refine)
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
            # Always clean up temp targets and park at NEUTRAL.
            if self._temp_targets:
                ctx.log(f"removing {len(self._temp_targets)} temporary targets")
                rdk.delete_items(self._temp_targets)
                self._temp_targets = []
            try:
                rdk.move_j(neutral)
            except Exception:
                pass

    # -- helpers ------------------------------------------------------------
    def _plan_poses(self, ctx, rdk, seed_T, look):
        ccfg = self.services.config.calibration
        candidates = generate_calibration_poses(
            seed_T, count=ccfg.pose_count, look_distance_mm=look,
            cone_half_angle_deg=ccfg.cone_half_angle_deg,
            roll_max_deg=ccfg.roll_max_deg, distance_jitter=ccfg.distance_jitter)
        ctx.log(f"checking reachability of {len(candidates)} candidate poses…")
        poses: list[tuple[str, np.ndarray]] = []
        for T in candidates:
            if len(poses) >= ccfg.pose_count:
                break
            if rdk.is_reachable(T):
                name = f"TasniCalib_{len(poses) + 1:02d}"
                rdk.add_target(name, T)
                self._temp_targets.append(name)
                poses.append((name, T))
        ctx.log(f"{len(poses)} reachable poses (of {ccfg.pose_count} wanted)")
        if len(poses) < 6:
            raise RuntimeError(
                f"only {len(poses)} reachable calibration poses found — move "
                f"{self.services.config.robodk.neutral_target} closer to the board "
                f"or into a more open part of the workspace")
        return poses

    def _capture(self, ctx, rdk, cam, board, K, dist, tool_pose, poses, run_dir):
        views, skipped = [], []
        total = len(poses)
        for i, (name, _T) in enumerate(poses):
            ctx.check_cancel()
            ctx.progress(i + 1, total, f"capturing {name}")
            rdk.move_j(name)
            time.sleep(self.services.config.calibration.settle_s)
            frame = cam.grab()
            det = board.detect(frame.color, K, dist,
                               min_corners=self.services.config.calibration.min_charuco_corners)
            if det is None:
                ctx.log(f"{name}: no board / too few corners — skipped")
                skipped.append(name)
                continue
            self._emit_frame(ctx, board.annotate(frame.color, det, K, dist, name),
                             run_dir / f"{name}.jpg")
            flange = rdk.tcp_pose_T() @ invert_T(tool_pose)   # true flange in frame
            views.append(CalibrationView(name, flange, det.R_target2cam,
                                         det.t_target2cam, det.corners, det.obj_points))
            ctx.log(f"{name}: {det.n_corners} corners")
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
