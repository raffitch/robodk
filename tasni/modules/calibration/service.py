"""The calibration *job* — orchestration that ties the core services to the
pure library. Equivalent of the old ``AutoCalibrate.main()``, but: runs off the
UI thread, streams an annotated live preview, splits out validation poses,
reports quality, and never auto-applies the result (the user applies after
reviewing the metrics).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from ...core.jobrunner import JobContext
from ...core.logging import get_logger, new_run_dir
from ...core.rdk_io import RdkIO
from .charuco import CharucoTarget
from .handeye import CalibrationView, estimate_board_in_base, refine, solve_tsai
from .quality import evaluate

log = get_logger("tasni.calibration")


@dataclass
class CalibrationParams:
    tool_name: str | None = None        # tool to write the result into (apply step)
    holdout_count: int | None = None    # override config.calibration.holdout_count
    refine: bool | None = None          # override config.calibration.refine
    run_mode: str | None = None         # "run_robot" (real) or "simulate"; UI defaults real
    save_frames: bool = True


@dataclass
class CalibrationResult:
    report: dict
    summary: str
    run_dir: str
    tool_name: str | None
    n_captured: int
    n_skipped: list[str] = field(default_factory=list)


class CalibrationJob:
    """Callable run by the JobRunner. Holds the solved transform for the
    separate apply step (so applying to the tool is an explicit user action)."""

    def __init__(self, services, params: CalibrationParams):
        self.services = services
        self.params = params
        self.solved_X: np.ndarray | None = None   # T_gripper_cam (cam2gripper)
        self.tool_name: str | None = params.tool_name
        self.result: CalibrationResult | None = None

    def __call__(self, ctx: JobContext) -> dict:
        cfg = self.services.config
        rdk: RdkIO = self.services.rdk
        cam = self.services.camera
        K, dist = cfg.camera.K, cfg.camera.dist
        board = CharucoTarget(cfg.board)

        holdout = (self.params.holdout_count if self.params.holdout_count is not None
                   else cfg.calibration.holdout_count)
        do_refine = (self.params.refine if self.params.refine is not None
                     else cfg.calibration.refine)

        applied_mode = rdk.apply_run_mode(self.params.run_mode)
        ctx.log(f"run mode: {applied_mode}"
                + ("  (REAL ROBOT)" if applied_mode == "run_robot" else "  (simulation)"))
        targets = rdk.list_targets()
        if not targets:
            raise RuntimeError(f"no targets found with prefix "
                               f"{cfg.robodk.target_prefix!r}")
        ctx.log(f"{len(targets)} calibration poses")

        stamp = time.strftime("%Y%m%d-%H%M%S")
        run_dir = new_run_dir("calibration", stamp)

        views: list[CalibrationView] = []
        skipped: list[str] = []
        for i, name in enumerate(targets):
            ctx.check_cancel()
            ctx.progress(i, len(targets), f"moving to {name}")
            rdk.move_j(name)
            time.sleep(cfg.calibration.settle_s)
            frame = cam.grab()
            det = board.detect(frame.color, K, dist,
                               min_corners=cfg.calibration.min_charuco_corners)
            if det is None:
                ctx.log(f"{name}: no board / too few corners — skipped")
                skipped.append(name)
                continue
            annotated = board.annotate(frame.color, det, K, dist, name)
            ok, jpeg = cv2.imencode(".jpg", annotated)
            if ok:
                ctx.frame(jpeg.tobytes())
                if self.params.save_frames:
                    cv2.imwrite(str(run_dir / f"{name.replace(' ', '_')}.jpg"), annotated)
            views.append(CalibrationView(
                name, rdk.target_pose_T(name),
                det.R_target2cam, det.t_target2cam, det.corners, det.obj_points))
            ctx.log(f"{name}: {det.n_corners} corners")

        if len(views) < holdout + 3:
            raise RuntimeError(
                f"only {len(views)} usable views; need at least {holdout + 3} "
                f"(holdout {holdout} + 3 to solve). Skipped: {skipped}")

        # Hold out the last K poses for validation (deterministic + spatially
        # distinct from the start of the dome).
        train = views[:-holdout] if holdout else views
        val = views[-holdout:] if holdout else []
        ctx.progress(len(targets), len(targets),
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

        (run_dir / "report.json").write_text(
            json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        (run_dir / "summary.txt").write_text(summary, encoding="utf-8")

        self.result = CalibrationResult(
            report=report.to_dict(), summary=summary, run_dir=str(run_dir),
            tool_name=self.tool_name, n_captured=len(views), n_skipped=skipped)
        # Returned dict becomes the JobRunner result + the "result" event payload.
        return {
            "summary": summary,
            "report": report.to_dict(),
            "run_dir": str(run_dir),
            "tool_name": self.tool_name,
            "n_captured": len(views),
            "n_skipped": skipped,
            "can_apply": True,
        }

    def apply_to_tool(self) -> str:
        """Write the solved camera pose into the chosen tool (explicit step)."""
        if self.solved_X is None:
            raise RuntimeError("nothing solved yet")
        if not self.tool_name:
            raise RuntimeError("no tool selected to apply the calibration to")
        self.services.rdk.set_tool_pose(self.tool_name, self.solved_X)
        return self.tool_name
