"""CalibrationModule — the WorkflowModule that plugs the calibration job into
the platform. Exposes a small REST surface and the UI panel; all the real work
lives in the pure library + the job in :mod:`service`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from ..base import ServiceContainer, WorkflowModule
from .service import CalibrationJob, CalibrationParams

if TYPE_CHECKING:  # pragma: no cover
    from fastapi import APIRouter


# Request bodies must live at module scope: with `from __future__ import
# annotations` the handler annotations are strings, and FastAPI resolves them
# via get_type_hints in module globals — a class defined inside router() can't
# be found there (it would be misread as a query param).
class RunBody(BaseModel):
    # Tool is forced to the Realsense camera; motion is forced to the real robot.
    holdout_count: int | None = None
    refine: bool | None = None


class CalibrationModule(WorkflowModule):
    id = "calibration"
    title = "Calibration"
    description = "ChArUco eye-in-hand hand-eye calibration (TSAI) with quality metrics."
    icon = "🎯"
    order = 10

    def __init__(self, services: ServiceContainer):
        super().__init__(services)
        self._active_job: CalibrationJob | None = None

    # -- REST ---------------------------------------------------------------
    def router(self) -> "APIRouter":
        from fastapi import APIRouter, HTTPException, Response

        router = APIRouter()
        services = self.services

        @router.get("/config")
        def get_config() -> dict:
            c = services.config
            return {
                "robot": c.robodk.robot_name,
                "camera_tool": c.robodk.camera_tool,
                "neutral_target": c.robodk.neutral_target,
                "board": vars(c.board),
                "camera": {"ip": c.camera.ip, "port": c.camera.port,
                           "resolution": c.camera.resolution},
                "calibration": vars(c.calibration),
            }

        @router.get("/targets")
        def get_targets() -> dict:
            try:
                return {"targets": services.rdk.list_targets()}
            except Exception as e:  # RoboDK not running / no station
                raise HTTPException(503, f"RoboDK unavailable: {e}")

        @router.get("/tools")
        def get_tools() -> dict:
            try:
                return {"tools": services.rdk.list_tools()}
            except Exception as e:
                raise HTTPException(503, f"RoboDK unavailable: {e}")

        @router.post("/connect")
        def connect() -> dict:
            """Open the cell's station (loads Tasni.rdk if RoboDK came up empty)
            and verify it's set up for RealSense calibration. First load of the
            ~117 MB station is slow."""
            c = services.config.robodk
            try:
                robot_ok = services.rdk.robot().Valid()
                tool_ok = services.rdk.item_exists(c.camera_tool)
                neutral_ok = services.rdk.item_exists(c.neutral_target)
                ready = robot_ok and tool_ok and neutral_ok
                missing = [n for n, ok in (
                    (c.robot_name, robot_ok), (f"tool {c.camera_tool!r}", tool_ok),
                    (f"target {c.neutral_target!r}", neutral_ok)) if not ok]
                return {"connected": True, "ready": ready,
                        "robot": c.robot_name, "robot_valid": robot_ok,
                        "tool": c.camera_tool, "tool_present": tool_ok,
                        "neutral": c.neutral_target, "neutral_present": neutral_ok,
                        "missing": missing}
            except Exception as e:
                raise HTTPException(503, f"RoboDK unavailable: {e}")

        @router.post("/preview")
        def preview() -> dict:
            """Move to NEUTRAL and grab one frame so the user can confirm the
            board is framed before a run. Publishes the annotated frame to the
            live preview; returns whether the board was detected."""
            if services.jobs.running:
                raise HTTPException(409, "a calibration run is in progress")
            import base64

            import cv2

            from ...core.events import JobEvent
            from .charuco import CharucoTarget
            c = services.config
            neutral = c.robodk.neutral_target
            try:
                if not services.rdk.item_exists(neutral):
                    raise HTTPException(400, f"target {neutral!r} not found in the station")
                services.rdk.apply_run_mode("run_robot")
                services.rdk.use_tool_and_frame(c.robodk.camera_tool, neutral)
                services.bus.publish(JobEvent("log", {"message": f"preview: moving to {neutral}"}))
                services.rdk.move_j(neutral)
                frame = services.camera.grab()
                board = CharucoTarget(c.board)
                det = board.detect(frame.color, c.camera.K, c.camera.dist,
                                   min_corners=c.calibration.min_charuco_corners)
                img = (board.annotate(frame.color, det, c.camera.K, c.camera.dist,
                                      f"{neutral} (board OK)")
                       if det is not None else frame.color)
                ok, jpeg = cv2.imencode(".jpg", img)
                if ok:
                    services.bus.publish(JobEvent("frame",
                        {"jpeg_b64": base64.b64encode(jpeg.tobytes()).decode("ascii")}))
                return {"target": neutral, "detected": det is not None,
                        "n_corners": det.n_corners if det is not None else 0}
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(503, f"preview failed: {e}")

        # -- calibration board (print-it-yourself) --------------------------
        @router.get("/board/spec")
        def board_spec(page: str = "A4") -> dict:
            from .board_pdf import board_spec as _spec
            return _spec(services.config.board, page).to_dict()

        @router.get("/board.png")
        def board_png():
            from .board_pdf import render_png
            return Response(render_png(services.config.board),
                            media_type="image/png",
                            headers={"Cache-Control": "no-store"})

        @router.get("/board.pdf")
        def board_pdf(page: str = "A4", download: bool = False):
            from .board_pdf import render_pdf
            pdf, spec = render_pdf(services.config.board, page)
            disp = "attachment" if download else "inline"
            fname = f"charuco_{spec.squares_x}x{spec.squares_y}_{spec.square_size_mm}mm_{page}.pdf"
            return Response(pdf, media_type="application/pdf",
                            headers={"Content-Disposition": f'{disp}; filename="{fname}"'})

        @router.post("/run")
        def run(body: RunBody) -> dict:
            if services.jobs.running:
                raise HTTPException(409, "a job is already running")
            params = CalibrationParams(holdout_count=body.holdout_count,
                                       refine=body.refine, mode="calibrate")
            self._active_job = CalibrationJob(services, params)
            services.jobs.start(self._active_job, name="calibration")
            return {"status": "started"}

        @router.post("/poses/preview")
        def poses_preview() -> dict:
            """Generate the calibration poses and leave them in RoboDK as
            TasniCalib_* targets to inspect — moves to NEUTRAL but does NOT
            capture or solve."""
            if services.jobs.running:
                raise HTTPException(409, "a job is already running")
            self._active_job = CalibrationJob(services, CalibrationParams(mode="preview"))
            services.jobs.start(self._active_job, name="pose-preview")
            return {"status": "started"}

        @router.post("/poses/clear")
        def poses_clear() -> dict:
            """Delete the generated TasniCalib_* targets from the station."""
            try:
                existing = services.rdk.list_targets("TasniCalib_")
                services.rdk.delete_items(existing)
                return {"cleared": len(existing)}
            except Exception as e:
                raise HTTPException(503, f"RoboDK unavailable: {e}")

        @router.post("/cancel")
        def cancel() -> dict:
            services.jobs.cancel()
            return {"status": "cancelling"}

        @router.post("/apply")
        def apply() -> dict:
            if self._active_job is None or self._active_job.solved_X is None:
                raise HTTPException(400, "no solved calibration to apply")
            try:
                tool = self._active_job.apply_to_tool()
            except Exception as e:
                raise HTTPException(400, str(e))
            return {"status": "applied", "tool": tool}

        @router.get("/status")
        def status() -> dict:
            return {
                "status": services.jobs.status,
                "running": services.jobs.running,
                "result": services.jobs.result,
                "error": services.jobs.error,
            }

        return router
