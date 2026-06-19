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
    tool_name: str | None = None
    holdout_count: int | None = None
    refine: bool | None = None
    run_mode: str | None = None        # "run_robot" (real) or "simulate"


class BoardBody(BaseModel):
    page: str = "A4"


class PreviewBody(BaseModel):
    run_mode: str | None = None


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
                "run_mode": c.robodk.run_mode,
                "target_prefix": c.robodk.target_prefix,
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
            and report what's there. First load of the ~117 MB station is slow."""
            try:
                targets = services.rdk.list_targets()
                tools = services.rdk.list_tools()
                robot_ok = services.rdk.robot().Valid()
                return {"connected": True, "robot": services.config.robodk.robot_name,
                        "robot_valid": robot_ok, "n_targets": len(targets),
                        "targets": targets, "tools": tools}
            except Exception as e:
                raise HTTPException(503, f"RoboDK unavailable: {e}")

        @router.post("/preview")
        def preview(body: PreviewBody) -> dict:
            """Move to the first calibration pose and grab one frame so the user
            can confirm the board is in view. Publishes the annotated frame to the
            live preview; returns whether the board was detected."""
            if services.jobs.running:
                raise HTTPException(409, "a calibration run is in progress")
            import base64

            import cv2

            from ...core.events import JobEvent
            from .charuco import CharucoTarget
            try:
                targets = services.rdk.list_targets()
                if not targets:
                    raise HTTPException(400, "no calibration targets in the station")
                mode = services.rdk.apply_run_mode(body.run_mode)
                first = targets[0]
                services.bus.publish(JobEvent("log",
                    {"message": f"preview: moving to {first} ({mode})"}))
                services.rdk.move_j(first)
                frame = services.camera.grab()
                board = CharucoTarget(services.config.board)
                det = board.detect(frame.color, services.config.camera.K,
                                   services.config.camera.dist,
                                   min_corners=services.config.calibration.min_charuco_corners)
                img = (board.annotate(frame.color, det, services.config.camera.K,
                                      services.config.camera.dist, f"{first} (preview)")
                       if det is not None else frame.color)
                ok, jpeg = cv2.imencode(".jpg", img)
                if ok:
                    services.bus.publish(JobEvent("frame",
                        {"jpeg_b64": base64.b64encode(jpeg.tobytes()).decode("ascii")}))
                return {"target": first, "detected": det is not None,
                        "n_corners": det.n_corners if det is not None else 0}
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(503, f"preview failed: {e}")

        # -- calibration board (print-it-yourself) --------------------------
        @router.get("/board/spec")
        def board_spec(page: str = "A4") -> dict:
            from .board_pdf import compute_spec
            spec = compute_spec(services.config.board, page)
            cur = services.config.board
            spec_d = spec.to_dict()
            # Does the live detection config already match this printed size?
            spec_d["matches_config"] = (
                abs(cur.square_size_mm - spec.square_size_mm) < 1e-6
                and abs(cur.marker_size_mm - spec.marker_size_mm) < 1e-6)
            spec_d["pages"] = ["A4", "A3", "Letter"]
            return spec_d

        @router.get("/board.pdf")
        def board_pdf(page: str = "A4", download: bool = False):
            from .board_pdf import render_pdf
            pdf, spec = render_pdf(services.config.board, page)
            disp = "attachment" if download else "inline"
            fname = f"charuco_{spec.squares_x}x{spec.squares_y}_{spec.square_size_mm}mm_{page}.pdf"
            return Response(pdf, media_type="application/pdf",
                            headers={"Content-Disposition": f'{disp}; filename="{fname}"'})

        @router.post("/board/use")
        def board_use(body: BoardBody) -> dict:
            """Sync the printed board's dimensions into the calibration config
            (in memory + persisted) so detection matches what was printed."""
            from .board_pdf import compute_spec
            from ...core.config import save_overrides
            spec = compute_spec(services.config.board, body.page)
            b = services.config.board
            b.square_size_mm = spec.square_size_mm
            b.marker_size_mm = spec.marker_size_mm
            save_overrides({"board": {"square_size_mm": b.square_size_mm,
                                      "marker_size_mm": b.marker_size_mm}})
            return {"applied": True, "board": vars(b)}

        @router.post("/run")
        def run(body: RunBody) -> dict:
            if services.jobs.running:
                raise HTTPException(409, "a job is already running")
            params = CalibrationParams(
                tool_name=body.tool_name,
                holdout_count=body.holdout_count,
                refine=body.refine,
                run_mode=body.run_mode,
            )
            self._active_job = CalibrationJob(services, params)
            services.jobs.start(self._active_job, name="calibration")
            return {"status": "started"}

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
