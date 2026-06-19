"""CalibrationModule — the WorkflowModule that plugs the calibration job into
the platform. Exposes a small REST surface and the UI panel; all the real work
lives in the pure library + the job in :mod:`service`.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ..base import ServiceContainer, WorkflowModule
from .service import CalibrationJob, CalibrationParams

if TYPE_CHECKING:  # pragma: no cover
    from fastapi import APIRouter

_HERE = Path(__file__).resolve().parent


class CalibrationModule(WorkflowModule):
    id = "calibration"
    title = "Calibration"
    description = "ChArUco eye-in-hand hand-eye calibration (TSAI) with quality metrics."

    def __init__(self, services: ServiceContainer):
        super().__init__(services)
        self._active_job: CalibrationJob | None = None

    # -- UI -----------------------------------------------------------------
    def panel_html(self) -> str:
        return (_HERE / "panel.html").read_text(encoding="utf-8")

    def panel_js(self) -> str:
        return (_HERE / "panel.js").read_text(encoding="utf-8")

    # -- REST ---------------------------------------------------------------
    def router(self) -> "APIRouter":
        from fastapi import APIRouter, HTTPException
        from pydantic import BaseModel

        router = APIRouter()
        services = self.services

        class RunBody(BaseModel):
            tool_name: str | None = None
            holdout_count: int | None = None
            refine: bool | None = None

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

        @router.post("/run")
        def run(body: RunBody) -> dict:
            if services.jobs.running:
                raise HTTPException(409, "a job is already running")
            params = CalibrationParams(
                tool_name=body.tool_name,
                holdout_count=body.holdout_count,
                refine=body.refine,
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
