"""CalibrationModule — the WorkflowModule that plugs the calibration job into
the platform. Exposes a small REST surface and the UI panel; all the real work
lives in the pure library + the job in :mod:`service`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from ..base import ServiceContainer, WorkflowModule
from .service import (
    CalibrationJob, CalibrationParams, SimTourJob, TARGET_PREFIX,
    apply_calibration, gate_thresholds, generate_calibration_targets)

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


class ApplyBody(BaseModel):
    # Apply the in-memory last run by default; pass a run_id (a TasniCalib run
    # stamp) to apply a past run loaded from disk — survives a server restart.
    run_id: str | None = None


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
            cc = c.calibration
            return {
                "robot": c.robodk.robot_name,
                "camera_tool": c.robodk.camera_tool,
                "board": c.board.model_dump(),
                "camera": {"ip": c.camera.ip, "port": c.camera.port,
                           "resolution": c.camera.resolution},
                "calibration": cc.model_dump(),
                "gate": {"ideal_distance_mm": cc.ideal_distance_mm,
                         "distance_tol_mm": cc.distance_tol_mm,
                         "max_tilt_deg": cc.max_tilt_deg},
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
                ready = robot_ok and tool_ok
                missing = [n for n, ok in (
                    (c.robot_name, robot_ok), (f"tool {c.camera_tool!r}", tool_ok))
                    if not ok]
                return {"connected": True, "ready": ready,
                        "robot": c.robot_name, "robot_valid": robot_ok,
                        "tool": c.camera_tool, "tool_present": tool_ok,
                        "missing": missing}
            except Exception as e:
                raise HTTPException(503, f"RoboDK unavailable: {e}")

        @router.post("/live/start")
        def live_start() -> dict:
            """Start the live aiming gate: stream annotated frames + gate readings
            over the WebSocket so the operator can jog the board into the ideal
            distance/angle band. Camera-only (no robot motion)."""
            if services.jobs.running:
                raise HTTPException(409, "a calibration run is in progress")
            if services.live.running:
                return {"status": "already running"}

            import cv2

            from ...core.aiming import evaluate_gate
            from .charuco import CharucoTarget
            c = services.config
            cc = c.calibration
            K, dist = c.camera.K, c.camera.dist
            board = CharucoTarget(c.board)
            th = gate_thresholds(cc)
            # Detection runs on the full-res frame; the transported image is
            # downscaled (the SVG HUD is drawn client-side, so this costs no
            # overlay sharpness) to keep the WebSocket light.
            PREVIEW_W = 960
            enc = [cv2.IMWRITE_JPEG_QUALITY, 75]

            def analyze(frame):
                det = board.detect(frame.color, K, dist, min_corners=cc.min_charuco_corners)
                reading = evaluate_gate(det, K, frame.color.shape, th,
                                        board_center_mm=board.board_center)
                # corners + axes confirm detection; the HUD draws the status text.
                img = (board.annotate(frame.color, det, K, dist, "")
                       if det is not None else frame.color)
                h, w = img.shape[:2]
                if w > PREVIEW_W:
                    img = cv2.resize(img, (PREVIEW_W, int(h * PREVIEW_W / w)),
                                     interpolation=cv2.INTER_AREA)
                ok, jpeg = cv2.imencode(".jpg", img, enc)
                return (jpeg.tobytes() if ok else b""), reading.to_dict()

            from ...core.camera_lease import CameraBusy
            try:
                services.live.start(analyze, fps=cc.preview_fps,
                                    timeout_s=cc.preview_timeout_s, color_only=True)
            except CameraBusy as e:
                raise HTTPException(409, str(e))
            return {"status": "started"}

        @router.post("/live/stop")
        def live_stop() -> dict:
            services.live.stop()
            return {"status": "stopped"}

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

        @router.post("/poses/generate")
        def poses_generate() -> dict:
            """Gate-gated target creation: confirm the board is in the ideal band
            (one authoritative grab), then generate reachable poses around the
            robot's CURRENT pose and leave them as TasniCalib_* to inspect. Refuses
            with 400 if the gate isn't green. No robot motion."""
            if services.jobs.running:
                raise HTTPException(409, "a job is already running")
            try:
                return generate_calibration_targets(services)
            except RuntimeError as e:
                raise HTTPException(400, str(e))
            except Exception as e:
                raise HTTPException(503, f"RoboDK/camera unavailable: {e}")

        @router.post("/poses/simulate")
        def poses_simulate() -> dict:
            """Dry-run the generated targets in RoboDK SIMULATE mode (no hardware):
            per-pose reachability + collision + return-to-start. A soft safety gate
            before the real Run; progress streams over the WebSocket and the verdict
            arrives as the job 'result' (name=sim_tour). No real-robot motion."""
            if services.jobs.running:
                raise HTTPException(409, "a job is already running")
            if len(services.rdk.list_targets(TARGET_PREFIX)) == 0:
                raise HTTPException(400, "no calibration targets to simulate — aim the "
                                    "camera until the gate is green and Create targets first")
            services.live.stop()    # free the camera thread; the dry tour owns the robot
            services.jobs.start(SimTourJob(services), name="sim_tour")
            return {"status": "started"}

        @router.post("/run")
        def run(body: RunBody) -> dict:
            if services.jobs.running:
                raise HTTPException(409, "a job is already running")
            if len(services.rdk.list_targets(TARGET_PREFIX)) == 0:
                raise HTTPException(400, "no calibration targets — aim the camera "
                                    "until the gate is green and Create targets first")
            services.live.stop()    # release the camera for the capture grabs
            params = CalibrationParams(holdout_count=body.holdout_count, refine=body.refine)
            self._active_job = CalibrationJob(services, params)
            services.jobs.start(self._active_job, name="calibration")
            return {"status": "started"}

        @router.post("/poses/clear")
        def poses_clear() -> dict:
            """Delete the generated TasniCalib_* targets from the station."""
            try:
                existing = services.rdk.list_targets(TARGET_PREFIX)
                services.rdk.delete_items(existing)
                return {"cleared": len(existing)}
            except Exception as e:
                raise HTTPException(503, f"RoboDK unavailable: {e}")

        @router.post("/cancel")
        def cancel() -> dict:
            services.jobs.cancel()
            return {"status": "cancelling"}

        @router.post("/apply")
        def apply(body: ApplyBody) -> dict:
            """Write the solved camera pose into the Realsense tool. With no
            ``run_id`` this applies the in-memory last run (the fast path); with a
            ``run_id`` it loads that past run from disk (survives a restart). Either
            way it records runs/calibration/active.json for the Dashboard."""
            from ...core.runs import RunNotFound

            if body.run_id is None and (
                    self._active_job is None or self._active_job.solved_X is None):
                raise HTTPException(400, "no solved calibration to apply — run a "
                                    "calibration first, or pass a run_id")
            try:
                return apply_calibration(services, job=self._active_job,
                                         run_id=body.run_id)
            except RunNotFound as e:
                raise HTTPException(404, str(e))
            except (RuntimeError, ValueError, KeyError) as e:
                raise HTTPException(400, str(e))
            except Exception as e:
                raise HTTPException(503, f"RoboDK unavailable: {e}")

        @router.get("/status")
        def status() -> dict:
            return {
                "status": services.jobs.status,
                "running": services.jobs.running,
                "result": services.jobs.result,
                "error": services.jobs.error,
            }

        return router
