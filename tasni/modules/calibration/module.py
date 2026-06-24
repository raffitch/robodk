"""CalibrationModule — the WorkflowModule that plugs the calibration job into
the platform. Exposes a small REST surface and the UI panel; all the real work
lives in the pure library + the job in :mod:`service`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from ...core.rdk_io import link_real_robot
from ..base import ServiceContainer, WorkflowModule
from .service import (
    BOARD_KEEPOUT_NAME, CalibrationJob, CalibrationParams, SimTourJob, TARGET_PREFIX,
    apply_calibration, apply_intrinsics, dry_tour_required, gate_thresholds,
    generate_calibration_targets)

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


class IntrSolveBody(BaseModel):
    # Fix k3 (the high-order radial term) to 0 by default — the D4xx RGB lens is
    # low-distortion and a free k3 overfits a limited image region.
    fix_k3: bool = True


class CalibrationModule(WorkflowModule):
    id = "calibration"
    title = "Calibration"
    description = "ChArUco eye-in-hand hand-eye calibration (TSAI) with quality metrics."
    icon = "🎯"
    order = 10

    def __init__(self, services: ServiceContainer):
        super().__init__(services)
        self._active_job: CalibrationJob | None = None
        # Dedicated RGB intrinsic-calibration session (camera-only; accumulates
        # auto-captured ChArUco views across the frame). Lazily created on first use.
        self._intr = None

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

        @router.get("/collision/status")
        def collision_status() -> dict:
            """Confirm RoboDK is actually evaluating collisions at the current pose
            (no robot motion). ``available: false`` means no collision map is set up,
            so target generation can't filter colliding poses. Also force-enables and
            reports the mounted-tool↔arm pairs (``guarded_tools``) RoboDK omits by
            default — the spindle/camera hitting the arm — so the chip reflects what
            Create targets will actually filter."""
            cc = services.config.calibration
            try:
                return services.rdk.collision_status(
                    ensure_pairs=cc.collision_self_pairs,
                    skip_trailing=cc.collision_skip_wrist_links)
            except Exception as e:
                raise HTTPException(503, f"RoboDK unavailable: {e}")

        @router.post("/connect")
        def connect() -> dict:
            """Open the cell's station (loads Tasni.rdk if RoboDK came up empty)
            and report ready only once the robot is actually detected.

            The first load of the ~117 MB station is slow, so this POLLS for the
            robot to appear (up to ``robodk.connect_timeout_s``) instead of judging
            the connection on a single slow query — the single-query approach is why
            the first click used to report a problem while a second click (by then
            the station is loaded) worked. On a transient socket error mid-load the
            session is reset so the next poll re-attaches cleanly."""
            import time

            c = services.config.robodk
            deadline = time.monotonic() + float(c.connect_timeout_s)
            last_err: Exception | None = None
            while True:
                try:
                    robot_ok = services.rdk.robot().Valid()
                    if robot_ok:
                        tool_ok = services.rdk.item_exists(c.camera_tool)
                        missing = [n for n, ok in (
                            (c.robot_name, robot_ok),
                            (f"tool {c.camera_tool!r}", tool_ok)) if not ok]
                        # Best-effort link the PHYSICAL robot so the operator no
                        # longer has to connect it by hand before a run (and so the
                        # model tracks the real arm). Never blocks readiness on it —
                        # the controller may be off; the real run re-ensures it.
                        robot_link = link_real_robot(services.rdk, c)
                        return {"connected": True, "ready": robot_ok and tool_ok,
                                "robot": c.robot_name, "robot_valid": robot_ok,
                                "tool": c.camera_tool, "tool_present": tool_ok,
                                "missing": missing, "robot_link": robot_link}
                    last_err = None        # connected; the robot just isn't loaded yet
                except Exception as e:     # socket/timeout while RoboDK is busy loading
                    last_err = e
                    try:
                        services.rdk.session.reset()   # re-attach on the next poll
                    except Exception:
                        pass
                if time.monotonic() >= deadline:
                    break
                time.sleep(1.0)
            if last_err is not None:
                raise HTTPException(503,
                    f"RoboDK didn't become ready within {float(c.connect_timeout_s):.0f}s — "
                    f"it may still be loading the station. Give it a moment and click "
                    f"Connect again. ({last_err})")
            return {"connected": True, "ready": False, "robot": c.robot_name,
                    "robot_valid": False, "tool": c.camera_tool,
                    "tool_present": False, "missing": [c.robot_name]}

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
                                    timeout_s=cc.preview_timeout_s, color_only=True,
                                    quality=cc.preview_jpeg_quality,
                                    codec=cc.preview_codec,
                                    bitrate=cc.preview_h264_bitrate_kbps)
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
            if dry_tour_required(services):
                raise HTTPException(409, "the camera tool was recreated from a past "
                                    "calibration — run the dry tour (Simulate) and let "
                                    "it pass before the real run")
            services.live.stop()    # release the camera for the capture grabs
            params = CalibrationParams(holdout_count=body.holdout_count, refine=body.refine)
            self._active_job = CalibrationJob(services, params)
            services.jobs.start(self._active_job, name="calibration")
            return {"status": "started"}

        @router.post("/poses/clear")
        def poses_clear() -> dict:
            """Delete the generated TasniCalib_* targets and the board keep-out box
            from the station."""
            try:
                existing = services.rdk.list_targets(TARGET_PREFIX)
                services.rdk.delete_items(existing)
                # Remove the platform stand-in too, so it doesn't linger as a stale
                # obstacle for the next aim (it's re-derived on the next Create targets).
                removed_keepout = False
                if services.rdk.item_exists(BOARD_KEEPOUT_NAME):
                    services.rdk.delete_items([BOARD_KEEPOUT_NAME])
                    removed_keepout = True
                return {"cleared": len(existing), "keepout_removed": removed_keepout}
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

        # -- RGB intrinsic calibration (camera-only; no robot motion) -------
        def _intr_session():
            from .intrinsics_calib import IntrinsicCalibSession
            if self._intr is None:
                self._intr = IntrinsicCalibSession(services.config.camera.size)
            return self._intr

        @router.get("/intrinsics/status")
        def intr_status() -> dict:
            return _intr_session().status()

        @router.post("/intrinsics/live/start")
        def intr_live_start() -> dict:
            """Start the intrinsic-capture preview: auto-capture diverse ChArUco views
            as the operator waves the board across the frame. Camera-only; runs on the
            shared live preview in JPEG (NOT h264, so corners stay crisp)."""
            if services.jobs.running:
                raise HTTPException(409, "a calibration run is in progress")

            import cv2

            from ...core.camera_lease import CameraBusy
            from .charuco import CharucoTarget
            from .intrinsics_calib import draw_overlay
            c = services.config
            session = _intr_session()
            board = CharucoTarget(c.board)
            PREVIEW_W = 960
            enc = [cv2.IMWRITE_JPEG_QUALITY, 80]

            def analyze(frame):
                found = board.detect_points(frame.color, min_corners=1)
                if found is None:
                    st = session.offer_none()
                    img = draw_overlay(frame.color, None, None, st)
                else:
                    corners, ids, obj = found
                    st = session.offer(corners, ids, obj)
                    img = draw_overlay(frame.color, corners, ids, st)
                h, w = img.shape[:2]
                if w > PREVIEW_W:
                    img = cv2.resize(img, (PREVIEW_W, int(h * PREVIEW_W / w)),
                                     interpolation=cv2.INTER_AREA)
                ok, jpeg = cv2.imencode(".jpg", img, enc)
                return (jpeg.tobytes() if ok else b""), st

            services.live.stop()    # switch the shared preview out of any aiming stream
            try:
                services.live.start(analyze, fps=c.calibration.preview_fps,
                                    timeout_s=c.calibration.preview_timeout_s,
                                    color_only=True, codec="jpeg")
            except CameraBusy as e:
                raise HTTPException(409, str(e))
            return {"status": "started", **session.status()}

        @router.post("/intrinsics/live/stop")
        def intr_live_stop() -> dict:
            services.live.stop()
            return {"status": "stopped"}

        @router.post("/intrinsics/reset")
        def intr_reset() -> dict:
            session = _intr_session()
            session.reset()
            return session.status()

        @router.post("/intrinsics/solve")
        def intr_solve(body: IntrSolveBody) -> dict:
            """Solve K + distortion from the captured views (cv2.calibrateCamera) and
            save a report. Stops the capture preview first (frees the camera)."""
            import json
            import time

            from ...core.logging import new_run_dir
            if self._intr is None or self._intr.count < 1:
                raise HTTPException(400, "no captured views — start intrinsic capture "
                                    "and move the board around the frame first")
            services.live.stop()
            try:
                report = self._intr.solve(services.config.camera.K, fix_k3=body.fix_k3)
            except RuntimeError as e:
                raise HTTPException(400, str(e))
            except Exception as e:
                raise HTTPException(400, f"intrinsic solve failed: {e}")
            stamp = time.strftime("%Y%m%d-%H%M%S")
            run_dir = new_run_dir("calibration", f"intrinsics-{stamp}")
            (run_dir / "report.json").write_text(json.dumps(report, indent=2),
                                                 encoding="utf-8")
            return {**report, "run_dir": str(run_dir)}

        @router.post("/intrinsics/apply")
        def intr_apply() -> dict:
            """Write the solved K + distortion into the camera config (live + persisted
            to tasni.config.json) — takes effect immediately, no restart needed."""
            if self._intr is None or self._intr.last_report is None:
                raise HTTPException(400, "no solved intrinsics to apply — solve first")
            try:
                return apply_intrinsics(services, self._intr.last_report)
            except Exception as e:
                raise HTTPException(400, f"failed to apply intrinsics: {e}")

        return router
