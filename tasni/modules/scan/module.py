"""ScanModule — plugs the scan workflow into the platform.

Small REST surface mirroring the calibration module: connect, the live depth-standoff
gate, gate-gated target creation, a dry tour, the capture+fuse Run, review (a 3D
preview the browser fetches), and Insert. All real work lives in :mod:`service` +
the pure ``reconstruct``/``plane``/``depth_gate`` libraries.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from ...core.rdk_io import link_real_robot
from ..base import ServiceContainer, WorkflowModule
from ..calibration.service import SimTourJob
from .service import (ScanCaptureJob, ScanParams, ScanResult, generate_scan_targets,
                      insert_scan, live_scan_telemetry_payload, LockedScanSurface,
                      lock_scan_surface, stabilize_live_scan_payload)

if TYPE_CHECKING:  # pragma: no cover
    from fastapi import APIRouter


class InsertBody(BaseModel):
    # Insert the in-memory last scan by default; pass a run_id to insert a past scan
    # loaded from disk (survives a server restart).
    run_id: str | None = None


class CollisionIgnoreBody(BaseModel):
    pair: str


class SurfaceLockBody(BaseModel):
    mode: str = "auto"       # "auto" | "crop"


class ScanModule(WorkflowModule):
    id = "scan"
    title = "Scan"
    description = "3D-scan a work surface → fused mesh + a working frame + rectangle."
    icon = "📷"
    order = 20

    def __init__(self, services: ServiceContainer):
        super().__init__(services)
        self._active_job: ScanCaptureJob | None = None
        self._reference_result: ScanResult | None = None  # set by reference-mode locate
        self._planned_voxel_m: float | None = None         # set by /poses/generate for /run
        self._planned_crop_mm: tuple[float, float] | None = None
        self._planned_surface_size_mm: tuple[float, float] | None = None
        self._locked_surface: LockedScanSurface | None = None

    def router(self) -> "APIRouter":
        from fastapi import APIRouter, HTTPException, Response

        router = APIRouter()
        services = self.services

        @router.get("/config")
        def get_config() -> dict:
            c = services.config
            sc = c.scan
            return {
                "robot": c.robodk.robot_name,
                "camera_tool": c.robodk.camera_tool,
                "camera": {"ip": c.camera.ip, "port": c.camera.port,
                           "resolution": c.camera.resolution},
                "scan": sc.model_dump(),
                "gate": {"ideal_distance_mm": sc.ideal_distance_mm,
                         "distance_tol_mm": sc.distance_tol_mm,
                         "max_tilt_deg": sc.max_tilt_deg},
            }

        @router.post("/connect")
        def connect() -> dict:
            """Open the cell's station and report ready once robot + camera tool are
            present (polls through the slow first load of the 117 MB station)."""
            import time

            c = services.config.robodk
            deadline = time.monotonic() + float(c.connect_timeout_s)
            last_err: Exception | None = None
            while True:
                try:
                    robot_ok = services.rdk.robot().Valid()
                    if robot_ok:
                        tool_ok = services.rdk.item_exists(c.camera_tool)
                        missing = [n for n, ok in ((c.robot_name, robot_ok),
                                                   (f"tool {c.camera_tool!r}", tool_ok))
                                   if not ok]
                        # Best-effort link the physical robot (same as calibration).
                        robot_link = link_real_robot(services.rdk, c)
                        return {"connected": True, "ready": robot_ok and tool_ok,
                                "robot": c.robot_name, "robot_valid": robot_ok,
                                "tool": c.camera_tool, "tool_present": tool_ok,
                                "missing": missing, "robot_link": robot_link}
                    last_err = None
                except Exception as e:
                    last_err = e
                    try:
                        services.rdk.session.reset()
                    except Exception:
                        pass
                if time.monotonic() >= deadline:
                    break
                time.sleep(1.0)
            raise HTTPException(503,
                f"RoboDK didn't become ready within {float(c.connect_timeout_s):.0f}s — "
                f"it may still be loading the station. ({last_err})" if last_err else
                "RoboDK connected but the robot isn't loaded yet — try Connect again.")

        @router.get("/targets")
        def get_targets() -> dict:
            try:
                return {"targets": services.rdk.list_targets(services.config.scan.target_prefix)}
            except Exception as e:
                raise HTTPException(503, f"RoboDK unavailable: {e}")

        @router.get("/collision/status")
        def collision_status() -> dict:
            sc = services.config.scan
            try:
                return services.rdk.collision_status(
                    ensure_pairs=sc.collision_self_pairs,
                    skip_trailing=sc.collision_skip_wrist_links,
                    ignore_pairs=sc.collision_ignore_pairs)
            except Exception as e:
                raise HTTPException(503, f"RoboDK unavailable: {e}")

        @router.post("/collision/ignore")
        def collision_ignore(body: CollisionIgnoreBody) -> dict:
            pair = body.pair.strip()
            if not pair or "↔" not in pair:
                raise HTTPException(400, "invalid collision pair")
            sc = services.config.scan
            if pair not in sc.collision_ignore_pairs:
                sc.collision_ignore_pairs.append(pair)
                from ...core.config import save_overrides
                save_overrides({"scan": {
                    "collision_ignore_pairs": sc.collision_ignore_pairs}})
            try:
                return services.rdk.collision_status(
                    ensure_pairs=sc.collision_self_pairs,
                    skip_trailing=sc.collision_skip_wrist_links,
                    ignore_pairs=sc.collision_ignore_pairs)
            except Exception as e:
                raise HTTPException(503, f"RoboDK unavailable: {e}")

        @router.post("/live/start")
        def live_start() -> dict:
            """Start the uninterrupted color preview using Calibration's transport.

            Depth is deliberately excluded from this live socket. Create targets
            performs the authoritative distance/tilt/surface check without coupling
            those slower readings to video FPS. Camera-only, no robot motion."""
            if services.jobs.running:
                raise HTTPException(409, "a scan run is in progress")
            if services.live.running:
                return {"status": "already running"}

            import cv2

            from ...core.camera_lease import CameraBusy
            c = services.config
            sc = c.scan
            PREVIEW_W = 960
            enc = [cv2.IMWRITE_JPEG_QUALITY, sc.preview_jpeg_quality]
            last_ideal_mm = None
            last_metrics = None

            def analyze(frame):
                nonlocal last_ideal_mm, last_metrics
                # Color-only video: draw ONLY a thin reticle marking where the gate
                # samples standoff/tilt. The HUD overlays all numbers, so we bake no
                # text here (that was the overlapping-text bug).
                img = frame.color.copy()
                h, w = img.shape[:2]
                cw, ch = int(w * sc.center_patch_frac), int(h * sc.center_patch_frac)
                x0, y0 = (w - cw) // 2, (h - ch) // 2
                cv2.rectangle(img, (x0, y0), (x0 + cw, y0 + ch), (120, 200, 160), 1)
                if w > PREVIEW_W:
                    img = cv2.resize(img, (PREVIEW_W, int(h * PREVIEW_W / w)),
                                     interpolation=cv2.INTER_AREA)
                ok, jpeg = cv2.imencode(".jpg", img, enc)
                metrics = live_scan_telemetry_payload(
                    getattr(frame, "telemetry", None), sc,
                    previous_ideal_mm=last_ideal_mm,
                    camera_cfg=c.camera)
                if metrics:
                    metrics = stabilize_live_scan_payload(metrics, last_metrics, sc)
                    last_metrics = metrics
                    last_ideal_mm = metrics.get("ideal_distance_mm", last_ideal_mm)
                return (jpeg.tobytes() if ok else b""), metrics

            # Use the exact same proven color transport as Calibration. Scan depth is
            # intentionally NOT interleaved into this socket: interrupting the video
            # to obtain HUD values is what caused the repeated FPS/no-signal/timeout
            # cycle. The authoritative depth reading remains Create targets.
            preview_codec = c.calibration.preview_codec
            kwargs = dict(
                fps=sc.preview_fps,
                timeout_s=sc.preview_timeout_s,
                color_only=True,
                quality=sc.preview_jpeg_quality,
                codec=preview_codec,
                bitrate=c.calibration.preview_h264_bitrate_kbps,
                scan_telemetry=True,
            )
            try:
                services.live.start(analyze, **kwargs)
            except CameraBusy as e:
                raise HTTPException(409, str(e))
            return {"status": "started"}

        @router.post("/live/stop")
        def live_stop() -> dict:
            services.live.stop()
            return {"status": "stopped"}

        @router.post("/surface/lock")
        def surface_lock(body: SurfaceLockBody) -> dict:
            if services.jobs.running:
                raise HTTPException(409, "a job is already running")
            force_crop = body.mode == "crop"
            if body.mode not in ("auto", "crop"):
                raise HTTPException(400, "surface lock mode must be 'auto' or 'crop'")
            try:
                self._locked_surface = lock_scan_surface(services, force_crop=force_crop)
                gate = self._locked_surface.gate_payload
                crop = gate.get("crop_size_mm")
                extent = gate.get("extent_mm")
                return {
                    "status": "locked",
                    "gate": gate,
                    "surface_mode": "crop" if crop else "full",
                    "extent_mm": extent,
                    "crop_size_mm": crop,
                }
            except RuntimeError as e:
                self._locked_surface = None
                raise HTTPException(400, str(e))
            except Exception as e:
                self._locked_surface = None
                raise HTTPException(503, f"camera/RoboDK unavailable: {e}")

        @router.post("/surface/unlock")
        def surface_unlock() -> dict:
            self._locked_surface = None
            return {"status": "unlocked"}

        @router.post("/poses/generate")
        def poses_generate() -> dict:
            if services.jobs.running:
                raise HTTPException(409, "a job is already running")
            if self._locked_surface is None:
                raise HTTPException(400, "lock and review the surface first")
            try:
                result_dict = generate_scan_targets(services, self._locked_surface)
                # Reference mode returns a ready ScanResult with no targets.
                if result_dict.get("mode") == "reference" and "_scan_result" in result_dict:
                    self._reference_result = result_dict.pop("_scan_result")
                    self._active_job = None
                    self._planned_voxel_m = None
                    self._planned_crop_mm = None
                    self._planned_surface_size_mm = None
                else:
                    self._reference_result = None
                    self._planned_voxel_m = result_dict.get("voxel_size_m")
                    crop = result_dict.get("crop_size_mm")
                    self._planned_crop_mm = tuple(crop) if crop is not None else None
                    extent = result_dict.get("extent_mm")
                    self._planned_surface_size_mm = (
                        tuple(extent) if crop is None and extent is not None else None)
                self._locked_surface = None
                return result_dict
            except RuntimeError as e:
                raise HTTPException(400, str(e))
            except Exception as e:
                raise HTTPException(503, f"RoboDK/camera unavailable: {e}")

        @router.post("/poses/simulate")
        def poses_simulate() -> dict:
            """Dry-run the TasniScan_* targets in SIMULATE (reachability + collisions
            + return-to-start), reusing the calibration dry tour with the scan prefix."""
            sc = services.config.scan
            if services.jobs.running:
                raise HTTPException(409, "a job is already running")
            if len(services.rdk.list_targets(sc.target_prefix)) == 0:
                raise HTTPException(400, "no scan targets to simulate — aim the camera "
                                    "until the gate is green and Create targets first")
            services.live.stop()
            services.jobs.start(SimTourJob(
                services, target_prefix=sc.target_prefix,
                collision_self_pairs=sc.collision_self_pairs,
                collision_skip_wrist_links=sc.collision_skip_wrist_links), name="sim_tour")
            return {"status": "started"}

        @router.post("/poses/clear")
        def poses_clear() -> dict:
            try:
                existing = services.rdk.list_targets(services.config.scan.target_prefix)
                services.rdk.delete_items(existing)
                return {"cleared": len(existing)}
            except Exception as e:
                raise HTTPException(503, f"RoboDK unavailable: {e}")

        @router.post("/run")
        def run() -> dict:
            sc = services.config.scan
            if services.jobs.running:
                raise HTTPException(409, "a job is already running")
            if len(services.rdk.list_targets(sc.target_prefix)) == 0:
                raise HTTPException(400, "no scan targets — aim the camera until the "
                                    "gate is green and Create targets first")
            services.live.stop()
            self._active_job = ScanCaptureJob(services, ScanParams(
                voxel_size_m=self._planned_voxel_m,
                crop_size_mm=self._planned_crop_mm,
                surface_size_mm=self._planned_surface_size_mm))
            services.jobs.start(self._active_job, name="scan")
            return {"status": "started"}

        @router.post("/cancel")
        def cancel() -> dict:
            services.jobs.cancel()
            return {"status": "cancelling"}

        @router.get("/status")
        def status() -> dict:
            return {"status": services.jobs.status, "running": services.jobs.running,
                    "result": services.jobs.result, "error": services.jobs.error}

        @router.get("/result")
        def result() -> dict:
            """Metadata of the last scan (plane frame/rectangle in mm + mesh stats) for
            the review UI. The point cloud itself comes from /preview.bin."""
            if self._active_job is not None and self._active_job.result is not None:
                return self._active_job.result.report
            if self._reference_result is not None:
                return self._reference_result.report
            raise HTTPException(404, "no scan result yet — run a scan first")

        @router.get("/preview.bin")
        def preview_bin(run_id: str | None = None):
            """The decimated fused cloud as binary for the Three.js viewer:
            ``<uint32 N><float32 N*3 xyz mm><float32 N*3 rgb 0..1>`` (little-endian)."""
            import numpy as np

            if run_id is not None:
                from ...core import runs
                try:
                    data = np.load(runs.run_dir("scan", run_id) / "preview.npz")
                    pts = np.asarray(data["points_mm"], np.float32)
                    cols = np.asarray(data["colors"], np.float32)
                except Exception as e:
                    raise HTTPException(404, f"no preview for run {run_id}: {e}")
            elif self._active_job is not None and self._active_job.result is not None:
                pts = self._active_job.result.preview_points_mm.astype(np.float32)
                cols = self._active_job.result.preview_colors.astype(np.float32)
            else:
                raise HTTPException(404, "no scan result yet — run a scan first")
            n = int(len(pts))
            blob = (np.array([n], "<u4").tobytes()
                    + np.ascontiguousarray(pts, "<f4").tobytes()
                    + np.ascontiguousarray(cols, "<f4").tobytes())
            return Response(content=blob, media_type="application/octet-stream",
                            headers={"Cache-Control": "no-store"})

        @router.post("/insert")
        def insert(body: InsertBody) -> dict:
            """Create the work frame + rectangle (+ fused mesh) in the station."""
            from ...core.runs import RunNotFound

            has_job_result = (self._active_job is not None
                              and self._active_job.result is not None)
            if body.run_id is None and not has_job_result and self._reference_result is None:
                raise HTTPException(400, "no scan to insert — run a scan first, or pass a run_id")
            try:
                if body.run_id is None and not has_job_result and self._reference_result is not None:
                    return insert_scan(services, result=self._reference_result)
                return insert_scan(services, job=self._active_job, run_id=body.run_id)
            except RunNotFound as e:
                raise HTTPException(404, str(e))
            except (RuntimeError, ValueError, KeyError) as e:
                raise HTTPException(400, str(e))
            except Exception as e:
                raise HTTPException(503, f"RoboDK unavailable: {e}")

        return router
