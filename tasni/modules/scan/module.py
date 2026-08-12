"""ScanModule — plugs the scan workflow into the platform.

Small REST surface mirroring the calibration module: connect, the live depth-standoff
gate, gate-gated target creation, a dry tour, the capture+fuse Run, review (a 3D
preview the browser fetches), and Insert. All real work lives in :mod:`service` +
the pure ``reconstruct``/``plane``/``depth_gate`` libraries.
"""
from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

from pydantic import BaseModel

from ...core.config import save_overrides
from ...core.events import JobEvent
from ...core.rdk_io import link_real_robot
from ..base import ServiceContainer, WorkflowModule
from ..calibration.service import SimTourJob
from .color_boundary import color_work_boundary
from .sam_boundary import SamBoundaryWorker
from .service import (annotate_pose_liveness, camera_pose_moved, ScanCaptureJob,
                      ScanParams, ScanResult, generate_scan_targets, insert_scan,
                      live_scan_telemetry_payload, LockedScanSurface, lock_scan_surface,
                      stabilize_live_scan_payload)

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


class SurfaceRegionBody(BaseModel):
    width_mm: float
    height_mm: float


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
        # §11 provenance (Task 5): the locked survey's boundary_provenance/to_dict(),
        # threaded from /poses/generate into the /run ScanParams so the run report
        # (and eventually insert's active.json) carry the same provenance as the
        # targets actually created. None/"" whenever the lock built no survey record.
        self._planned_provenance: str | None = None
        self._planned_survey: dict | None = None
        self._targets_token: str = ""
        self._locked_surface: LockedScanSurface | None = None
        # Task 8: the identity of the CURRENT lock, independent of
        # self._locked_surface (which poses_generate() clears once it has
        # consumed the lock to build targets, forcing a fresh lock before the
        # next generate). This field instead tracks "what is locked right now"
        # across a generate — updated by surface_lock() (to the new lock's
        # token) and cleared by surface_unlock() — so run() can tell whether
        # self._targets_token still matches the surface that is currently
        # locked, or whether the operator re-locked/unlocked since generating.
        self._current_lock_token: str | None = None
        # Set by POST /live/refresh; consumed by the live analyze loop to drop the
        # anti-jitter hold + pose anchor and re-read fresh at the current robot pose.
        self._live_refresh = threading.Event()
        # Background SAM boundary worker (lives with the live preview; None for the
        # classical colour engine). Runs the ~450 ms/frame model off the video thread.
        self._sam_worker: "SamBoundaryWorker | None" = None
        # Operator-declared work-region size (mm), passed into every surface_lock()
        # call (Task 3: provenance is decided solely by force_crop, not by whether
        # this happens to be passed — safe to always pass). Defaults from config,
        # refreshed by POST /surface/region and persisted there under the same
        # scan.work_crop_mm key. NOTE: this host-side value is authoritative; the
        # Jetson's live overlay square (server/server_unicast_syncronous.py,
        # WORK_CROP_MM) is display-only and intentionally left out of sync — that
        # server change is out of scope here.
        self._user_region_mm: tuple[float, float] = tuple(
            float(v) for v in services.config.scan.work_crop_mm)

    def surface_region(self, body: SurfaceRegionBody) -> dict:
        """Declare the work-region size (mm) used for crop locks.

        A real bound method (not a router() closure) so it is directly callable
        — by tests with fake services, and by the POST /surface/region route
        below, which is a thin wrapper mirroring surface_lock's registration.
        """
        from fastapi import HTTPException

        w, h = float(body.width_mm), float(body.height_mm)
        if not (100.0 <= w <= 4000.0 and 100.0 <= h <= 4000.0):
            raise HTTPException(422, "region dimensions must be 100-4000 mm")
        self._user_region_mm = (w, h)
        save_overrides({"scan": {"work_crop_mm": [w, h]}})
        return {"user_region_mm": [w, h]}

    def surface_lock(self, body: SurfaceLockBody) -> dict:
        """Freeze the current measured surface (auto or forced crop) so target
        generation has a stable geometry to build from.

        A real bound method (not a router() closure) — same pattern as
        surface_region, so it is directly callable by tests. Records this
        lock's identity in ``self._current_lock_token`` so run()'s Task 8
        guard can tell whether generated targets still match the surface that
        is currently locked, or whether the operator re-locked/unlocked since.
        """
        from fastapi import HTTPException

        services = self.services
        if services.jobs.running:
            raise HTTPException(409, "a job is already running")
        force_crop = body.mode == "crop"
        if body.mode not in ("auto", "crop"):
            raise HTTPException(400, "surface lock mode must be 'auto' or 'crop'")
        try:
            self._locked_surface = lock_scan_surface(
                services, force_crop=force_crop, user_region_mm=self._user_region_mm)
            self._current_lock_token = self._locked_surface.lock_token
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
            self._current_lock_token = None
            raise HTTPException(400, str(e))
        except Exception as e:
            self._locked_surface = None
            self._current_lock_token = None
            raise HTTPException(503, f"camera/RoboDK unavailable: {e}")

    def surface_unlock(self) -> dict:
        """Drop the current lock.

        Also invalidates any targets already generated from it, for run()'s
        Task 8 guard: self._current_lock_token -> None, so a stale
        self._targets_token no longer matches and run() refuses.
        """
        self._locked_surface = None
        self._current_lock_token = None
        return {"status": "unlocked"}

    def poses_generate(self) -> dict:
        """Generate TasniScan_* targets from the currently locked surface.

        A real bound method (not a router() closure) — same pattern as
        surface_region — so the Task 8 lock-token guard is directly testable.
        """
        from fastapi import HTTPException

        services = self.services
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
                self._planned_provenance = None
                self._planned_survey = None
                self._targets_token = ""
            else:
                self._reference_result = None
                self._planned_voxel_m = result_dict.get("voxel_size_m")
                crop = result_dict.get("crop_size_mm")
                self._planned_crop_mm = tuple(crop) if crop is not None else None
                extent = result_dict.get("extent_mm")
                self._planned_surface_size_mm = (
                    tuple(extent) if crop is None and extent is not None else None)
                self._planned_provenance = result_dict.get("boundary_provenance")
                self._planned_survey = result_dict.get("survey")
                self._targets_token = result_dict.get("lock_token") or ""
            self._locked_surface = None
            return result_dict
        except RuntimeError as e:
            raise HTTPException(400, str(e))
        except Exception as e:
            raise HTTPException(503, f"RoboDK/camera unavailable: {e}")

    def run(self) -> dict:
        """Start the capture+fuse job for the currently generated TasniScan_*
        targets.

        A real bound method (not a router() closure) — same pattern as
        surface_region — so the Task 8 lock-token guard is directly testable.

        Task 8 guard: refuses before the job starts if the targets were
        generated for a surface lock that is no longer the current one (the
        operator re-locked or unlocked since generating) — running them could
        drive the arm to poses computed for geometry that has since moved.
        """
        from fastapi import HTTPException

        services = self.services
        sc = services.config.scan
        if services.jobs.running:
            raise HTTPException(409, "a job is already running")
        token = self._targets_token
        if token and self._current_lock_token != token:
            raise HTTPException(409, "targets predate the current surface lock "
                                "— regenerate targets before running")
        if len(services.rdk.list_targets(sc.target_prefix)) == 0:
            raise HTTPException(400, "no scan targets — aim the camera until the "
                                "gate is green and Create targets first")
        services.live.stop()
        self._active_job = ScanCaptureJob(services, ScanParams(
            voxel_size_m=self._planned_voxel_m,
            crop_size_mm=self._planned_crop_mm,
            surface_size_mm=self._planned_surface_size_mm,
            boundary_provenance=self._planned_provenance,
            survey=self._planned_survey))
        services.jobs.start(self._active_job, name="scan")
        return {"status": "started"}

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
            anchor_pose_T = None
            last_driver_check = 0.0
            driver_ok = False

            def _publish_boundary(cb):
                # The one boundary contract the HUD consumes (identical for every engine).
                services.bus.publish(JobEvent("boundary", {
                    "outline_uv": cb["outline_uv"],
                    "polygon_uv": cb["polygon_uv"],
                    "overruns": cb["overruns"],
                    "contrast": cb["contrast"],
                }))

            def _color_boundary(color):
                # Classical colour segmenter; also the SAM worker's abstain fallback.
                return color_work_boundary(
                    color, reticle_frac=sc.center_patch_frac,
                    min_color_dist=sc.color_boundary_min_color_dist,
                    seg_width=sc.color_boundary_seg_width)

            def analyze(frame):
                nonlocal last_ideal_mm, last_metrics, anchor_pose_T
                nonlocal last_driver_check, driver_ok
                if self._live_refresh.is_set():
                    # Operator pressed Refresh: re-read at the current pose. Drop the
                    # hold (last_metrics) and the pose anchor (anchor_pose_T) so the next
                    # few frames re-settle a fresh reading HERE — clears a stale overlay
                    # when RoboDK is not mirroring the arm and a lateral jog slipped past
                    # the hold. KEEP last_ideal_mm so the distance target stays continuous
                    # (no flash to accurate_min while it re-frames).
                    self._live_refresh.clear()
                    last_metrics = None
                    anchor_pose_T = None
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
                # Live work boundary, published every video frame independent of the
                # ~1 Hz depth telemetry + the anti-jitter freeze, so the HUD's blue
                # rectangle tracks the object in real time. Segmented from the raw color
                # frame (before the reticle is drawn). A visual aid; depth still drives the
                # gates + the lock. SAM engines hand the frame to the background worker
                # (inference runs off THIS video thread so ~450 ms/frame never hitches the
                # preview); the classical colour engine segments inline (sub-frame cost).
                if sc.color_boundary_enabled:
                    if self._sam_worker is not None:
                        self._sam_worker.submit(frame.color.copy())
                    elif sc.boundary_engine == "color":
                        try:
                            cb = _color_boundary(frame.color)
                        except Exception:
                            cb = None
                        if cb is not None:
                            _publish_boundary(cb)
                metrics = live_scan_telemetry_payload(
                    getattr(frame, "telemetry", None), sc,
                    previous_ideal_mm=last_ideal_mm,
                    camera_cfg=c.camera)
                if metrics:
                    # HARD anti-jitter gate: RoboDK mirrors the physical arm, so the
                    # camera pose is the true "did the robot move" signal. While it
                    # sits inside the tolerance of the anchor pose (where the current
                    # reading was taken), hold the reading instead of chasing the
                    # RealSense plane-fit noise. Re-anchor the moment the arm moves.
                    pose_T = None
                    try:
                        pose_T = services.rdk.camera_pose_T()
                    except Exception:
                        pose_T = None
                    # Driver-connected status is a cheap RoboDK round trip but not
                    # free at video frame rate — recheck at most every 2 s, same
                    # caching pattern as last_ideal_mm. Any failure (RoboDK busy,
                    # station reloading, etc.) reads as "not live": a driver-status
                    # probe must never take down the live preview.
                    now = time.monotonic()
                    if now - last_driver_check >= 2.0:
                        last_driver_check = now
                        try:
                            driver_ok = bool(services.rdk.robot_connected()[0])
                        except Exception:
                            driver_ok = False
                    moved = camera_pose_moved(
                        pose_T, anchor_pose_T,
                        sc.live_hold_pose_trans_mm, sc.live_hold_pose_rot_deg)
                    metrics = stabilize_live_scan_payload(
                        metrics, last_metrics, sc,
                        robot_static=(not moved and pose_T is not None))
                    # Called AFTER stabilize_live_scan_payload so the freeze/hold logic
                    # above (which copies dicts wholesale in places) cannot clobber it.
                    metrics = annotate_pose_liveness(
                        metrics, pose_T=pose_T, driver_ok=driver_ok)
                    # Re-anchor to the current pose only while the reading is LIVE (not
                    # frozen). During a hold we keep comparing against the pose where it
                    # engaged, so a transient model-pose blip can't drift the anchor and
                    # cascade into a spurious release; while live/settling the anchor
                    # tracks the arm frame-to-frame so a genuine jog is caught promptly.
                    if pose_T is not None and not metrics.get("held"):
                        anchor_pose_T = pose_T
                    last_metrics = metrics
                    # Latch the target only while the surface is FRAMED (full). In crop
                    # the telemetry HOLDS this latched value instead of collapsing to
                    # accurate_min, so a brief over-nudge into overrun doesn't move the
                    # goalpost. A genuinely oversized surface never frames, so the latch
                    # stays None and crop correctly falls back to the work-close standoff.
                    if metrics.get("surface_mode") == "full":
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
            # Boundary engine: spin up the SAM worker unless the classical colour engine
            # is selected (or the boundary layer is off). A missing model / onnxruntime
            # does NOT fail here — the worker detects it on its own thread and flips to the
            # colour fallback (or idle), logging once, so the video always starts.
            self._sam_worker = None
            if sc.color_boundary_enabled and sc.boundary_engine in ("sam", "sam_then_color"):
                fallback = _color_boundary if sc.boundary_engine == "sam_then_color" else None
                self._sam_worker = SamBoundaryWorker(
                    _publish_boundary,
                    model_dir=sc.sam_model_dir,
                    encoder_file=sc.sam_encoder_file,
                    decoder_file=sc.sam_decoder_file,
                    min_score=sc.sam_min_score,
                    max_fill_frac=sc.sam_max_fill_frac,
                    point_uv=(0.5, 0.5),
                    fallback=fallback,
                    log=print)
            try:
                services.live.start(analyze, **kwargs)
            except CameraBusy as e:
                if self._sam_worker is not None:
                    self._sam_worker.stop()
                    self._sam_worker = None
                raise HTTPException(409, str(e))
            return {"status": "started"}

        @router.post("/live/stop")
        def live_stop() -> dict:
            services.live.stop()
            if self._sam_worker is not None:
                self._sam_worker.stop()
                self._sam_worker = None
            return {"status": "stopped"}

        @router.post("/live/refresh")
        def live_refresh() -> dict:
            """Re-read the live surface at the current robot pose.

            Drops the anti-jitter hold + pose anchor so the overlay/readouts re-settle
            fresh where the arm is NOW — the escape hatch for a stale projection when
            RoboDK is not mirroring the arm (driver monitoring off) and a lateral jog
            slipped past the hold + vision escape. Keeps the video streaming (no
            teardown) and the distance target continuous.
            """
            if not services.live.running:
                raise HTTPException(409, "live preview is not running")
            self._live_refresh.set()
            return {"status": "refreshing"}

        @router.post("/surface/lock")
        def surface_lock(body: SurfaceLockBody) -> dict:
            return self.surface_lock(body)

        @router.post("/surface/unlock")
        def surface_unlock() -> dict:
            return self.surface_unlock()

        @router.post("/surface/region")
        def surface_region(body: SurfaceRegionBody) -> dict:
            return self.surface_region(body)

        @router.post("/poses/generate")
        def poses_generate() -> dict:
            return self.poses_generate()

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
            return self.run()

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
