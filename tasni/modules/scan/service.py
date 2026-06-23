"""Scan orchestration — the gate, target generation, capture+fuse job, and insert.

Mirrors ``modules/calibration/service.py`` (same flow: live gate -> Create targets
-> dry tour -> Run -> review -> apply/insert) but for scanning a work surface:

  1. Depth standoff gate (no ChArUco board): jog the camera to look down at the
     table until the HUD lamps are green.
  2. ``generate_scan_targets``: reachable cone poses around the gated standoff seed,
     left as ``TasniScan_*`` (its own prefix — never the calibration targets).
  3. ``ScanCaptureJob``: visit the targets, grab depth+color, **fuse** (TSDF) into a
     mesh, fit the work **plane -> frame + rectangle**, hold the result for review.
  4. ``insert_scan``: on the user's click, create the frame + rectangle (+ mesh) in
     the open station.

Decoupled from calibration: it uses the *stored* camera tool offset + intrinsics to
register views — it never runs calibration. It only WARNS if the tool offset looks
like the flange (no calibration on file). Reuses calibration's camera-tool / lease /
pose-generation helpers so the two modules share one implementation.
"""
from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from ...core import runs
from ...core.camera import CameraError
from ...core.events import JobEvent
from ...core.jobrunner import JobContext
from ...core.logging import get_logger, new_run_dir
from ...core.rdk_io import RdkIO
# Reuse calibration's shared orchestration helpers (one implementation).
from ..calibration.poses import (generate_calibration_poses, select_diverse,
                                  viewing_angle_span)
from ..calibration.service import (_camera_hold, dry_tour_required,
                                    ensure_camera_tool)
from .depth_gate import ScanGateThresholds, evaluate_depth_gate
from .plane import work_plane_from_points
from .reconstruct import (ScanView, cloud_points_m, crop_box, decimate_for_preview,
                          fuse_views, look_point_from_views, save_mesh)

log = get_logger("tasni.scan")

# Minimum posed views for a usable fusion (a flat table tolerates few, but more =
# better coverage; below this the mesh/plane are unreliable).
SCAN_MIN_VIEWS = 4

# Names of the items insert creates in the station (replaced on each insert).
FRAME_NAME = "Tasni Work Frame"
RECT_NAME = "Tasni Work Surface"
MESH_NAME = "Tasni Scan Mesh"


def scan_gate_thresholds(scfg) -> ScanGateThresholds:
    """One source of truth so the live preview and the authoritative grab gate
    identically (mirrors calibration's ``gate_thresholds``)."""
    return ScanGateThresholds(
        ideal_distance_mm=scfg.ideal_distance_mm,
        distance_tol_mm=scfg.distance_tol_mm,
        max_tilt_deg=scfg.max_tilt_deg,
        center_patch_frac=scfg.center_patch_frac,
        min_valid_depth_frac=scfg.min_valid_depth_frac)


def _log_pub(services):
    return lambda m: services.bus.publish(JobEvent("log", {"message": m}))


def generate_scan_targets(services) -> dict:
    """Gate-gated scan-target creation (synchronous, no robot motion).

    Stops the live preview, grabs one authoritative depth frame, and refuses unless a
    surface is centred at the ideal standoff + roughly fronto-parallel. On success the
    robot's current camera pose is the seed: reachable cone poses are generated and
    written as ``TasniScan_*`` (prior ones cleared). Raises ``RuntimeError`` if not
    ready / too few reachable poses.
    """
    cfg = services.config
    scfg = cfg.scan
    rdk: RdkIO = services.rdk
    cam = services.camera
    tool_name = cfg.robodk.camera_tool
    K = cfg.camera.K
    prefix = scfg.target_prefix
    pub = _log_pub(services)

    ensure_camera_tool(services, log=pub)
    if services.live.running:
        services.live.stop()

    tool_pose = rdk.use_camera_tool(tool_name)
    # DECOUPLING: the scan uses whatever calibration is on file; it never runs one.
    # But if the camera tool has ~no offset, calibration was never applied and the
    # poses orbit the FLANGE, not the camera — and the fused cloud will be
    # misregistered. Warn loudly; do NOT block (calibration is done "every blue moon").
    tool_offset_mm = float(np.linalg.norm(np.asarray(tool_pose)[:3, 3]))
    if tool_offset_mm < 15.0:
        services.bus.publish(JobEvent("log", {"message":
            f"WARNING: the {tool_name!r} tool is only ~{tool_offset_mm:.0f} mm off the "
            f"flange (≈ no calibration on file) — the scan will register views against "
            f"the FLANGE, so the fused mesh + work frame may be off. Run Calibration "
            f"once for an accurate scan; proceeding with the stored offset for now."}))

    th = scan_gate_thresholds(scfg)
    with _camera_hold(services, "scan-target-creation"):
        frame = cam.grab(with_depth=True, timeout=scfg.grab_timeout_s)
    reading = evaluate_depth_gate(frame.depth, K, th, depth_scale=scfg.depth_scale)

    ok, jpeg = cv2.imencode(".jpg", frame.color)
    if ok:
        services.bus.publish(JobEvent("frame",
            {"jpeg_b64": base64.b64encode(jpeg.tobytes()).decode("ascii")}))
    services.bus.publish(JobEvent("gate", {**reading.to_dict(), "live": False}))

    if not reading.ok:
        bad = [name for name, good in reading.gates.items() if not good]
        dist = reading.distance_mm and round(reading.distance_mm)
        tilt = reading.tilt_deg and round(reading.tilt_deg, 1)
        raise RuntimeError(
            "surface not in the ideal band — fix " + ", ".join(bad)
            + f" (distance {dist} mm, tilt {tilt}°, valid depth "
            + f"{round(reading.valid_frac * 100)}%). Jog the camera to look down at "
            + "the table until all HUD lamps are green, then create targets.")

    seed_T = rdk.camera_pose_T()
    try:
        seed_joints = rdk.current_joints()
    except Exception:
        seed_joints = None
    look = float(reading.distance_mm)

    prior = rdk.list_targets(prefix)
    if prior:
        rdk.delete_items(prior)

    candidates = generate_calibration_poses(
        seed_T, count=scfg.pose_count, look_distance_mm=look,
        cone_half_angle_deg=scfg.cone_half_angle_deg,
        roll_max_deg=scfg.roll_max_deg, distance_jitter=scfg.distance_jitter)
    reachable = [(i, T) for i, T in enumerate(candidates) if rdk.is_reachable(T)]
    n_reach = len(reachable)
    if n_reach < SCAN_MIN_VIEWS:
        raise RuntimeError(
            f"only {n_reach} reachable poses around this view (need >= {SCAN_MIN_VIEWS}) "
            f"— jog to a more open part of the workspace (still framing the table) and retry")

    # Collision screen (identical machinery to calibration; uses scan's knobs).
    guard_skip = None
    if scfg.collision_filter and scfg.collision_self_pairs:
        guard_skip = scfg.collision_skip_wrist_links
        guard = rdk.ensure_mounted_tool_collision_pairs(scfg.collision_skip_wrist_links)
        n_pairs = (guard or {}).get("pairs_enabled", 0)
        services.bus.publish(JobEvent("log", {"message":
            f"collision guard: enabled {n_pairs} tool↔arm pair(s) "
            f"(RoboDK omits these by default)" if n_pairs else
            "WARNING: collision guard enabled 0 tool↔arm pairs — confirm the camera "
            "is mounted on the robot in RoboDK"}))

    n_collide = 0
    col_checked = False
    reach_joints: list = [None] * n_reach
    if scfg.collision_filter:
        mask, col_checked, jts = rdk.screen_collisions([T for _, T in reachable],
                                                       guard_skip=guard_skip)
        kept = [k for k in range(n_reach) if mask[k]]
        if col_checked:
            n_collide = n_reach - len(kept)
        reachable = [reachable[k] for k in kept]
        reach_joints = [jts[k] for k in kept]
        services.bus.publish(JobEvent("log", {"message":
            f"collision screen: {'ACTIVE' if col_checked else 'unavailable'}; swept "
            f"{n_reach} reachable pose(s), {n_collide} collided and were dropped"}))
        if col_checked and len(reachable) < SCAN_MIN_VIEWS:
            raise RuntimeError(
                f"only {len(reachable)} collision-free poses ({n_collide} of {n_reach} "
                f"would collide) — jog to a more open part of the workspace and retry")

    n_usable = len(reachable)
    reach_T = [T for _, T in reachable]
    sel = select_diverse(reach_T, min(scfg.pose_count, n_usable), seed_fwd=seed_T[:3, 2])
    chosen = [(reachable[k][0], reachable[k][1], reach_joints[k]) for k in sel]

    n_backfilled = 0
    created: list[str] = []
    for _, T, joints in chosen:
        if joints is None:
            joints = rdk.solve_joints_for_pose(T, seed_joints)
            if joints is not None:
                n_backfilled += 1
        name = f"{prefix}{len(created) + 1:02d}"
        rdk.add_target(name, T, joints=joints)
        created.append(name)

    _, eff_max, eff_mean = viewing_angle_span([T for _, T, _ in chosen], seed_T[:3, 2])
    services.bus.publish(JobEvent("log", {"message":
        f"created {len(created)} scan targets (standoff ~{look:.0f} mm; "
        f"{n_reach}/{len(candidates)} candidates reachable; effective cone "
        f"~{eff_max:.0f}° of {scfg.cone_half_angle_deg:.0f}°) — inspect them in RoboDK"}))

    return {"created": len(created), "targets": created, "look_distance_mm": look,
            "gate": reading.to_dict(), "candidates_reachable": n_reach,
            "candidates_total": len(candidates), "collisions_checked": col_checked,
            "candidates_collided": n_collide, "effective_cone_deg": round(eff_max, 1),
            "camera_tool_offset_mm": round(tool_offset_mm, 1),
            "calibration_on_file": tool_offset_mm >= 15.0}


# -- capture + reconstruct job ----------------------------------------------
@dataclass
class ScanParams:
    save_artifacts: bool = True


@dataclass
class ScanResult:
    report: dict                       # JSON-serializable (plane in mm + mesh stats)
    run_dir: str
    frame_T_mm: np.ndarray             # 4x4 base->work-frame (RoboDK mm units)
    corners_mm: np.ndarray             # (4,3) rectangle corners (mm)
    mesh_obj_path: str | None
    preview_points_mm: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))
    preview_colors: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float32))


def _result_report(wp, frame_T_mm, corners_mm, *, n_views, n_points, mesh,
                   run_dir, stamp) -> dict:
    return {
        "module": "scan", "stamp": stamp, "run_dir": str(run_dir),
        "n_views": int(n_views), "n_points": int(n_points),
        "mesh_vertices": int(len(mesh.vertices)),
        "mesh_triangles": int(len(mesh.triangles)),
        "mesh_file": "mesh.obj",
        "plane": {
            "frame_T_mm": np.asarray(frame_T_mm, float).tolist(),
            "corners_mm": np.asarray(corners_mm, float).tolist(),
            "size_mm": [float(wp.size[0] * 1000.0), float(wp.size[1] * 1000.0)],
            "normal": wp.normal.tolist(),
            "inlier_frac": float(wp.inlier_frac),
            "inlier_count": int(wp.inlier_count),
        },
    }


class ScanCaptureJob:
    """Callable run by the JobRunner: visit ``TasniScan_*``, fuse, fit the work plane,
    and hold the result for the separate insert step (insert is an explicit action)."""

    def __init__(self, services, params: ScanParams | None = None):
        self.services = services
        self.params = params or ScanParams()
        self.tool_name: str = services.config.robodk.camera_tool
        self.result: ScanResult | None = None

    def __call__(self, ctx: JobContext) -> dict:
        cfg = self.services.config
        scfg = cfg.scan
        rdk: RdkIO = self.services.rdk
        cam = self.services.camera
        K = cfg.camera.K
        width, height = cfg.camera.size
        prefix = scfg.target_prefix

        ensure_camera_tool(self.services, log=ctx.log)
        if dry_tour_required(self.services):
            raise RuntimeError(
                "the camera tool was recreated from a past calibration and has not "
                "passed a dry tour since — run the dry tour (Simulate) first.")

        targets = rdk.list_targets(prefix)
        if len(targets) < SCAN_MIN_VIEWS:
            raise RuntimeError(
                f"only {len(targets)} {prefix}* targets; need >= {SCAN_MIN_VIEWS}. Aim "
                f"the camera at the table until the gate is green and Create targets first.")

        if self.services.live.running:
            self.services.live.stop()

        applied_mode = rdk.apply_run_mode("run_robot")
        ctx.log(f"run mode: {applied_mode} (REAL ROBOT); {len(targets)} targets to visit")
        rdk.use_camera_tool(self.tool_name)
        try:
            start_joints = rdk.current_joints()
        except Exception:
            start_joints = None

        stamp = time.strftime("%Y%m%d-%H%M%S")
        run_dir = new_run_dir("scan", stamp)

        try:
            with _camera_hold(self.services, "scan-run"):
                views, skipped = self._capture(ctx, rdk, cam, targets, scfg)
            if len(views) < SCAN_MIN_VIEWS:
                raise RuntimeError(
                    f"only {len(views)} usable views (need >= {SCAN_MIN_VIEWS}). "
                    f"Skipped (no depth): {skipped}")

            ctx.progress(len(targets), len(targets), "fusing")
            ctx.log(f"fusing {len(views)} views (TSDF voxel {scfg.voxel_size_m * 1000:.0f} mm)…")
            res = fuse_views(views, K, width, height, voxel_size_m=scfg.voxel_size_m,
                             sdf_trunc_m=scfg.sdf_trunc_m, depth_scale=scfg.depth_scale,
                             depth_min_m=scfg.depth_min_m, depth_max_m=scfg.depth_max_m)

            # Isolate the work surface (the "top layer"): crop to a box around where the
            # camera was aimed so the FLOOR/walls don't dominate the fit (the cause of a
            # room-sized plane). Falls back to the full cloud if the crop is too thin.
            mesh, cloud = res.mesh, res.cloud
            if scfg.roi_enabled:
                center_mm = look_point_from_views(views)
                if center_mm is None:
                    ctx.log("ROI: no central depth to locate the aim — using the full cloud")
                else:
                    cm = center_mm / 1000.0
                    roi = dict(radius_m=scfg.roi_radius_m, below_m=scfg.roi_below_m,
                               above_m=scfg.roi_above_m)
                    c_cloud = crop_box(cloud, cm, **roi)
                    n0, n1 = len(cloud.points), len(c_cloud.points)
                    if n1 >= 500:
                        cloud = c_cloud
                        mesh = crop_box(mesh, cm, **roi)
                        ctx.log(f"ROI: cropped to a {2 * scfg.roi_radius_m:.1f} m box around the "
                                f"aim (surface Z≈{center_mm[2]:.0f} mm); kept {n1}/{n0} pts "
                                f"(floor/walls dropped)")
                    else:
                        ctx.log(f"ROI crop would keep only {n1} pts — using the full cloud "
                                f"(widen scan.roi_radius_m if the surface was clipped)")

            pts = cloud_points_m(cloud)
            if len(pts) == 0:
                raise RuntimeError("fusion produced an empty cloud — check the standoff "
                                   "band / depth range, or that depth is being received")

            ctx.log("fitting the work plane + rectangle…")
            wp = work_plane_from_points(
                pts, distance=scfg.ransac_distance_m, n_iterations=scfg.ransac_iterations,
                min_inlier_frac=scfg.min_inlier_frac)
            # metres -> mm for RoboDK (rotation is unitless; translation + corners scale)
            frame_T_mm = wp.frame_T.copy()
            frame_T_mm[:3, 3] *= 1000.0
            corners_mm = wp.corners * 1000.0
            pp_m, cc = decimate_for_preview(cloud, scfg.preview_max_points)
            preview_mm = (pp_m * 1000.0).astype(np.float32)

            report = _result_report(wp, frame_T_mm, corners_mm, n_views=len(views),
                                     n_points=len(pts), mesh=mesh, run_dir=run_dir,
                                     stamp=stamp)
            mesh_obj = None
            if self.params.save_artifacts:
                save_mesh(mesh, str(run_dir / "mesh.obj"))
                save_mesh(mesh, str(run_dir / "mesh.ply"))
                mesh_obj = str(run_dir / "mesh.obj")
                np.savez_compressed(run_dir / "preview.npz",
                                    points_mm=preview_mm, colors=cc)
                (run_dir / "report.json").write_text(json.dumps(report, indent=2),
                                                     encoding="utf-8")
                runs.write_meta("scan", stamp, {"module": "scan", "stamp": stamp,
                                                "tool_name": self.tool_name})

            self.result = ScanResult(
                report=report, run_dir=str(run_dir), frame_T_mm=frame_T_mm,
                corners_mm=corners_mm, mesh_obj_path=mesh_obj,
                preview_points_mm=preview_mm, preview_colors=cc)

            sz = report["plane"]["size_mm"]
            ctx.log(f"fused {len(views)} views -> {len(pts)} pts, "
                    f"{len(res.mesh.vertices)} mesh verts; work surface "
                    f"{sz[0]:.0f} x {sz[1]:.0f} mm (plane inliers "
                    f"{report['plane']['inlier_frac']:.0%}). Review, then Insert.")
            return {"kind": "scan", "run_dir": str(run_dir), "can_insert": True,
                    **report}
        finally:
            if start_joints is not None:
                try:
                    ctx.log("returning to start pose")
                    rdk.move_j_joints(start_joints)
                except Exception:
                    pass

    def _capture(self, ctx, rdk, cam, targets, scfg):
        """Visit each target and gather a depth+color view per pose. Burst mode (if
        enabled and the server supports it) buffers frames on the Jetson and pulls
        them in one transfer at the end; otherwise grab per pose. Falls back to the
        per-pose path if the burst handshake is rejected (a pre-burst server)."""
        if scfg.burst_capture:
            try:
                return self._capture_burst(ctx, rdk, cam, targets, scfg)
            except CameraError as e:
                ctx.log(f"burst capture unavailable ({e}); using per-pose grab")
        return self._capture_per_pose(ctx, rdk, cam, targets, scfg)

    def _capture_per_pose(self, ctx, rdk, cam, targets, scfg):
        views: list[ScanView] = []
        skipped: list[str] = []
        total = len(targets)
        for i, name in enumerate(targets):
            ctx.check_cancel()
            ctx.progress(i + 1, total, f"capturing {name}")
            rdk.move_j(name)
            time.sleep(scfg.settle_s)
            frame = cam.grab(with_depth=True, timeout=scfg.grab_timeout_s)
            if frame.depth is None:
                ctx.log(f"{name}: no depth — skipped")
                skipped.append(name)
                continue
            pose = rdk.camera_pose_T()                 # uses the STORED tool offset
            views.append(ScanView(color=frame.color, depth=frame.depth, pose_T=pose))
            ok, jpeg = cv2.imencode(".jpg", frame.color)
            if ok:
                ctx.frame(jpeg.tobytes())
            ctx.log(f"{name}: captured ({np.count_nonzero(frame.depth)} depth px)")
        return views, skipped

    def _capture_burst(self, ctx, rdk, cam, targets, scfg):
        """Fast tour: at each pose the Jetson buffers the depth+color frame (a quick
        round-trip returning a thumbnail), then all frames are pulled in ONE burst and
        the Jetson buffer is dropped. The per-pose camera pose is still recorded as
        each view's extrinsic, so the fused result is identical to the per-pose path —
        only the network cost moves out of the robot loop.

        Alignment: the server buffers one frame per CAP that returns a thumbnail (a
        ``None`` thumbnail means it skipped that pose — no valid frame / buffer full),
        so ``fetch_all`` returns exactly the buffered ones, in order. We therefore pair
        the returned frames against only the poses whose CAP buffered a frame — never
        by raw target index, which would misalign every view after a skip."""
        skipped: list[str] = []
        captured: list = []          # (name, pose) for each CAP that buffered a frame
        total = len(targets)
        with cam.burst(timeout=scfg.grab_timeout_s) as bs:
            for i, name in enumerate(targets):
                ctx.check_cancel()
                ctx.progress(i + 1, total, f"capturing {name}")
                rdk.move_j(name)
                time.sleep(scfg.settle_s)
                thumb = bs.capture()                   # Jetson grabs + buffers the frame
                if thumb is None:                      # server skipped -> not in fetch_all
                    ctx.log(f"{name}: no frame buffered — skipped")
                    skipped.append(name)
                    continue
                try:
                    pose = rdk.camera_pose_T()         # uses the STORED tool offset
                except Exception:
                    pose = None
                captured.append((name, pose))
                ctx.frame(thumb)                       # live per-pose thumbnail strip
                ctx.log(f"{name}: captured (buffered on Jetson)")
            ctx.progress(total, total, "downloading buffered frames…")
            ctx.log("transferring all buffered frames from the Jetson in one burst…")
            frames = bs.fetch_all()
            bs.clear()                                 # delete the buffer on the Jetson

        if len(frames) != len(captured):
            ctx.log(f"WARNING: Jetson returned {len(frames)} frame(s) but {len(captured)} "
                    f"were buffered — pairing the overlap (some views may be dropped)")
        views: list[ScanView] = []
        for (name, pose), fr in zip(captured, frames):
            if fr is None or fr.depth is None or pose is None:
                ctx.log(f"{name}: no depth/pose — skipped")
                skipped.append(name)
                continue
            views.append(ScanView(color=fr.color, depth=fr.depth, pose_T=pose))
        ctx.log(f"burst transfer complete: {len(frames)} frame(s), {len(views)} usable")
        return views, skipped


# -- insert (the explicit "apply") ------------------------------------------
def insert_scan(services, *, job: "ScanCaptureJob | None" = None,
                run_id: str | None = None) -> dict:
    """Create the work frame + rectangle (+ fused mesh) in the open station.

    Two sources (mirrors ``apply_calibration``): an explicit ``run_id`` loads the
    plane + mesh from disk (survives a restart), else the in-memory last job. Records
    ``runs/scan/active.json``. Raises ``RuntimeError`` if there is nothing to insert.
    """
    rdk: RdkIO = services.rdk
    if run_id is not None:
        report = runs.load_report("scan", run_id)
        plane = report["plane"]
        frame_T_mm = np.asarray(plane["frame_T_mm"], float)
        corners_mm = np.asarray(plane["corners_mm"], float)
        rd = runs.run_dir("scan", run_id)
        mesh_obj = rd / report.get("mesh_file", "mesh.obj")
        mesh_path = str(mesh_obj) if mesh_obj.is_file() else None
        stamp_id, source = run_id, "run_id"
    elif job is not None and job.result is not None:
        r = job.result
        frame_T_mm, corners_mm = r.frame_T_mm, r.corners_mm
        mesh_path = r.mesh_obj_path
        report = r.report
        stamp_id, source = report.get("stamp"), "memory"
    else:
        raise RuntimeError("no scan to insert — run a scan first, or pass a run_id")

    frame = rdk.add_frame(FRAME_NAME, frame_T_mm)
    rect = rdk.add_rectangle(RECT_NAME, corners_mm)
    mesh_inserted = False
    if mesh_path:
        item = rdk.add_mesh_file(MESH_NAME, mesh_path)
        mesh_inserted = bool(getattr(item, "Valid", lambda: False)())

    payload = {
        "module": "scan", "run_id": stamp_id, "source": source,
        "applied_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "frame": FRAME_NAME, "rectangle": RECT_NAME,
        "mesh": MESH_NAME if mesh_inserted else None,
        "size_mm": report.get("plane", {}).get("size_mm"),
    }
    runs.write_active("scan", payload)
    return {"status": "inserted", "frame": FRAME_NAME, "rectangle": RECT_NAME,
            "mesh": MESH_NAME if mesh_inserted else None, "run_id": stamp_id,
            "source": source, "active": payload,
            "frame_valid": bool(getattr(frame, "Valid", lambda: True)()),
            "rectangle_valid": bool(getattr(rect, "Valid", lambda: True)())}
