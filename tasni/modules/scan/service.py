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
from ..calibration.poses import (generate_calibration_poses, projected_corner_coverage,
                                  select_diverse, select_diverse_with_coverage,
                                  viewing_angle_span)
from ..calibration.service import (
    BOARD_KEEPOUT_NAME as CALIB_BOARD_KEEPOUT_NAME,
    TARGET_PREFIX as CALIB_TARGET_PREFIX,
    _camera_hold, dry_tour_required, ensure_camera_tool, ensure_real_robot_link)
from .depth_gate import ScanGateThresholds, evaluate_depth_gate
from .plane import bounded_work_plane, work_plane_from_points
from .planner import ScanPlan, plan_scan
from .survey import SurveyThresholds, survey_surface
from .reconstruct import (ScanView, clean_measured_surface_mesh, cloud_points_m,
                          crop_box, fuse_views, look_point_from_views,
                          mesh_preview_points, planar_rectangle_mesh, save_mesh)

log = get_logger("tasni.scan")

# Minimum posed views for a usable fusion (a flat table tolerates few, but more =
# better coverage; below this the mesh/plane are unreliable).
SCAN_MIN_VIEWS = 4

# Names of the items insert creates in the station (replaced on each insert).
FRAME_NAME = "Tasni Work Frame"
RECT_NAME = "Tasni Work Surface"
MESH_NAME = "Tasni Scan Mesh"


@dataclass
class LockedScanSurface:
    frame: object
    reading: object
    survey: object
    gate_payload: dict
    seed_T: np.ndarray
    seed_joints: object
    locked_at: float


def _large_surface_crop_mm(scfg, K, image_size, look_mm: float) -> list[float]:
    """The generic work-square size used when the surface overruns the view.

    Fixed (``scfg.work_crop_mm``, default 1.0×1.0 m) rather than a fraction of the FOV:
    the operator aims the reticle at the work area and we project a standard square on
    the plane around it. ``K``/``image_size``/``look_mm`` are unused now (kept so the
    call sites — which have them handy — need not change)."""
    return [float(scfg.work_crop_mm[0]), float(scfg.work_crop_mm[1])]


def _outline_edge_angle_deg(outline_uv) -> float | None:
    uv = np.asarray(outline_uv, dtype=float).reshape(-1, 2)
    if len(uv) < 2:
        return None
    edges = np.roll(uv, -1, axis=0) - uv
    edge = edges[int(np.argmax(np.linalg.norm(edges, axis=1)))]
    angle = float(np.degrees(np.arctan2(edge[1], edge[0])))
    return ((angle + 45.0) % 90.0) - 45.0


def _aspect_ratio(values) -> float | None:
    try:
        a, b = [abs(float(v)) for v in values[:2]]
    except Exception:
        return None
    lo = max(min(a, b), 1e-9)
    return max(a, b) / lo


def _planned_surface_standoff_mm(
    scfg, K, image_size, reading, survey, full_frame_valid_frac: float | None = None
) -> float:
    """Best standoff for the measured surface, matching the target planner."""
    if survey is not None and getattr(survey, "detected", False):
        if (not getattr(survey, "fully_framed", False)
                and full_frame_valid_frac is not None
                and full_frame_valid_frac >= 0.95):
            return float(scfg.accurate_min_mm)
        try:
            plan = plan_scan(survey, K, image_size, scfg, cam_to_base_T=None)
            return float(plan.standoff_mm)
        except Exception:
            pass
    if getattr(reading, "distance_mm", None) is not None:
        return float(np.clip(float(reading.distance_mm),
                             float(scfg.accurate_min_mm),
                             float(scfg.accurate_max_mm)))
    return float(scfg.ideal_distance_mm)


def lock_scan_surface(services) -> LockedScanSurface:
    """Freeze one authoritative RGBD measurement and the matching robot pose."""
    cfg = services.config
    scfg = cfg.scan
    rdk: RdkIO = services.rdk
    K = cfg.camera.K
    ensure_camera_tool(services, log=_log_pub(services))
    if services.live.running:
        services.live.stop()
    rdk.use_camera_tool(cfg.robodk.camera_tool)

    with _camera_hold(services, "scan-surface-lock"):
        frame = services.camera.grab(with_depth=True, timeout=scfg.grab_timeout_s)
    reading = evaluate_depth_gate(
        frame.depth, K, scan_gate_thresholds(scfg), depth_scale=scfg.depth_scale)
    survey = survey_surface(
        frame.depth, K, _survey_thresholds(scfg), depth_scale=scfg.depth_scale)
    depth = np.asarray(frame.depth) if frame.depth is not None else np.zeros((0, 0))
    full_frame_valid_frac = float(np.mean(depth > 0)) if depth.size else 0.0
    surface_overruns_view = bool(
        survey.detected and not survey.fully_framed and full_frame_valid_frac >= 0.95)
    gate_payload = scan_gate_payload(reading, survey)
    ideal_distance = _planned_surface_standoff_mm(
        scfg, K, cfg.camera.size, reading, survey, full_frame_valid_frac)
    gate_payload["ideal_distance_mm"] = ideal_distance
    gate_payload["distance_tol_mm"] = float(scfg.distance_tol_mm)
    gate_payload["surface_mode"] = "crop" if surface_overruns_view else "full"
    final_gates = {
        "detected": bool(reading.gates.get("detected")),
        "distance": (
            reading.distance_mm is not None
            and abs(float(reading.distance_mm) - ideal_distance) <= float(scfg.distance_tol_mm)
        ),
        "angle": bool(reading.gates.get("angle")),
    }
    if survey.detected and survey.fully_framed:
        if survey.centroid_cam_mm is not None:
            final_gates["center"] = bool(
                abs(float(survey.centroid_cam_mm[0])) <= float(scfg.center_tol_mm)
                and abs(float(survey.centroid_cam_mm[1])) <= float(scfg.center_tol_mm))
            gate_payload["move_cam"] = [
                float(survey.centroid_cam_mm[0]),
                float(survey.centroid_cam_mm[1]),
                float((reading.distance_mm or ideal_distance) - ideal_distance),
            ]
            gate_payload["center_tol_mm"] = float(scfg.center_tol_mm)
        aspect = _aspect_ratio(survey.extent_mm) if survey.extent_mm is not None else None
        if (survey.outline_uv and len(survey.outline_uv) >= 2
                and (aspect is None or aspect >= float(scfg.edge_gate_min_aspect))):
            uv = np.asarray(survey.outline_uv, float)
            edges = np.roll(uv, -1, axis=0) - uv
            edge = edges[int(np.argmax(np.linalg.norm(edges, axis=1)))]
            angle = float(np.degrees(np.arctan2(edge[1], edge[0])))
            angle = ((angle + 45.0) % 90.0) - 45.0
            final_gates["edge"] = abs(angle) <= float(scfg.edge_align_tol_deg)
            gate_payload["yaw_a_deg"] = -angle
            gate_payload["edge_align_tol_deg"] = float(scfg.edge_align_tol_deg)
    elif survey.detected and not surface_overruns_view:
        final_gates["framed"] = False
    gate_payload["gates"] = {**gate_payload.get("gates", {}), **final_gates}
    gate_payload["ok"] = all(final_gates.values())
    if surface_overruns_view and reading.distance_mm is not None:
        gate_payload["crop_size_mm"] = _large_surface_crop_mm(
            scfg, K, cfg.camera.size, float(reading.distance_mm))

    ok, jpeg = cv2.imencode(".jpg", frame.color)
    if ok:
        services.bus.publish(JobEvent("frame", {
            "jpeg_b64": base64.b64encode(jpeg.tobytes()).decode("ascii")}))
    services.bus.publish(JobEvent("gate", {**gate_payload, "live": False}))

    if not gate_payload["ok"]:
        bad = [name for name, good in final_gates.items() if not good]
        raise RuntimeError(
            "surface is not ready — fix " + ", ".join(bad)
            + f" (distance {reading.distance_mm and round(reading.distance_mm)} mm, "
            + f"target {round(ideal_distance)} mm, "
            + f"tilt {reading.tilt_deg and round(reading.tilt_deg, 1)}°).")
    seed_T = rdk.camera_pose_T()
    try:
        seed_joints = rdk.current_joints()
    except Exception:
        seed_joints = None
    return LockedScanSurface(
        frame=frame, reading=reading, survey=survey, gate_payload=gate_payload,
        seed_T=np.asarray(seed_T, float), seed_joints=seed_joints,
        locked_at=time.monotonic())


def scan_gate_thresholds(scfg) -> ScanGateThresholds:
    """One source of truth so the live preview and the authoritative grab gate
    identically (mirrors calibration's ``gate_thresholds``)."""
    return ScanGateThresholds(
        ideal_distance_mm=scfg.ideal_distance_mm,
        distance_tol_mm=scfg.distance_tol_mm,
        max_tilt_deg=scfg.max_tilt_deg,
        center_patch_frac=scfg.center_patch_frac,
        min_valid_depth_frac=scfg.min_valid_depth_frac)


def _survey_thresholds(scfg) -> SurveyThresholds:
    return SurveyThresholds(
        accurate_min_mm=scfg.accurate_min_mm,
        accurate_max_mm=scfg.accurate_max_mm,
        survey_max_tilt_deg=scfg.survey_max_tilt_deg,
        grid_target_px=scfg.grid_target_px,
        work_crop_mm=tuple(scfg.work_crop_mm),
    )


def _backproject_depth(depth: np.ndarray, K: np.ndarray, *,
                       depth_scale: float = 1000.0) -> np.ndarray:
    """Back-project a raw uint16 depth image to camera-frame 3D points (mm)."""
    d = np.asarray(depth, float)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    ys, xs = np.nonzero(d > 0)
    if len(ys) == 0:
        return np.zeros((0, 3), float)
    z_mm = d[ys, xs] / float(depth_scale) * 1000.0
    return np.column_stack([(xs - cx) / fx * z_mm, (ys - cy) / fy * z_mm, z_mm])


def _densify_quad(corners: np.ndarray, n: int = 6) -> np.ndarray:
    """Grid of ``n*n`` points bilinearly filling a cyclic-ordered quad (4,3).

    ``corners`` are the oriented-rectangle corners in consecutive order (as
    :func:`plane._oriented_rectangle` returns them). Used to turn the 4-corner
    surface footprint into a point cloud the coverage selector/metric can tile —
    4 corners alone only land in the grid's corner cells.
    """
    c = np.asarray(corners, float).reshape(4, 3)
    s, t = np.meshgrid(np.linspace(0.0, 1.0, n), np.linspace(0.0, 1.0, n))
    s = s.ravel()[:, None]
    t = t.ravel()[:, None]
    bottom = c[0] + s * (c[1] - c[0])    # edge c0->c1
    top = c[3] + s * (c[2] - c[3])       # edge c3->c2
    return bottom + t * (top - bottom)


def _surface_footprint_base(survey, seed_T: np.ndarray) -> np.ndarray | None:
    """The measured surface rectangle as a grid of points in the robot base frame.

    Returns ``None`` when the survey has no trustworthy rectangle (camera-frame
    corners). ``seed_T`` is the camera pose in base, so it maps the survey's
    camera-frame corners into base — the frame the candidate poses live in.
    """
    corners_cam = getattr(survey, "corners_cam_mm", None)
    if corners_cam is None:
        return None
    corners_cam = np.asarray(corners_cam, float).reshape(-1, 3)
    if corners_cam.shape[0] != 4:
        return None
    R = np.asarray(seed_T[:3, :3], float)
    t = np.asarray(seed_T[:3, 3], float)
    corners_base = (R @ corners_cam.T).T + t
    return _densify_quad(corners_base, n=6)


def _save_views(views, K, width, height, run_dir, *, depth_scale, log) -> None:
    """Persist each captured view (color JPEG + 16-bit depth PNG + camera pose) under
    ``<run>/views/`` for a later camera-perspective coverage overlay.

    Diagnostic only (``scan.save_views``). The depth is written as a single-channel
    16-bit PNG (lossless, the raw mm units), color as JPEG; ``views.json`` records K,
    image size, depth scale and each view's base->camera pose so the fused cloud can
    be re-projected into any view.
    """
    from pathlib import Path

    vdir = Path(run_dir) / "views"
    vdir.mkdir(parents=True, exist_ok=True)
    meta = {"K": np.asarray(K, float).tolist(), "size": [int(width), int(height)],
            "depth_scale": float(depth_scale), "views": []}
    for i, v in enumerate(views):
        cv2.imwrite(str(vdir / f"view_{i:02d}.jpg"), v.color,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(str(vdir / f"depth_{i:02d}.png"),
                    np.ascontiguousarray(np.asarray(v.depth, np.uint16)))
        meta["views"].append({"index": i,
                              "pose_T_mm": np.asarray(v.pose_T, float).tolist()})
    (vdir / "views.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log(f"diagnostics: saved {len(views)} per-view color+depth frames to {vdir.name}/ "
        f"(scan.save_views) — enables the camera-perspective coverage overlay")


def _reference_locate(services, frame, survey, seed_T: np.ndarray,
                      plan: ScanPlan) -> "ScanResult":
    """Reference mode: fit plane + rectangle from a single survey depth frame.

    No robot tour, no TSDF fusion. Returns a ScanResult with mesh_obj_path=None.
    The result is ready to insert (frame + rectangle) without a Run step.
    """
    cfg = services.config
    scfg = cfg.scan
    K = cfg.camera.K
    pub = _log_pub(services)

    pts_cam_mm = _backproject_depth(frame.depth, K, depth_scale=scfg.depth_scale)
    if len(pts_cam_mm) == 0:
        raise RuntimeError("reference locate: no valid depth pixels in the survey frame")

    R = np.asarray(seed_T[:3, :3], float)
    t = np.asarray(seed_T[:3, 3], float)
    pts_base_m = ((R @ pts_cam_mm.T).T + t) / 1000.0   # mm → m

    try:
        wp = work_plane_from_points(
            pts_base_m, distance=scfg.ransac_distance_m,
            n_iterations=scfg.ransac_iterations, min_inlier_frac=scfg.min_inlier_frac)
    except ValueError as e:
        raise RuntimeError(f"reference locate: plane fit failed — {e}") from e

    frame_T_mm = wp.frame_T.copy()
    frame_T_mm[:3, 3] *= 1000.0
    corners_mm = wp.corners * 1000.0

    stamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = new_run_dir("scan", stamp)
    sz = [float(wp.size[0] * 1000.0), float(wp.size[1] * 1000.0)]

    report = {
        "module": "scan", "stamp": stamp, "run_dir": str(run_dir),
        "mode": "reference",
        "n_views": 1, "n_points": int(len(pts_base_m)),
        "mesh_vertices": 0, "mesh_triangles": 0, "mesh_file": None,
        "plane": {
            "frame_T_mm": frame_T_mm.tolist(),
            "corners_mm": corners_mm.tolist(),
            "size_mm": sz,
            "normal": wp.normal.tolist(),
            "inlier_frac": float(wp.inlier_frac),
            "inlier_count": int(wp.inlier_count),
        },
    }
    try:
        (run_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        runs.write_meta("scan", stamp, {"module": "scan", "stamp": stamp, "mode": "reference",
                                        "tool_name": cfg.robodk.camera_tool})
    except Exception:
        pass

    pub(f"reference surface: {sz[0]:.0f}×{sz[1]:.0f} mm from single frame "
        f"(standoff ~{survey.standoff_mm and round(survey.standoff_mm)} mm, "
        f"inliers {wp.inlier_frac:.0%}). Review, then Insert.")

    return ScanResult(report=report, run_dir=str(run_dir),
                      frame_T_mm=frame_T_mm, corners_mm=corners_mm, mesh_obj_path=None)


def _log_pub(services):
    return lambda m: services.bus.publish(JobEvent("log", {"message": m}))


def scan_gate_payload(reading, survey) -> dict:
    """Publish the centre-patch gate plus full-frame survey overlays.

    The centre patch stays authoritative for target creation because it preserves
    the calibration-style workflow: use the current reachable camera pose as the
    cone seed, then orbit the measured standoff. The full-frame survey remains
    useful for the HUD and for voxel planning when its extent is trustworthy.
    """
    payload = reading.to_dict()
    if survey is not None and survey.detected:
        payload.update({
            "fully_framed": survey.fully_framed,
            "outline_uv": survey.outline_uv,
            "grid_uv": survey.grid_uv,
            "grid_spacing_mm": survey.grid_spacing_mm,
            "extent_mm": list(survey.extent_mm) if survey.extent_mm is not None else None,
            "fov_deg": list(survey.fov_deg),
            "points_uv": survey.points_uv,
        })
        payload["gates"] = {**payload.get("gates", {}),
                            "framed": bool(survey.fully_framed)}
    return payload


def live_scan_telemetry_payload(raw: dict | None, scfg,
                                previous_ideal_mm: float | None = None,
                                camera_cfg=None) -> dict:
    """Apply workstation scan thresholds to compact Jetson plane telemetry."""
    if not raw:
        return {}
    stamp = raw.get("timestamp")
    if stamp is not None and time.time() - float(stamp) > 2.0:
        return {}
    th = scan_gate_thresholds(scfg)
    detected = bool(raw.get("detected"))
    valid_frac = float(raw.get("valid_frac", 0.0))
    if not detected:
        return {
            "detected": False, "distance_mm": None, "tilt_deg": None,
            "valid_frac": valid_frac,
            "gates": {"detected": False, "distance": False, "angle": False},
            "ok": False,
            "ideal_distance_mm": th.ideal_distance_mm,
            "distance_tol_mm": th.distance_tol_mm,
            "max_tilt_deg": th.max_tilt_deg,
            "live": True,
        }
    distance = float(raw["distance_mm"])
    tilt = float(raw["tilt_deg"])
    fully_framed = raw.get("fully_framed")
    surface_mode = raw.get("surface_mode", "full")
    outline_uv = raw.get("outline_uv")
    corners_color = raw.get("rectangle_corners_color_mm")
    if camera_cfg is not None and corners_color is not None:
        corners = np.asarray(corners_color, dtype=np.float64).reshape(-1, 3)
        projected, _ = cv2.projectPoints(
            corners, np.zeros(3), np.zeros(3), camera_cfg.K, camera_cfg.dist)
        W, H = camera_cfg.size
        calibrated_uv = projected.reshape(-1, 2) / np.array([W, H], dtype=float)
        if np.all(np.isfinite(calibrated_uv)):
            outline_uv = calibrated_uv.tolist()
            edge_angle = _outline_edge_angle_deg(calibrated_uv)
            color_margin = 0.015
            fully_framed = bool(raw.get("depth_fully_framed")) and bool(np.all(
                (calibrated_uv[:, 0] >= color_margin)
                & (calibrated_uv[:, 0] <= 1.0 - color_margin)
                & (calibrated_uv[:, 1] >= color_margin)
                & (calibrated_uv[:, 1] <= 1.0 - color_margin)))
            max_center_span = float(np.max(np.abs(calibrated_uv - 0.5)))
            raw = {
                **raw,
                "edge_angle_deg": edge_angle,
                "color_fit_standoff_per_margin_mm":
                    distance * 2.0 * max_center_span,
            }
    ideal_distance = float(th.ideal_distance_mm)
    fit_per_margin = raw.get("color_fit_standoff_per_margin_mm")
    extent = raw.get("extent_mm")
    if surface_mode == "crop":
        # The surface overruns the view: do not chase an impossible whole-table
        # framing distance. Work close, then project the generic reticle work square.
        ideal_distance = float(scfg.accurate_min_mm)
    elif fit_per_margin is not None:
        # Continuous on both sides of the color-frame boundary, so moving toward
        # the recommendation cannot flip the policy from 300 to 500 and back.
        ideal_distance = float(np.clip(
            float(fit_per_margin) * float(scfg.frame_margin),
            float(scfg.accurate_min_mm), float(scfg.accurate_max_mm)))
        candidate = round(ideal_distance / 10.0) * 10.0
        # Recommendation deadband: 410/420 is sensor/fitting noise, not a useful
        # instruction. Hold the previous target until the estimate moves >=20 mm.
        if previous_ideal_mm is not None and abs(candidate - previous_ideal_mm) < 20.0:
            ideal_distance = float(previous_ideal_mm)
        else:
            ideal_distance = candidate
    elif extent is not None and camera_cfg is not None:
        try:
            sx, sy = [float(v) for v in extent]
            W, H = camera_cfg.size
            K = camera_cfg.K
            ideal_distance = float(np.clip(
                max(float(scfg.frame_margin) * sx * float(K[0, 0]) / float(W),
                    float(scfg.frame_margin) * sy * float(K[1, 1]) / float(H)),
                float(scfg.accurate_min_mm), float(scfg.accurate_max_mm)))
        except Exception:
            pass
    crop_size = None
    if surface_mode == "crop":
        # Generic fixed work square (the surface overruns the view; its edges are not
        # trustworthy). Matches the host lock/run crop and the server's live overlay.
        crop_size = [float(scfg.work_crop_mm[0]), float(scfg.work_crop_mm[1])]
    gates = {
        "detected": True,
        "distance": abs(distance - ideal_distance) <= th.distance_tol_mm,
        "angle": tilt <= th.max_tilt_deg,
    }
    center_cam = raw.get("surface_center_cam_mm")
    edge_angle = raw.get("edge_angle_deg")
    finite_surface = surface_mode == "full"
    edge_aspect = _aspect_ratio(raw.get("rectangle_size_mm") or extent or [])
    edge_gate_reliable = (
        edge_aspect is None or edge_aspect >= float(scfg.edge_gate_min_aspect))
    if finite_surface and center_cam is not None:
        gates["center"] = bool(
            abs(float(center_cam[0])) <= float(scfg.center_tol_mm)
            and abs(float(center_cam[1])) <= float(scfg.center_tol_mm))
    if finite_surface and edge_angle is not None and edge_gate_reliable:
        gates["edge"] = abs(float(edge_angle)) <= float(scfg.edge_align_tol_deg)
    if fully_framed is not None:
        gates["framed"] = bool(fully_framed)
    ok_gates = dict(gates)
    if surface_mode == "crop":
        ok_gates.pop("framed", None)
    return {
        "detected": True,
        "distance_mm": distance,
        "tilt_deg": tilt,
        "valid_frac": valid_frac,
        "gates": gates,
        # Large crop planes intentionally remain unframed; finite platforms must
        # frame their measured edges before the one-second lock hold can complete.
        "ok": all(bool(v) for v in ok_gates.values()),
        "ideal_distance_mm": ideal_distance,
        "distance_tol_mm": th.distance_tol_mm,
        "max_tilt_deg": th.max_tilt_deg,
        "move_cam": [
            float(center_cam[0]) if finite_surface and center_cam is not None else 0.0,
            float(center_cam[1]) if finite_surface and center_cam is not None else 0.0,
            distance - ideal_distance,
        ],
        "center_tol_mm": float(scfg.center_tol_mm),
        "yaw_a_deg": (-float(edge_angle)
                      if finite_surface and edge_angle is not None and edge_gate_reliable else None),
        "edge_align_tol_deg": float(scfg.edge_align_tol_deg),
        "tilt_b_deg": raw.get("tilt_b_deg"),
        "tilt_c_deg": raw.get("tilt_c_deg"),
        "normal_cam": raw.get("normal_cam"),
        "centroid_cam_mm": raw.get("centroid_cam_mm"),
        "fully_framed": fully_framed,
        "surface_mode": surface_mode,
        "extent_mm": extent,
        "rectangle_size_mm": raw.get("rectangle_size_mm"),
        "crop_size_mm": crop_size,
        "outline_uv": outline_uv,
        "visible_outline_uv": raw.get("visible_outline_uv"),
        "points_uv": raw.get("points_uv"),
        "measurement_ts": stamp,
        "live": True,
    }


def _as_float_array(value, shape_last: int | None = None) -> np.ndarray | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=float)
    except Exception:
        return None
    if not np.all(np.isfinite(arr)):
        return None
    if shape_last is not None and (arr.ndim == 0 or arr.shape[-1] != shape_last):
        return None
    return arr


def _lerp(a, b, alpha: float):
    return (1.0 - alpha) * np.asarray(a, dtype=float) + alpha * np.asarray(b, dtype=float)


def _smooth_scalar(prev: dict, cur: dict, key: str, alpha: float) -> None:
    if prev.get(key) is None or cur.get(key) is None:
        return
    try:
        cur[key] = float(_lerp(float(prev[key]), float(cur[key]), alpha))
    except Exception:
        pass


def _smooth_vector(prev: dict, cur: dict, key: str, alpha: float,
                   *, shape_last: int | None = None) -> None:
    a = _as_float_array(prev.get(key), shape_last)
    b = _as_float_array(cur.get(key), shape_last)
    if a is None or b is None or a.shape != b.shape:
        return
    out = _lerp(a, b, alpha)
    if key == "normal_cam":
        n = float(np.linalg.norm(out))
        if n > 1e-9:
            out = out / n
    cur[key] = out.tolist()


def _payload_center_mm(payload: dict) -> np.ndarray | None:
    move = _as_float_array(payload.get("move_cam"), 3)
    if move is None:
        return None
    return move[:2]


def _payload_outline_uv(payload: dict) -> np.ndarray | None:
    outline = _as_float_array(payload.get("outline_uv"), 2)
    if outline is None or outline.ndim != 2 or len(outline) < 3:
        return None
    return outline


def _align_polygon_like(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Return candidate with cyclic/reversed corner order closest to reference."""
    ref = np.asarray(reference, dtype=float)
    cand = np.asarray(candidate, dtype=float)
    if ref.shape != cand.shape or cand.ndim != 2 or len(cand) < 3:
        return cand
    variants = []
    for arr in (cand, cand[::-1]):
        for shift in range(len(arr)):
            variants.append(np.roll(arr, shift, axis=0))
    return min(variants, key=lambda v: float(np.mean(np.linalg.norm(v - ref, axis=1))))


def _should_reset_live_smoothing(prev: dict, cur: dict, scfg) -> bool:
    if not prev or not prev.get("detected") or not cur.get("detected"):
        return True
    if prev.get("distance_mm") is not None and cur.get("distance_mm") is not None:
        if abs(float(cur["distance_mm"]) - float(prev["distance_mm"])) > float(scfg.live_aim_reset_distance_mm):
            return True
    return False


def stabilize_live_scan_payload(current: dict, previous: dict | None, scfg) -> dict:
    """Temporal smoothing for live scan HUD telemetry only.

    The lock/create-target path still grabs an authoritative raw RGBD frame. This
    filter prevents per-frame RealSense plane/edge noise from making a static robot
    look like it is moving in the live aiming UI.
    """
    if not current or current.get("live") is not True:
        return current
    if previous is None or _should_reset_live_smoothing(previous, current, scfg):
        return current

    out = dict(current)
    if previous.get("surface_mode") != current.get("surface_mode"):
        out["surface_mode"] = previous.get("surface_mode")
        out["crop_size_mm"] = previous.get("crop_size_mm", out.get("crop_size_mm"))
        for key in ("outline_uv", "visible_outline_uv", "grid_uv", "points_uv"):
            if previous.get(key) is not None:
                out[key] = previous.get(key)
    if previous.get("fully_framed") != current.get("fully_framed"):
        out["fully_framed"] = previous.get("fully_framed")
    for key in ("outline_uv", "visible_outline_uv"):
        prev_poly = _as_float_array(previous.get(key), 2)
        cur_poly = _as_float_array(out.get(key), 2)
        if prev_poly is not None and cur_poly is not None and prev_poly.shape == cur_poly.shape:
            out[key] = _align_polygon_like(prev_poly, cur_poly).tolist()
    alpha = float(np.clip(getattr(scfg, "live_aim_smoothing_alpha", 0.35), 0.05, 1.0))
    for key in (
        "distance_mm", "tilt_deg", "tilt_b_deg", "tilt_c_deg",
        "yaw_a_deg", "ideal_distance_mm",
    ):
        _smooth_scalar(previous, out, key, alpha)
    for key, shape_last in (
        ("move_cam", 3),
        ("normal_cam", 3),
        ("centroid_cam_mm", 3),
        ("extent_mm", None),
        ("rectangle_size_mm", None),
        ("outline_uv", 2),
        ("visible_outline_uv", 2),
    ):
        _smooth_vector(previous, out, key, alpha, shape_last=shape_last)

    gates = dict(out.get("gates") or {})
    prev_gates = dict(previous.get("gates") or {})
    distance = out.get("distance_mm")
    ideal = out.get("ideal_distance_mm")
    tilt = out.get("tilt_deg")
    if distance is not None and ideal is not None:
        tol = float(out.get("distance_tol_mm", scfg.distance_tol_mm))
        if prev_gates.get("distance"):
            tol += float(getattr(scfg, "live_aim_distance_hysteresis_mm", 20.0))
        gates["distance"] = abs(float(distance) - float(ideal)) <= tol
    if tilt is not None:
        tol = float(out.get("max_tilt_deg", scfg.max_tilt_deg))
        if prev_gates.get("angle"):
            tol += float(getattr(scfg, "live_aim_angle_hysteresis_deg", 1.0))
        gates["angle"] = float(tilt) <= tol
    if "center" in gates and out.get("move_cam") is not None:
        mv = _as_float_array(out.get("move_cam"), 3)
        if mv is not None:
            tol = float(out.get("center_tol_mm", scfg.center_tol_mm))
            if prev_gates.get("center"):
                tol += float(getattr(scfg, "live_aim_center_hysteresis_mm", 15.0))
            gates["center"] = (
                abs(float(mv[0])) <= tol
                and abs(float(mv[1])) <= tol)
    if "edge" in gates and out.get("yaw_a_deg") is not None:
        tol = float(out.get("edge_align_tol_deg", scfg.edge_align_tol_deg))
        if prev_gates.get("edge"):
            tol += float(getattr(scfg, "live_aim_edge_hysteresis_deg", 2.0))
        gates["edge"] = abs(float(out["yaw_a_deg"])) <= tol
    if out.get("fully_framed") is not None:
        gates["framed"] = bool(out.get("fully_framed"))
    ok_gates = dict(gates)
    if out.get("surface_mode") == "crop":
        ok_gates.pop("framed", None)
    out["gates"] = gates
    out["ok"] = all(bool(v) for v in ok_gates.values())
    out["stabilized"] = True
    return out


def generate_scan_targets(services, locked: LockedScanSurface | None = None) -> dict:
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
    W, H = cfg.camera.size
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

    if locked is None:
        locked = lock_scan_surface(services)
    elif time.monotonic() - locked.locked_at > 120.0:
        raise RuntimeError("locked surface expired — reposition and lock it again")
    frame, reading, survey = locked.frame, locked.reading, locked.survey
    gate_payload = locked.gate_payload
    seed_T, seed_joints = locked.seed_T, locked.seed_joints
    current_T = np.asarray(rdk.camera_pose_T(), float)
    moved_mm = float(np.linalg.norm(current_T[:3, 3] - seed_T[:3, 3]))
    rel_R = seed_T[:3, :3].T @ current_T[:3, :3]
    moved_deg = float(np.degrees(np.arccos(np.clip(
        (np.trace(rel_R) - 1.0) / 2.0, -1.0, 1.0))))
    if moved_mm > 5.0 or moved_deg > 1.5:
        raise RuntimeError(
            f"robot moved after surface lock ({moved_mm:.1f} mm, {moved_deg:.1f}°) — "
            "reposition and lock the surface again")

    look = float(reading.distance_mm or scfg.look_distance_mm)
    target_center = None
    target_count = scfg.pose_count
    target_cone_deg = scfg.cone_half_angle_deg
    target_normal = None
    min_perpendicular_mm = None
    plan = None
    planned_voxel_m = scfg.voxel_size_m
    extent_mm = (list(survey.extent_mm)
                 if survey.detected and survey.extent_mm is not None else None)
    crop_size_mm = None
    if survey.detected and survey.extent_mm is not None:
        if survey.fully_framed:
            plan = plan_scan(survey, K, (W, H), scfg, cam_to_base_T=seed_T)
            planned_voxel_m = plan.voxel_size_m
            if plan.mode == "quality" and plan.aims:
                look = float(plan.standoff_mm)
                target_center = np.asarray(plan.aims[0].point_base_mm, float)
                target_normal = -np.asarray(plan.aims[0].view_dir_base, float)
                min_perpendicular_mm = float(plan.aims[0].min_perpendicular_mm)
                # Preserve an operator-configured denser tour (12 by default), while
                # still allowing a raised-surface plan to request more viewpoints.
                target_count = max(int(scfg.pose_count), int(plan.aims[0].n_views))
                target_cone_deg = float(plan.cone_half_angle_deg)
            pub(f"survey: {survey.extent_mm[0]:.0f}×{survey.extent_mm[1]:.0f} mm surface "
                f"at {survey.standoff_mm:.0f} mm; planned scan targets "
                f"(standoff {look:.0f} mm, cone {target_cone_deg:.0f}°, "
                f"views {target_count}), "
                f"voxel={planned_voxel_m*1000:.1f} mm")
            for w in plan.warnings:
                pub(f"WARNING (survey): {w}")
        else:
            # The intended surface continues beyond the image. Define a useful,
            # camera-centred work region instead of pretending the visible border is
            # the table edge. The final multi-view fit preserves its inclination.
            crop_size_mm = _large_surface_crop_mm(scfg, K, (W, H), look)
            pub("survey: surface outline touches the image border; using the stable "
                f"centre plane and a {crop_size_mm[0]:.0f}×{crop_size_mm[1]:.0f} mm "
                "camera-centred work crop")
    else:
        pub("survey: no trustworthy full-frame extent; using the stable centre-patch "
            "standoff gate and default cone/voxel settings for targets")

    prior = rdk.list_targets(prefix)
    if prior:
        rdk.delete_items(prior)
    calib_prior = rdk.list_targets(CALIB_TARGET_PREFIX)
    removed_calib_keepout = False
    if calib_prior:
        rdk.delete_items(calib_prior)
    if rdk.item_exists(CALIB_BOARD_KEEPOUT_NAME):
        rdk.delete_items([CALIB_BOARD_KEEPOUT_NAME])
        removed_calib_keepout = True
    if calib_prior or removed_calib_keepout:
        pub(f"cleared {len(calib_prior)} calibration target(s)"
            + (" and board keep-out" if removed_calib_keepout else "")
            + " before creating scan targets")

    candidates = generate_calibration_poses(
        seed_T, count=target_count, look_distance_mm=look,
        cone_half_angle_deg=target_cone_deg,
        roll_max_deg=scfg.roll_max_deg, distance_jitter=scfg.distance_jitter,
        target_center=target_center, target_normal=target_normal,
        min_perpendicular_mm=min_perpendicular_mm)
    reachable = [(i, T) for i, T in enumerate(candidates) if rdk.is_reachable(T)]
    n_reach = len(reachable)
    reachable_before_collision = list(reachable)
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
    collision_filter_bypassed = False
    pair_examples: list[str] = []
    reach_joints: list = [None] * n_reach
    if scfg.collision_filter:
        mask, col_checked, jts, col_details = rdk.screen_collisions(
            [T for _, T in reachable],
            guard_skip=guard_skip,
            ignore_pairs=scfg.collision_ignore_pairs,
            return_details=True)
        kept = [k for k in range(n_reach) if mask[k]]
        if col_checked:
            n_collide = n_reach - len(kept)
        reachable = [reachable[k] for k in kept]
        reach_joints = [jts[k] for k in kept]
        services.bus.publish(JobEvent("log", {"message":
            f"collision screen: {'ACTIVE' if col_checked else 'unavailable'}; swept "
            f"{n_reach} reachable pose(s), {n_collide} collided and were dropped"}))
        if col_checked and n_collide:
            for d in col_details.get("poses", []):
                if d.get("collides") and d.get("pairs"):
                    for p in d["pairs"]:
                        if p not in pair_examples:
                            pair_examples.append(p)
                        if len(pair_examples) >= 8:
                            break
                if len(pair_examples) >= 8:
                    break
            if pair_examples:
                services.bus.publish(JobEvent("log", {"message":
                    "collision pairs causing dropped scan targets: "
                    + "; ".join(pair_examples)}))
        if col_checked and len(reachable) < SCAN_MIN_VIEWS:
            if scfg.collision_filter_hard_fail:
                raise RuntimeError(
                    f"only {len(reachable)} collision-free poses ({n_collide} of {n_reach} "
                    f"would collide) — jog to a more open part of the workspace and retry")
            collision_filter_bypassed = True
            reachable = reachable_before_collision
            reach_joints = [None] * len(reachable)
            services.bus.publish(JobEvent("log", {"message":
                "WARNING: RoboDK reported too many scan candidate poses as colliding, "
                "so target creation is continuing with reachable poses only. This is "
                "often a noisy/stale collision map or oversized wall/fixture collision "
                "geometry. Inspect the targets in RoboDK and run the dry tour before "
                "moving the real robot; set scan.collision_filter_hard_fail to true "
                "for strict refusal."}))

    n_usable = len(reachable)
    reach_T = [T for _, T in reachable]
    # The scan's whole job is to TILE the surface — every region needs to land in
    # frame across the kept views. Plain rotation-diversity selection (what hand-eye
    # calibration wants) maximizes geodesic rotation spread but is azimuth-blind, so
    # the kept set can cluster to one side and leave a patch of the board uncovered
    # in every view. When the survey gives a trustworthy rectangle, select for
    # surface COVERAGE first (rotation spread as the tie-break) — mirroring
    # calibration's intrinsic-coverage selection.
    footprint_base = _surface_footprint_base(survey, seed_T)
    if footprint_base is not None:
        sel = select_diverse_with_coverage(
            reach_T, min(target_count, n_usable), footprint_base, K, (W, H),
            seed_fwd=seed_T[:3, 2])
    else:
        sel = select_diverse(reach_T, min(target_count, n_usable), seed_fwd=seed_T[:3, 2])
    chosen = [(reachable[k][0], reachable[k][1], reach_joints[k]) for k in sel]

    _, eff_max, eff_mean = viewing_angle_span([T for _, T, _ in chosen], seed_T[:3, 2])
    # Predicted surface coverage: fraction of the footprint grid the kept views tile.
    # A low value is exactly the "one part of the board never captured" failure, so
    # surface it BEFORE the run rather than discovering the hole in the fused mesh.
    surface_coverage = None
    if footprint_base is not None:
        surface_coverage, _ = projected_corner_coverage(
            [T for _, T, _ in chosen], footprint_base, K, (W, H))
    if (surface_coverage is not None
            and surface_coverage < float(scfg.min_surface_coverage)):
        message = (
            f"the chosen views tile only {surface_coverage:.0%} of the surface "
            f"(< {scfg.min_surface_coverage:.0%}) — part of the surface would not "
            "be captured. Re-seed at a more central/open view or move farther back "
            "until the surface stays framed, then Create targets again.")
        hard_fail_coverage = (
            getattr(scfg, "surface_coverage_hard_fail", False)
            and crop_size_mm is None)
        if hard_fail_coverage:
            raise RuntimeError(message)
        services.bus.publish(JobEvent("log", {"message": f"WARNING: {message}"}))

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
    extent_txt = (f"; extent {round(extent_mm[0])}×{round(extent_mm[1])} mm"
                  if extent_mm else "")
    collide_note = (f"; collision filter bypassed after {n_collide} reported collision(s)"
                    if collision_filter_bypassed else
                    (f"; {n_collide} dropped for collision" if col_checked and n_collide
                     else ("; collision-checked" if col_checked
                           else "; collisions NOT checked")))
    cover_note = (f"; predicted surface coverage {surface_coverage:.0%}"
                  if surface_coverage is not None else "")
    services.bus.publish(JobEvent("log", {"message":
        f"created {len(created)} scan targets (standoff ~{look:.0f} mm{extent_txt}; "
        f"{n_reach}/{len(candidates)} candidates reachable; effective cone "
        f"~{eff_max:.0f}° of {target_cone_deg:.0f}°{collide_note}{cover_note}) — "
        f"inspect them in RoboDK"}))
    return {"mode": "quality", "created": len(created), "targets": created,
            "look_distance_mm": look,
            "gate": gate_payload, "candidates_reachable": n_reach,
            "candidates_total": len(candidates), "collisions_checked": col_checked,
            "candidates_collided": n_collide, "effective_cone_deg": round(eff_max, 1),
            "collision_pairs": pair_examples,
            "surface_coverage": (round(surface_coverage, 3)
                                 if surface_coverage is not None else None),
            "planned_cone_deg": target_cone_deg, "planned_views": target_count,
            "camera_tool_offset_mm": round(tool_offset_mm, 1),
            "calibration_on_file": tool_offset_mm >= 15.0,
            "collision_filter_enabled": scfg.collision_filter,
            "collision_filter_bypassed": collision_filter_bypassed,
            "extent_mm": extent_mm,
            "crop_size_mm": crop_size_mm,
            "voxel_size_m": planned_voxel_m,
            "plan": plan.to_dict() if plan is not None else None}


# -- capture + reconstruct job ----------------------------------------------
@dataclass
class ScanParams:
    save_artifacts: bool = True
    voxel_size_m: float | None = None   # None → use ScanConfig default
    crop_size_mm: tuple[float, float] | None = None
    surface_size_mm: tuple[float, float] | None = None


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
                   run_dir, stamp, voxel_size_m: float, mesh_spacing_m: float,
                   frames_per_pose: int, mesh_stats: dict | None = None,
                   coverage: dict | None = None,
                   mesh_kind: str = "fitted_flat_surface") -> dict:
    return {
        "module": "scan", "stamp": stamp, "run_dir": str(run_dir),
        "n_views": int(n_views), "n_points": int(n_points),
        "mesh_vertices": int(len(mesh.vertices)),
        "mesh_triangles": int(len(mesh.triangles)),
        "mesh_file": "mesh.obj",
        "mesh_kind": mesh_kind,
        "measured_mesh_file": "measured_tsdf_mesh.ply",
        "reference_mesh_file": "work_surface_rect.obj",
        "raw_mesh_file": "raw_tsdf_mesh.ply",
        "mesh_cleaning": mesh_stats or {},
        "coverage": coverage or {},
        "quality": {
            "voxel_size_mm": float(voxel_size_m * 1000.0),
            "surface_mesh_spacing_mm": float(mesh_spacing_m * 1000.0),
            "frames_per_pose": int(frames_per_pose),
        },
        "plane": {
            "frame_T_mm": np.asarray(frame_T_mm, float).tolist(),
            "corners_mm": np.asarray(corners_mm, float).tolist(),
            "size_mm": [float(wp.size[0] * 1000.0), float(wp.size[1] * 1000.0)],
            "normal": wp.normal.tolist(),
            "inlier_frac": float(wp.inlier_frac),
            "inlier_count": int(wp.inlier_count),
        },
    }


def _surface_coverage(points_m: np.ndarray, wp, *, bin_m: float,
                      edge_band_m: float) -> dict:
    """Occupancy of measured mesh vertices inside the work rectangle."""
    pts = np.asarray(points_m, dtype=float).reshape(-1, 3)
    empty_edges = {"x_min": 0.0, "x_max": 0.0, "y_min": 0.0, "y_max": 0.0}
    if len(pts) == 0:
        return {"point_count": 0, "fill": 0.0, "interior": 0.0,
                "edges": empty_edges, "weakest_edge": 0.0}
    R = np.asarray(wp.frame_T[:3, :3], dtype=float)
    origin = np.asarray(wp.frame_T[:3, 3], dtype=float)
    local = (pts - origin) @ R
    corners_local = (np.asarray(wp.corners, dtype=float) - origin) @ R
    xmin, xmax = float(corners_local[:, 0].min()), float(corners_local[:, 0].max())
    ymin, ymax = float(corners_local[:, 1].min()), float(corners_local[:, 1].max())
    inside = ((local[:, 0] >= xmin) & (local[:, 0] <= xmax)
              & (local[:, 1] >= ymin) & (local[:, 1] <= ymax))
    if not np.any(inside):
        return {"point_count": int(len(pts)), "fill": 0.0, "interior": 0.0,
                "edges": empty_edges, "weakest_edge": 0.0}
    B = max(float(bin_m), 1e-6)
    nx = max(1, int(np.ceil((xmax - xmin) / B)))
    ny = max(1, int(np.ceil((ymax - ymin) / B)))
    ix = np.floor((local[inside, 0] - xmin) / B).astype(int)
    iy = np.floor((local[inside, 1] - ymin) / B).astype(int)
    ix = np.clip(ix, 0, nx - 1)
    iy = np.clip(iy, 0, ny - 1)
    occ = np.zeros((nx, ny), bool)
    occ[ix, iy] = True
    band = max(1, int(round(float(edge_band_m) / B)))
    interior = occ[band:-band, band:-band].mean() if nx > 2 * band and ny > 2 * band else occ.mean()
    edges = {
        "x_min": float(occ[:band, :].mean()),
        "x_max": float(occ[-band:, :].mean()),
        "y_min": float(occ[:, :band].mean()),
        "y_max": float(occ[:, -band:].mean()),
    }
    return {
        "point_count": int(len(pts)),
        "bin_mm": float(B * 1000.0),
        "edge_band_mm": float(band * B * 1000.0),
        "fill": float(occ.mean()),
        "interior": float(interior),
        "edges": edges,
        "weakest_edge": float(min(edges.values())),
    }


def _surface_quality_reasons(coverage: dict, mesh_stats: dict, scfg) -> list[str]:
    """Reasons a fitted flat surface is not backed by enough measured depth."""
    reasons: list[str] = []
    point_count = int(coverage.get("point_count") or 0)
    fill = float(coverage.get("fill") or 0.0)
    weakest_edge = float(coverage.get("weakest_edge") or 0.0)
    if point_count < int(scfg.min_actual_surface_points):
        reasons.append(
            f"only {point_count} supported measured mesh vertices "
            f"(< {int(scfg.min_actual_surface_points)})")
    if fill < float(scfg.min_actual_fill_coverage):
        reasons.append(
            f"measured surface fill {fill:.0%} "
            f"(< {float(scfg.min_actual_fill_coverage):.0%})")
    if weakest_edge < float(scfg.min_actual_edge_coverage):
        reasons.append(
            f"weakest edge support {weakest_edge:.0%} "
            f"(< {float(scfg.min_actual_edge_coverage):.0%})")
    if bool(mesh_stats.get("support_fallback")) and int(mesh_stats.get("combined_vertices") or 0) == 0:
        reasons.append("no measured vertices had repeated multi-view depth support")
    return reasons


def _combine_depth_frames(frames) -> tuple[np.ndarray, np.ndarray]:
    """Median-fuse same-pose RGBD frames, ignoring zero-depth holes."""
    colors = [np.asarray(f.color, dtype=np.float32) for f in frames]
    color = np.clip(np.mean(np.stack(colors, axis=0), axis=0), 0, 255).astype(np.uint8)
    depths = [np.asarray(f.depth) for f in frames if f.depth is not None]
    if len(depths) == 1:
        return color, np.ascontiguousarray(depths[0])
    stack = np.stack(depths, axis=0)
    masked = np.ma.masked_equal(stack, 0)
    depth = np.ma.median(masked, axis=0).filled(0)
    return color, np.ascontiguousarray(depth.astype(stack.dtype, copy=False))


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
        if applied_mode == "run_robot":
            ensure_real_robot_link(rdk, self.services.config.robodk, log=ctx.log)
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

            if start_joints is not None:
                ctx.log("returning to start pose before fusion")
                rdk.move_j_joints(start_joints)
                start_joints = None

            if scfg.save_views and self.params.save_artifacts:
                _save_views(views, K, width, height, run_dir,
                            depth_scale=scfg.depth_scale, log=ctx.log)

            ctx.progress(len(targets), len(targets), "fusing")
            voxel_m = (self.params.voxel_size_m
                       if self.params.voxel_size_m is not None else scfg.voxel_size_m)
            ctx.log(f"fusing {len(views)} views (TSDF voxel {voxel_m * 1000:.1f} mm)…")
            res = fuse_views(views, K, width, height, voxel_size_m=voxel_m,
                             sdf_trunc_m=scfg.sdf_trunc_m, depth_scale=scfg.depth_scale,
                             depth_min_m=scfg.depth_min_m, depth_max_m=scfg.depth_max_m)

            # Isolate the work surface (the "top layer"): crop to a box around where the
            # camera was aimed so the FLOOR/walls don't dominate the fit (the cause of a
            # room-sized plane). Falls back to the full cloud if the crop is too thin.
            raw_mesh, cloud = res.mesh, res.cloud
            center_mm = look_point_from_views(views)
            if scfg.roi_enabled:
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
                        raw_mesh = crop_box(raw_mesh, cm, **roi)
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
            if self.params.crop_size_mm is not None and center_mm is not None:
                wp = bounded_work_plane(
                    wp, center_mm / 1000.0,
                    (self.params.crop_size_mm[0] / 1000.0,
                     self.params.crop_size_mm[1] / 1000.0))
                ctx.log(
                    f"large surface: bounded work region to "
                    f"{self.params.crop_size_mm[0]:.0f}×"
                    f"{self.params.crop_size_mm[1]:.0f} mm around the camera aim")
            elif self.params.surface_size_mm is not None and center_mm is not None:
                wp = bounded_work_plane(
                    wp, center_mm / 1000.0,
                    (self.params.surface_size_mm[0] / 1000.0,
                     self.params.surface_size_mm[1] / 1000.0))
                ctx.log(
                    f"locked surface: bounded work region to "
                    f"{self.params.surface_size_mm[0]:.0f}×"
                    f"{self.params.surface_size_mm[1]:.0f} mm from the surface lock")

            # Keep the measured TSDF surface as diagnostic evidence, but insert a
            # dense fitted plane for the operator-facing flat-surface workflow. The
            # raw TSDF topology preserves RealSense validity holes from printed
            # ChArUco texture; projecting that topology flat still looks patterned.
            reference_mesh = planar_rectangle_mesh(
                wp.corners, spacing_m=scfg.surface_mesh_spacing_m)
            measured_mesh, mesh_stats = clean_measured_surface_mesh(
                raw_mesh, views, wp, K, width, height,
                plane_band_m=scfg.measured_mesh_plane_band_m,
                rect_margin_m=scfg.measured_mesh_rect_margin_m,
                support_tolerance_m=scfg.measured_mesh_support_tolerance_m,
                min_support_views=scfg.measured_mesh_min_support_views,
                min_support_ratio=scfg.measured_mesh_min_support_ratio,
                min_normal_dot=scfg.measured_mesh_min_normal_dot,
                depth_scale=scfg.depth_scale,
                depth_min_m=scfg.depth_min_m,
                depth_max_m=scfg.depth_max_m,
                keep_largest_component=scfg.measured_mesh_keep_largest_component,
                project_to_plane=scfg.measured_mesh_project_to_plane,
                neutral_color=scfg.measured_mesh_neutral_color)
            if len(measured_mesh.triangles) == 0:
                ctx.log("WARNING: measured mesh cleaning produced no triangles; "
                        "using only the fitted flat surface mesh")
                mesh_stats["fallback_mesh"] = "fitted_flat_surface"
            mesh = reference_mesh
            # metres -> mm for RoboDK (rotation is unitless; translation + corners scale)
            frame_T_mm = wp.frame_T.copy()
            frame_T_mm[:3, 3] *= 1000.0
            corners_mm = wp.corners * 1000.0
            pp_m, cc = mesh_preview_points(mesh, max_points=scfg.preview_max_points)
            preview_mm = (pp_m * 1000.0).astype(np.float32)
            coverage = _surface_coverage(
                np.asarray(measured_mesh.vertices, dtype=float), wp,
                bin_m=scfg.actual_coverage_bin_m,
                edge_band_m=scfg.actual_coverage_edge_band_m)
            quality_reasons = _surface_quality_reasons(coverage, mesh_stats, scfg)
            if coverage["weakest_edge"] < float(scfg.min_actual_edge_coverage):
                ctx.log(
                    f"WARNING: measured mesh edge support is weak "
                    f"(weakest edge {coverage['weakest_edge']:.0%}, "
                    f"interior {coverage['interior']:.0%}); expect visible gaps "
                    f"or re-scan with better edge coverage")
            if quality_reasons and getattr(scfg, "actual_coverage_hard_fail", False):
                raise RuntimeError(
                    "scan rejected: the fitted work surface is not backed by enough "
                    "measured depth (" + "; ".join(quality_reasons) + "). "
                    "Move farther back so the whole surface stays framed in every "
                    "target, then lock and create targets again.")

            report = _result_report(wp, frame_T_mm, corners_mm, n_views=len(views),
                                     n_points=len(pts), mesh=mesh, run_dir=run_dir,
                                     stamp=stamp, voxel_size_m=voxel_m,
                                     mesh_spacing_m=scfg.surface_mesh_spacing_m,
                                     frames_per_pose=scfg.frames_per_pose,
                                     mesh_stats=mesh_stats, coverage=coverage,
                                     mesh_kind="fitted_flat_surface")
            mesh_obj = None
            if self.params.save_artifacts:
                save_mesh(mesh, str(run_dir / "mesh.obj"))
                save_mesh(mesh, str(run_dir / "mesh.ply"))
                save_mesh(measured_mesh, str(run_dir / "measured_tsdf_mesh.obj"))
                save_mesh(measured_mesh, str(run_dir / "measured_tsdf_mesh.ply"))
                save_mesh(reference_mesh, str(run_dir / "work_surface_rect.obj"))
                save_mesh(raw_mesh, str(run_dir / "raw_tsdf_mesh.ply"))
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
                    f"{len(mesh.vertices)} fitted flat mesh verts "
                    f"({len(mesh.triangles)} tris); work surface "
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
        frames_per_pose = max(1, int(scfg.frames_per_pose))
        for i, name in enumerate(targets):
            ctx.check_cancel()
            ctx.progress(i + 1, total, f"capturing {name}")
            rdk.move_j(name)
            time.sleep(scfg.settle_s)
            frames = []
            for _ in range(frames_per_pose):
                frame = cam.grab(with_depth=True, timeout=scfg.grab_timeout_s)
                if frame.depth is not None:
                    frames.append(frame)
            if not frames:
                ctx.log(f"{name}: no depth — skipped")
                skipped.append(name)
                continue
            color, depth = _combine_depth_frames(frames)
            pose = rdk.camera_pose_T()                 # uses the STORED tool offset
            views.append(ScanView(color=color, depth=depth, pose_T=pose))
            ok, jpeg = cv2.imencode(".jpg", color)
            if ok:
                ctx.frame(jpeg.tobytes())
            suffix = f", median of {len(frames)} frame(s)" if frames_per_pose > 1 else ""
            ctx.log(f"{name}: captured ({np.count_nonzero(depth)} depth px{suffix})")
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
        frames_per_pose = max(1, int(scfg.frames_per_pose))
        with cam.burst(timeout=scfg.grab_timeout_s) as bs:
            for i, name in enumerate(targets):
                ctx.check_cancel()
                ctx.progress(i + 1, total, f"capturing {name}")
                rdk.move_j(name)
                time.sleep(scfg.settle_s)
                buffered = 0
                for rep in range(frames_per_pose):
                    thumb = bs.capture()               # Jetson grabs + buffers the frame
                    if thumb is None:
                        continue
                    try:
                        pose = rdk.camera_pose_T()     # uses the STORED tool offset
                    except Exception:
                        pose = None
                    captured.append((name, pose))
                    buffered += 1
                    if rep == 0:
                        ctx.frame(thumb)               # one thumbnail per target
                if buffered == 0:
                    ctx.log(f"{name}: no frame buffered — skipped")
                    skipped.append(name)
                else:
                    suffix = f" x{buffered}" if frames_per_pose > 1 else ""
                    ctx.log(f"{name}: captured (buffered on Jetson{suffix})")
            ctx.progress(total, total, "downloading buffered frames…")
            ctx.log("transferring all buffered frames from the Jetson in one burst…")
            frames = bs.fetch_all()
            bs.clear()                                 # delete the buffer on the Jetson

        if len(frames) != len(captured):
            ctx.log(f"WARNING: Jetson returned {len(frames)} frame(s) but {len(captured)} "
                    f"were buffered — pairing the overlap (some views may be dropped)")
        grouped: dict[str, dict] = {}
        for (name, pose), fr in zip(captured, frames):
            if pose is None:
                continue
            g = grouped.setdefault(name, {"pose": pose, "frames": []})
            if fr is not None and fr.depth is not None:
                g["frames"].append(fr)
        views: list[ScanView] = []
        for name, g in grouped.items():
            if not g["frames"]:
                ctx.log(f"{name}: no depth/pose — skipped")
                skipped.append(name)
                continue
            color, depth = _combine_depth_frames(g["frames"])
            views.append(ScanView(color=color, depth=depth, pose_T=g["pose"]))
        ctx.log(f"burst transfer complete: {len(frames)} frame(s), {len(views)} usable")
        return views, skipped


# -- insert (the explicit "apply") ------------------------------------------
def insert_scan(services, *, job: "ScanCaptureJob | None" = None,
                run_id: str | None = None,
                result: "ScanResult | None" = None) -> dict:
    """Create the work frame + rectangle (+ fused mesh) in the open station.

    Three sources: a direct ``result`` (reference-mode single-frame locate), an
    explicit ``run_id`` loaded from disk (survives restart), or the in-memory last
    job. Records ``runs/scan/active.json``. Raises ``RuntimeError`` if nothing to insert.
    """
    rdk: RdkIO = services.rdk
    if result is not None:
        r = result
        frame_T_mm, corners_mm = r.frame_T_mm, r.corners_mm
        mesh_path = r.mesh_obj_path
        report = r.report
        stamp_id, source = report.get("stamp"), "reference"
    elif run_id is not None:
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
