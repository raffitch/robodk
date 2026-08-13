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
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime
from types import SimpleNamespace

import cv2
import numpy as np

from ...core import runs
from ...core.camera import CameraError
# latest_characterization/CHARACTERIZATION_DIR live in tasni.core.characterize
# (Task 16 review, Finding 2 — moved OUT of tools/characterize_distance.py,
# which is excluded from packaging, so a tasni/ module never depends on the
# tools/ scripts directory; see that module's docstring for the full reasoning).
from ...core.characterize import CHARACTERIZATION_DIR, latest_characterization
from ...core.events import JobEvent
from ...core.jobrunner import JobContext
from ...core.logging import get_logger, new_run_dir
from ...core.rdk_io import RdkIO
# Reuse calibration's shared orchestration helpers (one implementation).
from ..calibration.poses import (frame_aim_offsets, generate_calibration_poses,
                                  projected_corner_coverage, select_diverse,
                                  select_diverse_with_coverage, viewing_angle_span)
from ..calibration.service import (
    BOARD_KEEPOUT_NAME as CALIB_BOARD_KEEPOUT_NAME,
    TARGET_PREFIX as CALIB_TARGET_PREFIX,
    _camera_hold, dry_tour_required, ensure_camera_tool, ensure_real_robot_link)
from .classifier import classify_compact
from .color_boundary import color_work_boundary
from .corner_evidence import extract_corner_evidence
from .depth_gate import ScanGateThresholds, evaluate_depth_gate
from .five_position import FivePositionSurvey
from .plane import (bounded_work_plane, fit_plane, rectangle_in_frame,
                    reticle_plane_square, work_plane_from_points)
from .planner import (ScanPlan, _largest_contiguous_empty_block, _tile_grid_dims,
                     plan_rect_tour, plan_scan)
from .sam_boundary import sam_work_boundary
from .survey import SurveyThresholds, survey_surface
from .survey_contract import (
    MODE_COMPACT, MODE_FIVE_POSITION, MODE_USER_SPECIFIED, PROVENANCE_BY_MODE,
    CaptureRecord, LockedWorkframeSurvey, camera_calibration_id,
    frame_from_rectangle, order_corners_clockwise, refresh_robot_state)
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

# five_position_capture's corner-detection retry (Task 13 review, remedy ii): a
# small, fixed sanity floor -- just enough real depth for survey_surface's RANSAC
# to even attempt a plane fit -- deliberately NOT the same value as
# ScanConfig.survey_corner_min_plane_coverage_frac, the MEANINGFUL corner-adequacy
# gate applied afterward; see the call site for why reusing one value for both
# would make the coverage gate unreachable dead code.
_CORNER_DETECT_SANITY_FRAC = 0.02


@dataclass
class LockedScanSurface:
    frame: object
    reading: object
    survey: object
    gate_payload: dict
    seed_T: np.ndarray
    seed_joints: object
    locked_at: float
    survey_record: "LockedWorkframeSurvey | None" = None
    lock_token: str = ""


def _crop_gate_payload(gate_payload: dict, scfg, K, image_size, look_mm: float,
                       survey=None, user_region_mm: tuple[float, float] | None = None
                       ) -> tuple[dict, "np.ndarray | None"]:
    """Force the locked surface into reticle-crop mode.

    This is the operator escape hatch for cases where a finite rectangle/dot overlay
    jitters or is partially missing even though the center depth plane is usable.
    ``user_region_mm``, when given, declares the crop rectangle's size explicitly
    (the operator-specified work region) instead of the generic ``scfg.work_crop_mm``
    square.

    Pure: neither ``gate_payload`` nor ``survey`` is mutated. Returns
    ``(payload, corners_cam_mm)`` — a new payload dict, and the crop rectangle's
    corners (camera frame, mm) so the caller can fold them into its own survey
    object (e.g. via ``dataclasses.replace``) instead of this function reaching into
    someone else's object. ``corners_cam_mm`` is ``None`` when the crop rectangle
    could not be computed (survey not detected / not enough info) — the payload is
    still valid in that case, just without an updated overlay.
    """
    out = dict(gate_payload)
    gates = dict(out.get("gates") or {})
    gates.pop("center", None)
    gates.pop("edge", None)
    gates["framed"] = False
    out["gates"] = gates
    out["ok"] = all(bool(gates.get(k)) for k in ("detected", "distance", "angle"))
    out["surface_mode"] = "crop"
    out["fully_framed"] = False
    out["crop_size_mm"] = _large_surface_crop_mm(scfg, K, image_size, look_mm, user_region_mm)
    out["rectangle_size_mm"] = list(out["crop_size_mm"])
    corners_cam_mm = None
    if survey is not None and getattr(survey, "detected", False):
        try:
            corners, _u, _v, _reticle = reticle_plane_square(
                np.asarray(survey.normal_cam, float),
                np.asarray(survey.centroid_cam_mm, float),
                tuple(out["crop_size_mm"]))
            corners_cam_mm = np.asarray(corners, float)
            W, H = image_size
            fx, fy = float(K[0, 0]), float(K[1, 1])
            cx, cy = float(K[0, 2]), float(K[1, 2])
            outline = []
            for p in corners_cam_mm:
                if float(p[2]) <= 0:
                    continue
                outline.append([
                    (float(p[0]) * fx / float(p[2]) + cx) / float(W),
                    (float(p[1]) * fy / float(p[2]) + cy) / float(H),
                ])
            if len(outline) >= 3:
                out["outline_uv"] = outline
                out["grid_uv"] = None
        except Exception:
            corners_cam_mm = None
    return out, corners_cam_mm


def _large_surface_crop_mm(scfg, K, image_size, look_mm: float,
                           user_region_mm: tuple[float, float] | None = None) -> list[float]:
    """The work-square size used when the surface overruns the view (or is forced).

    Fixed (``scfg.work_crop_mm``, default 1.0×1.0 m) rather than a fraction of the FOV:
    the operator aims the reticle at the work area and we project a standard square on
    the plane around it. ``user_region_mm``, when given, overrides this with an
    explicit operator-declared (length, width) in mm. ``K``/``image_size``/``look_mm``
    are unused now (kept so the call sites — which have them handy — need not change)."""
    if user_region_mm is not None:
        return [float(user_region_mm[0]), float(user_region_mm[1])]
    return [float(scfg.work_crop_mm[0]), float(scfg.work_crop_mm[1])]


def _plane_rms_mm(depth, K, *, stride: int = 8) -> float:
    """Quick plane-fit RMS (mm) of the fused lock depth, for the quality report."""
    d = np.asarray(depth, dtype=float)[::stride, ::stride]
    v, u = np.nonzero(d > 0)
    if len(v) < 50:
        return float("nan")
    z = d[v, u]
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    pts = np.stack([(u * stride - cx) / fx * z, (v * stride - cy) / fy * z, z], axis=1)
    try:
        normal, centroid, _ = fit_plane(pts, distance=6.0)
    except Exception:
        return float("nan")
    res = (pts - centroid) @ np.asarray(normal, dtype=float)
    return float(np.sqrt(np.mean(res ** 2)))


def _survey_record_from_lock(survey, seed_T, snapshot, camera_cfg, *, mode, n_frames,
                             measurement_ts, valid_frac, plane_rms_mm) -> "LockedWorkframeSurvey | None":
    """Build the immutable §11 contract from the authoritative lock acquisition."""
    if survey.corners_cam_mm is None:
        return None
    T = np.asarray(seed_T, dtype=float)
    R, t = T[:3, :3], T[:3, 3]
    corners_base = np.asarray(survey.corners_cam_mm, dtype=float) @ R.T + t
    normal_base = R @ np.asarray(survey.normal_cam, dtype=float)
    if normal_base[2] < 0:
        normal_base = -normal_base
    centroid_base = R @ np.asarray(survey.centroid_cam_mm, dtype=float) + t
    corners_base = order_corners_clockwise(corners_base, normal_base)
    frame_T = frame_from_rectangle(corners_base, normal_base)
    e1 = float(np.linalg.norm(corners_base[1] - corners_base[0]))
    e2 = float(np.linalg.norm(corners_base[2] - corners_base[1]))
    record = CaptureRecord(
        kind="compact" if mode == MODE_COMPACT else "center", robot=snapshot,
        measurement_ts=float(measurement_ts), captured_at=snapshot.fetched_at,
        n_frames=int(n_frames), standoff_mm=float(survey.standoff_mm),
        tilt_deg=float(survey.tilt_deg), valid_frac=float(valid_frac),
        plane_rms_mm=float(plane_rms_mm),
        plane_normal_base=tuple(normal_base), plane_point_base=tuple(centroid_base))
    quality = {
        "plane_rms_mm": float(plane_rms_mm), "standoff_mm": float(survey.standoff_mm),
        "tilt_deg": float(survey.tilt_deg), "valid_frac": float(valid_frac),
        "measure_frames": int(n_frames),
    }
    return LockedWorkframeSurvey(
        mode=mode, boundary_provenance=PROVENANCE_BY_MODE[mode], captures=(record,),
        plane_normal_base=tuple(normal_base), plane_point_base=tuple(centroid_base),
        corners_base=tuple(map(tuple, corners_base)),
        center_base=tuple(corners_base.mean(axis=0)),
        frame_T_base=tuple(map(tuple, frame_T)), size_mm=(max(e1, e2), min(e1, e2)),
        quality=quality, calibration_id=camera_calibration_id(camera_cfg),
        locked_robot=snapshot, locked_at=snapshot.fetched_at)


def _outline_edge_angle_deg(outline_uv) -> float | None:
    uv = np.asarray(outline_uv, dtype=float).reshape(-1, 2)
    if len(uv) < 2:
        return None
    edges = np.roll(uv, -1, axis=0) - uv
    edge = edges[int(np.argmax(np.linalg.norm(edges, axis=1)))]
    angle = float(np.degrees(np.arctan2(edge[1], edge[0])))
    return ((angle + 45.0) % 90.0) - 45.0


def _project_color_corners_uv(corners_color, camera_cfg):
    """Project color-camera 3D corners (mm) to normalized uv through the workstation
    RealSense calibration (K + distortion). Returns an ``(N, 2)`` array in 0-1 image
    coords, or ``None`` if any projected point is non-finite."""
    if corners_color is None:
        return None
    corners = np.asarray(corners_color, dtype=np.float64).reshape(-1, 3)
    projected, _ = cv2.projectPoints(
        corners, np.zeros(3), np.zeros(3), camera_cfg.K, camera_cfg.dist)
    W, H = camera_cfg.size
    uv = projected.reshape(-1, 2) / np.array([W, H], dtype=float)
    return uv if np.all(np.isfinite(uv)) else None


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


def _authoritative_acquisition(services, *, owner: str):
    """Shared step-and-measure acquisition core (spec §9) for every
    authoritative scan capture -- the compact/crop lock (:func:`lock_scan_surface`)
    AND the guided five-position survey (:func:`five_position_capture`) both
    call this, so the two capture paths cannot silently drift apart.

    Mounts the camera tool, stops any live preview (so the capture never
    contends the camera lease with the video loop), grabs + fuses
    ``scfg.surface_measure_frames`` depth+colour frames, then computes BOTH
    readiness views of the fused frame -- the coarse centre-patch gate
    (``reading``) and the full-frame plane survey (``measurement``) -- plus
    an explicit REAL robot-state refresh (``snapshot``, a genuine double-read
    of the robot's joints; see :func:`survey_contract.refresh_robot_state`).

    Returns ``(frame, n_frames, reading, measurement, snapshot, raw_frames)``
    where ``frame`` is a ``SimpleNamespace(color, depth, timestamp)`` --
    bundling colour/depth/timestamp together (rather than as separate tuple
    members) is deliberate: every downstream consumer of a "capture"
    (``CaptureRecord.measurement_ts``, ``_survey_record_from_lock``) needs the
    frame's timestamp, and the acquisition site (this function) is the only
    place that legitimately reads ``frames[-1].timestamp`` off the raw camera
    frames.

    ``raw_frames`` (Task 18) is the pre-fusion list of individually-grabbed
    camera frames (each with its own ``.depth``) that ``frame`` was median-
    fused from -- returning the list itself costs nothing extra (they are
    already held here for ``_combine_depth_frames``); it exists so
    ``lock_scan_surface`` can independently re-survey each raw frame for
    ``classify_compact``'s multi-frame rectangle-identity gate (spec §6) --
    fusion collapses frame-to-frame disagreement before a single
    ``survey_surface`` call ever sees it, so that gate needs the un-fused
    frames. Callers that do not need per-frame evidence (``five_position_
    capture``) simply ignore this element; the RANSAC re-survey itself is
    deferred to (and only ever paid by) the caller that asks for it, not run
    here.
    """
    cfg = services.config
    scfg = cfg.scan
    rdk: RdkIO = services.rdk
    K = cfg.camera.K
    ensure_camera_tool(services, log=_log_pub(services))
    if services.live.running:
        services.live.stop()
    rdk.use_camera_tool(cfg.robodk.camera_tool)

    measure_frames = max(1, int(getattr(scfg, "surface_measure_frames", 1)))
    frames = []
    with _camera_hold(services, owner):
        for _ in range(measure_frames):
            fr = services.camera.grab(with_depth=True, timeout=scfg.grab_timeout_s)
            if fr.depth is not None:
                frames.append(fr)
    if not frames:
        raise RuntimeError("surface measurement failed - no depth frames received")
    color, depth = _combine_depth_frames(frames)
    frame = SimpleNamespace(
        color=color, depth=depth,
        timestamp=getattr(frames[-1], "timestamp", time.time()))
    reading = evaluate_depth_gate(
        frame.depth, K, scan_gate_thresholds(scfg), depth_scale=scfg.depth_scale)
    measurement = survey_surface(
        frame.depth, K, _survey_thresholds(scfg), depth_scale=scfg.depth_scale)
    snapshot = refresh_robot_state(rdk)
    return frame, len(frames), reading, measurement, snapshot, frames


def lock_scan_surface(services, *, force_crop: bool = False,
                      user_region_mm: tuple[float, float] | None = None) -> LockedScanSurface:
    """Freeze one authoritative RGBD measurement and the matching robot pose."""
    cfg = services.config
    scfg = cfg.scan
    rdk: RdkIO = services.rdk
    K = cfg.camera.K
    frame, n_frames, reading, survey, snapshot, raw_frames = _authoritative_acquisition(
        services, owner="scan-surface-lock")
    depth = np.asarray(frame.depth) if frame.depth is not None else np.zeros((0, 0))
    full_frame_valid_frac = float(np.mean(depth > 0)) if depth.size else 0.0
    surface_overruns_view = bool(
        survey.detected and not survey.fully_framed and full_frame_valid_frac >= 0.95)
    crop_mode = force_crop or surface_overruns_view
    gate_payload = scan_gate_payload(reading, survey)
    ideal_distance = _planned_surface_standoff_mm(
        scfg, K, cfg.camera.size, reading, survey, full_frame_valid_frac)
    gate_payload["ideal_distance_mm"] = ideal_distance
    gate_payload["distance_tol_mm"] = float(scfg.distance_tol_mm)
    gate_payload["surface_mode"] = "crop" if crop_mode else "full"
    final_gates = {
        "detected": bool(reading.gates.get("detected")),
        "distance": (
            reading.distance_mm is not None
            and abs(float(reading.distance_mm) - ideal_distance) <= float(scfg.distance_tol_mm)
        ),
        "angle": bool(reading.gates.get("angle")),
    }
    if not force_crop and survey.detected and survey.fully_framed:
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
        if survey.outline_uv and len(survey.outline_uv) >= 2:
            uv = np.asarray(survey.outline_uv, float)
            edges = np.roll(uv, -1, axis=0) - uv
            edge = edges[int(np.argmax(np.linalg.norm(edges, axis=1)))]
            angle = float(np.degrees(np.arctan2(edge[1], edge[0])))
            angle = ((angle + 45.0) % 90.0) - 45.0
            gate_payload["yaw_a_deg"] = -angle
            gate_payload["edge_align_tol_deg"] = float(scfg.edge_align_tol_deg)
            # EDGE A: advisory alignment lamp (never part of ``ok`` — see
            # live_scan_telemetry_payload). Meaningful only for an elongated platform;
            # a near-square platform's edge yaw is ambiguous, so the lamp reads OK.
            edge_aspect = _aspect_ratio(survey.extent_mm)
            edge_meaningful = bool(edge_aspect is not None
                                   and edge_aspect >= float(scfg.edge_gate_min_aspect))
            final_gates["edge"] = (abs(angle) <= float(scfg.edge_align_tol_deg)
                                   if edge_meaningful else True)
    elif survey.detected and not crop_mode:
        final_gates["framed"] = False
    gate_payload["gates"] = {**gate_payload.get("gates", {}), **final_gates}
    ok_gates = dict(final_gates)
    ok_gates.pop("edge", None)          # advisory — never blocks readiness
    if survey.detected and survey.fully_framed:
        # Once a finite rectangle is fully framed, its measured centroid is the
        # target center used by the planner. Reticle-centering is useful guidance,
        # but it should not make the operator chase per-frame X/Y jitter.
        ok_gates.pop("center", None)
    gate_payload["ok"] = all(ok_gates.values())
    if crop_mode and reading.distance_mm is not None:
        gate_payload, crop_corners_cam_mm = _crop_gate_payload(
            gate_payload, scfg, K, cfg.camera.size, float(reading.distance_mm),
            survey=survey, user_region_mm=user_region_mm)
        if crop_corners_cam_mm is not None:
            # Rebind to a corrected local copy rather than mutating the shared
            # ``survey`` object in place (_crop_gate_payload is pure). This same
            # local is used for the rest of the lock (record building below, and
            # the LockedScanSurface.survey the caller/generate_scan_targets reads).
            survey = replace(survey, corners_cam_mm=crop_corners_cam_mm)

    # Robot-state snapshot (§9): already an authoritative double-read of joints +
    # the derived camera pose, tagged stationary/moving (see
    # _authoritative_acquisition) -- fetched before the survey record / gate event
    # are built, so even on a failing gate the operator's HUD reflects the robot
    # state the failed reading was taken at.
    seed_T = snapshot.camera_T_np()
    seed_joints = list(snapshot.joints)

    # Boundary provenance (spec §1 / §12): a survey record is built only when the
    # boundary was either MEASURED (the normal compact path, crop_mode False, AND
    # -- Task 18 -- classify_compact's spec-§6 entry conditions all hold) or
    # EXPLICITLY DECLARED by the operator (force_crop — today's "region" mode).
    # crop_mode alone is NOT that discriminator: it also goes true automatically
    # when the surface overruns the camera view (surface_overruns_view), and that
    # is the system silently falling back, not the operator specifying anything —
    # tagging it "user specified" would be a false provenance claim. In that
    # auto-overrun-without-force_crop case, leave survey_record/boundary_provenance
    # unset and surface a warning instead; everything else about the lock (gate
    # readiness, the crop overlay/outline_uv, etc.) is unaffected.
    record = None
    if force_crop:
        mode = MODE_USER_SPECIFIED
        plane_rms = _plane_rms_mm(depth, K)
        record = _survey_record_from_lock(
            survey, seed_T, snapshot, services.config.camera,
            mode=mode, n_frames=n_frames, measurement_ts=frame.timestamp,
            valid_frac=gate_payload.get("valid_frac", 0.0), plane_rms_mm=plane_rms)
    elif not crop_mode:
        # Compact path (spec §6, Task 18): "detected + fully framed" alone is not
        # sufficient to call the boundary MEASURED -- classify_compact (built +
        # unit-tested in an earlier task, never wired in until now) is the single
        # source of truth for the guard band / segmentation-confirmed boundary /
        # centering / tilt / rectangle-identity gates. A failure follows the EXACT
        # same honest-provenance shape as the auto-overrun branch below: no
        # record, no boundary_provenance, a warning naming which condition(s)
        # failed -- never a raise, so the lock still completes and issues a
        # lock_token either way.
        if not survey.detected:
            # Task 18 review, Minor: ``detected`` alone already decides
            # ineligibility (classify_compact adds "no surface detected" and
            # every other gate reads not-ok against None/empty inputs too), so
            # skip the two expensive calls below (~1.75 s of RANSAC across
            # surface_measure_frames raw frames, plus a real segmentation pass)
            # and reach the identical conclusion for free.
            eligibility = classify_compact(survey, None, None, scfg, outline_history=[])
            gate_payload["compact_eligibility"] = eligibility.to_dict()
            gate_payload.setdefault("warnings", []).extend(eligibility.reasons)
        elif not survey.fully_framed:
            # Task 18 review, Important 5: once fully_framed is False,
            # survey_surface has already REPLACED corners_cam_mm/outline_uv with
            # a generic reticle square centred on the reticle (see its own
            # comment) -- not a measured boundary. classify_compact must never
            # judge that fabricated square: it would trivially satisfy gates a
            # genuinely unmeasured edge says nothing about (e.g. "centered"),
            # misreporting WHY the lock isn't ready. Chose this over passing
            # "the measured corners instead" because survey_surface's returned
            # SurveyMeasurement does not retain the pre-substitution corners at
            # all (recomputing them here would duplicate survey_surface's own
            # RANSAC + oriented-rectangle logic at the call site) -- same honest
            # shape as the auto-overrun branch below: warn, no record, no
            # compact_eligibility key (nothing was actually classified).
            gate_payload.setdefault("warnings", []).append(
                "the surface is not fully framed, so its boundary is unmeasured — "
                "reframe it fully in view, declare a region, or run a "
                "multi-position survey before relying on this lock's geometry.")
        else:
            classify_scfg, identity_note = _compact_identity_scfg(scfg)
            if identity_note:
                gate_payload.setdefault("warnings", []).append(identity_note)
            outline_history = _survey_outline_history(raw_frames, K, scfg)
            boundary = _work_boundary(scfg, frame.color)
            eligibility = classify_compact(
                survey, survey.outline_uv, boundary, classify_scfg,
                outline_history=outline_history)
            gate_payload["compact_eligibility"] = eligibility.to_dict()
            if eligibility.eligible:
                mode = MODE_COMPACT
                plane_rms = _plane_rms_mm(depth, K)
                record = _survey_record_from_lock(
                    survey, seed_T, snapshot, services.config.camera,
                    mode=mode, n_frames=n_frames, measurement_ts=frame.timestamp,
                    valid_frac=gate_payload.get("valid_frac", 0.0), plane_rms_mm=plane_rms)
            else:
                # Task 18 review, Critical 2c: the guard-band reason alone gave
                # no direction -- enrich just that one warning line with an
                # actionable backoff hint (classify_compact's own `reasons`
                # tuple, already unit-tested verbatim in test_scan_classifier.py,
                # is left untouched; the enrichment happens only in what this
                # lock publishes).
                warnings = gate_payload.setdefault("warnings", [])
                for reason in eligibility.reasons:
                    if not eligibility.guard_ok and "guard region" in reason:
                        reason = reason + " — " + _guard_violation_backoff_hint(
                            survey.outline_uv, scfg.compact_guard_uv, reading.distance_mm)
                    warnings.append(reason)
    else:
        gate_payload.setdefault("warnings", []).append(
            "the surface overruns the camera view, so its boundary is unverified — "
            "declare a region (crop mode) or run a multi-position survey before "
            "relying on this lock's geometry.")

    # Calibration-age gate (Task 16, spec §10): a distance-characterization sweep
    # (tools/characterize_distance.py) is what actually validates the measurement
    # chain's error budget at the working standoff -- without one on file (or once
    # it goes stale) this lock's plane/rectangle has no verified accuracy behind
    # it. Checked here regardless of which branch above ran: staleness is a
    # camera/measurement-chain concern, orthogonal to the boundary-provenance
    # warning (a surface can be perfectly framed with a stale characterization, or
    # vice versa).
    #
    # Task 16 review Finding 1: the warning text is appended to gate_payload
    # UNCONDITIONALLY when stale -- including the hard-fail branch -- and
    # hard-fail additionally forces gate_payload["ok"] = False. Without this,
    # the gate/frame telemetry published just below would be byte-for-byte
    # identical to a fully healthy lock right up until the RuntimeError below
    # fires, so a client driven by the event stream could show "surface ready"
    # for a moment before the call errors. The raise itself is still deferred
    # past the telemetry publish (mirroring the "gate not ok" raise at the end
    # of this function) so the operator's HUD sees the now-correctly-flagged
    # gate event before the lock is refused, rather than failing silently.
    characterization = latest_characterization(CHARACTERIZATION_DIR)
    stale = characterization is None
    if not stale:
        try:
            measured_at = datetime.fromisoformat(str(characterization.get("date")))
            age_days = (datetime.now() - measured_at).total_seconds() / 86400.0
            stale = age_days > float(scfg.calibration_max_age_days)
        except (TypeError, ValueError):
            stale = True    # an unparsable date is untrustworthy -- treat as stale
    calibration_hard_fail = False
    if stale:
        gate_payload.setdefault("warnings", []).append(
            "calibration verification missing or expired")
        if scfg.calibration_expiry_hard_fail:
            calibration_hard_fail = True
            gate_payload["ok"] = False
    elif record is not None and characterization.get("dstar_mm") is not None:
        record.quality["dstar_mm"] = float(characterization["dstar_mm"])

    # Snapshot the survey record into gate_payload LAST (bundled minor, Task 16
    # review): record.quality may just have gained "dstar_mm" above, and
    # to_dict() is a deep copy -- taking this snapshot any earlier would
    # publish a survey block that never carries dstar_mm even on a healthy,
    # freshly-characterized lock.
    if record is not None:
        gate_payload["survey"] = record.to_dict()
        gate_payload["boundary_provenance"] = record.boundary_provenance

    ok, jpeg = cv2.imencode(".jpg", frame.color)
    if ok:
        services.bus.publish(JobEvent("frame", {
            "jpeg_b64": base64.b64encode(jpeg.tobytes()).decode("ascii")}))
    services.bus.publish(JobEvent("gate", {
        **gate_payload, "live": False, "measure_frames": n_frames}))

    if calibration_hard_fail:
        raise RuntimeError("calibration verification missing or expired")

    if not gate_payload["ok"]:
        bad = [name for name, good in final_gates.items() if not good]
        raise RuntimeError(
            "surface is not ready — fix " + ", ".join(bad)
            + f" (distance {reading.distance_mm and round(reading.distance_mm)} mm, "
            + f"target {round(ideal_distance)} mm, "
            + f"tilt {reading.tilt_deg and round(reading.tilt_deg, 1)}°).")
    return LockedScanSurface(
        frame=frame, reading=reading, survey=survey, gate_payload=gate_payload,
        seed_T=np.asarray(seed_T, float), seed_joints=seed_joints,
        locked_at=time.monotonic(), survey_record=record, lock_token=uuid.uuid4().hex)


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
        frame_margin_uv=float(getattr(scfg, "live_frame_margin_uv", 0.02)),
        work_crop_mm=tuple(scfg.work_crop_mm),
    )


def _compact_identity_scfg(scfg) -> "tuple[object, str | None]":
    """Adapt classify_compact's rectangle-identity frame requirement (spec §6) to
    what a single lock's acquisition can actually supply (Task 18).

    ``_authoritative_acquisition`` grabs ``scfg.surface_measure_frames`` raw
    frames per lock; ``classify_compact`` hard-codes ``scfg.compact_identity_
    frames`` consecutive frames as its evidence requirement and has no
    parameter to override that (see its own module docstring -- it is reviewed,
    not to be modified). If an operator configures fewer measure-frames than
    the identity requirement, the gate is UNSATISFIABLE by construction: every
    compact lock would warn/reject regardless of how consistent the surface
    genuinely is. Since ``surface_measure_frames`` caps how much identity
    evidence can ever exist, adapt the requirement down to that cap.

    Never adapt below 2, though: with 0 or 1 samples there is nothing to
    compare against, and ``rectangle_identity_consistent`` would pass
    VACUOUSLY (a single outline trivially "agrees" with itself) -- exactly the
    loophole the identity gate exists to close. Below that floor, the nominal
    (now unmeetable) requirement is left in place instead, so the gate fails
    honestly rather than rubber-stamping zero real evidence.

    Returns ``(scfg_for_classify_compact, warning_or_None)``.
    """
    nominal = int(scfg.compact_identity_frames)
    available = int(scfg.surface_measure_frames)
    if available >= nominal:
        return scfg, None
    if available < 2:
        return scfg, (
            f"surface_measure_frames ({available}) is below 2 -- too few frames to "
            "ever establish rectangle-identity consistency (compact_identity_frames "
            f"requires {nominal}); the identity gate cannot pass until "
            "surface_measure_frames is raised.")
    return scfg.model_copy(update={"compact_identity_frames": available}), (
        f"surface_measure_frames ({available}) is below compact_identity_frames "
        f"({nominal}); the identity check used {available} frame(s) instead.")


#: Task 18 review round 2: the origin-distance start-corner rule below is
#: unambiguous with a large margin EXCEPT near an in-plane rotation of
#: 45/135/225/315 degrees for a rectangle centred in view, where two corners
#: become nearly equidistant from the origin by construction (an exact
#: algebraic tie for an exactly-centred, exactly-45-degree square) -- see
#: ``_canonicalize_outline_uv``'s own docstring for the measured before/after
#: numbers. Below this gap (normalized uv), the two smallest origin-distances
#: are treated as "too close to trust" and the tie-break described there
#: takes over instead. Comfortably above the ~0.001-0.03 uv corner-position
#: noise measured at 0.5-2 mm depth noise (this and the original Critical 1
#: regression test), and comfortably inside the window where origin-distance
#: is genuinely near-degenerate: a numeric sweep (0-45 degrees in 1-degree
#: steps, three aspect ratios -- 1:1, 1.36:1, 1.67:1) shows the gap crosses
#: below 0.05 only within roughly 3-6 degrees of a 45-degree multiple
#: (e.g. square: 0.073 at 40 degrees, 0.044 at 42, 0.000 at 45), and is back
#: above 0.15 by 10 degrees away (e.g. square: 0.145 at 35 degrees) -- so this
#: fallback engages only in the narrow band where it is actually needed.
_CANONICALIZE_TIE_EPS_UV = 0.05


def _canonicalize_outline_uv(outline_uv) -> list:
    """Canonicalize a 4-corner outline's corner order for CROSS-FRAME comparison
    (Task 18 review, Critical 1; tie-break hardened in review round 2).

    ``_oriented_rectangle`` (``plane.py``) fits each frame's rectangle
    independently via its own min-area-rectangle search, so the SAME physical
    rectangle can legitimately come back with a different STARTING corner
    and/or winding direction from one frame to the next once real depth noise
    perturbs which axis the fit calls "first" -- measured directly: at 1.0 mm
    of synthetic depth noise (within the D435i's real ~0.5-2 mm RMS band at
    400-500 mm), a frame's corners came back as an exact cyclic rotation of a
    noise-free reference frame's, reading as 0.8455 normalized-uv "drift" by
    naive corner-for-corner (by-index) comparison against a 0.04 tolerance --
    while the best relabelling of the SAME two outlines drifts only 0.0012.
    ``rectangle_identity_consistent`` compares corner-for-corner BY INDEX, so
    without this, a rectangle that has not moved at all can spuriously fail
    the identity gate on ordinary sensor noise (see ``test_scan_job.py``'s
    dedicated noisy-frame regression test for the full before/after numbers).

    Normalizes winding via the shoelace signed area (reversing to make it
    consistently non-negative). The start corner is then the one nearest the
    image origin (0, 0) -- EXCEPT when the two smallest origin-distances are
    within ``_CANONICALIZE_TIE_EPS_UV`` of each other, which happens near an
    in-plane rotation of 45 degrees for a rectangle centred in view (two
    corners become nearly equidistant from the origin by construction; review
    round 2 measured an EXACT tie for a perfectly centred, perfectly
    45-degree square, and a near-tie -- 0.0026-0.0137 normalized-uv gap, well
    inside real sensor-noise range -- empirically on this project's own
    rendering + fitting pipeline at 43.6-43.7 degrees, the pipeline's actual
    crossing point once its own small fitting asymmetries are accounted for).
    In that regime the start corner is instead the LEXICOGRAPHICALLY smallest
    (u, then v) -- chosen because it is providably non-degenerate exactly
    where origin-distance is degenerate: for any rectangle with half-extents
    a, b > 0 at rotation theta, the four corners' u-coordinates (relative to
    centroid) are +-(a*cos(theta)+b*sin(theta)) and +-(a*cos(theta)-b*sin(theta));
    the minimum of the four is UNIQUE for every theta except theta = 0/90/180/
    270 degrees (where a*cos(theta) or b*sin(theta) is exactly 0) -- i.e.
    lexicographic order's own degenerate angles are a full 45 degrees away
    from origin-distance's, so the two criteria's fragile zones never
    coincide. (Verified both algebraically and with a numeric sweep; at
    theta=45 degrees exactly the lexicographic gap is close to the
    rectangle's own scale, not a near-tie.) At theta = 0/90/180/270 degrees
    themselves origin-distance is fully unambiguous (its own gap is at its
    MAXIMUM there), so this fallback is never actually exercised at those
    angles -- the two rules' blind spots are complementary, not shared.

    Two representations of the same physical rectangle that differ only in
    starting corner and/or winding map to the IDENTICAL canonical form.
    """
    uv = np.asarray(outline_uv, dtype=float).reshape(-1, 2)
    if len(uv) < 3:
        return uv.tolist()
    signed_area = 0.5 * np.sum(
        uv[:, 0] * np.roll(uv[:, 1], -1) - np.roll(uv[:, 0], -1) * uv[:, 1])
    if signed_area < 0:
        uv = uv[::-1]
    dist = np.linalg.norm(uv, axis=1)
    order = np.argsort(dist)
    if len(dist) >= 2 and (dist[order[1]] - dist[order[0]]) < _CANONICALIZE_TIE_EPS_UV:
        start = int(min(range(len(uv)), key=lambda i: (uv[i, 0], uv[i, 1])))
    else:
        start = int(order[0])
    return np.roll(uv, -start, axis=0).tolist()


def _survey_outline_history(raw_frames, K, scfg) -> list:
    """Independently survey each RAW (pre-fusion) depth frame from this lock's
    acquisition and collect its CANONICALIZED rectangle outline, for classify_
    compact's §6 rectangle-identity gate (Task 18).

    ``_authoritative_acquisition`` median-fuses ``scfg.surface_measure_frames``
    raw depth frames into ONE depth image before the single ``survey_surface``
    call that produces the lock's own ``survey`` -- so that fused result alone
    carries no frame-to-frame evidence at all. This reruns ``survey_surface``
    on each raw frame independently -- real per-frame RANSAC plane fits, NOT
    the fused outline repeated -- so a genuinely unstable rectangle can be told
    apart from a stable one. Each outline is passed through
    ``_canonicalize_outline_uv`` before being appended (Critical 1) so ordinary
    per-frame corner-order/winding drift from the independent fits does not
    read as rectangle movement.

    MEASURED cost (1280x720, this machine): ~0.35 s per ``survey_surface`` pass,
    so ~1.75 s total for the default 5 raw frames -- plus a further ~450 ms for
    ``_work_boundary``'s SAM pass when the configured engine tries it. Paid
    ONLY here, i.e. only on the compact (non-crop, fully-framed) lock path --
    never on a crop/force_crop lock, and short-circuited entirely when the
    survey did not detect a surface at all (see ``lock_scan_surface``).

    A raw frame that fails to detect a plane on its own contributes nothing to
    the history (skipped, not padded with a placeholder) -- that omission is
    itself real evidence of instability, and padding it would manufacture
    agreement that was never measured.
    """
    th = _survey_thresholds(scfg)
    history = []
    for fr in raw_frames:
        if fr.depth is None:
            continue
        m = survey_surface(fr.depth, K, th, depth_scale=scfg.depth_scale)
        if m.outline_uv:
            history.append(_canonicalize_outline_uv(m.outline_uv))
    return history


def _guard_violation_backoff_hint(raw_corners_uv, guard: float, standoff_mm) -> str:
    """Turn a failed ``compact_guard_uv`` check into an ACTIONABLE hint (Task 18
    review, Critical 2c) instead of a bare "leaves the guard region" with no
    direction: roughly how much farther to back the camera off.

    Backing off (increasing standoff) shrinks a FIXED real-world rectangle's
    projected uv-distance-from-centre by the same factor, to first order
    (pinhole projection: apparent size is inversely proportional to standoff).
    The corner presently deepest into the guard band sets the required scale:
    if it sits at normalized distance ``d`` from image centre but the guard
    band only allows up to ``0.5 - guard``, standoff must grow by roughly
    ``d / (0.5 - guard)`` to bring it back inside.
    """
    generic = "back off (increase standoff) to bring the boundary further inside the frame"
    if raw_corners_uv is None or standoff_mm is None:
        return generic
    uv = np.asarray(raw_corners_uv, dtype=float).reshape(-1, 2)
    d = float(np.max(np.abs(uv - 0.5)))
    allowed = 0.5 - float(guard)
    if d <= 0.0 or allowed <= 0.0 or d <= allowed:
        return generic
    scale = d / allowed
    extra_mm = float(standoff_mm) * (scale - 1.0)
    return (f"back off (increase standoff) by roughly {extra_mm:.0f} mm "
            f"(~{(scale - 1.0) * 100:.0f}% farther) so the boundary clears the guard margin")


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


def _deproject_plane_points_mm(depth, K, T_base_cam, *, plane_normal_cam,
                               plane_point_cam, band_mm: float,
                               stride: int = 6) -> "tuple[np.ndarray, float, float]":
    """Deproject a strided grid of depth pixels that lie within ``band_mm`` of the
    ALREADY-FITTED local plane to BASE-frame millimetres.

    Returns ``(points_base_mm, purity_frac, coverage_frac)``:

    - ``purity_frac`` = (points kept by the band) / (points with ANY valid depth in
      the stride grid). "Of the depth we actually got, how much of it is trustworthy
      (on the work plane)?" Robust to whether the missing/background portion of the
      frame returns real-but-off-plane depth (a nearby floor/fixture) OR no depth at
      all (open space beyond the D435i's range): a genuinely good capture reads near
      1.0 in BOTH cases (a real floor drags it down; open space simply isn't counted
      in either the numerator or denominator, so it does not drag the ratio down the
      way a naive whole-frame fraction would). See ``five_position_capture`` for why
      this -- not the compact lock's centre-patch fraction -- is what a CORNER step's
      ``CaptureRecord.valid_frac`` needs to be (Task 13 review Finding, remedy ii).
    - ``coverage_frac`` = (points kept by the band) / (all stride-grid positions,
      valid or not). "How much of the FULL FRAME is actually table?" This is what
      catches "genuinely too little table in view" (e.g. a corner capture whose
      table sliver is tiny) -- a case ``purity_frac`` alone cannot: a tiny-but-clean
      sliver against an empty (0-depth) background also reads ``purity_frac`` near
      1.0, since there is nothing else to dilute it.

    Explicit millimetres throughout -- deliberately the SAME pattern as
    ``corner_evidence._deproject_base`` (depth is the raw uint16 RealSense frame,
    read directly as millimetres; no ``depth_scale`` division). This module does
    NOT reuse ``_backproject_depth`` above for this: that helper returns
    CAMERA-frame points scaled by ``depth_scale``, an extra parameter/convention a
    caller has to know to apply correctly, whereas the five-position survey (the
    one caller of this function) needs BASE-frame mm points and nothing else --
    matching ``corner_evidence``'s own unambiguous, self-contained convention
    keeps the whole five-position pipeline's units consistent end to end.

    ``stride`` subsamples the frame (every ``stride``-th row/col): the per-capture
    local plane this feeds (``rect_fit.fit_global_plane``, via
    ``FivePositionSurvey.add_capture``) only needs a representative sample of this
    one position's surface, not every pixel.

    ``plane_normal_cam``/``plane_point_cam`` (CAMERA frame -- exactly what
    ``survey_surface`` already fit via RANSAC for this same depth frame,
    ``measurement.normal_cam``/``measurement.centroid_cam_mm``) and ``band_mm`` are
    REQUIRED, not optional: Task 13 review Finding 1 found that deprojecting every
    valid pixel unfiltered (this function's original behaviour) is NOT "plane
    inliers" the brief's own spec asked for. A five-position CORNER capture aims the
    camera at a table corner (``corner_hint_uv=(0.5, 0.5)``), so a large fraction of
    the frame legitimately looks PAST the table's two edges at background (floor,
    fixtures) tens of centimetres away on a real D435i.

    MEASURED failure (corrected mechanism -- Task 13 re-review follow-up): on a
    synthetic 300x300 mm table with a floor 750 mm below and a corner capture whose
    reticle sits exactly on the corner (73.75% of the frame is floor, 26.25% table),
    the unfiltered points fed into ``fit_global_plane`` produced per-set RMS
    **384.26 mm** against the ``survey_coplanar_reject_mm`` gate of 8 mm. This is
    NOT "off-plane points measured against the correct plane" (the docstring's
    original, wrong explanation) -- ``fit_global_plane`` always re-derives its OWN
    plane via an internal RANSAC pass over whatever points it is given; it never
    receives or uses this function's ``plane_normal_cam``/``plane_point_cam`` at
    all. With the floor as the 73.75% MAJORITY of the unfiltered points, THAT
    internal RANSAC locks onto the floor (verified: its recovered plane's centroid
    sits at z=-750, not z=0), and the reported 384.26 mm is the residual of the
    MINORITY 567 table points against that WRONG (floor) plane:
    ``750 mm (table-to-floor distance) * sqrt(567/2160) = 384.2 mm`` -- matching the
    measurement (the 1593 floor points fit their own RANSAC-recovered plane almost
    exactly, residual ~0). Filtering to a band around the plane ``survey_surface``
    already fit for THIS exact frame (a real, independent detector -- not
    ``fit_global_plane``'s own internal RANSAC) removes the floor points before
    ``fit_global_plane`` ever sees them, so its internal RANSAC has nothing left to
    lock onto but the table -- verified: the same fixture now measures **0.00 mm**
    per-set RMS (only exact-zero floating-point residual from the synthetic
    ray-plane intersection remains). See the Task 13 review fix reports for the
    full before/after measurement and this mechanism trace.
    """
    d = np.asarray(depth, dtype=float)
    if d.ndim != 2 or d.size == 0:
        return np.zeros((0, 3), dtype=float), 0.0, 0.0
    h, w = d.shape
    ys, xs = np.mgrid[0:h:stride, 0:w:stride]
    n_grid_total = int(ys.size)
    z = d[ys, xs]
    valid = z > 0
    n_valid = int(np.count_nonzero(valid))
    if n_valid == 0:
        return np.zeros((0, 3), dtype=float), 0.0, 0.0
    xs_v = xs[valid].astype(float)
    ys_v = ys[valid].astype(float)
    z_v = z[valid].astype(float)
    fx, fy, cx, cy = float(K[0][0]), float(K[1][1]), float(K[0][2]), float(K[1][2])
    x_cam = (xs_v - cx) / fx * z_v
    y_cam = (ys_v - cy) / fy * z_v
    pts_cam = np.column_stack([x_cam, y_cam, z_v])

    n = np.asarray(plane_normal_cam, dtype=float)
    n = n / max(float(np.linalg.norm(n)), 1e-9)
    p0 = np.asarray(plane_point_cam, dtype=float)
    dist = np.abs((pts_cam - p0) @ n)
    keep = dist <= float(band_mm)
    n_inliers = int(np.count_nonzero(keep))
    pts_cam = pts_cam[keep]

    purity_frac = n_inliers / n_valid
    coverage_frac = n_inliers / n_grid_total

    if len(pts_cam) == 0:
        return np.zeros((0, 3), dtype=float), purity_frac, coverage_frac

    pts_cam_h = np.column_stack([pts_cam, np.ones(len(pts_cam))])
    T = np.asarray(T_base_cam, dtype=float)
    return (pts_cam_h @ T.T)[:, :3], purity_frac, coverage_frac


def _work_boundary(scfg, color) -> "dict | None":
    """Segment the object under the reticle with the configured boundary engine.

    Mirrors the live-preview dispatch in ``module.py``'s ``/live/start`` (colour
    only for ``"color"``; SAM with an optional colour fallback for ``"sam"`` /
    ``"sam_then_color"``) but runs INLINE/synchronously rather than through
    ``SamBoundaryWorker``'s background thread: that worker exists so ~450 ms/frame
    SAM inference never hitches the ~6 fps live video, but a compact lock or a
    five-position capture is a single one-shot authoritative measurement, not a
    video frame, so a blocking call is the right shape here (and the simplest
    one that cannot get out of sync with the frame it was asked to segment).

    Returns the FULL boundary dict -- ``{outline_uv, polygon_uv, fill_frac,
    border_touch, overruns[, contrast]}`` (see ``color_work_boundary`` /
    ``sam_work_boundary``) -- or ``None`` when every configured engine
    abstains. ``lock_scan_surface`` (Task 18) reads ``overruns`` directly for
    ``classify_compact``'s segmentation-confirmed-boundary gate;
    ``_boundary_polygon_uv`` below is a thin backward-compatible wrapper for
    ``five_position_capture``'s corner-evidence extraction, which only ever
    needed the polygon.
    """
    if not scfg.color_boundary_enabled:
        return None
    cb = None
    if scfg.boundary_engine in ("sam", "sam_then_color"):
        try:
            cb = sam_work_boundary(
                color, model_dir=scfg.sam_model_dir, encoder_file=scfg.sam_encoder_file,
                decoder_file=scfg.sam_decoder_file, min_score=scfg.sam_min_score,
                max_fill_frac=scfg.sam_max_fill_frac, point_uv=(0.5, 0.5))
        except Exception:
            cb = None
        if cb is None and scfg.boundary_engine == "sam_then_color":
            try:
                cb = color_work_boundary(
                    color, reticle_frac=scfg.center_patch_frac,
                    min_color_dist=scfg.color_boundary_min_color_dist,
                    seg_width=scfg.color_boundary_seg_width)
            except Exception:
                cb = None
    else:
        try:
            cb = color_work_boundary(
                color, reticle_frac=scfg.center_patch_frac,
                min_color_dist=scfg.color_boundary_min_color_dist,
                seg_width=scfg.color_boundary_seg_width)
        except Exception:
            cb = None
    return cb


def _boundary_polygon_uv(scfg, color) -> "np.ndarray | None":
    """Backward-compatible wrapper around :func:`_work_boundary` returning just
    ``polygon_uv`` (normalized 0-1 image coords, a CLOSED contour -- see the
    ``closed=True`` note where this feeds ``extract_corner_evidence``), or
    ``None`` when every configured engine abstains. Used by
    ``five_position_capture``'s corner-evidence extraction only."""
    cb = _work_boundary(scfg, color)
    if cb is None:
        return None
    return np.asarray(cb["polygon_uv"], dtype=float)


def five_position_capture(services, survey: FivePositionSurvey) -> dict:
    """One authoritative step-and-measure acquisition for the guided five-position
    workframe survey (spec §7/§9) -- whichever position ``survey.step`` currently
    expects (``"center"`` or ``"corner1"``..``"corner4"``).

    Reuses ``_authoritative_acquisition`` -- the SAME camera-hold / grab / fuse /
    readiness-survey / robot-state-refresh sequence :func:`lock_scan_surface` uses
    -- so the two authoritative capture paths cannot silently drift apart; only the
    logic AFTER acquisition (evidence extraction, ``CaptureRecord`` shape, and what
    to do with the result) differs, matching what is genuinely different between a
    single compact lock and one position of a five-position survey.

    A five-position survey combines FIVE separately-registered positions, so
    (unlike the compact lock, which merely records ``robot.stationary``) a moving
    robot at capture time is rejected here explicitly and immediately -- before any
    further per-frame compute -- rather than silently accepted: this is the core
    safety contract of the whole feature (a stale/live pose blend at any one
    position would corrupt the cross-position plane/rectangle fit in a way no
    downstream check could distinguish from a real geometry error). The rejection
    is INTENTIONALLY worded differently from ``FivePositionSurvey.add_capture``'s
    own "robot was moving during the capture" message (Task 13 review Finding 4):
    the two are genuinely different layers checking genuinely different things (this
    one never even builds a ``CaptureRecord`` -- ``add_capture`` is not called at
    all -- whereas ``add_capture``'s check is a defence-in-depth backstop for a
    record that DID get built), and distinct text keeps a test able to tell which
    layer actually fired instead of both raising byte-identical strings.

    Also checks the RoboDK driver is actually connected to the physical controller
    (Task 13 review Finding 2) BEFORE trusting anything ``_authoritative_acquisition``
    reads: ``refresh_robot_state`` reads ``current_joints()`` twice and calls them
    "stationary" when they agree, but that only means the arm didn't move ACCORDING
    TO THE MODEL POSE RODK IS TRACKING -- with the driver down (or never connected),
    RoboDK's model freezes at its last commanded pose while the physical arm can be
    anywhere, so a dead link reads as a perfectly "stationary" robot every time. A
    five-position survey is uniquely exposed to this: it *depends* on the pose
    genuinely differing across five captures, so a frozen mirror would silently
    register all five positions at the same fictional pose -- checked first (before
    the camera grab, not just before trusting the snapshot) since it is a cheap
    RoboDK round trip next to a multi-second Wi-Fi depth grab, so a disconnected
    driver fails in milliseconds instead of after burning that grab for nothing.
    Scoped to this function only -- ``lock_scan_surface``/``_authoritative_acquisition``
    are untouched, so the compact/crop lock path's behaviour does not change.
    """
    K = services.config.camera.K
    scfg = services.config.scan
    rdk = services.rdk
    kind = survey.step
    if kind == "review":
        raise RuntimeError("survey already has all five captures")

    driver_ok, driver_msg = rdk.robot_connected()
    if not driver_ok:
        raise RuntimeError(
            f"{kind} capture: RoboDK is not connected to the physical robot "
            f"controller ({driver_msg or 'driver not ready'}) - a robot-state read "
            "cannot be trusted while the driver link is down; reconnect the driver "
            "and remeasure")

    frame, n_frames, reading, measurement, snapshot, _raw_frames = _authoritative_acquisition(
        services, owner="scan-five-position")
    if not snapshot.stationary:
        raise RuntimeError(
            f"{kind} capture: the robot moved between the two robot-state reads "
            "taken for this capture - stop the robot and remeasure")

    if kind != "center" and not measurement.detected:
        # _authoritative_acquisition's own survey_surface() call uses
        # _survey_thresholds(scfg)'s min_valid_depth_frac default (0.3 -- a
        # centre-patch-style "is there a substantial surface in frame" sanity
        # floor, appropriate for the compact lock and the CENTRE step, where a
        # well-aimed capture fills most of the frame). A well-aimed CORNER
        # capture legitimately cannot clear that: centring the reticle on a
        # 90-degree corner caps real coverage at 25% of the frame (see
        # survey_corner_min_plane_coverage_frac's own config comment) -- below
        # the 0.3 floor -- so a capture at a corner's geometric BEST would be
        # misreported as "no usable surface plane detected" before this
        # function's own purity/coverage checks ever run (this is the "two
        # thresholds uncoordinated" observation from the Task 13 review). Rather
        # than touch _authoritative_acquisition (the review's protected
        # refactor) or SurveyThresholds' own default (shared with the compact
        # lock), re-run survey_surface here on the frame already fetched -- a
        # second, cheap, pure-numpy RANSAC pass, not a second camera grab --
        # scoped to corner steps only, at the small fixed sanity floor
        # (_CORNER_DETECT_SANITY_FRAC, not survey_corner_min_plane_coverage_frac
        # -- see that constant's own comment for why reusing one value for both
        # would make the coverage gate below unreachable).
        corner_th = replace(_survey_thresholds(scfg),
                            min_valid_depth_frac=_CORNER_DETECT_SANITY_FRAC)
        measurement = survey_surface(frame.depth, K, corner_th, depth_scale=scfg.depth_scale)

    if not measurement.detected:
        raise RuntimeError(
            f"{kind} capture: no usable surface plane detected - reposition and recapture")

    T_base_cam = snapshot.camera_T_np()
    plane_points_base, plane_purity_frac, plane_coverage_frac = _deproject_plane_points_mm(
        frame.depth, K, T_base_cam, plane_normal_cam=measurement.normal_cam,
        plane_point_cam=measurement.centroid_cam_mm,
        band_mm=float(scfg.survey_plane_inlier_band_mm))

    # PRECONDITION (logged for hardware validation, not fixed here -- Task 13 review
    # round 3): purity_frac and coverage_frac are only as trustworthy as the plane
    # `measurement` (survey_surface's own RANSAC) actually selected. If the real
    # background at a corner is itself a large, COHERENT surface -- e.g. this exact
    # ~26/74 table/floor geometry, measured in round 1 -- survey_surface's RANSAC can
    # lock onto the BACKGROUND instead of the table (majority wins the vote). When
    # that happens, both metrics are computed relative to the WRONG plane: floor
    # points read as "inliers," table points read as "off-plane," so purity/coverage
    # both come back HIGH and this capture is silently accepted with FLOOR geometry
    # mislabelled as table geometry -- surfacing only later, and only as a confusing
    # whole-survey "not coplanar" rejection at finish() that does not name the real
    # cause. This module's own tests sidestep the failure by overriding the returned
    # plane with independently-computed ground truth; on real hardware there is no
    # such override. Plane SELECTION is survey_surface's responsibility, not this
    # function's, and is explicitly out of scope here.
    #
    # CaptureRecord.valid_frac: the CENTRE step keeps the coarse centre-patch metric
    # (`reading.valid_frac`) -- the reticle really does sit ON the surface there, so
    # "is the centre of frame valid?" is the right question, matching the compact
    # lock's own convention (Task 13 review, remedy ii). A CORNER step's reticle
    # straddles the table/background boundary BY DESIGN (it is aimed AT the corner),
    # so that same question is the WRONG one: a corner shot aimed past the table into
    # open space beyond the D435i's reliable range returns zeros there and would be
    # spuriously rejected ("not enough valid depth") even though the visible table
    # portion is perfectly good. `plane_purity_frac` (see _deproject_plane_points_mm)
    # answers the question that actually matters for a corner -- of the depth we DID
    # get, how much of it is trustworthy -- and is robust to whether the missing
    # portion of the frame is silent (0 depth) or a real, off-plane surface.
    #
    # Coverage is gated SEPARATELY, before add_capture, using a corner-specific
    # threshold: purity_frac alone cannot catch "genuinely too little table in view"
    # (a tiny-but-clean sliver against silence also reads purity_frac ~1.0), and the
    # SHARED add_capture gate (scfg.min_valid_depth_frac, default 0.5, tuned for the
    # compact lock's typically near-full-frame centre-patch coverage) cannot use a
    # looser number for corners without changing behaviour for every OTHER step and
    # path that field feeds -- so it is checked here instead, against a threshold
    # sized for what a corner can ever legitimately achieve: centring the reticle
    # exactly on a 90-degree corner caps real table coverage at 25% of the frame (a
    # geometric ceiling, not a fixture artefact -- confirmed by ray-tracing all four
    # corners of the test table), so scan.survey_corner_min_plane_coverage_frac
    # defaults well below that (0.10), admitting normal aiming imprecision while
    # still refusing a genuinely-too-small sliver.
    if kind != "center":
        if plane_coverage_frac < float(scfg.survey_corner_min_plane_coverage_frac):
            raise RuntimeError(
                f"{kind} capture: only {plane_coverage_frac:.0%} of the frame is on "
                "the work plane (< "
                f"{float(scfg.survey_corner_min_plane_coverage_frac):.0%} minimum) - "
                "too little of the table is in view; move closer to the corner or "
                "reposition and recapture")
        valid_frac = plane_purity_frac
    else:
        valid_frac = float(reading.valid_frac)

    R, t = T_base_cam[:3, :3], T_base_cam[:3, 3]
    normal_base = R @ np.asarray(measurement.normal_cam, dtype=float)
    if normal_base[2] < 0:
        normal_base = -normal_base
    point_base = R @ np.asarray(measurement.centroid_cam_mm, dtype=float) + t

    record = CaptureRecord(
        kind=kind, robot=snapshot, measurement_ts=float(frame.timestamp),
        captured_at=snapshot.fetched_at, n_frames=int(n_frames),
        standoff_mm=float(measurement.standoff_mm), tilt_deg=float(measurement.tilt_deg),
        valid_frac=valid_frac, plane_rms_mm=_plane_rms_mm(frame.depth, K),
        plane_normal_base=tuple(float(v) for v in normal_base),
        plane_point_base=tuple(float(v) for v in point_base))

    evidence = None
    if kind != "center":
        polygon_uv = _boundary_polygon_uv(scfg, frame.color)
        if polygon_uv is not None:
            # CRITICAL: production boundary polygons (color_work_boundary /
            # sam_work_boundary, both via mask_to_boundary's cv2.findContours) are
            # CLOSED contours -- closed=True lets the corner's arm walk wrap around
            # the array end to find its other arm when the corner sits near the
            # contour's start/end vertex. Omitting it fails safe (returns None, an
            # operator-visible "no usable edge evidence" rejection) but silently
            # loses real wraparound coverage; see corner_evidence.extract_corner_evidence's
            # own docstring.
            evidence = extract_corner_evidence(
                frame.depth, K, polygon_uv, T_base_cam,
                corner_hint_uv=(0.5, 0.5), closed=True)

    state = survey.add_capture(record, plane_points_base, evidence)
    services.bus.publish(JobEvent("survey", state))
    return state


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
    trimmed_color = raw.get("trimmed_corners_color_mm")
    if camera_cfg is not None and corners_color is not None:
        calibrated_uv = _project_color_corners_uv(corners_color, camera_cfg)
        if calibrated_uv is not None:
            edge_angle = _outline_edge_angle_deg(calibrated_uv)
            # TRUST THE FITTED RECTANGLE: if its calibrated corners sit inside the
            # colour frame with a margin, the object is bounded in view — draw that
            # rectangle and mark it framed. The server's depth_fully_framed test keys
            # off RAW depth-inlier pixels touching the border, so a few stray fringe
            # points made a well-margined block read as an overrun and fall back to
            # the generic work square. The projected rectangle corners (built from the
            # trimmed fit) are the reliable "does the object fit the view" signal, so
            # the host overrides the server's over-eager crop here. The framing test
            # deliberately uses the RAW corners (the full object extent).
            frame_margin = float(getattr(scfg, "live_frame_margin_uv", 0.02))
            rectangle_in_frame = bool(np.all(
                (calibrated_uv[:, 0] >= frame_margin)
                & (calibrated_uv[:, 0] <= 1.0 - frame_margin)
                & (calibrated_uv[:, 1] >= frame_margin)
                & (calibrated_uv[:, 1] <= 1.0 - frame_margin)))
            fully_framed = rectangle_in_frame
            surface_mode = "full" if rectangle_in_frame else "crop"
            if rectangle_in_frame:
                # Draw the density/colour-TRIMMED rectangle so the LIVE overlay hugs
                # the surface the same way the locked/inserted work rectangle does
                # (the survey lock trims via plane._density_extent_1d). Both are now
                # trimmed, so the box no longer visibly shrinks on lock. Fall back to
                # the raw fitted rectangle if the server did not send trimmed corners
                # (a pre-deploy server) — never worse than before. When the object
                # overruns, keep the server's generic reticle square (raw outline_uv).
                trimmed_uv = (_project_color_corners_uv(trimmed_color, camera_cfg)
                              if trimmed_color is not None else None)
                outline_uv = (trimmed_uv if trimmed_uv is not None
                              else calibrated_uv).tolist()
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
    # Framing standoff = the standoff that frames the *physical* surface with a border.
    # This is distance-invariant (it depends on the object's size, not where the camera
    # is now), so a parked operator sees a steady target instead of a moving goalpost.
    # It is only trustworthy while the object is BOUNDED in view — once it overruns, the
    # rectangle/extent are clipped to the frame and the estimate is meaningless.
    framing_standoff = None
    if fit_per_margin is not None:
        framing_standoff = float(np.clip(
            float(fit_per_margin) * float(scfg.frame_margin),
            float(scfg.accurate_min_mm), float(scfg.accurate_max_mm)))
    elif extent is not None and camera_cfg is not None:
        try:
            sx, sy = [float(v) for v in extent]
            W, H = camera_cfg.size
            K = camera_cfg.K
            framing_standoff = float(np.clip(
                max(float(scfg.frame_margin) * sx * float(K[0, 0]) / float(W),
                    float(scfg.frame_margin) * sy * float(K[1, 1]) / float(H)),
                float(scfg.accurate_min_mm), float(scfg.accurate_max_mm)))
        except Exception:
            framing_standoff = None

    if surface_mode == "crop":
        # The surface overruns the view, so its live rectangle/extent are clipped to the
        # frame — the framing standoff cannot be re-measured here. If a target was already
        # latched while the surface WAS framed (``previous_ideal_mm``), HOLD it: a small
        # over-nudge past the framing distance must NOT collapse the goal to accurate_min
        # and then drive the operator even closer (which deepens the overrun). That was
        # the "target says 590, I move toward it, it jumps to 300, I can never reach it"
        # bug. Only fall back to the work-close accurate standoff when nothing was ever
        # framed — a genuinely oversized surface (whole table), where working close and
        # projecting the generic reticle square is the right policy.
        ideal_distance = (float(previous_ideal_mm) if previous_ideal_mm is not None
                          else float(scfg.accurate_min_mm))
    elif framing_standoff is not None:
        candidate = round(framing_standoff / 10.0) * 10.0
        # Recommendation deadband: 410/420 is sensor/fitting noise, not a useful
        # instruction. Hold the previous target until the estimate moves >=20 mm.
        if previous_ideal_mm is not None and abs(candidate - previous_ideal_mm) < 20.0:
            ideal_distance = float(previous_ideal_mm)
        else:
            ideal_distance = candidate
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
    if finite_surface and center_cam is not None:
        gates["center"] = bool(
            abs(float(center_cam[0])) <= float(scfg.center_tol_mm)
            and abs(float(center_cam[1])) <= float(scfg.center_tol_mm))
    if fully_framed is not None:
        gates["framed"] = bool(fully_framed)
    # EDGE A: the platform edge's yaw alignment. It is a meaningful reading only for
    # an elongated platform (the long edge defines the work-frame X); a near-square
    # platform or the generic crop has an ambiguous edge, so the lamp is advisory.
    # It is ADVISORY only (never part of ``ok``) so it informs without making lock
    # harder — the lamp reflects reality instead of showing a permanent "·".
    edge_aspect = _aspect_ratio(raw.get("rectangle_size_mm") or extent)
    edge_meaningful = bool(finite_surface and edge_angle is not None
                           and edge_aspect is not None
                           and edge_aspect >= float(scfg.edge_gate_min_aspect))
    gates["edge"] = (abs(float(edge_angle)) <= float(scfg.edge_align_tol_deg)
                     if edge_meaningful else True)
    ok_gates = dict(gates)
    ok_gates.pop("edge", None)          # advisory — never blocks readiness
    if surface_mode == "crop":
        ok_gates.pop("framed", None)
    elif fully_framed:
        # For a finite platform, the rectangle centroid is the planning target.
        # Keep X/Y as guidance, not a hard readiness gate.
        ok_gates.pop("center", None)
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
                      if finite_surface and edge_angle is not None else None),
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


def _polygon_area(poly: np.ndarray | None) -> float:
    if poly is None:
        return 0.0
    pts = np.asarray(poly, dtype=float)
    if pts.ndim != 2 or len(pts) < 3 or pts.shape[1] < 2:
        return 0.0
    x, y = pts[:, 0], pts[:, 1]
    return abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))) * 0.5


def _rectangles_consistent(prev: dict, cur: dict, scfg) -> bool:
    """True when live full-rectangle telemetry looks like the same static view."""
    if prev.get("surface_mode") != "full" or cur.get("surface_mode") != "full":
        return False
    if prev.get("fully_framed") is not True or cur.get("fully_framed") is not True:
        return False
    prev_poly = _payload_outline_uv(prev)
    cur_poly = _payload_outline_uv(cur)
    if prev_poly is None or cur_poly is None or prev_poly.shape != cur_poly.shape:
        return False
    aligned = _align_polygon_like(prev_poly, cur_poly)
    mean_corner_motion = float(np.mean(np.linalg.norm(aligned - prev_poly, axis=1)))
    if mean_corner_motion > float(getattr(scfg, "live_rect_latch_outline_uv", 0.04)):
        return False
    return True


def _should_reset_live_smoothing(prev: dict, cur: dict, scfg) -> bool:
    if not prev or not prev.get("detected") or not cur.get("detected"):
        return True
    if prev.get("distance_mm") is not None and cur.get("distance_mm") is not None:
        if abs(float(cur["distance_mm"]) - float(prev["distance_mm"])) > float(scfg.live_aim_reset_distance_mm):
            return True
    return False


def camera_pose_moved(cur_T, ref_T, trans_tol_mm: float, rot_tol_deg: float) -> bool:
    """True when the camera pose left ``ref_T`` by more than the given tolerances.

    RoboDK mirrors the physical arm (position monitoring is on after connect), so
    the camera pose is the authoritative "did the robot actually move" signal —
    unlike the RealSense plane fit, whose per-frame noise makes a parked arm look
    like it is drifting. ``ref_T`` is the pose the current live reading was anchored
    to; a real jog crosses these tolerances, sensor noise never does. ``None`` on
    either side reads as *moved* (fail open → fall back to plain smoothing).
    """
    if cur_T is None or ref_T is None:
        return True
    cur = np.asarray(cur_T, float)
    ref = np.asarray(ref_T, float)
    if cur.shape != (4, 4) or ref.shape != (4, 4):
        return True
    if float(np.linalg.norm(cur[:3, 3] - ref[:3, 3])) > float(trans_tol_mm):
        return True
    rel = ref[:3, :3].T @ cur[:3, :3]
    ang = float(np.degrees(np.arccos(np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0))))
    return ang > float(rot_tol_deg)


def _vision_says_moved(current: dict, previous: dict, scfg) -> bool:
    """True when the depth reading shifted far more than static sensor noise.

    A safety net for the hold: the live rectangle is computed on the Jetson from
    depth, so it reprojects when the camera physically moves even if RoboDK is not
    mirroring the arm (driver not monitoring) and the pose gate would wrongly read
    "static". Uses only the STABLE signals — standoff and tilt magnitude (≈1 mm /
    ≈0.5° noise) — which have a clean gap above the noise floor, unlike the ~80 mm
    rectangle-centroid jitter. A real dolly-in/out or level change releases the
    hold; per-frame noise never does. (A pure lateral jog with the driver off is the
    one case neither signal catches, and is an accepted degraded-mode limitation.)
    """
    pd, cd = previous.get("distance_mm"), current.get("distance_mm")
    if pd is not None and cd is not None:
        if abs(float(cd) - float(pd)) > float(scfg.live_hold_vision_distance_mm):
            return True
    pt, ct = previous.get("tilt_deg"), current.get("tilt_deg")
    if pt is not None and ct is not None:
        if abs(float(ct) - float(pt)) > float(scfg.live_hold_vision_tilt_deg):
            return True
    return False


def _hold_scan_payload(current: dict, previous: dict) -> dict:
    """Freeze every pose-derived readout at the previous live reading.

    Used when the robot is parked: the operator must see rock-steady X/Y/Z + tilt
    A/B/C + rectangle. Only the liveness bookkeeping (timestamp, valid fraction) and
    the config-driven tolerances refresh, so the frame never goes stale and the HUD
    stays green without chasing sensor noise. The gates/ok are held too — a static
    scene cannot change readiness.
    """
    out = dict(previous)
    out["live"] = True
    out["held"] = True
    # Keep the frame fresh (>2 s stale telemetry is dropped elsewhere) but hold
    # everything geometry/pose derived.
    for key in ("measurement_ts", "valid_frac"):
        if current.get(key) is not None:
            out[key] = current[key]
    return out


def stabilize_live_scan_payload(current: dict, previous: dict | None, scfg,
                                *, robot_static: bool = False) -> dict:
    """Temporal smoothing for live scan HUD telemetry only.

    The lock/create-target path still grabs an authoritative raw RGBD frame. This
    filter prevents per-frame RealSense plane/edge noise from making a static robot
    look like it is moving in the live aiming UI.

    ``robot_static`` (from the camera pose, see :func:`camera_pose_moved`) is the
    hard gate: when the arm has not moved, every pose-derived readout is *held* at
    the previous reading rather than merely smoothed, so a parked robot shows zero
    jitter on X/Y/Z and A/B/C. Smoothing/hysteresis still runs while the arm moves.
    """
    if not current or current.get("live") is not True:
        return current

    # Robot-parked HOLD with symmetric hysteresis. The RealSense plane fit is noisy
    # enough on a bare surface (measured ~6° tilt / ~50 mm centroid swing at 0.8 m with
    # the arm PARKED) that the freeze must debounce BOTH edges: ENGAGE only after the
    # reading has been still for ``live_hold_settle_frames`` (so the rectangle/tilt
    # average first), and RELEASE only after motion PERSISTS for
    # ``live_hold_release_frames``. Without a release debounce a single noisy frame drops
    # the hold, re-latches a fresh noisy sample, and the operator sees the readout sit
    # still and then suddenly jump. Motion is signalled by the RoboDK camera pose (it
    # mirrors the arm) via ``robot_static``, with a large vision shift as the fallback
    # when the driver is not monitoring.
    have_context = bool(previous is not None and previous.get("detected")
                        and current.get("detected") and previous.get("live") is True)
    was_held = bool(previous and previous.get("held"))
    moved_now = (not robot_static) or _vision_says_moved(current, previous or {}, scfg)
    static_frames = 0
    moving_frames = 0
    if have_context and was_held:
        if not moved_now:
            out = _hold_scan_payload(current, previous)
            out["static_frames"] = int(previous.get("static_frames", 0)) + 1
            out["moving_frames"] = 0
            return out
        # Moving while held: ride out brief noise blips — only release once the motion
        # persists, so per-frame plane-fit noise can never break a parked hold.
        moving_frames = int(previous.get("moving_frames", 0)) + 1
        release = max(1, int(getattr(scfg, "live_hold_release_frames", 2)))
        if moving_frames < release:
            out = _hold_scan_payload(current, previous)
            out["static_frames"] = int(previous.get("static_frames", 0))
            out["moving_frames"] = moving_frames
            return out
        # else: motion confirmed → release, fall through to live smoothing below.
    elif have_context and not moved_now:
        # Settling toward a fresh hold: average a few frames before freezing so the
        # latched rectangle/tilt is not a single half-settled sample.
        static_frames = int(previous.get("static_frames", 0)) + 1
        settle = max(1, int(getattr(scfg, "live_hold_settle_frames", 5)))
        if static_frames >= settle:
            out = _hold_scan_payload(current, previous)
            out["static_frames"] = static_frames
            out["moving_frames"] = 0
            return out
        # else: fall through to smoothing to keep settling; counter carried below.
    if previous is None or _should_reset_live_smoothing(previous, current, scfg):
        res = dict(current)
        res["static_frames"] = static_frames
        res["moving_frames"] = moving_frames
        return res

    out = dict(current)
    rect_consistent = _rectangles_consistent(previous, current, scfg)
    stable_frames = (int(previous.get("rect_stable_frames", 1)) + 1
                     if rect_consistent else 1)
    out["rect_stable_frames"] = stable_frames
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
            aligned = _align_polygon_like(prev_poly, cur_poly)
            if key == "outline_uv":
                prev_area = _polygon_area(prev_poly)
                cur_area = _polygon_area(aligned)
                # Live RealSense validity can drop an edge or low-texture band for
                # one telemetry frame. Do not let the blue work rectangle shrink to
                # that partial footprint; only grow/refine it, or reset on a real
                # distance change handled above.
                if prev_area > 0.0 and cur_area < 0.98 * prev_area:
                    out[key] = prev_poly.tolist()
                    for size_key in ("extent_mm", "rectangle_size_mm"):
                        if previous.get(size_key) is not None:
                            out[size_key] = previous.get(size_key)
                    continue
            out[key] = aligned.tolist()
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
            latch_frames = int(getattr(scfg, "live_rect_latch_frames", 3))
            if (out.get("surface_mode") == "full"
                    and out.get("fully_framed") is True
                    and stable_frames >= latch_frames
                    and previous.get("ok")
                    and not gates["center"]):
                prev_mv = _as_float_array(previous.get("move_cam"), 3)
                if prev_mv is not None:
                    out["move_cam"] = [float(prev_mv[0]), float(prev_mv[1]), float(mv[2])]
                    gates["center"] = True
                    out["center_latched"] = True
    # EDGE A recomputed from the smoothed yaw with hysteresis (advisory; see
    # live_scan_telemetry_payload). Preserved through smoothing rather than dropped so
    # the EDGE A lamp keeps reflecting live alignment instead of going blank.
    yaw = out.get("yaw_a_deg")
    edge_aspect = _aspect_ratio(out.get("rectangle_size_mm") or out.get("extent_mm"))
    edge_meaningful = bool(out.get("surface_mode") == "full" and yaw is not None
                           and edge_aspect is not None
                           and edge_aspect >= float(scfg.edge_gate_min_aspect))
    if edge_meaningful:
        tol = float(out.get("edge_align_tol_deg", scfg.edge_align_tol_deg))
        if prev_gates.get("edge"):
            tol += float(getattr(scfg, "live_aim_edge_hysteresis_deg", 10.0))
        gates["edge"] = abs(float(yaw)) <= tol
    else:
        gates["edge"] = True
    if out.get("fully_framed") is not None:
        gates["framed"] = bool(out.get("fully_framed"))
    ok_gates = dict(gates)
    ok_gates.pop("edge", None)          # advisory — never blocks readiness
    if out.get("surface_mode") == "crop":
        ok_gates.pop("framed", None)
    elif out.get("fully_framed") is True:
        ok_gates.pop("center", None)
    out["gates"] = gates
    out["ok"] = all(bool(v) for v in ok_gates.values())
    out["stabilized"] = True
    # Carry the debounce counters so the hold engages after N still frames and releases
    # only after M moved frames (symmetric hysteresis).
    out["static_frames"] = static_frames
    out["moving_frames"] = moving_frames
    return out


def annotate_pose_liveness(metrics: dict, *, pose_T, driver_ok: bool) -> dict:
    """Mark whether the pose-derived readouts (X/Y move_cam, jog guidance) reflect the
    PHYSICAL robot right now, or a stale/model pose.

    RoboDK's camera pose only mirrors the real arm while the driver is connected and
    actively monitoring (see :meth:`RdkIO.robot_connected`). Without that link,
    ``camera_pose_T()`` still returns *a* pose — RoboDK's last-known/model pose — so
    the smoothing/hold logic upstream runs exactly the same either way. This flag is
    the HUD's only signal to tell the operator the difference: ``True`` only when a
    connected driver AND an actual pose reading both back the current readout.
    Pure/no I/O so it is testable without hardware.
    """
    metrics["pose_live"] = bool(driver_ok and pose_T is not None)
    return metrics


def _clear_prior_scan_targets(rdk: RdkIO, prefix: str, pub) -> None:
    """Delete previous ``TasniScan_*`` targets and any stray calibration
    targets/keep-out before writing a fresh target set.

    Shared by the single-aim and tiled-tour generators (Task 12) — pure code
    motion out of ``generate_scan_targets``, same behaviour as before.
    """
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


def _screen_scan_collision_candidates(
    services, scfg, reachable: list[tuple[int, np.ndarray]], n_reach: int,
) -> tuple[list[tuple[int, np.ndarray]], list, bool, int, bool, list[str]]:
    """Collision-screen reachable scan candidates.

    Shared by the single-aim and tiled-tour generators (Task 12) — pure code
    motion out of ``generate_scan_targets``, same semantics as before: a
    STRICT hard-fail by default (a production target set the real robot is
    about to run), with a soft bypass only when the operator has explicitly
    opted into it via ``scan.collision_filter_hard_fail = False``.

    Returns ``(reachable, reach_joints, col_checked, n_collide,
    collision_filter_bypassed, pair_examples)`` — ``reachable`` narrowed to
    the collision-free survivors (or, on a soft bypass, restored to the full
    pre-collision reachable set).
    """
    rdk: RdkIO = services.rdk
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
    reachable_before_collision = list(reachable)
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
    return (reachable, reach_joints, col_checked, n_collide,
            collision_filter_bypassed, pair_examples)


def _generate_tiled_scan_targets(
    services, locked: LockedScanSurface, seed_T: np.ndarray, seed_joints,
    tool_offset_mm: float, pub,
) -> dict:
    """Tour a five-position-surveyed rectangle too large for one camera view
    (Task 12): plan a TILED close-range tour (:func:`plan_rect_tour`) and feed
    it through the SAME per-aim candidate generation / reachability /
    collision / diversity machinery the single-aim path uses — just once per
    tile instead of once for the whole surface, accumulating candidates
    across tiles (brief step 3).

    Reachability/collision screening is reused verbatim via
    :func:`_screen_scan_collision_candidates`; diversity selection reuses
    :func:`select_diverse`, but per tile (each tile already IS the coverage
    unit — unlike the single-aim orbit, there is no single wide footprint to
    spread rotation over). Targets are named ``TasniScan_T{tile:02d}_{k}``.
    """
    cfg = services.config
    scfg = cfg.scan
    rdk: RdkIO = services.rdk
    K = cfg.camera.K
    W, H = cfg.camera.size
    prefix = scfg.target_prefix
    rec = locked.survey_record

    plan = plan_rect_tour(rec.corners_np(), np.asarray(rec.plane_normal_base, float),
                          K, (W, H), scfg)
    pub(f"five-position survey: tiling a {rec.size_mm[0]:.0f}×{rec.size_mm[1]:.0f} mm "
        f"surface into {len(plan.aims)} tile(s) at {plan.standoff_mm:.0f} mm standoff, "
        f"voxel={plan.voxel_size_m * 1000:.1f} mm")

    _clear_prior_scan_targets(rdk, prefix, pub)

    # Per-aim candidate generation (reuses generate_calibration_poses exactly
    # as the single-aim path does — once per tile, centred/aimed at that
    # tile's own AimPoint), accumulated across all tiles.
    tile_candidates: list[tuple[int, np.ndarray]] = []
    for t, aim in enumerate(plan.aims):
        cands = generate_calibration_poses(
            seed_T, count=aim.n_views, look_distance_mm=aim.standoff_mm,
            cone_half_angle_deg=aim.cone_half_angle_deg, roll_max_deg=aim.roll_max_deg,
            distance_jitter=scfg.distance_jitter,
            target_center=np.asarray(aim.point_base_mm, float),
            target_normal=-np.asarray(aim.view_dir_base, float),
            min_perpendicular_mm=aim.min_perpendicular_mm)
        tile_candidates.extend((t, T) for T in cands)

    reachable = [(t, T) for t, T in tile_candidates if rdk.is_reachable(T)]
    n_reach = len(reachable)
    if n_reach < SCAN_MIN_VIEWS:
        raise RuntimeError(
            f"only {n_reach} reachable pose(s) across {len(plan.aims)} tile(s) of the "
            f"surveyed rectangle (need >= {SCAN_MIN_VIEWS}) — jog to a more open part of "
            "the workspace and retry")

    (reachable, reach_joints, col_checked, n_collide, collision_filter_bypassed,
     pair_examples) = _screen_scan_collision_candidates(services, scfg, reachable, n_reach)

    # Per-tile diversity selection: each tile already targets its own patch of
    # the surface, so diversity is spread WITHIN a tile's own reachable,
    # collision-free candidates (angle variety for TSDF fusion), not across
    # tiles the way the single-aim path spreads across a shared footprint.
    by_tile: dict[int, list[int]] = {}
    for k, (t, _T) in enumerate(reachable):
        by_tile.setdefault(t, []).append(k)

    chosen: list[tuple[int, np.ndarray, object]] = []   # (tile, T, joints)
    for t, aim in enumerate(plan.aims):
        idxs = by_tile.get(t, [])
        if not idxs:
            continue
        tile_T = [reachable[k][1] for k in idxs]
        want = min(int(aim.n_views), len(tile_T))
        sel = select_diverse(tile_T, want, seed_fwd=seed_T[:3, 2])
        for s in sel:
            k = idxs[s]
            chosen.append((t, reachable[k][1], reach_joints[k]))

    present_tiles = {t for t, _, _ in chosen}
    empty_lin = sorted(set(range(len(plan.aims))) - present_tiles)
    empty_tiles = len(empty_lin)
    if not chosen:
        raise RuntimeError(
            "no reachable, collision-free poses survived tiling across the surveyed "
            "rectangle — jog to a more open part of the workspace and retry")

    _, eff_max, eff_mean = viewing_angle_span([T for _, T, _ in chosen], seed_T[:3, 2])

    # Coverage prediction across ALL tiles (§10 hard gate, ambiguity
    # resolution #5, refined post-review — Findings 2/3): the brief's own
    # prescription was to feed the survey's whole densified rectangle (via
    # _densify_quad) into the EXISTING projected_corner_coverage call, the
    # same one the single-aim path uses. Measured this is NOT a valid metric
    # here, on BOTH sides of the failure: for a SINGLE, correctly-covered
    # tile (a 120x90 mm rectangle inside one ~286x214 mm tile footprint) it
    # reports 0.5 regardless of pose diversity — a false FAIL, because the
    # whole-rectangle footprint only ever lands in the centre columns of that
    # one pose's own 4x3 image-frame grid. For the MULTI-tile case it does
    # the opposite: on the brief's own 2000x1200 mm / 64-tile example it
    # reports EXACTLY 1.000 whether or not a 3x3 block of 9 tiles is missing
    # (checked at densification n=6, 12 and 40) — a false PASS, because each
    # SURVIVING tile's own pose already fills nearly all of the shared 4x3
    # grid on its own; the grid is indexed by position WITHIN the frame, not
    # in the world, so it cannot tell "world region A was seen" from "world
    # region B was seen instead."
    #
    # What DOES detect a missing tile is tile_coverage_frac: the fraction of
    # tiles that produced >= 1 reachable/collision-free pose. But the
    # fraction alone still has a blind spot (Finding 2): it treats one large
    # CONTIGUOUS hole (a single, real, unscanned patch — e.g. the reported
    # 3x3/9-tile block, which passes a 0.85 fraction gate at 0.859) exactly
    # like the same count of misses SCATTERED across the grid (which are
    # mostly absorbed by neighbouring tiles' own overlap margin). So this
    # also gates on the largest 4-connected contiguous block of empty tiles
    # (see _largest_contiguous_empty_block + survey_tour_max_contiguous_
    # empty_tiles) — a scattered handful of edge-of-workspace misses passes,
    # a single sizeable hole does not, regardless of what the fraction says.
    nx, ny, _foot_w, _foot_h = _tile_grid_dims(rec.corners_np(), K, (W, H), scfg)
    empty_ij = {divmod(t, ny) for t in empty_lin}
    largest_block = _largest_contiguous_empty_block(empty_ij)
    tile_coverage_frac = 1.0 - empty_tiles / len(plan.aims)
    surface_coverage = tile_coverage_frac

    max_contig = int(scfg.survey_tour_max_contiguous_empty_tiles)
    problems: list[str] = []
    if surface_coverage < float(scfg.min_surface_coverage):
        problems.append(
            f"only {surface_coverage:.0%} of the surveyed rectangle is covered "
            f"(< {scfg.min_surface_coverage:.0%} min_surface_coverage)")
    if largest_block > max_contig:
        problems.append(
            f"a contiguous block of {largest_block} adjacent empty tile(s) forms a "
            f"single unscanned hole (> {max_contig} allowed)")
    if problems:
        empty_names = ", ".join(f"T{t + 1:02d}" for t in empty_lin[:20])
        if len(empty_lin) > 20:
            empty_names += f", … ({len(empty_lin) - 20} more)"
        message = (
            "; ".join(problems) + f" — {empty_tiles} of {len(plan.aims)} tile(s) have "
            f"zero reachable/collision-free poses: {empty_names}. Part of the surface "
            "would not be captured. Re-run the five-position survey from a more open "
            "pose, or reposition and retry.")
        if getattr(scfg, "surface_coverage_hard_fail", False):
            raise RuntimeError(message)
        services.bus.publish(JobEvent("log", {"message": f"WARNING: {message}"}))

    created: list[str] = []
    n_backfilled = 0
    per_tile_counts: dict[int, int] = {}
    for t, T, joints in chosen:
        if joints is None:
            joints = rdk.solve_joints_for_pose(T, seed_joints)
            if joints is not None:
                n_backfilled += 1
        k = per_tile_counts.get(t, 0) + 1
        per_tile_counts[t] = k
        name = f"{prefix}T{t + 1:02d}_{k}"
        rdk.add_target(name, T, joints=joints)
        created.append(name)

    collide_note = (f"; collision filter bypassed after {n_collide} reported collision(s)"
                    if collision_filter_bypassed else
                    (f"; {n_collide} dropped for collision" if col_checked and n_collide
                     else ("; collision-checked" if col_checked
                           else "; collisions NOT checked")))
    cover_note = f"; predicted coverage {surface_coverage:.0%}"
    services.bus.publish(JobEvent("log", {"message":
        f"created {len(created)} scan targets across {len(plan.aims) - empty_tiles}/"
        f"{len(plan.aims)} tile(s) of the surveyed rectangle (standoff "
        f"~{plan.standoff_mm:.0f} mm; {n_reach}/{len(tile_candidates)} candidates "
        f"reachable; effective cone ~{eff_max:.0f}°{collide_note}{cover_note}) — "
        f"inspect them in RoboDK"}))

    return {"mode": "quality", "created": len(created), "targets": created,
            "look_distance_mm": plan.standoff_mm,
            "gate": locked.gate_payload, "candidates_reachable": n_reach,
            "candidates_total": len(tile_candidates), "collisions_checked": col_checked,
            "candidates_collided": n_collide, "effective_cone_deg": round(eff_max, 1),
            "collision_pairs": pair_examples,
            "surface_coverage": round(surface_coverage, 3),
            "empty_tile_count": empty_tiles,
            "largest_contiguous_empty_tiles": largest_block,
            "planned_cone_deg": float(scfg.flat_cone_deg),
            # PLANNED figure (mirrors the single-aim path's target_count
            # semantics — the intended capture count, not how many actually
            # got created after reachability/collision/selection): the sum of
            # each tile's own n_views, i.e. what a fully-reachable, fully-
            # collision-free tour would have produced.
            "planned_views": int(sum(a.n_views for a in plan.aims)),
            "boundary_views_enabled": False, "boundary_aim_offsets": 0,
            "camera_tool_offset_mm": round(tool_offset_mm, 1),
            "calibration_on_file": tool_offset_mm >= 15.0,
            "collision_filter_enabled": scfg.collision_filter,
            "collision_filter_bypassed": collision_filter_bypassed,
            "extent_mm": [float(rec.size_mm[0]), float(rec.size_mm[1])],
            "crop_size_mm": None,
            "voxel_size_m": plan.voxel_size_m,
            "plan": plan.to_dict(),
            "tile_count": len(plan.aims),
            "tiles_with_targets": len(plan.aims) - empty_tiles,
            "boundary_provenance": rec.boundary_provenance,
            "survey": rec.to_dict(),
            "lock_token": locked.lock_token}


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

    # Task 12, ambiguity resolution #4: a five-position-surveyed surface (too
    # large for one camera view) is a completely different geometry problem —
    # tile it with a close-range tour instead of the single-aim orbit below.
    # Taken as the FIRST branch so every other path (compact/user-specified/
    # crop/reference/quality) is completely untouched.
    if locked.survey_record is not None and locked.survey_record.mode == MODE_FIVE_POSITION:
        return _generate_tiled_scan_targets(services, locked, seed_T, seed_joints,
                                            tool_offset_mm, pub)

    look = float(reading.distance_mm or scfg.look_distance_mm)
    target_center = None
    target_count = scfg.pose_count
    target_cone_deg = scfg.cone_half_angle_deg
    target_normal = None
    min_perpendicular_mm = None
    boundary_aim_offsets = None
    boundary_views_enabled = False
    plan = None
    planned_voxel_m = scfg.voxel_size_m
    extent_mm = (list(survey.extent_mm)
                 if survey.detected and survey.extent_mm is not None else None)
    crop_size_mm = None
    force_crop_plan = gate_payload.get("surface_mode") == "crop"
    if survey.detected and survey.extent_mm is not None:
        if force_crop_plan:
            # Operator chose / auto detected large-platform crop. Do not chase the
            # measured finite rectangle; use the current reticle plane as the work
            # region and keep the seed-pose cone.
            crop_size_mm = gate_payload.get("crop_size_mm") or _large_surface_crop_mm(
                scfg, K, (W, H), look)
            pub("survey: using reticle crop mode; ignoring unstable/full rectangle "
                f"edges and creating a {crop_size_mm[0]:.0f}×{crop_size_mm[1]:.0f} mm "
                "camera-centred work crop")
        elif survey.fully_framed:
            plan = plan_scan(survey, K, (W, H), scfg, cam_to_base_T=seed_T)
            planned_voxel_m = plan.voxel_size_m
            if plan.mode == "reference":
                # The surface frames cleanly but is too large/far to capture within
                # the camera's accurate depth band in a quality tour. Place a single-
                # frame reference rectangle directly (no robot motion, no fusion) and
                # return it for immediate review/insert — the frontend renders this
                # mode without a tour. Previously this branch fell through and created
                # a quality tour anyway (orbiting a surface it could not accurately
                # fuse); _reference_locate was dead code.
                for w in plan.warnings:
                    pub(f"WARNING (survey): {w}")
                prior_scan = rdk.list_targets(prefix)
                if prior_scan:
                    rdk.delete_items(prior_scan)
                result = _reference_locate(services, frame, survey, seed_T, plan)
                return {"mode": "reference", "created": 0, "targets": [],
                        "look_distance_mm": float(plan.standoff_mm),
                        "extent_mm": extent_mm, "voxel_size_m": plan.voxel_size_m,
                        "crop_size_mm": None,
                        "camera_tool_offset_mm": round(tool_offset_mm, 1),
                        "calibration_on_file": tool_offset_mm >= 15.0,
                        "gate": gate_payload, "_scan_result": result}
            if plan.mode == "quality" and plan.aims:
                look = float(plan.standoff_mm)
                target_center = np.asarray(plan.aims[0].point_base_mm, float)
                target_normal = -np.asarray(plan.aims[0].view_dir_base, float)
                min_perpendicular_mm = float(plan.aims[0].min_perpendicular_mm)
                # Preserve an operator-configured denser tour (12 by default), while
                # still allowing a raised-surface plan to request more viewpoints.
                target_count = max(int(scfg.pose_count), int(plan.aims[0].n_views))
                target_cone_deg = float(plan.cone_half_angle_deg)
                boundary_views = max(0, int(getattr(scfg, "boundary_views", 0)))
                if boundary_views:
                    boundary_aim_offsets = frame_aim_offsets(
                        K, (W, H),
                        edge_fraction=float(getattr(scfg, "boundary_aim_edge_fraction", 0.30)))
                    target_count += boundary_views
                    target_cone_deg = min(
                        float(getattr(scfg, "raised_cone_deg", target_cone_deg)),
                        target_cone_deg + float(getattr(scfg, "boundary_cone_extra_deg", 0.0)))
                    boundary_views_enabled = True
            pub(f"survey: {survey.extent_mm[0]:.0f}×{survey.extent_mm[1]:.0f} mm surface "
                f"at {survey.standoff_mm:.0f} mm; planned scan targets "
                f"(standoff {look:.0f} mm, cone {target_cone_deg:.0f}°, "
                f"views {target_count}"
                + (f", +{scfg.boundary_views} boundary" if boundary_views_enabled else "")
                + "), "
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

    _clear_prior_scan_targets(rdk, prefix, pub)

    candidates = generate_calibration_poses(
        seed_T, count=target_count, look_distance_mm=look,
        cone_half_angle_deg=target_cone_deg,
        roll_max_deg=scfg.roll_max_deg, distance_jitter=scfg.distance_jitter,
        target_center=target_center, target_normal=target_normal,
        min_perpendicular_mm=min_perpendicular_mm,
        aim_offsets=boundary_aim_offsets)
    reachable = [(i, T) for i, T in enumerate(candidates) if rdk.is_reachable(T)]
    n_reach = len(reachable)
    if n_reach < SCAN_MIN_VIEWS:
        raise RuntimeError(
            f"only {n_reach} reachable poses around this view (need >= {SCAN_MIN_VIEWS}) "
            f"— jog to a more open part of the workspace (still framing the table) and retry")

    (reachable, reach_joints, col_checked, n_collide, collision_filter_bypassed,
     pair_examples) = _screen_scan_collision_candidates(services, scfg, reachable, n_reach)

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
    boundary_note = ("; boundary-biased views enabled"
                     if boundary_views_enabled else "")
    cover_note += boundary_note
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
            "boundary_views_enabled": boundary_views_enabled,
            "boundary_aim_offsets": (len(boundary_aim_offsets)
                                     if boundary_aim_offsets is not None else 0),
            "camera_tool_offset_mm": round(tool_offset_mm, 1),
            "calibration_on_file": tool_offset_mm >= 15.0,
            "collision_filter_enabled": scfg.collision_filter,
            "collision_filter_bypassed": collision_filter_bypassed,
            "extent_mm": extent_mm,
            "crop_size_mm": crop_size_mm,
            "voxel_size_m": planned_voxel_m,
            "plan": plan.to_dict() if plan is not None else None,
            # §11 provenance (Task 5): absent (None) whenever the lock built no
            # survey_record (e.g. a silent auto-crop overrun) — never fabricated.
            "boundary_provenance": (locked.survey_record.boundary_provenance
                                    if locked and locked.survey_record else None),
            "survey": (locked.survey_record.to_dict()
                       if locked and locked.survey_record else None),
            "lock_token": locked.lock_token if locked else ""}


# -- capture + reconstruct job ----------------------------------------------
@dataclass
class ScanParams:
    save_artifacts: bool = True
    voxel_size_m: float | None = None   # None → use ScanConfig default
    crop_size_mm: tuple[float, float] | None = None
    surface_size_mm: tuple[float, float] | None = None
    # §11 provenance threaded from the locked surface (Task 3/5): who/what decided
    # the boundary, and the LockedWorkframeSurvey.to_dict() it came from. Both are
    # None when the lock never built a survey record (e.g. an auto-crop overrun) —
    # that is an honest absence, never a fabricated/defaulted string (spec §1/§12).
    boundary_provenance: str | None = None
    survey: dict | None = None


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
                   mesh_kind: str = "fitted_flat_surface",
                   provenance: str | None = None,
                   survey: dict | None = None) -> dict:
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
        # §11 provenance (Task 5): carried through from ScanParams, which came from
        # generate_scan_targets's locked-surface survey record. None when the lock
        # built no record — an honest absence, never a fabricated string.
        "boundary_provenance": provenance,
        "survey": survey,
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
                                     mesh_kind="fitted_flat_surface",
                                     provenance=self.params.boundary_provenance,
                                     survey=self.params.survey)
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

    # Downstream modules place work *in* this frame, so publish the rectangle in
    # frame coordinates too. The frame origin is a corner, so its centre has to be
    # derived from the corners; the (X, Y) extents alone cannot give the sign.
    corners_frame_mm = rectangle_in_frame(frame_T_mm, corners_mm)
    center_frame_mm = corners_frame_mm.mean(axis=0)
    # §11 provenance (Task 5): read from the SAME resolved report as the geometry
    # above — never a different source — so provenance can never disagree with what
    # was actually inserted. Absent (None) whenever the run carried no survey
    # record; never fabricated or defaulted to a measured-sounding string.
    boundary_provenance = report.get("boundary_provenance")
    survey_quality = (report.get("survey") or {}).get("quality")
    payload = {
        "module": "scan", "run_id": stamp_id, "source": source,
        "applied_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "frame": FRAME_NAME, "rectangle": RECT_NAME,
        "mesh": MESH_NAME if mesh_inserted else None,
        "size_mm": report.get("plane", {}).get("size_mm"),
        "rectangle_corners_frame_mm": corners_frame_mm.tolist(),
        "rectangle_center_frame_mm": [float(center_frame_mm[0]),
                                      float(center_frame_mm[1])],
        "boundary_provenance": boundary_provenance,
        "survey_quality": survey_quality,
    }
    runs.write_active("scan", payload)
    return {"status": "inserted", "frame": FRAME_NAME, "rectangle": RECT_NAME,
            "mesh": MESH_NAME if mesh_inserted else None, "run_id": stamp_id,
            "source": source, "active": payload,
            "frame_valid": bool(getattr(frame, "Valid", lambda: True)()),
            "rectangle_valid": bool(getattr(rect, "Valid", lambda: True)())}
