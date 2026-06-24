"""The calibration *job* + the gate-gated target generation — the orchestration
that ties the core services to the pure library. Real-robot, RealSense-only.

Flow (no taught NEUTRAL):

  1. Live aiming gate (LivePreview): the operator jogs the robot until the board
     sits at the ideal distance + angle and all HUD lamps go green.
  2. ``generate_calibration_targets`` (gated): one authoritative grab confirms the
     gate, then reachable viewpoints are generated around the robot's *current*
     pose and left in the station as ``TasniCalib_*`` to inspect in RoboDK.
  3. ``CalibrationJob`` visits those targets on the real robot, detects ChArUco,
     records the true flange pose, solves TSAI (+refine) and reports quality.

The job never auto-applies (the user applies after reviewing the metrics) and
returns the robot to its starting joints when it finishes.
"""
from __future__ import annotations

import json
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path

import base64

import cv2
import numpy as np

from ...core import runs
from ...core.aiming import GateThresholds, evaluate_gate
from ...core.events import JobEvent
from ...core.geometry import Rt_to_T, transform_points
from ...core.jobrunner import JobContext
from ...core.logging import get_logger, new_run_dir
from ...core.rdk_io import RdkIO
from .charuco import CharucoTarget
from .handeye import (CalibrationView, cross_validate, estimate_board_in_base,
                      refine, reject_outliers, solve_best, solve_handeye)
from .intrinsics import verify_intrinsics
from .poses import (board_visible_fraction, generate_calibration_poses,
                    select_diverse, viewing_angle_span)
from .quality import evaluate

log = get_logger("tasni.calibration")

TARGET_PREFIX = "TasniCalib_"
# A collision OBJECT spanning the board footprint + margin, down toward the floor —
# a conservative stand-in for the physical platform the board sits on (which is often
# bigger than, or absent from, the station CAD). Created at generation, left in the
# station to inspect, removed by Clear targets.
BOARD_KEEPOUT_NAME = "TasniBoardKeepout"

# Hand-eye needs a minimum well-conditioned training set; below this the linear
# solve is under-constrained regardless of how good each view's reprojection is.
MIN_TRAIN_VIEWS = 6


def _camera_hold(services, owner: str):
    """Scoped camera lease for a capture grab. Non-blocking: if the camera is held
    elsewhere (e.g. the live preview wasn't stopped), this raises ``CameraBusy``
    with the holder's label rather than racing the unicast socket. Degrades to a
    no-op when the container has no lease (older fakes in tests)."""
    lease = getattr(services, "camera_lease", None)
    return lease.hold(owner) if lease is not None else nullcontext()


def dry_tour_required(services) -> bool:
    """True while a recreated camera tool still needs a passing dry tour before the
    real run is allowed (see :func:`ensure_camera_tool`)."""
    return bool(getattr(services, "calib_dry_tour_required", False))


def _set_dry_tour_required(services, value: bool) -> None:
    setattr(services, "calib_dry_tour_required", value)


def _last_good_camera_pose(services):
    """``(X_cam2gripper, run_id)`` from the currently-applied calibration on disk,
    or ``None`` if nothing has been applied yet. This is the *real* mounting offset
    of the camera tool — the only safe pose to recreate a deleted tool at."""
    active = runs.read_active("calibration")
    if not active:
        return None
    run_id = active.get("run_id")
    if not run_id:
        return None
    try:
        report = runs.load_report("calibration", run_id)
        X = np.asarray(report["X_cam2gripper"], dtype=float)
    except (runs.RunNotFound, KeyError, ValueError, TypeError):
        return None
    if X.shape != (4, 4) or not np.all(np.isfinite(X)):
        return None
    return X, run_id


def ensure_camera_tool(services, *, log=None) -> dict:
    """Make sure the camera tool exists; self-heal a deleted-and-saved tool from
    the last applied calibration.

    Why not just recreate it at identity: the generated ``TasniCalib_*`` targets are
    *camera* (TCP) poses, and the robot drives the active TCP to them. A tool with an
    identity mount would put the **flange** where the camera should go — a whole
    camera-stick-out shift in arm configuration, i.e. a collision risk. So we only
    recreate it at the *real* offset recovered from the last calibration; with no
    calibration history there is no safe offset to invent, so we refuse and ask the
    operator to re-add the camera (with its 3D model).

    Returns ``{"present": True, "restored": run_id|None}``. Raises ``RuntimeError``
    only when the tool is missing AND unrecoverable.
    """
    rdk: RdkIO = services.rdk
    tool_name = services.config.robodk.camera_tool
    if rdk.item_exists(tool_name):
        return {"present": True, "restored": None}

    recovered = _last_good_camera_pose(services)
    if recovered is None:
        raise RuntimeError(
            f"tool {tool_name!r} not found and no prior calibration to restore its "
            f"mounting offset from — re-add the RealSense camera (its 3D model + a "
            f"tool named {tool_name!r}) on the flange in RoboDK. Auto-recreating it "
            f"at a guessed offset is unsafe: the robot would drive the flange, not "
            f"the camera, to each target and could collide.")
    X, run_id = recovered
    rdk.create_tool(tool_name, X)
    # Latch the dry-tour gate: the run refuses until a dry tour passes (the
    # recreated tool has no 3D model, so collisions are unchecked — verify motion
    # in simulation before driving the real arm).
    _set_dry_tour_required(services, True)
    msg = (f"{tool_name!r} tool was missing — recreated it from the last applied "
           f"calibration ({run_id}). VERIFY its position in RoboDK and run the dry "
           f"tour before moving the real robot. Note: the recreated tool has no 3D "
           f"model, so the collision check cannot see the camera body.")
    if log:
        log(msg)
    return {"present": True, "restored": run_id}


def gate_thresholds(ccfg) -> GateThresholds:
    """Build the gate thresholds from a CalibrationConfig (one source of truth so
    the live preview and the authoritative generate-grab gate identically)."""
    return GateThresholds(min_corners=ccfg.min_charuco_corners,
                          ideal_distance_mm=ccfg.ideal_distance_mm,
                          distance_tol_mm=ccfg.distance_tol_mm,
                          max_tilt_deg=ccfg.max_tilt_deg,
                          center_tol_mm=ccfg.center_tol_mm,
                          invert_x=ccfg.jog_invert_x,
                          invert_y=ccfg.jog_invert_y,
                          invert_z=ccfg.jog_invert_z)


def generate_calibration_targets(services) -> dict:
    """Gate-gated target creation (synchronous, no robot motion).

    Stops the live preview, takes one authoritative frame, and refuses unless the
    board is detected at the ideal distance + angle. On success the robot's
    current pose is the seed: reachable cone+roll poses are generated and written
    into the station as ``TasniCalib_*`` (prior ones cleared first). Returns the
    count created and the gate reading. Raises ``RuntimeError`` if not ready.
    """
    cfg = services.config
    ccfg = cfg.calibration
    rdk: RdkIO = services.rdk
    cam = services.camera
    tool_name = cfg.robodk.camera_tool
    K, dist = cfg.camera.K, cfg.camera.dist

    ensure_camera_tool(
        services,
        log=lambda m: services.bus.publish(JobEvent("log", {"message": m})))

    # Free the camera (unicast) so our authoritative grab gets the frame.
    if services.live.running:
        services.live.stop()

    tool_pose = rdk.use_camera_tool(tool_name)
    # The seed + all generated poses are CAMERA (active-TCP) poses. If the Realsense
    # tool has no real mounting offset (PoseTool ≈ identity), its TCP sits AT the
    # flange, so the orbit centres on the flange axis, not where the camera looks —
    # which looks and behaves like "targets for the flange, not the camera". Warn so
    # the operator sets the approximate camera mount (calibration then refines it).
    tool_offset_mm = float(np.linalg.norm(np.asarray(tool_pose)[:3, 3]))
    if tool_offset_mm < 15.0:
        services.bus.publish(JobEvent("log", {"message":
            f"WARNING: the {tool_name!r} tool is only ~{tool_offset_mm:.0f} mm from the "
            f"flange (≈ no offset) — poses will be generated for the FLANGE, not the "
            f"camera. Set the camera's approximate mounting pose on the {tool_name!r} "
            f"tool in RoboDK (calibration refines it), or apply a prior calibration "
            f"first, then re-create targets."}))
    board = CharucoTarget(cfg.board)
    with _camera_hold(services, "target-creation"):
        frame = cam.grab(color_only=True)
    det = board.detect(frame.color, K, dist, min_corners=ccfg.min_charuco_corners)
    reading = evaluate_gate(det, K, frame.color.shape, gate_thresholds(ccfg),
                            board_center_mm=board.board_center)

    # Show the operator exactly the frame we judged.
    img = (board.annotate(frame.color, det, K, dist, "TARGET CREATION")
           if det is not None else frame.color)
    ok, jpeg = cv2.imencode(".jpg", img)
    if ok:
        services.bus.publish(JobEvent("frame",
            {"jpeg_b64": base64.b64encode(jpeg.tobytes()).decode("ascii")}))
    services.bus.publish(JobEvent("gate", {**reading.to_dict(), "live": False}))

    if not reading.ok:
        bad = [name for name, good in reading.gates.items() if not good]
        raise RuntimeError(
            "board not in the ideal band — fix " + ", ".join(bad)
            + f" (distance {reading.distance_mm and round(reading.distance_mm)} mm, "
            + f"tilt {reading.tilt_deg and round(reading.tilt_deg, 1)}°). "
            + "Jog the robot until all HUD lamps are green, then create targets.")

    # The seed is the CAMERA pose (computed explicitly from the flange + the Realsense
    # tool offset), not whatever the active TCP happens to be — so the generated poses
    # orbit the camera's view, and the IK that locks each target drives the camera to
    # it. tcp_pose_T()/Pose() would silently be the flange if the tool isn't active.
    seed_T = rdk.camera_pose_T()
    try:
        # The gate/seed config — anchors the IK branch when locking targets to joints.
        seed_joints = rdk.current_joints()
    except Exception:
        seed_joints = None
    look = float(reading.distance_mm)

    prior = rdk.list_targets(TARGET_PREFIX)
    if prior:
        rdk.delete_items(prior)

    # The board's pose in the base frame, from the seed detection — used for the
    # visibility filter (below) and the keep-out box. seed_T is the camera pose in
    # base; det.* is the board in the camera; so seed_T @ T_cam_board = T_base_board.
    T_base_board = seed_T @ Rt_to_T(det.R_target2cam, det.t_target2cam)
    board_pts_base = transform_points(T_base_board, board.all_obj_points)

    # Add a conservative platform stand-in (board footprint + margin, down toward the
    # floor) as a collision object, so a pose that grazes the real platform — bigger
    # than the station CAD — is caught by the baseline-relative screen below. Left in
    # the station to inspect; Clear targets removes it. Best-effort.
    keepout_added = False
    if ccfg.board_keepout:
        try:
            box = rdk.add_keepout_box(
                BOARD_KEEPOUT_NAME, board_pts_base,
                margin_mm=ccfg.board_keepout_margin_mm,
                above_mm=ccfg.board_keepout_above_mm,
                depth_mm=ccfg.board_keepout_depth_mm)
            keepout_added = bool(box is not None and getattr(box, "Valid", lambda: True)())
            if keepout_added:
                services.bus.publish(JobEvent("log", {"message":
                    f"board keep-out: added {BOARD_KEEPOUT_NAME!r} (board footprint + "
                    f"{ccfg.board_keepout_margin_mm:.0f} mm, {ccfg.board_keepout_depth_mm:.0f} mm "
                    f"deep) as a platform stand-in — poses grazing the platform are dropped"}))
        except Exception as e:   # noqa: BLE001 - the keep-out is a bonus, never abort
            services.bus.publish(JobEvent("log", {"message":
                f"board keep-out: could not add the platform stand-in "
                f"({type(e).__name__}: {e}); continuing without it"}))

    candidates = generate_calibration_poses(
        seed_T, count=ccfg.pose_count, look_distance_mm=look,
        cone_half_angle_deg=ccfg.cone_half_angle_deg,
        roll_max_deg=ccfg.roll_max_deg, distance_jitter=ccfg.distance_jitter)
    # IK-filter ALL candidates first, then choose a diverse spread — NOT the first
    # N reachable (which clusters at the narrow-cone seed and starves hand-eye
    # conditioning; see select_diverse / the workspace-edge finding in robot_probe).
    reachable = [(i, T) for i, T in enumerate(candidates) if rdk.is_reachable(T)]
    n_reach = len(reachable)
    if n_reach < MIN_TRAIN_VIEWS:
        raise RuntimeError(
            f"only {n_reach} reachable poses around this view (need "
            f">= {MIN_TRAIN_VIEWS}) — jog to a more open part of the workspace "
            f"(still framing the board) and retry")

    # Drop poses where the robot collides with its mounted tooling (e.g. a spindle)
    # or the cell — evaluated in SIMULATE before any target is written, so a
    # would-collide pose never becomes a TasniCalib_* target. The sweep also returns
    # the exact joint configuration it checked: we store those as JOINT targets so
    # the pose that was collision-checked is the one actually visited (a cartesian
    # target can otherwise be reached in a different, colliding IK branch — the
    # cause of a "filtered but still colliding" target). Degrades to a no-op where
    # the station has no collision map (col_checked False -> nothing dropped).
    #
    # First close RoboDK's default blind spot: it excludes a tool from colliding
    # with its own robot, so a flange-mounted spindle/camera hitting an arm link
    # is never reported. This call discovers the flange bodies and reports them; the
    # pairs are (re)enabled INSIDE screen_collisions (via guard_skip) *after* it turns
    # checking on, because setCollisionActive(ON) rebuilds the default map and would
    # wipe pairs enabled here — the reason the spindle-into-A4 target-12 slipped through.
    guard = None
    guard_skip = None
    if ccfg.collision_filter and ccfg.collision_self_pairs:
        guard_skip = ccfg.collision_skip_wrist_links
        guard = rdk.ensure_mounted_tool_collision_pairs(ccfg.collision_skip_wrist_links)
        n_pairs = (guard or {}).get("pairs_enabled", 0)
        if n_pairs:
            services.bus.publish(JobEvent("log", {"message":
                f"collision guard: enabled {n_pairs} tool↔arm pair(s) for "
                f"{', '.join(guard['tools']) or 'mounted tools'} vs links "
                f"{guard['links']} (RoboDK omits these by default)"}))
        else:
            services.bus.publish(JobEvent("log", {"message":
                "WARNING: collision guard enabled 0 tool↔arm pairs — no flange "
                "tool/object was found, so a spindle-vs-arm collision can't be "
                "filtered. Confirm the camera/spindle are mounted on the robot in "
                "RoboDK."}))

    n_collide = 0
    col_checked = False
    collision_filter_bypassed = False   # retained for the API/UI shape; never set now
    reach_joints: list = [None] * n_reach
    if ccfg.collision_filter:
        # Baseline-relative screen: drops only poses that introduce a NEW collision
        # beyond the constant ones present at the safe seed (robot-base↔pedestal, each
        # tool↔its wrist, a parked axis↔wall). Obstacle pairs enabled so a tool
        # entering the board pedestal is seen; the path is swept so a mid-move bump is
        # too. A genuinely-colliding pose is ALWAYS dropped — never shipped.
        mask, col_checked, jts = rdk.screen_collisions(
            [T for _, T in reachable], guard_skip=guard_skip,
            obstacle_pairs=ccfg.collision_obstacle_pairs,
            baseline_relative=ccfg.collision_baseline_relative,
            path_samples=ccfg.collision_path_samples)
        kept = [k for k in range(n_reach) if mask[k]]
        if col_checked:
            n_collide = n_reach - len(kept)
        reachable = [reachable[k] for k in kept]
        reach_joints = [jts[k] for k in kept]       # locked configs (may be None)
        services.bus.publish(JobEvent("log", {"message":
            f"collision screen: checking {'ACTIVE' if col_checked else 'unavailable'}; "
            f"swept {n_reach} reachable pose(s), {n_collide} introduced a new "
            f"collision and were dropped"}))
        if not col_checked:
            # The station/build can't evaluate collisions (no collision map). Nothing
            # was dropped — proceed with reachable poses (inspect + dry-run), unless a
            # hard gate is configured.
            if ccfg.collision_filter_hard_fail:
                raise RuntimeError(
                    "collision checking is unavailable on this station (no collision "
                    "map) and collision_filter_hard_fail is set — set up Tools → "
                    "Collision Map in RoboDK and save, or clear the hard-fail flag.")
            services.bus.publish(JobEvent("log", {"message":
                "WARNING: collisions could NOT be checked (no station collision map) — "
                "creating reachable poses only. Inspect them in RoboDK and run the dry "
                "tour before moving the real robot."}))
        elif len(reachable) < MIN_TRAIN_VIEWS:
            # Enough poses genuinely collide that too few clean ones remain. Refuse —
            # never ship a colliding target (the operator chose refuse-with-guidance).
            raise RuntimeError(
                f"only {len(reachable)} collision-free pose(s) around this view — "
                f"{n_collide} of {n_reach} reachable poses introduce a NEW collision "
                f"(the mounted tooling swinging into the arm, or a tool entering the "
                f"board pedestal). Re-seed at a more open part of the workspace (still "
                f"framing the board), or clear the obstruction, and Create targets again.")

    # Visibility pre-filter: a pose can be reachable AND collision-free yet aim so
    # the board clips the frame edge (or leaves view), which today only surfaces as
    # a skipped capture *after* the robot has driven there. Project the board (its
    # corners placed in the base frame from the seed detection) into each surviving
    # candidate's image and drop poses that don't keep enough of the board in frame —
    # so the pre-run guarantee becomes reachable + collision-free + board-in-frame.
    # Cheap pure-numpy pinhole; degrades to a no-op if it would starve the solve.
    n_offframe = 0
    vis_checked = False
    if ccfg.visibility_filter and reachable:
        img_size = cfg.camera.size      # board_pts_base computed above (from the seed)
        vis = [board_visible_fraction(T, board_pts_base, K, img_size,
                                      margin_frac=ccfg.board_visible_margin_frac)
               for _, T in reachable]
        keep = [k for k, f in enumerate(vis) if f >= ccfg.min_board_visible_frac]
        vis_checked = True
        if len(keep) >= MIN_TRAIN_VIEWS:
            n_offframe = len(reachable) - len(keep)
            reachable = [reachable[k] for k in keep]
            reach_joints = [reach_joints[k] for k in keep]
            services.bus.publish(JobEvent("log", {"message":
                f"visibility screen: {n_offframe} pose(s) would clip the board out of "
                f"frame and were dropped; {len(reachable)} keep the board in view"}))
        else:
            # A framing gate must never starve the solve: keep all reachable poses
            # and warn — the capture step still skips any that truly clip.
            services.bus.publish(JobEvent("log", {"message":
                f"WARNING: only {len(keep)} reachable pose(s) keep the board fully in "
                f"frame (need >= {MIN_TRAIN_VIEWS}) — visibility filter not applied. "
                f"Inspect the targets; some may clip the board and be skipped at capture."}))

    n_usable = len(reachable)
    reach_T = [T for _, T in reachable]
    sel = select_diverse(reach_T, min(ccfg.pose_count, n_usable),
                         seed_fwd=seed_T[:3, 2])
    chosen = [(reachable[k][0], reachable[k][1], reach_joints[k])
              for k in sel]                         # index-sorted -> spiral naming

    # Lock every target to a joint configuration solved with the *camera* tool
    # active, so selecting/visiting it reproduces the camera (TCP) at the viewpoint
    # regardless of which tool the RoboDK GUI has active. A bare cartesian target
    # stores only a TCP pose, which RoboDK drives the *currently active* tool to —
    # with the flange selected that puts the FLANGE where the camera should be (the
    # "flange visits the TCP" the operator reported). screen_collisions already
    # locks the collision-checked config; here we back-fill any pose it left
    # unlocked (collision filter off, or no IK branch near the seed).
    n_backfilled = 0
    locked: list = []
    for _, T, joints in chosen:
        if joints is None:
            joints = rdk.solve_joints_for_pose(T, seed_joints)
            if joints is not None:
                n_backfilled += 1
        locked.append((T, joints))
    n_cartesian = sum(1 for _, j in locked if j is None)

    created: list[str] = []
    for T, joints in locked:
        name = f"{TARGET_PREFIX}{len(created) + 1:02d}"
        rdk.add_target(name, T, joints=joints)
        created.append(name)
    services.bus.publish(JobEvent("log", {"message":
        f"targets stored as JOINT targets locked to the camera TCP "
        f"(tool offset {tool_offset_mm:.0f} mm off the flange): "
        f"{len(created) - n_cartesian}/{len(created)} locked"
        + (f", {n_backfilled} back-filled by IK" if n_backfilled else "")
        + (f"; WARNING {n_cartesian} left cartesian (no IK branch) — those will "
           f"follow whatever tool is active in the GUI" if n_cartesian else "")}))

    # Effective cone: how much of the configured cone the kept poses actually span.
    # At an edge-of-workspace seed the wide (diversity-rich) poses are unreachable,
    # so this can be far narrower than cone_half_angle_deg — warn BEFORE capture
    # rather than discovering it from motion_diversity after a full run.
    _, eff_max, eff_mean = viewing_angle_span([T for _, T, _ in chosen], seed_T[:3, 2])
    collide_note = (f"; collision filter bypassed after {n_collide} reported collision(s)"
                    if collision_filter_bypassed else
                    (f"; {n_collide} dropped for collision" if col_checked and n_collide
                     else ("; collision-checked" if col_checked
                           else "; collisions NOT checked (no station collision map)")))
    vis_note = (f"; {n_offframe} dropped off-frame" if vis_checked and n_offframe
                else ("; board-in-frame checked" if vis_checked else ""))
    services.bus.publish(JobEvent("log",
        {"message": f"created {len(created)} calibration targets "
                    f"(working distance ~{look:.0f} mm; {n_reach}/{len(candidates)} "
                    f"candidates reachable{collide_note}{vis_note}; effective cone "
                    f"~{eff_max:.0f}° of {ccfg.cone_half_angle_deg:.0f}°) — inspect "
                    f"them in RoboDK"}))
    if eff_max < 0.5 * ccfg.cone_half_angle_deg:
        services.bus.publish(JobEvent("log",
            {"message": f"WARNING: reachable poses span only ~{eff_max:.0f}° "
                        f"(mean {eff_mean:.0f}°) — narrow rotational diversity, "
                        f"hand-eye may be poorly conditioned. Consider re-seeding "
                        f"at a more central, open view."}))
    if len(created) - ccfg.holdout_count < MIN_TRAIN_VIEWS:
        services.bus.publish(JobEvent("log",
            {"message": f"NOTE: {len(created)} targets minus {ccfg.holdout_count} "
                        f"holdout leaves < {MIN_TRAIN_VIEWS} training views; the "
                        f"holdout will be auto-reduced at solve time to keep "
                        f"{MIN_TRAIN_VIEWS} training poses."}))
    _ = tool_pose  # (kept active on the robot for the upcoming run)
    return {"created": len(created), "targets": created,
            "look_distance_mm": look, "gate": reading.to_dict(),
            "candidates_reachable": n_reach, "candidates_total": len(candidates),
            "collisions_checked": col_checked, "candidates_collided": n_collide,
            "collision_filter_enabled": ccfg.collision_filter,
            "collision_filter_bypassed": collision_filter_bypassed,
            "visibility_checked": vis_checked,
            "poses_offframe_dropped": n_offframe,
            "board_keepout_added": keepout_added,
            "effective_cone_deg": round(eff_max, 1),
            "camera_tool_offset_mm": round(tool_offset_mm, 1),
            "targets_joint_locked": len(created) - n_cartesian,
            "targets_cartesian": n_cartesian,
            "collision_guard": guard}


def _split_views(views: list, holdout: int, strategy: str, seed: int):
    """Train/validation split. 'shuffle' (seeded, unbiased) avoids the bias of
    'tail', where the deterministic-spiral pose order makes the last poses
    systematically the widest-angle views."""
    if not holdout:
        return list(views), []
    if strategy == "tail":
        return views[:-holdout], views[-holdout:]
    idx = list(range(len(views)))
    np.random.default_rng(seed).shuffle(idx)
    val_i = set(idx[:holdout])
    return ([v for i, v in enumerate(views) if i not in val_i],
            [views[i] for i in idx[:holdout]])


def _active_quality(report: dict) -> dict:
    """The handful of metrics the Dashboard shows for the currently-applied run."""
    train = report.get("train") or {}
    val = report.get("validation") or {}
    bc = report.get("board_consistency_mm") or {}
    diag = report.get("diagnosis") or {}
    return {
        "verdict": diag.get("verdict"),
        "train_rms_px": train.get("rms_px"),
        "val_rms_px": val.get("rms_px"),
        "board_consistency_rms_mm": bc.get("rms"),
    }


def apply_calibration(services, *, job: "CalibrationJob | None" = None,
                      run_id: str | None = None) -> dict:
    """Write a solved camera pose into the Realsense tool and record provenance.

    Two sources of the solve:
      * ``run_id`` — load ``report.json`` (the solved ``X_cam2gripper``) + ``meta.json``
        from disk. This survives a server restart, when the in-memory last job is gone.
      * else ``job`` — the in-memory last run (the fast path right after a Run).
    On success ``runs/calibration/active.json`` records which run is now live in the
    cell (run-id, date, tool, key metrics) so the Dashboard can show "cell calibrated".
    Raises ``RuntimeError`` if there is nothing to apply.
    """
    rdk: RdkIO = services.rdk
    if run_id is not None:
        report = runs.load_report("calibration", run_id)
        meta = runs.load_meta("calibration", run_id) or {}
        tool = meta.get("tool_name") or services.config.robodk.camera_tool
        X = np.asarray(report["X_cam2gripper"], dtype=float)
        source, stamp_id = "run_id", run_id
    elif job is not None and job.solved_X is not None:
        report = job.result.report if job.result else {}
        tool, X = job.tool_name, job.solved_X
        source = "memory"
        stamp_id = Path(job.result.run_dir).name if job.result else None
    else:
        raise RuntimeError("no solved calibration to apply")

    rdk.set_tool_pose(tool, X)
    payload = {
        "module": "calibration",
        "run_id": stamp_id,
        "applied_at": time.strftime("%Y-%m-%dT%H:%M:%S"),  # caller-stamped; core stays clock-free
        "tool": tool,
        "source": source,
        "refined": report.get("refined"),
        "method": report.get("method"),
        "quality": _active_quality(report),
    }
    runs.write_active("calibration", payload)
    return {"status": "applied", "tool": tool, "run_id": stamp_id,
            "source": source, "active": payload}


def intrinsics_present(services) -> bool:
    """True once calibrated (non-factory) intrinsics have been applied — a marker
    :func:`apply_intrinsics` writes. Absent ⇒ the cell is still on the factory K /
    zero distortion, so the auto path should calibrate them on the next run."""
    return runs.read_active("intrinsics") is not None


def apply_intrinsics(services, report: dict, *, source: str = "manual") -> dict:
    """Write a solved camera matrix + lens distortion into the camera config.

    Mutates the **live** config (so the very next grab / hand-eye solve uses the new
    intrinsics with no restart) AND persists to ``tasni.config.json`` (so it
    survives one). Only the *active* resolution's K is replaced — the other
    resolutions are preserved — while ``dist_coeffs`` is resolution-independent
    (OpenCV distortion operates in normalised coords) so it applies to all. Records
    an ``intrinsics`` active-marker (so :func:`intrinsics_present` is True and the
    auto path won't redo it); ``source`` is "manual" (dedicated capture) or "auto"
    (derived from a hand-eye run's views).
    """
    from ...core.config import save_overrides

    cam = services.config.camera
    res = cam.resolution
    K = np.asarray(report["K"], dtype=float)
    dist = [float(x) for x in report["dist"]]
    full = {r: (K.tolist() if r == res else [row[:] for row in rows])
            for r, rows in cam.intrinsics.items()}
    full[res] = K.tolist()                 # in case the active res wasn't in the map
    cam.intrinsics = full                   # validate_assignment re-checks the shape
    cam.dist_coeffs = dist
    save_overrides({"camera": {"intrinsics": full, "dist_coeffs": dist}})
    runs.write_active("intrinsics", {
        "module": "intrinsics", "source": source, "resolution": res,
        "applied_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "rms_px": report.get("rms_px"), "n_views": report.get("n_views"),
        "coverage_pct": report.get("coverage_pct"), "fix_k3": report.get("fix_k3"),
    })
    return {"status": "applied", "resolution": res, "K": K.tolist(), "dist": dist,
            "source": source}


@dataclass
class TourPoseResult:
    """One pose's verdict on the simulated dry tour."""
    name: str
    reachable: bool
    collision: bool | None        # resting-pose collision (None = not checked)
    ok: bool
    error: str | None = None
    transit: bool | None = None   # collision while SWEEPING into this pose (None = not checked)
    collision_pairs: list[str] | None = None

    def to_dict(self) -> dict:
        return {"name": self.name, "reachable": self.reachable,
                "collision": self.collision, "ok": self.ok, "error": self.error,
                "transit": self.transit, "collision_pairs": self.collision_pairs}


class SimTourJob:
    """Dry-run the generated ``TasniCalib_*`` targets in RoboDK **simulate** mode.

    Visits each target with the robot in simulation (no hardware motion), recording
    per-pose reachability + (if the build supports it) collision state, then returns
    to the start joints — the same return-to-start guarantee the real run makes. It
    is a **soft gate**: failures warn prominently but never block Run (the operator
    may still proceed once the cell is clear). Runs through the JobRunner, so it can
    never overlap a real calibration run. Restores the prior run mode on the way out
    so a dry tour can't leave the station silently in RUN_ROBOT.
    """

    def __init__(self, services, *, target_prefix: str = TARGET_PREFIX,
                 collision_self_pairs: bool | None = None,
                 collision_skip_wrist_links: int | None = None):
        self.services = services
        self.tool_name: str = services.config.robodk.camera_tool
        # Parameterised so the scan module reuses this exact dry tour for its own
        # TasniScan_* targets + collision knobs; default to calibration's values so
        # the calibration call sites are unchanged.
        cc = services.config.calibration
        self.target_prefix = target_prefix
        self.collision_self_pairs = (cc.collision_self_pairs
                                     if collision_self_pairs is None else collision_self_pairs)
        self.collision_skip_wrist_links = (cc.collision_skip_wrist_links
                                           if collision_skip_wrist_links is None
                                           else collision_skip_wrist_links)

    def __call__(self, ctx: JobContext) -> dict:
        rdk: RdkIO = self.services.rdk

        ensure_camera_tool(self.services, log=ctx.log)

        targets = rdk.list_targets(self.target_prefix)
        if not targets:
            raise RuntimeError(
                f"no {self.target_prefix}* targets to simulate — aim the camera until "
                f"the gate is green and click Create targets first")

        prior_mode = rdk.current_run_mode()
        rdk.apply_run_mode("simulate")
        ctx.log(f"dry run (SIMULATE) — visiting {len(targets)} targets, no hardware motion")
        rdk.use_camera_tool(self.tool_name)
        collisions_on = rdk.set_collision_checking(True)
        # Same blind spot as Create targets: RoboDK won't check a tool against its
        # own robot, so enable the mounted-tool↔arm pairs before the tour or the
        # spindle-into-A4 case stays invisible here too.
        ccfg = self.services.config.calibration
        if collisions_on and self.collision_self_pairs:
            guard = rdk.ensure_mounted_tool_collision_pairs(
                self.collision_skip_wrist_links)
            if guard and guard.get("pairs_enabled"):
                ctx.log(f"collision guard: {guard['pairs_enabled']} tool↔arm pair(s) "
                        f"enabled ({', '.join(guard['tools']) or 'mounted tools'})")
            # Also check the mounted tools against the static obstacles (board
            # pedestal, walls, …) — RoboDK omits tool↔object pairs by default, the
            # reason a tool dipping into the pedestal mid-tour went unseen.
            if ccfg.collision_obstacle_pairs:
                obs = rdk.ensure_obstacle_collision_pairs()
                if obs and obs.get("pairs_enabled"):
                    ctx.log(f"collision guard: {obs['pairs_enabled']} tool↔object "
                            f"pair(s) enabled ({', '.join(obs['objects']) or 'objects'})")
        try:
            start_joints = rdk.current_joints()
        except Exception:
            start_joints = None
        # Baseline collision pair-set at the safe start config: the constant cell
        # artifacts (robot base↔pedestal, tool↔its wrist, parked axis↔wall) to
        # subtract so the tour reports only NEW collisions — a genuine bump.
        baseline = set()
        if collisions_on and ccfg.collision_baseline_relative:
            bk = rdk.collision_pair_keys()
            baseline = bk if bk is not None else set()
        samples = ccfg.collision_path_samples

        results: list[TourPoseResult] = []
        total = len(targets)
        # The real run drives start -> t1 -> t2 -> ... -> tN -> start. Resting-pose
        # checks alone miss a tool that sweeps THROUGH an arm link mid-move yet
        # clears both endpoints, so sweep each segment (the config we came from to
        # the next target's config) with MoveJ_Test, the same swept API the pose
        # filter uses. prev_joints starts at the seed/start config.
        prev_joints = start_joints
        try:
            for i, name in enumerate(targets):
                ctx.check_cancel()
                ctx.progress(i + 1, total, f"checking {name}")
                reachable = rdk.is_reachable(rdk.target_pose_T(name))
                collision: bool | None = None
                transit: bool | None = None
                err: str | None = None
                pairs: list[str] | None = None
                if reachable:
                    dest = rdk.target_joints(name)
                    if collisions_on and prev_joints is not None and dest is not None:
                        if ccfg.collision_baseline_relative:
                            transit = rdk.path_new_collisions(prev_joints, dest,
                                                              baseline, samples)
                            if transit:
                                _, pairs = rdk.new_collisions_here(baseline)
                        else:
                            ncol = rdk.move_j_test(prev_joints, dest)
                            transit = None if ncol is None else bool(ncol)
                            if transit:
                                pairs = getattr(rdk, "collision_pairs", lambda: [])()
                    try:
                        rdk.move_j(name)
                    except Exception as e:   # noqa: BLE001 - a sim move failure is a fail, not a crash
                        reachable, err = False, str(e)
                    if reachable and collisions_on:
                        if ccfg.collision_baseline_relative:
                            collision, newp = rdk.new_collisions_here(baseline)
                            if collision:
                                pairs = newp or pairs
                        else:
                            n_col = rdk.collisions()
                            collision = None if n_col is None else bool(n_col)
                            if collision:
                                pairs = getattr(rdk, "collision_pairs", lambda: [])() or pairs
                    if reachable:
                        try:
                            prev_joints = rdk.current_joints()
                        except Exception:
                            prev_joints = dest if dest is not None else prev_joints
                ok = reachable and not bool(collision) and not bool(transit)
                results.append(TourPoseResult(name, reachable, collision, ok, err,
                                              transit=transit, collision_pairs=pairs))
                flag = ("OK" if ok else "UNREACHABLE" if not reachable
                        else "TRANSIT-COLLISION" if transit else "COLLISION")
                pair_txt = f" ({'; '.join(pairs[:3])})" if pairs else ""
                ctx.log(f"{name}: {flag}{pair_txt}")

            # Return-to-start (the guarantee the real run makes; verify it here too) —
            # and sweep the path back, since that move runs on the real arm as well.
            returned = False
            return_path_ok = True
            if start_joints is not None:
                if collisions_on and prev_joints is not None:
                    if ccfg.collision_baseline_relative:
                        if rdk.path_new_collisions(prev_joints, start_joints,
                                                   baseline, samples):
                            return_path_ok = False
                            ctx.log("return-to-start path introduces a NEW collision")
                    else:
                        ncol = rdk.move_j_test(prev_joints, start_joints)
                        if ncol:
                            return_path_ok = False
                            ctx.log(f"return-to-start path COLLIDES ({ncol} pair(s))")
                try:
                    rdk.move_j_joints(start_joints)
                    returned = True
                except Exception:
                    returned = False

            n_pass = sum(1 for r in results if r.ok)
            n_unreachable = sum(1 for r in results if not r.reachable)
            n_collision = sum(1 for r in results if r.collision)
            n_transit = sum(1 for r in results if r.transit)
            all_ok = n_pass == total and returned and return_path_ok
            # A clean dry tour clears the restored-tool safety latch, re-enabling the
            # real run (only meaningful when a recreated tool armed it).
            if all_ok and dry_tour_required(self.services):
                _set_dry_tour_required(self.services, False)
                ctx.log("restored-tool safety latch cleared (dry tour passed) — "
                        "the real run is enabled again")
            ctx.log(f"dry run complete: {n_pass}/{total} poses OK"
                    + (f"; {n_transit} transit collision(s)" if n_transit else "")
                    + f"; return-to-start {'ok' if returned and return_path_ok else 'FAILED'}"
                    + ("" if collisions_on else "; collisions not checked on this build"))
            return {
                "kind": "sim_tour",
                "total": total,
                "passed": n_pass,
                "unreachable": n_unreachable,
                "collisions": n_collision,
                "transit_collisions": n_transit,
                "collisions_checked": collisions_on,
                "returned_to_start": returned and return_path_ok,
                "all_ok": all_ok,
                "poses": [r.to_dict() for r in results],
            }
        finally:
            rdk.set_collision_checking(False)
            rdk.set_run_mode_raw(prior_mode)
            if start_joints is not None:
                try:
                    rdk.move_j_joints(start_joints)
                except Exception:
                    pass


@dataclass
class CalibrationParams:
    holdout_count: int | None = None    # override config.calibration.holdout_count
    refine: bool | None = None          # override config.calibration.refine
    save_frames: bool = True


@dataclass
class CalibrationResult:
    report: dict
    summary: str
    run_dir: str
    tool_name: str
    n_captured: int
    n_skipped: list[str] = field(default_factory=list)


class CalibrationJob:
    """Callable run by the JobRunner. Visits the pre-generated ``TasniCalib_*``
    targets, solves, and holds the result for the separate apply step (so writing
    the tool pose is an explicit user action)."""

    def __init__(self, services, params: CalibrationParams):
        self.services = services
        self.params = params
        self.solved_X: np.ndarray | None = None   # T_flange_cam (cam2flange)
        self.tool_name: str = services.config.robodk.camera_tool
        self.result: CalibrationResult | None = None

    def __call__(self, ctx: JobContext) -> dict:
        cfg = self.services.config
        rdk: RdkIO = self.services.rdk
        cam = self.services.camera
        ccfg = cfg.calibration
        K, dist = cfg.camera.K, cfg.camera.dist
        board = CharucoTarget(cfg.board)

        ensure_camera_tool(self.services, log=ctx.log)
        if dry_tour_required(self.services):
            raise RuntimeError(
                "the camera tool was recreated from a past calibration and has not "
                "passed a dry tour since — run the dry tour (Simulate) and let it "
                "pass before moving the real robot, so a wrong tool position can't "
                "drive the arm into a collision.")

        targets = rdk.list_targets(TARGET_PREFIX)
        holdout = (self.params.holdout_count if self.params.holdout_count is not None
                   else ccfg.holdout_count)
        if len(targets) < MIN_TRAIN_VIEWS:
            raise RuntimeError(
                f"only {len(targets)} {TARGET_PREFIX}* targets; need >= "
                f"{MIN_TRAIN_VIEWS}. Aim the camera until the gate is green and "
                f"click Create targets first.")

        # The camera (unicast) must be ours for the capture grabs.
        if self.services.live.running:
            self.services.live.stop()

        applied_mode = rdk.apply_run_mode("run_robot")
        ctx.log(f"run mode: {applied_mode} (REAL ROBOT)")
        tool_pose = rdk.use_camera_tool(self.tool_name)
        ctx.log(f"tool: {self.tool_name}; {len(targets)} targets to visit")

        try:
            start_joints = rdk.current_joints()
        except Exception:
            start_joints = None

        stamp = time.strftime("%Y%m%d-%H%M%S")
        run_dir = new_run_dir("calibration", stamp)

        try:
            # Own the camera for the whole capture so the live preview can't sneak
            # a grab in between poses (non-blocking; fails fast if still held).
            with _camera_hold(self.services, "calibration-run"):
                views, skipped = self._capture(ctx, rdk, cam, board, K, dist,
                                               tool_pose, targets, run_dir)
            do_refine = (self.params.refine if self.params.refine is not None
                         else ccfg.refine)
            if len(views) < MIN_TRAIN_VIEWS:
                raise RuntimeError(
                    f"only {len(views)} usable views; need >= {MIN_TRAIN_VIEWS} "
                    f"training poses. Skipped (no board): {skipped}")

            # Auto intrinsic calibration (once, under the hood): if the cell has no
            # calibrated intrinsics yet, derive K + distortion from THESE captured
            # board views — no separate step, no extra motion — apply them, then
            # recompute each view's board pose with the better model so the hand-eye
            # solve below uses it. Gated on the marker (so it runs only when missing).
            # Degrades to a no-op (keeps the configured intrinsics) on any failure.
            report_intrinsics_auto = None
            if ccfg.auto_intrinsics and not intrinsics_present(self.services):
                try:
                    from .intrinsics_calib import solve_intrinsics
                    obj_l = [v.obj_points.astype(np.float32) for v in views]
                    img_l = [v.corners.reshape(-1, 1, 2).astype(np.float32) for v in views]
                    intr = solve_intrinsics(obj_l, img_l, cfg.camera.size, K,
                                            fix_k3=ccfg.intrinsics_fix_k3)
                    apply_intrinsics(self.services, intr, source="auto")
                    K, dist = cfg.camera.K, cfg.camera.dist     # reload the applied model
                    for v in views:                              # re-pose with new K/dist
                        ok, rvec, tvec = cv2.solvePnP(
                            v.obj_points.astype(np.float64),
                            v.corners.reshape(-1, 1, 2).astype(np.float64), K, dist)
                        if ok:
                            v.R_target2cam = cv2.Rodrigues(rvec)[0]
                            v.t_target2cam = tvec.reshape(3)
                    cov = int(round(intr["coverage_pct"] * 100))
                    ctx.log(f"auto intrinsics (none on file): calibrated K+distortion from "
                            f"{intr['n_views']} captured views — fit RMS {intr['rms_px']:.3f} px, "
                            f"coverage {cov}% (k3 {'fixed' if intr['fix_k3'] else 'free'}); applied")
                    if intr["coverage_pct"] < 0.6:
                        ctx.log("NOTE: the board stays centred in hand-eye poses, so intrinsic "
                                "edge coverage is thin — for best distortion accuracy run the "
                                "dedicated Camera intrinsics capture (Step 0) and re-run.")
                    report_intrinsics_auto = {**intr, "source": "auto"}
                except Exception as e:   # noqa: BLE001 - the optional step must never abort a run
                    ctx.log(f"auto intrinsics skipped ({type(e).__name__}: {e}); "
                            f"continuing with the configured intrinsics")

            # Scale the holdout down so the training set never drops below
            # MIN_TRAIN_VIEWS — a thin capture (poses lost to reachability /
            # detection) should spend its views on the solve, not validation.
            eff_holdout = min(holdout, max(0, len(views) - MIN_TRAIN_VIEWS))
            if eff_holdout < holdout:
                ctx.log(f"holdout reduced {holdout} -> {eff_holdout} to keep "
                        f">= {MIN_TRAIN_VIEWS} training views ({len(views)} usable)")
            holdout = eff_holdout

            train, val = _split_views(views, holdout, ccfg.holdout_strategy,
                                      ccfg.split_seed)
            ctx.progress(len(targets), len(targets), "solving")

            def _solve(vs):
                """Solve one view set, honouring solver_method ("best" tries all)."""
                if ccfg.solver_method == "best":
                    return solve_best(vs, K, dist)            # (X, method, ranking)
                return solve_handeye(vs, ccfg.solver_method), ccfg.solver_method, None

            X, method, ranking = _solve(train)

            # Robust pass: drop training views whose reprojection is an outlier
            # (a mis-detected board or bad pose drags the linear solve) and re-solve
            # on the survivors. Conservative — a clean capture drops nothing. Held-out
            # validation views are never touched (that would bias the metric).
            rejected: list[str] = []
            if ccfg.reject_outliers:
                T_bt0 = estimate_board_in_base(train, X)
                kept, dropped, thr = reject_outliers(
                    train, X, T_bt0, K, dist, abs_px=ccfg.outlier_px,
                    factor=ccfg.outlier_factor, min_keep=MIN_TRAIN_VIEWS)
                if dropped and len(kept) < len(train):
                    ctx.log(f"outlier rejection: dropped {len(dropped)} view(s) over "
                            f"{thr:.2f}px ({', '.join(dropped)}); re-solving on {len(kept)}")
                    train, rejected = kept, dropped
                    X, method, ranking = _solve(train)

            T_bt = estimate_board_in_base(train, X)
            if do_refine:
                X, T_bt = refine(train, X, T_bt, K, dist)
            ctx.log(f"solver: {method}{' (+refine)' if do_refine else ''}"
                    + (f"; ranking " + ", ".join(f"{m} {r:.2f}px" for m, r in ranking)
                       if ranking else ""))
            xcheck = (verify_intrinsics(train, K, dist, cfg.camera.size)
                      if ccfg.verify_intrinsics else None)
            # Pass the configured strategy (not the winning method) so a "best" run
            # re-selects per fold — an honest cross-val that prices in the selection.
            cv_rms = cross_validate(train, ccfg.solver_method, K, dist, ccfg.cross_val_folds)
            report = evaluate(train, val, X, T_bt, K, dist, refined=do_refine,
                              method=method, method_ranking=ranking,
                              intrinsics_check=xcheck, cross_val_rms_px=cv_rms,
                              rejected_views=rejected)
            self.solved_X = X

            report_dict = report.to_dict()
            if report_intrinsics_auto is not None:
                report_dict["intrinsics_auto"] = report_intrinsics_auto
            summary = report.summary()
            for line in summary.splitlines():
                ctx.log(line)
            if report_intrinsics_auto is not None:                # already logged above
                summary += (
                    f"\nintrinsics: auto-calibrated from {report_intrinsics_auto['n_views']} "
                    f"views (fit RMS {report_intrinsics_auto['rms_px']:.3f} px, coverage "
                    f"{int(report_intrinsics_auto['coverage_pct'] * 100)}%); applied")
            (run_dir / "report.json").write_text(json.dumps(report_dict, indent=2),
                                                 encoding="utf-8")
            (run_dir / "summary.txt").write_text(summary, encoding="utf-8")
            # A tiny sidecar so apply-by-run-id (after a server restart, when the
            # in-memory job is gone) knows which tool this run solved for. The
            # solved transform itself already lives in report.json.
            (run_dir / "meta.json").write_text(json.dumps(
                {"module": "calibration", "stamp": stamp,
                 "tool_name": self.tool_name, "method": method,
                 "refined": do_refine}, indent=2), encoding="utf-8")

            self.result = CalibrationResult(
                report=report_dict, summary=summary, run_dir=str(run_dir),
                tool_name=self.tool_name, n_captured=len(views), n_skipped=skipped)
            return {
                "summary": summary, "report": report_dict, "run_dir": str(run_dir),
                "tool_name": self.tool_name, "n_captured": len(views),
                "n_skipped": skipped, "can_apply": True,
            }
        finally:
            # Return to where the run started. The generated targets are left in
            # the station (the user created them deliberately); Clear poses removes
            # them.
            if start_joints is not None:
                try:
                    ctx.log("returning to start pose")
                    rdk.move_j_joints(start_joints)
                except Exception:
                    pass

    # -- helpers ------------------------------------------------------------
    def _grab_frames(self, cam, n: int):
        """Grab ``n`` frames as fresh as possible for median detection. Uses a
        held stream when the client supports it (no per-frame reconnect) and falls
        back to one-shot grabs; ``n == 1`` is a single grab (the old behaviour)."""
        if n <= 1:
            return [cam.grab(color_only=True)]
        stream = getattr(cam, "stream", None)
        if stream is not None:
            try:
                out = []
                with stream(color_only=True) as s:
                    for k in range(n):
                        out.append(s.read(drain=(k == 0)))
                return out
            except Exception as e:   # noqa: BLE001 - capture must go on
                log.warning("stream burst failed (%s); using one-shot grabs", e)
        return [cam.grab(color_only=True) for _ in range(n)]

    def _capture(self, ctx, rdk, cam, board, K, dist, tool_pose, targets, run_dir):
        views, skipped = [], []
        total = len(targets)
        ccfg = self.services.config.calibration
        for i, name in enumerate(targets):
            ctx.check_cancel()
            ctx.progress(i + 1, total, f"capturing {name}")
            rdk.move_j(name)
            time.sleep(ccfg.settle_s)
            images = [f.color for f in self._grab_frames(cam, ccfg.frames_per_pose)]
            # Accept a pose into the solve only with the higher SOLVE corner floor
            # (not the low aiming-detection floor): a weak few-corner view gives a
            # noisy per-view board pose that drags the linear hand-eye solve.
            det = board.detect_median(images, K, dist,
                                      min_corners=ccfg.min_charuco_corners_solve)
            if det is None:
                ctx.log(f"{name}: board not seen with >= "
                        f"{ccfg.min_charuco_corners_solve} corners — skipped")
                skipped.append(name)
                continue
            rep = images[len(images) // 2]                 # a representative frame
            self._emit_frame(ctx, board.annotate(rep, det, K, dist, name),
                             run_dir / f"{name}.jpg")
            # The flange (gripper2base) the hand-eye solve needs, derived from the
            # active TCP and its offset — robust to which tool is active (vs the old
            # tcp_pose_T() @ inv(tool_pose), which was the FLANGE only if the camera
            # tool happened to be the active TCP).
            flange = rdk.flange_pose_T()
            views.append(CalibrationView(name, flange, det.R_target2cam,
                                         det.t_target2cam, det.corners, det.obj_points))
            n_frames = len(images)
            ctx.log(f"{name}: {det.n_corners} corners"
                    + (f" (median of {n_frames} frames)" if n_frames > 1 else ""))
        return views, skipped

    def _emit_frame(self, ctx, img, save_path=None) -> None:
        ok, jpeg = cv2.imencode(".jpg", img)
        if ok:
            ctx.frame(jpeg.tobytes())
            if save_path is not None and self.params.save_frames:
                cv2.imwrite(str(save_path), img)

    def apply_to_tool(self) -> str:
        """Write the solved camera pose into the Realsense tool (explicit step)."""
        if self.solved_X is None:
            raise RuntimeError("nothing solved yet")
        self.services.rdk.set_tool_pose(self.tool_name, self.solved_X)
        return self.tool_name
