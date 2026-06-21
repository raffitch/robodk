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
from ...core.geometry import invert_T
from ...core.jobrunner import JobContext
from ...core.logging import get_logger, new_run_dir
from ...core.rdk_io import RdkIO
from .charuco import CharucoTarget
from .handeye import (CalibrationView, cross_validate, estimate_board_in_base,
                      refine, reject_outliers, solve_best, solve_handeye)
from .intrinsics import verify_intrinsics
from .poses import generate_calibration_poses, select_diverse, viewing_angle_span
from .quality import evaluate

log = get_logger("tasni.calibration")

TARGET_PREFIX = "TasniCalib_"

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

    if not rdk.item_exists(tool_name):
        raise RuntimeError(
            f"tool {tool_name!r} not found — mount the RealSense camera "
            f"(3D model + a tool named {tool_name!r}) on the flange in RoboDK")

    # Free the camera (unicast) so our authoritative grab gets the frame.
    if services.live.running:
        services.live.stop()

    tool_pose = rdk.use_camera_tool(tool_name)
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

    seed_T = rdk.tcp_pose_T()
    look = float(reading.distance_mm)

    prior = rdk.list_targets(TARGET_PREFIX)
    if prior:
        rdk.delete_items(prior)

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

    reach_T = [T for _, T in reachable]
    sel = select_diverse(reach_T, min(ccfg.pose_count, n_reach),
                         seed_fwd=seed_T[:3, 2])
    chosen = [reachable[k] for k in sel]            # index-sorted -> spiral naming
    created: list[str] = []
    for _, T in chosen:
        name = f"{TARGET_PREFIX}{len(created) + 1:02d}"
        rdk.add_target(name, T)
        created.append(name)

    # Effective cone: how much of the configured cone the kept poses actually span.
    # At an edge-of-workspace seed the wide (diversity-rich) poses are unreachable,
    # so this can be far narrower than cone_half_angle_deg — warn BEFORE capture
    # rather than discovering it from motion_diversity after a full run.
    _, eff_max, eff_mean = viewing_angle_span([T for _, T in chosen], seed_T[:3, 2])
    services.bus.publish(JobEvent("log",
        {"message": f"created {len(created)} calibration targets "
                    f"(working distance ~{look:.0f} mm; {n_reach}/{len(candidates)} "
                    f"candidates reachable; effective cone ~{eff_max:.0f}° of "
                    f"{ccfg.cone_half_angle_deg:.0f}°) — inspect them in RoboDK"}))
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
            "effective_cone_deg": round(eff_max, 1)}


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


@dataclass
class TourPoseResult:
    """One pose's verdict on the simulated dry tour."""
    name: str
    reachable: bool
    collision: bool | None        # None = collisions not checked on this build
    ok: bool
    error: str | None = None

    def to_dict(self) -> dict:
        return {"name": self.name, "reachable": self.reachable,
                "collision": self.collision, "ok": self.ok, "error": self.error}


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

    def __init__(self, services):
        self.services = services
        self.tool_name: str = services.config.robodk.camera_tool

    def __call__(self, ctx: JobContext) -> dict:
        rdk: RdkIO = self.services.rdk

        if not rdk.item_exists(self.tool_name):
            raise RuntimeError(
                f"tool {self.tool_name!r} not found — mount the RealSense camera "
                f"(3D model + a tool named {self.tool_name!r}) on the flange in RoboDK")

        targets = rdk.list_targets(TARGET_PREFIX)
        if not targets:
            raise RuntimeError(
                "no TasniCalib_* targets to simulate — aim the camera until the "
                "gate is green and click Create targets first")

        prior_mode = rdk.current_run_mode()
        rdk.apply_run_mode("simulate")
        ctx.log(f"dry run (SIMULATE) — visiting {len(targets)} targets, no hardware motion")
        rdk.use_camera_tool(self.tool_name)
        collisions_on = rdk.set_collision_checking(True)
        try:
            start_joints = rdk.current_joints()
        except Exception:
            start_joints = None

        results: list[TourPoseResult] = []
        total = len(targets)
        try:
            for i, name in enumerate(targets):
                ctx.check_cancel()
                ctx.progress(i + 1, total, f"checking {name}")
                reachable = rdk.is_reachable(rdk.target_pose_T(name))
                collision: bool | None = None
                err: str | None = None
                if reachable:
                    try:
                        rdk.move_j(name)
                    except Exception as e:   # noqa: BLE001 - a sim move failure is a fail, not a crash
                        reachable, err = False, str(e)
                    if reachable and collisions_on:
                        n_col = rdk.collisions()
                        collision = None if n_col is None else bool(n_col)
                ok = reachable and not bool(collision)
                results.append(TourPoseResult(name, reachable, collision, ok, err))
                flag = "OK" if ok else ("UNREACHABLE" if not reachable else "COLLISION")
                ctx.log(f"{name}: {flag}")

            # Return-to-start (the guarantee the real run makes; verify it here too).
            returned = False
            if start_joints is not None:
                try:
                    rdk.move_j_joints(start_joints)
                    returned = True
                except Exception:
                    returned = False

            n_pass = sum(1 for r in results if r.ok)
            n_unreachable = sum(1 for r in results if not r.reachable)
            n_collision = sum(1 for r in results if r.collision)
            all_ok = n_pass == total and returned
            ctx.log(f"dry run complete: {n_pass}/{total} poses OK; "
                    f"return-to-start {'ok' if returned else 'FAILED'}"
                    + ("" if collisions_on else "; collisions not checked on this build"))
            return {
                "kind": "sim_tour",
                "total": total,
                "passed": n_pass,
                "unreachable": n_unreachable,
                "collisions": n_collision,
                "collisions_checked": collisions_on,
                "returned_to_start": returned,
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

        if not rdk.item_exists(self.tool_name):
            raise RuntimeError(
                f"tool {self.tool_name!r} not found — mount the RealSense camera "
                f"(3D model + a tool named {self.tool_name!r}) on the flange in RoboDK")

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

            summary = report.summary()
            for line in summary.splitlines():
                ctx.log(line)
            (run_dir / "report.json").write_text(json.dumps(report.to_dict(), indent=2),
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
                report=report.to_dict(), summary=summary, run_dir=str(run_dir),
                tool_name=self.tool_name, n_captured=len(views), n_skipped=skipped)
            return {
                "summary": summary, "report": report.to_dict(), "run_dir": str(run_dir),
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
            det = board.detect_median(images, K, dist,
                                      min_corners=ccfg.min_charuco_corners)
            if det is None:
                ctx.log(f"{name}: no board / too few corners — skipped")
                skipped.append(name)
                continue
            rep = images[len(images) // 2]                 # a representative frame
            self._emit_frame(ctx, board.annotate(rep, det, K, dist, name),
                             run_dir / f"{name}.jpg")
            flange = rdk.tcp_pose_T() @ invert_T(tool_pose)   # true flange in frame
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
