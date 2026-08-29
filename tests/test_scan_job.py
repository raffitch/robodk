"""Job-level scan test with fake core services — no RoboDK, real Open3D fusion.

Drives the real path:
  generate_scan_targets (depth gate -> seed=current pose -> reachable TasniScan_*)
  -> ScanCaptureJob (visit, grab synthetic depth, TSDF fuse, fit work plane)
  -> insert_scan (create frame + rectangle + mesh)

The fake camera renders depth of a flat 300x300 mm "table" at base z=0 from the
robot's current pose, so the fused surface + work frame are checked end to end.

    py -3.10 tests/test_scan_job.py
"""
from __future__ import annotations

import re
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import geometry_fixtures as gf  # noqa: E402
from tasni.core import runs  # noqa: E402
from tasni.core.camera_lease import CameraLease  # noqa: E402
from tasni.core.config import AppConfig, ScanConfig  # noqa: E402
from tasni.core.geometry import Rt_to_T  # noqa: E402
from tasni.core.jobrunner import JobContext  # noqa: E402
from tasni.modules.scan import service as scan_service  # noqa: E402
from tasni.modules.scan.classifier import CompactEligibility  # noqa: E402
from tasni.modules.scan.corner_evidence import CornerEvidence  # noqa: E402
from tasni.modules.scan.five_position import FivePositionSurvey  # noqa: E402
from tasni.modules.scan.survey_contract import (  # noqa: E402
    MODE_COMPACT, MODE_FIVE_POSITION, MODE_USER_SPECIFIED, PROVENANCE_COMPACT,
    PROVENANCE_USER_SPECIFIED, CaptureRecord, RobotStateSnapshot)

W, H = 320, 240
K = np.array([[300.0, 0, 160.0], [0, 300.0, 120.0], [0, 0, 1.0]])
TABLE_HALF_MM = 150.0
_ORIG_ROOT = runs.REPO_ROOT
# A fixed, deterministic, non-zero "camera frame timestamp (server clock)" so tests
# can assert the locked survey record's measurement_ts is real (not the 0.0 default).
FRAME_TIMESTAMP = 1_700_000_000.5


class _Ctx(JobContext):
    def __init__(self): self.frames = 0
    def progress(self, *a, **k): pass
    def log(self, *a, **k): pass
    def frame(self, *a, **k): self.frames += 1
    def check_cancel(self): pass


def _look_at(cam_pos, target):
    cam_pos = np.asarray(cam_pos, float)
    z = np.asarray(target, float) - cam_pos
    z /= np.linalg.norm(z)
    a = np.array([1.0, 0, 0]) if abs(z[2]) > 0.9 else np.array([0, 0, 1.0])
    x = np.cross(a, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return Rt_to_T(np.column_stack([x, y, z]), cam_pos)


def _render(T_base_cam, table_half_mm=None, noise_mm=0.0, rng=None, rotation_deg=0.0):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    dirs_cam = np.stack([(us - cx) / fx, (vs - cy) / fy, np.ones_like(us, float)], -1)
    R, t = T_base_cam[:3, :3], T_base_cam[:3, 3]
    dirs_base = dirs_cam @ R.T
    dz = dirs_base[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (0.0 - t[2]) / dz
    P = t + s[..., None] * dirs_base
    half_mm = TABLE_HALF_MM if table_half_mm is None else table_half_mm
    # Task 18 review rounds 2/3: an in-plane (world Z axis) rotation of the
    # table square itself, independent of camera pose -- used to test the
    # identity gate against a rectangle rotated near 45 degrees relative to
    # the image axes while centred in view (the orientation earlier per-frame
    # canonicalization attempts had trouble with).
    if rotation_deg:
        th = np.radians(rotation_deg)
        cr, sr = np.cos(th), np.sin(th)
        Px = P[..., 0] * cr + P[..., 1] * sr
        Py = -P[..., 0] * sr + P[..., 1] * cr
    else:
        Px, Py = P[..., 0], P[..., 1]
    valid = ((np.abs(Px) <= half_mm) & (np.abs(Py) <= half_mm)
             & (s > 0) & np.isfinite(s))
    if noise_mm > 0:
        # Task 18 review, Critical 1: REAL per-pixel depth noise (Gaussian,
        # independently drawn per call when the caller passes a shared `rng`
        # across several _render() calls in one lock -- i.e. genuinely
        # different per-frame noise, not the same noise repeated), rounded to
        # whole mm like a real integer-mm depth stream. Only perturbs where
        # `valid` was already True -- noise never creates or removes coverage.
        r = rng if rng is not None else np.random.default_rng()
        s = np.where(valid, np.clip(np.round(s + r.normal(0.0, noise_mm, s.shape)), 1.0, None), s)
    depth = np.where(valid, s, 0).astype(np.uint16)
    # Task 18: color must genuinely distinguish the table from its surroundings so
    # a REAL segmentation boundary engine (color_work_boundary / sam_work_boundary,
    # now wired into lock_scan_surface's classify_compact call) can find it --
    # a flat uniform-gray frame (the old fixture) makes every boundary engine
    # abstain (verified empirically), which would make classify_compact reject
    # every synthetic lock as "boundary not confirmed by segmentation" regardless
    # of geometry. Two-tone, keyed on the same `valid` mask as depth, mirrors how
    # a real RGB frame actually looks (a light table against a darker surround).
    color = np.where(np.repeat(valid[..., None], 3, axis=2), 200, 50).astype(np.uint8)
    return SimpleNamespace(color=color, depth=depth, timestamp=FRAME_TIMESTAMP,
                            geometry=gf.aligned(K, (W, H)))


def _build_fakes(mount_mm=(40.0, -15.0, 55.0)):
    # Task 18: 440 mm (was 420) -- straight down, close framed standoff, with
    # enough margin that the 300x300 mm table's raw rectangle clears
    # compact_guard_uv on BOTH image axes (the 320x240 frame is not square, so
    # the vertical axis is the tighter one; at 420 mm the table's raw corners sat
    # at v=0.058/0.942, just OUTSIDE the default 0.06 guard band -- confirmed
    # empirically). classify_compact now genuinely enforces that gate (it was
    # never wired in, hence never exercised, before this task), so a scene that
    # merely LOOKED like a comfortably-framed compact capture needs to actually
    # be one. 440 mm was chosen (over going further) because the "distance"
    # gate's own planner-computed ideal standoff for this table sits at ~413-416
    # mm independent of the CURRENT camera z (it comes from the measured extent,
    # not from where the camera happens to be) -- so moving the seed too far
    # trades a guard-band failure for a distance-gate failure instead (confirmed
    # empirically: 460 mm clears guard_uv but misses distance_tol_mm=50 by ~2
    # mm). 440 mm sits in the middle of the feasible window, comfortably inside
    # both gates (~25 mm of the 50 mm distance budget spent either way, guard
    # margin v=0.079..0.921 vs the 0.06/0.94 band) -- and leaves size_mm
    # ~296x296 (unchanged within measurement noise from the old 420 mm seed).
    seed_T = _look_at((0, 0, 440), (0, 0, 0))
    state = {"cam": seed_T, "targets": {}, "joints": {}}
    mount = Rt_to_T(np.eye(3), np.asarray(mount_mm, float))

    class FakeRdk:
        def __init__(self): self.inserted = {}
        def item_exists(self, name): return True
        def apply_run_mode(self, mode=None): return "run_robot"
        def connect_robot(self, ip="", *, timeout_s=10.0, poll_s=0.4):
            return True, "ROBOTCOM_READY"
        def robot_connection_params(self): return {"ip": "10.0.0.5", "port": 7000}
        # (ready, message) for the physical driver link -- Task 13 review Finding 2's
        # five_position_capture check. Healthy by default so every pre-existing test
        # (which assumes a connected driver) is unaffected; tests for the disconnected
        # case override this attribute directly.
        def robot_connected(self): return True, "ROBOTCOM_READY"
        def use_camera_tool(self, tool): return mount
        def camera_pose_T(self): return state["cam"]
        # A constant 6-element list (not the old opaque "HOME" sentinel): the new
        # explicit robot-state refresh (survey_contract.refresh_robot_state) reads
        # this numerically twice to decide stationary-vs-moving, so it must be a
        # real joint vector. Fixed across calls -> always reads as stationary here.
        def current_joints(self): return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        def move_j_joints(self, j): state["cam"] = seed_T
        def is_reachable(self, T): return True
        def screen_collisions(self, poses, *, guard_skip=None, **kw):
            out = [True] * len(poses), False, [None] * len(poses)
            if kw.get("return_details"):
                return (*out, {"poses": []})
            return out
        def solve_joints_for_pose(self, T, seed=None):
            return ("joints", float(T[0, 3]), float(T[1, 3]), float(T[2, 3]))
        def ensure_mounted_tool_collision_pairs(self, skip_trailing=2):
            return {"tools": ["Realsense"], "links": [0, 1, 2, 3, 4],
                    "pairs_enabled": 10, "pairs_failed": 0, "dof": 6}
        def list_targets(self, prefix=""):
            return sorted(n for n in state["targets"] if n.startswith(prefix))
        def add_target(self, name, T, joints=None):
            state["targets"][name] = T; state["joints"][name] = joints
        def delete_items(self, names):
            for n in list(names): state["targets"].pop(n, None)
        def move_j(self, name): state["cam"] = state["targets"][name]
        def add_frame(self, name, T, parent=None):
            # Keyed by name: insert_scan adds the corner frame AND the centre frame,
            # so a single "frame" slot would hide one of them.
            self.inserted["frame" if name == scan_service.FRAME_NAME else name] =                 np.asarray(T, float)
            return SimpleNamespace(Valid=lambda: True)
        def add_rectangle(self, name, corners, parent=None, color=None):
            self.inserted["rect"] = np.asarray(corners, float)
            return SimpleNamespace(Valid=lambda: True)
        def add_mesh_file(self, name, path, parent=None, color=None):
            self.inserted["mesh"] = path
            return SimpleNamespace(Valid=lambda: True)

    class FakeBurst:
        """Mimics _BurstSession: CAP renders+buffers the frame at the current pose,
        GET returns them all in order, CLEAR records that the Jetson buffer dropped."""
        def __init__(self): self._buf = []; self.cleared = False
        def capture(self):
            self._buf.append(_render(state["cam"]))
            return b"thumb"                       # non-empty -> shows in the strip
        def fetch_all(self): return list(self._buf)
        def clear(self): self.cleared = True

    class FakeCamera:
        def __init__(self): self.last_burst = None; self.grabs = 0
        def grab(self, with_depth=False, timeout=None, color_only=False):
            self.grabs += 1
            return _render(state["cam"])
        @contextmanager
        def burst(self, timeout=None):
            b = FakeBurst()
            self.last_burst = b
            yield b

    cfg = AppConfig()
    cfg.camera.intrinsics = {"320x240": K.tolist()}
    cfg.camera.resolution = "320x240"
    cfg.scan.pose_count = 8
    cfg.scan.flat_views = 8
    cfg.scan.cone_half_angle_deg = 30.0
    cfg.scan.voxel_size_m = 0.005
    cfg.scan.frames_per_pose = 1
    # Task 18: force the deterministic classical-CV boundary engine for these fakes
    # rather than the default "sam_then_color" -- the two-tone synthetic color
    # above is a good target for it, and it avoids a real ONNX inference pass (plus
    # its unpredictable-on-synthetic-images confidence score) on every lock in
    # every test in this file.
    cfg.scan.boundary_engine = "color"
    services = SimpleNamespace(config=cfg, rdk=FakeRdk(), camera=FakeCamera(),
                               camera_lease=CameraLease(),
                               bus=SimpleNamespace(publish=lambda *a, **k: None),
                               live=SimpleNamespace(running=False, stop=lambda: None),
                               calib_dry_tour_required=False)
    return services, state


def _expected_framed_views(services) -> int:
    scfg = services.config.scan
    return max(int(scfg.pose_count), int(scfg.flat_views)) + int(scfg.boundary_views)


def test_generate_run_insert():
    try:
        import open3d  # noqa: F401
    except Exception:
        print("[skip] open3d not installed — `pip install -e .[scan]`")
        return
    services, state = _build_fakes()
    rdk = services.rdk

    gen = scan_service.generate_scan_targets(services)
    expected_views = _expected_framed_views(services)
    assert gen["created"] == expected_views, gen
    assert all(n.startswith("TasniScan_") for n in gen["targets"])
    assert gen["gate"]["ok"] is True
    assert gen["calibration_on_file"] is True
    # The full-frame survey now drives the actual scan geometry: this 300 mm square
    # scans near the closest standoff that still frames its measured boundary with the
    # configured comfortable border (frame_margin 1.12).
    assert 405 < gen["look_distance_mm"] < 430
    target_z = [float(state["targets"][name][2, 3]) for name in gen["targets"]]
    assert min(target_z) >= gen["look_distance_mm"] - 1.0, target_z
    assert gen["planned_cone_deg"] == min(
        services.config.scan.raised_cone_deg,
        services.config.scan.flat_cone_deg + services.config.scan.boundary_cone_extra_deg)
    assert gen["planned_views"] == expected_views
    assert gen["boundary_views_enabled"] is True
    assert gen["boundary_aim_offsets"] == 9

    with tempfile.TemporaryDirectory() as t:
        runs.REPO_ROOT = Path(t)

        def fake_new_run_dir(mid, stamp):
            d = Path(t) / "runs" / mid / stamp
            d.mkdir(parents=True, exist_ok=True)
            return d
        orig = scan_service.new_run_dir
        scan_service.new_run_dir = fake_new_run_dir
        try:
            job = scan_service.ScanCaptureJob(services, scan_service.ScanParams())
            res = job(_Ctx())
            assert res["kind"] == "scan" and res["can_insert"] is True
            assert res["n_views"] == expected_views
            assert res["mesh_vertices"] > 0
            sz = res["plane"]["size_mm"]
            assert 240 < sz[0] < 360 and 240 < sz[1] < 360, sz   # ~300 x 300 mm table
            assert res["plane"]["inlier_frac"] > 0.8
            # frame Z (col 2) points up out of the table
            fT = np.asarray(res["plane"]["frame_T_mm"], float)
            assert float(fT[:3, 2] @ [0, 0, 1]) > 0.99, fT[:3, 2]
            # targets persist (user-created)
            assert len(rdk.list_targets("TasniScan_")) == expected_views

            # Insert from the in-memory result -> frame + rectangle + mesh created.
            out = scan_service.insert_scan(services, job=job)
            assert out["status"] == "inserted"
            assert "frame" in rdk.inserted and "rect" in rdk.inserted and "mesh" in rdk.inserted
            assert rdk.inserted["rect"].shape == (4, 3)
            active = runs.read_active("scan")
            assert active["frame"] == scan_service.FRAME_NAME
            # The rectangle is published in FRAME coordinates too, so downstream
            # modules (extrusion) can centre on the surface. The frame origin is a
            # corner, so the recorded centre must be ~half the extent, never (0, 0).
            corners_frame = np.asarray(active["rectangle_corners_frame_mm"], float)
            assert corners_frame.shape == (4, 3)
            np.testing.assert_allclose(corners_frame[:, 2], 0, atol=1e-6)
            centre = np.asarray(active["rectangle_center_frame_mm"], float)
            size = active["size_mm"]
            # Half the extent on each axis in MAGNITUDE, but not necessarily positive:
            # frame +Y is Z x X, which can point away from the rectangle (it does here,
            # Y spans -295..0). So (size/2, size/2) is NOT the centre — it has to come
            # from the corners, which is why insert publishes them.
            np.testing.assert_allclose(np.abs(centre), [size[0] / 2, size[1] / 2], atol=1e-6)
            assert np.linalg.norm(centre) > 1.0, centre
            np.testing.assert_allclose(centre, corners_frame[:, :2].mean(axis=0), atol=1e-6)

            # ...and the same middle is published as a FRAME under the corner frame, so
            # it is saved with the station and survives the runs directory being cleared.
            assert out["center_frame"] == scan_service.CENTER_NAME
            assert active["center_frame"] == scan_service.CENTER_NAME
            centre_T = rdk.inserted[scan_service.CENTER_NAME]
            np.testing.assert_allclose(centre_T[:3, :3], np.eye(3), atol=1e-9)
            np.testing.assert_allclose(centre_T[:3, 3], corners_frame.mean(axis=0), atol=1e-6)

            # Insert by run_id (from disk) also works.
            rdk.inserted.clear()
            stamp = res["stamp"]
            out2 = scan_service.insert_scan(services, run_id=stamp)
            assert out2["source"] == "run_id" and "frame" in rdk.inserted
        finally:
            scan_service.new_run_dir = orig
            runs.REPO_ROOT = _ORIG_ROOT
    print(f"[scan] gen {expected_views} ->", res["n_views"], "views fused;",
          res["mesh_vertices"], "verts; surface",
          tuple(round(s) for s in res["plane"]["size_mm"]), "mm; inserted")


def test_provenance_flows_lock_to_insert():
    """The §11 boundary provenance + survey record must thread end to end: a
    normal (fully framed, non-crop) lock builds a real LockedWorkframeSurvey,
    generate_scan_targets carries its provenance/survey/lock_token in its
    return dict, ScanParams/ScanCaptureJob carry them into the run report, and
    insert_scan carries them into the runs/scan/active.json payload -- reading
    provenance and geometry from the SAME resolved source (job.result)."""
    try:
        import open3d  # noqa: F401
    except Exception:
        print("[skip] open3d not installed — `pip install -e .[scan]`")
        return
    services, state = _build_fakes()

    locked = scan_service.lock_scan_surface(services)
    assert locked.survey_record is not None
    gen = scan_service.generate_scan_targets(services, locked)
    assert gen["boundary_provenance"] == locked.survey_record.boundary_provenance
    assert gen["survey"] == locked.survey_record.to_dict()
    assert gen["lock_token"] == locked.lock_token
    assert gen["lock_token"] != ""

    with tempfile.TemporaryDirectory() as t:
        runs.REPO_ROOT = Path(t)

        def fake_new_run_dir(mid, stamp):
            d = Path(t) / "runs" / mid / stamp
            d.mkdir(parents=True, exist_ok=True)
            return d
        orig = scan_service.new_run_dir
        scan_service.new_run_dir = fake_new_run_dir
        try:
            params = scan_service.ScanParams(
                boundary_provenance=gen["boundary_provenance"], survey=gen["survey"])
            job = scan_service.ScanCaptureJob(services, params)
            result = job(_Ctx())
            assert result["boundary_provenance"] == gen["boundary_provenance"]
            assert result["survey"] == gen["survey"]

            out = scan_service.insert_scan(services, result=job.result)
            assert out["status"] == "inserted"
            active = runs.read_active("scan")
            assert active["boundary_provenance"] == gen["boundary_provenance"]
            assert active["survey_quality"] == gen["survey"]["quality"]
        finally:
            scan_service.new_run_dir = orig
            runs.REPO_ROOT = _ORIG_ROOT
    print("[provenance] lock ->", gen["boundary_provenance"],
          "-> report -> insert payload survey_quality", active["survey_quality"])


def test_provenance_absent_when_survey_record_is_none():
    """A record-less lock's provenance must flow as ABSENT (None) end to end --
    never fabricated, never defaulted to a measured-sounding string.

    Adaptive-scan Task 3 removed the auto-overrun fallback this test originally
    used to obtain a record-less lock (that case now refuses with
    LargeSurfaceRequired instead of silently cropping). The one remaining
    legitimate record-less path is a fully framed surface that classify_compact
    rejects, so that is how the lock is produced now; the pipeline assertions
    are unchanged."""
    try:
        import open3d  # noqa: F401
    except Exception:
        print("[skip] open3d not installed — `pip install -e .[scan]`")
        return
    global TABLE_HALF_MM
    saved = TABLE_HALF_MM
    try:
        services, state = _build_fakes()
        # The generic work region can reach farther than the cone-limited tour
        # views capture on this synthetic table, which would otherwise trip the
        # (unrelated) measured-edge-coverage hard-fail gate -- this test is about
        # provenance flow, not synthetic mesh coverage.
        services.config.scan.actual_coverage_hard_fail = False
        ineligible = CompactEligibility(
            eligible=False, reasons=("rectangle is not sufficiently centered",),
            guard_ok=True, boundary_ok=True, centered_ok=False, tilt_ok=True,
            identity_ok=True, coverage_ok=True, predicted_coverage=None)
        orig = _patch_classify_compact(lambda *a, **k: ineligible)
        try:
            locked = scan_service.lock_scan_surface(services)
        finally:
            scan_service.classify_compact = orig
        assert locked.survey_record is None
        assert locked.lock_token != ""

        gen = scan_service.generate_scan_targets(services, locked)
        assert gen["boundary_provenance"] is None
        assert gen["survey"] is None
        assert gen["lock_token"] == locked.lock_token

        with tempfile.TemporaryDirectory() as t:
            runs.REPO_ROOT = Path(t)

            def fake_new_run_dir(mid, stamp):
                d = Path(t) / "runs" / mid / stamp
                d.mkdir(parents=True, exist_ok=True)
                return d
            orig = scan_service.new_run_dir
            scan_service.new_run_dir = fake_new_run_dir
            try:
                params = scan_service.ScanParams(
                    boundary_provenance=gen["boundary_provenance"], survey=gen["survey"])
                job = scan_service.ScanCaptureJob(services, params)
                result = job(_Ctx())
                assert result["boundary_provenance"] is None
                assert result["survey"] is None

                out = scan_service.insert_scan(services, result=job.result)
                assert out["status"] == "inserted"
                active = runs.read_active("scan")
                assert active["boundary_provenance"] is None
                assert active["survey_quality"] is None
            finally:
                scan_service.new_run_dir = orig
                runs.REPO_ROOT = _ORIG_ROOT
    finally:
        TABLE_HALF_MM = saved
    print("[provenance absent] auto-crop overrun -> no survey_record -> "
          "report/insert carry None throughout, no crash, no fabricated string")


def test_lock_then_create_targets_reuses_frozen_surface():
    services, state = _build_fakes()
    locked = scan_service.lock_scan_surface(services)
    assert locked.gate_payload["ok"] is True
    assert locked.survey.detected is True
    assert services.camera.grabs == services.config.scan.surface_measure_frames
    gen = scan_service.generate_scan_targets(services, locked)
    assert gen["created"] == _expected_framed_views(services)

    # A moved robot invalidates the frozen measurement instead of generating a
    # trajectory around stale geometry.
    locked2 = scan_service.lock_scan_surface(services)
    state["cam"] = locked2.seed_T.copy()
    state["cam"][0, 3] += 10.0
    try:
        scan_service.generate_scan_targets(services, locked2)
        raise AssertionError("expected moved robot to invalidate the lock")
    except RuntimeError as e:
        assert "moved after surface lock" in str(e), e
    print("[surface lock] frozen RGBD reused; 10 mm post-lock motion refused")


def test_lock_builds_locked_workframe_survey_compact():
    """A normal (fully-framed, non-crop) lock now also freezes the immutable §11
    LockedWorkframeSurvey contract alongside the legacy gate/survey/frame fields."""
    services, _state = _build_fakes()
    locked = scan_service.lock_scan_surface(services)
    rec = locked.survey_record
    assert rec is not None and rec.mode == MODE_COMPACT
    assert rec.boundary_provenance == PROVENANCE_COMPACT
    assert rec.corners_np().shape == (4, 3)
    assert rec.frame_np()[2, 2] > 0.99            # +Z up
    assert rec.calibration_id.startswith("cam-")
    assert rec.locked_robot.stationary is True
    assert locked.lock_token != ""
    # measurement_ts must be the real camera-frame timestamp (server clock), not the
    # 0.0 default that silently landed there when the record read a nonexistent
    # gate_payload["measurement_ts"] key.
    assert rec.captures[0].measurement_ts == FRAME_TIMESTAMP
    assert rec.captures[0].measurement_ts != 0.0
    # Task 18 no-regression pin: MODE_COMPACT must now flow through a REAL (not
    # monkeypatched) classify_compact call that reports every §6 gate satisfied --
    # not just an unconditional label as before this task wired it in.
    elig = locked.gate_payload["compact_eligibility"]
    assert elig["eligible"] is True and elig["reasons"] == (), elig
    assert elig["guard_ok"] and elig["boundary_ok"] and elig["centered_ok"]
    assert elig["tilt_ok"] and elig["identity_ok"]
    print("[survey record] compact lock ->", rec.mode, rec.boundary_provenance,
          "measurement_ts", rec.captures[0].measurement_ts)


def test_lock_crop_is_user_specified_with_declared_size():
    """force_crop + an explicit user_region_mm declares the boundary instead of
    measuring it: the record is USER_SPECIFIED and its size is the declared region,
    reusing the same camera->base transform path as the compact/measured case."""
    services, _state = _build_fakes()
    locked = scan_service.lock_scan_surface(
        services, force_crop=True, user_region_mm=(1200.0, 900.0))
    rec = locked.survey_record
    assert rec is not None
    assert rec.mode == MODE_USER_SPECIFIED
    assert rec.boundary_provenance == PROVENANCE_USER_SPECIFIED
    assert sorted(rec.size_mm, reverse=True) == [1200.0, 900.0]
    # Task 18: classify_compact is a compact-path-only gate -- an operator-declared
    # region is already honestly labeled and must never be run through it.
    assert "compact_eligibility" not in locked.gate_payload
    print("[survey record] crop lock -> user-specified", rec.size_mm)


def test_prepare_frame_result_converts_the_locked_survey_verbatim():
    """Plan Task 4: the frame-only route is a CONVERSION, not a re-measurement.

    What gets inserted must be bit-for-bit the geometry the operator reviewed, so
    the result's frame/corners are asserted identical to the locked record's -- no
    refit from another cloud can creep in. It also creates no targets and needs no
    scan captures, which is the whole point (no robot motion for a working frame).
    """
    services, _state = _build_fakes()
    locked = scan_service.lock_scan_surface(services, force_crop=True,
                                            user_region_mm=(1200.0, 900.0))
    rec = locked.survey_record
    result = scan_service.prepare_frame_result(services, locked)

    assert np.allclose(result.frame_T_mm, rec.frame_np())      # verbatim, not refit
    assert np.allclose(result.corners_mm, rec.corners_np())
    assert result.mesh_obj_path is None
    r = result.report
    assert r["mode"] == "frame_only" and r["mesh_file"] is None
    assert r["acquisition_mode"] == MODE_USER_SPECIFIED
    assert r["boundary_provenance"] == PROVENANCE_USER_SPECIFIED
    assert r["surface_scope"] == scan_service.SCOPE_DECLARED_REGION
    assert r["calibration_id"] == rec.calibration_id
    assert sorted(r["plane"]["size_mm"], reverse=True) == [1200.0, 900.0]
    print("[frame-only] prepared", r["plane"]["size_mm"], "captures", r["captures"])


def test_prepare_frame_result_refuses_when_the_robot_moved():
    """Same guard the motion path has: geometry frozen at a pose the robot has
    since left is not the geometry in front of the camera now."""
    services, state = _build_fakes()
    locked = scan_service.lock_scan_surface(services, force_crop=True,
                                            user_region_mm=(1200.0, 900.0))
    state["cam"] = _look_at((0, 0, 480), (0, 0, 0))       # drive the camera away
    try:
        scan_service.prepare_frame_result(services, locked)
        raise AssertionError("expected a refusal after the robot moved")
    except RuntimeError as e:
        assert "robot moved" in str(e), e
    print("[frame-only] robot moved after lock -> refused")


def test_prepare_frame_result_requires_a_survey_record():
    """No measured/declared boundary record => nothing trustworthy to insert."""
    services, _state = _build_fakes()
    locked = scan_service.lock_scan_surface(services, force_crop=True,
                                            user_region_mm=(1200.0, 900.0))
    locked.survey_record = None
    try:
        scan_service.prepare_frame_result(services, locked)
        raise AssertionError("expected a refusal without a survey record")
    except RuntimeError as e:
        assert "no measured boundary survey" in str(e), e
    print("[frame-only] missing survey record -> refused")


def test_entire_platform_overrun_refuses_instead_of_auto_cropping():
    """Adaptive-scan plan Task 3, the load-bearing invariant.

    A surface that overruns the camera view used to silently become the generic
    ``work_crop_mm`` square (a SYSTEM fallback carrying a warning and no survey
    record). Under ``entire_platform`` scope that fallback is now refused outright:
    a fabricated square is not the platform, so "entire platform" can never reach
    Insert through it. The refusal is structured, not a bare message, so the UI can
    offer the five-position survey as the one action that CAN measure this boundary.
    """
    global TABLE_HALF_MM
    saved = TABLE_HALF_MM
    TABLE_HALF_MM = 1000.0
    try:
        services, state = _build_fakes()
        state["cam"] = _look_at((0, 0, 310), (0, 0, 0))
        try:
            scan_service.lock_scan_surface(services)          # default: entire_platform
            raise AssertionError("expected LargeSurfaceRequired for an overrun platform")
        except scan_service.LargeSurfaceRequired as e:
            payload = e.payload
        assert payload["error"] == "large_surface_required"
        assert payload["primary_action"] == "survey_full_platform"
        assert payload["surface_scope"] == scan_service.SCOPE_ENTIRE_PLATFORM
        assert scan_service.SCOPE_DECLARED_REGION in payload["alternatives"]
        # 2026-08-30 false-refusal fix: the message must never claim the colour
        # view is narrower than the depth view when it is not -- _build_fakes
        # renders through gf.aligned (depth registered 1:1 to colour, the legacy/
        # archived convention), so colour_fov_deg and depth_fov_deg must come out
        # numerically equal here, and the message must NOT assert a gap that does
        # not exist. The old wording ("the platform overruns the camera view") is
        # also gone -- it named an undifferentiated "camera view" the operator had
        # no way to check against what they could plainly see.
        assert payload["colour_fov_deg"] == payload["depth_fov_deg"], payload
        assert "narrower" not in payload["message"], payload["message"]
        assert "the platform overruns the camera view" not in payload["message"]
        assert payload["standoff_mm"] is not None
        assert payload["extent_mm"] is not None and len(payload["extent_mm"]) == 2
    finally:
        TABLE_HALF_MM = saved
    print("[scope] entire-platform overrun -> refused:", payload["message"][:60])


def test_large_surface_message_names_the_true_colour_vs_depth_fov_gap():
    """2026-08-30 false-refusal investigation, cause 3: the operator reported
    ``"i can see the floor and the frame, so the platform is not larger than
    camera view"`` -- the refusal WAS correct (a real capture proved the RANSAC
    plane fit can genuinely extend past the colour frame, picking up something
    coplanar beyond the platform's visible edges through the wider native depth
    FOV), but its message named an undifferentiated "camera view" the operator
    could not verify against what they saw.

    ``_build_fakes`` (used by ``test_entire_platform_overrun_refuses_instead_of_
    auto_cropping`` above) renders through ``gf.aligned``, where depth is
    registered 1:1 to colour -- it cannot exercise this fix at all, since
    colour_fov and depth_fov come out identical there by construction. This
    tests the message-building helpers directly against a REAL ``gf.offset``
    geometry with a depth FOV wider than colour's, proving the fix actually
    names the gap (and the correct numbers) when one genuinely exists.
    """
    depth_K = np.array([[90.0, 0, 80.0], [0, 90.0, 60.0], [0, 0, 1.0]])
    offset_geom = gf.offset(color_K=K, color_size=(W, H), depth_K=depth_K, depth_size=(160, 120))
    aligned_geom = gf.aligned(K, (W, H))

    class _FakeSurvey:
        fov_deg = (
            float(np.degrees(2.0 * np.arctan(W / (2.0 * K[0, 0])))),
            float(np.degrees(2.0 * np.arctan(H / (2.0 * K[1, 1])))),
        )

    survey = _FakeSurvey()

    fov_aligned = scan_service._large_surface_fov_context(survey, aligned_geom)
    fov_offset = scan_service._large_surface_fov_context(survey, offset_geom)

    # gf.aligned: depth_K == color_K, depth_size == color_size by construction ->
    # the two FOVs must come out numerically identical.
    assert fov_aligned["colour_fov_deg"] == fov_aligned["depth_fov_deg"], fov_aligned

    # gf.offset with a much smaller depth fx/fy: the depth FOV is genuinely wider
    # on both axes.
    assert fov_offset["depth_fov_deg"][0] > fov_offset["colour_fov_deg"][0] + 1.0, fov_offset
    assert fov_offset["depth_fov_deg"][1] > fov_offset["colour_fov_deg"][1] + 1.0, fov_offset

    msg_aligned = scan_service._large_surface_message(
        "intro.", fov_aligned, [900.0, 500.0], 450.0)
    msg_offset = scan_service._large_surface_message(
        "intro.", fov_offset, [900.0, 500.0], 450.0)

    assert "narrower" not in msg_aligned, msg_aligned
    assert "narrower" in msg_offset, msg_offset
    # The actual depth FOV numbers must appear in the message, not just a vague claim.
    assert f"{fov_offset['depth_fov_deg'][0]:.0f}" in msg_offset, (fov_offset, msg_offset)
    assert f"{fov_offset['colour_fov_deg'][0]:.0f}" in msg_offset, (fov_offset, msg_offset)
    # The measured extent + standoff are named too, not just asserted in the abstract.
    assert "900" in msg_offset and "450" in msg_offset, msg_offset
    print("[fov message] aligned ->", msg_aligned[:70])
    print("[fov message] offset  ->", msg_offset[:110])


def test_full_frame_valid_frac_uses_colour_registration_not_raw_depth_r_important1():
    """Task 10 review, Important 1: ``lock_scan_surface`` used to take
    ``full_frame_valid_frac`` (the >= 0.95 "the whole view is surface" signal that
    gates the ``LargeSurfaceRequired`` invariant above, and ``_planned_surface_aim``'s
    quality-mode shortcut) as ``np.mean(depth > 0)`` on the RAW native depth image.
    That equalled "fraction of the colour view with depth" only under the old
    aligned stream (depth == colour image, 1:1); protocol 2's depth FOV is wider
    than colour's, so a platform that fills the ENTIRE colour view can leave the
    wider native depth image's periphery unfilled (background/out of range),
    silently making both >= 0.95 gates unreachable.

    Builds a REAL (offset) registration where a bounded plane fully covers the
    colour camera's FOV (so the colour-registered fraction must read ~1.0) but
    stops well inside the (deliberately much wider) native depth FOV, leaving its
    periphery genuinely empty -- reproducing the exact geometry of the defect: a
    platform that overruns nothing from the colour view's perspective still reads
    well under 0.95 raw-depth-frame coverage. Proves ``_colour_frame_valid_frac``
    (not ``np.mean(depth > 0)``) is what backs the >= 0.95 decision."""
    depth_K = np.array([[90.0, 0, 80.0], [0, 90.0, 60.0], [0, 0, 1.0]])
    geom = gf.offset(color_K=K, color_size=(W, H), depth_K=depth_K, depth_size=(160, 120))
    half_x, half_y = 300.0, 220.0     # covers the colour FOV (~267x200mm at z=500) with margin
    xs, ys = np.meshgrid(np.linspace(-half_x, half_x, 200), np.linspace(-half_y, half_y, 160))
    plane = np.column_stack([xs.ravel(), ys.ravel(), np.full(xs.size, 500.0)])
    depth = gf.render_depth_in_depth_camera(plane, geom)

    raw_frac = float(np.mean(depth > 0))
    assert raw_frac < 0.95, (
        f"fixture is not discriminating -- raw depth-frame fraction {raw_frac} "
        f"must be BELOW the 0.95 gate for this test to prove anything")

    frame = SimpleNamespace(depth=depth, geometry=geom)
    cfg = SimpleNamespace(camera=SimpleNamespace(K=K, dist=None))
    colour_frac = scan_service._colour_frame_valid_frac(frame, cfg, stride=8)
    assert colour_frac >= 0.95, (colour_frac, raw_frac)   # the FIX must clear the gate
    print("[full-frame frac] raw depth-frame", round(raw_frac, 3),
          "vs colour-registered", round(colour_frac, 3), "-> gate now correctly reachable")


def test_declared_region_still_crops_the_same_overrun_surface():
    """The other half of Task 3: the crop is not gone, it is now OPT-IN.

    Explicitly declaring a work region on the very same overrunning surface still
    locks, and is labeled user-specified (boundary declared, not measured) — so the
    capability the auto-fallback provided is preserved with honest provenance.
    """
    global TABLE_HALF_MM
    saved = TABLE_HALF_MM
    TABLE_HALF_MM = 1000.0
    try:
        services, state = _build_fakes()
        state["cam"] = _look_at((0, 0, 310), (0, 0, 0))
        locked = scan_service.lock_scan_surface(
            services, surface_scope=scan_service.SCOPE_DECLARED_REGION)
        assert locked.gate_payload["ok"] is True, locked.gate_payload
        assert locked.gate_payload["surface_mode"] == "crop", locked.gate_payload
        assert locked.surface_scope == scan_service.SCOPE_DECLARED_REGION
        assert locked.survey_record is not None
        assert locked.survey_record.mode == MODE_USER_SPECIFIED
        assert locked.gate_payload["boundary_provenance"] == PROVENANCE_USER_SPECIFIED
        assert locked.lock_token != ""
        # Still not the compact path -- a declared region is never classified.
        assert "compact_eligibility" not in locked.gate_payload
    finally:
        TABLE_HALF_MM = saved
    print("[scope] declared region on the same surface -> user-specified lock")


# --- Task 18: wire classify_compact (spec §6 entry conditions) into the lock ---
#
# classifier.py's own gate math (guard band, boundary overrun, centering, tilt,
# rectangle-identity) is unit-tested in test_scan_classifier.py; these tests
# cover the WIRING -- lock_scan_surface must call it on the compact path only,
# feed it REAL per-frame evidence (not the fused frame's outline repeated), and
# follow the exact same honest-provenance shape Task 3 established for the
# auto-overrun branch when it rejects.

def _patch_classify_compact(fake):
    """Manual save/restore for scan_service.classify_compact -- same pattern as
    _patch_latest_characterization above (this module doubles as a standalone
    script; see the __main__ block, so no pytest fixtures)."""
    orig = scan_service.classify_compact
    scan_service.classify_compact = fake
    return orig


def test_lock_real_guard_band_rejection_at_close_standoff():
    """Task 18 review, Important 4: every other rejection test in this section
    monkeypatches classify_compact with a canned CompactEligibility, so none of
    them prove the real gate MATHS actually reach the lock -- only that the
    lock branches correctly on whatever classify_compact returns. This is the
    one real (non-monkeypatched) guard-band failure in the suite: a genuinely
    closer standoff whose raw rectangle measurably leaves compact_guard_uv's
    margin (fully framed, comfortably inside the distance/angle bands, nothing
    else wrong) -- confirmed directly against the real production pipeline
    (survey_surface + classify_compact) before writing this test:
    v-range 0.0375..0.9625 against the 0.04 guard band, i.e. genuinely just
    outside it, not a synthetic/canned violation.

    (The original 420 mm scene this plan's own review used to find Critical 2
    no longer reproduces a guard failure once Critical 2's own fix -- lowering
    compact_guard_uv 0.06 -> 0.04 -- is applied: 420 mm's v-range 0.0583..0.9417
    clears 0.04 comfortably. 400 mm is the closest-available equivalent scene
    against the FIXED config, verified the same way.)
    """
    services, state = _build_fakes()
    state["cam"] = _look_at((0, 0, 400), (0, 0, 0))
    locked = scan_service.lock_scan_surface(services)
    assert locked.gate_payload["ok"] is True, locked.gate_payload   # distance/angle/framed all fine
    elig = locked.gate_payload["compact_eligibility"]
    assert elig["guard_ok"] is False, elig
    assert elig["eligible"] is False, elig
    assert locked.survey_record is None
    assert "survey" not in locked.gate_payload
    assert "boundary_provenance" not in locked.gate_payload
    # Critical 2c: the published warning is actionable, not just a bare "leaves
    # the guard region" -- it names the fix and roughly how much.
    assert any("guard region" in w and "back off" in w and "mm" in w
              for w in locked.gate_payload["warnings"]), locked.gate_payload
    assert locked.lock_token != ""
    print("[compact eligibility] real 400 mm guard-band violation -> rejected with an actionable hint:",
          [w for w in locked.gate_payload["warnings"] if "guard region" in w][0])


def test_lock_never_classifies_the_fabricated_reticle_square_when_not_fully_framed():
    """Task 18 review, Important 5: once survey.fully_framed is False,
    survey_surface has already REPLACED corners_cam_mm/outline_uv with a
    generic reticle square (not a measured boundary) -- see survey.py's own
    comment. The compact branch is reachable in exactly that state (detected,
    not fully framed, but NOT overrunning enough to trip the auto-crop 0.95
    valid_frac threshold -- a realistic "table runs off one edge" scene): a
    huge table with the camera aimed near one edge, so part of the frame is
    genuinely open space, not a full overrun. Before this fix, classify_compact
    would have been asked to judge the fabricated square, trivially satisfying
    gates a real unmeasured edge says nothing about (e.g. "centered") and
    misreporting why the lock refused. Confirmed first against the real
    pipeline: this exact scene gives valid_frac=0.787 (< 0.95, so NOT
    crop_mode) with detected=True, fully_framed=False.

    final_gates["framed"] (pre-existing, untouched by Task 18) still forces
    gate_payload["ok"]=False here, so the lock still raises -- but the
    published gate JobEvent (what a live HUD actually sees, one tick BEFORE
    the raise) must not carry a fabricated per-gate classification.
    """
    services, state = _build_fakes()
    global TABLE_HALF_MM
    saved = TABLE_HALF_MM
    TABLE_HALF_MM = 1000.0
    events = []
    services.bus = SimpleNamespace(publish=lambda e: events.append(e))
    try:
        state["cam"] = _look_at((900, 0, 440), (900, 0, 0))
        try:
            scan_service.lock_scan_surface(services)
            raise AssertionError("expected the not-fully-framed lock to refuse (framed gate)")
        except RuntimeError:
            pass
    finally:
        TABLE_HALF_MM = saved
    gate_events = [e for e in events if e.type == "gate"]
    assert gate_events, "expected the gate event to still be published before the raise"
    payload = gate_events[-1].payload
    assert payload["surface_mode"] == "full", payload   # not auto-crop -- valid_frac < 0.95
    assert "compact_eligibility" not in payload, payload
    assert "survey" not in payload and "boundary_provenance" not in payload
    assert any("not fully framed" in w for w in payload.get("warnings", [])), payload
    print("[compact eligibility] not-fully-framed partial overrun -> "
          "no fabricated classification, honest warning instead")


def _patch_sam_work_boundary(fake):
    orig = scan_service.sam_work_boundary
    scan_service.sam_work_boundary = fake
    return orig


def test_lock_falls_back_to_color_when_sam_abstains():
    """Task 18 review, Important 6: production's default boundary_engine is
    "sam_then_color" (these tests otherwise force "color" for determinism/speed
    -- see _build_fakes's own comment), but lock_scan_surface's compact branch
    never invoked ANY boundary engine before this task, so the SAM-first
    dispatch was untested on the lock's actual critical path. If SAM abstains
    (returns None -- a real, common outcome: low confidence, an unfamiliar
    scene, weights not loaded), sam_then_color's OWN contract is to fall back
    to colour, not to treat the abstention as an overrun. Proven end to end
    through the real (non-monkeypatched) _work_boundary dispatch: only
    sam_work_boundary is forced to abstain here; the fixture's real two-tone
    colour frame lets the REAL color_work_boundary fallback actually succeed.
    """
    services, _state = _build_fakes()
    services.config.scan.boundary_engine = "sam_then_color"
    orig = _patch_sam_work_boundary(lambda *a, **k: None)
    try:
        locked = scan_service.lock_scan_surface(services)
    finally:
        scan_service.sam_work_boundary = orig
    elig = locked.gate_payload["compact_eligibility"]
    assert elig["boundary_ok"] is True, elig
    assert elig["eligible"] is True, elig
    assert locked.survey_record is not None
    assert locked.survey_record.mode == MODE_COMPACT
    print("[compact eligibility] SAM abstains -> falls back to color -> still MODE_COMPACT")


def test_lock_rejects_when_every_boundary_engine_abstains():
    """Task 18 review, Important 6 (the failure half): when EVERY configured
    boundary engine abstains (SAM low-confidence AND colour low-contrast --
    both real, independently-documented failure modes elsewhere in this
    module), _work_boundary correctly returns None and classify_compact's
    boundary_ok reads False -- proven through the real dispatch, not assumed.
    This is the "a SAM abstention silently gives boundary_ok=False" case the
    review named: confirmed here it produces the SAME honest no-record/warning
    shape as every other §6 rejection, not a crash or a silent false-accept.
    """
    services, _state = _build_fakes()
    services.config.scan.boundary_engine = "sam_then_color"
    orig_sam = _patch_sam_work_boundary(lambda *a, **k: None)
    orig_color = scan_service.color_work_boundary
    scan_service.color_work_boundary = lambda *a, **k: None
    try:
        locked = scan_service.lock_scan_surface(services)
    finally:
        scan_service.sam_work_boundary = orig_sam
        scan_service.color_work_boundary = orig_color
    elig = locked.gate_payload["compact_eligibility"]
    assert elig["boundary_ok"] is False, elig
    assert elig["eligible"] is False, elig
    assert locked.survey_record is None
    assert "survey" not in locked.gate_payload
    assert "boundary_provenance" not in locked.gate_payload
    assert any("boundaries not confirmed" in w for w in locked.gate_payload["warnings"]), \
        locked.gate_payload
    assert locked.lock_token != ""
    print("[compact eligibility] SAM + color both abstain -> honest rejection, no record")


def test_lock_wires_classify_compact_and_rejects_when_ineligible():
    """The core Task 18 behaviour: when classify_compact reports the surface
    ineligible, lock_scan_surface must follow the EXACT same honest-provenance
    shape the pre-existing auto-overrun branch uses (Task 3) -- no survey record,
    no boundary_provenance/survey keys, a warning naming why -- but the lock still
    completes (a lock_token is still issued, gate_payload["ok"] is untouched)."""
    services, _state = _build_fakes()
    fake_result = CompactEligibility(
        eligible=False, reasons=("rectangle is not sufficiently centered",),
        guard_ok=True, boundary_ok=True, centered_ok=False, tilt_ok=True,
        identity_ok=True, coverage_ok=True, predicted_coverage=None)
    orig = _patch_classify_compact(lambda *a, **k: fake_result)
    try:
        locked = scan_service.lock_scan_surface(services)
    finally:
        scan_service.classify_compact = orig
    assert locked.survey_record is None
    assert "survey" not in locked.gate_payload
    assert "boundary_provenance" not in locked.gate_payload
    assert locked.gate_payload["ok"] is True, locked.gate_payload
    assert locked.lock_token != ""
    assert locked.gate_payload["compact_eligibility"]["eligible"] is False
    assert "rectangle is not sufficiently centered" in locked.gate_payload["warnings"]
    print("[compact eligibility] ineligible -> no record, warning:",
          locked.gate_payload["warnings"][-1])


def test_lock_reports_the_failing_gate_for_each_of_the_five_conditions():
    """§6 lists five independent hard gates; each must, ON ITS OWN, block the
    compact label and surface an operator-readable reason naming it -- proven
    per-gate via a canned CompactEligibility (classify_compact's own gate math is
    unit-tested separately; this is testing the wiring/branching, not re-deriving
    that math from synthetic geometry)."""
    cases = [
        ("guard_ok", "raw (untrimmed) boundary leaves the guard region"),
        ("boundary_ok", "four physical boundaries not confirmed by segmentation"),
        ("centered_ok", "rectangle is not sufficiently centered"),
        ("tilt_ok", "plane tilt exceeds the survey tolerance"),
        ("identity_ok", "rectangle identity not consistent across the multi-frame acquisition"),
    ]
    for failing_field, reason in cases:
        services, _state = _build_fakes()
        flags = {"guard_ok": True, "boundary_ok": True, "centered_ok": True,
                 "tilt_ok": True, "identity_ok": True}
        flags[failing_field] = False
        fake_result = CompactEligibility(
            eligible=False, reasons=(reason,), coverage_ok=True,
            predicted_coverage=None, **flags)
        orig = _patch_classify_compact(lambda *a, **k: fake_result)
        try:
            locked = scan_service.lock_scan_surface(services)
        finally:
            scan_service.classify_compact = orig
        assert locked.survey_record is None, failing_field
        assert "survey" not in locked.gate_payload, failing_field
        assert "boundary_provenance" not in locked.gate_payload, failing_field
        # substring, not equality: the guard-region reason is enriched with an
        # actionable backoff hint at the call site (Task 18 review, Critical 2c)
        # before it is published as a warning -- classify_compact's own reason
        # text (asserted verbatim in test_scan_classifier.py) is still a prefix.
        assert any(reason in w for w in locked.gate_payload["warnings"]), \
            (failing_field, locked.gate_payload)
        assert locked.gate_payload["compact_eligibility"][failing_field] is False, failing_field
        assert locked.lock_token != "", failing_field
    print("[compact eligibility] each of the 5 §6 gates independently blocks the compact label")


def test_lock_survey_outline_history_reflects_independent_per_frame_surveys():
    """The identity gate needs REAL per-frame evidence, not the fused frame's
    outline repeated n times -- proven by making the raw frames within a single
    lock genuinely differ (alternating table half-extent) and checking the REAL
    (non-monkeypatched) classify_compact call rejects on identity alone while
    every other §6 gate still passes. An implementation that fed
    outline_history=[survey.outline_uv] * n (the fused result repeated) could
    never fail this -- a value trivially agrees with itself no matter what the
    raw frames actually looked like."""
    services, state = _build_fakes()
    cam = services.camera
    calls = {"n": 0}

    def jittered_grab(with_depth=False, timeout=None, color_only=False):
        calls["n"] += 1
        cam.grabs += 1
        # Only the THIRD raw frame shrinks (150 -> 120 mm half-extent); the other
        # four stay nominal. Per-pixel median fusion is a per-pixel UNION, not an
        # average -- any pixel reached by ANY frame ends up valid in the fused
        # result (confirmed empirically) -- so a single smaller frame does not
        # shrink the fused/"survey" rectangle at all (still governed by the four
        # nominal frames): fused fully_framed/guard_ok/centered_ok all stay
        # exactly as in the unjittered baseline. But THAT one frame's own
        # independent outline differs from the others by ~0.10 normalized uv --
        # 2.5x compact_identity_tol_uv's default 0.04 -- real evidence a naive
        # "repeat the fused outline n times" implementation could never produce.
        half = 120.0 if calls["n"] == 3 else TABLE_HALF_MM
        return _render(state["cam"], table_half_mm=half)
    cam.grab = jittered_grab

    locked = scan_service.lock_scan_surface(services)
    assert calls["n"] == services.config.scan.surface_measure_frames
    elig = locked.gate_payload["compact_eligibility"]
    assert elig["identity_ok"] is False, elig
    assert elig["guard_ok"] and elig["boundary_ok"] and elig["centered_ok"] and elig["tilt_ok"], elig
    assert elig["eligible"] is False
    assert locked.survey_record is None
    assert "survey" not in locked.gate_payload
    assert "boundary_provenance" not in locked.gate_payload
    assert any("identity" in w for w in locked.gate_payload["warnings"]), locked.gate_payload
    assert locked.gate_payload["ok"] is True, locked.gate_payload   # the lock itself still succeeds
    assert locked.lock_token != ""
    print("[compact eligibility] genuinely differing raw frames -> identity gate correctly rejects")


def test_lock_identity_gate_accepts_a_stationary_rectangle_under_real_depth_noise():
    """Task 18 review, Critical 1: each raw frame's outline comes from an
    INDEPENDENT _oriented_rectangle fit (plane.py), so real per-frame depth
    noise can shift which corner that fit calls "first" -- a cyclic rotation
    -- even though the physical rectangle never moved.
    rectangle_identity_consistent compares corner-for-corner BY INDEX, so
    without aligning each subsequent outline to a common reference first, a
    genuinely stationary rectangle can spuriously FAIL the identity gate on
    ordinary sensor noise (measured directly on this exact fixture: 1.0 mm of
    per-pixel Gaussian depth noise -- squarely inside the D435i's real
    ~0.5-2 mm RMS band at 400-500 mm standoff -- produced a 0.8455
    normalized-uv "drift" by naive by-index comparison against the 0.04
    tolerance, purely from corner reordering; aligning each frame to the
    first successfully-surveyed frame via _align_polygon_like -- review
    round 3 -- drops that to ~0.001, matching the best possible corner
    correspondence). Complements the test above: that one proves a REAL
    difference is still caught; this one proves REAL noise without a real
    difference is not spuriously rejected. Both directions matter."""
    services, state = _build_fakes()
    cam = services.camera
    rng = np.random.default_rng(7)   # one shared RNG -> genuinely different noise per frame

    def noisy_grab(with_depth=False, timeout=None, color_only=False):
        cam.grabs += 1
        return _render(state["cam"], noise_mm=1.0, rng=rng)
    cam.grab = noisy_grab

    locked = scan_service.lock_scan_surface(services)
    elig = locked.gate_payload["compact_eligibility"]
    assert elig["identity_ok"] is True, elig
    assert elig["eligible"] is True, elig
    assert locked.survey_record is not None
    assert locked.survey_record.mode == MODE_COMPACT
    print("[compact eligibility] 1.0 mm real per-frame depth noise -> "
          "identity gate still accepts a stationary rectangle")


def test_lock_identity_gate_accepts_a_near_45_degree_centred_rectangle_under_noise():
    """Task 18 review history: round 1's per-frame canonicalization (start
    corner = nearest the image origin) had an EXACT algebraic tie for a
    perfectly centred, perfectly 45-degree-rotated square -- two corners are
    equidistant from the origin by construction, so noise could flip which
    one "wins" frame to frame. Round 2 tried a secondary per-frame rule
    (lexicographic tie-break) and fixed this exact angle, but review round 3
    found that ANY purely-geometric per-frame labelling rule has its OWN
    degenerate rotation somewhere (a square's 4-fold symmetry guarantees it)
    -- round 2's fix relocated the singularity to ~37-41 degrees rather than
    removing it. Round 3 replaced per-frame canonicalization entirely with
    cross-frame alignment (_align_polygon_like, aligning every frame to the
    first successfully-surveyed one) -- a comparison, not a labelling rule,
    with no orientation-dependent singularity (see
    test_align_polygon_like_has_no_degenerate_rotation_across_full_sweep for
    the full-range proof). This test keeps the ORIGINAL near-45-degree,
    centred, noisy scene as a concrete end-to-end regression -- it is no
    longer "the" hard case for this implementation, just one of many that all
    now behave identically well.

    classify_compact places no constraint on in-plane yaw (only centring via
    compact_center_tol_uv), so an operator placing a platform near 45 degrees
    to the camera's roll axis is plausible, not theoretical.

    This project's own rendering + fitting pipeline (_render's perspective
    projection through _oriented_rectangle's min-area-rectangle fit) has a
    small internal asymmetry -- confirmed by a fine-grained noiseless sweep
    before writing this test in round 2 -- so round 1's actual failure
    crossing on THIS fixture was 43.6-43.7 degrees rather than exactly 45.0;
    43.65 degrees is used directly, documented rather than silently
    substituted (matching this task's own precedent for adjusting a
    reviewer-specified scene to what the real pipeline actually produces --
    see the guard-band test's 400 mm vs. the originally-cited 420 mm). 115 mm
    half-extent is the largest that still clears every OTHER gate (detected,
    fully framed, guard, centred) for a 45-degree-rotated table on this
    synthetic 320x240 camera -- a rotated square's corners reach sqrt(2)
    further than an axis-aligned one of the same size, confirmed by a direct
    sweep in round 2.

    The pre-existing "distance" gate (untouched by Task 18) does not clear at
    this table size/standoff combination for a rotated shape (its planner
    was tuned for axis-aligned surfaces), so the lock still raises for that
    unrelated reason -- exactly like the not-fully-framed test above, this
    reads the identity gate off the published `gate` JobEvent one tick before
    the raise rather than off a successful return.
    """
    services, state = _build_fakes()
    cam = services.camera
    # seed=14 independently confirmed in round 2 (before either fix existed)
    # to trip the round-1 bug at this exact angle: max cross-frame drift
    # 0.5814 against a 0.04 tolerance -- see the round-2 fix report section
    # for the full seed sweep.
    rng = np.random.default_rng(14)
    events = []
    services.bus = SimpleNamespace(publish=lambda e: events.append(e))

    def rotated_noisy_grab(with_depth=False, timeout=None, color_only=False):
        cam.grabs += 1
        return _render(state["cam"], table_half_mm=115.0, rotation_deg=43.65,
                       noise_mm=1.0, rng=rng)
    cam.grab = rotated_noisy_grab

    try:
        scan_service.lock_scan_surface(services)
    except RuntimeError:
        pass
    gate_events = [e for e in events if e.type == "gate"]
    assert gate_events, "expected the gate event to still be published before the raise"
    payload = gate_events[-1].payload
    elig = payload.get("compact_eligibility")
    assert elig is not None, payload
    assert elig["identity_ok"] is True, elig
    print("[compact eligibility] near-45-degree centred rotation + 1.0 mm noise -> "
          "identity gate still accepts (cross-frame alignment, review round 3)")


def test_align_polygon_like_has_no_degenerate_rotation_across_full_sweep():
    """Task 18 review round 3 -- the reviewer's own acceptance bar, automated.

    Rounds 1 and 2 each canonicalized a SINGLE outline's corner order using a
    property of that outline in isolation (nearest the image origin; then a
    lexicographic tie-break). Both had a degenerate in-plane rotation: round 1
    near 45 degrees, and round 2's own fix MOVED it to ~37-41 degrees instead
    (the reviewer measured single-frame wrong-correspondence rates up to
    ~38% there at 0.01-0.02 normalized-uv noise -- worse than round 1's own
    45-degree band). A square's 4-fold rotational symmetry means no
    per-frame, purely-geometric labelling rule can avoid this; it can only
    relocate the singularity.

    _align_polygon_like (`tasni/modules/scan/service.py`, already written and
    reviewed for the live-preview rectangle stabiliser) sidesteps the whole
    class of bug: it answers "is this the same physical rectangle as a
    REFERENCE" (a comparison, searching all 4 cyclic rotations of the
    candidate and its reversal for the closest match) rather than "what is
    this corner's canonical label" (an absolute per-outline rule that always
    has somewhere to hide a singularity). ``_survey_outline_history`` now
    aligns every frame after the first to that first frame via this function
    instead of canonicalizing each one independently.

    This test is the reviewer's stated acceptance bar: sweep in-plane
    rotation across the FULL 0-90 degree range (a rectangle's own 4-fold
    symmetry makes that the whole story -- rotating further just repeats it)
    in 5-degree steps, at both cited noise levels, and assert there is NO
    rotation with an elevated single-frame flip rate -- not "better on
    average". Each trial independently relabels a reference and a candidate
    copy of the same true rectangle with a random cyclic shift + random
    reversal (mimicking _oriented_rectangle's uncontrolled per-frame
    ordering), adds independent Gaussian noise to each, aligns the noisy
    candidate to the noisy reference via the REAL _align_polygon_like, and
    checks the aligned result against the noiseless correctly-corresponding
    target -- a "flip" is when alignment picked a WRONG correspondence (error
    on the order of the rectangle's own scale), not just ordinary noise
    (error on the order of the noise level itself).
    """
    from tasni.modules.scan.service import _align_polygon_like

    def true_corners(theta_deg, a=0.2, b=0.2, cx=0.5, cy=0.5):
        th = np.radians(theta_deg)
        c, s = np.cos(th), np.sin(th)
        local = np.array([[a, b], [-a, b], [-a, -b], [a, -b]])
        rot = np.array([[c, -s], [s, c]])
        return (rot @ local.T).T + np.array([cx, cy])

    def relabel(corners, shift, flip):
        arr = corners[::-1] if flip else corners.copy()
        return np.roll(arr, shift, axis=0)

    # A correctly-aligned result differs from the noiseless target only by
    # noise (<= ~0.02 here); a WRONG correspondence differs by ~ the
    # rectangle's own scale (edge/diagonal length, ~0.28-0.57 for a=b=0.2 --
    # see this task's fix report for the measured range). 0.05 sits with
    # comfortable margin on both sides of that gap.
    FLIP_ERROR_THRESHOLD_UV = 0.05
    N_TRIALS = 300
    # Generous headroom over the ~0.2% noise floor this exact sweep measured
    # for the real fix (see the fix report's full table) -- and nowhere near
    # round 2's 5-38% at its own bad angles, so this remains a meaningful
    # bar, not a rubber stamp.
    MAX_ACCEPTABLE_FLIP_RATE = 0.03

    rng = np.random.default_rng(2026)
    worst = (None, None, -1.0)
    results = []
    for theta in range(0, 91, 5):
        for noise in (0.01, 0.02):
            flips = 0
            for _ in range(N_TRIALS):
                true_c = true_corners(theta)
                ref_true = relabel(true_c, int(rng.integers(0, 4)), bool(rng.integers(0, 2)))
                cand_true = relabel(true_c, int(rng.integers(0, 4)), bool(rng.integers(0, 2)))
                ref_noisy = ref_true + rng.normal(0.0, noise, ref_true.shape)
                cand_noisy = cand_true + rng.normal(0.0, noise, cand_true.shape)
                aligned = _align_polygon_like(ref_noisy, cand_noisy)
                err = float(np.mean(np.linalg.norm(aligned - ref_true, axis=1)))
                if err > FLIP_ERROR_THRESHOLD_UV:
                    flips += 1
            rate = flips / N_TRIALS
            results.append((theta, noise, rate))
            if rate > worst[2]:
                worst = (theta, noise, rate)
    worst_theta, worst_noise, worst_rate = worst
    assert worst_rate <= MAX_ACCEPTABLE_FLIP_RATE, (
        f"elevated flip rate {worst_rate:.1%} at theta={worst_theta} deg, "
        f"noise={worst_noise} -- a degenerate rotation exists; full sweep: {results}")
    print(f"[align sweep] 0-90 deg step 5, {N_TRIALS} trials/point x 2 noise levels: "
          f"worst = theta={worst_theta} deg, noise={worst_noise}, flip_rate={worst_rate:.2%} "
          "-- no elevated rotation found")


def test_lock_adapts_identity_frame_requirement_when_measure_frames_is_lower():
    """If compact_identity_frames (default 5) exceeds surface_measure_frames, the
    identity gate is UNSATISFIABLE by construction -- classify_compact has no
    override parameter (it reads scfg.compact_identity_frames directly), so the
    lock adapts the requirement down to what this acquisition can actually
    supply, and warns that it did so, rather than silently failing every lock
    under this configuration. (Adaptation has a floor of 2 -- see the next test.)
    """
    services, _state = _build_fakes()
    services.config.scan.surface_measure_frames = 3
    locked = scan_service.lock_scan_surface(services)
    elig = locked.gate_payload["compact_eligibility"]
    assert elig["identity_ok"] is True, elig
    assert elig["eligible"] is True, elig
    assert locked.survey_record is not None
    assert any("surface_measure_frames" in w and "3" in w
               for w in locked.gate_payload.get("warnings", [])), locked.gate_payload
    print("[compact eligibility] measure_frames=3 < identity_frames=5 -> adapted, still eligible")


def test_lock_never_vacuously_passes_identity_with_a_single_frame():
    """A single frame can never demonstrate MULTI-frame identity consistency --
    adapting the requirement down to 1 would make rectangle_identity_consistent
    trivially pass (nothing to disagree with), exactly the loophole the gate
    exists to close. The adaptation therefore has a floor of 2: below that, the
    nominal (unmet) compact_identity_frames requirement is left in place, so the
    gate fails honestly instead of rubber-stamping zero real evidence."""
    services, _state = _build_fakes()
    services.config.scan.surface_measure_frames = 1
    locked = scan_service.lock_scan_surface(services)
    elig = locked.gate_payload["compact_eligibility"]
    assert elig["identity_ok"] is False, elig
    assert elig["eligible"] is False, elig
    assert locked.survey_record is None
    assert any("identity" in w for w in locked.gate_payload["warnings"]), locked.gate_payload
    assert any("surface_measure_frames" in w for w in locked.gate_payload["warnings"]), \
        locked.gate_payload
    assert locked.lock_token != ""
    print("[compact eligibility] measure_frames=1 -> identity gate never vacuously passes")


def test_lock_gate_event_carries_survey_and_provenance():
    """The gate JobEvent published at lock time now carries the record's to_dict()
    and boundary_provenance, so the live HUD can show the locked polygon as-is."""
    services, _state = _build_fakes()
    events = []
    services.bus = SimpleNamespace(publish=lambda e: events.append(e))
    scan_service.lock_scan_surface(services)
    gate_events = [e for e in events if e.type == "gate"]
    assert gate_events, "expected at least one gate JobEvent"
    payload = gate_events[-1].payload
    assert payload["live"] is False
    assert "outline_uv" in payload            # locked polygon is displayable as-is
    assert payload["boundary_provenance"]
    assert payload["survey"]["mode"] == MODE_COMPACT
    print("[gate event] carries survey + boundary_provenance:",
          payload["boundary_provenance"])


# --- Task 16: calibration-age gate ---------------------------------------------
#
# lock_scan_surface reads tasni.core.characterize.latest_characterization
# (imported into scan_service's own namespace) after building the survey record,
# and warns (or hard-fails) when the on-file characterization is missing/stale.
# Patched the same way this file already patches survey_surface/color_work_
# boundary/extract_corner_evidence elsewhere (manual save/restore, not the
# pytest `monkeypatch` fixture) -- this module doubles as a standalone script
# (see the __main__ block), so every test here must run without fixtures.

def _patch_latest_characterization(fake):
    orig = scan_service.latest_characterization
    scan_service.latest_characterization = fake
    return orig


def test_lock_warns_when_characterization_missing():
    services, _state = _build_fakes()
    orig = _patch_latest_characterization(lambda root: None)
    try:
        locked = scan_service.lock_scan_surface(services)
    finally:
        scan_service.latest_characterization = orig
    assert "calibration verification missing or expired" in locked.gate_payload.get("warnings", [])
    print("[characterization gate] missing on-file characterization -> warning appended")


def test_lock_warns_when_characterization_stale():
    services, _state = _build_fakes()
    stale_date = (datetime.now() - timedelta(days=100)).isoformat()
    orig = _patch_latest_characterization(
        lambda root: {"date": stale_date, "dstar_mm": 400.0})
    try:
        locked = scan_service.lock_scan_surface(services)
    finally:
        scan_service.latest_characterization = orig
    assert "calibration verification missing or expired" in locked.gate_payload.get("warnings", [])
    print("[characterization gate] 100-day-old characterization (> 30 day default) -> warning")


def test_lock_records_dstar_into_survey_quality_when_fresh():
    services, _state = _build_fakes()
    fresh_date = datetime.now().isoformat()
    orig = _patch_latest_characterization(
        lambda root: {"date": fresh_date, "dstar_mm": 417.5})
    try:
        locked = scan_service.lock_scan_surface(services)
    finally:
        scan_service.latest_characterization = orig
    assert not locked.gate_payload.get("warnings")
    assert locked.survey_record is not None
    assert locked.survey_record.quality.get("dstar_mm") == pytest.approx(417.5)
    print("[characterization gate] fresh characterization -> no warning; "
          "dstar_mm recorded into the survey record's quality dict")


def test_lock_hard_fails_when_characterization_missing_and_hard_fail_enabled():
    services, _state = _build_fakes()
    services.config.scan.calibration_expiry_hard_fail = True
    orig = _patch_latest_characterization(lambda root: None)
    try:
        try:
            scan_service.lock_scan_surface(services)
            raise AssertionError("expected the missing characterization to hard-fail the lock")
        except RuntimeError as e:
            assert "calibration verification missing or expired" in str(e), e
    finally:
        scan_service.latest_characterization = orig
    print("[characterization gate] calibration_expiry_hard_fail=True -> RuntimeError, lock refused")


def test_lock_hard_fail_gate_event_carries_the_warning_before_raising():
    """Task 16 review, Finding 1: the hard-fail RuntimeError must not be the
    only signal. Before that fix, the published gate JobEvent in the
    hard-fail branch was byte-for-byte identical to a healthy lock (no
    warning text, ok computed earlier and never touched), so a client driven
    by the event stream could show "surface ready" a moment before the call
    errors -- directly contradicting the stated rationale for deferring the
    raise past the publish in the first place. Now the published event must
    itself show the lock as refused (ok=False) and carry the same warning
    text a soft (non-hard-fail) stale characterization would."""
    services, _state = _build_fakes()
    services.config.scan.calibration_expiry_hard_fail = True
    events = []
    services.bus = SimpleNamespace(publish=lambda e: events.append(e))
    orig = _patch_latest_characterization(lambda root: None)
    try:
        try:
            scan_service.lock_scan_surface(services)
            raise AssertionError("expected the hard-fail path to raise")
        except RuntimeError:
            pass
    finally:
        scan_service.latest_characterization = orig
    gate_events = [e for e in events if e.type == "gate"]
    assert gate_events, "expected the gate event to still be published before the raise"
    payload = gate_events[-1].payload
    assert payload["ok"] is False, payload
    assert "calibration verification missing or expired" in payload.get("warnings", []), payload
    print("[characterization gate] hard-fail gate event carries ok=False + warning before raising")


def test_surface_region_route_updates_lock_dimensions():
    """POST /surface/region persists an operator-declared work-region size and
    feeds it into subsequent surface_lock() calls (Task 3's user_region_mm
    param) without changing the force_crop-only provenance discriminator.

    ScanModule's route handlers are nested closures inside router() (see
    surface_lock), so this exercises the module's real bound method
    (self.surface_region), which the router registers as a thin wrapper — the
    same object the fixtures in this file already use for the fake services.
    """
    import tasni.modules.scan.module as scan_module
    from fastapi import HTTPException

    services, _state = _build_fakes()
    saved = {}
    orig_save_overrides = scan_module.save_overrides
    scan_module.save_overrides = lambda updates: saved.update(updates)
    try:
        mod = scan_module.ScanModule(services)
        body = scan_module.SurfaceRegionBody(width_mm=1200.0, height_mm=900.0)
        out = mod.surface_region(body)
        assert out["user_region_mm"] == [1200.0, 900.0]
        assert saved == {"scan": {"work_crop_mm": [1200.0, 900.0]}}
        assert mod._user_region_mm == (1200.0, 900.0)

        try:
            mod.surface_region(scan_module.SurfaceRegionBody(width_mm=50.0, height_mm=900.0))
            raise AssertionError("expected out-of-range region to be rejected")
        except HTTPException as e:
            assert e.status_code == 422
        # A rejected update must not clobber the last accepted region.
        assert mod._user_region_mm == (1200.0, 900.0)
    finally:
        scan_module.save_overrides = orig_save_overrides
    print("[surface region] declared 1200x900 mm persisted; out-of-range rejected")


def test_targets_report_surface_coverage_from_footprint():
    """A fully-framed surface now drives COVERAGE-aware view selection (mirroring
    calibration), so target creation reports the predicted surface coverage and the
    survey carries the rectangle corners the footprint grid is built from."""
    services, _state = _build_fakes()
    locked = scan_service.lock_scan_surface(services)
    # The survey exposes the oriented-rectangle corners (camera frame) the scan
    # transforms to base + densifies into the coverage footprint.
    corners = np.asarray(locked.survey.corners_cam_mm, float)
    assert corners.shape == (4, 3), corners.shape

    gen = scan_service.generate_scan_targets(services, locked)
    assert gen["created"] == _expected_framed_views(services), gen
    assert gen["boundary_views_enabled"] is True, gen
    # Small table, fully framed in every view -> the kept views tile (essentially)
    # the whole footprint, so coverage is reported and high (no missed region).
    assert gen["surface_coverage"] is not None, gen
    assert gen["surface_coverage"] >= 0.85, gen["surface_coverage"]
    print("[surface coverage] reported", f"{gen['surface_coverage']:.0%}",
          "from the measured rectangle footprint")


def test_burst_capture_path():
    """With scan.burst_capture on, the job captures via the burst session, fuses the
    same table, and CLEARs the Jetson buffer (no data left on the device)."""
    try:
        import open3d  # noqa: F401
    except Exception:
        print("[skip] open3d not installed — `pip install -e .[scan]`")
        return
    services, state = _build_fakes()
    services.config.scan.burst_capture = True

    gen = scan_service.generate_scan_targets(services)
    expected_views = _expected_framed_views(services)
    assert gen["created"] == expected_views, gen

    with tempfile.TemporaryDirectory() as t:
        runs.REPO_ROOT = Path(t)

        def fake_new_run_dir(mid, stamp):
            d = Path(t) / "runs" / mid / stamp
            d.mkdir(parents=True, exist_ok=True)
            return d
        orig = scan_service.new_run_dir
        scan_service.new_run_dir = fake_new_run_dir
        try:
            job = scan_service.ScanCaptureJob(services, scan_service.ScanParams())
            res = job(_Ctx())
            assert res["kind"] == "scan" and res["n_views"] == expected_views, res
            sz = res["plane"]["size_mm"]
            assert 240 < sz[0] < 360 and 240 < sz[1] < 360, sz   # ~300 x 300 mm table
            assert res["plane"]["inlier_frac"] > 0.8
            assert services.camera.last_burst is not None
            assert services.camera.last_burst.cleared, "burst buffer must be cleared on the Jetson"
        finally:
            scan_service.new_run_dir = orig
            runs.REPO_ROOT = _ORIG_ROOT
    print(f"[scan burst] gen {expected_views} ->", res["n_views"],
          "views fused via burst; buffer cleared")


def test_save_views_persists_per_pose_frames():
    """scan.save_views writes each pose's color+depth+pose under <run>/views/ for a
    later camera-perspective coverage overlay (off by default, opt-in diagnostic)."""
    try:
        import open3d  # noqa: F401
    except Exception:
        print("[skip] open3d not installed — `pip install -e .[scan]`")
        return
    import json as _json

    services, _state = _build_fakes()
    services.config.scan.save_views = True
    scan_service.generate_scan_targets(services)

    with tempfile.TemporaryDirectory() as t:
        runs.REPO_ROOT = Path(t)

        def fake_new_run_dir(mid, stamp):
            d = Path(t) / "runs" / mid / stamp
            d.mkdir(parents=True, exist_ok=True)
            return d
        orig = scan_service.new_run_dir
        scan_service.new_run_dir = fake_new_run_dir
        try:
            job = scan_service.ScanCaptureJob(services, scan_service.ScanParams())
            res = job(_Ctx())
            vdir = Path(res["run_dir"]) / "views"
            assert vdir.is_dir(), "views/ dir not created"
            assert len(list(vdir.glob("view_*.jpg"))) == res["n_views"]
            assert len(list(vdir.glob("depth_*.png"))) == res["n_views"]
            meta = _json.loads((vdir / "views.json").read_text())
            assert len(meta["views"]) == res["n_views"]
            # views.json now carries the full CameraGeometry (Task 10), not a bare
            # K/size pair -- the fakes render through gf.aligned(K, (W, H)), so its
            # own "size"/"K" fields (the legacy_aligned convention) still line up.
            geom = meta["camera_geometry"]
            assert geom["size"] == [W, H] and len(geom["K"]) == 3
            assert len(meta["views"][0]["pose_T_mm"]) == 4   # 4x4 pose persisted
        finally:
            scan_service.new_run_dir = orig
            runs.REPO_ROOT = _ORIG_ROOT
    print("[save_views] persisted", res["n_views"], "color+depth frames + poses")


def test_generate_targets_when_survey_touches_border():
    """A border-touching surface still gets a target tour — under DECLARED scope.

    Pre-Task-3 this exercised the silent auto-crop: generate with no explicit lock
    would quietly plan over the generic square. That fallback is gone (the default
    entire-platform scope now refuses with LargeSurfaceRequired — covered by
    test_entire_platform_overrun_refuses_instead_of_auto_cropping). The capability
    itself survives as an explicit declared-region lock, and target creation must
    still use the current-pose cone rather than fail because the surface reaches
    the image border.
    """
    global TABLE_HALF_MM
    saved = TABLE_HALF_MM
    TABLE_HALF_MM = 1000.0
    try:
        services, state = _build_fakes()
        state["cam"] = _look_at((0, 0, 310), (0, 0, 0))
        locked = scan_service.lock_scan_surface(
            services, surface_scope=scan_service.SCOPE_DECLARED_REGION)
        gen = scan_service.generate_scan_targets(services, locked)
        assert gen["created"] == 8, gen
        assert gen["boundary_views_enabled"] is False, gen
        assert gen["boundary_aim_offsets"] == 0, gen
        assert gen["gate"]["ok"] is True, gen["gate"]
        # _crop_gate_payload reports framed=False for the HUD (the surface really
        # does overrun the view) while excluding it from ok — same as before Task 3.
        assert gen["gate"]["gates"].get("framed") is False, gen["gate"]
        assert 300 <= gen["look_distance_mm"] <= 340, gen["look_distance_mm"]
        # The work region is the generic fixed square (scan.work_crop_mm default
        # 1000×1000) declared around the reticle, not an adaptive FOV-fraction.
        assert gen["crop_size_mm"] is not None
        assert gen["crop_size_mm"] == [1000.0, 1000.0], gen["crop_size_mm"]
    finally:
        TABLE_HALF_MM = saved
    print("[survey border] declared region -> created", gen["created"],
          "crop", gen["crop_size_mm"])


def test_manual_crop_ignores_unstable_framed_rectangle():
    services, _state = _build_fakes()
    locked = scan_service.lock_scan_surface(services, force_crop=True)
    assert locked.gate_payload["ok"] is True, locked.gate_payload
    assert locked.gate_payload["surface_mode"] == "crop", locked.gate_payload
    assert locked.gate_payload["crop_size_mm"] == [1000.0, 1000.0]
    assert locked.gate_payload["gates"].get("framed") is False

    gen = scan_service.generate_scan_targets(services, locked)
    assert gen["created"] == services.config.scan.pose_count, gen
    assert gen["crop_size_mm"] == [1000.0, 1000.0], gen
    assert gen["boundary_views_enabled"] is False, gen
    # The normal framed-rectangle path plans a closer standoff from the measured
    # extent; manual crop keeps the reticle/center-plane fallback -- i.e. the
    # fixture's own seed distance (440 mm, Task 18 -- was 420 before the seed
    # moved to clear compact_guard_uv; see _build_fakes's own comment).
    assert abs(gen["look_distance_mm"] - 440.0) < 1.0, gen["look_distance_mm"]
    print("[manual crop] fully framed but unstable rectangle -> reticle crop targets")


def test_scan_collision_filter_bypasses_noisy_wall_map_by_default():
    """Scan's collision gate is STRICT by default (Task 8, §10) — a soft bypass
    is not appropriate for a production target set. This test now exercises the
    soft path deliberately, via an explicit opt-out, to prove the bypass still
    works for an operator who consciously chooses it.

    If RoboDK reports every reachable candidate as colliding (for example an oversized
    wall collision mesh), target creation still leaves reachable targets for operator
    inspection when scan.collision_filter_hard_fail is explicitly disabled.
    """
    services, _state = _build_fakes()
    services.config.scan.collision_filter_hard_fail = False  # explicit opt-in to the soft path

    def all_collide(poses, *, guard_skip=None, **kw):
        out = [False] * len(poses), True, [None] * len(poses)
        if kw.get("return_details"):
            return (*out, {"poses": []})
        return out

    services.rdk.screen_collisions = all_collide
    gen = scan_service.generate_scan_targets(services)
    assert gen["created"] == _expected_framed_views(services), gen
    assert gen["collision_filter_bypassed"] is True, gen
    assert gen["candidates_collided"] == gen["candidates_reachable"], gen
    assert len(services.rdk.list_targets("TasniScan_")) == _expected_framed_views(services)
    print("[scan collision bypass] wall/noisy map reported all colliding -> created", gen["created"])


def test_scan_collision_filter_hard_fail_can_still_refuse():
    services, _state = _build_fakes()
    services.config.scan.collision_filter_hard_fail = True

    def all_collide(poses, *, guard_skip=None, **kw):
        out = [False] * len(poses), True, [None] * len(poses)
        if kw.get("return_details"):
            return (*out, {"poses": []})
        return out

    services.rdk.screen_collisions = all_collide
    try:
        scan_service.generate_scan_targets(services)
        raise AssertionError("expected hard-fail scan collision filter to refuse")
    except RuntimeError as e:
        assert "collision-free poses" in str(e), e
    assert services.rdk.list_targets("TasniScan_") == []
    print("[scan collision hard fail] strict mode refused noisy collision map")


def test_generate_refuses_when_too_far():
    services, state = _build_fakes()
    state["cam"] = _look_at((0, 0, 900), (0, 0, 0))      # 900 mm > 500 +/- 150
    try:
        scan_service.generate_scan_targets(services)
        raise AssertionError("expected refusal — surface out of the standoff band")
    except RuntimeError as e:
        assert "distance" in str(e)
    assert services.rdk.list_targets("TasniScan_") == []
    print("[gate refusal] too-far standoff refused, nothing created")


def test_generate_reference_mode_for_oversized_framed_surface():
    """A fully framed surface whose framing standoff exceeds the accurate depth band
    -> reference mode: a single-frame rectangle placed directly (no tour, no fusion),
    returned as mode='reference' + _scan_result. Previously this fell through to a
    quality tour and _reference_locate was dead code (never invoked).

    The reachable window is narrow (framing needs a far standoff while the distance
    gate caps it near accurate_max), so the test lowers accurate_max_mm to trigger it
    with comfortable geometric margins.
    """
    global TABLE_HALF_MM
    saved = TABLE_HALF_MM
    # 4:3 frame (240 px tall) binds vertically: framing needs Z >= ~2.60*half, the
    # distance gate caps Z <= accurate_max+50, and reference needs d_fit = ~2.625*half
    # > accurate_max. half=205, accurate_max=500, Z=540 satisfies all three.
    TABLE_HALF_MM = 205.0
    try:
        services, state = _build_fakes()
        services.config.scan.accurate_max_mm = 500.0
        state["cam"] = _look_at((0, 0, 540), (0, 0, 0))
        with tempfile.TemporaryDirectory() as t:
            runs.REPO_ROOT = Path(t)

            def fake_new_run_dir(mid, stamp):
                d = Path(t) / "runs" / mid / stamp
                d.mkdir(parents=True, exist_ok=True)
                return d
            orig = scan_service.new_run_dir
            scan_service.new_run_dir = fake_new_run_dir
            try:
                gen = scan_service.generate_scan_targets(services)
                assert gen["mode"] == "reference", gen
                assert gen["created"] == 0 and gen["targets"] == [], gen
                assert "_scan_result" in gen, gen
                r = gen["_scan_result"]
                assert r.report["mode"] == "reference", r.report
                assert r.mesh_obj_path is None, "reference mode places no fused mesh"
                assert r.report["plane"]["size_mm"][0] > 300.0, r.report["plane"]
                # No tour targets were created for reference mode.
                assert services.rdk.list_targets("TasniScan_") == []
            finally:
                scan_service.new_run_dir = orig
                runs.REPO_ROOT = _ORIG_ROOT
    finally:
        TABLE_HALF_MM = saved
    print("[reference mode] oversized framed surface -> single-frame rectangle, no tour")


def test_generate_accepts_dynamic_near_quality_distance():
    services, state = _build_fakes()
    state["cam"] = _look_at((0, 0, 420), (0, 0, 0))
    gen = scan_service.generate_scan_targets(services)
    assert gen["created"] == _expected_framed_views(services), gen
    assert 405 < gen["look_distance_mm"] < 430, gen["look_distance_mm"]
    print("[dynamic distance] near quality-band surface accepted at",
          round(gen["look_distance_mm"]), "mm")


def test_warns_but_proceeds_without_calibration():
    """Decoupling: a near-identity tool offset (no calibration) must NOT block —
    it warns and still creates targets (calibration_on_file=False)."""
    services, state = _build_fakes(mount_mm=(0.0, 0.0, 2.0))   # ~no offset
    gen = scan_service.generate_scan_targets(services)
    assert gen["created"] == _expected_framed_views(services)
    assert gen["calibration_on_file"] is False
    print("[decoupled] no calibration on file -> warned, still created", gen["created"])


def test_run_without_targets_errors():
    services, _state = _build_fakes()
    try:
        scan_service.ScanCaptureJob(services, scan_service.ScanParams())(_Ctx())
        raise AssertionError("expected run to require targets")
    except RuntimeError as e:
        assert "targets" in str(e)
    print("[run needs targets] refused")


def _build_fakes_with_jobs(**kw):
    """_build_fakes() plus a fake JobRunner-shaped services.jobs (Task 8's guard
    tests drive ScanModule.run() directly, which touches services.jobs before
    the real scan job ever needs to execute — a real ScanCaptureJob run belongs
    to test_generate_run_insert, not to these route-guard tests)."""
    services, state = _build_fakes(**kw)
    started = {}
    services.jobs = SimpleNamespace(
        running=False,
        start=lambda job, name="job": started.update(job=job, name=name))
    return services, state, started


def test_run_refuses_targets_from_a_previous_lock():
    """§Task 8: the operator UNLOCKED the surface after generating targets — the
    targets now predate the current lock (there is none) and must be refused,
    not silently run against stale geometry."""
    import tasni.modules.scan.module as scan_module

    services, _state, _started = _build_fakes_with_jobs()
    mod = scan_module.ScanModule(services)
    mod.surface_lock(scan_module.SurfaceLockBody(mode="auto"))
    mod.poses_generate()
    mod.surface_unlock()                       # locked state changed after generation
    with pytest.raises(Exception, match="regenerate"):
        mod.run()
    print("[lock guard] targets orphaned by unlock -> run refused")


def test_run_refuses_targets_from_a_relocked_surface():
    """§Task 8: the operator RE-LOCKED (not just unlocked) after generating
    targets — a fresh lock_token means the targets were computed for the
    PREVIOUS lock's geometry, which may since have moved."""
    import tasni.modules.scan.module as scan_module

    services, _state, _started = _build_fakes_with_jobs()
    mod = scan_module.ScanModule(services)
    mod.surface_lock(scan_module.SurfaceLockBody(mode="auto"))
    mod.poses_generate()
    mod.surface_lock(scan_module.SurfaceLockBody(mode="auto"))   # re-lock, no re-generate
    with pytest.raises(Exception, match="regenerate"):
        mod.run()
    print("[lock guard] targets orphaned by re-lock -> run refused")


def test_run_starts_normally_after_lock_and_generate():
    """The guard must not block the ordinary lock -> generate -> run flow: this
    is the happy path every real scan takes, so it has to keep working."""
    import tasni.modules.scan.module as scan_module

    services, _state, started = _build_fakes_with_jobs()
    mod = scan_module.ScanModule(services)
    mod.surface_lock(scan_module.SurfaceLockBody(mode="auto"))
    mod.poses_generate()
    result = mod.run()
    assert result == {"status": "started"}
    assert started.get("name") == "scan"
    print("[lock guard] normal lock -> generate -> run not blocked")


def test_run_not_blocked_when_targets_token_was_never_set():
    """No-guard case: targets present in RoboDK but this ScanModule instance
    never generated them (e.g. an older flow, or a fresh process after a
    restart) -- _targets_token defaults to "" and must not be treated as a
    stale-lock mismatch. /run only enforces the ordinary 'targets exist' gate."""
    import tasni.modules.scan.module as scan_module

    services, state, started = _build_fakes_with_jobs()
    mod = scan_module.ScanModule(services)
    assert mod._targets_token == ""
    services.rdk.add_target("TasniScan_0", state["cam"])
    result = mod.run()
    assert result == {"status": "started"}
    assert started.get("name") == "scan"
    print("[lock guard] no-guard case (token never set) does not block run")


def test_sparse_measured_support_is_rejected():
    cfg = AppConfig().scan
    coverage = {
        "point_count": 19,
        "fill": 0.00018,
        "weakest_edge": 0.0,
    }
    mesh_stats = {
        "support_fallback": True,
        "combined_vertices": 0,
    }
    reasons = scan_service._surface_quality_reasons(coverage, mesh_stats, cfg)
    assert any("only 19 supported" in r for r in reasons), reasons
    assert any("fill 0%" in r for r in reasons), reasons
    assert any("weakest edge support 0%" in r for r in reasons), reasons
    assert any("repeated multi-view" in r for r in reasons), reasons
    print("[surface quality] sparse measured support rejected:", len(reasons), "reasons")


# -- Task 12: five-position survey -> tiled tour wiring ---------------------
# generate_scan_targets' new FIRST branch (taken when the locked surface has a
# survey_record with mode=MODE_FIVE_POSITION) plans plan_rect_tour instead of
# the single-aim orbit. This reuses Task 11's real FivePositionSurvey happy
# path (mirrors tests/test_five_position.py's _run_all/_survey helpers) to
# build a genuinely validated LockedWorkframeSurvey, rather than hand-rolling
# one -- so this also exercises Task 11's own gates on the way in.
_FP_CLOCK = [100.0]


def _fp_snap():
    return RobotStateSnapshot((0.0,) * 6, tuple(map(tuple, np.eye(4))), _FP_CLOCK[0], True)


def _fp_record(kind, standoff=350.0, tilt=1.0):
    snap = RobotStateSnapshot((0.0,) * 6, tuple(map(tuple, np.eye(4))), _FP_CLOCK[0], True)
    return CaptureRecord(kind=kind, robot=snap, measurement_ts=1.0, captured_at=_FP_CLOCK[0],
                         n_frames=5, standoff_mm=standoff, tilt_deg=tilt, valid_frac=0.9,
                         plane_rms_mm=0.6, plane_normal_base=(0, 0, 1),
                         plane_point_base=(1000, 800, 0))


def _fp_plane_points(center, n=300, seed=0):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-150, 150, (n, 2)) + np.asarray(center, float)
    return np.column_stack([xy, rng.normal(0, 0.3, n)])


def _fp_corner_evidence(corners, i, seed=None):
    c, prev_c, next_c = corners[i], corners[(i - 1) % 4], corners[(i + 1) % 4]
    rng = np.random.default_rng(seed if seed is not None else i)
    pts = []
    for other in (prev_c, next_c):
        t = np.linspace(0.02, 0.30, 40)[:, None]
        seg = c + t * (other - c)
        seg = seg + np.column_stack([rng.normal(0, 0.4, 40), rng.normal(0, 0.4, 40),
                                     np.zeros(40)])
        pts.append(seg)
    return CornerEvidence(corner_uv=(0.5, 0.5), corner_base_mm=tuple(c),
                          edge_points_base=np.concatenate(pts, axis=0))


def _five_position_locked_survey(corners: np.ndarray):
    """Task 11's happy-path survey (center + four corners, in order) driven to
    completion -> a real, fully-gated LockedWorkframeSurvey (mode =
    MODE_FIVE_POSITION) for ``corners``, not a hand-rolled stand-in."""
    s = FivePositionSurvey(ScanConfig(), clock=lambda: _FP_CLOCK[0])
    s.add_capture(_fp_record("center"), _fp_plane_points(corners.mean(axis=0)[:2]), None)
    for i in range(4):
        s.add_capture(_fp_record(f"corner{i + 1}"),
                      _fp_plane_points(corners[i][:2], seed=i + 1),
                      _fp_corner_evidence(corners, i, seed=i + 1))
    assert s.step == "review"
    return s.finish(calibration_id="cam-test-five-position", locked_robot=_fp_snap())


def _locked_five_position_scan_surface(state, corners: np.ndarray, *,
                                       lock_token="five-pos-test"):
    rec = _five_position_locked_survey(corners)
    assert rec.mode == MODE_FIVE_POSITION
    return scan_service.LockedScanSurface(
        frame=None, reading=None, survey=None, gate_payload={},
        seed_T=state["cam"].copy(), seed_joints=[0.0] * 6,
        locked_at=time.monotonic(), survey_record=rec, lock_token=lock_token)


# A rectangle too large for one camera view at this fake's K/resolution
# (900 x 700 mm; the fake 320x240 camera's tile footprint is ~286x214 mm).
_FP_LARGE_CORNERS = np.array([[300.0, 400.0, 0.0], [1200.0, 400.0, 0.0],
                              [1200.0, 1100.0, 0.0], [300.0, 1100.0, 0.0]])


def test_five_position_tiled_tour_spans_multiple_tiles():
    """A five-position locked surface (a platform surveyed because it is too
    large for one camera view) is planned as a TILED close-range tour
    (plan_rect_tour), not the single-aim orbit -- generate_scan_targets' new
    FIRST branch (Task 12 ambiguity resolution #4). The 900x700 mm rectangle
    needs > 1 tile at this fake camera's K/resolution, so the created targets
    must span more than one tile, named TasniScan_T{tile:02d}_{k}."""
    services, state = _build_fakes()
    locked = _locked_five_position_scan_surface(state, _FP_LARGE_CORNERS)

    gen = scan_service.generate_scan_targets(services, locked)
    assert gen["mode"] == "quality", gen
    assert gen["plan"]["mode"] == "large_survey", gen["plan"]
    assert gen["tile_count"] > 1, gen["tile_count"]
    assert gen["created"] > 0, gen

    tile_ids = set()
    for name in gen["targets"]:
        m = re.match(r"TasniScan_T(\d+)_\d+$", name)
        assert m, name
        tile_ids.add(int(m.group(1)))
    assert len(tile_ids) > 1, tile_ids
    assert gen["surface_coverage"] is not None
    assert gen["surface_coverage"] >= services.config.scan.min_surface_coverage
    assert len(services.rdk.list_targets("TasniScan_")) == gen["created"]
    # Follow-up 5 (post-review): "planned_views" must report the PLANNED
    # figure (sum of every tile's own n_views), the same semantics the
    # single-aim path uses for that key -- not silently alias "created" (this
    # scenario is fully reachable so the two numbers happen to coincide here;
    # the contiguous-hole test below exercises a case where they differ).
    assert gen["planned_views"] == sum(a["n_views"] for a in gen["plan"]["aims"])
    print(f"[five-position tiled tour] {gen['tile_count']} tiles planned, "
          f"{len(tile_ids)} produced targets, {gen['created']} targets total, "
          f"coverage {gen['surface_coverage']:.0%}")


def test_five_position_tiled_tour_coverage_gate_catches_missing_tiles():
    """§10's coverage hard gate must apply ACROSS ALL TILES (ambiguity
    resolution #5), not just report near-100% because whichever tiles DID get
    a pose each fill their own close-range frame on their own. Force roughly
    half the surveyed rectangle's tiles to have zero reachable poses (as if
    that half were outside the workspace) and confirm the gate refuses rather
    than silently shipping a target set that leaves half the surface unscanned."""
    services, state = _build_fakes()
    locked = _locked_five_position_scan_surface(state, _FP_LARGE_CORNERS)

    midpoint_y = float(_FP_LARGE_CORNERS[:, 1].mean())
    real_is_reachable = services.rdk.is_reachable
    services.rdk.is_reachable = lambda T: float(np.asarray(T)[1, 3]) < midpoint_y
    try:
        with pytest.raises(RuntimeError, match="coverage"):
            scan_service.generate_scan_targets(services, locked)
    finally:
        services.rdk.is_reachable = real_is_reachable
    print("[coverage gate] missing-tile scenario (half the rectangle "
          "unreachable) correctly refused instead of reporting a false-pass")


# A many-tile rectangle at this fake camera's K/config (300 mm standoff,
# frame_margin 1.12 -> footprint ~286x214 mm, default overlap 0.30 -> nominal
# spacing ~200x150 mm) -- comfortably >= 4 tiles per axis regardless of a few
# mm of survey measurement noise, and regardless of which physical edge
# order_corners_clockwise happens to label "x" vs "y" for the recovered
# rectangle (that labelling is not under this test's control -- see below).
_FP_GRID_CORNERS = np.array([[300.0, 400.0, 0.0], [1800.0, 400.0, 0.0],
                             [1800.0, 1550.0, 0.0], [300.0, 1550.0, 0.0]])


def test_five_position_tiled_tour_contiguous_hole_caught_even_when_fraction_passes():
    """Finding 2 (post-review): a CONTIGUOUS block of empty tiles is a single,
    real, unscanned hole in the surface — unlike the same COUNT of misses
    scattered across the grid (mostly absorbed by neighbouring tiles' own
    overlap), it slips right past the tile-completeness FRACTION gate alone.
    Drop a 2x2 block (4 tiles, comfortably above the reported case's own
    9-of-64 in relative terms is not required — 4 tiles is already > the
    survey_tour_max_contiguous_empty_tiles=2 default and, on this grid,
    leaves the fraction comfortably clear of 0.85): the fraction alone PASSES
    — but the scan must still be refused by the new largest-contiguous-empty-
    block gate, and the refusal message must name which tiles are empty (the
    brief's own complaint: it used to report only a count)."""
    from tasni.modules.scan.planner import _tile_grid_dims, plan_rect_tour

    services, state = _build_fakes()
    scfg = services.config.scan
    # Narrow the cone for JUST this test so the reachability filter below (a
    # geometric "nearest tile centre" classifier, since a real is_reachable()
    # has no notion of "which tile") cannot misclassify a wide-cone candidate
    # into the wrong tile — a pure test-robustness knob, independent of (and
    # not needed by) the tiling math itself.
    scfg.flat_cone_deg = 8.0
    locked = _locked_five_position_scan_surface(state, _FP_GRID_CORNERS)
    rec = locked.survey_record
    K = services.config.camera.K
    W, H = services.config.camera.size

    plan = plan_rect_tour(rec.corners_np(), np.asarray(rec.plane_normal_base), K, (W, H), scfg)
    # nx/ny come from _tile_grid_dims itself -- order_corners_clockwise (Task
    # 11) re-derives the recovered rectangle's own corner order from the
    # fitted geometry, not from _FP_GRID_CORNERS' row order, so which physical
    # edge ends up "x" vs "y" (and therefore the exact nx/ny split) is not
    # under this test's control; only that BOTH axes have >= 4 tiles is.
    nx, ny, _fw, _fh = _tile_grid_dims(rec.corners_np(), K, (W, H), scfg)
    assert nx >= 4 and ny >= 4, (nx, ny)
    assert len(plan.aims) == nx * ny

    all_centers = np.array([a.point_base_mm for a in plan.aims])[:, :2]
    bi, bj = nx // 2 - 1, ny // 2 - 1
    block_lin = {(bi + di) * ny + (bj + dj) for di in range(2) for dj in range(2)}
    assert len(block_lin) == 4
    expected_fraction = 1.0 - 4 / (nx * ny)
    assert expected_fraction >= scfg.min_surface_coverage, expected_fraction  # sanity

    def is_reachable(T):
        p = np.asarray(T)[:2, 3]
        nearest = int(np.argmin(np.linalg.norm(all_centers - p, axis=1)))
        return nearest not in block_lin

    real_is_reachable = services.rdk.is_reachable
    services.rdk.is_reachable = is_reachable
    try:
        with pytest.raises(RuntimeError) as exc:
            scan_service.generate_scan_targets(services, locked)
        msg = str(exc.value)
        assert "contiguous" in msg.lower(), msg
        for t in sorted(block_lin):
            assert f"T{t + 1:02d}" in msg, (t, msg)

        # Proves the OLD fraction-only gate would have silently PASSED this
        # exact scenario: disabling just the new block check (a large
        # allowance) must now succeed, with the reported fraction confirming
        # it clears min_surface_coverage (0.85) on its own.
        scfg.survey_tour_max_contiguous_empty_tiles = 1000
        gen = scan_service.generate_scan_targets(services, locked)
        assert gen["surface_coverage"] >= scfg.min_surface_coverage, gen["surface_coverage"]
        assert gen["empty_tile_count"] == 4, gen["empty_tile_count"]
        assert gen["largest_contiguous_empty_tiles"] == 4, gen["largest_contiguous_empty_tiles"]
        # Follow-up 5 (post-review): with 4 whole tiles missing, "planned"
        # (every tile's own n_views, ignoring reachability) and "created"
        # (what actually landed in RoboDK) must now DIFFER -- this is exactly
        # the case that would have caught the old bug (planned_views silently
        # aliasing len(created), which cannot distinguish the two by
        # definition).
        planned = sum(a["n_views"] for a in gen["plan"]["aims"])
        assert gen["planned_views"] == planned, (gen["planned_views"], planned)
        assert gen["created"] < gen["planned_views"], gen
        print(f"[contiguous hole] {nx}x{ny} grid; fraction={gen['surface_coverage']:.3f} "
              f"(>= {scfg.min_surface_coverage}) would have PASSED alone; the contiguous "
              f"block gate correctly caught the 4-tile hole "
              f"(largest={gen['largest_contiguous_empty_tiles']}); planned "
              f"{gen['planned_views']} vs created {gen['created']}")
    finally:
        services.rdk.is_reachable = real_is_reachable


def test_five_position_tiled_tour_small_rectangle_is_single_tile():
    """A rectangle small enough to fit this fake camera's own footprint still
    goes through the tiled-tour branch (mode=five_position), collapsing to a
    single tile -- the branch must not require multiple tiles to work.

    Sized well clear of the ~200x150 mm single/double-tile boundary at this
    fake's K/config (not right AT it), so the survey's own measurement noise
    (a few mm, like any real fit) cannot tip it into a second tile.
    """
    services, state = _build_fakes()
    small = np.array([[940.0, 730.0, 0.0], [1060.0, 730.0, 0.0],
                      [1060.0, 820.0, 0.0], [940.0, 820.0, 0.0]])
    locked = _locked_five_position_scan_surface(state, small)

    gen = scan_service.generate_scan_targets(services, locked)
    assert gen["tile_count"] == 1, gen["tile_count"]
    assert gen["created"] > 0, gen
    for name in gen["targets"]:
        assert name.startswith("TasniScan_T01_"), name
    print("[five-position tiled tour] small rectangle -> single tile,",
          gen["created"], "targets")


# -- Task 13: survey capture orchestration + REST routes --------------------
# five_position_capture() is the ORCHESTRATION that connects the pure Task 11
# state machine + Task 10 corner extractor to real hardware: one authoritative
# step-and-measure acquisition per operator position, reusing the exact same
# camera-hold/grab/fuse/robot-refresh sequence lock_scan_surface() uses (now
# factored out as _authoritative_acquisition, shared by both). The brief's own
# Step-1 test snippet assumed an illustrative `scan_services` fixture and a
# JobEvent shape (`.kind`/`.data`) that don't exist in this file/codebase --
# adapted below to the real `_build_fakes()` fixture and JobEvent's actual
# fields (`.type`/`.payload`), keeping the assertions semantically identical.

def test_five_position_capture_uses_fresh_robot_state():
    """One authoritative five-position capture (the initial "center" step)
    against the real synthetic-table fake camera: must succeed, record a
    genuine (fresh, double-read) robot-state snapshot, advance the survey
    machine to "corner1", and publish a "survey" JobEvent carrying the same
    state -- the exact contract the guided UI polls / listens for."""
    services, _state = _build_fakes()
    events = []
    services.bus = SimpleNamespace(publish=lambda e: events.append(e))
    survey = FivePositionSurvey(services.config.scan)

    state = scan_service.five_position_capture(services, survey)

    assert state["step"] == "corner1"          # center accepted, machine advanced
    assert survey._accepted["center"].record.robot.stationary is True
    assert survey._accepted["center"].evidence is None  # no corner evidence for "center"
    # Bundled minor (fix round): an earlier review found measurement_ts silently
    # pinned to 0.0 on the sibling (compact-lock) path when it read a key that
    # didn't exist; pin it here too rather than assume the SimpleNamespace/
    # timestamp plumbing (_authoritative_acquisition's returned `frame`) is right.
    assert survey._accepted["center"].record.measurement_ts == FRAME_TIMESTAMP
    assert survey._accepted["center"].record.measurement_ts != 0.0
    survey_events = [e for e in events if e.type == "survey"]
    assert survey_events and survey_events[-1].payload["step"] == "corner1"
    print("[five-position capture] center accepted -> step corner1; JobEvent published")


def test_five_position_capture_rejects_disconnected_driver():
    """Review Finding 2: refresh_robot_state's "stationary" read only means the
    real arm didn't move ACCORDING TO ROBODK'S OWN MODEL -- with the driver
    down (or never linked), that model freezes at its last commanded pose
    while the physical arm can be anywhere, so a dead link reads as a
    perfectly "stationary" robot every time. A five-position survey is
    uniquely exposed: it depends on the pose genuinely differing across five
    captures, so a frozen mirror would silently register all five at the same
    fictional pose. Checked BEFORE the (multi-second, Wi-Fi) camera grab, so
    this must fail fast -- verified below via the fake camera's own grab
    counter, which must stay at zero."""
    services, _state = _build_fakes()
    services.rdk.robot_connected = lambda: (False, "ROBOTCOM_DISCONNECTED")
    survey = FivePositionSurvey(services.config.scan)

    try:
        scan_service.five_position_capture(services, survey)
        raise AssertionError("expected a disconnected driver to be rejected")
    except RuntimeError as e:
        assert "not connected" in str(e) or "driver" in str(e), e
    assert survey.step == "center"                  # never reached add_capture
    assert services.camera.grabs == 0, (             # failed BEFORE the slow grab
        "driver-liveness check must run before the camera grab, not after")
    print("[five-position capture] disconnected driver rejected before any camera grab")


def test_five_position_capture_rejects_moving_robot():
    """Core safety contract (Task 13): a robot that MOVED between the two
    refresh_robot_state reads must be rejected immediately -- before the
    capture is ever handed to the survey state machine -- not merely
    recorded and silently accepted. This is stricter than the compact lock
    (which still records robot.stationary=False without refusing) because a
    five-position survey combines FIVE separately-registered positions, so a
    stale/live pose blend at any one of them would corrupt the cross-position
    plane/rectangle fit in a way no downstream check could distinguish from a
    real geometry error.

    Review Finding 4: five_position_capture's own rejection and
    FivePositionSurvey.add_capture's backstop rejection used to raise
    byte-identical strings, so a test asserting only on message text could not
    tell WHICH layer fired -- deleting the early orchestration-layer check
    would have left a same-looking test green. Made load-bearing two ways:
    (1) the two messages are now worded differently (checked below), and (2)
    add_capture is spied on and asserted NEVER CALLED -- proving the rejection
    happens before a CaptureRecord is even built, not merely inside the state
    machine's own defence-in-depth check."""
    services, _state = _build_fakes()
    survey = FivePositionSurvey(services.config.scan)

    calls = {"n": 0}

    def moving_joints():
        calls["n"] += 1
        return [0.0, 0.0, 0.0, 0.0, 0.0, float(calls["n"])]   # different every read

    services.rdk.current_joints = moving_joints
    add_capture_calls = []
    orig_add_capture = survey.add_capture
    survey.add_capture = lambda *a, **k: add_capture_calls.append((a, k)) or orig_add_capture(*a, **k)
    try:
        scan_service.five_position_capture(services, survey)
        raise AssertionError("expected a moving robot to be rejected")
    except RuntimeError as e:
        msg = str(e)
        assert "moved" in msg, e                      # five_position_capture's own wording
        assert "was moving during the capture" not in msg, (  # NOT add_capture's wording
            "message matches add_capture's backstop text -- cannot tell which layer fired")
    assert add_capture_calls == [], "add_capture must never be reached for a moving robot"
    assert survey.step == "center"       # rejected capture must not have been accepted
    print("[five-position capture] moving robot correctly rejected before add_capture was ever called")


# -- Review Finding 1: _deproject_plane_points_mm must filter to PLANE
# INLIERS, not every valid depth pixel. A corner capture aims the camera at a
# table corner (corner_hint_uv=(0.5, 0.5)), so on real hardware a meaningful
# fraction of the frame legitimately looks PAST the table's two near edges at
# background (floor, fixtures) far off the work plane. _render() (the fixture
# used everywhere else in this file) returns 0 depth outside the synthetic
# table, which cannot exercise this at all -- these fixtures render a REAL
# floor plane 750 mm below the table so the off-plane background returns
# real, valid depth, exactly like a D435i pointed at a table near a floor.
_FLOOR_Z_MM = -750.0
# The four physical table corners (matching the 300x300 mm table _render()
# already uses) -- a five-position survey visits these in order.
_FIVE_POS_WORLD_CORNERS = {
    "corner1": (150.0, 150.0), "corner2": (150.0, -150.0),
    "corner3": (-150.0, -150.0), "corner4": (-150.0, 150.0),
}


def _render_with_floor(T_base_cam, *, table_z=0.0, floor_z=_FLOOR_Z_MM):
    """Like _render(), but rays that miss the (table_z-height) table hit a
    real floor plane floor_z below instead of returning 0 -- so a corner
    capture's off-plane background is real, valid depth, not silence."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    dirs_cam = np.stack([(us - cx) / fx, (vs - cy) / fy, np.ones_like(us, float)], -1)
    R, t = T_base_cam[:3, :3], T_base_cam[:3, 3]
    dirs_base = dirs_cam @ R.T
    dz = dirs_base[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        s_table = (table_z - t[2]) / dz
    P_table = t + s_table[..., None] * dirs_base
    on_table = ((np.abs(P_table[..., 0]) <= TABLE_HALF_MM) & (np.abs(P_table[..., 1]) <= TABLE_HALF_MM)
                & (s_table > 0) & np.isfinite(s_table))
    with np.errstate(divide="ignore", invalid="ignore"):
        s_floor = (floor_z - t[2]) / dz
    valid_floor = (s_floor > 0) & np.isfinite(s_floor)
    depth = np.where(on_table, s_table, np.where(valid_floor, s_floor, 0)).astype(np.uint16)
    color = np.full((H, W, 3), 128, np.uint8)
    return SimpleNamespace(color=color, depth=depth, timestamp=FRAME_TIMESTAMP,
                            geometry=gf.aligned(K, (W, H)))


def _corner_polygon_uv(cx, cy, arm=0.3):
    """A minimal open/closed 3-vertex corner polygon (uv) whose two axis-
    aligned arms stay inside the TABLE quadrant as seen by a camera centred
    directly above world corner (cx, cy) and looking straight down -- matches
    this file's _look_at basis (image u = world Y, image v = world X, derived
    and verified by direct ray-tracing against _render_with_floor): stepping
    INTO the table from image-centre (where the reticle sits, matching the
    corner_hint_uv=(0.5, 0.5) five_position_capture always uses) is -u when
    cy > 0 else +u, and -v when cx > 0 else +v."""
    u_dir = -1.0 if cy > 0 else 1.0
    v_dir = -1.0 if cx > 0 else 1.0
    u0, v0 = 0.5, 0.5
    p_minus = (u0 + u_dir * arm, v0)
    p_plus = (u0, v0 + v_dir * arm)
    return [p_minus, (u0, v0), p_plus]


def _render_table_only(T_base_cam, *, table_z=0.0):
    """Like _render(), but the table sits at table_z instead of always 0 --
    used only to compute a ground-truth reference plane (see
    _true_table_plane_cam), never as a capture's actual depth frame."""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    dirs_cam = np.stack([(us - cx) / fx, (vs - cy) / fy, np.ones_like(us, float)], -1)
    R, t = T_base_cam[:3, :3], T_base_cam[:3, 3]
    dirs_base = dirs_cam @ R.T
    dz = dirs_base[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (table_z - t[2]) / dz
    P = t + s[..., None] * dirs_base
    valid = ((np.abs(P[..., 0]) <= TABLE_HALF_MM) & (np.abs(P[..., 1]) <= TABLE_HALF_MM)
             & (s > 0) & np.isfinite(s))
    return np.where(valid, s, 0).astype(np.uint16)


def _true_table_plane_cam(cx, cy, *, table_z=0.0):
    """The REAL table plane (camera frame), computed from a table-only render
    at the same pose -- ground truth, independent of whether RANSAC would
    itself pick the table over a majority-background frame (a separate
    question from Finding 1's -- see the "isolates" note on
    _patch_boundary_and_plane_for_five_position_background below)."""
    from tasni.modules.scan.survey import SurveyThresholds, survey_surface as real_survey_surface

    T = _look_at((cx, cy, 420.0), (cx, cy, table_z))
    depth_clean = _render_table_only(T, table_z=table_z)
    th = SurveyThresholds(accurate_min_mm=300.0, accurate_max_mm=800.0, survey_max_tilt_deg=6.0,
                          grid_target_px=64, frame_margin_uv=0.02, work_crop_mm=(1000.0, 1000.0),
                          min_valid_depth_frac=0.05)
    m = real_survey_surface(depth_clean, gf.aligned(K, (W, H)), K, None, th)
    assert m.detected, "table-only reference render must be detected"
    return m.normal_cam, m.centroid_cam_mm, m.standoff_mm, m.tilt_deg


def _render_zero_background(T_base_cam, *, table_z=0.0):
    """Like _render_with_floor, but the background is SILENT (0 depth) rather
    than a real off-plane surface -- the D435i pointed past the table's edge
    into open space beyond its reliable range, not at a nearby floor. Same
    SimpleNamespace(color, depth, timestamp) shape as every other render
    helper in this file."""
    depth = _render_table_only(T_base_cam, table_z=table_z)
    color = np.full((H, W, 3), 128, np.uint8)
    return SimpleNamespace(color=color, depth=depth, timestamp=FRAME_TIMESTAMP,
                            geometry=gf.aligned(K, (W, H)))


def _build_fakes_five_position_background(table_z_by_kind=None, *, background="floor"):
    """A five-position-survey fake harness whose pose depends on
    state["cur_kind"] (set by the caller before each capture) -- "center"
    looks straight down at the table centre; "cornerN" looks straight down
    at that physical corner, matching how the guided UI instructs the
    operator to centre the reticle on each position in turn.

    ``background`` selects which of Finding 1's two failure modes the camera
    renders beyond the table edges: ``"floor"`` (default) is a REAL off-plane
    surface (a floor 750 mm below); ``"zero"`` is silence (0 depth, open
    space beyond the D435i's reliable range) -- the mode remedy (ii)
    (corner-appropriate valid_frac) specifically targets, since that mode is
    invisible to the coarse centre-patch metric's replacement unless the
    fixture actually renders it.
    """
    table_z_by_kind = table_z_by_kind or {}
    render = _render_with_floor if background == "floor" else _render_zero_background
    state = {"cam": _look_at((0.0, 0.0, 420.0), (0.0, 0.0, 0.0)), "cur_kind": "center"}
    mount = Rt_to_T(np.eye(3), np.array([40.0, -15.0, 55.0]))

    def _pose_for(kind):
        cx, cy = (0.0, 0.0) if kind == "center" else _FIVE_POS_WORLD_CORNERS[kind]
        return _look_at((cx, cy, 420.0), (cx, cy, 0.0)), cx, cy

    class FakeRdk:
        def item_exists(self, name): return True
        def use_camera_tool(self, tool): return mount
        def camera_pose_T(self): return state["cam"]
        def current_joints(self): return [0.0] * 6
        def robot_connected(self): return True, "ROBOTCOM_READY"

    class FakeCamera:
        def __init__(self): self.grabs = 0
        def grab(self, with_depth=False, timeout=None, color_only=False):
            self.grabs += 1
            T, _cx, _cy = _pose_for(state["cur_kind"])
            state["cam"] = T
            tz = table_z_by_kind.get(state["cur_kind"], 0.0)
            return render(T, table_z=tz)

    cfg = AppConfig()
    cfg.camera.intrinsics = {"320x240": K.tolist()}
    cfg.camera.resolution = "320x240"
    services = SimpleNamespace(config=cfg, rdk=FakeRdk(), camera=FakeCamera(),
                               camera_lease=CameraLease(),
                               bus=SimpleNamespace(publish=lambda *a, **k: None),
                               live=SimpleNamespace(running=False, stop=lambda: None))
    return services, state


def _patch_boundary_and_plane_for_five_position_background(table_z_by_kind):
    """Context-manager-free patch/unpatch pair for the two module-level names
    five_position_capture() looks up (scan_service.survey_surface,
    scan_service.color_work_boundary), used by both Finding-1 pipeline tests.

    The survey_surface patch delegates to the REAL function and only
    overrides the plane fields (normal/centroid/standoff/tilt) with the known-
    true table plane -- it does NOT fabricate detected/valid_frac, both of
    which come from the real function on the real (floor-containing) depth.
    This deliberately ISOLATES Finding 1's actual mechanism (does the
    DEPROJECTION correctly filter background once a plane is known) from a
    separate, real, but out-of-scope question this fixture surfaced during
    development: an operator centring the reticle EXACTLY on a 90-degree
    table corner gets AT MOST 25% real table coverage (an exact geometric
    ceiling -- image-centre lines always quarter a rectangular frame), so
    RANSAC's own plane vote would favour a majority background over the
    table. That is a characteristic of survey_surface's plane SELECTION,
    pre-existing to Task 13 and not touched by this fix -- flagged in the fix
    report, not silently worked around here.
    """
    orig_survey_surface = scan_service.survey_surface
    orig_color_boundary = scan_service.color_work_boundary

    def patched_survey_surface(depth, geometry, K_, dist_, th, **kw):
        real = orig_survey_surface(depth, geometry, K_, dist_, th, **kw)
        if not real.detected:
            return real
        import dataclasses
        kind = patched_survey_surface.cur
        cx, cy = (0.0, 0.0) if kind == "center" else _FIVE_POS_WORLD_CORNERS[kind]
        tz = table_z_by_kind.get(kind, 0.0)
        normal, centroid, standoff, tilt = _true_table_plane_cam(cx, cy, table_z=tz)
        return dataclasses.replace(real, normal_cam=normal, centroid_cam_mm=centroid,
                                   standoff_mm=standoff, tilt_deg=tilt)
    patched_survey_surface.cur = "center"

    def patched_color_boundary(color, **kw):
        kind = patched_survey_surface.cur
        if kind not in _FIVE_POS_WORLD_CORNERS:
            return None
        cx, cy = _FIVE_POS_WORLD_CORNERS[kind]
        poly = _corner_polygon_uv(cx, cy)
        return {"outline_uv": poly, "polygon_uv": poly, "fill_frac": 0.1,
                "border_touch": 0.0, "overruns": False, "contrast": 50.0}

    scan_service.survey_surface = patched_survey_surface
    scan_service.color_work_boundary = patched_color_boundary

    def unpatch():
        scan_service.survey_surface = orig_survey_surface
        scan_service.color_work_boundary = orig_color_boundary

    return patched_survey_surface, unpatch


def test_deproject_plane_points_mm_filters_background_to_plane_inliers():
    """The exact before/after measurement the review asked for: on a
    synthetic corner capture with a REAL floor 750 mm below the table (25%
    table / 75% floor -- an operator centred exactly on the corner), the
    UNFILTERED deprojection (this function's original behaviour, reproduced
    here via an effectively-infinite band) mixes in the floor and inflates
    fit_global_plane's per-set RMS to ~384 mm against the 8 mm
    survey_coplanar_reject_mm gate -- i.e. it would refuse every real corner
    capture as "not coplanar." Filtering to a tight band around the plane
    survey_surface already fit for this exact frame (this function's fixed
    behaviour) collapses that to ~0 mm (only floating-point residue from the
    synthetic ray-plane intersection remains)."""
    cx, cy = 150.0, 150.0
    T = _look_at((cx, cy, 420.0), (cx, cy, 0.0))
    normal_cam, centroid_cam_mm, standoff_mm, _tilt = _true_table_plane_cam(cx, cy)
    assert abs(standoff_mm - 420.0) < 1.0, standoff_mm     # sanity: the real table plane

    depth_bg = _render_with_floor(T).depth
    bg_frac = float(np.mean(depth_bg > 1000))
    assert bg_frac > 0.7, bg_frac      # a genuinely majority-background frame

    geom = gf.aligned(K, (W, H))
    pts_before, purity_before, coverage_before = scan_service._deproject_plane_points_mm(
        depth_bg, geom, T, plane_normal_cam=normal_cam, plane_point_cam=centroid_cam_mm,
        band_mm=1.0e6)                 # effectively unfiltered -- reproduces the pre-fix behaviour
    pts_after, purity_after, coverage_after = scan_service._deproject_plane_points_mm(
        depth_bg, geom, T, plane_normal_cam=normal_cam, plane_point_cam=centroid_cam_mm,
        band_mm=6.0)                   # the scan.survey_plane_inlier_band_mm default

    from tasni.modules.scan.rect_fit import fit_global_plane
    rms_before = fit_global_plane([pts_before]).per_set_rms_mm[0]
    rms_after = fit_global_plane([pts_after]).per_set_rms_mm[0]

    assert rms_before > 300.0, rms_before      # measured 384.26 mm
    assert rms_after < 1.0, rms_after          # measured 0.00 mm
    assert len(pts_after) < len(pts_before)    # background points were actually dropped
    assert np.all(np.abs(pts_after[:, 2]) < 1.0), "filtered points must be on the table (z~0), not the floor"
    # purity/coverage after the fix: with band_mm=1e6 every valid pixel survives, so
    # purity_before is trivially 1.0 (not informative -- band=1e6 disables filtering
    # entirely); the AFTER values are the real per-corner-step metrics five_position_
    # capture now uses (remedy ii) -- with a REAL majority-background floor filling
    # the frame, purity/coverage are both ~26%, correctly reflecting how much of this
    # particular capture is genuinely trustworthy table.
    assert purity_before == 1.0, purity_before
    assert 0.2 < purity_after < 0.3, purity_after
    assert 0.2 < coverage_after < 0.3, coverage_after
    print(f"[Finding 1] bg_frac={bg_frac:.3f} n_before={len(pts_before)} rms_before={rms_before:.2f} mm "
          f"-> n_after={len(pts_after)} rms_after={rms_after:.4f} mm; "
          f"purity_after={purity_after:.3f} coverage_after={coverage_after:.3f}")


def test_five_position_survey_accepts_captures_with_zero_background_depth():
    """Finding 1 remedy (ii)'s end-to-end positive proof: a full five-position
    survey (center + 4 corners) driven through the REAL five_position_capture,
    with SILENT (0 depth) background beyond every corner's near edges -- the
    D435i pointed past the table into open space beyond its reliable range,
    not at a nearby floor -- is ACCEPTED at every step and finish() succeeds.

    This is exactly the scenario the coarse centre-patch metric got wrong
    (mode (b) of the original Finding 1): the reticle straddles the table/
    background boundary by design at a corner, so a patch centred there sees
    a lot of 0-depth void even on a perfectly good capture, and would have
    been spuriously rejected as "not enough valid depth." The plane-inlier
    PURITY metric (survivors of the band filter / all pixels with ANY valid
    depth) is robust to this: with nothing else providing depth, virtually
    every valid pixel IS on the table, so purity reads near 1.0 regardless of
    how little of the FRAME the table fills.

    Verified against the code before this fix: with CaptureRecord.valid_frac
    still fed the CENTRE-PATCH metric for corner steps, this exact fixture
    fails at add_capture with "not enough valid depth in the capture" (the
    centre 25% patch, centred on the corner, is itself mostly the 0-depth
    void) -- reproduced by temporarily reverting five_position_capture's
    valid_frac line back to `float(reading.valid_frac)` and re-running this
    test, which then fails exactly that way."""
    services, state = _build_fakes_five_position_background(background="zero")
    services.config.scan.boundary_engine = "color"
    survey = FivePositionSurvey(services.config.scan)
    patched, unpatch = _patch_boundary_and_plane_for_five_position_background({})
    try:
        for kind in ("center", "corner1", "corner2", "corner3", "corner4"):
            state["cur_kind"] = kind
            patched.cur = kind
            st = scan_service.five_position_capture(services, survey)
            pts = survey._accepted[kind].plane_points_base
            assert np.all(np.abs(pts[:, 2]) < 1.0), (kind, pts[:, 2].min(), pts[:, 2].max())
            if kind != "center":
                purity = survey._accepted[kind].record.valid_frac
                assert purity > 0.9, (kind, purity)   # near-pure: nothing else provides depth
        assert survey.step == "review"
        record = survey.finish(calibration_id="cam-test",
                               locked_robot=scan_service.refresh_robot_state(services.rdk))
        assert record.mode == MODE_FIVE_POSITION
        assert max(record.quality["per_position_rms_mm"]) < 1.0, record.quality
    finally:
        unpatch()
    print("[Finding 1 remedy ii] full 5-position survey with SILENT background -> "
          "accepted (purity ~1.0 at every corner) + finish() succeeded")


def test_five_position_survey_accepts_well_aimed_corner_with_real_background():
    """Finding 1 remedy (ii) round-3 ruling: a corner capture whose reticle sits
    EXACTLY on the table corner -- the BEST POSSIBLE aim -- against a REAL,
    coherent off-plane surface (a floor 750 mm below, filling ~74% of the
    frame; only ~26.25% is genuinely trustworthy table) MUST BE ACCEPTED.
    Aiming at corners with real background in frame (a floor, fixtures, a
    wall within D435i range) is the entire premise of the five-position
    path -- rejecting the geometric best case a corner shot can ever achieve
    would make the feature unusable on any real cell.

    Round 2 fed corner steps' CaptureRecord.valid_frac the new plane-inlier
    PURITY metric but left it gated by five_position.py's SHARED
    min_valid_depth_frac (0.5): since purity for a well-aimed corner with a
    real background is the true table fraction (~26%), and 26% < 0.5 always,
    that left EVERY such capture rejected regardless of aim quality -- the
    bug this test now pins. five_position.py now gates corner steps against
    survey_corner_min_plane_coverage_frac (0.10) instead of the shared 0.5,
    so ~26% clears it with room to spare.

    Table fraction produced by this fixture: 26.25% (567 of 2160 stride-grid
    points; see test_deproject_plane_points_mm_filters_background_to_plane_inliers,
    which measures the identical geometry directly)."""
    services, state = _build_fakes_five_position_background(background="floor")
    services.config.scan.boundary_engine = "color"
    survey = FivePositionSurvey(services.config.scan)
    patched, unpatch = _patch_boundary_and_plane_for_five_position_background({})
    try:
        state["cur_kind"] = "center"
        patched.cur = "center"
        scan_service.five_position_capture(services, survey)   # center: unaffected, accepted
        assert survey.step == "corner1"

        state["cur_kind"] = "corner1"
        patched.cur = "corner1"
        st = scan_service.five_position_capture(services, survey)
        assert st["step"] == "corner2"
        purity = survey._accepted["corner1"].record.valid_frac
        assert 0.2 < purity < 0.3, purity   # the true ~26% table fraction, not a fabricated pass
    finally:
        unpatch()
    print(f"[Finding 1 remedy ii, round 3] well-aimed corner + real background "
          f"(purity={purity:.3f}, ~26% table) -> ACCEPTED")


def test_five_position_capture_rejects_genuinely_bad_aim_with_real_background():
    """The other half of the round-3 ruling's regression pin: a corner capture
    with a REAL background must still be REJECTED when the aim is genuinely
    bad -- well below the ~25% geometric ceiling a well-aimed shot achieves,
    not merely "a real background is present." Shrinks the table to the SAME
    80x80 mm sliver the zero-background coverage-gate test uses, but pairs it
    with the REAL floor fixture instead of silence: with a real background,
    purity and coverage are numerically the same quantity (both denominators
    equal when the background returns real depth), so BOTH the coverage gate
    (five_position_capture, checked first) and the now-step-aware purity gate
    (five_position.add_capture) would independently catch this -- verified
    (see the fix report) that the coverage gate fires first in practice, with
    a message naming corner1 and "too little of the table is in view", proving
    the accept test above is not simply "the gate never fires."

    Table fraction produced by this fixture: 4.23% (3249 of 76800 pixels) --
    comfortably below both the 10% gate and the ~25% ceiling."""
    tiny_half_mm = 40.0
    cx, cy = _FIVE_POS_WORLD_CORNERS["corner1"]
    T = _look_at((cx, cy, 420.0), (cx, cy, 0.0))

    def render_tiny_with_floor(T_base_cam):
        fx, fy, ccx, ccy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        us, vs = np.meshgrid(np.arange(W), np.arange(H))
        dirs_cam = np.stack([(us - ccx) / fx, (vs - ccy) / fy, np.ones_like(us, float)], -1)
        R, t = T_base_cam[:3, :3], T_base_cam[:3, 3]
        dirs_base = dirs_cam @ R.T
        dz = dirs_base[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            s_table = (0.0 - t[2]) / dz
        P_table = t + s_table[..., None] * dirs_base
        on_table = ((np.abs(P_table[..., 0] - cx) <= tiny_half_mm)
                    & (np.abs(P_table[..., 1] - cy) <= tiny_half_mm)
                    & (s_table > 0) & np.isfinite(s_table))
        with np.errstate(divide="ignore", invalid="ignore"):
            s_floor = (_FLOOR_Z_MM - t[2]) / dz
        valid_floor = (s_floor > 0) & np.isfinite(s_floor)
        depth = np.where(on_table, s_table, np.where(valid_floor, s_floor, 0)).astype(np.uint16)
        color = np.full((H, W, 3), 128, np.uint8)
        return SimpleNamespace(color=color, depth=depth, timestamp=FRAME_TIMESTAMP,
                               geometry=gf.aligned(K, (W, H)))

    services, state = _build_fakes_five_position_background(background="floor")
    real_grab = services.camera.grab

    def grab_bad_aim_for_corner1(*a, **kw):
        if state["cur_kind"] == "corner1":
            state["cam"] = T
            return render_tiny_with_floor(T)
        return real_grab(*a, **kw)

    services.camera.grab = grab_bad_aim_for_corner1
    services.config.scan.boundary_engine = "color"
    survey = FivePositionSurvey(services.config.scan)
    patched, unpatch = _patch_boundary_and_plane_for_five_position_background({})
    try:
        state["cur_kind"] = "center"
        patched.cur = "center"
        scan_service.five_position_capture(services, survey)
        assert survey.step == "corner1"

        state["cur_kind"] = "corner1"
        patched.cur = "corner1"
        try:
            scan_service.five_position_capture(services, survey)
            raise AssertionError("expected the genuinely bad-aim corner capture to be rejected")
        except RuntimeError as e:
            msg = str(e)
            # Actionable for the operator: names the step, the problem, and a fix --
            # not a bare "not enough valid depth" that doesn't say WHAT to do at a
            # corner specifically (round-3 ruling item 3).
            assert "corner1" in msg and "too little of the table is in view" in msg, msg
        assert survey.step == "corner1"   # rejected capture must not have been accepted
    finally:
        unpatch()
    print("[Finding 1 remedy ii, round 3] genuinely bad-aim corner (4.2% table) + "
          "real background -> correctly REJECTED with an actionable message")


def test_five_position_capture_rejects_tiny_table_sliver_via_coverage_gate():
    """Finding 1 remedy (ii)'s coverage-specific gate (the NEW corner-specific
    config key, scan.survey_corner_min_plane_coverage_frac): purity alone
    cannot catch "genuinely too little table in view" when the background is
    SILENT, since a tiny-but-clean sliver against a 0-depth void also reads
    purity ~1.0 -- exactly the blind spot the review flagged. Shrinks the
    table to a sliver far too small to be a usable corner capture (a fraction
    of the ~25% ceiling a well-aimed corner shot achieves) while keeping the
    background silent, and confirms coverage (not purity) is what catches it."""
    # 80x80 mm sliver, vs the normal 300x300 mm table: ~4.3% coverage -- above
    # five_position_capture's own small RANSAC-attempt sanity floor (2%, just
    # enough real depth to try a plane fit at all) but comfortably below
    # scan.survey_corner_min_plane_coverage_frac (0.10), so this pins the
    # COVERAGE gate specifically, not the earlier detection retry.
    tiny_half_mm = 40.0
    cx, cy = _FIVE_POS_WORLD_CORNERS["corner1"]
    T = _look_at((cx, cy, 420.0), (cx, cy, 0.0))

    def render_tiny(T_base_cam):
        fx, fy, ccx, ccy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        us, vs = np.meshgrid(np.arange(W), np.arange(H))
        dirs_cam = np.stack([(us - ccx) / fx, (vs - ccy) / fy, np.ones_like(us, float)], -1)
        R, t = T_base_cam[:3, :3], T_base_cam[:3, 3]
        dirs_base = dirs_cam @ R.T
        dz = dirs_base[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (0.0 - t[2]) / dz
        P = t + s[..., None] * dirs_base
        valid = ((np.abs(P[..., 0] - cx) <= tiny_half_mm) & (np.abs(P[..., 1] - cy) <= tiny_half_mm)
                 & (s > 0) & np.isfinite(s))
        depth = np.where(valid, s, 0).astype(np.uint16)
        color = np.full((H, W, 3), 128, np.uint8)
        return SimpleNamespace(color=color, depth=depth, timestamp=FRAME_TIMESTAMP,
                               geometry=gf.aligned(K, (W, H)))

    # Unit-level sanity first: prove purity alone would be misled, coverage is not.
    tiny_frame = render_tiny(T)
    normal_cam, centroid_cam_mm, _standoff_mm, _tilt = _true_table_plane_cam(cx, cy)
    _pts, purity, coverage = scan_service._deproject_plane_points_mm(
        tiny_frame.depth, tiny_frame.geometry, T, plane_normal_cam=normal_cam,
        plane_point_cam=centroid_cam_mm, band_mm=6.0)
    assert purity > 0.9, purity          # the blind spot: silent bg -> purity looks fine
    assert coverage < 0.05, coverage     # but coverage correctly shows almost nothing is there

    # Now end to end: a normal "center" capture, then a corner1 whose camera grab is
    # swapped for the tiny-sliver render (everything else -- pose, driver, boundary
    # stub -- stays the real fixture machinery every other pipeline test uses).
    services, state = _build_fakes_five_position_background(background="zero")
    real_grab = services.camera.grab

    def grab_tiny_for_corner1(*a, **kw):
        if state["cur_kind"] == "corner1":
            state["cam"] = T
            return render_tiny(T)
        return real_grab(*a, **kw)

    services.camera.grab = grab_tiny_for_corner1
    services.config.scan.boundary_engine = "color"
    survey = FivePositionSurvey(services.config.scan)
    patched, unpatch = _patch_boundary_and_plane_for_five_position_background({})
    try:
        state["cur_kind"] = "center"
        patched.cur = "center"
        scan_service.five_position_capture(services, survey)
        assert survey.step == "corner1"

        state["cur_kind"] = "corner1"
        patched.cur = "corner1"
        try:
            scan_service.five_position_capture(services, survey)
            raise AssertionError("expected the tiny table sliver to be rejected by the coverage gate")
        except RuntimeError as e:
            assert "too little of the table is in view" in str(e), e
        assert survey.step == "corner1"    # rejected capture must not have been accepted
    finally:
        unpatch()
    print(f"[Finding 1 remedy ii] tiny table sliver: purity={purity:.3f} (looks fine) "
          f"coverage={coverage:.3f} (correctly low) -> coverage gate rejected it")


def test_five_position_survey_still_rejects_genuine_noncoplanarity_with_background():
    """Finding 1's end-to-end negative proof: the SAME pipeline must still
    REFUSE a genuinely non-coplanar surface -- proving the per-capture
    plane-inlier filter (which only removes THIS capture's own off-plane
    background) does not also defeat the SEPARATE cross-position coplanarity
    check (fit_global_plane's per-set RMS against the pooled global plane).
    Uses the SILENT-background fixture (background="zero") so every
    individual capture is accepted on its own merits (matching the
    "accepts" test above) -- corner4's table is rendered 50 mm above the
    other four positions', so each capture is still internally flat (passes
    add_capture on its own), but finish() must catch the cross-position
    discrepancy."""
    table_z_by_kind = {"corner4": 50.0}
    # BOTH the fake camera (the actual rendered depth) and the plane stub (what
    # survey_surface is told the true plane is) must agree on the shift, or the
    # deprojection band filter and the rendered depth describe two different
    # heights and every point gets filtered out as "off-plane" -- a fixture
    # bug, not a production one; caught by this test itself failing loudly
    # ("too few plane points") rather than silently mis-testing something else.
    services, state = _build_fakes_five_position_background(table_z_by_kind, background="zero")
    services.config.scan.boundary_engine = "color"
    survey = FivePositionSurvey(services.config.scan)
    patched, unpatch = _patch_boundary_and_plane_for_five_position_background(table_z_by_kind)
    try:
        for kind in ("center", "corner1", "corner2", "corner3", "corner4"):
            state["cur_kind"] = kind
            patched.cur = kind
            st = scan_service.five_position_capture(services, survey)     # each capture still accepted
            assert st["step"] != kind
        assert survey.step == "review"
        try:
            survey.finish(calibration_id="cam-test",
                          locked_robot=scan_service.refresh_robot_state(services.rdk))
            raise AssertionError("expected the shifted corner4 to be rejected as non-coplanar")
        except RuntimeError as e:
            assert "not coplanar" in str(e) and "corner4" in str(e), e
    finally:
        unpatch()
    print("[Finding 1] genuinely non-coplanar corner4 (+50 mm) still rejected through the real pipeline")


def test_deproject_plane_points_mm_regression_pin_against_pre_fix_behaviour():
    """Regression pin (verified against pre-fix code): with
    scan.survey_plane_inlier_band_mm set to reproduce the ORIGINAL unfiltered
    behaviour (an effectively-infinite band), the exact same genuinely flat,
    coplanar five-position survey from the accept test above is INCORRECTLY
    REJECTED by finish() -- proving this fixture reproduces the reported bug,
    not just a synthetic RMS number in isolation."""
    services, state = _build_fakes_five_position_background()
    services.config.scan.boundary_engine = "color"
    services.config.scan.survey_plane_inlier_band_mm = 1.0e6      # simulate the pre-fix (unfiltered) code
    survey = FivePositionSurvey(services.config.scan)
    patched, unpatch = _patch_boundary_and_plane_for_five_position_background({})
    try:
        for kind in ("center", "corner1", "corner2", "corner3", "corner4"):
            state["cur_kind"] = kind
            patched.cur = kind
            scan_service.five_position_capture(services, survey)
        assert survey.step == "review"
        try:
            survey.finish(calibration_id="cam-test",
                          locked_robot=scan_service.refresh_robot_state(services.rdk))
            raise AssertionError(
                "expected the pre-fix (unfiltered) band to reproduce the reported "
                "not-coplanar failure on a genuinely flat table")
        except RuntimeError as e:
            assert "not coplanar" in str(e), e
            print(f"[Finding 1 regression pin] pre-fix band reproduces the bug: {e}")
    finally:
        unpatch()


def test_five_position_capture_corner_step_passes_closed_true_to_corner_evidence():
    """CRITICAL per the Task 10/13 interface note: extract_corner_evidence's
    `closed` flag is NOT auto-detected, and every production boundary polygon
    (color_work_boundary / sam_work_boundary, both via mask_to_boundary's
    cv2.findContours) is a genuinely CLOSED contour. five_position_capture
    must pass closed=True for every corner step -- verified here by spying on
    the real call (not merely trusting the source reads correctly)."""
    services, _state = _build_fakes()
    services.config.scan.boundary_engine = "color"   # skip the SAM attempt entirely
    survey = FivePositionSurvey(services.config.scan)
    scan_service.five_position_capture(services, survey)   # center -> corner1
    assert survey.step == "corner1"

    # The fake's flat-grey colour frame has zero contrast, so the REAL
    # color_work_boundary would abstain (return None) here -- stub it so the
    # corner-evidence step actually runs, and spy on extract_corner_evidence
    # to capture exactly what it was called with.
    fake_polygon = [[0.5, 0.3], [0.7, 0.5], [0.5, 0.7], [0.3, 0.5]]
    orig_color_boundary = scan_service.color_work_boundary
    orig_extract = scan_service.extract_corner_evidence
    captured_kwargs = {}

    def fake_color_boundary(*_a, **_k):
        return {"outline_uv": fake_polygon, "polygon_uv": fake_polygon,
                "fill_frac": 0.1, "border_touch": 0.0, "overruns": False, "contrast": 50.0}

    def spy_extract(depth, K, polygon_uv, T_base_cam, **kwargs):
        captured_kwargs.update(kwargs)
        return CornerEvidence(corner_uv=(0.5, 0.5), corner_base_mm=(1000.0, 500.0, 0.0),
                              edge_points_base=np.zeros((25, 3)))

    scan_service.color_work_boundary = fake_color_boundary
    scan_service.extract_corner_evidence = spy_extract
    try:
        state = scan_service.five_position_capture(services, survey)
        assert state["step"] == "corner2"
    finally:
        scan_service.color_work_boundary = orig_color_boundary
        scan_service.extract_corner_evidence = orig_extract

    assert captured_kwargs.get("closed") is True, captured_kwargs
    assert captured_kwargs.get("corner_hint_uv") == (0.5, 0.5), captured_kwargs
    print("[five-position capture] corner boundary -> extract_corner_evidence(closed=True) confirmed")


def test_survey_begin_state_capture_cancel_routes():
    """The five-position guided-survey routes (Task 13), driven through the
    real bound methods -- same pattern as the surface_lock/poses_generate
    route tests above."""
    import tasni.modules.scan.module as scan_module

    services, _state, _started = _build_fakes_with_jobs()
    mod = scan_module.ScanModule(services)

    assert mod.survey_state() == {"step": None}     # inactive

    begun = mod.survey_begin()
    assert begun["step"] == "center"
    assert mod.survey_state()["step"] == "center"

    captured = mod.survey_capture()
    assert captured["step"] == "corner1"
    assert mod.survey_state()["step"] == "corner1"

    cancelled = mod.survey_cancel()
    assert cancelled == {"status": "cancelled"}
    assert mod.survey_state() == {"step": None}
    print("[survey routes] begin -> capture -> state -> cancel wired correctly")


def test_survey_routes_require_an_active_survey():
    import tasni.modules.scan.module as scan_module
    from fastapi import HTTPException

    services, _state, _started = _build_fakes_with_jobs()
    mod = scan_module.ScanModule(services)
    checks = (
        mod.survey_capture,
        lambda: mod.survey_recapture(scan_module.SurveyRecaptureBody(kind="center")),
        mod.survey_finish,
    )
    for call in checks:
        try:
            call()
            raise AssertionError("expected a 400 without an active survey")
        except HTTPException as e:
            assert e.status_code == 400, e.status_code
    print("[survey guards] capture/recapture/finish all refuse without an active survey")


def test_survey_recapture_route_rejects_unknown_kind():
    import tasni.modules.scan.module as scan_module
    from fastapi import HTTPException

    services, _state, _started = _build_fakes_with_jobs()
    mod = scan_module.ScanModule(services)
    mod.survey_begin()
    try:
        mod.survey_recapture(scan_module.SurveyRecaptureBody(kind="bogus"))
        raise AssertionError("expected an unknown recapture kind to be rejected")
    except HTTPException as e:
        assert e.status_code == 400

    # A known kind not yet captured is a harmless no-op (mirrors
    # FivePositionSurvey.recapture's own dict.pop(kind, None) semantics).
    out = mod.survey_recapture(scan_module.SurveyRecaptureBody(kind="corner1"))
    assert out["step"] == "center"
    print("[survey recapture] unknown kind -> 400; known-but-uncaptured kind -> no-op")


def test_survey_finish_route_locks_and_sets_current_lock_token():
    """POST /survey/finish (Task 13, ambiguity resolution #3): once a survey
    is complete, finishing it must store a LockedScanSurface with a FRESH
    lock_token AND set self._current_lock_token to that SAME token, mirroring
    surface_lock()'s own bookkeeping exactly -- otherwise poses_generate()'s
    Task 8 "targets predate the current lock" guard could not tell a
    five-position lock from no lock at all."""
    import tasni.modules.scan.module as scan_module

    services, state, _started = _build_fakes_with_jobs()
    mod = scan_module.ScanModule(services)
    # A fixed clock matching _fp_record's captured_at (100.0), same as
    # _five_position_locked_survey above -- survey_begin() (real time.monotonic)
    # would make every synthetic _fp_record instantly "stale".
    mod._five_survey = FivePositionSurvey(services.config.scan, clock=lambda: _FP_CLOCK[0])
    mod._five_survey.add_capture(
        _fp_record("center"), _fp_plane_points(_FP_LARGE_CORNERS.mean(axis=0)[:2]), None)
    for i in range(4):
        mod._five_survey.add_capture(
            _fp_record(f"corner{i + 1}"), _fp_plane_points(_FP_LARGE_CORNERS[i][:2], seed=i + 1),
            _fp_corner_evidence(_FP_LARGE_CORNERS, i, seed=i + 1))
    assert mod._five_survey.step == "review"

    out = mod.survey_finish()
    assert out["status"] == "locked"
    assert "discrepancy_mm" in out and "corner_agreement_mm" in out  # record.quality spread in

    locked = mod._locked_surface
    assert locked is not None and locked.survey_record is not None
    assert locked.survey_record.mode == MODE_FIVE_POSITION
    assert locked.lock_token != ""
    assert mod._current_lock_token == locked.lock_token             # ambiguity resolution #3
    assert locked.gate_payload["surface_mode"] == "five_position"
    assert locked.gate_payload["ok"] is True
    assert mod._five_survey is None                                 # consumed
    assert mod.survey_state() == {"step": None}
    print("[survey finish] locked with a fresh lock_token; self._current_lock_token mirrors it")


def test_survey_finish_route_rejects_incomplete_survey():
    import tasni.modules.scan.module as scan_module
    from fastapi import HTTPException

    services, _state, _started = _build_fakes_with_jobs()
    mod = scan_module.ScanModule(services)
    mod.survey_begin()
    try:
        mod.survey_finish()
        raise AssertionError("expected an incomplete survey to be rejected")
    except HTTPException as e:
        assert e.status_code == 400
        assert "incomplete" in str(e.detail), e.detail
    assert mod._five_survey is not None    # not consumed by a failed finish
    print("[survey finish] incomplete survey -> 400, survey left intact for the operator")


def test_survey_capture_unexpected_exception_surfaces_as_500_not_hardware_claim():
    """Task 19 (Defect 1b): survey_capture()'s except-chain used to map EVERY
    non-(RuntimeError|ValueError|KeyError) exception to a 503 "RoboDK/camera
    unavailable" -- exactly what happened for the real numpy/robomath.Mat
    TypeError (Defect 1a, see test_survey_contract.py): the operator saw a
    false hardware claim instead of the real bug, and the traceback was never
    logged, costing real diagnosis time at the cell. A genuinely unexpected
    exception must now surface as a 500 naming its real type, with the full
    traceback logged server-side (not swallowed)."""
    import logging as _logging
    import tasni.modules.scan.module as scan_module
    from fastapi import HTTPException

    services, _state, _started = _build_fakes_with_jobs()
    mod = scan_module.ScanModule(services)
    mod.survey_begin()

    def boom(*a, **k):
        raise TypeError("only size-1 arrays can be converted to Python scalars")
    orig = scan_module.five_position_capture
    scan_module.five_position_capture = boom

    records = []
    handler = _logging.Handler()
    handler.emit = lambda record: records.append(record)
    scan_module.log.addHandler(handler)
    try:
        try:
            mod.survey_capture()
            raise AssertionError("expected the unexpected exception to propagate")
        except HTTPException as e:
            assert e.status_code == 500, e.status_code
            assert "TypeError" in str(e.detail), e.detail
            assert "size-1 arrays" in str(e.detail), e.detail
            assert "unavailable" not in str(e.detail).lower(), e.detail
    finally:
        scan_module.five_position_capture = orig
        scan_module.log.removeHandler(handler)

    assert any(r.exc_info is not None for r in records), (
        "the full traceback must be logged server-side, not swallowed")
    print("[survey_capture] unexpected exception -> 500 with real type, traceback logged")


def test_survey_finish_unexpected_exception_surfaces_as_500_not_hardware_claim():
    """Task 19 (Defect 1b): the identical fix, applied to survey_finish()'s
    except-chain (it had the same 503 "RoboDK unavailable" catch-all)."""
    import logging as _logging
    import tasni.modules.scan.module as scan_module
    from fastapi import HTTPException

    services, state, _started = _build_fakes_with_jobs()
    mod = scan_module.ScanModule(services)
    mod._five_survey = FivePositionSurvey(services.config.scan, clock=lambda: _FP_CLOCK[0])
    mod._five_survey.add_capture(
        _fp_record("center"), _fp_plane_points(_FP_LARGE_CORNERS.mean(axis=0)[:2]), None)
    for i in range(4):
        mod._five_survey.add_capture(
            _fp_record(f"corner{i + 1}"), _fp_plane_points(_FP_LARGE_CORNERS[i][:2], seed=i + 1),
            _fp_corner_evidence(_FP_LARGE_CORNERS, i, seed=i + 1))
    assert mod._five_survey.step == "review"
    mod._five_survey.finish = lambda **k: (_ for _ in ()).throw(
        AttributeError("'NoneType' object has no attribute 'x'"))

    records = []
    handler = _logging.Handler()
    handler.emit = lambda record: records.append(record)
    scan_module.log.addHandler(handler)
    try:
        try:
            mod.survey_finish()
            raise AssertionError("expected the unexpected exception to propagate")
        except HTTPException as e:
            assert e.status_code == 500, e.status_code
            assert "AttributeError" in str(e.detail), e.detail
            assert "unavailable" not in str(e.detail).lower(), e.detail
    finally:
        scan_module.log.removeHandler(handler)

    assert any(r.exc_info is not None for r in records), (
        "the full traceback must be logged server-side, not swallowed")
    print("[survey_finish] unexpected exception -> 500 with real type, traceback logged")


def test_survey_recapture_unexpected_exception_surfaces_as_500_not_hardware_claim():
    """Task 19 (Defect 1b): survey_recapture() previously had NO catch-all at
    all (only ValueError -> 400), so an unexpected exception there would
    propagate as an unhandled 500 from FastAPI's default middleware with no
    scan-module traceback logged. Now behaves like survey_capture/finish."""
    import logging as _logging
    import tasni.modules.scan.module as scan_module
    from fastapi import HTTPException

    services, _state, _started = _build_fakes_with_jobs()
    mod = scan_module.ScanModule(services)
    mod.survey_begin()
    mod._five_survey.recapture = lambda kind: (_ for _ in ()).throw(TypeError("boom"))

    records = []
    handler = _logging.Handler()
    handler.emit = lambda record: records.append(record)
    scan_module.log.addHandler(handler)
    try:
        try:
            mod.survey_recapture(scan_module.SurveyRecaptureBody(kind="center"))
            raise AssertionError("expected the unexpected exception to propagate")
        except HTTPException as e:
            assert e.status_code == 500, e.status_code
            assert "TypeError" in str(e.detail), e.detail
    finally:
        scan_module.log.removeHandler(handler)

    assert any(r.exc_info is not None for r in records)
    print("[survey_recapture] unexpected exception -> 500 with real type, traceback logged")


class _FakeSamWorker:
    """Records .stop() instead of spinning a real background thread/ONNX
    session -- used by both Finding 3/5 tests below to prove a worker is
    stopped, not merely dereferenced."""
    def __init__(self, *a, **k): self.stopped = False
    def stop(self): self.stopped = True
    def submit(self, *a, **k): pass


class _FakeLive:
    """A services.live fake that actually implements .start()/.running --
    every OTHER fake in this file (_build_fakes's SimpleNamespace) has no
    .start() at all, which is exactly why the restart path had zero
    coverage (Finding 5): was_running was always False, so the restart
    branch never even executed."""
    def __init__(self): self.running = False; self.start_calls = 0
    def start(self, analyze, **kwargs): self.running = True; self.start_calls += 1
    def stop(self): self.running = False


def test_live_start_stops_existing_sam_worker_before_replacing_it():
    """Review Finding 3: _authoritative_acquisition stops the VIDEO loop for
    a capture but never touches self._sam_worker (unlike /live/stop), so the
    prior worker's daemon thread is orphaned -- never .stop()ped -- the
    moment live_start() is called again (as survey_capture()'s restart
    does). Reproduced directly here: calling the SAME /live/start closure
    twice in a row must stop the FIRST worker before building the second,
    not just drop the reference."""
    import tasni.modules.scan.module as scan_module

    services, _state, _started = _build_fakes_with_jobs()
    services.live = _FakeLive()
    services.config.scan.boundary_engine = "sam"   # forces a SamBoundaryWorker

    orig_worker_cls = scan_module.SamBoundaryWorker
    scan_module.SamBoundaryWorker = _FakeSamWorker
    try:
        mod = scan_module.ScanModule(services)
        mod.router()               # wires self._live_start (real app-startup path)
        mod._live_start()
        first = mod._sam_worker
        assert first is not None and not first.stopped

        services.live.stop()       # matches _authoritative_acquisition's own stop()
        mod._live_start()          # the restart survey_capture() performs
        assert first.stopped, "orphaned SamBoundaryWorker thread leaked across a restart"
        assert mod._sam_worker is not None and mod._sam_worker is not first
        assert not mod._sam_worker.stopped
    finally:
        scan_module.SamBoundaryWorker = orig_worker_cls
    print("[live restart] prior SamBoundaryWorker stopped before being replaced")


def test_survey_capture_restarts_live_preview_and_its_sam_worker_cleanly():
    """Review Finding 5: the /live/start restart path (survey_capture's
    finally block) had zero coverage -- every existing fake's services.live
    has no .start() at all, so was_running was always False and the branch
    never ran. Covers BOTH the restart itself AND its interaction with
    Finding 3's worker-leak fix, driven through the real route method (not
    just the raw closure), on a genuine "center" capture against the
    synthetic table fixture."""
    import tasni.modules.scan.module as scan_module

    services, _state, _started = _build_fakes_with_jobs()
    services.live = _FakeLive()
    services.config.scan.boundary_engine = "sam"

    orig_worker_cls = scan_module.SamBoundaryWorker
    scan_module.SamBoundaryWorker = _FakeSamWorker
    try:
        mod = scan_module.ScanModule(services)
        mod.router()                        # wires self._live_start
        mod._live_start()                   # the operator's initial /live/start
        first_worker = mod._sam_worker
        assert first_worker is not None and not first_worker.stopped
        services.live.running = True        # the preview WAS running before the capture

        mod.survey_begin()
        result = mod.survey_capture()       # five_position_capture stops live internally

        assert result["step"] == "corner1"
        assert services.live.running is True, "live preview must be restarted after the capture"
        assert services.live.start_calls == 2, services.live.start_calls  # initial + restart
        assert first_worker.stopped, "the pre-capture SamBoundaryWorker must not leak across the restart"
        assert mod._sam_worker is not None and mod._sam_worker is not first_worker
        assert not mod._sam_worker.stopped
    finally:
        scan_module.SamBoundaryWorker = orig_worker_cls
    print("[survey capture restart] live preview restarted; prior SamBoundaryWorker "
          "stopped, not leaked (start_calls=%d)" % services.live.start_calls)


def test_poses_generate_rejects_out_of_range_overlap_with_400():
    """Bundled cleanup (ambiguity resolution #5, a review finding from the
    previous task): poses_generate()'s except-chain used to map ONLY
    RuntimeError to 400, letting a config-validation ValueError (e.g. an
    out-of-range survey_tour_overlap, raised by planner._tile_grid_dims) fall
    through to the generic 503 "RoboDK/camera unavailable" -- misleading,
    since nothing was actually unavailable. Now mirrors insert()'s own
    convention: except (RuntimeError, ValueError, KeyError) -> 400."""
    import tasni.modules.scan.module as scan_module
    from fastapi import HTTPException

    services, state, _started = _build_fakes_with_jobs()
    services.config.scan.survey_tour_overlap = 1.5   # out of [0.0, 1.0)
    mod = scan_module.ScanModule(services)
    mod._locked_surface = _locked_five_position_scan_surface(state, _FP_LARGE_CORNERS)

    try:
        mod.poses_generate()
        raise AssertionError("expected the out-of-range overlap to be rejected")
    except HTTPException as e:
        assert e.status_code == 400, e.status_code
        assert "survey_tour_overlap" in str(e.detail), e.detail
    print("[poses_generate] out-of-range survey_tour_overlap -> 400, not 503")


if __name__ == "__main__":
    test_generate_run_insert()
    test_provenance_flows_lock_to_insert()
    test_provenance_absent_when_survey_record_is_none()
    test_lock_then_create_targets_reuses_frozen_surface()
    test_lock_builds_locked_workframe_survey_compact()
    test_lock_crop_is_user_specified_with_declared_size()
    test_prepare_frame_result_converts_the_locked_survey_verbatim()
    test_prepare_frame_result_refuses_when_the_robot_moved()
    test_prepare_frame_result_requires_a_survey_record()
    test_entire_platform_overrun_refuses_instead_of_auto_cropping()
    test_large_surface_message_names_the_true_colour_vs_depth_fov_gap()
    test_declared_region_still_crops_the_same_overrun_surface()
    test_lock_real_guard_band_rejection_at_close_standoff()
    test_lock_never_classifies_the_fabricated_reticle_square_when_not_fully_framed()
    test_lock_falls_back_to_color_when_sam_abstains()
    test_lock_rejects_when_every_boundary_engine_abstains()
    test_lock_wires_classify_compact_and_rejects_when_ineligible()
    test_lock_reports_the_failing_gate_for_each_of_the_five_conditions()
    test_lock_survey_outline_history_reflects_independent_per_frame_surveys()
    test_lock_identity_gate_accepts_a_stationary_rectangle_under_real_depth_noise()
    test_lock_identity_gate_accepts_a_near_45_degree_centred_rectangle_under_noise()
    test_align_polygon_like_has_no_degenerate_rotation_across_full_sweep()
    test_lock_adapts_identity_frame_requirement_when_measure_frames_is_lower()
    test_lock_never_vacuously_passes_identity_with_a_single_frame()
    test_lock_gate_event_carries_survey_and_provenance()
    test_lock_warns_when_characterization_missing()
    test_lock_warns_when_characterization_stale()
    test_lock_records_dstar_into_survey_quality_when_fresh()
    test_lock_hard_fails_when_characterization_missing_and_hard_fail_enabled()
    test_lock_hard_fail_gate_event_carries_the_warning_before_raising()
    test_surface_region_route_updates_lock_dimensions()
    test_targets_report_surface_coverage_from_footprint()
    test_save_views_persists_per_pose_frames()
    test_burst_capture_path()
    test_generate_targets_when_survey_touches_border()
    test_manual_crop_ignores_unstable_framed_rectangle()
    test_scan_collision_filter_bypasses_noisy_wall_map_by_default()
    test_scan_collision_filter_hard_fail_can_still_refuse()
    test_generate_refuses_when_too_far()
    test_generate_reference_mode_for_oversized_framed_surface()
    test_generate_accepts_dynamic_near_quality_distance()
    test_warns_but_proceeds_without_calibration()
    test_run_without_targets_errors()
    test_run_refuses_targets_from_a_previous_lock()
    test_run_refuses_targets_from_a_relocked_surface()
    test_run_starts_normally_after_lock_and_generate()
    test_run_not_blocked_when_targets_token_was_never_set()
    test_sparse_measured_support_is_rejected()
    test_five_position_tiled_tour_spans_multiple_tiles()
    test_five_position_tiled_tour_coverage_gate_catches_missing_tiles()
    test_five_position_tiled_tour_contiguous_hole_caught_even_when_fraction_passes()
    test_five_position_tiled_tour_small_rectangle_is_single_tile()
    test_five_position_capture_uses_fresh_robot_state()
    test_five_position_capture_rejects_disconnected_driver()
    test_five_position_capture_rejects_moving_robot()
    test_deproject_plane_points_mm_filters_background_to_plane_inliers()
    test_five_position_survey_accepts_captures_with_zero_background_depth()
    test_five_position_survey_accepts_well_aimed_corner_with_real_background()
    test_five_position_capture_rejects_genuinely_bad_aim_with_real_background()
    test_five_position_capture_rejects_tiny_table_sliver_via_coverage_gate()
    test_five_position_survey_still_rejects_genuine_noncoplanarity_with_background()
    test_deproject_plane_points_mm_regression_pin_against_pre_fix_behaviour()
    test_five_position_capture_corner_step_passes_closed_true_to_corner_evidence()
    test_survey_begin_state_capture_cancel_routes()
    test_survey_routes_require_an_active_survey()
    test_survey_recapture_route_rejects_unknown_kind()
    test_survey_finish_route_locks_and_sets_current_lock_token()
    test_survey_finish_route_rejects_incomplete_survey()
    test_live_start_stops_existing_sam_worker_before_replacing_it()
    test_survey_capture_restarts_live_preview_and_its_sam_worker_cleanly()
    test_poses_generate_rejects_out_of_range_overlap_with_400()
    print("\nScan job (gate -> generate -> run -> insert) tests passed.")


def _corroboration_scene(shift_x_mm: float):
    from types import SimpleNamespace
    K = np.array([[600.0, 0.0, 640.0], [0.0, 600.0, 360.0], [0.0, 0.0, 1.0]])
    camera_cfg = SimpleNamespace(K=K, dist=np.zeros(5), size=(1280, 720))
    survey = SimpleNamespace(
        normal_cam=np.array([0.0, 0.0, -1.0]),
        centroid_cam_mm=np.array([0.0, 0.0, 500.0]),
        extent_mm=(300.0, 200.0),
        corners_cam_mm=np.array([[-150.0, -100.0, 500.0], [150.0, -100.0, 500.0],
                                 [150.0, 100.0, 500.0], [-150.0, 100.0, 500.0]]))
    xs = np.array([-150.0, 150.0, 150.0, -150.0]) + shift_x_mm
    ys = np.array([-100.0, -100.0, 100.0, 100.0])
    u = (xs / 500.0 * 600.0 + 640.0) / 1280.0
    v = (ys / 500.0 * 600.0 + 360.0) / 720.0
    return np.column_stack([u, v]), survey, camera_cfg


def test_vision_boundary_rejected_when_laterally_shifted():
    """Review finding: side lengths were the ONLY corroboration, so a shifted
    segmentation (same extent, moved centre) replaced the work-frame corners
    ~40 mm off under a green lock."""
    from tasni.core.config import ScanConfig
    polygon_uv, survey, camera_cfg = _corroboration_scene(shift_x_mm=40.0)
    corners, info = scan_service._corners_from_boundary_on_plane(
        polygon_uv, survey, camera_cfg, ScanConfig())
    assert corners is None
    assert "centre" in info["reason"]
    assert info["boundary_source"] == "depth"


def test_vision_boundary_accepted_when_centred():
    from tasni.core.config import ScanConfig
    polygon_uv, survey, camera_cfg = _corroboration_scene(shift_x_mm=5.0)
    corners, info = scan_service._corners_from_boundary_on_plane(
        polygon_uv, survey, camera_cfg, ScanConfig())
    assert corners is not None
    assert info["boundary_source"] == "vision"
    assert info["center_offset_mm"] <= 6.0


def test_vision_boundary_updates_extent_for_the_planner():
    """Review finding: only corners_cam_mm was replaced at the hybrid lock;
    plan_scan reads survey.extent_mm, so the tour was sized from the smaller
    depth rectangle and the gate payload contradicted the record."""
    from dataclasses import dataclass

    @dataclass
    class _S:
        corners_cam_mm: object
        extent_mm: object

    survey = _S(corners_cam_mm=None, extent_mm=(280.0, 200.0))
    corners = np.zeros((4, 3))
    out = scan_service._survey_with_vision_boundary(
        survey, corners, {"vision_extent_mm": [320.0, 240.0]})
    assert out.extent_mm == (320.0, 240.0)
    assert out.corners_cam_mm is corners
    out2 = scan_service._survey_with_vision_boundary(survey, corners, {})
    assert out2.extent_mm == (280.0, 200.0)


def test_live_distance_gate_clamped_to_accurate_band():
    """Audit A2: a crop-latched ideal of 800 must not accept 930 mm just because
    |930-800| <= distance_tol_mm — the window's top is accurate_max_mm."""
    import time as _time
    from tasni.core.config import ScanConfig
    scfg = ScanConfig()
    raw = {"detected": True, "distance_mm": 930.0, "tilt_deg": 1.0,
           "valid_frac": 0.9, "surface_mode": "crop",
           "_received_at": _time.time(), "timestamp": _time.time()}
    out = scan_service.live_scan_telemetry_payload(raw, scfg, previous_ideal_mm=800.0)
    assert out["ideal_distance_mm"] == 800.0
    assert out["gates"]["distance"] is False, out
    ok = dict(raw, distance_mm=780.0)
    out2 = scan_service.live_scan_telemetry_payload(ok, scfg, previous_ideal_mm=800.0)
    assert out2["gates"]["distance"] is True, out2


def test_stabilize_does_not_hand_back_the_unclamped_window():
    """stabilize_live_scan_payload re-gates distance AFTER the payload builder,
    so it has to use the same clamped window (audit A2) — otherwise the live HUD
    goes green past accurate_max_mm even though the builder said no."""
    from tasni.core.config import ScanConfig
    scfg = ScanConfig()

    def frame(distance_mm, distance_gate):
        return {"detected": True, "live": True, "surface_mode": "crop",
                "distance_mm": distance_mm, "ideal_distance_mm": 800.0,
                "tilt_deg": 1.0, "distance_tol_mm": scfg.distance_tol_mm,
                "max_tilt_deg": scfg.max_tilt_deg,
                "gates": {"detected": True, "distance": distance_gate,
                          "angle": True}}

    # Previous frame green at 900 mm (what the UNCLAMPED gate used to publish:
    # |900-800| <= distance_tol_mm), so hysteresis widens the window. 930 mm is
    # still far past accurate_max_mm and must NOT be re-gated green.
    out = scan_service.stabilize_live_scan_payload(
        frame(930.0, False), frame(900.0, True), scfg, robot_static=False)
    assert out["gates"]["distance"] is False, out
    assert out["distance_window_mm"][1] == scfg.accurate_max_mm
    # ...and a reading inside the band still gates green through the same path.
    ok = scan_service.stabilize_live_scan_payload(
        frame(790.0, True), frame(780.0, True), scfg, robot_static=False)
    assert ok["gates"]["distance"] is True, ok


def test_standoff_window_exempts_reference_mode_far_edge_only():
    """Reference mode exists for a surface too big to frame inside the accurate
    band, and the planner pins its ideal AT accurate_max_mm — clamping the far
    edge there would make the mode unreachable (it builds no mesh and runs no
    tour, so standing past the band is the point). The NEAR edge still clamps:
    below MinZ the camera returns nothing usable in any mode."""
    from tasni.core.config import ScanConfig
    scfg = ScanConfig()          # accurate band 300..800, tol 150
    lo, hi = scan_service.standoff_accept_window_mm(800.0, scfg)
    assert (lo, hi) == (650.0, 800.0)
    lo_r, hi_r = scan_service.standoff_accept_window_mm(
        800.0, scfg, reference_mode=True)
    assert (lo_r, hi_r) == (650.0, 950.0)
    # near edge clamped in BOTH modes
    for ref in (False, True):
        lo_n, _ = scan_service.standoff_accept_window_mm(350.0, scfg, reference_mode=ref)
        assert lo_n == scfg.accurate_min_mm


def test_stabilize_honours_a_published_reference_window():
    """stabilize must reuse the window the payload builder published — only the
    builder knows this surface is heading for reference mode."""
    from tasni.core.config import ScanConfig
    scfg = ScanConfig()

    def frame(distance_mm, gate):
        return {"detected": True, "live": True, "surface_mode": "full",
                "distance_mm": distance_mm, "ideal_distance_mm": 800.0,
                "tilt_deg": 1.0, "distance_tol_mm": scfg.distance_tol_mm,
                "distance_window_mm": [650.0, 950.0],   # reference: far edge open
                "max_tilt_deg": scfg.max_tilt_deg,
                "gates": {"detected": True, "distance": gate, "angle": True}}

    out = scan_service.stabilize_live_scan_payload(
        frame(900.0, True), frame(890.0, True), scfg, robot_static=False)
    assert out["gates"]["distance"] is True, out


def test_plane_rms_none_when_starved_never_nan():
    """Review finding: NaN here kills the /ws JSON on the client and 500s the
    lock/insert responses (FastAPI renders with allow_nan=False)."""
    K = np.array([[600.0, 0.0, 32.0], [0.0, 600.0, 32.0], [0.0, 0.0, 1.0]])
    frame = SimpleNamespace(depth=np.zeros((64, 64), np.uint16),
                            geometry=gf.aligned(K, (64, 64)))
    cfg = SimpleNamespace(camera=SimpleNamespace(K=K, dist=None))
    assert scan_service._plane_rms_mm(frame, cfg) is None


def test_finite_or_none_guards_payload_metrics():
    assert scan_service._finite_or_none(float("nan")) is None
    assert scan_service._finite_or_none(float("inf")) is None
    assert scan_service._finite_or_none(1.25) == 1.25
    assert scan_service._finite_or_none(None) is None
