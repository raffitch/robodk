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
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tasni.core import runs  # noqa: E402
from tasni.core.camera_lease import CameraLease  # noqa: E402
from tasni.core.config import AppConfig, ScanConfig  # noqa: E402
from tasni.core.geometry import Rt_to_T  # noqa: E402
from tasni.core.jobrunner import JobContext  # noqa: E402
from tasni.modules.scan import service as scan_service  # noqa: E402
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


def _render(T_base_cam):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    dirs_cam = np.stack([(us - cx) / fx, (vs - cy) / fy, np.ones_like(us, float)], -1)
    R, t = T_base_cam[:3, :3], T_base_cam[:3, 3]
    dirs_base = dirs_cam @ R.T
    dz = dirs_base[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (0.0 - t[2]) / dz
    P = t + s[..., None] * dirs_base
    valid = ((np.abs(P[..., 0]) <= TABLE_HALF_MM) & (np.abs(P[..., 1]) <= TABLE_HALF_MM)
             & (s > 0) & np.isfinite(s))
    depth = np.where(valid, s, 0).astype(np.uint16)
    color = np.full((H, W, 3), 128, np.uint8)
    return SimpleNamespace(color=color, depth=depth, timestamp=FRAME_TIMESTAMP)


def _build_fakes(mount_mm=(40.0, -15.0, 55.0)):
    seed_T = _look_at((0, 0, 420), (0, 0, 0))           # straight down, close framed standoff
    state = {"cam": seed_T, "targets": {}, "joints": {}}
    mount = Rt_to_T(np.eye(3), np.asarray(mount_mm, float))

    class FakeRdk:
        def __init__(self): self.inserted = {}
        def item_exists(self, name): return True
        def apply_run_mode(self, mode=None): return "run_robot"
        def connect_robot(self, ip="", *, timeout_s=10.0, poll_s=0.4):
            return True, "ROBOTCOM_READY"
        def robot_connection_params(self): return {"ip": "10.0.0.5", "port": 7000}
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
            self.inserted["frame"] = np.asarray(T, float)
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
    """An auto-detected crop overrun (the surface overruns the view, no
    force_crop) builds NO survey_record -- a silent SYSTEM fallback, not an
    operator declaration (Task 3's honest-provenance rule). The pipeline must
    still complete end to end: provenance is simply ABSENT throughout (None),
    never fabricated, never defaulted to a measured-sounding string."""
    try:
        import open3d  # noqa: F401
    except Exception:
        print("[skip] open3d not installed — `pip install -e .[scan]`")
        return
    global TABLE_HALF_MM
    saved = TABLE_HALF_MM
    TABLE_HALF_MM = 1000.0
    try:
        services, state = _build_fakes()
        state["cam"] = _look_at((0, 0, 310), (0, 0, 0))
        # The crop's fixed 1000x1000 mm work region reaches farther than the 8
        # cone-limited tour views actually capture on this synthetic table, which
        # would otherwise trip the (unrelated) measured-edge-coverage hard-fail
        # gate -- this test is about provenance flow, not synthetic mesh coverage.
        services.config.scan.actual_coverage_hard_fail = False
        locked = scan_service.lock_scan_surface(services)
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
    print("[survey record] crop lock -> user-specified", rec.size_mm)


def test_lock_auto_crop_overrun_builds_no_survey_record_but_warns():
    """An auto-detected crop (the surface overruns the view, no force_crop) is a
    silent SYSTEM fallback, not an operator declaration — tagging it USER_SPECIFIED
    would be a false provenance claim (spec §1/§12). No survey record is built; the
    gate payload instead carries a human-readable warning. Everything else about the
    lock (readiness, the crop overlay, the lock_token) still works as before.
    """
    global TABLE_HALF_MM
    saved = TABLE_HALF_MM
    TABLE_HALF_MM = 1000.0
    try:
        services, state = _build_fakes()
        state["cam"] = _look_at((0, 0, 310), (0, 0, 0))
        locked = scan_service.lock_scan_surface(services)
        assert locked.gate_payload["ok"] is True, locked.gate_payload
        assert locked.gate_payload["surface_mode"] == "crop", locked.gate_payload
        assert locked.survey_record is None
        assert "boundary_provenance" not in locked.gate_payload
        assert "survey" not in locked.gate_payload
        assert locked.gate_payload.get("warnings"), locked.gate_payload
        assert locked.lock_token != ""
    finally:
        TABLE_HALF_MM = saved
    print("[survey record] auto-overrun crop -> no record, warning:",
          locked.gate_payload["warnings"][-1])


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
            assert meta["size"] == [W, H] and len(meta["K"]) == 3
            assert len(meta["views"][0]["pose_T_mm"]) == 4   # 4x4 pose persisted
        finally:
            scan_service.new_run_dir = orig
            runs.REPO_ROOT = _ORIG_ROOT
    print("[save_views] persisted", res["n_views"], "color+depth frames + poses")


def test_generate_targets_when_survey_touches_border():
    """A full-frame survey can mark FRAMED red while the old centre gate is valid.

    Target creation should still use the current-pose cone, not fail just because the
    measured surface reaches the image border.
    """
    global TABLE_HALF_MM
    saved = TABLE_HALF_MM
    TABLE_HALF_MM = 1000.0
    try:
        services, state = _build_fakes()
        state["cam"] = _look_at((0, 0, 310), (0, 0, 0))
        gen = scan_service.generate_scan_targets(services)
        assert gen["created"] == 8, gen
        assert gen["boundary_views_enabled"] is False, gen
        assert gen["boundary_aim_offsets"] == 0, gen
        assert gen["gate"]["ok"] is True, gen["gate"]
        assert gen["gate"]["gates"].get("framed") is False, gen["gate"]
        assert 300 <= gen["look_distance_mm"] <= 340, gen["look_distance_mm"]
        # The surface overruns the view, so the work region is the generic fixed
        # square (scan.work_crop_mm default 1000×1000), not an adaptive FOV-fraction.
        assert gen["crop_size_mm"] is not None
        assert gen["crop_size_mm"] == [1000.0, 1000.0], gen["crop_size_mm"]
    finally:
        TABLE_HALF_MM = saved
    print("[survey border] framed red but centre gate OK -> created", gen["created"],
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
    # extent; manual crop keeps the reticle/center-plane fallback.
    assert abs(gen["look_distance_mm"] - 420.0) < 1.0, gen["look_distance_mm"]
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
    survey_events = [e for e in events if e.type == "survey"]
    assert survey_events and survey_events[-1].payload["step"] == "corner1"
    print("[five-position capture] center accepted -> step corner1; JobEvent published")


def test_five_position_capture_rejects_moving_robot():
    """Core safety contract (Task 13): a robot that MOVED between the two
    refresh_robot_state reads must be rejected immediately -- before the
    capture is ever handed to the survey state machine -- not merely
    recorded and silently accepted. This is stricter than the compact lock
    (which still records robot.stationary=False without refusing) because a
    five-position survey combines FIVE separately-registered positions, so a
    stale/live pose blend at any one of them would corrupt the cross-position
    plane/rectangle fit in a way no downstream check could distinguish from a
    real geometry error."""
    services, _state = _build_fakes()
    survey = FivePositionSurvey(services.config.scan)

    calls = {"n": 0}

    def moving_joints():
        calls["n"] += 1
        return [0.0, 0.0, 0.0, 0.0, 0.0, float(calls["n"])]   # different every read

    services.rdk.current_joints = moving_joints
    try:
        scan_service.five_position_capture(services, survey)
        raise AssertionError("expected a moving robot to be rejected")
    except RuntimeError as e:
        assert "moving" in str(e), e
    assert survey.step == "center"       # rejected capture must not have been accepted
    print("[five-position capture] moving robot correctly rejected before acceptance")


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
    test_lock_auto_crop_overrun_builds_no_survey_record_but_warns()
    test_lock_gate_event_carries_survey_and_provenance()
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
    test_five_position_tiled_tour_small_rectangle_is_single_tile()
    test_five_position_capture_uses_fresh_robot_state()
    test_five_position_capture_rejects_moving_robot()
    test_five_position_capture_corner_step_passes_closed_true_to_corner_evidence()
    test_survey_begin_state_capture_cancel_routes()
    test_survey_routes_require_an_active_survey()
    test_survey_recapture_route_rejects_unknown_kind()
    test_survey_finish_route_locks_and_sets_current_lock_token()
    test_survey_finish_route_rejects_incomplete_survey()
    test_poses_generate_rejects_out_of_range_overlap_with_400()
    print("\nScan job (gate -> generate -> run -> insert) tests passed.")
