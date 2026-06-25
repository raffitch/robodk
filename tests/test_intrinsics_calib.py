"""Tests for the dedicated RGB intrinsic calibration (intrinsics_calib.py) and the
apply-to-config path (service.apply_intrinsics). No camera, no RoboDK:

* the solver recovers known K + distortion from projected ChArUco corners,
* ``fix_k3`` actually pins k3 to 0 (and freeing it recovers a non-zero k3),
* the auto-capture gate records only stable, novel views with enough corners,
* applying writes K + dist into the live config and persists them, preserving the
  other resolutions' K.

    py -3.10 tests/test_intrinsics_calib.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tasni.core import config as config_mod  # noqa: E402
from tasni.core.config import AppConfig  # noqa: E402
from tasni.modules.calibration import service as service_mod  # noqa: E402
from tasni.modules.calibration.intrinsics_calib import (  # noqa: E402
    GRID_X, GRID_Y, MIN_CORNERS, IntrinsicCalibSession, _View,
    coverage_from_corners, solve_intrinsics)

K_TRUE = np.array([[908.1, 0, 650.2], [0, 908.14, 366.6], [0, 0, 1]])
SIZE = (1280, 720)


def _grid_obj():
    """A 7x5 planar corner grid (35 pts), board frame z=0, in mm."""
    xs = (np.arange(7) - 3) * 30.0
    ys = (np.arange(5) - 2) * 30.0
    return np.array([[x, y, 0.0] for y in ys for x in xs], dtype=np.float64)


def _project_views(K, dist, *, n=18, seed=0):
    """Build n synthetic views (corners, ids, obj) by projecting a grid from varied
    poses that keep it in front of the camera."""
    obj = _grid_obj()
    ids = np.arange(obj.shape[0], dtype=np.int32).reshape(-1, 1)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        rvec = np.deg2rad(rng.uniform(-25, 25, 3)).reshape(3, 1)
        tvec = np.array([rng.uniform(-120, 120), rng.uniform(-80, 80),
                         rng.uniform(420, 560)]).reshape(3, 1)
        img, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        out.append((img.astype(np.float32).reshape(-1, 1, 2), ids.copy(),
                    obj.astype(np.float32)))
    return out


def _load_views(session, views):
    for i, (corners, ids, obj) in enumerate(views):
        session._views.append(_View(corners, ids, obj, (i % GRID_X, (i // GRID_X) % GRID_Y)))
        session._cells[(i // GRID_X) % GRID_Y, i % GRID_X] += 1


def test_solver_recovers_known_intrinsics():
    dist = np.array([0.105, -0.20, 0.0015, -0.0008, 0.0])      # k3 = 0
    s = IntrinsicCalibSession(SIZE)
    _load_views(s, _project_views(K_TRUE, dist, n=18))
    # Solve from a deliberately-wrong guess to prove convergence.
    K0 = K_TRUE + np.array([[6, 0, -5], [0, -7, 4], [0, 0, 0]])
    rep = s.solve(K0, fix_k3=True)
    assert rep["rms_px"] < 0.05, rep["rms_px"]
    assert abs(rep["fx"] - 908.1) < 0.5 and abs(rep["fy"] - 908.14) < 0.5
    assert abs(rep["cx"] - 650.2) < 0.5 and abs(rep["cy"] - 366.6) < 0.5
    assert np.allclose(rep["dist"], dist, atol=2e-3)
    print(f"[recover] rms {rep['rms_px']:.4f}px  dist {[round(x,4) for x in rep['dist']]}")


def test_fix_k3_pins_k3_and_freeing_recovers_it():
    dist = np.array([0.10, -0.18, 0.0, 0.0, 0.35])             # real non-zero k3
    views = _project_views(K_TRUE, dist, n=20, seed=2)

    fixed = IntrinsicCalibSession(SIZE)
    _load_views(fixed, views)
    rf = fixed.solve(K_TRUE, fix_k3=True)
    assert rf["dist"][4] == 0.0, "fix_k3 must hold k3 at 0"

    free = IntrinsicCalibSession(SIZE)
    _load_views(free, views)
    rr = free.solve(K_TRUE, fix_k3=False)
    assert abs(rr["dist"][4] - 0.35) < 0.05, rr["dist"][4]
    assert rr["rms_px"] < 0.05
    print(f"[fix_k3] fixed k3={rf['dist'][4]}  free k3={rr['dist'][4]:.3f}")


def test_autocapture_only_keeps_stable_novel_views():
    s = IntrinsicCalibSession(SIZE)
    (c0, i0, o0), (c1, i1, o1) = _project_views(K_TRUE, np.zeros(5), n=2, seed=5)

    # First sighting: no previous frame -> not "stable" -> not captured.
    st = s.offer(c0, i0, o0)
    assert st["detected"] and not st["captured"] and st["count"] == 0
    # Same view again: still vs the previous frame + novel -> captured.
    st = s.offer(c0, i0, o0)
    assert st["captured"] and st["count"] == 1
    # Held in place: stable but a near-duplicate of a kept view -> not captured.
    st = s.offer(c0, i0, o0)
    assert not st["captured"] and st["count"] == 1
    # Move to a clearly different pose: first frame jumps (not stable), then settles.
    assert not s.offer(c1, i1, o1)["captured"]
    st = s.offer(c1, i1, o1)
    assert st["captured"] and st["count"] == 2

    # Too few corners is never captured, even if stable + novel.
    few = slice(0, MIN_CORNERS - 3)
    s.offer(c0[few], i0[few], o0[few])
    st = s.offer(c0[few], i0[few], o0[few])
    assert not st["captured"] and st["count"] == 2
    print(f"[autocapture] kept {st['count']} of many offers; coverage "
          f"{int(st['coverage_pct']*100)}% on a {GRID_X}x{GRID_Y} grid")


def test_solve_intrinsics_function_and_coverage():
    dist = np.array([0.09, -0.16, 0.0, 0.0, 0.0])
    views = _project_views(K_TRUE, dist, n=16, seed=11)
    obj = [c[2] for c in views]
    img = [c[0] for c in views]
    rep = solve_intrinsics(obj, img, SIZE, K_TRUE, fix_k3=True)
    assert rep["rms_px"] < 0.05 and rep["n_views"] == 16
    assert np.allclose(rep["dist"], dist, atol=2e-3)
    pct, cells = coverage_from_corners(img, SIZE)
    assert rep["coverage_pct"] == pct and 0.0 < pct <= 1.0
    assert len(cells) == GRID_Y and len(cells[0]) == GRID_X
    print(f"[solve_intrinsics] rms {rep['rms_px']:.4f}px  coverage {int(pct*100)}%")


def test_verify_intrinsics_compares_against_configured_not_zero():
    """The self-check must warn when the configured distortion DISAGREES with the
    lens — not merely because the lens has distortion. So: zero config + a distorting
    lens warns; applying the matching distortion clears it (the false-positive that
    kept the verdict BORDERLINE after auto-intrinsics applied the right values)."""
    from tasni.modules.calibration.intrinsics import verify_intrinsics

    dist_true = np.array([0.10, -0.18, 0.0, 0.0, 0.0])
    raw = _project_views(K_TRUE, dist_true, n=14, seed=9)
    views = [SimpleNamespace(obj_points=o, corners=c) for c, _i, o in raw]

    w_zero = verify_intrinsics(views, K_TRUE, np.zeros((5, 1)), SIZE)
    assert w_zero["warn"] is True, "zero-config vs a distorting lens must warn"

    w_ok = verify_intrinsics(views, K_TRUE, dist_true.reshape(-1, 1), SIZE)
    assert w_ok["warn"] is False, f"correct config must NOT warn: {w_ok['note']}"
    print(f"[verify] zero-cfg warn={w_zero['warn']} | correct-cfg warn={w_ok['warn']} "
          f"('{w_ok['note']}')")


def test_apply_intrinsics_updates_live_config_persists_and_marks():
    cfg = AppConfig()
    services = SimpleNamespace(config=cfg)
    res = cfg.camera.resolution
    others = {r for r in cfg.camera.intrinsics if r != res}
    K_new = [[900.0, 0, 640.0], [0, 901.0, 360.0], [0, 0, 1]]
    dist_new = [0.11, -0.21, 0.001, -0.001, 0.0]
    report = {"K": K_new, "dist": dist_new, "rms_px": 0.3, "n_views": 18,
              "coverage_pct": 0.9, "fix_k3": True}

    marker: dict = {}
    orig_runs = service_mod.runs
    service_mod.runs = SimpleNamespace(
        read_active=lambda m: marker.get(m),
        write_active=lambda m, payload: marker.__setitem__(m, payload))
    with tempfile.TemporaryDirectory() as tmp:
        cfg_path = Path(tmp) / "tasni.config.json"
        orig = config_mod.config_file_path
        config_mod.config_file_path = lambda: cfg_path        # redirect persistence
        try:
            assert service_mod.intrinsics_present(services) is False
            out = service_mod.apply_intrinsics(services, report, source="auto")
        finally:
            config_mod.config_file_path = orig
            service_mod.runs = orig_runs

        assert np.allclose(cfg.camera.K, K_new)               # live mutation, no restart
        assert np.allclose(cfg.camera.dist.reshape(-1), dist_new)
        assert others.issubset(set(cfg.camera.intrinsics))    # other resolutions kept
        import json
        saved = json.loads(cfg_path.read_text())
        assert saved["camera"]["dist_coeffs"] == dist_new
        assert saved["camera"]["intrinsics"][res] == K_new
        assert out["source"] == "auto"
        assert marker.get("intrinsics", {}).get("source") == "auto"   # marker written
    print(f"[apply] live K+dist set; persisted; marker written; kept {sorted(others)}")


if __name__ == "__main__":
    test_solver_recovers_known_intrinsics()
    test_fix_k3_pins_k3_and_freeing_recovers_it()
    test_solve_intrinsics_function_and_coverage()
    test_verify_intrinsics_compares_against_configured_not_zero()
    test_autocapture_only_keeps_stable_novel_views()
    test_apply_intrinsics_updates_live_config_persists_and_marks()
    print("\nIntrinsic calibration tests passed.")
