"""Run-artifact registry (core/runs.py) + apply-by-run-id / provenance.

No RoboDK, no camera — a temp ``runs/`` tree and a fake rdk. Covers:
  * path-traversal guard (untrusted module/stamp can't climb out of runs/)
  * load_report / load_meta round-trip + RunNotFound when missing
  * list_runs newest-first + limit + skips the active.json pointer file
  * write_active / read_active atomic round-trip
  * apply_calibration: by run_id (from disk, survives restart) AND in-memory,
    both write the tool and record runs/calibration/active.json provenance

    py -3.10 tests/test_runs.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tasni.core import runs  # noqa: E402


@pytest.fixture
def tmp(tmp_path: Path) -> Path:
    """Alias pytest's builtin ``tmp_path`` so these tests run under pytest as
    well as via the ``__main__`` block below (which passes ``tmp`` positionally)."""
    return tmp_path
from tasni.core.config import AppConfig  # noqa: E402
from tasni.modules.calibration import service as service_mod  # noqa: E402

X_TRUE = [[1, 0, 0, 40], [0, 1, 0, -15], [0, 0, 1, 55], [0, 0, 0, 1]]


def _write_run(root: Path, stamp: str, *, tool="Realsense", verdict="pass",
               train=0.4, val=0.6) -> None:
    d = root / "runs" / "calibration" / stamp
    d.mkdir(parents=True, exist_ok=True)
    report = {
        "refined": True, "method": "PARK", "X_cam2gripper": X_TRUE,
        "train": {"rms_px": train}, "validation": {"rms_px": val},
        "board_consistency_mm": {"rms": 0.9},
        "diagnosis": {"verdict": verdict, "headline": "ok"},
    }
    (d / "report.json").write_text(json.dumps(report), encoding="utf-8")
    (d / "meta.json").write_text(json.dumps(
        {"module": "calibration", "stamp": stamp, "tool_name": tool}), encoding="utf-8")


# -- path-traversal guard ---------------------------------------------------
def test_path_traversal_rejected():
    for bad in ("..", "../secrets", "a/b", "a\\b", "", "."):
        try:
            runs.run_dir("calibration", bad, root=Path("X"))
            raise AssertionError(f"expected rejection of stamp {bad!r}")
        except ValueError:
            pass
        try:
            runs.run_dir(bad, "20260101-000000", root=Path("X"))
            raise AssertionError(f"expected rejection of module {bad!r}")
        except ValueError:
            pass
    print("[guard] separators / .. / empty rejected for module + stamp")


def test_load_report_roundtrip_and_missing(tmp: Path):
    _write_run(tmp, "20260620-101010")
    rep = runs.load_report("calibration", "20260620-101010", root=tmp)
    assert rep["method"] == "PARK" and rep["X_cam2gripper"] == X_TRUE
    meta = runs.load_meta("calibration", "20260620-101010", root=tmp)
    assert meta["tool_name"] == "Realsense"
    # missing run -> RunNotFound (a FileNotFoundError subclass)
    try:
        runs.load_report("calibration", "19990101-000000", root=tmp)
        raise AssertionError("expected RunNotFound")
    except runs.RunNotFound:
        pass
    assert runs.load_meta("calibration", "19990101-000000", root=tmp) is None
    print("[load] report/meta round-trip; missing -> RunNotFound")


def test_list_runs_orders_and_skips_active(tmp: Path):
    for s in ("20260101-000000", "20260620-101010", "20260315-120000"):
        _write_run(tmp, s)
    runs.write_active("calibration", {"run_id": "20260620-101010"}, root=tmp)
    listed = runs.list_runs(limit=20, root=tmp)
    stamps = [r["stamp"] for r in listed]
    assert stamps == sorted(stamps, reverse=True)        # newest first
    assert "active.json" not in stamps                   # the pointer file is not a run
    assert all(r["module"] == "calibration" for r in listed)
    assert len(runs.list_runs(limit=2, root=tmp)) == 2   # limit honoured
    print("[list] newest-first, limited, active.json skipped")


def test_write_active_atomic_roundtrip(tmp: Path):
    runs.write_active("calibration", {"run_id": "A", "tool": "Realsense"}, root=tmp)
    runs.write_active("calibration", {"run_id": "B", "tool": "Realsense"}, root=tmp)  # overwrite
    got = runs.read_active("calibration", root=tmp)
    assert got["run_id"] == "B"
    # no stray .tmp left behind
    assert not (tmp / "runs" / "calibration" / "active.json.tmp").exists()
    assert runs.read_active("scan", root=tmp) is None    # absent -> None
    print("[active] atomic overwrite, no .tmp residue, absent -> None")


# -- apply_calibration ------------------------------------------------------
class _FakeRdk:
    def __init__(self): self.applied = None
    def set_tool_pose(self, tool, T): self.applied = (tool, np.asarray(T))


def _services(tmp: Path):
    return SimpleNamespace(config=AppConfig(), rdk=_FakeRdk())


def test_apply_by_run_id_from_disk(tmp: Path):
    # The in-memory job is GONE (server restarted) — only disk remains.
    _write_run(tmp, "20260620-090000", tool="Realsense", verdict="pass", val=0.55)
    runs.REPO_ROOT = tmp                     # redirect the default root for this call
    svc = _services(tmp)
    try:
        out = service_mod.apply_calibration(svc, job=None, run_id="20260620-090000")
    finally:
        runs.REPO_ROOT = _ORIG_ROOT
    assert out["status"] == "applied" and out["tool"] == "Realsense"
    assert out["source"] == "run_id" and out["run_id"] == "20260620-090000"
    # tool written with the on-disk transform
    assert svc.rdk.applied[0] == "Realsense"
    assert np.allclose(svc.rdk.applied[1], np.asarray(X_TRUE, float))
    # provenance recorded
    active = runs.read_active("calibration", root=tmp)
    assert active["run_id"] == "20260620-090000"
    assert active["quality"]["verdict"] == "pass"
    assert active["quality"]["val_rms_px"] == 0.55
    assert active["source"] == "run_id" and "applied_at" in active
    print("[apply run_id]", active["applied_at"], active["quality"])


def test_apply_in_memory_job(tmp: Path):
    runs.REPO_ROOT = tmp
    svc = _services(tmp)
    job = SimpleNamespace(
        solved_X=np.asarray(X_TRUE, float), tool_name="Realsense",
        result=SimpleNamespace(
            report={"refined": True, "method": "TSAI",
                    "train": {"rms_px": 0.3}, "validation": {"rms_px": 0.4},
                    "board_consistency_mm": {"rms": 0.8},
                    "diagnosis": {"verdict": "pass"}},
            run_dir=str(tmp / "runs" / "calibration" / "20260620-110000")))
    try:
        out = service_mod.apply_calibration(svc, job=job, run_id=None)
    finally:
        runs.REPO_ROOT = _ORIG_ROOT
    assert out["source"] == "memory" and out["run_id"] == "20260620-110000"
    assert svc.rdk.applied[0] == "Realsense"
    active = runs.read_active("calibration", root=tmp)
    assert active["method"] == "TSAI" and active["quality"]["verdict"] == "pass"
    print("[apply memory]", active["run_id"], active["quality"]["verdict"])


def test_apply_nothing_to_apply(tmp: Path):
    try:
        service_mod.apply_calibration(_services(tmp), job=None, run_id=None)
        raise AssertionError("expected a refusal")
    except RuntimeError as e:
        assert "nothing" in str(e).lower() or "no solved" in str(e).lower()
    print("[apply none] refused with no job and no run_id")


_ORIG_ROOT = runs.REPO_ROOT


if __name__ == "__main__":
    import tempfile

    test_path_traversal_rejected()
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        test_load_report_roundtrip_and_missing(tmp)
        test_list_runs_orders_and_skips_active(tmp)
        test_write_active_atomic_roundtrip(tmp)
        test_apply_by_run_id_from_disk(tmp)
        test_apply_in_memory_job(tmp)
        test_apply_nothing_to_apply(tmp)
    print("\nRun-registry + apply-by-run-id tests passed.")
