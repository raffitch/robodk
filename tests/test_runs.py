"""Run-artifact registry (core/runs.py) + apply-by-run-id / provenance.

No RoboDK, no camera — a temp ``runs/`` tree and a fake rdk. Covers:
  * path-traversal guard (untrusted module/stamp can't climb out of runs/)
  * load_report / load_meta round-trip + RunNotFound when missing
  * list_runs newest-first + limit + skips the active.json pointer file
  * list_runs reports size/file-count and flags the applied run
  * ONLY a registered module's buckets are listed/deletable (the 2026-08-30 loss)
  * delete_run removes the folder, reports what it freed, and refuses to climb out
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

#: What the webapp passes as ``modules=``: the registered module ids (see
#: ``tasni/modules/registry.py::build_registry``).
MODULES = ("calibration", "scan", "extrusion")


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
    listed = runs.list_runs(limit=20, root=tmp, modules=MODULES)
    stamps = [r["stamp"] for r in listed]
    assert stamps == sorted(stamps, reverse=True)        # newest first
    assert "active.json" not in stamps                   # the pointer file is not a run
    assert all(r["module"] == "calibration" for r in listed)
    assert len(runs.list_runs(limit=2, root=tmp, modules=MODULES)) == 2   # limit honoured
    print("[list] newest-first, limited, active.json skipped")


def test_list_runs_reports_size_and_applied(tmp: Path):
    tmp = tmp / "sizes"                      # own root: __main__ shares one tmp dir
    _write_run(tmp, "20260101-000000")
    _write_run(tmp, "20260620-101010")
    (tmp / "runs" / "calibration" / "20260101-000000" / "cloud.ply").write_bytes(b"x" * 2048)
    runs.write_active("calibration", {"run_id": "20260620-101010"}, root=tmp)
    by_stamp = {r["stamp"]: r for r in runs.list_runs(limit=10, root=tmp, modules=MODULES)}
    old_run, applied = by_stamp["20260101-000000"], by_stamp["20260620-101010"]
    assert old_run["files"] == 3 and old_run["bytes"] > 2048     # report + meta + cloud
    assert applied["files"] == 2 and old_run["bytes"] > applied["bytes"]
    assert applied["active"] is True and old_run["active"] is False
    assert runs.active_run_id("calibration", root=tmp) == "20260620-101010"
    assert runs.active_run_id("scan", root=tmp) is None
    print("[list] size/file-count reported, applied run flagged")


def test_only_module_buckets_are_listed_and_deletable(tmp: Path):
    """Regression for the 2026-08-30 data loss.

    ``runs/`` is not only run buckets — a figures backup and a characterization
    archive were parked there. ``list_runs`` treated every directory under ``runs/``
    as a module, so those appeared as ordinary deletable rows and "select all"
    swept them (``runs/`` is gitignored: unrecoverable). Only a registered module's
    bucket may be listed, and the delete guard must refuse the rest even when the
    URL is hand-crafted rather than clicked.
    """
    tmp = tmp / "buckets"                    # own root: __main__ shares one tmp dir
    _write_run(tmp, "20260830-101010")                       # a real module run
    # a module's own variant bucket ("<id>-<kind>") is still a real run bucket
    sim = tmp / "runs" / "extrusion-quick-simulation" / "20260830-120000"
    sim.mkdir(parents=True)
    (sim / "report.json").write_text("{}", encoding="utf-8")
    # ...and these two are not runs at all: nobody registered them
    backup = tmp / "runs" / "_figures-backup-pre-3733d1b" / "figures"
    backup.mkdir(parents=True)
    (backup / "plan.png").write_bytes(b"x" * 4096)
    archive = tmp / "runs" / "characterization" / "20260813-090000"
    archive.mkdir(parents=True)
    (archive / "report.json").write_text("{}", encoding="utf-8")
    # a non-run folder parked *inside* a real bucket must not become a row either
    parked = tmp / "runs" / "calibration" / "board-photos"
    parked.mkdir(parents=True)
    (parked / "a.png").write_bytes(b"x" * 16)

    listed = {(r["module"], r["stamp"])
              for r in runs.list_runs(limit=50, root=tmp, modules=MODULES)}
    assert listed == {("calibration", "20260830-101010"),
                      ("extrusion-quick-simulation", "20260830-120000")}, listed

    # Defence in depth: the same refusal without going through the listing.
    for module, stamp in (("_figures-backup-pre-3733d1b", "figures"),   # unknown bucket
                          ("characterization", "20260813-090000"),      # unknown bucket
                          ("calibration", "board-photos")):             # not a run
        try:
            runs.delete_run(module, stamp, root=tmp, modules=MODULES)
            raise AssertionError(f"expected refusal of {module}/{stamp}")
        except ValueError:
            pass
    assert (backup / "plan.png").is_file()
    assert (archive / "report.json").is_file()
    assert (parked / "a.png").is_file()

    # the allowlist is required — a caller that forgets it fails loudly
    try:
        runs.list_runs(limit=50, root=tmp)   # type: ignore[call-arg]
        raise AssertionError("expected list_runs to require modules=")
    except TypeError:
        pass
    # a real run in a real bucket still deletes
    out = runs.delete_run("extrusion-quick-simulation", "20260830-120000",
                          root=tmp, modules=MODULES)
    assert out["files"] == 1 and not sim.exists()
    print("[buckets] non-module dirs neither listed nor deletable; real runs still go")


def test_delete_run_frees_and_guards(tmp: Path):
    tmp = tmp / "deletes"                    # own root: __main__ shares one tmp dir
    _write_run(tmp, "20260101-000000")
    _write_run(tmp, "20260620-101010")
    nested = tmp / "runs" / "calibration" / "20260101-000000" / "clouds"
    nested.mkdir()
    (nested / "a.ply").write_bytes(b"x" * 4096)
    runs.write_active("calibration", {"run_id": "20260620-101010"}, root=tmp)

    out = runs.delete_run("calibration", "20260101-000000", root=tmp, modules=MODULES)
    assert out["files"] == 3 and out["bytes"] > 4096            # counts nested files
    assert not (tmp / "runs" / "calibration" / "20260101-000000").exists()
    # only that run went: the sibling and the active.json pointer are untouched
    assert (tmp / "runs" / "calibration" / "20260620-101010").is_dir()
    assert runs.read_active("calibration", root=tmp)["run_id"] == "20260620-101010"

    # already gone / not a run dir -> RunNotFound (the pointer file is not a run)
    for stamp in ("20260101-000000", "active.json"):
        try:
            runs.delete_run("calibration", stamp, root=tmp, modules=MODULES)
            raise AssertionError(f"expected RunNotFound for {stamp!r}")
        except runs.RunNotFound:
            pass
    # a recursive delete must never be steerable out of the runs tree
    for bad in ("..", "../..", "a/b", "a\\b", "", str(tmp)):
        try:
            runs.delete_run("calibration", bad, root=tmp, modules=MODULES)
            raise AssertionError(f"expected rejection of stamp {bad!r}")
        except ValueError:
            pass
    # ...and the allowlist is not optional: forgetting it is a loud TypeError,
    # never a silent "delete anything under runs/".
    try:
        runs.delete_run("calibration", "20260620-101010", root=tmp)  # type: ignore[call-arg]
        raise AssertionError("expected delete_run to require modules=")
    except TypeError:
        pass
    assert (tmp / "runs" / "calibration" / "20260620-101010").is_dir()
    assert tmp.exists() and (tmp / "runs").is_dir()
    print("[delete]", out["files"], "files /", out["bytes"], "bytes freed; guards hold")


def test_api_refuses_non_module_dirs(tmp: Path):
    """The same guarantee at the HTTP surface: GET /api/runs never offers a
    non-module directory, and a hand-crafted DELETE /api/runs/<anything>/<x> is
    refused (400) instead of recursively deleting it."""
    from fastapi.testclient import TestClient

    from tasni.webapp.server import create_app

    tmp = tmp / "api"                        # own root: __main__ shares one tmp dir
    _write_run(tmp, "20260830-131313")
    backup = tmp / "runs" / "_figures-backup-pre-3733d1b" / "figures"
    backup.mkdir(parents=True)
    (backup / "plan.png").write_bytes(b"x" * 512)
    (tmp / "runs" / "characterization" / "20260813-090000").mkdir(parents=True)

    runs.REPO_ROOT = tmp                     # redirect the default root for this call
    try:
        client = TestClient(create_app(AppConfig()))
        listed = client.get("/api/runs?limit=50").json()["runs"]
        assert [(r["module"], r["stamp"]) for r in listed] == [
            ("calibration", "20260830-131313")], listed
        for module, stamp in (("_figures-backup-pre-3733d1b", "figures"),
                              ("characterization", "20260813-090000")):
            r = client.delete(f"/api/runs/{module}/{stamp}")
            assert r.status_code == 400, (module, r.status_code, r.text)
            assert "run bucket" in r.text
        assert (backup / "plan.png").is_file()
        assert (tmp / "runs" / "characterization" / "20260813-090000").is_dir()
        # ...while a real run still deletes through the same endpoint
        assert client.delete("/api/runs/calibration/20260830-131313").status_code == 200
        assert not (tmp / "runs" / "calibration" / "20260830-131313").exists()
    finally:
        runs.REPO_ROOT = _ORIG_ROOT
    print("[api] non-module dirs: not listed, DELETE -> 400, files intact")


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
        test_list_runs_reports_size_and_applied(tmp)
        test_only_module_buckets_are_listed_and_deletable(tmp)
        test_delete_run_frees_and_guards(tmp)
        test_api_refuses_non_module_dirs(tmp)
        test_write_active_atomic_roundtrip(tmp)
        test_apply_by_run_id_from_disk(tmp)
        test_apply_in_memory_job(tmp)
        test_apply_nothing_to_apply(tmp)
    print("\nRun-registry + apply-by-run-id tests passed.")
