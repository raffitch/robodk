"""The running build must be distinguishable from the checked-out one.

Two live-cell test cycles were spent exercising stale code: the app caches
imported modules, so editing tasni/**.py changes nothing until it restarts, and
the run report recorded `git rev-parse HEAD` at report time -- naming a commit
the process had never loaded.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from tasni.core import build_info as build_info_module
from tasni.core.build_info import build_info, staleness_warning
from tasni.core.config import AppConfig
from tasni.webapp.server import create_app


def test_a_current_process_reports_its_loaded_build_and_no_warning():
    info = build_info()

    assert info["loaded_sha"] == build_info_module.LOADED_SHA
    assert info["stale"] is False
    assert info["changed_since_start"] == []
    assert "warning" not in info
    assert staleness_warning() is None
    # The checked-out commit is still reported, but under a name that says what
    # it actually is, so it cannot be mistaken for the running build.
    assert "git_commit_checked_out" in info
    assert "git_commit" not in info


def test_editing_a_packaged_source_file_marks_the_process_stale(monkeypatch):
    """Simulate an edit landing after start, which is exactly what bit us."""
    edited = dict(build_info_module._LOADED_MTIMES)
    victim = next(iter(edited))
    edited[victim] = edited[victim] - 1.0          # a different mtime == an edit
    monkeypatch.setattr(build_info_module, "_LOADED_MTIMES", edited)

    info = build_info()

    assert info["stale"] is True
    assert victim in info["changed_since_start"]
    assert info["changed_count"] >= 1
    warning = info["warning"]
    assert "STALE" in warning.upper() and "Restart" in warning
    assert victim.split("/")[-1] in warning or victim in warning
    assert staleness_warning() == warning


def test_a_deleted_packaged_file_also_counts_as_changed(monkeypatch):
    ghost = "tasni/core/a_module_that_was_removed.py"
    monkeypatch.setattr(build_info_module, "_LOADED_MTIMES",
                        {**build_info_module._LOADED_MTIMES, ghost: 1.0})

    info = build_info()

    assert info["stale"] is True
    assert ghost in info["changed_since_start"]


def test_health_exposes_the_running_build(monkeypatch):
    monkeypatch.setattr("tasni.webapp.server.tcp_probe", lambda host, port: True)
    health = TestClient(create_app(AppConfig())).get("/api/health").json()

    assert health["build"]["loaded_sha"] == build_info_module.LOADED_SHA
    assert health["build"]["stale"] is False
