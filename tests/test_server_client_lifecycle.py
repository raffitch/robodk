"""The camera server must never leak a client socket when a handler dies.

``handle_client`` closes ``conn`` on its normal-return paths only. When the
RealSense pipeline wedges, ``getFrames`` raises ``RuntimeError: Frame didn't
arrive within 5000``, the exception escapes the handler and the thread dies with
a traceback -- leaving the accepted socket open forever in CLOSE_WAIT. Against
``listen(5)`` a handful of those stop new clients connecting, so a *recoverable*
camera stall degrades into a server that has to be restarted by hand. Observed
on the cell 2026-08-28: 10 leaked CLOSE_WAIT sockets after the camera wedged.

    py -3.10 -m pytest tests/test_server_client_lifecycle.py
"""
from __future__ import annotations

import socket
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
sys.modules.setdefault("pyrealsense2", SimpleNamespace())
sys.modules.setdefault("turbojpeg", SimpleNamespace())

from server import server_unicast_syncronous as srv  # noqa: E402


def _pair():
    a, b = socket.socketpair()
    return a, b


def test_socket_is_closed_when_the_handler_raises(monkeypatch):
    """The wedged-camera case: librealsense raises straight out of the handler."""
    def wedged(conn, addr):
        raise RuntimeError("Frame didn't arrive within 5000")

    monkeypatch.setattr(srv, "handle_client", wedged)
    conn, peer = _pair()
    try:
        srv._serve_client(conn, ("10.12.172.19", 44796))     # must not propagate
        assert conn.fileno() == -1, "client socket leaked after handler raised"
    finally:
        conn.close()
        peer.close()


def test_socket_is_closed_when_the_handler_returns_normally(monkeypatch):
    monkeypatch.setattr(srv, "handle_client", lambda conn, addr: None)
    conn, peer = _pair()
    try:
        srv._serve_client(conn, ("10.12.172.19", 44797))
        assert conn.fileno() == -1
    finally:
        conn.close()
        peer.close()


def test_a_dead_handler_does_not_stop_the_next_client(monkeypatch):
    """The actual operational failure: one wedge must not poison the backlog."""
    calls = []

    def flaky(conn, addr):
        calls.append(addr)
        if len(calls) == 1:
            raise RuntimeError("Frame didn't arrive within 5000")

    monkeypatch.setattr(srv, "handle_client", flaky)
    for port in (1, 2):
        conn, peer = _pair()
        try:
            srv._serve_client(conn, ("10.12.172.19", port))
            assert conn.fileno() == -1
        finally:
            conn.close()
            peer.close()
    assert len(calls) == 2, "second client was never served"


def test_accept_loop_uses_the_guarded_wrapper():
    """Guard against the wrapper existing but main() still spawning the raw
    handler -- that would pass every test above and leak in production."""
    import inspect
    source = inspect.getsource(srv.main)
    assert "_serve_client" in source
    assert "target=handle_client" not in source
