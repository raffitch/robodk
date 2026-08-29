"""Camera connect ladder: prefer the direct LAN address, fall back to Tailscale.

The Jetson is reachable two ways from the workstation: directly on the cell's
Wi-Fi (fast, but a DHCP address that can move) and over Tailscale (always
routable, but relayed and slower). The client should *try direct first* and fall
back, then remember which one won so a per-pose ``grab()`` loop does not re-pay
the ladder on every frame.

The ordering logic is a pure function (``_candidates``) so ordering/dedup/cache
behaviour is deterministic; the actual failover is exercised against real
loopback sockets. ``192.0.2.x`` is TEST-NET-1 (RFC 5737) — guaranteed
unroutable, so it stands in for "the LAN path is down".

    py -3.10 -m pytest tests/test_camera_failover.py
"""
from __future__ import annotations

import json
import socket
import struct
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

from tasni.core.camera import CameraClient, CameraError  # noqa: E402
from tasni.core.config import AppConfig, CameraConfig  # noqa: E402
from tasni.webapp.server import create_app  # noqa: E402

DEAD = "192.0.2.1"          # RFC 5737 TEST-NET-1: guaranteed unroutable
DEAD2 = "192.0.2.2"


@pytest.fixture
def listener():
    """A real listening socket on loopback. A backlog completes the TCP
    handshake without an accept loop, so ``connect()`` genuinely succeeds."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(8)
    yield srv.getsockname()[1]      # the ephemeral port
    srv.close()


def _cfg(**kw) -> CameraConfig:
    kw.setdefault("connect_probe_timeout_s", 0.2)
    return CameraConfig(**kw)


# --- ordering (pure) -------------------------------------------------------

def test_tries_lan_ip_before_tailscale_ip():
    client = CameraClient(_cfg(lan_ip="10.12.171.70", ip="100.123.63.127"))
    assert client._candidates() == ["10.12.171.70", "100.123.63.127"]


def test_empty_lan_ip_disables_the_direct_path():
    client = CameraClient(_cfg(lan_ip="", ip="100.123.63.127"))
    assert client._candidates() == ["100.123.63.127"]


def test_duplicate_lan_and_tailscale_ip_is_tried_once():
    client = CameraClient(_cfg(lan_ip="100.123.63.127", ip="100.123.63.127"))
    assert client._candidates() == ["100.123.63.127"]


def test_cached_host_short_circuits_the_ladder():
    client = CameraClient(_cfg(lan_ip="10.12.171.70", ip="100.123.63.127"))
    client._host = "100.123.63.127"
    assert client._candidates() == ["100.123.63.127"]


# --- failover (real sockets) ----------------------------------------------

def test_connects_directly_when_lan_is_reachable(listener):
    client = CameraClient(_cfg(lan_ip="127.0.0.1", ip=DEAD, port=listener))
    sock, host = client._connect(timeout=0.2)
    try:
        assert host == "127.0.0.1"
        assert client.active_host == "127.0.0.1"
    finally:
        sock.close()


def test_falls_back_to_tailscale_when_lan_is_unreachable(listener):
    client = CameraClient(_cfg(lan_ip=DEAD, ip="127.0.0.1", port=listener))
    sock, host = client._connect(timeout=0.2)
    try:
        assert host == "127.0.0.1"
        assert client.active_host == "127.0.0.1"
    finally:
        sock.close()


def test_stale_cached_host_is_dropped_and_the_ladder_reruns(listener):
    """Moving between networks must recover without restarting the app."""
    client = CameraClient(_cfg(lan_ip=DEAD, ip="127.0.0.1", port=listener))
    client._host = DEAD2                     # cached from a previous network
    sock, host = client._connect(timeout=0.2)
    try:
        assert host == "127.0.0.1"
    finally:
        sock.close()


def test_error_names_every_host_tried_when_all_are_down():
    client = CameraClient(_cfg(lan_ip=DEAD, ip=DEAD2, port=9))
    with pytest.raises(CameraError) as excinfo:
        client._connect(timeout=0.2)
    message = str(excinfo.value)
    assert DEAD in message and DEAD2 in message


def test_active_host_defaults_to_configured_ip_before_first_connect():
    client = CameraClient(_cfg(lan_ip="10.12.171.70", ip="100.123.63.127"))
    assert client.active_host == "100.123.63.127"


# --- the dashboard resolves the route without a capture --------------------

def test_resolve_via_picks_the_first_reachable_candidate_and_caches_it():
    client = CameraClient(_cfg(lan_ip="10.12.171.70", ip="100.123.63.127"))
    reachable = {"100.123.63.127"}
    host, ok = client.resolve_via(lambda h, p: h in reachable)
    assert (host, ok) == ("100.123.63.127", True)
    assert client.active_host == "100.123.63.127"


def test_resolve_via_drops_a_stale_cached_host_and_reladders():
    client = CameraClient(_cfg(lan_ip="10.12.171.70", ip="100.123.63.127"))
    client._host = "10.12.171.70"                    # cached from the cell LAN
    host, ok = client.resolve_via(lambda h, p: h == "100.123.63.127")
    assert (host, ok) == ("100.123.63.127", True)


def test_resolve_via_reports_unreachable_without_caching_a_winner():
    client = CameraClient(_cfg(lan_ip="10.12.171.70", ip="100.123.63.127"))
    host, ok = client.resolve_via(lambda h, p: False)
    assert ok is False
    assert host == "100.123.63.127"                  # the configured fallback


def test_health_reports_the_direct_path_before_any_capture(listener):
    """The dashboard is what the operator reads. It must walk the ladder itself
    rather than echo the configured fallback until something happens to capture."""
    cfg = AppConfig()
    cfg.camera.lan_ip = "127.0.0.1"
    cfg.camera.ip = DEAD
    cfg.camera.port = listener
    status = TestClient(create_app(cfg)).get("/api/health").json()["camera"]
    assert status["route"] == "Direct/LAN"
    assert status["endpoint"] == f"127.0.0.1:{listener}"
    assert status["ok"] is True


# --- the real capture paths go through the ladder --------------------------

_GREETING = {  # a minimal protocol-2 greeting, just enough for CameraGeometry.from_greeting
    "protocol": 2, "depth_unit_mm": 0.1,
    "depth": {"width": 4, "height": 4, "fx": 1.0, "fy": 1.0, "ppx": 2.0, "ppy": 2.0},
    "color": {"width": 4, "height": 4, "fx": 1.0, "fy": 1.0, "ppx": 2.0, "ppy": 2.0},
    "depth_to_color": {"rotation_row_major": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                       "translation_mm": [0.0, 0.0, 0.0]},
}


def _fake_camera(port_holder: list, mode: str = "frame") -> threading.Thread:
    """Minimal stand-in for the Jetson server: one client, one reply.

    ``frame`` serves a single color-only frame in the server's wire format
    (``<I depth_len><I color_len><d ts>`` + JPEG, mirroring
    server_unicast_syncronous.py); ``burst`` answers the MODE BURST handshake,
    followed by the protocol-2 greeting (server_unicast_syncronous.py sends it
    right after ``BURST READY\\n``, before any frame -- burst frames are still
    depth+color and CameraClient.burst() now reads that greeting same as
    grab()/stream())."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(4)
    port_holder.append(srv.getsockname()[1])

    def serve():
        try:
            conn, _ = srv.accept()
            with conn:
                conn.settimeout(3.0)
                try:
                    conn.recv(64)          # MODE line (color-only / burst)
                except OSError:
                    pass
                if mode == "burst":
                    conn.sendall(b"BURST READY\n" + json.dumps(_GREETING).encode() + b"\n")
                else:
                    ok, jpeg = cv2.imencode(".jpg", np.zeros((4, 4, 3), np.uint8))
                    assert ok
                    blob = jpeg.tobytes()
                    conn.sendall(struct.pack("<IId", 0, len(blob), 1.5) + blob)
        except OSError:
            pass
        finally:
            srv.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    return thread


def test_grab_reaches_the_camera_over_the_direct_path():
    holder: list = []
    _fake_camera(holder)
    client = CameraClient(_cfg(lan_ip="127.0.0.1", ip=DEAD, port=holder[0]))
    frame = client.grab(color_only=True, timeout=3.0)
    assert frame.timestamp == 1.5
    assert client.active_host == "127.0.0.1"


def test_stream_reaches_the_camera_over_the_direct_path():
    holder: list = []
    _fake_camera(holder)
    client = CameraClient(_cfg(lan_ip="127.0.0.1", ip=DEAD, port=holder[0]))
    with client.stream(color_only=True, timeout=3.0) as feed:
        assert feed.read().timestamp == 1.5
    assert client.active_host == "127.0.0.1"


def test_burst_reaches_the_camera_over_the_direct_path():
    holder: list = []
    _fake_camera(holder, mode="burst")
    client = CameraClient(_cfg(lan_ip="127.0.0.1", ip=DEAD, port=holder[0]))
    with client.burst(timeout=3.0):
        pass
    assert client.active_host == "127.0.0.1"
