"""The server's handshake parse is the version gate: a depth client without V2 is refused."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))

from server.handshake import parse_handshake  # noqa: E402


def test_full_v2_is_the_only_accepted_depth_stream():
    assert parse_handshake(b"MODE FULL V2\n") == {
        "mode": "full", "v2": True, "codec": "jpeg", "quality": None,
        "bitrate": 4000, "scan_telemetry": False, "depth_requested": True}


def test_no_handshake_and_old_full_are_depth_requests_without_v2():
    for req in (b"", b"MODE FULL\n", b"garbage"):
        p = parse_handshake(req)
        assert p["mode"] == "full" and p["depth_requested"] and not p["v2"], req


def test_burst_needs_v2_too():
    assert parse_handshake(b"MODE BURST\n")["v2"] is False
    p = parse_handshake(b"MODE BURST V2\n")
    assert p["mode"] == "burst" and p["v2"] and p["depth_requested"]


def test_color_and_telemetry_are_unchanged_and_never_depth():
    p = parse_handshake(b"MODE COLOR H264 B6000 SCAN\n")
    assert p["mode"] == "color" and p["codec"] == "h264" and p["bitrate"] == 6000
    assert p["scan_telemetry"] and not p["depth_requested"]
    assert parse_handshake(b"MODE COLOR Q5\n")["quality"] == 10        # clamped low
    assert parse_handshake(b"MODE COLOR Q999\n")["quality"] == 100     # clamped high
    assert parse_handshake(b"C")["mode"] == "color"
    assert parse_handshake(b"MODE TELEMETRY\n")["mode"] == "telemetry"
    assert parse_handshake(b"MODE TELEMETRY\n")["depth_requested"] is False
