"""livepreview.py — the shared preview loop (frame + gate publishing).

Regression coverage for the scan-preview flap: the scan gate arrives on a SEPARATE
telemetry channel piggybacked on the video stream (~0.4 s cadence), so most early
frames carry no metrics. The loop must HOLD ONE stream open until telemetry lands —
not break+reconnect on every metrics-less frame (which starved telemetry, so the HUD
flapped 'no signal'<->'streaming' and 'ready'<->'hold').

    py -3.10 tests/test_livepreview.py
"""
from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tasni.core.livepreview import LivePreview  # noqa: E402


class _FakeBus:
    def __init__(self):
        self.events = []

    def publish(self, ev):
        self.events.append(ev)


def _run_until(pred, lp, *, timeout_s=3.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if pred():
            return
        time.sleep(0.01)


def test_scan_preview_holds_one_stream_when_telemetry_is_sparse():
    # Frames before the telemetry channel delivers carry telemetry=None (-> empty
    # metrics); a later frame carries a populated reading.
    tel = {"gates": {"detected": True, "distance": True, "angle": True}, "ok": True}
    frames = [SimpleNamespace(color=0, depth=None, telemetry=None),
              SimpleNamespace(color=0, depth=None, telemetry=None),
              SimpleNamespace(color=0, depth=None, telemetry=tel)]

    class FakeStream:
        def __init__(self):
            self.i = 0

        def read(self, *, with_depth=False, drain=False):
            f = frames[min(self.i, len(frames) - 1)]   # repeat the last (telemetry) frame
            self.i += 1
            return f

    class FakeCamera:
        def __init__(self):
            self.opens = 0

        @contextmanager
        def stream(self, **kw):
            self.opens += 1
            yield FakeStream()

        def grab(self, **kw):
            raise AssertionError("scan preview must not grab depth (no interleave)")

    cam, bus = FakeCamera(), _FakeBus()
    lp = LivePreview(cam, bus, lease=None)

    def analyze(frame):
        return b"jpg", (dict(frame.telemetry) if frame.telemetry else {})

    lp.start(analyze, fps=200, scan_telemetry=True)
    try:
        _run_until(lambda: any(e.type == "gate" for e in list(bus.events)), lp)
    finally:
        lp.stop()

    gates = [e for e in bus.events if e.type == "gate"]
    # The whole point: ONE connection, held open — not a reconnect per metrics-less frame.
    assert cam.opens == 1, f"stream reconnected {cam.opens}x — it must hold one open"
    assert gates, "telemetry frame never produced a gate (stream torn down too early?)"
    assert gates[-1].payload.get("ok") is True
    assert all(not e.payload.get("error") for e in gates), "stream errored/flapped"
    assert any(e.type == "frame" for e in bus.events), "no video frames published"
    print("[livepreview] scan stream held 1 connection across",
          sum(e.type == "frame" for e in bus.events), "frames; gate published")


if __name__ == "__main__":
    test_scan_preview_holds_one_stream_when_telemetry_is_sparse()
    print("\nlivepreview tests passed.")
