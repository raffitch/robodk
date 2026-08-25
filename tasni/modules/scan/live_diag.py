"""Latency instrumentation for the scan live-preview loop.

This exists to settle Defect 2 (the live readout lagging ~10 s behind the arm on
the real cell) by *measurement* rather than by guessing between the candidates,
which are not distinguishable from the operator's description alone:

* **Host RoboDK stall** — ``camera_pose_T()`` / ``robot_connected()`` are
  synchronous RoboDK round trips (and each re-resolves the robot ``Item`` first,
  so they are 3 and 2 RPCs respectively). Under a live driver these can block.
  Signature: ``pose_ms`` / ``driver_ms`` high **and** ``loop`` interval high.
* **Jetson telemetry cadence** — the server computes the plane fit, rectangle,
  density trim and coverage grid inline on its frame feeder, nominally every
  ``SCAN_TELEMETRY_PERIOD_S`` (1 s). Signature: ``loop`` interval and the RPCs
  are *fine* but ``telemetry`` update rate is far below 1 Hz.
* **The 2 s staleness drop** in ``live_scan_telemetry_payload`` — it compares the
  host's ``time.time()`` against a timestamp stamped on the *Jetson*. That is a
  cross-machine wall-clock subtraction, and a Nano has no RTC battery, so a
  skewed clock silently discards every payload. Signature: ``dropped`` climbing,
  and ``age`` far from ~0 (negative age = Jetson clock ahead of the host).
* **The hold/freeze logic** — signature: ``held`` at or near the frame count
  while the arm is being jogged.

One line every ``period_s`` distinguishes all four in a single cell run, which is
what the defect note asks for ("do not guess between them").

The probe is pure bookkeeping over injected clocks: it performs no I/O beyond the
``emit`` callback, and — because it sits in the video hot path — it never raises
into its caller.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from statistics import median

__all__ = ["LiveLatencyProbe"]


def _p50(values: list[float]) -> float | None:
    return float(median(values)) if values else None


def _max(values: list[float]) -> float | None:
    return float(max(values)) if values else None


def _ms(seconds: float | None) -> float | None:
    return None if seconds is None else seconds * 1000.0


class LiveLatencyProbe:
    """Accumulate live-loop timings and emit a periodic one-line summary.

    ``emit`` takes the formatted line (typically ``log.info``). ``clock`` is a
    monotonic source used for every duration; ``wall`` is a wall-clock source used
    only to age Jetson telemetry stamps against, since those are wall-clock on the
    far side of the link.

    ``period_s <= 0`` disables the probe outright: nothing is recorded (rather than
    recorded-but-never-emitted, which would grow the sample lists without bound over
    a long preview) and nothing is emitted.
    """

    def __init__(self, emit, *, period_s: float = 5.0,
                 clock=time.monotonic, wall=time.time):
        self._emit = emit
        self._period_s = float(period_s)
        self._enabled = self._period_s > 0.0
        self._clock = clock
        self._wall = wall
        self._window_start = clock()
        self._last_iter: float | None = None
        # Deliberately NOT cleared per window: distinct-stamp counting has to span
        # the window boundary or the first frame of every window re-counts the
        # payload the previous window already reported.
        self._last_stamp: float | None = None
        self._reset()

    def _reset(self) -> None:
        self._frames = 0
        self._intervals: list[float] = []
        self._rpc: dict[str, list[float]] = {}
        self._tel_ages: list[float] = []
        self._tel_updates = 0
        self._dropped = 0
        self._held = 0
        self._no_telemetry = 0

    # -- recording ---------------------------------------------------------
    def note_iteration(self) -> None:
        """Mark one pass of the live loop (call once per analysed frame)."""
        if not self._enabled:
            return
        now = self._clock()
        if self._last_iter is not None:
            self._intervals.append(now - self._last_iter)
        self._last_iter = now
        self._frames += 1

    @contextmanager
    def timing(self, name: str):
        """Time a named call. Records in ``finally`` so a call that stalls *and
        then raises* (RoboDK busy) is still measured — the failing case is exactly
        the one worth seeing."""
        if not self._enabled:
            yield
            return
        t0 = self._clock()
        try:
            yield
        finally:
            self._rpc.setdefault(name, []).append(self._clock() - t0)

    def note_telemetry(self, telemetry) -> None:
        """Record the raw Jetson telemetry dict this frame read (or ``None``)."""
        if not self._enabled:
            return
        try:
            if not telemetry:
                self._no_telemetry += 1
                return
            stamp = telemetry.get("timestamp")
            if stamp is None:
                return
            stamp = float(stamp)
        except (AttributeError, TypeError, ValueError):
            return
        # Cross-machine subtraction on purpose: this is the very quantity the 2 s
        # staleness gate tests, so a clock skew must show up here rather than be
        # normalised away.
        self._tel_ages.append(self._wall() - stamp)
        if stamp != self._last_stamp:
            self._last_stamp = stamp
            self._tel_updates += 1

    def note_result(self, metrics, *, telemetry_present: bool = False) -> None:
        """Record what the frame's metrics ended up as."""
        if not self._enabled:
            return
        try:
            if not metrics:
                if telemetry_present:
                    self._dropped += 1
                return
            if metrics.get("held"):
                self._held += 1
        except AttributeError:
            return

    # -- reporting ---------------------------------------------------------
    def summary(self) -> dict:
        elapsed = max(self._clock() - self._window_start, 1e-9)
        out = {
            "elapsed_s": elapsed,
            "frames": self._frames,
            "fps": self._frames / elapsed if self._frames else 0.0,
            "loop_ms_p50": _ms(_p50(self._intervals)),
            "loop_ms_max": _ms(_max(self._intervals)),
            "tel_updates": self._tel_updates,
            "tel_hz": self._tel_updates / elapsed if self._tel_updates else 0.0,
            "tel_age_ms_p50": _ms(_p50(self._tel_ages)),
            "dropped": self._dropped,
            "held": self._held,
            "no_telemetry": self._no_telemetry,
        }
        for name, samples in self._rpc.items():
            out[f"{name}_ms_p50"] = _ms(_p50(samples))
            out[f"{name}_ms_max"] = _ms(_max(samples))
        return out

    def _format(self, s: dict) -> str:
        def num(v, digits=0):
            return "-" if v is None else f"{v:.{digits}f}"

        parts = [
            f"live-latency {s['elapsed_s']:.1f}s: {s['frames']} frames "
            f"({s['fps']:.1f} fps)",
            f"loop p50/max {num(s['loop_ms_p50'])}/{num(s['loop_ms_max'])} ms",
        ]
        for name in sorted(self._rpc):
            parts.append(f"{name} p50/max {num(s[f'{name}_ms_p50'])}/"
                         f"{num(s[f'{name}_ms_max'])} ms")
        parts.append(
            f"telemetry {s['tel_updates']} upd ({s['tel_hz']:.2f} Hz) "
            f"age p50 {num(s['tel_age_ms_p50'])} ms")
        parts.append(
            f"dropped {s['dropped']} held {s['held']} "
            f"no-tel {s['no_telemetry']}")
        return " | ".join(parts)

    def flush_if_due(self) -> None:
        """Emit and restart the window if ``period_s`` has elapsed. Silent when the
        window saw nothing at all, and never propagates a failure from ``emit``."""
        if not self._enabled:
            return
        now = self._clock()
        if now - self._window_start < self._period_s:
            return
        # summary() divides by (now - _window_start), so the window must NOT be
        # restarted until after it is taken — otherwise every rate in the emitted
        # line is divided by ~0, which is precisely what makes the telemetry Hz
        # figure (the slow-Jetson discriminator) useless.
        if self._frames or self._tel_ages or self._no_telemetry:
            try:
                self._emit(self._format(self.summary()))
            except Exception:  # noqa: BLE001 - diagnostics must never break preview
                pass
        self._window_start = now
        self._reset()
