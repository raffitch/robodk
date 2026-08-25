"""Live-loop latency probe (Defect 2 Phase-1 instrumentation).

These tests pin the *measurement* semantics, not a fix: the probe exists to tell a
cell run apart from a host stall, so each number it reports has to mean exactly one
thing. All hardware-free (injected clocks).
"""
import pytest

from tasni.modules.scan.live_diag import LiveLatencyProbe


class _Clock:
    """Injected monotonic + wall clocks that advance only when told to."""

    def __init__(self, mono=1000.0, wall=5000.0):
        self.mono = mono
        self.wall = wall

    def advance(self, dt, *, wall_dt=None):
        self.mono += dt
        self.wall += dt if wall_dt is None else wall_dt


def _probe(clock, period_s=5.0):
    emitted = []
    p = LiveLatencyProbe(emitted.append, period_s=period_s,
                         clock=lambda: clock.mono, wall=lambda: clock.wall)
    return p, emitted


def test_emits_nothing_before_the_period_and_once_after():
    clock = _Clock()
    p, emitted = _probe(clock)
    for _ in range(4):
        p.note_iteration()
        clock.advance(1.0)
        p.flush_if_due()
    assert emitted == []          # 4 s elapsed, period is 5 s
    p.note_iteration()
    clock.advance(1.5)
    p.flush_if_due()
    assert len(emitted) == 1
    p.note_iteration()
    p.flush_if_due()
    assert len(emitted) == 1      # window restarted, not re-emitted


def test_loop_interval_and_rpc_timings_are_reported_separately():
    """The discriminator: a stalled RoboDK RPC shows up in BOTH pose_ms and the
    loop interval; a slow Jetson shows up in neither."""
    clock = _Clock()
    p, _ = _probe(clock)
    for i, rpc_s in enumerate((0.002, 0.5, 0.004)):
        p.note_iteration()
        with p.timing("pose"):
            clock.advance(rpc_s)
        clock.advance(0.1)        # rest of the loop body
    s = p.summary()
    assert s["frames"] == 3
    assert s["pose_ms_max"] == pytest.approx(500.0, abs=1.0)
    assert s["pose_ms_p50"] == pytest.approx(4.0, abs=1.0)
    # intervals: between iteration 1->2 (0.002+0.1) and 2->3 (0.5+0.1)
    assert s["loop_ms_max"] == pytest.approx(600.0, abs=1.0)


def test_timing_records_even_when_the_call_raises():
    """RoboDK probes are wrapped in try/except; a raising call must still be timed
    or a stalling-then-failing RPC would be invisible."""
    clock = _Clock()
    p, _ = _probe(clock)
    p.note_iteration()
    with pytest.raises(RuntimeError):
        with p.timing("driver"):
            clock.advance(0.25)
            raise RuntimeError("RoboDK busy")
    assert p.summary()["driver_ms_max"] == pytest.approx(250.0, abs=1.0)


def test_repeated_telemetry_stamps_count_as_one_update():
    """The reader thread is latest-wins, so the SAME payload is re-read every frame.
    Update cadence must count distinct stamps, else it just mirrors video fps."""
    clock = _Clock()
    p, _ = _probe(clock)
    for stamp in (5000.0, 5000.0, 5000.0, 5001.0, 5001.0):
        p.note_telemetry({"timestamp": stamp})
        p.note_iteration()
        clock.advance(0.2)
    s = p.summary()
    assert s["tel_updates"] == 2
    assert s["frames"] == 5


def test_telemetry_age_exposes_clock_skew_without_clamping():
    """Age is host wall-clock minus the JETSON's stamp — a cross-machine subtraction.
    A Jetson clock running ahead yields a NEGATIVE age; clamping it would hide the
    skew that silently trips the 2 s staleness drop."""
    clock = _Clock(wall=5000.0)
    p, _ = _probe(clock)
    p.note_telemetry({"timestamp": 5008.0})     # stamped 8 s in the future
    assert p.summary()["tel_age_ms_p50"] == pytest.approx(-8000.0, abs=1.0)


def test_counts_dropped_frames_and_holds_and_missing_telemetry():
    clock = _Clock()
    p, _ = _probe(clock)
    # telemetry arrived but the staleness gate emptied the payload
    p.note_telemetry({"timestamp": 4990.0})
    p.note_result({}, telemetry_present=True)
    # a normal held frame
    p.note_telemetry({"timestamp": 5000.0})
    p.note_result({"held": True}, telemetry_present=True)
    # a live frame
    p.note_result({"held": False}, telemetry_present=True)
    # no telemetry on the socket at all
    p.note_telemetry(None)
    p.note_result({}, telemetry_present=False)
    s = p.summary()
    assert (s["dropped"], s["held"], s["no_telemetry"]) == (1, 1, 1)


def test_summary_is_safe_with_no_samples():
    """flush_if_due runs on a loop that may not have completed a single frame."""
    clock = _Clock()
    p, emitted = _probe(clock)
    s = p.summary()
    assert s["frames"] == 0 and s["fps"] == 0.0
    assert s["loop_ms_p50"] is None and s["tel_age_ms_p50"] is None
    clock.advance(10.0)
    p.flush_if_due()
    assert emitted == []          # nothing observed -> nothing to report


def test_malformed_telemetry_never_raises():
    """The probe sits in the video hot path; it must never take down the preview."""
    clock = _Clock()
    p, _ = _probe(clock)
    for bad in ({"timestamp": None}, {"timestamp": "not-a-number"}, {}, None, 42):
        p.note_telemetry(bad)
    assert p.summary()["tel_updates"] == 0


def test_emit_failure_is_swallowed():
    clock = _Clock()

    def boom(_msg):
        raise OSError("log sink closed")

    p = LiveLatencyProbe(boom, period_s=1.0,
                         clock=lambda: clock.mono, wall=lambda: clock.wall)
    p.note_iteration()
    clock.advance(2.0)
    p.flush_if_due()              # must not propagate


def test_emitted_rates_use_the_window_that_just_closed():
    """Regression: the emitted line must divide by the ELAPSED window, not by a
    window already reset to zero — fps and the telemetry Hz (the discriminator that
    separates a slow Jetson from a host stall) are otherwise meaningless."""
    clock = _Clock()
    p, emitted = _probe(clock, period_s=5.0)
    for i in range(10):
        p.note_iteration()
        p.note_telemetry({"timestamp": clock.wall})   # a distinct stamp each frame
        clock.advance(0.5)
    p.note_iteration()
    p.flush_if_due()
    line = emitted[0]
    assert "5.0s" in line          # window length, not ~0
    assert "(2.2 fps)" in line     # 11 frames / 5.0 s
    assert "(2.00 Hz)" in line     # 10 distinct stamps / 5.0 s


def test_window_restarts_from_the_flush_not_from_the_last_frame():
    """Two consecutive windows must each report their own elapsed time."""
    clock = _Clock()
    p, emitted = _probe(clock, period_s=2.0)
    for _ in range(2):
        for _ in range(4):
            p.note_iteration()
            clock.advance(0.5)
        p.flush_if_due()
    assert len(emitted) == 2
    assert "2.0s" in emitted[0] and "2.0s" in emitted[1]


def test_disabled_probe_records_nothing_and_never_emits():
    """period_s <= 0 is the off switch. It must stop RECORDING too — a probe that
    accumulates samples it will never emit grows without bound over a long preview."""
    clock = _Clock()
    emitted = []
    p = LiveLatencyProbe(emitted.append, period_s=0.0,
                         clock=lambda: clock.mono, wall=lambda: clock.wall)
    for _ in range(50):
        p.note_iteration()
        with p.timing("pose"):
            clock.advance(0.01)
        p.note_telemetry({"timestamp": clock.wall})
        p.note_result({"held": True}, telemetry_present=True)
        clock.advance(1.0)
        p.flush_if_due()
    s = p.summary()
    assert emitted == []
    assert s["frames"] == 0 and s["tel_updates"] == 0 and s["held"] == 0
    assert "pose_ms_p50" not in s


def test_disabled_timing_still_runs_the_wrapped_block():
    clock = _Clock()
    p = LiveLatencyProbe(lambda _m: None, period_s=0.0,
                         clock=lambda: clock.mono, wall=lambda: clock.wall)
    ran = []
    with p.timing("pose"):
        ran.append(True)
    assert ran == [True]


def test_formatted_line_names_every_discriminator():
    clock = _Clock()
    p, emitted = _probe(clock, period_s=1.0)
    p.note_iteration()
    with p.timing("pose"):
        clock.advance(0.01)
    with p.timing("driver"):
        clock.advance(0.02)
    p.note_telemetry({"timestamp": clock.wall - 0.1})
    p.note_result({"held": True}, telemetry_present=True)
    clock.advance(1.0)
    p.note_iteration()
    p.flush_if_due()
    line = emitted[0]
    for token in ("fps", "loop", "pose", "driver", "telemetry", "age", "held"):
        assert token in line
