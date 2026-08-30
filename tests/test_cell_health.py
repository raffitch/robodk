"""``tools/cell_health.py`` must not FAIL a healthy cell.

The morning readiness check proves each device option from the camera service's own
journal READ-BACK lines rather than by opening the device. That design is right --
the service holds the D435i exclusively -- but it makes every check a REGEX against
text produced somewhere else in this repo, and a regex that does not match anything
the server has ever printed is indistinguishable, in the report, from a genuinely
broken cell. It reports FAIL with a remedy ("py -3.10 tools/jetson_deploy.py deploy")
that cannot possibly help, hours before a capture run.

That had already happened: the ``stream resolutions`` pattern was written as
``depth \\(1280, 720\\) colour \\(1920, 1080\\)`` while the server prints
``depth (1280, 720), colour (1920, 1080), protocol 2`` -- one missing comma, and the
check could never go green on any cell, ever.

So these tests do not hand-write the expected log lines. They GENERATE them from the
production code:

  * the device-option lines come from calling ``server.rs_config.configure_depth_sensor``
    with a fake sensor that answers the way the real D435i does (its ``depth_units``
    read-back is float32(1e-4), i.e. 9.999999747378752e-05, which is why the
    ``depth_unit_mm`` line really reads 0.09999999747378752 and not 0.1);
  * the startup banner is lifted out of ``server/server_unicast_syncronous.py``'s own
    source with the module's real ``DEPTH_SIZE``/``COLOR_SIZE`` substituted in.

so a change to either side of the contract fails here instead of on the cell.

    py -3.10 -m pytest tests/test_cell_health.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "server"))
sys.path.insert(0, str(ROOT / "tools"))
sys.modules.setdefault("pyrealsense2", SimpleNamespace())
sys.modules.setdefault("turbojpeg", SimpleNamespace())

import cell_health  # noqa: E402
from server import rs_config  # noqa: E402
from server import server_unicast_syncronous as srv  # noqa: E402


# What the D435i actually reports back after depth_units is set to 1e-4: the device
# stores it as a float32, so the read-back is float32(1e-4) widened to double. This
# is the reason ``depth_unit_mm`` prints 0.09999999747378752 in the live journal.
DEVICE_DEPTH_UNITS = 9.999999747378752e-05

FAKE_RS = SimpleNamespace(option=SimpleNamespace(
    emitter_enabled="emitter_enabled", laser_power="laser_power",
    visual_preset="visual_preset", depth_units="depth_units",
    auto_exposure_priority="auto_exposure_priority",
    asic_temperature="asic_temperature", projector_temperature="projector_temperature",
    global_time_enabled="global_time_enabled"), __version__="2.55.1")


class HealthyDepthSensor:
    """The cell as it is configured today: laser pinned to 150 by the systemd unit,
    the visual preset left as-is on Custom (0), and depth_units accepted."""

    SUPPORTED = ("emitter_enabled", "laser_power", "visual_preset", "depth_units")
    RANGES = {"emitter_enabled": (0.0, 1.0), "laser_power": (0.0, 360.0),
              "visual_preset": (0.0, 5.0), "depth_units": (1e-05, 0.01)}

    def __init__(self):
        self.values = {"laser_power": 150.0, "visual_preset": 0.0}
        self.depth_scale = 0.001

    def supports(self, opt):
        return opt in self.SUPPORTED

    def get_option_range(self, opt):
        lo, hi = self.RANGES[opt]
        return SimpleNamespace(min=lo, max=hi)

    def set_option(self, opt, value):
        if opt == "depth_units":                     # quantised to float32 by the device
            self.values[opt] = DEVICE_DEPTH_UNITS
            self.depth_scale = DEVICE_DEPTH_UNITS
        else:
            self.values[opt] = float(value)

    def get_option(self, opt):
        return self.values.get(opt, 0.0)

    def get_depth_scale(self):
        return self.depth_scale


def _startup_banner() -> str:
    """The exact banner the live server prints, taken from its own source.

    It lives inside ``if __name__ == '__main__':`` so it cannot be called from here.
    Reading the literal out of the source (and substituting the module's real size
    constants) is what keeps this test coupled to the actual line rather than to a
    copy of it that is free to drift -- which is exactly how the missing comma
    survived review.
    """
    src = (ROOT / "server" / "server_unicast_syncronous.py").read_text(encoding="utf-8")
    # Pin the ONE literal this test depends on, not "the first Initiating line in
    # the file": that server is edited constantly and by other people, so match on
    # the two size fields the banner is here to carry and require exactly one hit.
    # Unrelated edits elsewhere in the module cannot move this; a change to the
    # banner itself still fails here, which is the whole point.
    hits = [m for m in re.findall(r'print\(f"(Initiating[^"]*)"\)', src)
            if "{DEPTH_SIZE}" in m and "{COLOR_SIZE}" in m]
    assert len(hits) == 1, (
        "expected exactly one single-line print(f\"Initiating ... {DEPTH_SIZE} ... "
        f"{{COLOR_SIZE}} ...\") startup banner in the server; found {len(hits)}. "
        "If the banner was reworded or wrapped, update cell_health's 'stream "
        "resolutions' pattern and this lift together.")
    line = (hits[0]
            .replace("{DEPTH_SIZE}", str(srv.DEPTH_SIZE))
            .replace("{COLOR_SIZE}", str(srv.COLOR_SIZE)))
    assert "{" not in line, f"unsubstituted f-string field in the banner: {line!r}"
    return line


def healthy_journal() -> str:
    """The startup lines a healthy cell really writes, produced by the real code."""
    lines = [_startup_banner()]
    rs_config.configure_depth_sensor(
        HealthyDepthSensor(), FAKE_RS,
        laser_power=150.0,        # pinned by Environment=RS_LASER_POWER=150 in the unit
        visual_preset=-1,         # leave-alone, so the Custom preset is reported as found
        log=lines.append)
    return "\n".join(lines)


@pytest.mark.parametrize("title,pattern,_remedy", cell_health.JOURNAL_CHECKS,
                         ids=[c[0] for c in cell_health.JOURNAL_CHECKS])
def test_every_journal_check_matches_a_healthy_cell(title, pattern, _remedy):
    """No check may FAIL on a cell that is configured exactly as intended.

    A pattern that cannot match healthy output is worse than no check at all: the
    report says NOT READY and prints a remedy that changes nothing.
    """
    journal = healthy_journal()
    assert re.search(pattern, journal), (
        f"cell_health's {title!r} check cannot match a healthy cell.\n"
        f"pattern: {pattern}\njournal:\n{journal}")


def test_the_stream_resolution_check_reads_the_servers_own_banner():
    """The specific regression: the banner separates the two tuples with a COMMA.

    Pinned separately from the parametrised sweep above so the intent survives even
    if the banner is one day reworded -- and so a future edit that drops the comma
    from the pattern again fails with this name on it.
    """
    pattern = dict((c[0], c[1]) for c in cell_health.JOURNAL_CHECKS)["stream resolutions"]
    banner = _startup_banner()
    assert "depth (1280, 720), colour (1920, 1080)" in banner, banner
    assert re.search(pattern, banner), (pattern, banner)


def test_journal_checks_do_not_match_a_misconfigured_cell():
    """The counter-test: these patterns must still be able to say NO.

    A pattern loosened until it matches everything would pass the sweep above and be
    useless on the cell, so each check is also shown to reject the wrong value.
    """
    class Wrong(HealthyDepthSensor):
        def __init__(self):
            super().__init__()
            self.values = {"laser_power": 300.0, "visual_preset": 3.0}

        def set_option(self, opt, value):             # a device that ignores writes
            if opt == "depth_units":
                self.depth_scale = 0.001              # still 1 mm words
                self.values[opt] = 0.001
            elif opt == "emitter_enabled":
                self.values[opt] = 0.0                # emitter stayed off
            else:
                self.values[opt] = float(value)

    lines = ["Initiating Jetson-Realsense Wi-Fi Server: depth (640, 480), "
             "colour (1280, 720), protocol 2"]
    rs_config.configure_depth_sensor(Wrong(), FAKE_RS, laser_power=300.0,
                                     visual_preset=-1, log=lines.append)
    journal = "\n".join(lines)
    for title, pattern, _ in cell_health.JOURNAL_CHECKS:
        assert not re.search(pattern, journal), (
            f"{title!r} matched a MISCONFIGURED cell: {pattern}\n{journal}")


# ===========================================================================
# A check that reports OK on a broken condition is worse than no check at all.
# Everything below pins a check that used to do exactly that.
# ===========================================================================

import numpy as np                                                    # noqa: E402
from tasni.core.camera import CameraClient                            # noqa: E402
from tasni.core.config import CameraConfig                            # noqa: E402


@pytest.fixture(autouse=True)
def _clean_results():
    """``record()`` appends to a module global; every test starts from empty."""
    cell_health.RESULTS.clear()
    yield
    cell_health.RESULTS.clear()


def _depth_frame(distance_mm, wire_unit_mm, *, noise_mm=6.0, seed=0, shape=(720, 1280)):
    """A depth image as the SERVER would put it on the wire.

    ``wire_unit_mm`` is the real mm-per-word of the stream. The server clips every
    frame through ``rs.threshold_filter(0.15, 1.5)`` before shipping it, so the
    words are clipped in TRUE metric -- which is the invariant the scale check
    leans on, and the reason this helper clips too.
    """
    rng = np.random.default_rng(seed)
    mm = rng.normal(distance_mm, noise_mm, shape)
    mm = np.clip(mm, cell_health.DEPTH_CLIP_MIN_MM, cell_health.DEPTH_CLIP_MAX_MM)
    return np.rint(mm / wire_unit_mm).astype(np.uint16)


# --------------------------------------------------- defect 1: the depth scale

def test_depth_scale_accepts_a_healthy_protocol_2_frame():
    """0.1 mm words on the wire, 0.1 mm/word in the greeting, a 450 mm surface."""
    status, detail, _ = cell_health.evaluate_depth_scale(_depth_frame(450.0, 0.1), 0.1)
    assert status == cell_health.OK, detail
    assert "450" in detail


def test_depth_scale_rejects_1mm_words_behind_a_0_1mm_greeting():
    """THE point of this check.

    A Jetson serving the pre-protocol-2 1 mm stream while the greeting still says
    0.1 mm/word makes the host read 45 mm for a 450 mm surface -- every distance,
    every point cloud, every measured ring height a tenth of the truth, with no
    error anywhere. This is the exact condition the old check's FAIL text claimed
    to detect and could not.
    """
    status, detail, remedy = cell_health.evaluate_depth_scale(_depth_frame(450.0, 1.0), 0.1)
    assert status == cell_health.FAIL, f"a 10x scale error reported {status}: {detail}"
    assert "45 mm" in detail, detail
    assert "deploy" in remedy


def test_depth_scale_rejects_the_mismatch_in_the_other_direction():
    """0.1 mm words read with a 1 mm/word greeting: everything reads 10x LONG,
    beyond the 1.5 m the server's threshold filter can even emit."""
    status, detail, _ = cell_health.evaluate_depth_scale(_depth_frame(450.0, 0.1), 1.0)
    assert status == cell_health.FAIL, detail
    assert "FARTHER" in detail, detail


def test_the_old_min_step_statistic_could_not_tell_those_two_apart():
    """Why the check had to be rewritten rather than tightened.

    ``min(diff(unique(words)))`` is 1 word on any real noisy patch no matter how
    big a word is, so ``min_step * depth_unit_mm`` always returned depth_unit_mm --
    it re-tested a number the 'depth unit' check had already asserted, and passed
    the broken stream. Pinned so nobody reintroduces it as a "cheaper" check.
    """
    healthy, broken = _depth_frame(450.0, 0.1), _depth_frame(450.0, 1.0)
    steps = {name: float(np.diff(np.unique(f[f > 0])).min()) * 0.1
             for name, f in (("healthy", healthy), ("broken", broken))}
    assert steps["healthy"] == steps["broken"] == pytest.approx(0.1)
    # ... while the check that replaced it separates them.
    assert cell_health.evaluate_depth_scale(healthy, 0.1)[0] == cell_health.OK
    assert cell_health.evaluate_depth_scale(broken, 0.1)[0] == cell_health.FAIL


@pytest.mark.parametrize("depth,why", [
    (np.zeros((720, 1280), np.uint16), "an empty scene"),
    (_depth_frame(450.0, 0.1)[:8, :8], "a handful of valid pixels"),
])
def test_depth_scale_says_NOT_CHECKED_rather_than_OK_without_evidence(depth, why):
    """Nothing in view (or everything beyond the 1.5 m clip) is not evidence of a
    healthy word size. Saying so is the honest answer; OK would be a lie and FAIL
    would be a false alarm on a camera that is merely pointed at the ceiling."""
    status, detail, remedy = cell_health.evaluate_depth_scale(depth, 0.1)
    assert status == cell_health.WARN, f"{why} reported {status}: {detail}"
    assert "NOT CHECKED" in detail
    assert "UNVERIFIED" in remedy


# ------------------------------------ defect 2: the route, and the silent skip

LAN_IP, TS_IP = "10.12.171.70", "100.123.63.127"
REFUSAL = b"ERR protocol 2 required; send MODE FULL V2\n"


class FakeSocket:
    def __init__(self, reply=REFUSAL):
        self._reply, self.closed = reply, False

    def sendall(self, data):
        pass

    def settimeout(self, t):
        pass

    def recv(self, n):
        return self._reply[:n]

    def close(self):
        self.closed = True


def _connector(reachable, attempts):
    """A ``socket.create_connection`` stand-in that only ``reachable`` hosts answer."""
    def connect(address, timeout=None):
        host, port = address
        attempts.append(host)
        if host not in reachable:
            raise OSError(f"[fake] no route to {host}:{port}")
        return FakeSocket()
    return connect


def _client(**over):
    cfg = CameraConfig(ip=TS_IP, lan_ip=LAN_IP, port=1024, timeout_s=1.0,
                       connect_probe_timeout_s=0.1, resolution="1920x1080", **over)
    return CameraClient(cfg)


def _fake_frame(depth=None, *, unit=0.1, color_size=(1920, 1080)):
    geom = SimpleNamespace(depth_unit_mm=unit, color_size=color_size)
    return SimpleNamespace(geometry=geom,
                           depth=_depth_frame(450.0, 0.1) if depth is None else depth)


def _by_title():
    return {r[1]: r for r in cell_health.RESULTS}


def test_camera_checks_walk_the_same_host_ladder_as_a_real_capture():
    """The check probed ``cam.ip`` -- the TAILSCALE address -- while CameraClient
    tries ``cam.lan_ip`` FIRST. With Tailscale down on the workstation (routine),
    that FAILed against a cell that captures perfectly over the LAN."""
    attempts = []
    client = _client()
    cell_health.check_camera_protocol(
        client, connect=_connector({LAN_IP}, attempts), grab=lambda c: _fake_frame())
    assert attempts[0] == LAN_IP, f"probed {attempts} -- the LAN route must come first"
    route = _by_title()["camera route"]
    assert route[0] == cell_health.OK, route
    assert LAN_IP in route[2]
    assert _by_title()["stale-client refusal"][0] == cell_health.OK


def test_the_resolved_host_is_reused_for_the_grab():
    """Not a second ladder walk: the grab must land on the host the probe won,
    or the report describes one route while the capture uses another."""
    grabbed = []

    def grab(client):
        grabbed.append(client.active_host)
        return _fake_frame()

    cell_health.check_camera_protocol(
        _client(), connect=_connector({LAN_IP}, []), grab=grab)
    assert grabbed == [LAN_IP]


def test_a_failed_grab_does_not_silently_drop_the_remaining_checks():
    """The old code returned on a failed grab, so 'depth unit', 'greeting vs
    config' and the quantisation check simply vanished from the report -- and a
    check that is missing reads exactly like a check that passed."""
    def boom(_client):
        raise RuntimeError("[fake] no frame")

    cell_health.check_camera_protocol(
        _client(), connect=_connector({LAN_IP}, []), grab=boom)
    seen = _by_title()
    for title in cell_health.CAMERA_CHECK_TITLES:
        assert title in seen, f"{title!r} vanished from the report: {sorted(seen)}"
    assert seen["depth grab"][0] == cell_health.FAIL
    for title in ("depth unit", "greeting vs config", "depth scale"):
        assert seen[title][0] == cell_health.WARN
        assert "NOT CHECKED" in seen[title][2]


def test_an_unreachable_camera_does_not_silently_drop_the_remaining_checks():
    def never(_client):
        raise AssertionError("must not grab from an unreachable camera")

    attempts = []
    cell_health.check_camera_protocol(
        _client(), connect=_connector(set(), attempts), grab=never)
    assert set(attempts) == {LAN_IP, TS_IP}, f"both routes must be tried: {attempts}"
    seen = _by_title()
    assert seen["camera route"][0] == cell_health.FAIL
    for title in cell_health.CAMERA_CHECK_TITLES:
        assert title in seen, f"{title!r} vanished from the report: {sorted(seen)}"
        assert seen[title][0] == cell_health.WARN
        assert "NOT CHECKED" in seen[title][2]


def test_the_scale_check_runs_on_the_real_frame_the_camera_returned():
    """The wiring, not the maths: a broken stream must reach the report as FAIL."""
    cell_health.check_camera_protocol(
        _client(), connect=_connector({LAN_IP}, []),
        grab=lambda c: _fake_frame(_depth_frame(450.0, 1.0)))
    assert _by_title()["depth scale"][0] == cell_health.FAIL


# ----------------------------------------- defect 3: which process is reported

def test_backend_freshness_reports_the_OLDEST_matching_process():
    """max() green-lights the exact trap this check exists to catch: a STALE
    backend still holding :8000 while any newer python process also matches."""
    stale, fresh = 1_700_000_000_000, 1_700_000_900_000
    start, unreadable = cell_health._oldest_start(
        [{"CreationDate": f"/Date({fresh})/"}, {"CreationDate": f"/Date({stale})/"}])
    assert start == stale / 1000.0
    assert unreadable == 0


def test_an_unreadable_start_time_does_not_poison_the_reduction():
    ok = 1_700_000_000_000
    start, unreadable = cell_health._oldest_start(
        [{"CreationDate": None}, {"CreationDate": f"/Date({ok})/"}])
    assert start == ok / 1000.0 and unreadable == 1


def _fake_ps(payload):
    def run(cmd, **kw):
        return SimpleNamespace(stdout=payload, stderr="", returncode=0)
    return run


def test_a_stale_backend_beside_a_fresh_process_is_reported_STALE(monkeypatch):
    """End to end through check_backend, which is where the reduction is used."""
    newest_t = 1_700_000_600.0
    monkeypatch.setattr(cell_health, "_newest_source",
                        lambda: (ROOT / "tasni" / "core" / "config.py", newest_t))
    monkeypatch.setattr(cell_health.subprocess, "run", _fake_ps(
        '[{"ProcessId": 1, "CreationDate": "/Date(1700000000000)/"},'
        ' {"ProcessId": 2, "CreationDate": "/Date(1700000900000)/"}]'))
    cell_health.check_backend()
    fresh = _by_title()["backend freshness"]
    assert fresh[0] == cell_health.FAIL, f"stale backend reported {fresh[0]}: {fresh[2]}"
    assert "STALE" in fresh[3]


def test_the_config_file_counts_as_backend_source():
    """create_app() calls load_config() ONCE and hands the AppConfig to the
    services, so an edited tasni.config.json is as stale as an edited module."""
    assert (ROOT / "tasni.config.json") in set(cell_health._source_files())


def test_jetson_and_tooling_sources_are_deliberately_excluded():
    """server/ runs on the JETSON (the 'jetson code' commit check covers it) and
    tools/ is not imported by the backend -- and this script lives there, so
    counting it would make the freshness check fire on its own edits."""
    rel = {f.relative_to(ROOT).parts[0] for f in cell_health._source_files()}
    assert "server" not in rel and "tools" not in rel, sorted(rel)


# ------------------------------------------------ defect 4: an honest summary

def test_a_backend_that_is_not_running_is_never_summarised_as_no_blockers(monkeypatch):
    """The summary printed "READY WITH NOTES ... nothing blocks a capture run" for
    a cell with no backend at all. You cannot capture without one."""
    monkeypatch.setattr(cell_health.subprocess, "run", _fake_ps(""))
    cell_health.check_backend()
    assert _by_title()["backend process"][4] is True, "not-running must block capture"
    code, lines = cell_health.summarise(cell_health.RESULTS)
    text = "\n".join(lines)
    assert "blocks a capture run" not in text, text
    assert "before you can capture" in text, text
    assert cell_health.START_CMD in text, "the summary must name what to do"
    assert code == 0, "a backend that has not been started yet is not a broken cell"


def test_an_occupied_camera_also_blocks_a_capture_run():
    """The server is unicast; while another client holds it a run cannot start."""
    cell_health.record(cell_health.WARN, "camera clients", "1 connection(s) already open",
                       "close it", blocks_capture=True)
    _, lines = cell_health.summarise(cell_health.RESULTS)
    assert "camera clients" in "\n".join(lines)


def test_ordinary_warnings_still_read_as_ready_with_notes():
    """The counter-test: the blocker bucket must not swallow every warning, or
    the operator learns to ignore the headline."""
    cell_health.record(cell_health.WARN, "working tree", "3 uncommitted change(s)", "commit")
    code, lines = cell_health.summarise(cell_health.RESULTS)
    text = "\n".join(lines)
    assert "READY WITH NOTES" in text and "none of them blocks a capture run" in text
    assert code == 0


def test_failures_still_exit_nonzero_and_list_their_remedies():
    cell_health.record(cell_health.FAIL, "depth scale", "10x short", "deploy the server")
    cell_health.record(cell_health.WARN, "backend process", "not running", "start it",
                       blocks_capture=True)
    code, lines = cell_health.summarise(cell_health.RESULTS)
    text = "\n".join(lines)
    assert code == 1 and "NOT READY" in text
    assert "deploy the server" in text
    assert "start it" in text, "blockers must still be listed alongside failures"
