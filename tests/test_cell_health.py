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
    m = re.search(r'print\(f"(Initiating[^"]*)"\)', src)
    assert m, "the startup banner is no longer a single-line print(f\"Initiating...\")"
    line = (m.group(1)
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
