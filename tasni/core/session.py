"""RoboDK connection service.

Two modes (see :class:`~tasni.core.config.RoboDKConfig`):

* ``attach``  — bind to the RoboDK GUI instance the user already has open. This
  is the default for the control panel: that instance has the station, the
  ``Target*`` items and the tool to calibrate already loaded.
* ``isolated`` — spawn a private, headless ``-NEWINSTANCE -NOUI`` RoboDK, like
  ``rdk_session.py`` does for the extract/sync scripts. Used by tests and for
  bench runs where no GUI is open; optionally loads a ``.rdk`` station file.

``robolink`` is imported lazily so the rest of the platform (and the pure-math
unit tests) import without RoboDK installed.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .config import RoboDKConfig

if TYPE_CHECKING:  # pragma: no cover
    from robolink import Robolink

ISOLATED_ARGS = ["-NEWINSTANCE", "-NOUI", "-SKIPINI", "-EXIT_LAST_COM"]
_REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_station_path(station_path: str | None) -> Path | None:
    """Resolve a configured station path (relative -> repo root). None if unset
    or missing on disk."""
    if not station_path:
        return None
    p = Path(station_path)
    if not p.is_absolute():
        p = _REPO_ROOT / p
    return p if p.exists() else None


class RdkSession:
    """Lazily-opened RoboDK connection described by a :class:`RoboDKConfig`."""

    def __init__(self, config: RoboDKConfig):
        self.config = config
        self._rdk: "Robolink | None" = None

    @property
    def rdk(self) -> "Robolink":
        """The live ``Robolink`` handle, connecting (and loading the station on
        first access)."""
        if self._rdk is None:
            self._rdk = self._connect()
            self._ensure_station(self._rdk)
        return self._rdk

    def _connect(self) -> "Robolink":
        from robolink import Robolink

        if self.config.connection == "isolated":
            rdk = Robolink(args=ISOLATED_ARGS, quit_on_close=True)
            station = resolve_station_path(self.config.station_path)
            if station:
                rdk.AddFile(str(station))
            return rdk
        # "attach": default constructor binds to a running instance (starts one
        # with a window if none is running).
        return Robolink()

    def _ensure_station(self, rdk: "Robolink") -> None:
        """If the configured robot isn't present (e.g. RoboDK opened empty),
        open the cell's station file into this instance so the app drives the
        real cell instead of a blank station. No-op if it's already loaded."""
        station = resolve_station_path(self.config.station_path)
        if station is None:
            return
        if rdk.Item(self.config.robot_name).Valid():
            return  # station with our robot already loaded — don't reload
        rdk.AddFile(str(station))
        active = rdk.ActiveStation()
        if active.Valid():
            active.setName(self.config.station_name)

    def close(self) -> None:
        if self._rdk is not None:
            try:
                self._rdk.Finish()
            finally:
                self._rdk = None

    def __enter__(self) -> "RdkSession":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
