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

from typing import TYPE_CHECKING

from .config import RoboDKConfig

if TYPE_CHECKING:  # pragma: no cover
    from robolink import Robolink

ISOLATED_ARGS = ["-NEWINSTANCE", "-NOUI", "-SKIPINI", "-EXIT_LAST_COM"]


class RdkSession:
    """Lazily-opened RoboDK connection described by a :class:`RoboDKConfig`."""

    def __init__(self, config: RoboDKConfig):
        self.config = config
        self._rdk: "Robolink | None" = None

    @property
    def rdk(self) -> "Robolink":
        """The live ``Robolink`` handle, connecting on first access."""
        if self._rdk is None:
            self._rdk = self._connect()
        return self._rdk

    def _connect(self) -> "Robolink":
        from robolink import Robolink

        if self.config.connection == "isolated":
            rdk = Robolink(args=ISOLATED_ARGS, quit_on_close=True)
            if self.config.station_path:
                rdk.AddFile(self.config.station_path)
            return rdk
        # "attach": default constructor binds to a running instance (starts one
        # with a window if none is running).
        return Robolink()

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
