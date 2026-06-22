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
            rdk = self._connect()              # connect + widen timeout + quiet collisions
            try:
                self._ensure_station(rdk)      # load the station if RoboDK came up empty
            except Exception:
                # The connected handle would otherwise leak (and in isolated mode
                # strand a headless RoboDK) when the load/first-query fails. Tear it
                # down before re-raising so a retry re-attaches cleanly.
                try:
                    rdk.Finish()
                except Exception:
                    pass
                raise
            self._post_connect(rdk)            # quiet collisions again (AddFile may re-arm them)
            self._rdk = rdk
        return self._rdk

    def _connect(self) -> "Robolink":
        from robolink import Robolink

        if self.config.connection == "isolated":
            rdk = Robolink(args=ISOLATED_ARGS, quit_on_close=True)
            self._prime(rdk)
            station = resolve_station_path(self.config.station_path)
            if station:
                rdk.AddFile(str(station))
            return rdk
        # "attach": default constructor binds to a running instance (starts one
        # with a window if none is running).
        rdk = Robolink()
        self._prime(rdk)
        return rdk

    def _prime(self, rdk: "Robolink") -> None:
        """Make the freshly-opened connection ready for the heavy cell, BEFORE any
        existence query runs.

        Two things, in order:

        * Widen the socket timeout from robolink's 10 s default. Setting the
          ``TIMEOUT`` *attribute* alone is not enough — ordinary queries like
          ``Item()`` never re-apply it to the live socket, so the first heavy
          existence query would still run at the 10 s the constructor left on the
          socket and time out (the "first Connect fails, second works" bug).
          ``_setTimeout`` pushes the value onto the live socket immediately.
        * Quiet global collision checking. A station saved with checking ON makes
          RoboDK recompute the whole map on the very first query against the 117 MB
          cell — the slowness that tripped the old timeout. Doing it here (before
          ``_ensure_station``'s ``Item().Valid()``) keeps that first query fast.
          The collision MAP / pair config is preserved; the app re-enables checking
          only transiently for the pose filter + dry tour. Best-effort."""
        val = float(self.config.connect_timeout_s)
        try:
            rdk._setTimeout(val)               # attribute + live socket, under the lock
        except Exception:
            try:
                rdk.TIMEOUT = val
            except Exception:
                pass
        self._disable_collisions(rdk)

    def _post_connect(self, rdk: "Robolink") -> None:
        """Quiet collisions again once the station is loaded — loading a station
        saved with checking ON can re-arm it. Idempotent with :meth:`_prime`."""
        self._disable_collisions(rdk)

    def _disable_collisions(self, rdk: "Robolink") -> None:
        if not self.config.disable_collisions_on_connect:
            return
        try:
            from robolink import COLLISION_OFF

            rdk.setCollisionActive(COLLISION_OFF)
        except Exception:
            pass

    def reset(self) -> None:
        """Drop the cached handle so the next access re-attaches. Used by the
        connect poll to recover from a half-open handle while the station is still
        materialising (vs :meth:`close`, this is just a synonym for clarity)."""
        self.close()

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
            except Exception:
                # A headless instance may already have torn the socket down
                # (e.g. a Free-license popup closing it); disconnect is best-effort.
                pass
            finally:
                self._rdk = None

    def __enter__(self) -> "RdkSession":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
