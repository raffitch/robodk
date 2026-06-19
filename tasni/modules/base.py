"""The module contract: ServiceContainer (the shared core, injected) + the
WorkflowModule ABC every workflow implements.

A module is a *leaf*: it gets the core services through a
:class:`ServiceContainer` and contributes (a) REST endpoints and (b) a UI panel.
It must not open sockets, import ``robolink``, or spawn threads itself — those
are the core's job. Keeping that boundary is what lets scan / aruco / printing
modules drop in later exactly the way calibration does.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..core.camera import CameraClient
from ..core.config import AppConfig
from ..core.events import EventBus
from ..core.jobrunner import JobRunner
from ..core.rdk_io import RdkIO
from ..core.session import RdkSession

if TYPE_CHECKING:  # pragma: no cover - only for type hints
    from fastapi import APIRouter


@dataclass
class ServiceContainer:
    """Everything a module is allowed to touch — the shared core, bundled."""

    config: AppConfig
    session: RdkSession
    rdk: RdkIO
    camera: CameraClient
    bus: EventBus
    jobs: JobRunner

    @classmethod
    def build(cls, config: AppConfig | None = None) -> "ServiceContainer":
        from ..core.config import load_config

        config = config or load_config()
        session = RdkSession(config.robodk)
        bus = EventBus()
        return cls(
            config=config,
            session=session,
            rdk=RdkIO(session),
            camera=CameraClient(config.camera),
            bus=bus,
            jobs=JobRunner(bus),
        )


class WorkflowModule(ABC):
    """Base class for a pluggable workflow (calibration, scan, print, ...)."""

    #: stable url-safe id, e.g. "calibration"
    id: str
    #: human title shown in the sidebar
    title: str
    #: one-line description
    description: str = ""

    def __init__(self, services: ServiceContainer):
        self.services = services

    @abstractmethod
    def router(self) -> "APIRouter":
        """Return the module's FastAPI router (mounted at /api/modules/<id>)."""

    @abstractmethod
    def panel_html(self) -> str:
        """Return the HTML fragment rendered when this module's tab is active."""

    def meta(self) -> dict:
        return {"id": self.id, "title": self.title, "description": self.description}
