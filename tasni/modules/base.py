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
from ..core.camera_lease import CameraLease
from ..core.config import AppConfig
from ..core.events import EventBus
from ..core.jobrunner import JobRunner
from ..core.livepreview import LivePreview
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
    camera_lease: CameraLease
    bus: EventBus
    jobs: JobRunner
    live: LivePreview

    @classmethod
    def build(cls, config: AppConfig | None = None) -> "ServiceContainer":
        from ..core.config import load_config

        config = config or load_config()
        session = RdkSession(config.robodk)
        bus = EventBus()
        camera = CameraClient(config.camera)
        # One lease guards the unicast camera; both the live preview and the
        # capture grabs take it, so they can never race the single-client socket.
        lease = CameraLease()
        return cls(
            config=config,
            session=session,
            rdk=RdkIO(session),
            camera=camera,
            camera_lease=lease,
            bus=bus,
            jobs=JobRunner(bus),
            live=LivePreview(camera, bus, lease=lease),
        )


class WorkflowModule(ABC):
    """Base class for a pluggable workflow (calibration, scan, print, ...).

    A module is the *backend* half of a workflow: stable metadata + a REST
    router. Its UI is a React page in ``tasni/webui`` keyed off ``id`` (the
    frontend module registry). Adding a workflow = this class + a component +
    one registry line.
    """

    #: stable url-safe id, e.g. "calibration" — also the frontend component key
    id: str
    #: human title shown in the sidebar / dashboard
    title: str
    #: one-line description
    description: str = ""
    #: emoji/icon shown on the dashboard card and sidebar (optional)
    icon: str = "•"
    #: sidebar ordering hint (lower = higher up)
    order: int = 100

    def __init__(self, services: ServiceContainer):
        self.services = services

    @abstractmethod
    def router(self) -> "APIRouter":
        """Return the module's FastAPI router (mounted at /api/modules/<id>)."""

    def meta(self) -> dict:
        return {"id": self.id, "title": self.title,
                "description": self.description, "icon": self.icon,
                "order": self.order}
