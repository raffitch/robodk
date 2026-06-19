"""Module registry — discovers the workflow modules and hands them their services.

For now modules are registered explicitly in :func:`build_registry`. The same
shape (id -> module) generalizes to ``importlib.metadata`` entry-points later,
so third-party / future modules can register without editing this file.
"""
from __future__ import annotations

from .base import ServiceContainer, WorkflowModule


class ModuleRegistry:
    def __init__(self) -> None:
        self._modules: dict[str, WorkflowModule] = {}

    def register(self, module: WorkflowModule) -> None:
        if module.id in self._modules:
            raise ValueError(f"duplicate module id: {module.id!r}")
        self._modules[module.id] = module

    def get(self, module_id: str) -> WorkflowModule:
        return self._modules[module_id]

    def all(self) -> list[WorkflowModule]:
        return list(self._modules.values())


def build_registry(services: ServiceContainer) -> ModuleRegistry:
    """Instantiate and register every built-in module against ``services``."""
    from .calibration.module import CalibrationModule

    registry = ModuleRegistry()
    registry.register(CalibrationModule(services))
    return registry
