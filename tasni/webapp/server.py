"""FastAPI shell: builds the shared services + module registry, mounts each
module's router/panel, and bridges the job event bus to the browser over a
WebSocket. The shell knows nothing calibration-specific — it just renders the
registered modules, which is the whole point of the platform.
"""
from __future__ import annotations

import asyncio

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from ..core.config import AppConfig, load_config
from ..modules.base import ServiceContainer
from ..modules.registry import build_registry
from .static import STATIC_DIR


def create_app(config: AppConfig | None = None) -> FastAPI:
    config = config or load_config()
    services = ServiceContainer.build(config)
    registry = build_registry(services)

    app = FastAPI(title="tasni", version="0.1.0")
    app.state.services = services
    app.state.registry = registry

    @app.on_event("startup")
    async def _bind_loop() -> None:
        # Worker-thread job events hop onto this loop to reach the WebSocket.
        services.bus.bind_loop(asyncio.get_running_loop())

    @app.get("/api/modules")
    def list_modules() -> dict:
        return {"modules": [m.meta() for m in registry.all()]}

    def _module(module_id: str):
        try:
            return registry.get(module_id)
        except KeyError:
            raise HTTPException(404, f"no such module: {module_id}")

    @app.get("/api/modules/{module_id}/panel.html", response_class=HTMLResponse)
    def panel_html(module_id: str) -> str:
        return _module(module_id).panel_html()

    @app.get("/api/modules/{module_id}/panel.js")
    def panel_js(module_id: str) -> PlainTextResponse:
        return PlainTextResponse(_module(module_id).panel_js(),
                                 media_type="application/javascript")

    for module in registry.all():
        app.include_router(module.router(), prefix=f"/api/modules/{module.id}")

    @app.websocket("/ws")
    async def ws(websocket: WebSocket) -> None:
        await websocket.accept()
        queue = services.bus.subscribe()
        try:
            while True:
                event = await queue.get()
                await websocket.send_json(event.to_dict())
        except WebSocketDisconnect:
            pass
        finally:
            services.bus.unsubscribe(queue)

    # The SPA shell + assets at /. (mounted last so /api/* and /ws win.)
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
    return app
