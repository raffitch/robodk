"""FastAPI shell: builds the shared services + module registry, exposes the
platform API (modules, health, runs, job events), and — in prod — serves the
built React app from ``tasni/webui/dist``. The shell knows nothing
calibration-specific; it just lists the registered modules.

Dev: run Vite (``tasni/webui``) on :5173 proxying /api + /ws here. Prod:
``npm run build`` then this serves dist/ as a single origin. See start.sh.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..core.config import AppConfig, load_config
from ..core.health import ROBODK_API_PORT, tcp_probe
from ..core.logging import REPO_ROOT
from ..modules.base import ServiceContainer
from ..modules.registry import build_registry

DIST_DIR = Path(__file__).resolve().parents[1] / "webui" / "dist"


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

    # -- platform API -------------------------------------------------------
    @app.get("/api/modules")
    def list_modules() -> dict:
        mods = sorted(registry.all(), key=lambda m: (m.order, m.title))
        return {"modules": [m.meta() for m in mods]}

    @app.get("/api/health")
    def health() -> dict:
        cam = services.config.camera
        robodk_ok = tcp_probe("127.0.0.1", ROBODK_API_PORT)
        # Don't probe the camera mid-capture — the unicast server serves one
        # client and a probe would steal the frame the job expects.
        if services.jobs.running:
            camera = {"ok": None, "detail": "in use by running job"}
        else:
            camera = {"ok": tcp_probe(cam.ip, cam.port),
                      "detail": f"{cam.ip}:{cam.port}"}
        return {
            "robodk": {"ok": robodk_ok, "detail": f"API :{ROBODK_API_PORT}"},
            "camera": camera,
            "job": {"status": services.jobs.status, "running": services.jobs.running},
        }

    @app.get("/api/runs")
    def runs(limit: int = 20) -> dict:
        """Recent run-artifact folders across all modules, newest first."""
        root = REPO_ROOT / "runs"
        items = []
        if root.exists():
            for module_dir in root.iterdir():
                if not module_dir.is_dir():
                    continue
                for run in module_dir.iterdir():
                    if run.is_dir():
                        items.append({"module": module_dir.name, "stamp": run.name,
                                      "path": str(run)})
        items.sort(key=lambda r: r["stamp"], reverse=True)
        return {"runs": items[:limit]}

    for module in sorted(registry.all(), key=lambda m: m.order):
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

    # -- serve the built SPA (prod). In dev, Vite serves the UI itself. -----
    if DIST_DIR.exists():
        app.mount("/assets", StaticFiles(directory=str(DIST_DIR / "assets")),
                  name="assets")

        @app.get("/{full_path:path}")
        def spa(full_path: str):
            # Anything not matched above falls back to index.html (client routing).
            candidate = DIST_DIR / full_path
            if full_path and candidate.is_file():
                return FileResponse(candidate)
            return FileResponse(DIST_DIR / "index.html")
    else:
        @app.get("/")
        def no_build():
            return JSONResponse(
                {"detail": "UI not built. Run `start.sh` (dev) or "
                           "`cd tasni/webui && npm run build` (prod)."},
                status_code=200)

    return app
