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

from ..core import runs as runs_registry
from ..core.build_info import build_info
from ..core.config import AppConfig, load_config
from ..core.health import ROBODK_API_PORT, connection_route, tcp_probe
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
        # client and a probe would steal the frame the lease holder expects. The
        # lease's owner label gives the precise holder ("live-preview",
        # "calibration-run", ...); fall back to the coarse job/live flags.
        if services.camera_lease.held:
            busy = f"in use by {services.camera_lease.owner}"
        elif services.jobs.running:
            busy = "in use by running job"
        elif services.live.running:
            busy = "in use by live preview"
        else:
            busy = ""
        if busy:
            # Mid-capture the running client has already resolved the host, so
            # report it rather than opening a competing probe.
            cam_host, camera_ok, state, lead = (
                services.camera.active_host, None, "in_use", busy)
        else:
            # Idle: walk the ladder here. The client only resolves the host as a
            # side effect of a capture, so without this the dashboard would echo
            # the configured fallback ("Tailscale") until the operator happened
            # to start a preview — reporting a route we are not actually on.
            cam_host, camera_ok = services.camera.resolve_via(tcp_probe)
            state = "connected" if camera_ok else "offline"
            lead = "connected" if camera_ok else "offline/unreachable"
        camera_route = connection_route(cam_host)
        camera_endpoint = f"{cam_host}:{cam.port}"
        camera = {
            "route": camera_route, "endpoint": camera_endpoint,
            "ok": camera_ok, "state": state,
            "detail": f"{lead} via {camera_route} · {camera_endpoint}",
        }
        return {
            "robodk": {"ok": robodk_ok, "detail": f"API :{ROBODK_API_PORT}"},
            "camera": camera,
            "job": {"status": services.jobs.status, "running": services.jobs.running},
            # Whether this process is still running the code that is on disk.
            # Editing tasni/**.py does nothing until the app restarts, and a cell
            # test against stale code looks exactly like a failed fix.
            "build": build_info(),
        }

    @app.get("/api/rdk/status")
    def rdk_status() -> dict:
        """Current shared RoboDK session state without opening the station.

        Module pages use this on mount so switching Calibration -> Scan keeps the
        UI connected when the backend already has a live RoboDK handle.
        """
        c = services.config.robodk
        if not services.session.is_open:
            return {"connected": False, "ready": False, "robot": c.robot_name,
                    "tool": c.camera_tool, "missing": [c.robot_name]}
        try:
            robot_ok = services.rdk.robot().Valid()
            tool_ok = services.rdk.item_exists(c.camera_tool) if robot_ok else False
            missing = [n for n, ok in (
                (c.robot_name, robot_ok),
                (f"tool {c.camera_tool!r}", tool_ok)) if not ok]
            try:
                link_ok, link_msg = services.rdk.robot_connected()
            except Exception:
                link_ok, link_msg = False, ""
            params = services.rdk.robot_connection_params()
            return {"connected": True, "ready": robot_ok and tool_ok,
                    "robot": c.robot_name, "robot_valid": robot_ok,
                    "tool": c.camera_tool, "tool_present": tool_ok,
                    "missing": missing,
                    "robot_link": {"connected": link_ok, "message": link_msg,
                                   "ip": params.get("ip", ""),
                                   "configured": bool(params.get("ip"))}}
        except Exception as e:
            return {"connected": False, "ready": False, "robot": c.robot_name,
                    "tool": c.camera_tool, "missing": [c.robot_name],
                    "error": str(e)}

    # The only directories under runs/ this API may enumerate or delete: the
    # registered module ids (plus their own ``<id>-<kind>`` buckets — see
    # runs.is_run_bucket). Anything else parked in runs/ — a figures backup, a
    # measurement archive — is invisible here and therefore undeletable. The
    # registry is fixed once built, so resolve the set once.
    run_module_ids = frozenset(m.id for m in registry.all())

    @app.get("/api/runs")
    def runs(limit: int = 20) -> dict:
        """Recent run-artifact folders across the registered modules, newest first."""
        return {"runs": runs_registry.list_runs(limit, modules=run_module_ids)}

    @app.get("/api/runs/active")
    def active_run(module: str) -> dict:
        """The currently-applied run for a module (its ``active.json`` pointer), so
        the Dashboard can show e.g. "cell calibrated: <date> · <quality>". ``None``
        until something has been applied."""
        try:
            return {"active": runs_registry.read_active(module)}
        except ValueError as e:                 # rejected module segment
            raise HTTPException(400, str(e))

    @app.delete("/api/runs/{module}/{stamp}")
    def delete_run(module: str, stamp: str, force: bool = False) -> dict:
        """Delete one run folder and every file in it (Dashboard housekeeping).

        Three guards, because this is the one endpoint that destroys data:
        * never while a job runs — the folder being deleted may be the one the
          robot is writing takes into right now;
        * never the run currently applied to the cell without ``force=true``, so
          clearing old runs cannot silently take out the live calibration/scan. The
          409 is structured so the UI can offer "delete anyway" instead of guessing;
        * never outside a registered module's run bucket, and never a directory
          that does not look like a run — ``runs/`` also holds things nobody
          registered (a figures backup, an archive) and this endpoint must not be
          able to reach them, however the URL is hand-crafted. Both refusals are
          400s: the request names something that is not a run, so retrying it
          unchanged (or with ``force``) can never be right.
        """
        if services.jobs.running:
            raise HTTPException(409, "a job is running — stop it before deleting runs")
        # Forgiving by design: an unreadable/absent pointer means "nothing applied".
        # A bad or unknown module segment is rejected below, by delete_run's guards.
        was_active = runs_registry.active_run_id(module) == stamp
        if was_active and not force:
            raise HTTPException(409, {
                "code": "run_is_active",
                "message": f"{module}/{stamp} is the run currently applied to the cell."})
        try:
            out = runs_registry.delete_run(module, stamp, modules=run_module_ids)
        except runs_registry.RunNotFound as e:
            raise HTTPException(404, str(e))
        except ValueError as e:
            raise HTTPException(400, str(e))
        except OSError as e:                    # in use / permission (common on Windows)
            raise HTTPException(409, f"could not delete {module}/{stamp}: {e}")
        try:
            registry.get(module).on_runs_deleted({stamp})
        except KeyError:                        # a run folder with no live module
            pass
        # The pointer keeps its own copy of the payload, so it stays valid as
        # provenance — but re-applying *by run id* can no longer find the files.
        out["active_pointer_dangling"] = was_active
        return out

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
            # Anything not matched above falls back to index.html (client routing)
            # -- except an API path. Serving the app for an unmatched /api route
            # turns "this endpoint does not exist" into a 200 with an HTML body,
            # which a download link happily saves under whatever name it asked
            # for: a web page named paper-draft.docx, and no error anywhere.
            if full_path == "api" or full_path.startswith("api/"):
                raise HTTPException(404, f"no such API endpoint: /{full_path}")
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
