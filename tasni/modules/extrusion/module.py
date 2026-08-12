"""FastAPI integration for the Post-Extrusion Cylinder Test module."""
from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

from pydantic import BaseModel

from ...core.jobrunner import JobBusy
from ...core.logging import REPO_ROOT
from ..base import ServiceContainer, WorkflowModule
from .models import CylinderPlan, CylinderRecipe, CylinderSetup
from .service import (CylinderDryRunJob, CylinderPrintJob, geometry_preflight,
                      reprocess_saved_layer, station_requirements)
from .toolpath import corrected_cylinder_plan, generate_cylinder_plan

if TYPE_CHECKING:
    from fastapi import APIRouter


class GenerateBody(BaseModel):
    recipe: CylinderRecipe
    setup: CylinderSetup


class FingerprintBody(BaseModel):
    fingerprint: str


class PrintBody(FingerprintBody):
    confirm_live: bool = False


class ExtrusionModule(WorkflowModule):
    id = "extrusion"
    title = "Cylinder Test"
    description = "Generate, print, inspect, compare, and correct circular extrusion layers."
    icon = "◎"
    order = 30

    def __init__(self, services: ServiceContainer):
        super().__init__(services)
        self._plan: CylinderPlan | None = None
        self._geometry_preflight_fingerprint: str | None = None
        self._dry_run_fingerprint: str | None = None
        self._active_dry_job: CylinderDryRunJob | None = None
        self._active_print_job: CylinderPrintJob | None = None

    def _default_recipe(self) -> CylinderRecipe:
        c = self.services.config.extrusion
        return CylinderRecipe(
            radius_mm=c.radius_mm, layer_count=c.layer_count,
            layer_height_mm=c.layer_height_mm, bead_diameter_mm=c.bead_diameter_mm,
            robot_speed_mm_s=c.robot_speed_mm_s,
            travel_speed_mm_s=c.travel_speed_mm_s,
            path_rounding_mm=c.path_rounding_mm,
            extrusion_rate_pct=c.extrusion_rate_pct,
            points_per_circle=c.points_per_circle,
            correction_enabled=c.correction_enabled,
        )

    def _default_setup(self) -> dict:
        c = self.services.config.extrusion
        return {
            "print_tool": c.default_print_tool,
            "work_frame": c.default_work_frame,
            "inspection_tool": c.default_inspection_tool,
            "inspection_target": c.default_inspection_target,
            "center_x_mm": c.center_x_mm, "center_y_mm": c.center_y_mm,
            "orientation_rpy_deg": list(c.orientation_rpy_deg),
            "approach_clearance_mm": c.approach_clearance_mm,
            "retreat_clearance_mm": c.retreat_clearance_mm,
        }

    def _accept_dry_run(self, fingerprint: str) -> None:
        # A generation request is blocked while the job runs, but retain this
        # comparison so a stale completion can never approve a newer plan.
        if self._plan is not None and self._plan.fingerprint == fingerprint:
            self._dry_run_fingerprint = fingerprint

    def router(self) -> "APIRouter":
        from fastapi import APIRouter, HTTPException

        router = APIRouter()
        services = self.services

        @router.get("/config")
        def config() -> dict:
            c = services.config.extrusion
            return {
                "defaults": self._default_recipe().model_dump(mode="json"),
                "setup_defaults": self._default_setup(),
                "integration": {
                    "air_on_program": c.air_on_program,
                    "air_off_program": c.air_off_program,
                    "valve_outputs": c.valve_outputs,
                    "mapping_source": c.valve_mapping_source,
                    "mapping_verified": c.valve_mapping_verified,
                    "hardware_io_test_approved": c.hardware_io_test_approved,
                    "extrusion_rate_control": "record_only",
                },
                "live_print_enabled": bool(c.valve_mapping_verified and
                                           c.hardware_io_test_approved),
            }

        @router.post("/connect")
        def connect() -> dict:
            """Load/attach the station without moving or linking the real robot."""
            deadline = time.monotonic() + float(services.config.robodk.connect_timeout_s)
            last_error = None
            while time.monotonic() < deadline:
                try:
                    if services.rdk.robot().Valid():
                        return {"connected": True, "ready": True,
                                "robot": services.config.robodk.robot_name}
                except Exception as exc:
                    last_error = exc
                    services.session.reset()
                time.sleep(0.5)
            raise HTTPException(503, f"RoboDK station did not become ready: {last_error}")

        @router.get("/station-options")
        def station_options() -> dict:
            try:
                return {"tools": services.rdk.list_tools(),
                        "frames": services.rdk.list_frames(),
                        "targets": services.rdk.list_targets(prefix=""),
                        "programs": services.rdk.list_programs()}
            except Exception as exc:
                raise HTTPException(503, f"RoboDK unavailable: {exc}")

        @router.post("/generate")
        def generate(body: GenerateBody) -> dict:
            if services.jobs.running:
                raise HTTPException(409, "cannot regenerate while a robot job is running")
            self._plan = generate_cylinder_plan(body.recipe, body.setup)
            self._geometry_preflight_fingerprint = None
            self._dry_run_fingerprint = None
            self._active_dry_job = None
            self._active_print_job = None
            return self._plan.model_dump(mode="json")

        @router.post("/preflight")
        def preflight(body: FingerprintBody) -> dict:
            if self._plan is None or body.fingerprint != self._plan.fingerprint:
                raise HTTPException(409, "toolpath changed; generate the current recipe again")
            result = geometry_preflight(self._plan)
            if not services.session.is_open:
                result["station"] = {"ready": False,
                                     "error": "connect RoboDK to validate selected station items"}
            else:
                try:
                    result["station"] = station_requirements(
                        services.rdk, self._plan, services.config.extrusion)
                except Exception as exc:
                    result["station"] = {"ready": False, "error": str(exc)}
            if result["all_ok"]:
                self._geometry_preflight_fingerprint = self._plan.fingerprint
            return result

        @router.post("/dry-run")
        def dry_run(body: FingerprintBody) -> dict:
            if self._plan is None or body.fingerprint != self._plan.fingerprint:
                raise HTTPException(409, "toolpath changed; generate the current recipe again")
            if self._geometry_preflight_fingerprint != body.fingerprint:
                raise HTTPException(409, "run geometry preflight for this toolpath first")
            if services.jobs.running:
                raise HTTPException(409, "a job is already running")
            self._dry_run_fingerprint = None
            self._active_dry_job = CylinderDryRunJob(
                services, self._plan, on_pass=self._accept_dry_run)
            try:
                services.jobs.start(self._active_dry_job, name="extrusion-dry-run")
            except JobBusy as exc:
                raise HTTPException(409, str(exc))
            return {"status": "started", "mode": "DRY_RUN",
                    "fingerprint": self._plan.fingerprint}

        @router.post("/print")
        def live_print(body: PrintBody) -> dict:
            c = services.config.extrusion
            if self._plan is None or body.fingerprint != self._plan.fingerprint:
                raise HTTPException(409, "toolpath changed; generate and dry-run it again")
            if self._dry_run_fingerprint != body.fingerprint:
                raise HTTPException(409, "current toolpath has not passed a RoboDK dry run")
            if not c.hardware_io_test_approved:
                raise HTTPException(423, "live extrusion locked until the hardware I/O test is approved")
            if not body.confirm_live:
                raise HTTPException(400, "explicit live-run confirmation is required")
            if services.jobs.running:
                raise HTTPException(409, "a job is already running")
            self._active_print_job = CylinderPrintJob(services, self._plan)
            try:
                services.jobs.start(self._active_print_job, name="extrusion-print")
            except JobBusy as exc:
                raise HTTPException(409, str(exc))
            return {"status": "started", "mode": "LIVE_PRINT",
                    "fingerprint": self._plan.fingerprint}

        @router.post("/correction/apply")
        def apply_correction(body: FingerprintBody) -> dict:
            job = self._active_print_job
            if job is None or job.plan.fingerprint != body.fingerprint:
                raise HTTPException(409, "no matching measured print is available")
            if job.corrected_reference_xyz is None:
                raise HTTPException(409, "the matching print has no valid calculated correction")
            if services.jobs.running:
                raise HTTPException(409, "wait for the current job to finish")
            self._plan = corrected_cylinder_plan(job.plan, job.corrected_reference_xyz)
            self._geometry_preflight_fingerprint = None
            self._dry_run_fingerprint = None
            return self._plan.model_dump(mode="json")

        @router.post("/cancel")
        def cancel() -> dict:
            services.jobs.cancel()
            return {"status": "cancelling"}

        @router.post("/reset")
        def reset() -> dict:
            """Invalidate the generated plan and remove only Tasni-owned artifacts."""
            if services.jobs.running:
                raise HTTPException(409, "cannot reset while a robot job is running")
            removed: list[str] = []
            if services.session.is_open:
                try:
                    removed = services.rdk.cleanup_extrusion_artifacts()
                except Exception as exc:
                    raise HTTPException(
                        503, f"could not clean RoboDK extrusion artifacts: {exc}") from exc
            self._plan = None
            self._geometry_preflight_fingerprint = None
            self._dry_run_fingerprint = None
            self._active_dry_job = None
            self._active_print_job = None
            return {"status": "reset", "removed": removed}

        @router.get("/status")
        def status() -> dict:
            fingerprint = self._plan.fingerprint if self._plan else None
            return {
                "status": services.jobs.status, "running": services.jobs.running,
                "result": services.jobs.result, "error": services.jobs.error,
                "fingerprint": fingerprint,
                "geometry_preflight_passed": bool(
                    fingerprint and fingerprint == self._geometry_preflight_fingerprint),
                "dry_run_passed": bool(
                    fingerprint and fingerprint == self._dry_run_fingerprint),
                "hardware_io_test_approved": bool(
                    services.config.extrusion.hardware_io_test_approved),
                "live_print_enabled": bool(
                    fingerprint and fingerprint == self._dry_run_fingerprint and
                    services.config.extrusion.hardware_io_test_approved),
            }

        @router.get("/trials")
        def trials() -> dict:
            root = REPO_ROOT / "runs" / "extrusion"
            items = []
            recipe_keys: set[str] = set()
            total_layers = 0
            if root.is_dir():
                for path in sorted(root.iterdir(), reverse=True):
                    trial_file = path / "trial.json"
                    if path.is_dir() and trial_file.is_file():
                        data = json.loads(trial_file.read_text(encoding="utf-8"))
                        recipe_keys.add(json.dumps(data.get("recipe", {}), sort_keys=True))
                        layers = []
                        for manifest_path in sorted(path.glob("layer-*/manifest.json")):
                            try:
                                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                                layers.append({
                                    "layer_index": manifest.get("layer_index"),
                                    "metrics": manifest.get("metrics"),
                                    "valid": bool((manifest.get("metrics") or {}).get("valid")),
                                    "has_comparison": (manifest_path.parent / "comparison.png").is_file(),
                                })
                            except (OSError, ValueError, json.JSONDecodeError):
                                continue
                        data["layers_archived"] = len(layers)
                        data["layers"] = layers
                        total_layers += len(layers)
                        items.append(data)
            return {"summary": {"total_trials": len(items),
                                "total_layers": total_layers,
                                "total_recipes": len(recipe_keys)},
                    "trials": items}

        @router.post("/trials/{trial_id}/layers/{layer_index}/reprocess")
        def reprocess(trial_id: str, layer_index: int) -> dict:
            if services.jobs.running:
                raise HTTPException(409, "cannot reprocess while a robot job is running")
            try:
                return reprocess_saved_layer(
                    REPO_ROOT / "runs" / "extrusion", trial_id, layer_index)
            except (ValueError, FileNotFoundError) as exc:
                raise HTTPException(404, str(exc)) from exc
            except RuntimeError as exc:
                raise HTTPException(422, str(exc)) from exc

        return router
