"""FastAPI integration for the Post-Extrusion Cylinder Test module."""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

from pydantic import BaseModel

from ...core.logging import REPO_ROOT
from ..base import ServiceContainer, WorkflowModule
from .models import CylinderPlan, CylinderRecipe
from .service import geometry_preflight
from .toolpath import generate_cylinder_plan

if TYPE_CHECKING:
    from fastapi import APIRouter


class FingerprintBody(BaseModel):
    fingerprint: str


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

    def _default_recipe(self) -> CylinderRecipe:
        c = self.services.config.extrusion
        return CylinderRecipe(
            radius_mm=c.radius_mm, layer_count=c.layer_count,
            layer_height_mm=c.layer_height_mm, bead_diameter_mm=c.bead_diameter_mm,
            robot_speed_mm_s=c.robot_speed_mm_s,
            extrusion_rate_pct=c.extrusion_rate_pct,
            points_per_circle=c.points_per_circle,
            correction_enabled=c.correction_enabled,
        )

    def router(self) -> "APIRouter":
        from fastapi import APIRouter, HTTPException

        router = APIRouter()

        @router.get("/config")
        def config() -> dict:
            c = self.services.config.extrusion
            return {
                "defaults": self._default_recipe().model_dump(mode="json"),
                "integration": {
                    "extruder_tool": c.extruder_tool, "work_frame": c.work_frame,
                    "inspection_target": c.inspection_target,
                    "air_on_program": c.air_on_program, "air_off_program": c.air_off_program,
                    "valve_outputs": c.valve_outputs,
                    "mapping_source": c.valve_mapping_source,
                    "mapping_verified": c.valve_mapping_verified,
                    "hardware_io_test_approved": c.hardware_io_test_approved,
                },
                "live_print_enabled": bool(c.valve_mapping_verified and c.hardware_io_test_approved),
            }

        @router.post("/generate")
        def generate(recipe: CylinderRecipe) -> dict:
            self._plan = generate_cylinder_plan(recipe)
            # Regeneration always invalidates every pass tied to the previous hash.
            self._geometry_preflight_fingerprint = None
            self._dry_run_fingerprint = None
            return self._plan.model_dump(mode="json")

        @router.post("/preflight")
        def preflight(body: FingerprintBody) -> dict:
            if self._plan is None or body.fingerprint != self._plan.fingerprint:
                raise HTTPException(409, "toolpath changed; generate the current recipe again")
            result = geometry_preflight(self._plan)
            if result["all_ok"]:
                self._geometry_preflight_fingerprint = self._plan.fingerprint
            return result

        @router.post("/dry-run")
        def dry_run(body: FingerprintBody) -> dict:
            if self._plan is None or body.fingerprint != self._plan.fingerprint:
                raise HTTPException(409, "toolpath changed; generate the current recipe again")
            if self._geometry_preflight_fingerprint != body.fingerprint:
                raise HTTPException(409, "run geometry preflight for this toolpath first")
            missing = []
            c = self.services.config.extrusion
            if not self.services.session.is_open:
                missing.append("RoboDK station connection")
            else:
                for name in (c.extruder_tool, c.work_frame, c.inspection_target,
                             c.air_on_program, c.air_off_program):
                    if not self.services.rdk.item_exists(name):
                        missing.append(name)
            if missing:
                raise HTTPException(409, "RoboDK dry run is locked; missing: " + ", ".join(missing))
            raise HTTPException(501, "RoboDK cylinder program generation/dry-tour execution is the next implementation stage")

        @router.post("/print")
        def live_print(body: FingerprintBody) -> dict:
            c = self.services.config.extrusion
            if self._plan is None or body.fingerprint != self._plan.fingerprint:
                raise HTTPException(409, "toolpath changed; generate and dry-run it again")
            if self._dry_run_fingerprint != body.fingerprint:
                raise HTTPException(409, "current toolpath has not passed a RoboDK dry run")
            if not c.hardware_io_test_approved:
                raise HTTPException(423, "live extrusion locked until the hardware I/O test is approved")
            raise HTTPException(501, "live print execution is not implemented yet")

        @router.get("/status")
        def status() -> dict:
            fp = self._plan.fingerprint if self._plan else None
            return {
                "fingerprint": fp,
                "geometry_preflight_passed": bool(fp and fp == self._geometry_preflight_fingerprint),
                "dry_run_passed": bool(fp and fp == self._dry_run_fingerprint),
                "live_print_enabled": bool(self.services.config.extrusion.hardware_io_test_approved),
            }

        @router.get("/trials")
        def trials() -> dict:
            root = REPO_ROOT / "runs" / "extrusion"
            items = []
            if root.is_dir():
                for path in sorted(root.iterdir(), reverse=True):
                    trial_file = path / "trial.json"
                    if path.is_dir() and trial_file.is_file():
                        data = json.loads(trial_file.read_text(encoding="utf-8"))
                        data["layers_archived"] = len(list(path.glob("layer-*/manifest.json")))
                        items.append(data)
            return {"trials": items}

        return router
