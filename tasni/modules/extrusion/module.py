"""FastAPI integration for the Post-Extrusion Cylinder Test module."""
from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel, Field

from ...core.jobrunner import JobBusy
from ...core.logging import REPO_ROOT
from ..base import ServiceContainer, WorkflowModule
from .archive import _segment
from .figures import ensure_figure
from .inspection import inspection_plan
from .measure import (MODE as MEASURE_MODE, MeasureSession, RingCharacterizeJob,
                      RingMeasureJob, paper_summary)
from .models import CylinderPlan, CylinderRecipe, CylinderSetup
from .service import (CylinderDryRunJob, CylinderPrintJob, _utcnow, geometry_preflight,
                      reprocess_saved_layer, station_requirements)
from .surface import (CENTER_FRAME_NAME, SURFACE_OBJECT_NAME, active_scan_payload,
                      mesh_corners, resolve_platform_center, surface_fit)
from .toolpath import corrected_cylinder_plan, generate_cylinder_plan

if TYPE_CHECKING:
    from fastapi import APIRouter

try:  # FastAPI is optional at import time; the helpers below need HTTPException.
    from fastapi import HTTPException
except ImportError:  # pragma: no cover - exercised only without the web extra
    HTTPException = None  # type: ignore[assignment]


class GenerateBody(BaseModel):
    recipe: CylinderRecipe
    setup: CylinderSetup


class FingerprintBody(BaseModel):
    fingerprint: str


class QuickSimBody(FingerprintBody):
    layer_indices: list[int] = Field(default_factory=list)
    approve_full_plan: bool = False


class RestoreQuickSimBody(FingerprintBody):
    confirm_restore: bool = False


class PrintBody(FingerprintBody):
    confirm_live: bool = False
    collision_check_enabled: bool = True
    # Keep the generated curve/project/programs/targets in the station after the
    # run instead of cleaning them up, so a run can be examined afterwards.
    keep_artifacts: bool = False


class TcpSeedBody(BaseModel):
    print_tool: str
    work_frame: str


class MeasureSessionBody(BaseModel):
    note: str = ""


# Measure-only defaults collisions OFF (unlike every print/dry-run path, which defaults
# ON): the hand-placed ring stack is not in the station model, so RoboDK's check can only
# speak about cell furniture, and it was rejecting otherwise good camera-only inspection
# poses. IK/reachability screening still runs. Pass true to opt back in.
class MeasureLayerBody(FingerprintBody):
    layer_index: int = Field(ge=1)
    annotation: dict = Field(default_factory=dict)
    confirm_robot_motion: bool = False
    # How many takes one press buys, and how they are bought. `repeats` are
    # frames grabbed with the arm PARKED (sensing repeatability, seconds each);
    # `excursions` are whole trips out and back (which additionally measures the
    # arm's re-approach, and costs a trip each). Capped so a mistyped number
    # cannot commit the cell to a quarter of an hour of unattended motion.
    repeats: int = Field(default=1, ge=1, le=10)
    excursions: int = Field(default=1, ge=1, le=10)
    # One side-on RGB photo of the stack after the capture, via the taught
    # SideCapture / TowardsSideCapture targets. None = follow the configured
    # default (on); pass false to skip it for a take that does not need one.
    side_photo: bool | None = None
    confirm_close_range_tool_clear: bool = False
    collision_check_enabled: bool = False


class CharacterizeBody(BaseModel):
    confirm_robot_motion: bool = False
    confirm_close_range_tool_clear: bool = False
    collision_check_enabled: bool = False


class SurfaceCenterBody(BaseModel):
    """The wall footprint to fit while centring, so the answer carries its own check.

    ``work_frame`` is the frame selected in the UI: it chooses the coordinate system
    the returned centre is expressed in, not where the platform is.
    """

    radius_mm: float = Field(gt=0)
    bead_diameter_mm: float = Field(gt=0)
    work_frame: str = Field(min_length=1)


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
        self._quick_sim_fingerprint: str | None = None
        self._quick_sim_layers: set[int] = set()
        self._quick_sim_approves_full_plan = False
        self._dry_run_fingerprint: str | None = None
        self._active_quick_job: CylinderDryRunJob | None = None
        self._active_dry_job: CylinderDryRunJob | None = None
        self._active_print_job: CylinderPrintJob | None = None
        self._measure_session: MeasureSession | None = None
        self._active_measure_job = None
        self._restored_from: str | None = None

    def _measure_root(self):
        return REPO_ROOT / "runs" / "extrusion"

    def on_runs_deleted(self, stamps: set[str]) -> None:
        """Drop the cached measure session when its run folder was deleted.

        ``_session()`` only re-reads a session whose ``session.json`` still exists,
        so a cached handle to a deleted trial would otherwise survive forever — the
        Dashboard clears the run and the Extrusion page still shows its takes.
        """
        session = self._measure_session
        if session is not None and session.trial_id in stamps:
            self._measure_session = None
            self._restored_from = None

    def _platform(self, services: ServiceContainer, work_frame: str) -> dict | None:
        """The middle of the build platform expressed in ``work_frame``, or ``None``.

        Reads the STATION first — the scan-inserted centre frame, else the work-surface
        object's own mesh — and only then ``runs/scan/active.json``. That ordering is
        the point: the geometry a scan draws into RoboDK is saved with the station and
        survives a deleted run directory, so clearing sessions no longer orphans a
        placement whose surface is still sitting in the tree.

        Station reads are best-effort. A RoboDK hiccup must degrade to the disk pointer
        rather than fail the whole request, since every caller here is read-only.
        """
        center_xyz = corners = None
        if work_frame and services.session.is_open:
            try:
                center_xyz = services.rdk.frame_origin_in_frame(CENTER_FRAME_NAME, work_frame)
                points = services.rdk.object_mesh_in_frame(SURFACE_OBJECT_NAME, work_frame)
                corners = None if points is None else mesh_corners(points)
            except Exception:
                center_xyz = corners = None
        platform = resolve_platform_center(
            work_frame, center_frame_xyz=center_xyz, surface_corners=corners,
            active=active_scan_payload())
        if platform is None:
            return None
        note = "" if platform["extents_known"] else (
            f"the platform's size is not known in {work_frame!r}"
            + ("" if corners is None else " (its rectangle is not square-on to this frame)"))
        return {**platform, "note": note}

    def _session(self, *, create: bool = False) -> MeasureSession | None:
        """The MEASURE_ONLY session, always re-read from disk.

        A running job holds its OWN ``MeasureSession`` object and saves after
        every take, so the API's view must come from ``session.json`` and never
        from a cached copy that predates those saves.
        """
        root = self._measure_root()
        if (self._measure_session is not None
                and (root / self._measure_session.trial_id / "session.json").is_file()):
            self._measure_session = MeasureSession.load(root, self._measure_session.trial_id)
        elif self._measure_session is None:
            self._measure_session = MeasureSession.latest(root)
        if self._measure_session is None and create:
            if self._plan is None:
                raise HTTPException(
                    409, "generate coordinates first; a session records the plan it measures against")
            self._measure_session = MeasureSession.create(root, self._plan, note="")
        return self._measure_session

    def _restore_plan_from_session(self) -> str | None:
        """Rebuild the plan a session's characterization was applied as.

        The plan lives only in memory, so a backend restart mid-experiment used
        to leave the operator with no plan and a documented ritual ("press Apply
        FIRST") to remember -- and pressing *Center on scanned surface ->
        Generate* instead rebuilt the PRE-Apply plan and turned every later take
        into the stale-plan artifact. The session already records exactly what
        was applied, so restore that and nothing else.
        """
        if self._plan is not None:
            return self._restored_from
        session = self._session()
        applied = None if session is None else session.applied
        if not applied:
            return None
        try:
            plan = generate_cylinder_plan(
                CylinderRecipe.model_validate(applied["recipe"]),
                CylinderSetup.model_validate(applied["setup"]))
        except Exception:
            return None
        # Refuse to pretend: if the geometry no longer hashes to what was
        # applied, the recipe or the generator changed and the operator must
        # apply again deliberately.
        if plan.fingerprint != applied.get("fingerprint"):
            return None
        self._plan = plan
        self._invalidate_checks()
        self._restored_from = session.trial_id
        return self._restored_from

    def _session_base(self, session: MeasureSession):
        """What a characterization is applied ON TOP of: recipe + setup.

        The plan in memory first; then the plan the SESSION was created with, so
        a backend restart recovers the operator's own station selections and path
        resolution instead of config defaults -- which need not even name a tool
        (applying onto those raises a validation error, exactly when the operator
        is following "press Apply first" after a restart).
        """
        if self._plan is not None:
            return self._plan.recipe, self._plan.setup.model_dump(mode="json")
        trial_file = session.trial_dir / "trial.json"
        if trial_file.is_file():
            try:
                trial = json.loads(trial_file.read_text(encoding="utf-8"))
                return (CylinderRecipe.model_validate(trial["recipe"]),
                        CylinderSetup.model_validate(trial["setup"]).model_dump(mode="json"))
            except Exception:
                pass
        return self._default_recipe(), self._default_setup()

    def _invalidate_checks(self) -> None:
        self._geometry_preflight_fingerprint = None
        self._quick_sim_fingerprint = None
        self._quick_sim_layers.clear()
        self._quick_sim_approves_full_plan = False
        self._dry_run_fingerprint = None

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
            "inspection_auto": True,
            "center_x_mm": c.center_x_mm, "center_y_mm": c.center_y_mm,
            "build_plane_z_mm": c.build_plane_z_mm, "scan_run_id": None,
            "orientation_rpy_deg": list(c.orientation_rpy_deg),
            "maximum_tool_axis_spin_deg": c.max_tool_axis_spin_deg,
            "approach_clearance_mm": c.approach_clearance_mm,
            "retreat_clearance_mm": c.retreat_clearance_mm,
        }

    def _accept_dry_run(self, fingerprint: str) -> None:
        # A generation request is blocked while the job runs, but retain this
        # comparison so a stale completion can never approve a newer plan.
        if self._plan is not None and self._plan.fingerprint == fingerprint:
            self._dry_run_fingerprint = fingerprint

    def _accept_quick_sim(self, fingerprint: str, layer_indices: list[int],
                          *, approve_full_plan: bool = False) -> None:
        if self._plan is not None and self._plan.fingerprint == fingerprint:
            self._quick_sim_fingerprint = fingerprint
            self._quick_sim_layers.update(layer_indices)
            all_layers = {layer.layer_index for layer in self._plan.layers}
            self._quick_sim_approves_full_plan = bool(
                self._quick_sim_approves_full_plan or approve_full_plan
                or all_layers.issubset(self._quick_sim_layers))

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
                "measure_close_range_min_mm": c.measure_close_range_min_mm,
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

        @router.post("/current-tcp")
        def current_tcp(body: TcpSeedBody) -> dict:
            """Read only: selected print TCP expressed in the selected work frame."""
            if services.jobs.running:
                raise HTTPException(409, "cannot change the active tool/frame during a robot job")
            try:
                xyzrpw = services.rdk.current_tcp_xyzrpw(
                    body.print_tool, body.work_frame)
                return {"xyz_mm": xyzrpw[:3], "rpy_deg": xyzrpw[3:],
                        "tool": body.print_tool, "frame": body.work_frame}
            except Exception as exc:
                raise HTTPException(503, f"could not read selected TCP pose: {exc}")

        @router.get("/scan-surface")
        def scan_surface(work_frame: str = "") -> dict:
            """The platform this module would centre on, expressed in ``work_frame``."""
            platform = self._platform(services, work_frame)
            if platform is None:
                return {"applied": False, "available": False, "frame": work_frame,
                        "note": ("No platform is known for this work frame. Run the Scan "
                                 f"module and insert its result, or place a {CENTER_FRAME_NAME!r} "
                                 "frame in the middle of the platform yourself.")}
            return {"applied": True, "available": True, "note": "", **platform}

        @router.post("/center-on-surface")
        def center_on_surface(body: SurfaceCenterBody) -> dict:
            """Read only: the placement that puts this wall in the middle of the platform.

            The centre is resolved from the STATION first (see
            :func:`~.surface.resolve_platform_center`) and then expressed in the frame
            the operator selected, so the dropdown chooses the coordinate system while
            the cylinder still lands on the middle of the table.
            """
            if services.session.is_open and not services.rdk.item_exists_as(
                    body.work_frame, "frame"):
                raise HTTPException(
                    409, f"work frame {body.work_frame!r} is not in the open station")
            platform = self._platform(services, body.work_frame)
            if platform is None:
                raise HTTPException(
                    409, f"the middle of the platform is not known in {body.work_frame!r} — "
                         f"insert a scan, or add a {CENTER_FRAME_NAME!r} frame at the middle "
                         "of the platform and re-check")
            center_x, center_y = platform["center_mm"]
            fit = surface_fit(
                platform, center_x_mm=center_x, center_y_mm=center_y,
                outer_radius_mm=body.radius_mm + body.bead_diameter_mm / 2.0)
            return {
                "setup": {"work_frame": body.work_frame, "center_x_mm": center_x,
                          "center_y_mm": center_y,
                          # The platform's own height in this frame -- zero only while
                          # the selected frame happens to sit ON the surface.
                          "build_plane_z_mm": platform["center_z_mm"],
                          # Only a disk-sourced centre claims a run; a station-sourced
                          # one has none, and inventing one would arm the staleness
                          # check against a scan this placement never came from.
                          "scan_run_id": platform["run_id"]},
                "surface": {"applied": True, "available": True, "note": "", **platform},
                "fit": fit,
            }

        @router.post("/inspection-pose")
        def inspection_pose(body: FingerprintBody) -> dict:
            """Read only: where the camera will go to inspect each layer.

            Pure geometry — no station, no motion. The targets themselves are
            created (and collision-validated) per layer during the dry run, so
            this preview can be shown before RoboDK is even connected.
            """
            if self._plan is None or body.fingerprint != self._plan.fingerprint:
                raise HTTPException(409, "toolpath changed; generate the current recipe again")
            return inspection_plan(self._plan.recipe, self._plan.setup,
                                   K=services.config.camera.K,
                                   size_px=services.config.camera.size,
                                   config=services.config.extrusion)

        @router.post("/generate")
        def generate(body: GenerateBody) -> dict:
            if services.jobs.running:
                raise HTTPException(409, "cannot regenerate while a robot job is running")
            self._plan = generate_cylinder_plan(body.recipe, body.setup)
            self._restored_from = None
            self._geometry_preflight_fingerprint = None
            self._quick_sim_fingerprint = None
            self._quick_sim_layers.clear()
            self._quick_sim_approves_full_plan = False
            self._dry_run_fingerprint = None
            self._active_quick_job = None
            self._active_dry_job = None
            self._active_print_job = None
            # Same payload shape as GET /plan: the UI stores whichever it saw
            # last, so an extra key on one of them alone is a silent difference.
            return {**self._plan.model_dump(mode="json"), "restored_from": None}

        @router.get("/plan")
        def current_plan() -> dict:
            """Return the active plan so a reloaded UI can resume its workflow.

            After a backend restart this also brings back the plan a measurement
            session applied from its characterization, so the UI reopens on the
            ring it was measuring rather than on a stale default.
            """
            restored = self._restore_plan_from_session()
            if self._plan is None:
                raise HTTPException(404, "no cylinder plan has been generated")
            return {**self._plan.model_dump(mode="json"), "restored_from": restored}

        @router.post("/preflight")
        def preflight(body: FingerprintBody) -> dict:
            if self._plan is None or body.fingerprint != self._plan.fingerprint:
                raise HTTPException(409, "toolpath changed; generate the current recipe again")
            result = geometry_preflight(
                self._plan, surface=self._platform(services, self._plan.setup.work_frame),
                camera=services.config.camera, config=services.config.extrusion)
            if not services.session.is_open:
                result["station"] = {"ready": False,
                                     "error": "connect RoboDK to validate selected station items"}
            else:
                try:
                    station = station_requirements(
                        services.rdk, self._plan, services.config.extrusion)
                    if station["ready"]:
                        sampled_xyz = []
                        for layer in self._plan.layers:
                            layer_xyz = [[p.x_mm, p.y_mm, p.z_mm] for p in layer.points]
                            take = np.linspace(0, len(layer_xyz) - 1, 9, dtype=int)
                            sampled_xyz.extend(layer_xyz[index] for index in take)
                        reachability = services.rdk.extrusion_reachability_report(
                            points_xyz=np.asarray(sampled_xyz, dtype=float),
                            orientation_rpy_deg=self._plan.setup.orientation_rpy_deg,
                            print_tool=self._plan.setup.print_tool,
                            work_frame=self._plan.setup.work_frame,
                            maximum_tool_axis_spin_deg=(
                                self._plan.setup.maximum_tool_axis_spin_deg))
                        station["reachability"] = reachability
                        station["ready"] = bool(reachability["all_reachable"])
                    result["station"] = station
                except Exception as exc:
                    result["station"] = {"ready": False, "error": str(exc)}
            result["workflow_ready"] = bool(
                result["all_ok"] and result["station"]["ready"])
            if result["workflow_ready"]:
                self._geometry_preflight_fingerprint = self._plan.fingerprint
            else:
                self._geometry_preflight_fingerprint = None
            return result

        def require_simulation_ready(body: FingerprintBody) -> None:
            if self._plan is None or body.fingerprint != self._plan.fingerprint:
                raise HTTPException(409, "toolpath changed; generate the current recipe again")
            if self._geometry_preflight_fingerprint != body.fingerprint:
                raise HTTPException(409, "run geometry preflight for this toolpath first")
            if services.jobs.running:
                raise HTTPException(409, "a job is already running")

        @router.post("/quick-sim")
        def quick_sim(body: QuickSimBody) -> dict:
            require_simulation_ready(body)
            available = {layer.layer_index for layer in self._plan.layers}
            requested = sorted(set(body.layer_indices or available))
            invalid = sorted(set(requested) - available)
            if invalid:
                raise HTTPException(400, f"invalid layer selection: {invalid}")
            if not requested:
                raise HTTPException(400, "select at least one layer to simulate")
            self._active_quick_job = CylinderDryRunJob(
                services, self._plan,
                on_preview_pass=self._accept_quick_sim,
                check_collisions=False, layer_indices=requested,
                approve_full_plan=body.approve_full_plan)
            try:
                services.jobs.start(
                    self._active_quick_job, name="extrusion-quick-sim")
            except JobBusy as exc:
                raise HTTPException(409, str(exc))
            return {"status": "started", "mode": "QUICK_SIMULATION",
                    "collision_check_enabled": False,
                    "layer_indices": requested,
                    "approve_full_plan_on_pass": body.approve_full_plan,
                    "fingerprint": self._plan.fingerprint}

        @router.post("/quick-sim/restore-full-pass")
        def restore_full_quick_sim_pass(body: RestoreQuickSimBody) -> dict:
            """Restore a completed all-layer preview after a backend restart.

            Only an app-generated report for the exact active fingerprint, with
            every layer present and all safety/output-blocking fields intact, is
            accepted. Representative-layer overrides are deliberately not
            restored by this endpoint.
            """
            if not body.confirm_restore:
                raise HTTPException(400, "explicit restore confirmation is required")
            if self._plan is None or body.fingerprint != self._plan.fingerprint:
                raise HTTPException(409, "toolpath changed; run visual simulation again")
            expected = {layer.layer_index for layer in self._plan.layers}
            root = REPO_ROOT / "runs" / "extrusion-quick-simulation"
            candidates = sorted(
                root.glob("*/report.json") if root.is_dir() else [],
                key=lambda path: path.stat().st_mtime, reverse=True)
            for path in candidates:
                try:
                    report = json.loads(path.read_text(encoding="utf-8"))
                    layers = {int(item["layer_index"]) for item in report["layers"]}
                    valid = all(
                        float(item[phase]["percent_ok"]) >= 99.999
                        for item in report["layers"] for phase in ("path", "inspection"))
                except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                    continue
                if (report.get("kind") == "cylinder_quick_simulation"
                        and report.get("fingerprint") == body.fingerprint
                        and report.get("all_ok") is True
                        and report.get("returned_to_start") is True
                        and report.get("physical_outputs_blocked") is True
                        and report.get("collision_check_enabled") is False
                        and layers == expected and valid):
                    self._accept_quick_sim(
                        body.fingerprint, sorted(layers), approve_full_plan=True)
                    return {"status": "restored", "fingerprint": body.fingerprint,
                            "layer_indices": sorted(layers), "report": str(path)}
            raise HTTPException(
                409, "no completed all-layer quick simulation report matches this plan")

        @router.post("/dry-run")
        def dry_run(body: FingerprintBody) -> dict:
            require_simulation_ready(body)
            self._dry_run_fingerprint = None
            self._active_dry_job = CylinderDryRunJob(
                services, self._plan, on_pass=self._accept_dry_run,
                check_collisions=True)
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
                raise HTTPException(
                    409, "toolpath changed; generate and visually simulate it again")
            if (self._quick_sim_fingerprint != body.fingerprint
                    or not self._quick_sim_approves_full_plan):
                raise HTTPException(
                    409, "the full live run is not approved by the quick visual simulation; "
                         "simulate every layer or approve selected layers as representative")
            if not c.hardware_io_test_approved:
                raise HTTPException(423, "live extrusion locked until the hardware I/O test is approved")
            if not body.confirm_live:
                raise HTTPException(400, "explicit live-run confirmation is required")
            if services.jobs.running:
                raise HTTPException(409, "a job is already running")
            self._active_print_job = CylinderPrintJob(
                services, self._plan,
                check_collisions=body.collision_check_enabled,
                keep_artifacts=body.keep_artifacts)
            try:
                services.jobs.start(self._active_print_job, name="extrusion-print")
            except JobBusy as exc:
                raise HTTPException(409, str(exc))
            return {"status": "started", "mode": "LIVE_PRINT",
                    "collision_check_enabled": body.collision_check_enabled,
                    "keep_artifacts": body.keep_artifacts,
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
            self._quick_sim_fingerprint = None
            self._quick_sim_layers.clear()
            self._quick_sim_approves_full_plan = False
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
            self._quick_sim_fingerprint = None
            self._quick_sim_layers.clear()
            self._quick_sim_approves_full_plan = False
            self._dry_run_fingerprint = None
            self._active_quick_job = None
            self._active_dry_job = None
            self._active_print_job = None
            return {"status": "reset", "removed": removed}

        # -- ring-stack measure-only experiment (measure.py) ------------------
        # No dry-run, quick-sim or hardware-I/O gate here on purpose: the only
        # motion is the inspection move, which is collision-validated and
        # wrist-gated at execution exactly as it is inside the live print.

        @router.get("/measure/session")
        def measure_session() -> dict:
            session = self._session()
            return {"mode": MEASURE_MODE,
                    "session": None if session is None else session.to_json()}

        @router.post("/measure/session/new")
        def measure_session_new(body: MeasureSessionBody) -> dict:
            if self._plan is None:
                raise HTTPException(409, "generate coordinates first")
            if services.jobs.running:
                raise HTTPException(409, "a job is already running")
            self._measure_session = MeasureSession.create(
                self._measure_root(), self._plan, note=body.note)
            return {"mode": MEASURE_MODE, "session": self._measure_session.to_json()}

        def _require_measure_ready(confirm: bool) -> None:
            if self._plan is None:
                raise HTTPException(409, "generate coordinates first")
            if not confirm:
                raise HTTPException(400, "confirm that the robot may move to the inspection pose")
            if not services.session.is_open:
                raise HTTPException(409, "connect to RoboDK first (the camera move is a real robot motion)")
            if services.jobs.running:
                raise HTTPException(409, "a job is already running")

        @router.post("/measure/characterize")
        def measure_characterize(body: CharacterizeBody) -> dict:
            _require_measure_ready(body.confirm_robot_motion)
            session = self._session(create=True)
            self._active_measure_job = RingCharacterizeJob(
                services, self._plan, session,
                check_collisions=body.collision_check_enabled,
                close_range_tool_clear=body.confirm_close_range_tool_clear)
            try:
                services.jobs.start(self._active_measure_job, name="extrusion-characterize")
            except JobBusy as exc:
                raise HTTPException(409, str(exc))
            return {"status": "started", "mode": MEASURE_MODE, "trial_id": session.trial_id}

        @router.post("/measure/apply-characterization")
        def measure_apply() -> dict:
            session = self._session()
            if session is None or not session.characterizations:
                raise HTTPException(409, "characterize a ring first")
            if services.jobs.running:
                raise HTTPException(409, "wait for the current job to finish")
            found = session.characterizations[-1]
            self._restore_plan_from_session()
            base_recipe, base_setup = self._session_base(session)
            recipe = base_recipe.model_copy(update={
                "radius_mm": round(float(found["radius_mm"]), 1),
                "bead_diameter_mm": round(max(0.5, float(found["bead_width_mm"])), 1),
                "layer_height_mm": round(max(0.5, float(found["top_z_mean_mm"])), 1)})
            setup = CylinderSetup(**{**base_setup,
                                     "center_x_mm": float(found["center_mm"][0]),
                                     "center_y_mm": float(found["center_mm"][1]),
                                     "build_plane_z_mm": 0.0})
            self._plan = generate_cylinder_plan(recipe, setup)
            self._invalidate_checks()
            self._restored_from = None
            # Bind the session to this plan: every later take is scored against
            # it, and a restart rebuilds it from here.
            session.applied = {"characterization_index": found.get("index"),
                               "recipe": recipe.model_dump(mode="json"),
                               "setup": setup.model_dump(mode="json"),
                               "fingerprint": self._plan.fingerprint,
                               "applied_at": _utcnow()}
            session.save()
            return {**self._plan.model_dump(mode="json"), "restored_from": None}

        @router.post("/measure/layer")
        def measure_layer(body: MeasureLayerBody) -> dict:
            self._restore_plan_from_session()
            if self._plan is None or body.fingerprint != self._plan.fingerprint:
                raise HTTPException(409, "toolpath changed; generate coordinates again")
            if body.layer_index > len(self._plan.layers):
                raise HTTPException(400, f"layer_index must be 1..{len(self._plan.layers)}")
            # Data-integrity gate BEFORE the motion gates: scoring a take against
            # a plan the ring was never placed on silently produces a number the
            # paper cannot use, and detecting it needs no robot. (The second gate
            # here used to refuse layer N until layer N-1 had a measured top --
            # deleted with `floor_profile`, which measured WORSE on the only
            # stacked data there is; spec 2026-08-30 §2.4.)
            existing = self._session()
            applied = None if existing is None else existing.applied
            if applied and applied.get("fingerprint") != self._plan.fingerprint:
                raise HTTPException(
                    409,
                    "this session is bound to the plan applied from its characterization "
                    f"({str(applied.get('fingerprint'))[:10]}), but the current plan is "
                    f"{self._plan.fingerprint[:10]}. Press 'Apply to recipe & placement' "
                    "to measure against the characterized ring again — scoring a take "
                    "against a plan the ring was never placed on is the stale-plan "
                    "artifact (2026-08-28: a 15 mm centre offset that measured nothing).")
            _require_measure_ready(body.confirm_robot_motion)
            session = self._session(create=True)
            self._active_measure_job = RingMeasureJob(
                services, self._plan, session, body.layer_index,
                annotation=body.annotation, check_collisions=body.collision_check_enabled,
                close_range_tool_clear=body.confirm_close_range_tool_clear,
                repeats=body.repeats, excursions=body.excursions,
                side_photo=body.side_photo)
            try:
                services.jobs.start(self._active_measure_job, name="extrusion-measure")
            except JobBusy as exc:
                raise HTTPException(409, str(exc))
            return {"status": "started", "mode": MEASURE_MODE, "trial_id": session.trial_id,
                    "layer_index": body.layer_index, "take": session.next_take(body.layer_index),
                    "repeats": body.repeats, "excursions": body.excursions,
                    "takes_requested": body.repeats * body.excursions}

        @router.get("/status")
        def status() -> dict:
            fingerprint = self._plan.fingerprint if self._plan else None
            return {
                "status": services.jobs.status, "running": services.jobs.running,
                "result": services.jobs.result, "error": services.jobs.error,
                "fingerprint": fingerprint,
                "setup": (self._plan.setup.model_dump(mode="json")
                          if self._plan else None),
                "recipe": (self._plan.recipe.model_dump(mode="json")
                           if self._plan else None),
                "geometry_preflight_passed": bool(
                    fingerprint and fingerprint == self._geometry_preflight_fingerprint),
                "quick_sim_passed": bool(
                    fingerprint and fingerprint == self._quick_sim_fingerprint),
                "quick_sim_layers": (sorted(self._quick_sim_layers)
                                     if fingerprint == self._quick_sim_fingerprint else []),
                "quick_sim_live_approved": bool(
                    fingerprint and fingerprint == self._quick_sim_fingerprint
                    and self._quick_sim_approves_full_plan),
                "dry_run_passed": bool(
                    fingerprint and fingerprint == self._dry_run_fingerprint),
                "measure_session": (self._measure_session.trial_id
                                    if self._measure_session else None),
                "hardware_io_test_approved": bool(
                    services.config.extrusion.hardware_io_test_approved),
                "live_print_enabled": bool(
                    fingerprint and fingerprint == self._quick_sim_fingerprint and
                    self._quick_sim_approves_full_plan and
                    services.config.extrusion.hardware_io_test_approved),
            }

        @router.get("/trials")
        def trials() -> dict:
            root = self._measure_root()
            items = []
            recipe_keys: set[str] = set()
            total_trials = total_layers = 0
            measure_only_trials = measure_only_takes = 0
            if root.is_dir():
                for path in sorted(root.iterdir(), reverse=True):
                    trial_file = path / "trial.json"
                    if path.is_dir() and trial_file.is_file():
                        data = json.loads(trial_file.read_text(encoding="utf-8"))
                        mode = data.get("mode", "LIVE_PRINT")
                        data["mode"] = mode
                        layers = []
                        for manifest_path in sorted(path.glob("layer-*/manifest.json")):
                            try:
                                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                                layers.append({
                                    "layer_index": manifest.get("layer_index"),
                                    "take": manifest.get("take", 1),
                                    "annotation": manifest.get("annotation", {}),
                                    "metrics": manifest.get("metrics"),
                                    "geometry": manifest.get("geometry"),
                                    "valid": bool((manifest.get("metrics") or {}).get("valid")),
                                    "layer_dir": manifest_path.parent.name,
                                    "has_comparison": (manifest_path.parent / "comparison.png").is_file(),
                                    "has_side_view": (manifest_path.parent / "side.png").is_file(),
                                    "side_view": manifest.get("side_view"),
                                    "has_figures": (manifest_path.parent / "figures").is_dir(),
                                })
                            except (OSError, ValueError, json.JSONDecodeError):
                                continue
                        data["layers_archived"] = len(layers)
                        data["layers"] = layers
                        # A measurement session is not a print. Counting one as a
                        # trial would inflate exactly the number the paper cites.
                        if mode == "LIVE_PRINT":
                            recipe_keys.add(json.dumps(data.get("recipe", {}), sort_keys=True))
                            total_trials += 1
                            total_layers += len(layers)
                        else:
                            measure_only_trials += 1
                            measure_only_takes += len(layers)
                        items.append(data)
            return {"summary": {"total_trials": total_trials,
                                "total_layers": total_layers,
                                "total_recipes": len(recipe_keys),
                                "measure_only_trials": measure_only_trials,
                                "measure_only_takes": measure_only_takes},
                    "trials": items}

        # Raw frames and derived rasters the processing chain already writes.
        # Deliberately narrow: depth.npy and manifest.json are data, not pictures,
        # and reach the UI through /trials instead.
        SERVED_FILES = {"color.png": "image/png", "comparison.png": "image/png",
                        "segmentation.png": "image/png", "skeleton.png": "image/png",
                        # The side-on photo of the stack: a figure for the paper,
                        # taken after the layer's capture from the taught pose.
                        "side.png": "image/png"}
        FIGURE_TYPES = {"png": "image/png", "pdf": "application/pdf"}

        def _layer_directory(trial_id: str, layer_dir: str):
            """Resolve one take's directory, or 404. The segments come from a URL."""
            try:
                trial = _segment(trial_id, "trial id")
                name = _segment(layer_dir, "layer directory")
            except ValueError as exc:
                raise HTTPException(404, str(exc)) from exc
            if not name.startswith("layer-"):
                raise HTTPException(404, f"not a layer directory: {layer_dir!r}")
            root = self._measure_root().resolve()
            path = (root / trial / name).resolve()
            if root not in path.parents or not (path / "manifest.json").is_file():
                raise HTTPException(404, f"no archived take at {trial_id}/{layer_dir}")
            return path

        @router.get("/trials/{trial_id}/layers/{layer_dir}/files/{name}")
        def layer_file(trial_id: str, layer_dir: str, name: str):
            from fastapi.responses import FileResponse
            media = SERVED_FILES.get(name)
            if media is None:
                raise HTTPException(404, f"not a served file: {name!r}")
            path = _layer_directory(trial_id, layer_dir) / name
            if not path.is_file():
                raise HTTPException(404, f"{name} was not archived for this take")
            return FileResponse(path, media_type=media)

        @router.get("/trials/{trial_id}/layers/{layer_dir}/figures/{name}")
        def layer_figure(trial_id: str, layer_dir: str, name: str):
            """Render on first request, then serve. Never touches the robot."""
            from fastapi.responses import FileResponse
            media = FIGURE_TYPES.get(name.rpartition(".")[2])
            if media is None:
                raise HTTPException(404, f"not a figure: {name!r}")
            path = _layer_directory(trial_id, layer_dir)
            try:
                return FileResponse(ensure_figure(path, name), media_type=media)
            except (ValueError, FileNotFoundError) as exc:
                raise HTTPException(404, str(exc)) from exc
            except ImportError as exc:                       # the `figures` extra
                raise HTTPException(
                    503, f"figures need matplotlib (pip install -e .[figures]): {exc}") from exc

        @router.get("/trials/{trial_id}/figures/{name}")
        def trial_figure(trial_id: str, name: str):
            """Every layer's latest take in one picture -- plan and oblique."""
            from fastapi.responses import FileResponse
            media = FIGURE_TYPES.get(name.rpartition(".")[2])
            if media is None:
                raise HTTPException(404, f"not a figure: {name!r}")
            try:
                trial = _segment(trial_id, "trial id")
            except ValueError as exc:
                raise HTTPException(404, str(exc)) from exc
            root = self._measure_root().resolve()
            path = (root / trial).resolve()
            if root not in path.parents or not (path / "trial.json").is_file():
                raise HTTPException(404, f"no such trial: {trial_id}")
            try:
                return FileResponse(ensure_figure(path, name), media_type=media)
            except (ValueError, FileNotFoundError) as exc:
                raise HTTPException(404, str(exc)) from exc

        @router.get("/trials/{trial_id}/paper-draft.docx")
        def trial_paper_docx(trial_id: str):
            """The results as a Word file: real tables, paste straight into the paper.

            Rebuilt from the archive on every request, so it is always the run as
            it stands -- including a section naming what the run still owes.
            """
            from fastapi.responses import FileResponse
            from .paper_docx import build_paper_docx        # the `docx` extra
            try:
                trial = _segment(trial_id, "trial id")
            except ValueError as exc:
                raise HTTPException(404, str(exc)) from exc
            root = self._measure_root()
            if not (root / trial / "trial.json").is_file():
                raise HTTPException(404, f"trial does not exist: {trial}")
            try:
                path = build_paper_docx(
                    root, trial, robot_name=services.config.robodk.robot_name or None)
            except ImportError as exc:
                raise HTTPException(
                    503, "the Word draft needs python-docx "
                         f"(pip install -e .[docx]): {exc}") from exc
            return FileResponse(
                path,
                media_type="application/vnd.openxmlformats-officedocument."
                           "wordprocessingml.document",
                filename=f"{trial}-paper-draft.docx")

        @router.get("/trials/{trial_id}/paper-summary")
        def trial_paper_summary(trial_id: str) -> dict:
            try:
                return paper_summary(self._measure_root(), trial_id)
            except (FileNotFoundError, ValueError) as exc:
                raise HTTPException(404, str(exc)) from exc

        @router.post("/trials/{trial_id}/layers/{layer_index}/reprocess")
        def reprocess(trial_id: str, layer_index: int, take: int = 1) -> dict:
            if services.jobs.running:
                raise HTTPException(409, "cannot reprocess while a robot job is running")
            try:
                return reprocess_saved_layer(
                    REPO_ROOT / "runs" / "extrusion", trial_id, layer_index, take)
            except (ValueError, FileNotFoundError) as exc:
                raise HTTPException(404, str(exc)) from exc
            except RuntimeError as exc:
                raise HTTPException(422, str(exc)) from exc

        return router
