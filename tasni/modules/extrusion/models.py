"""Typed, unit-explicit records for cylinder trials and measured layers."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MaterialComponent(_Record):
    name: str
    quantity: float = Field(ge=0)
    unit: str


class CylinderRecipe(_Record):
    radius_mm: float = Field(gt=0, le=500)
    layer_count: int = Field(ge=1, le=100)
    layer_height_mm: float = Field(gt=0, le=50)
    bead_diameter_mm: float = Field(gt=0, le=50)
    robot_speed_mm_s: float = Field(gt=0, le=1000)
    travel_speed_mm_s: float = Field(default=200.0, gt=0, le=2000)
    path_rounding_mm: float = Field(default=1.0, ge=0, le=100)
    extrusion_rate_pct: float = Field(ge=0, le=100)
    points_per_circle: int = Field(default=180, ge=24, le=4000)
    correction_enabled: bool = False
    material: list[MaterialComponent] = Field(default_factory=list)


class CylinderSetup(_Record):
    """Every station/motion choice that changes what the robot will execute."""

    print_tool: str = Field(min_length=1)
    work_frame: str = Field(min_length=1)
    inspection_tool: str = Field(min_length=1)
    # Empty only in automatic mode, where the pose is derived from this placement
    # and a target is created per layer instead of taught once (see
    # ``modules/extrusion/inspection.py``). The generated name is deliberately NOT
    # stored here: it embeds the fingerprint, which is computed from this setup.
    inspection_target: str = ""
    inspection_auto: bool = False
    center_x_mm: float = 0.0
    center_y_mm: float = 0.0
    build_plane_z_mm: float = 0.0
    # Set only when the placement came from the scanned work surface; it makes the
    # originating scan part of the fingerprint, so re-scanning the table invalidates
    # a plan whose coordinates were derived from the previous surface.
    scan_run_id: str | None = None
    orientation_rpy_deg: tuple[float, float, float] = (0.0, 0.0, 0.0)
    maximum_tool_axis_spin_deg: float = Field(default=90.0, gt=0, le=180)
    approach_clearance_mm: float = Field(default=40.0, gt=0, le=500)
    retreat_clearance_mm: float = Field(default=60.0, gt=0, le=500)

    @model_validator(mode="after")
    def _target_or_auto(self) -> "CylinderSetup":
        if not self.inspection_auto and not self.inspection_target:
            raise ValueError(
                "inspection_target is required unless inspection_auto is set: "
                "manual inspection must name the taught target it moves to")
        return self


class PathPoint(_Record):
    x_mm: float
    y_mm: float
    z_mm: float


class LayerPath(_Record):
    layer_index: int = Field(ge=1)
    nominal_z_mm: float
    points: list[PathPoint]


class CylinderPlan(_Record):
    schema_version: str = "1.0"
    fingerprint: str
    recipe: CylinderRecipe
    setup: CylinderSetup
    layers: list[LayerPath]
    total_path_length_mm: float


class DeviationMetrics(_Record):
    mean_absolute_mm: float
    rms_mm: float
    maximum_mm: float
    measured_center_mm: tuple[float, float]
    measured_radius_mm: float
    path_completeness: float
    maximum_angular_gap_deg: float
    valid: bool
    # Fitted-circle centre minus the NOMINAL centre: the direct readout of a
    # bodily displacement of the ring (the paper's introduced-offset check).
    center_offset_mm: tuple[float, float] = (0.0, 0.0)
    center_offset_norm_mm: float = 0.0
    # Radial scatter about the FITTED circle: "ring is not round", separated
    # from "ring placed wrong".
    shape_rms_mm: float = 0.0
    shape_max_mm: float = 0.0
    warnings: list[str] = Field(default_factory=list)


class LayerManifest(_Record):
    schema_version: str = "1.0"
    trial_id: str
    layer_index: int = Field(ge=1)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    mode: str = "LIVE_PRINT"
    recipe: CylinderRecipe
    toolpath_fingerprint: str
    nominal_path_file: str = "nominal_path.json"
    commanded_path_file: str = "commanded_path.json"
    measured_path_file: str | None = None
    corrected_path_file: str | None = None
    color_file: str | None = None
    depth_file: str | None = None
    pointcloud_file: str | None = None
    metrics: DeviationMetrics | None = None
    processing: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
    valve_transitions: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
