"""Typed, unit-explicit records for cylinder trials and measured layers."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


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
    extrusion_rate_pct: float = Field(ge=0, le=100)
    points_per_circle: int = Field(default=180, ge=24, le=4000)
    correction_enabled: bool = False
    material: list[MaterialComponent] = Field(default_factory=list)


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
    metrics: DeviationMetrics | None = None
    processing: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)
    valve_transitions: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
