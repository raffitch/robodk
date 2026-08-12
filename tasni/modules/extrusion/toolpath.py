"""Deterministic circular-layer generation and dry-run identity handling."""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np

from .models import CylinderPlan, CylinderRecipe, CylinderSetup, LayerPath, PathPoint


def plan_fingerprint(recipe: CylinderRecipe, setup: CylinderSetup,
                     layers: list[LayerPath] | None = None) -> str:
    payload = {
        "recipe": recipe.model_dump(mode="json"),
        "setup": setup.model_dump(mode="json"),
        "commanded_layers": ([layer.model_dump(mode="json") for layer in layers]
                             if layers is not None else None),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def generate_cylinder_plan(recipe: CylinderRecipe, setup: CylinderSetup) -> CylinderPlan:
    """Generate closed, counter-clockwise circles in the work frame, in mm.

    Layer one is centred one bead radius above the build plane. Subsequent
    centre-lines advance by the configured layer height. The final point exactly
    repeats the first so RoboDK receives an explicitly closed extrusion segment.
    """
    theta = np.linspace(0.0, 2.0 * math.pi, recipe.points_per_circle + 1)
    x = setup.center_x_mm + recipe.radius_mm * np.cos(theta)
    y = setup.center_y_mm + recipe.radius_mm * np.sin(theta)
    x[-1], y[-1] = x[0], y[0]
    layers: list[LayerPath] = []
    for index in range(recipe.layer_count):
        z = (setup.build_plane_z_mm + recipe.bead_diameter_mm / 2.0
             + index * recipe.layer_height_mm)
        points = [PathPoint(x_mm=float(px), y_mm=float(py), z_mm=float(z))
                  for px, py in zip(x, y)]
        layers.append(LayerPath(layer_index=index + 1, nominal_z_mm=z, points=points))
    return CylinderPlan(
        fingerprint=plan_fingerprint(recipe, setup, layers), recipe=recipe, setup=setup, layers=layers,
        total_path_length_mm=float(recipe.layer_count * 2.0 * math.pi * recipe.radius_mm),
    )


def points_array(layer: LayerPath) -> np.ndarray:
    return np.array([[p.x_mm, p.y_mm, p.z_mm] for p in layer.points], dtype=float)


def corrected_cylinder_plan(plan: CylinderPlan, corrected_reference_xyz: np.ndarray) -> CylinderPlan:
    """Apply one measured radial compensation profile to every layer.

    This creates a NEW fingerprint and therefore requires a new dry run before it
    can print. Merely calculating correction can never count as executing it.
    """
    reference = np.asarray(corrected_reference_xyz, dtype=float)
    if reference.ndim != 2 or reference.shape[1] != 3 or len(reference) < 3:
        raise ValueError("corrected reference must be an Nx3 path")
    layers: list[LayerPath] = []
    for original in plan.layers:
        xyz = reference.copy()
        xyz[:, 2] = original.nominal_z_mm
        points = [PathPoint(x_mm=float(p[0]), y_mm=float(p[1]), z_mm=float(p[2]))
                  for p in xyz]
        layers.append(LayerPath(layer_index=original.layer_index,
                                nominal_z_mm=original.nominal_z_mm, points=points))
    return CylinderPlan(
        fingerprint=plan_fingerprint(plan.recipe, plan.setup, layers),
        recipe=plan.recipe, setup=plan.setup, layers=layers,
        total_path_length_mm=float(sum(np.linalg.norm(np.diff(points_array(layer), axis=0), axis=1).sum()
                                       for layer in layers)),
    )
