"""Deterministic circular-layer generation and dry-run identity handling."""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np

from .models import CylinderPlan, CylinderRecipe, LayerPath, PathPoint


def recipe_fingerprint(recipe: CylinderRecipe) -> str:
    payload = recipe.model_dump(mode="json", exclude={"material"})
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def generate_cylinder_plan(recipe: CylinderRecipe) -> CylinderPlan:
    """Generate closed, counter-clockwise circles in the work frame, in mm.

    Layer one is centred one bead radius above the build plane. Subsequent
    centre-lines advance by the configured layer height. The final point exactly
    repeats the first so RoboDK receives an explicitly closed extrusion segment.
    """
    theta = np.linspace(0.0, 2.0 * math.pi, recipe.points_per_circle + 1)
    x = recipe.radius_mm * np.cos(theta)
    y = recipe.radius_mm * np.sin(theta)
    x[-1], y[-1] = x[0], y[0]
    layers: list[LayerPath] = []
    for index in range(recipe.layer_count):
        z = recipe.bead_diameter_mm / 2.0 + index * recipe.layer_height_mm
        points = [PathPoint(x_mm=float(px), y_mm=float(py), z_mm=float(z))
                  for px, py in zip(x, y)]
        layers.append(LayerPath(layer_index=index + 1, nominal_z_mm=z, points=points))
    return CylinderPlan(
        fingerprint=recipe_fingerprint(recipe), recipe=recipe, layers=layers,
        total_path_length_mm=float(recipe.layer_count * 2.0 * math.pi * recipe.radius_mm),
    )


def points_array(layer: LayerPath) -> np.ndarray:
    return np.array([[p.x_mm, p.y_mm, p.z_mm] for p in layer.points], dtype=float)
