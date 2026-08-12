"""Versioned trial/layer archive writer with guarded path segments."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .models import CylinderPlan, LayerManifest

_SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _segment(value: str, label: str) -> str:
    if not _SAFE.fullmatch(value) or value in {".", ".."}:
        raise ValueError(f"invalid {label}: {value!r}")
    return value


class ExtrusionArchive:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def create_trial(self, trial_id: str, plan: CylinderPlan, *, provenance: dict | None = None) -> Path:
        trial = self.root / _segment(trial_id, "trial id")
        trial.mkdir(parents=True, exist_ok=False)
        payload = {
            "schema_version": "1.0", "trial_id": trial_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "toolpath_fingerprint": plan.fingerprint,
            "recipe": plan.recipe.model_dump(mode="json"),
            "layer_count": len(plan.layers), "provenance": provenance or {},
        }
        (trial / "trial.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return trial

    def write_layer(self, manifest: LayerManifest, *, nominal_xyz, commanded_xyz,
                    measured_xyz=None, corrected_xyz=None, color=None, depth=None) -> Path:
        trial = self.root / _segment(manifest.trial_id, "trial id")
        if not (trial / "trial.json").is_file():
            raise FileNotFoundError(f"trial does not exist: {manifest.trial_id}")
        layer = trial / f"layer-{manifest.layer_index:03d}"
        layer.mkdir(parents=False, exist_ok=False)
        self._json_path(layer / "nominal_path.json", nominal_xyz)
        self._json_path(layer / "commanded_path.json", commanded_xyz)
        if measured_xyz is not None:
            self._json_path(layer / "measured_path.json", measured_xyz)
        if corrected_xyz is not None:
            self._json_path(layer / "corrected_path.json", corrected_xyz)
        if color is not None:
            import cv2
            if not cv2.imwrite(str(layer / "color.png"), np.asarray(color)):
                raise OSError("failed to write color.png")
        if depth is not None:
            np.save(layer / "depth.npy", np.asarray(depth))
        (layer / "manifest.json").write_text(
            manifest.model_dump_json(indent=2), encoding="utf-8")
        return layer

    @staticmethod
    def _json_path(path: Path, xyz) -> None:
        pts = np.asarray(xyz, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3 or not np.isfinite(pts).all():
            raise ValueError("archived paths must be finite Nx3 arrays")
        path.write_text(json.dumps({"frame": "work", "units": "mm", "points": pts.tolist()},
                                   indent=2), encoding="utf-8")
