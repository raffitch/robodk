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

    @staticmethod
    def _layer_name(layer_index: int, take: int) -> str:
        """Take 1 keeps the historical ``layer-NNN`` name; repeats get a suffix."""
        if layer_index < 1 or take < 1:
            raise ValueError("layer index and take must be positive")
        return (f"layer-{layer_index:03d}" if take == 1
                else f"layer-{layer_index:03d}-take{take:02d}")

    def create_trial(self, trial_id: str, plan: CylinderPlan, *, provenance: dict | None = None,
                     mode: str = "LIVE_PRINT", experiment: dict | None = None) -> Path:
        trial = self.root / _segment(trial_id, "trial id")
        trial.mkdir(parents=True, exist_ok=False)
        payload = {
            "schema_version": "1.0", "trial_id": trial_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "toolpath_fingerprint": plan.fingerprint,
            "recipe": plan.recipe.model_dump(mode="json"),
            "setup": plan.setup.model_dump(mode="json"),
            "layer_count": len(plan.layers), "provenance": provenance or {},
            # A measurement session is not a print and must never be counted as
            # one; ``experiment`` carries what the operator was doing.
            "mode": mode, "experiment": experiment or {},
        }
        (trial / "trial.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return trial

    def layer_dir(self, trial_id: str, layer_index: int, *, take: int = 1,
                  require: bool = True) -> Path:
        layer = self.root / _segment(trial_id, "trial id") / self._layer_name(layer_index, take)
        if require and not (layer / "manifest.json").is_file():
            raise FileNotFoundError(f"archived layer does not exist: {trial_id}/{layer.name}")
        return layer

    def write_layer(self, manifest: LayerManifest, *, nominal_xyz, commanded_xyz,
                    measured_xyz=None, corrected_xyz=None, color=None, depth=None,
                    pointcloud_xyz=None,
                    derived_images: dict[str, np.ndarray] | None = None,
                    report: dict | None = None) -> Path:
        trial = self.root / _segment(manifest.trial_id, "trial id")
        if not (trial / "trial.json").is_file():
            raise FileNotFoundError(f"trial does not exist: {manifest.trial_id}")
        layer = trial / self._layer_name(manifest.layer_index, manifest.take)
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
        if pointcloud_xyz is not None:
            points = np.asarray(pointcloud_xyz, dtype=float)
            if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
                raise ValueError("archived point cloud must be a finite Nx3 array")
            np.save(layer / "height-or-pointcloud.npy", points)
        if derived_images:
            import cv2
            allowed = {"segmentation.png", "skeleton.png", "comparison.png"}
            for name, image in derived_images.items():
                if name not in allowed:
                    raise ValueError(f"unsupported derived image name: {name!r}")
                if not cv2.imwrite(str(layer / name), np.asarray(image)):
                    raise OSError(f"failed to write {name}")
        if report is not None:
            (layer / "report.json").write_text(json.dumps(report, indent=2),
                                                encoding="utf-8")
        (layer / "manifest.json").write_text(
            manifest.model_dump_json(indent=2), encoding="utf-8")
        return layer

    def write_characterization(self, trial_id: str, index: int, *, color, depth, measured_xyz,
                               derived_images: dict[str, np.ndarray], report: dict) -> Path:
        """Archive one ring characterization beside the trial's layers.

        Not a layer: it measures the ring before any recipe exists, so it must
        not appear under ``layer-*`` where the take listings and paper summary
        would count it as a measurement of a planned layer.
        """
        trial = self.root / _segment(trial_id, "trial id")
        if not (trial / "trial.json").is_file():
            raise FileNotFoundError(f"trial does not exist: {trial_id}")
        out = trial / f"characterize-{index:02d}"
        out.mkdir(parents=False, exist_ok=False)
        import cv2
        if not cv2.imwrite(str(out / "color.png"), np.asarray(color)):
            raise OSError("failed to write color.png")
        np.save(out / "depth.npy", np.asarray(depth))
        self._json_path(out / "measured_path.json", measured_xyz)
        allowed = {"segmentation.png", "skeleton.png", "comparison.png"}
        for name, image in derived_images.items():
            if name not in allowed:
                raise ValueError(f"unsupported derived image name: {name!r}")
            if not cv2.imwrite(str(out / name), np.asarray(image)):
                raise OSError(f"failed to write {name}")
        (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return out

    def rewrite_processing(self, manifest: LayerManifest, *, measured_xyz,
                           corrected_xyz=None, pointcloud_xyz=None,
                           derived_images: dict[str, np.ndarray], report: dict) -> Path:
        """Replace derived artifacts for a saved raw observation, never raw input."""
        layer = self.layer_dir(manifest.trial_id, manifest.layer_index, take=manifest.take)
        self._json_path(layer / "measured_path.json", measured_xyz)
        corrected_path = layer / "corrected_path.json"
        if corrected_xyz is not None:
            self._json_path(corrected_path, corrected_xyz)
        elif corrected_path.exists():
            corrected_path.unlink()
        if pointcloud_xyz is not None:
            points = np.asarray(pointcloud_xyz, dtype=float)
            if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
                raise ValueError("archived point cloud must be a finite Nx3 array")
            np.save(layer / "height-or-pointcloud.npy", points)
        import cv2
        allowed = {"segmentation.png", "skeleton.png", "comparison.png"}
        if set(derived_images) - allowed:
            raise ValueError("unsupported derived image name")
        for name, image in derived_images.items():
            if not cv2.imwrite(str(layer / name), np.asarray(image)):
                raise OSError(f"failed to write {name}")
        (layer / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
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
