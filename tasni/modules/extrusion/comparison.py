"""Nominal/measured comparison and bounded, opt-in radial compensation."""
from __future__ import annotations

import math

import numpy as np

from .models import DeviationMetrics


def _points(value) -> np.ndarray:
    pts = np.asarray(value, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 3 or not np.isfinite(pts).all():
        raise ValueError("path must contain at least three finite XYZ points")
    return pts


def fit_circle_xy(measured_xyz) -> tuple[np.ndarray, float]:
    pts = _points(measured_xyz)
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack((2.0 * x, 2.0 * y, np.ones(len(pts))))
    b = x * x + y * y
    cx, cy, c = np.linalg.lstsq(A, b, rcond=None)[0]
    radius = math.sqrt(max(0.0, c + cx * cx + cy * cy))
    return np.array([cx, cy], dtype=float), float(radius)


def compare_circle(measured_xyz, nominal_radius_mm: float, *,
                   min_points: int = 24, min_completeness: float = 0.90,
                   max_gap_deg: float = 30.0) -> DeviationMetrics:
    pts = _points(measured_xyz)
    center, measured_radius = fit_circle_xy(pts)
    radii = np.linalg.norm(pts[:, :2] - center, axis=1)
    deviation = radii - float(nominal_radius_mm)
    angles = np.mod(np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0]), 2 * math.pi)
    ordered = np.sort(np.unique(angles))
    gaps = np.diff(np.r_[ordered, ordered[0] + 2 * math.pi])
    gap_deg = float(np.degrees(gaps.max()))
    completeness = float(max(0.0, 1.0 - gaps.max() / (2 * math.pi)))
    warnings: list[str] = []
    if len(pts) < min_points:
        warnings.append(f"only {len(pts)} measured points; require {min_points}")
    if completeness < min_completeness:
        warnings.append(f"path completeness {completeness:.3f} below {min_completeness:.3f}")
    if gap_deg > max_gap_deg:
        warnings.append(f"maximum angular gap {gap_deg:.1f} deg exceeds {max_gap_deg:.1f} deg")
    return DeviationMetrics(
        mean_absolute_mm=float(np.mean(np.abs(deviation))),
        rms_mm=float(np.sqrt(np.mean(deviation * deviation))),
        maximum_mm=float(np.max(np.abs(deviation))),
        measured_center_mm=(float(center[0]), float(center[1])),
        measured_radius_mm=measured_radius,
        path_completeness=completeness,
        maximum_angular_gap_deg=gap_deg,
        valid=not warnings,
        warnings=warnings,
    )


def corrected_circle(measured_xyz, nominal_radius_mm: float, nominal_z_mm: float, *,
                     point_count: int = 180, gain: float = 1.0,
                     smoothing_points: int = 9, max_correction_mm: float = 10.0) -> np.ndarray:
    """Mirror measured radial error into a bounded, cyclically smoothed command."""
    pts = _points(measured_xyz)
    center, _ = fit_circle_xy(pts)
    angles = np.mod(np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0]), 2 * math.pi)
    radial_error = np.linalg.norm(pts[:, :2] - center, axis=1) - nominal_radius_mm
    order = np.argsort(angles)
    angles, radial_error = angles[order], radial_error[order]
    theta = np.linspace(0, 2 * math.pi, point_count, endpoint=False)
    ext_a = np.r_[angles[-1] - 2 * math.pi, angles, angles[0] + 2 * math.pi]
    ext_e = np.r_[radial_error[-1], radial_error, radial_error[0]]
    error = np.interp(theta, ext_a, ext_e)
    width = min(len(error), max(1, int(smoothing_points)))
    if width % 2 == 0 and width > 1:
        width -= 1
    if width > 1:
        pad = width // 2
        error = np.convolve(np.r_[error[-pad:], error, error[:pad]],
                            np.ones(width) / width, mode="valid")
    correction = np.clip(-gain * error, -max_correction_mm, max_correction_mm)
    radius = nominal_radius_mm + correction
    result = np.column_stack((radius * np.cos(theta), radius * np.sin(theta),
                              np.full(point_count, nominal_z_mm)))
    return np.vstack((result, result[0]))
