"""Substrate reference models for deposit segmentation (design 2026-08-30).

The segmentation question is "how high is this point above the surface the
deposit rests on" -- never "what colour is it" (a free-running auto-exposure
made that an uncalibrated quantity) and never "what is its Z in the work frame"
(the board was measured 1.2 mm below work Z=0 and tilted ~0.5 deg). One
contract answers it for every consumer; the fitted plane is the one provider
that ships. Further providers (a captured empty-plate reference, layer N-1's
measured top) plug into the same interface WHEN evidence demands them --
building them now was measured to be speculative (spec §11).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

import numpy as np


class SubstrateModel(Protocol):
    source: str
    sigma_mm: float

    def height(self, xyz: np.ndarray) -> np.ndarray: ...
    def floor_mm(self, k: float) -> float: ...
    def to_report(self) -> dict: ...


def _sigma_low(residual: np.ndarray) -> float:
    """One-sided scale: median minus p15.87. Deposit contaminates only the
    positive side, so the lower half of the residuals is pure sensor noise;
    for a Gaussian this equals sigma exactly, where a two-sided MAD is
    inflated by the bead (spec §3.3)."""
    return float(np.median(residual) - np.percentile(residual, 15.87))


@lru_cache(maxsize=None)
def _tukey_one_sided_bias(c_positive: float, c_negative: float) -> float:
    """Fisher-consistency correction for the one-sided Tukey biweight,
    in units of sigma.

    An asymmetric redescending weight (c+ != c-) is NOT a free lunch: for a
    clean, symmetric (Gaussian) residual population it does not converge to
    the true center, because points a given distance above the current fit
    are downweighted harder than points the same distance below it -- the
    IRLS fixed point is systematically pulled toward the more-gently-weighted
    (negative) side. That shift is provable and exactly reproducible (not
    sampling noise: it holds with zero bead contamination and is stable
    across iteration counts) -- confirmed both by closed-form solution here
    and by direct Monte Carlo simulation while implementing this module.
    Left uncorrected it puts the recovered plane measurably BELOW the true
    substrate, which is the opposite of "conservative" for a segmentation
    that measures height above it.

    This solves the one-dimensional location-family estimating equation
    E[w(Z - s) * (Z - s)] = 0 for a standard normal Z by bisection over a
    fixed, deterministic quadrature grid -- no RNG, no scipy, pure numpy.
    The result depends only on (c_positive, c_negative), so it is solved
    once per distinct pair and cached; `fit()` adds `-shift * sigma_mm` to
    the intercept once per call, which is cheap relative to the cached
    lookup.
    """
    z = np.linspace(-12.0, 12.0, 20_001)
    pdf = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)

    def estimating_eq(shift: float) -> float:
        r = z - shift
        c = np.where(r > 0.0, c_positive, c_negative)
        t = np.clip(np.abs(r / c), 0.0, 1.0)
        w = np.square(1.0 - np.square(t))
        return float(np.trapz(w * r * pdf, z))

    lo, hi = -3.0, 3.0
    g_lo = estimating_eq(lo)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        g_mid = estimating_eq(mid)
        if (g_mid > 0.0) == (g_lo > 0.0):
            lo, g_lo = mid, g_mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@dataclass(frozen=True)
class PlaneSubstrate:
    """z = a*x + b*y + c, fitted by deterministic one-sided IRLS.

    Positive residuals are down-weighted harder than negative ones (Tukey
    c+ = 2.0 vs c- = 4.685) because the deposit is the only thing that can sit
    ABOVE the surface. RANSAC was measured and rejected: Open3D 0.17 ignores
    its seed (a different plane per run) and one-sided IRLS beat it 0.064 mm
    to 0.191 mm mean |error| against ground truth (spec §3.2).

    That asymmetric weighting is exactly what makes the raw IRLS fixed point
    NOT the true plane for clean data -- see `_tukey_one_sided_bias` for the
    closed-form correction `fit()` applies to the intercept. Do not remove it
    to "simplify" back to the textbook (biased, for this weight shape) IRLS
    fixed point.
    """
    a: float
    b: float
    c: float
    sigma_mm: float
    inlier_fraction: float
    clamp_mm: tuple[float, float]
    source: str = "fitted_plane"

    @classmethod
    def fit(cls, xyz, *, clamp_mm=(1.0, 2.0), max_tilt_deg=25.0,
            iterations=12, c_positive=2.0, c_negative=4.685) -> "PlaneSubstrate":
        pts = np.asarray(xyz, dtype=float)
        if len(pts) < 50:
            raise RuntimeError(
                f"substrate fit needs at least 50 points, got {len(pts)} -- "
                "widen substrate_fit_radius_mm or check the depth stream")
        design = np.column_stack([pts[:, 0], pts[:, 1], np.ones(len(pts))])
        z = pts[:, 2]
        coeff, *_ = np.linalg.lstsq(design, z, rcond=None)   # plain LS seed
        for _ in range(iterations):
            residual = z - design @ coeff
            scale = _sigma_low(residual)
            if scale <= 0.0:
                break
            cutoff = np.where(residual > 0.0, c_positive * scale, c_negative * scale)
            t = np.clip(np.abs(residual / cutoff), 0.0, 1.0)
            weight = np.square(1.0 - np.square(t))           # Tukey biweight
            sw = np.sqrt(weight)
            coeff, *_ = np.linalg.lstsq(design * sw[:, None], z * sw, rcond=None)
        residual = z - design @ coeff
        sigma = float(max(_sigma_low(residual), 0.0))
        # The IRLS fixed point above is systematically offset from the true
        # plane by a known constant times sigma (see _tukey_one_sided_bias);
        # correct the intercept only -- x/y are ROI-centred, so the shift
        # loads onto the constant term, not the slope.
        coeff = coeff.copy()
        coeff[2] -= _tukey_one_sided_bias(c_positive, c_negative) * sigma
        residual = z - design @ coeff
        a, b, c = (float(v) for v in coeff)
        normal_z = 1.0 / math.sqrt(a * a + b * b + 1.0)
        tilt = math.degrees(math.acos(min(normal_z, 1.0)))
        if tilt > max_tilt_deg:
            raise RuntimeError(
                f"substrate fit refused: the recovered plane tilts {tilt:.1f} deg "
                f"off the work frame's up axis (limit {max_tilt_deg:g}) -- the "
                "neighbourhood is a wall, a fixture, or a mis-set work frame, "
                "and measuring against it would be silently wrong")
        inliers = float(np.mean(np.abs(residual) <= 3.0 * max(sigma, 1e-9)))
        return cls(a=a, b=b, c=c, sigma_mm=sigma, inlier_fraction=inliers,
                   clamp_mm=(float(clamp_mm[0]), float(clamp_mm[1])))

    def plane_z(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=float)
        return self.a * xy[..., 0] + self.b * xy[..., 1] + self.c

    def height(self, xyz: np.ndarray) -> np.ndarray:
        pts = np.asarray(xyz, dtype=float)
        return pts[:, 2] - self.plane_z(pts[:, :2])

    def floor_mm(self, k: float) -> float:
        lo, hi = self.clamp_mm
        return float(np.clip(k * self.sigma_mm, lo, hi))

    def tilt_deg(self) -> float:
        return math.degrees(math.acos(
            min(1.0 / math.sqrt(self.a ** 2 + self.b ** 2 + 1.0), 1.0)))

    def to_report(self) -> dict:
        return {"source": self.source,
                "sigma_mm": round(self.sigma_mm, 4),
                "tilt_deg": round(self.tilt_deg(), 3),
                "plane": [round(self.a, 6), round(self.b, 6), round(self.c, 4)],
                "inlier_fraction": round(self.inlier_fraction, 4)}
