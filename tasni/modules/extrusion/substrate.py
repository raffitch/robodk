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


#: P(Z < -3) for a standard normal -- the fraction of points a HEALTHY,
#: uncontaminated fit is expected to show more than 3 sigma below the plane.
#: Closed form (erfc), not a magic number: 0.5 * erfc(3/sqrt(2)).
_P_TAIL_3SIGMA = 0.5 * math.erfc(3.0 / math.sqrt(2.0))

#: Significance level for the breakdown count threshold below -- how often a
#: genuinely healthy fit may be refused by chance. 1e-4 keeps that at roughly
#: 1-in-10,000 while (measured, see task 4 report round 3) comfortably
#: catching every realized-majority-deposit fit tried down to n=50.
_BREAKDOWN_ALPHA = 1e-4

#: Clause (b) in fit(): refuse if sigma_mm exceeds this multiple of
#: clamp_mm[1]. MEASURED, not a round number picked by feel -- do not
#: tighten this without re-running BOTH sweeps that pin it (task 4 report,
#: round 4):
#:   - noise-magnitude sweep (n=20,000, pure noise, no bead): at 2.5x
#:     (5.0 mm) the false-fire cliff sits at noise_mm=5.0, with 0% fire
#:     through noise_mm=4.8 -- real per-take sigma is 0.44-0.61 mm (spec),
#:     so this keeps roughly 8-11x headroom. Round 3 briefly tried 2.0x
#:     (4.0 mm) to close a low-n catch gap in clause (a); that moved the
#:     cliff to noise_mm=4.0, i.e. onto a plausible-if-bad substrate reading
#:     instead of nowhere near one -- reverted (round 4 controller ruling).
#:   - low-n catch sweep for bead_frac 0.55/0.75: 2.5x measurably
#:     under-catches 2.0x's coverage at n=50-80 for the 0.55 case (see the
#:     round 4 report for the actual numbers) -- accepted, because clause
#:     (b) is a BACKSTOP behind clause (a) (the primary, well-validated
#:     detector), and a backstop belongs far from any plausible operating
#:     point so it never argues with real data, not tuned flush against the
#:     detector it backs up.
#: The "just refuse an unusably-high sigma outright" alternative was
#: considered and rejected: refusing whenever k*sigma saturates the clamp
#: fires at sigma > 0.667 mm -- inside normal operation -- so it cannot be
#: the criterion here.
_BREAKDOWN_SIGMA_MULT = 2.5


def _breakdown_below_count(residual: np.ndarray, k: float = 3.0) -> tuple[int, float]:
    """Count of points sitting more than k robust-sigma BELOW the fitted
    plane, and the robust sigma used -- the physical-invariant check (spec
    §3.5-adjacent, task 4 review Important 2): essentially nothing may sit
    far below the substrate, since it is the surface everything rests on. A
    large count below means the fit has locked onto the deposit's top
    instead of the table.

    Deliberately NOT `sigma_mm`/`_sigma_low`: once the fit has actually
    locked onto the deposit (its majority now sits at/above the true
    substrate), `_sigma_low`'s own p15.87 lands inside whichever cluster the
    fit is centred on -- for a hard two-cluster split it can walk all the way
    into the OTHER cluster and re-absorb the very substrate/deposit gap it
    should be flagging. Measured: it reads exactly 0.0 on every breakdown
    case tried while building this guard, healthy or broken alike, so a
    self-referential check is provably blind to this failure. The MAD
    (median absolute deviation from the median, x1.4826 for Gaussian scale)
    keeps its footing because it only needs the MAJORITY cluster's own
    spread, not the gap to the minority one -- robust up to just under 50%
    contamination, which is why this is one signal and not the only one
    (clause (b) in fit() covers the case this one measurably cannot:
    contamination at or just past the exact 50/50 tie, where MAD sits at its
    own breakdown point too).

    Returns a raw COUNT, not a fraction: `fit()` compares it against an
    n-adaptive threshold (`_poisson_upper_count`), not a fixed percentage --
    a fixed 1% cutoff false-fired on healthy small-n frames (task 4 review
    round 3: up to 18% of trials at n=80) because 1% of 50 points is half a
    point, so a single unlucky point trips it.

    If the MAD is degenerate (>=50% of residuals tied exactly at the
    median), falls through to the ordinary (non-robust) standard deviation
    rather than silently reporting a count of 0 -- a safety guard going
    quietly blind in a degenerate case is the wrong default. Only when
    THAT is also zero (every residual identical -- nothing to detect, by
    construction) does this return 0.
    """
    centered = residual - np.median(residual)
    mad_sigma = float(np.median(np.abs(centered))) * 1.4826
    if mad_sigma <= 0.0:
        mad_sigma = float(np.std(residual))
    if mad_sigma <= 0.0:
        return 0, 0.0
    count = int(np.sum(residual < -k * mad_sigma))
    return count, mad_sigma


def _poisson_upper_count(lam: float, alpha: float = _BREAKDOWN_ALPHA) -> int:
    """Smallest integer K such that P(Poisson(lam) > K) <= alpha, by direct
    summation of the Poisson pmf -- pure math/numpy, deterministic, no scipy.

    The count of points a healthy fit shows below -3*sigma is Binomial(n,
    p=_P_TAIL_3SIGMA); Poisson(n*p) is an excellent approximation here
    because p is tiny even when n is large (rare-event regime). This is what
    makes the breakdown-count threshold SCALE with n instead of being a
    fixed fraction or a fixed count -- exactly the fix task 4 review round 3
    asked for over a hard-coded 1% cutoff.
    """
    if lam <= 0.0:
        return 0
    p = math.exp(-lam)
    cdf = p
    k = 0
    while (1.0 - cdf) > alpha:
        k += 1
        p *= lam / k
        cdf += p
        if k > 10_000_000:      # safety valve; never reached for realistic n
            break
    return k


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
    once per distinct pair and cached; `fit()` negates it into
    `bias_correction_sigma` and multiplies by `sigma_mm` to get
    `bias_correction_mm`, which it adds to the intercept -- both are kept on
    the returned instance and surface in `to_report()` so the correction is
    never an invisible constant.
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
    g_lo, g_hi = estimating_eq(lo), estimating_eq(hi)
    if g_lo * g_hi >= 0.0:
        raise RuntimeError(
            f"substrate fit refused: no bias-correction root in [{lo:g}, {hi:g}] "
            f"for c_positive={c_positive:g}, c_negative={c_negative:g} "
            f"(g(lo)={g_lo:.3g}, g(hi)={g_hi:.3g}) -- this (c+, c-) pair falls "
            "outside the range this correction was verified for; use the "
            "spec-default constants or widen the bracket after re-deriving it")
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
    fixed point. The correction's effect on REAL cell data (not just the
    synthetic Gaussian scenario proving it analytically) is checked by the
    task 7 golden harness over 11 archived cell takes -- that is where its
    real-world accuracy claim is verified, not here.

    The correction is NOT unconditionally right, only right across the
    regime it was validated for: it is a net win against the uncorrected fit
    while the substrate is comfortably the majority of the fit region, but
    it multiplies by `sigma_mm`, and `sigma_mm` itself inflates as deposit
    fraction rises -- past a measured ~27-28% deposit fraction in the fit
    region the correction overshoots and makes the recovered plane WORSE
    than leaving it uncorrected, well before the breakdown guard below can
    fire (that guard is tuned to the ~50%+ regime where the fit locks onto
    the deposit outright, not this earlier, quieter degradation). The
    breakdown guard is a backstop for the extreme case, not a substitute for
    keeping the fit region substrate-majority in the first place.
    """
    a: float
    b: float
    c: float
    sigma_mm: float
    inlier_fraction: float
    clamp_mm: tuple[float, float]
    bias_correction_mm: float
    bias_correction_sigma: float
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
        # plane by a known constant times sigma (see _tukey_one_sided_bias).
        # Correcting only the intercept is not an ROI-centring assumption --
        # a vertical translation of z = ax + by + c is exactly `c += delta`
        # for ANY x/y origin; it is simply that the bias this module corrects
        # for is a location (not slope) bias. Keep both the applied mm shift
        # and the dimensionless constant on the instance (spec §4: every
        # frame reports its own health -- a correction that silently moves
        # every measured height must not be invisible to the report).
        bias_correction_sigma = -_tukey_one_sided_bias(c_positive, c_negative)
        bias_correction_mm = bias_correction_sigma * sigma
        coeff = coeff.copy()
        coeff[2] += bias_correction_mm
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
        # Breakdown guard (task 4 review Important 2): past ~50% deposit in
        # the fit region the IRLS above can converge onto the deposit's top
        # instead of the table -- the tilt guard cannot see it (the deposit
        # top is parallel to the substrate) and inlier_fraction, the number
        # an operator would trust, comes out 1.000 in exactly this failure.
        # Two independent checks, because the failure shows up differently
        # depending on how far past the tipping point the contamination is.
        # Clause (a)'s threshold is COUNT-based and n-adaptive (Poisson tail
        # of the ~0.135% a healthy fit is expected to show below -3 sigma),
        # not a fixed fraction: a fixed 1% cutoff false-fired on healthy
        # small-n frames (measured up to 18% of trials at n=80, task 4
        # review round 3) because 1% of 50 points is half a point.
        breakdown_count, _breakdown_sigma = _breakdown_below_count(residual)
        breakdown_threshold = _poisson_upper_count(len(pts) * _P_TAIL_3SIGMA)
        if breakdown_count > breakdown_threshold:
            raise RuntimeError(
                f"substrate fit refused: {breakdown_count} of {len(pts)} points "
                f"({breakdown_count / len(pts):.1%}) sit more than 3 robust-sigma "
                f"below the recovered plane (expected at most {breakdown_threshold} "
                "by chance on a clean fit this size) -- essentially nothing should "
                "sit far below the substrate that everything else rests on, so the "
                "fit has almost certainly locked onto the deposit's top instead of "
                "the table; widen or reposition the fit region so the true "
                "substrate is the majority of it")
        if sigma > _BREAKDOWN_SIGMA_MULT * clamp_mm[1]:
            raise RuntimeError(
                f"substrate fit refused: sigma_mm={sigma:.2f} is more than "
                f"{_BREAKDOWN_SIGMA_MULT:g}x the floor ceiling ({clamp_mm[1]:g} mm) "
                "-- the clamp ceiling exists precisely because k*sigma beyond it "
                "is meaningless, so sigma this far above it means the fit region "
                "is not describing a single substrate at all; widen or "
                "reposition the fit region so the true substrate is the "
                "majority of it")
        inliers = float(np.mean(np.abs(residual) <= 3.0 * max(sigma, 1e-9)))
        return cls(a=a, b=b, c=c, sigma_mm=sigma, inlier_fraction=inliers,
                   clamp_mm=(float(clamp_mm[0]), float(clamp_mm[1])),
                   bias_correction_mm=bias_correction_mm,
                   bias_correction_sigma=bias_correction_sigma)

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
                "inlier_fraction": round(self.inlier_fraction, 4),
                # audit trail for the Fisher-consistency correction applied to
                # the intercept: the actual mm shift, and the dimensionless
                # per-sigma constant it was derived from (bias_correction_mm
                # == bias_correction_sigma * sigma_mm).
                "bias_correction_mm": round(self.bias_correction_mm, 4),
                "bias_correction_sigma": round(self.bias_correction_sigma, 5)}
