"""Substrate reference models for deposit segmentation (design 2026-08-30).

The segmentation question is "how high is this point above the surface the
deposit rests on" -- never "what colour is it" (a free-running auto-exposure
made that an uncalibrated quantity) and never "what is its Z in the work frame"
(the board was measured 1.2 mm below work Z=0 and tilted ~0.5 deg). One
contract answers it for every consumer; the fitted plane is the one provider
that ships. Further providers (a captured empty-plate reference, layer N-1's
measured top) plug into the same interface WHEN evidence demands them --
building them now was measured to be speculative (spec §11).

This module also carries `compactness_filter`, the topology gate that takes
over the colour gate's one real job: distinguishing a deposit (a long
connected curve) from contamination that clears the height floor (a blob) --
by shape, not by an uncalibrated auto-exposed colour (spec §3.5).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

import cv2
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
    inflated by the bead (spec §3.3).

    This CAN return 0.0, on residuals so coarsely quantised that the median and
    p15.87 land on the same lattice level -- depth words as wide as the noise,
    read square-on to a flat surface, put ~68% of the residuals on one value.
    That is deliberately left as 0.0 rather than patched with a fallback
    estimator. Measured 2026-08-31 (task 7): on such a lattice there is no
    scale to recover -- a -2 sigma quantile lands on the same level, and an
    averaged (quantisation-immune) fallback returned 0.002, which is worse than
    zero because a tiny non-zero scale lets the IRLS loop iterate on nonsense
    (the recovered intercept drifted 0.13 mm and inlier_fraction fell to 0.000,
    where breaking out at zero keeps the sound least-squares seed). Zero
    propagates to `floor_mm`, which clamps -- the documented behaviour for a
    pathological fit. The real cell cannot produce this: protocol-2 depth is
    0.1 mm-worded, and even the 1 mm-worded pre-protocol-2 fixtures in
    tests/fixtures/extrusion measure sigma 0.76-0.86 mm because a real surface
    is never exactly square to the camera.
    """
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

#: FLOOR under clause (a)'s count threshold, as a fraction of n. The Poisson
#: tail above answers "how many points would a GAUSSIAN population put more
#: than 3 sigma below the plane" -- and real depth residuals are not Gaussian.
#: They carry a genuine low tail (quantisation lattice, dropout edges, the mat
#: the board rests on), so at large n the Poisson threshold converges on the
#: 0.135% Gaussian tail while healthy real frames sit well above it and every
#: one of them is refused. MEASURED 2026-08-31 on every real frame this repo
#: has -- 11 protocol-2 golden takes (0.1 mm depth words) and the 4 legacy
#: fixtures (1 mm words), all of which the chain measures correctly:
#:     healthy real below-fraction  0.069% .. 1.880%   (worst: layer-002)
#: against the same statistic on a genuine breakdown, the >=50%-deposit fit
#: that locks onto the bead top (synthetic sweep, n=100..300,000):
#:     bead_frac 0.75  20.0% .. 34.0%      bead_frac 0.90  ~10.0%
#: 5% therefore sits 2.66x above the worst healthy real frame, and 4x below the
#: two majority-deposit cases the tests below pin (both bead_frac 0.75). Note
#: that 4x is the margin against those PINNED cases, not the guard's worst case:
#: sweeping bead_frac further, the weakest breakdown that still locks onto the
#: bead reads ~9.9% (bead_frac 0.90), i.e. 1.97x this floor. Still caught, with
#: less room -- so treat 1.97x, not 4x, as the number to protect when touching
#: this. Without this floor the guard refused 6 of
#: the 11 golden takes and all 4 legacy fixtures -- i.e. it would have refused
#: the very archive the design was validated on. Do not lower it without
#: re-running BOTH: the healthy-real sweep above AND the low-n false-fire and
#: majority-deposit tests in tests/test_extrusion_substrate.py.
_BREAKDOWN_MIN_FRACTION = 0.05

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
        # Clause (a)'s threshold is COUNT-based and takes whichever of two
        # bounds is LOOSER, because each fixes the other's failure. The
        # n-adaptive Poisson tail (of the ~0.135% a Gaussian shows below
        # -3 sigma) keeps small n honest: a fixed 1% cutoff false-fired on
        # healthy small-n frames (measured up to 18% of trials at n=80, task 4
        # review round 3) because 1% of 50 points is half a point. The
        # measured fraction floor (_BREAKDOWN_MIN_FRACTION) keeps LARGE n
        # honest: real depth residuals are not Gaussian, so at n ~ 300,000 the
        # Poisson bound lands under every real frame's own low tail and refuses
        # all of them (measured 2026-08-31 -- see that constant).
        breakdown_count, _breakdown_sigma = _breakdown_below_count(residual)
        breakdown_threshold = max(
            _poisson_upper_count(len(pts) * _P_TAIL_3SIGMA),
            math.ceil(_BREAKDOWN_MIN_FRACTION * len(pts)))
        if breakdown_count > breakdown_threshold:
            raise RuntimeError(
                f"substrate fit refused: {breakdown_count} of {len(pts)} points "
                f"({breakdown_count / len(pts):.1%}) sit more than 3 robust-sigma "
                f"below the recovered plane (at most {breakdown_threshold} is "
                "normal for a clean fit this size) -- essentially nothing should "
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
        # `residual` here is POST bias-correction (recomputed at line 330 against
        # the corrected `coeff`), but `sigma` is the PRE-correction robust scale
        # (frozen at line 316, before `coeff[2] += bias_correction_mm`). The +-3
        # sigma window is therefore centred on the corrected plane but sized from
        # the uncorrected residual spread, leaving it asymmetric about the
        # corrected plane by ~bias_correction_sigma (~0.35 sigma). This is
        # deterministic, consistent across takes, and nothing downstream treats
        # inlier_fraction as a guard -- do not "fix" it by swapping in a
        # recomputed post-correction sigma, which would silently shift the
        # baseline for every archived take.
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


def compactness_filter(points, *, mm_per_pixel: float, bead_mm: float,
                       min_length_beads: float, min_points: int,
                       counts: dict | None = None) -> np.ndarray:
    """Drop connected components whose principal-axis extent is shorter than
    ``min_length_beads`` bead widths (spec §3.5).

    A deposit is a curve; contamination that clears the height floor (a speckle
    patch, a fixture corner, the 2026-08-29 checker patch) is compact. Occupancy
    raster at the chain's own pixel size, closed at a QUARTER bead width so the
    closing operation's own bridged-gap reach (a morphological close bridges up
    to roughly TWICE the structuring element's radius, not the radius itself)
    lands at about half a bead width -- sub-bead-scale speckle inside a real
    bead still closes, but two components separated by most of a bead width do
    not weld into one (review round 1: a half-bead RADIUS bridged up to a FULL
    bead width, silently switching the filter's rejection power off exactly at
    the scale of the 2026-08-29 checker patch, which sat 12 mm outside the
    ring). ``math.floor`` (not ``round``) so this holds for every bead/pixel
    recipe, not only ones that happen to round down. 8-connected labels give
    CONNECTIVITY only -- that raster's one job is answering "do these points
    form one blob", nothing more. Per-component extent is then measured on
    that component's own ORIGINAL float millimetre coordinates (looked up via
    each point's own rasterized-pixel label), using ``cv2.minAreaRect``'s
    longer side -- the true oriented (Feret) caliper, not a
    covariance-eigenvector projection (review round 1: for a near-isotropic
    footprint the eigenvectors of a covariance matrix with near-tied
    eigenvalues are numerically unstable, so the "principal axis" the
    projection lands on can swing by up to sqrt(2) depending purely on
    orientation). Do NOT run the caliper on the rasterized pixel coordinates
    (review round 2: doing so re-introduces exactly the grid-quantization the
    caliper swap was meant to escape -- np.rint's rounding is itself
    orientation-sensitive, and a perfect caliper computed on an
    orientation-sensitive input set is still orientation-sensitive; measured
    ~10% of rotations still flipping decision with the raster-coordinate
    caliper, unmoved by the eigenvector-to-minAreaRect swap alone). FAIL-OPEN:
    if the survivors would be fewer than ``min_points`` the cloud passes
    untouched and the bypass is recorded -- topology alone must never starve a
    thin real ring.
    """
    pts = np.asarray(points, dtype=float)
    if counts is None:
        counts = {}
    if not len(pts):
        counts["compactness_components"] = 0
        counts["compactness_kept_components"] = 0
        counts["compactness_bypassed"] = 0
        return pts
    xy = pts[:, :2]
    lo = xy.min(axis=0) - bead_mm
    size = np.ceil((xy.max(axis=0) + bead_mm - lo) / mm_per_pixel).astype(int) + 1
    if np.any(size > 4096):
        raise RuntimeError(f"compactness raster too large: {size[0]}x{size[1]}")
    pixels = np.rint((xy - lo) / mm_per_pixel).astype(int)
    mask = np.zeros((int(size[1]), int(size[0])), np.uint8)
    mask[pixels[:, 1], pixels[:, 0]] = 255
    close_px = max(1, math.floor(bead_mm / (4.0 * mm_per_pixel)))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * close_px + 1, 2 * close_px + 1))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    total, labels = cv2.connectedComponents(closed, connectivity=8)
    min_extent_mm = float(min_length_beads) * float(bead_mm)
    # The raster answers ONLY "do these points form one blob" (that is what
    # morphology/connected-components are for). Measuring extent on the raster
    # itself re-introduces the exact grid-quantization the caliper swap was
    # meant to escape -- np.rint's np.int rounding is orientation-sensitive in
    # a way `cv2.minAreaRect` cannot see past, no matter how good the caliper
    # is (review round 2: still ~10% of rotations flipped, unmoved by round
    # 1's fix, because the INPUT to minAreaRect was still quantized). So the
    # caliper below runs on each component's own ORIGINAL float millimetre
    # coordinates -- looked up via `point_labels`, each original point's
    # rasterized pixel's component label -- not on the pixel grid.
    point_labels = labels[pixels[:, 1], pixels[:, 0]]
    keep_labels = []
    for label in range(1, total):
        comp_xy = xy[point_labels == label].astype(np.float32)
        if len(comp_xy) < 2:
            extent = 0.0
        else:
            _, (w, h), _ = cv2.minAreaRect(comp_xy)
            extent = float(max(w, h))          # already in mm -- no pixel scale
        if extent >= min_extent_mm:
            keep_labels.append(label)
    counts["compactness_components"] = int(total - 1)
    counts["compactness_kept_components"] = len(keep_labels)
    keep = (np.isin(point_labels, keep_labels) if keep_labels
            else np.zeros(len(pts), bool))
    if int(keep.sum()) < int(min_points):
        counts["compactness_bypassed"] = 1
        return pts
    counts["compactness_bypassed"] = 0
    return pts[keep]
