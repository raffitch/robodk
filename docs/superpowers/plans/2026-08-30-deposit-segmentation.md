# Deposit Segmentation Without Colour — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the extrusion chain's chroma gate and constant work-frame floor with a
per-frame fitted-substrate reference (one-sided IRLS plane, derived threshold,
compactness filter), behind one `measure_take` seam that live, reprocess and figures all
share.

**Architecture:** A new `SubstrateModel` contract (`substrate.py`) answers "how high is
this point above the surface the deposit rests on"; `process_observation` consumes it
instead of colour + constant floor. One new entry point (`measure_take`) derives
`assemble_arcs` from the take itself so every path scores a take identically. A golden
harness over the 2026-08-30 archive brackets the whole change.

**Tech Stack:** Python 3.10 (`py -3.10`), numpy, OpenCV, Open3D (existing deps only —
the new estimator is pure numpy), pytest; React/TypeScript for one small webui edit.

**Spec:** `docs/superpowers/specs/2026-08-30-deposit-segmentation-design.md` — argue
from it; §12 lists the review amendments this plan already incorporates.

## Global Constraints

- **NEVER run the full pytest suite** (user memory: too slow, interrupted twice). Use
  `py -3.10 -m pytest tests/<file>.py -k <expr>` on the touched tests plus
  `py -3.10 -c "import tasni.modules.extrusion.processing"` import checks.
- `python` is not on PATH — always `py -3.10`.
- Never round-trip source through PowerShell `Get-Content`/`Set-Content` (mojibake trap);
  use the Read/Edit/Write tools or Git Bash.
- Commit after every task; **push when the plan completes** (working agreement: the user
  reviews from pushed history).
- The golden archive `runs/extrusion/20260830-202416-293b208d/` is git-ignored and
  exists only on the cell machine — golden tests must `skipif` on its absence and must
  **never write into `runs/`**.
- Determinism is a spec requirement (§3.2): no randomness anywhere in the new front end;
  a repeated fit must be bit-identical.
- Layer-2 golden takes are **expected to remain invalid** (spec §2.4). A change that
  makes them valid is a defect, not a fix.
- The frontend builds with `npm run build` in `tasni/webui` (only Task 6 touches it).
- Line endings: the repo is LF; don't let a tool rewrite whole files to CRLF.

---

### Task 1: The `measure_take` seam

Every scoring path must call one entry point that derives `assemble_arcs` from the
take's own layer index, ending the live/reprocess/figure divergence (spec §3.7).
Behaviour changes to be aware of (deliberate, spec §3.7): layer-1 reprocess, layer-1
take figures and the live-print layer-1 inspection flip from `assemble_arcs=False` to
`True`, matching the live measure path.

**Files:**
- Modify: `tasni/modules/extrusion/processing.py` (add `measure_take` after
  `process_observation`, ~line 908; reroute the internal call at `:1005`)
- Modify: `tasni/modules/extrusion/measure.py:863-873`
- Modify: `tasni/modules/extrusion/service.py:1051-1054` and `:1197-1213`
- Modify: `tasni/modules/extrusion/figures.py:829-837` and `:901-912`
- Test: `tests/test_extrusion_processing.py` (append)

**Interfaces:**
- Produces (transitional signature — Tasks 6/7 shrink it):
  ```python
  def measure_take(*, color, depth, geometry, T_work_camera, K, dist,
                   plan, layer, config, floor_profile=None,
                   stages=None) -> ProcessingResult
  ```
  Later tasks rely on: `measure_take` is THE seam; `process_observation` is called
  nowhere else outside `processing.py`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_extrusion_processing.py`):

```python
# --------------------------------------------------- one seam for every caller (§3.7)

def test_measure_take_derives_arc_assembly_from_the_layer_itself(monkeypatch):
    """assemble_arcs is a property of the take (isolated first layer), not of the
    caller -- the live/reprocess/figure divergence was the defect."""
    from tasni.modules.extrusion import processing
    seen = []
    monkeypatch.setattr(processing, "process_observation",
                        lambda **kw: seen.append(kw) or "sentinel")
    from tasni.core.config import ExtrusionConfig
    from tasni.modules.extrusion.models import CylinderRecipe, CylinderSetup
    from tasni.modules.extrusion.toolpath import generate_cylinder_plan
    plan = generate_cylinder_plan(
        CylinderRecipe(radius_mm=40.0, layer_count=2, layer_height_mm=5.0,
                       bead_diameter_mm=9.0, robot_speed_mm_s=75.0,
                       extrusion_rate_pct=0.0, points_per_circle=180),
        CylinderSetup(print_tool="LongCalibTool", work_frame="Tasni Work Frame",
                      inspection_tool="Realsense", inspection_auto=True,
                      center_x_mm=0.0, center_y_mm=0.0))
    # (construction idiom copied from scene_plan() in tests/test_extrusion_measure.py:73)
    common = dict(color=None, depth=None, geometry=None, T_work_camera=None,
                  K=None, dist=None, plan=plan, config=ExtrusionConfig())
    assert processing.measure_take(layer=plan.layers[0], **common) == "sentinel"
    assert seen[-1]["assemble_arcs"] is True
    processing.measure_take(layer=plan.layers[1], **common)
    assert seen[-1]["assemble_arcs"] is False


def test_no_caller_bypasses_the_seam():
    """Grep guard: outside processing.py, nothing in the extrusion module may call
    process_observation directly (spec §3.7)."""
    from pathlib import Path
    import tasni.modules.extrusion as ext
    root = Path(ext.__file__).parent
    offenders = [p.name for p in root.glob("*.py")
                 if p.name != "processing.py"
                 and "process_observation(" in p.read_text(encoding="utf-8")]
    assert not offenders, f"route these through measure_take: {offenders}"
```

If `CylinderRecipe`/`CylinderSetup` reject those minimal kwargs, copy the construction
idiom from the top of `tests/test_extrusion_measure.py` (its helpers build the same
models) — do not weaken the assertions.

- [ ] **Step 2: Run them to make sure they fail**

Run: `py -3.10 -m pytest tests/test_extrusion_processing.py -k "measure_take or bypasses" -v`
Expected: FAIL — `measure_take` does not exist; the grep guard finds 4 offenders
(`measure.py`, `service.py`, `figures.py` ×2 counts as one file).

- [ ] **Step 3: Implement `measure_take`** in `processing.py`, directly after
`process_observation`:

```python
def measure_take(*, color, depth, geometry, T_work_camera, K, dist,
                 plan, layer, config, floor_profile=None,
                 stages=None) -> ProcessingResult:
    """THE entry point for scoring one RGB-D take -- live, reprocess and figures.

    ``assemble_arcs`` is derived here, from the take itself: layer 1 is an
    isolated ring (no lower layer for assembly to fuse into), every higher layer
    keeps the deliberately strict no-assembly path. Callers choosing it
    independently is how the same archived take scored differently live, on the
    reprocess button and in its method figure (2026-08-30 handoff §6).
    """
    return process_observation(
        color=color, depth=depth, geometry=geometry, T_work_camera=T_work_camera,
        K=K, dist=dist, plan=plan, layer=layer, config=config,
        floor_profile=floor_profile, stages=stages,
        assemble_arcs=int(layer.layer_index) == 1)
```

If `LayerPath` has no `layer_index` attribute (verify: `service.py:1022` uses
`layer.layer_index`, so it does), stop and re-read the model — do not guess.

- [ ] **Step 4: Reroute the five external call sites + the internal one.** In each,
replace `process_observation(` with `measure_take(` and delete the caller's
`assemble_arcs=...` argument (and its justifying comment where it becomes wrong):
  - `measure.py:866-873` — keep `floor_profile=floor`; delete the
    `assemble_arcs=self.layer_index == 1` line and its comment (the seam owns it now);
    update the import at the top of the file.
  - `service.py:1051-1054` (live-print inspection) and `service.py:1197-1213`
    (reprocess) — imports likewise.
  - `figures.py:830-837` (take figure) and `figures.py:903-912` (characterization
    figure — drop its explicit `assemble_arcs=True`; `plan.layers[0]` derives it).
  - `processing.py:1005-1008` (`characterize_ring`'s refined pass — drop
    `assemble_arcs=True`).

- [ ] **Step 5: Run the new tests plus the neighbours the reroute touches**

Run: `py -3.10 -m pytest tests/test_extrusion_processing.py -k "measure_take or bypasses" -v`
then `py -3.10 -m pytest tests/test_extrusion_measure.py tests/test_extrusion_figures.py tests/test_extrusion_job.py -q`
Expected: new tests PASS. If an existing test pinned `assemble_arcs=False` for a
layer-1 reprocess/figure path, it is pinning the defect this task removes — update that
assertion to `True` and say so in its docstring; anything else that breaks is a real
regression to fix before proceeding.

- [ ] **Step 6: Commit**

```bash
git add tasni/modules/extrusion tests/test_extrusion_processing.py tests/test_extrusion_measure.py tests/test_extrusion_figures.py tests/test_extrusion_job.py
git commit -m "refactor(extrusion): one measure_take seam; assemble_arcs derived from the take"
```

---

### Task 2: Golden harness over the 2026-08-30 archive

Lands now, on the OLD chain, so the front-end swap (Task 7) is judged against a
baseline this same harness produced (spec §5). Read-only on `runs/`.

**Files:**
- Modify: `tasni/modules/extrusion/figures.py` — extract the take-input
  reconstruction out of `_compute_stages` (`:805-837`) into a public helper
- Create: `tests/test_extrusion_golden.py`

**Interfaces:**
- Produces: `figures.reconstruct_take_inputs(take: TakeData) -> dict | None` returning
  `{"plan": CylinderPlan, "layer": LayerPath, "config": ExtrusionConfig,
  "dist": np.ndarray | None}` — exactly the inputs `_compute_stages` feeds the seam
  (config payload resolution from manifest-then-trial provenance, `plan_for_archived_take`,
  the `_chroma_dist` choice). `_compute_stages` must call it, so the golden tests and
  the figure can never drift apart.

- [ ] **Step 1: Extract the helper.** Move `figures.py:809-837`'s reconstruction (trial
load, `config_payload` resolution, `plan_for_archived_take`, layer selection, the
dist choice) into `reconstruct_take_inputs(take)`, returning `None` where the current
code returns `None` (no trial file, no config payload, index out of range). Keep
`_compute_stages` behaviour identical: it calls the helper, then `measure_take` with
the helper's outputs. The characterization-figure path (`:846+`) is separate — leave it.

- [ ] **Step 2: Run the figure tests to prove the extraction changed nothing**

Run: `py -3.10 -m pytest tests/test_extrusion_figures.py -q`
Expected: PASS, same count as Task 1 Step 5.

- [ ] **Step 3: Write the golden tests** (`tests/test_extrusion_golden.py`):

```python
"""Golden reprocess of the 2026-08-30 cell archive (spec §5).

Read-only on runs/. Skips on machines without the archive (runs/ is
git-ignored). Layer-2 takes are EXPECTED INVALID -- a change that makes them
valid is the false positive this file exists to catch (spec §2.4).
"""
import json
from pathlib import Path

import numpy as np
import pytest

ARCHIVE = (Path(__file__).resolve().parents[1]
           / "runs" / "extrusion" / "20260830-202416-293b208d")
pytestmark = pytest.mark.skipif(
    not ARCHIVE.is_dir(),
    reason="golden archive not on this machine (runs/ is git-ignored)")

LAYER1 = ["layer-001"] + [f"layer-001-take{i:02d}" for i in range(2, 9)]
LAYER2 = ["layer-002", "layer-002-take02", "layer-002-take03"]


def _measure(name):
    import cv2
    from tasni.modules.extrusion import figures, processing
    take = figures.load_take(ARCHIVE / name)
    inputs = figures.reconstruct_take_inputs(take)
    assert inputs is not None, f"{name}: archive lacks reprocess provenance"
    color = cv2.imread(str(ARCHIVE / name / "color.png"), cv2.IMREAD_COLOR)
    return processing.measure_take(
        color=color, depth=take.depth, geometry=take.geometry,
        T_work_camera=take.T_work_camera, K=take.K, dist=inputs["dist"],
        plan=inputs["plan"], layer=inputs["layer"], config=inputs["config"])


def test_layer1_acceptance_holds():
    radii = []
    for name in LAYER1:
        result = _measure(name)
        assert result.metrics.valid, name
        assert result.metrics.path_completeness >= 0.990, (
            name, result.metrics.path_completeness)
        radii.append(result.metrics.measured_radius_mm)
    assert abs(float(np.mean(radii)) - 41.0) <= 0.10, radii
    assert float(np.std(radii, ddof=1)) <= 0.15, radii   # spec §2.1: σ stays measured


def test_layer2_stays_invalid_and_completeness_stays_honest():
    for name in LAYER2:
        archived = json.loads(
            (ARCHIVE / name / "report.json").read_text(encoding="utf-8"))
        result = _measure(name)
        assert not result.metrics.valid, (
            f"{name}: a 'fixed' layer-2 take is the false positive spec §2.4 pins")
        assert abs(result.metrics.path_completeness
                   - float(archived["metrics"]["path_completeness"])) <= 0.05, name
```

If a `_measure` call raises (an invalid take may raise rather than return — the
live path archives the error), catch `RuntimeError` in the layer-2 test and treat it
as invalid with completeness taken from the archived report only if the archived
report itself recorded an error; otherwise let it fail. Check what the three archived
layer-2 `report.json` files actually contain (`valid: false` with metrics, per the
handoff) and match the assertion to that reality — the numbers above are from spec
§2 and the 2026-08-30 handoff §1.

- [ ] **Step 4: Run the golden tests on the cell machine's archive**

Run: `py -3.10 -m pytest tests/test_extrusion_golden.py -v`
Expected: PASS (this is the old chain scoring its own archive; layer-1 acceptance was
measured to hold in spec §2.1 — old-chain radius σ is 0.056). If layer-2 raises
instead of returning invalid, adjust per Step 3's note and re-run.

- [ ] **Step 5: Commit**

```bash
git add tasni/modules/extrusion/figures.py tests/test_extrusion_golden.py
git commit -m "test(extrusion): golden harness over the 2026-08-30 archive, old-chain baseline"
```

---

### Task 3: `ExtrusionConfig.from_archive` — retire keys without breaking archives

`_Model` is `extra="forbid"` (`config.py:41`) and three call sites re-validate the
ARCHIVED `processing_config` payload (`service.py:1213`, `figures.py` — now inside
`reconstruct_take_inputs` — and `figures.py:874`). Without this shim, Tasks 6–7's field
deletions make every existing archive unreprocessable (spec §3.6).

**Files:**
- Modify: `tasni/core/config.py` (near `ExtrusionConfig`, `:713`)
- Modify: `tasni/modules/extrusion/service.py:1213`,
  `tasni/modules/extrusion/figures.py` (the two `ExtrusionConfig.model_validate(config_payload)` sites)
- Test: `tests/test_extrusion_processing.py` (append)

**Interfaces:**
- Produces: `RETIRED_EXTRUSION_CONFIG_KEYS: frozenset[str]` (module level, starts
  empty) and `ExtrusionConfig.from_archive(payload: dict) -> ExtrusionConfig`.
  Tasks 6 and 7 add the retired names to the frozenset.

- [ ] **Step 1: Write the failing test:**

```python
def test_archived_configs_with_retired_keys_still_validate():
    """extra='forbid' + archived processing_config payloads means a field can
    never be deleted without this shim (spec §3.6): retired keys are DROPPED,
    never reinterpreted, and unknown keys still fail loudly."""
    import pytest
    from tasni.core import config as cfg
    payload = cfg.ExtrusionConfig().model_dump()
    payload["deposit_min_saturation"] = 60          # will be retired by Task 7
    with_retired = dict(payload, **{k: 1 for k in cfg.RETIRED_EXTRUSION_CONFIG_KEYS})
    assert cfg.ExtrusionConfig.from_archive(with_retired) is not None
    with pytest.raises(Exception):
        cfg.ExtrusionConfig.from_archive(dict(payload, not_a_field_ever=1))
```

(While the frozenset is empty and `deposit_min_saturation` is still a live field the
first assert exercises the pass-through; the test's value compounds as keys retire.)

- [ ] **Step 2: Run it to make sure it fails**

Run: `py -3.10 -m pytest tests/test_extrusion_processing.py -k retired -v`
Expected: FAIL — no `RETIRED_EXTRUSION_CONFIG_KEYS`.

- [ ] **Step 3: Implement** in `core/config.py`, above `ExtrusionConfig`:

```python
# Config fields deleted from ExtrusionConfig but still present in archived
# processing_config payloads (every archive dumps the full config per take).
# from_archive() drops them so extra="forbid" keeps refusing genuinely unknown
# keys without refusing history. Grown by the 2026-08-30 substrate change.
RETIRED_EXTRUSION_CONFIG_KEYS: frozenset = frozenset()
```

and on `ExtrusionConfig`:

```python
    @classmethod
    def from_archive(cls, payload: dict) -> "ExtrusionConfig":
        """Validate an ARCHIVED processing_config: retired keys are dropped
        (never reinterpreted); anything else unknown still fails loudly."""
        return cls.model_validate({k: v for k, v in dict(payload).items()
                                   if k not in RETIRED_EXTRUSION_CONFIG_KEYS})
```

- [ ] **Step 4: Switch the archived-payload call sites** — `service.py:1213` and both
figure sites (`reconstruct_take_inputs`, `figures.py:874`) from
`ExtrusionConfig.model_validate(...)` to `ExtrusionConfig.from_archive(...)`. Live
config construction stays `model_validate` — only archive payloads go through the shim.

- [ ] **Step 5: Run**

Run: `py -3.10 -m pytest tests/test_extrusion_processing.py -k retired -v && py -3.10 -m pytest tests/test_extrusion_golden.py tests/test_extrusion_figures.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tasni/core/config.py tasni/modules/extrusion/service.py tasni/modules/extrusion/figures.py tests/test_extrusion_processing.py
git commit -m "feat(config): from_archive strips retired keys so field retirement cannot orphan archives"
```

---

### Task 4: `PlaneSubstrate` — deterministic one-sided IRLS (spec §3.2–§3.4)

**Files:**
- Create: `tasni/modules/extrusion/substrate.py`
- Test: create `tests/test_extrusion_substrate.py`

**Interfaces:**
- Produces:
  ```python
  class SubstrateModel(Protocol):
      source: str
      sigma_mm: float
      def height(self, xyz: np.ndarray) -> np.ndarray: ...
      def floor_mm(self, k: float) -> float: ...
      def to_report(self) -> dict: ...

  PlaneSubstrate.fit(xyz, *, clamp_mm=(1.0, 2.0), max_tilt_deg=25.0) -> PlaneSubstrate
  ```
  Task 7 consumes `fit`, `height`, `floor_mm`, `to_report`, `sigma_mm`, `source`.

- [ ] **Step 1: Write the failing tests** (`tests/test_extrusion_substrate.py`):

```python
"""PlaneSubstrate: the fitted-substrate reference (spec §3.2-§3.4).

Synthetic surfaces with a known bead: the estimator must recover the plane the
bead sits on, one-sidedly (deposit only ever contaminates from above), score its
own noise from the uncontaminated lower half, and do it all bit-identically."""
import math

import numpy as np
import pytest

from tasni.modules.extrusion.substrate import PlaneSubstrate


def _tilted_scene(*, tilt_deg=0.55, noise_mm=0.5, bead_height_mm=8.0,
                  bead_fraction=0.15, n=20_000, seed=7):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-120.0, 120.0, size=(n, 2))
    slope = math.tan(math.radians(tilt_deg))
    z = slope * xy[:, 0] - 1.2 + rng.normal(0.0, noise_mm, n)
    bead = rng.random(n) < bead_fraction
    z[bead] += bead_height_mm            # one-sided contamination, like a real bead
    return np.column_stack([xy, z]), slope


def test_recovers_the_plane_under_one_sided_contamination():
    pts, slope = _tilted_scene()
    fit = PlaneSubstrate.fit(pts)
    assert fit.a == pytest.approx(slope, abs=0.002)      # tilt recovered
    assert fit.c == pytest.approx(-1.2, abs=0.15)        # offset recovered
    assert fit.sigma_mm == pytest.approx(0.5, rel=0.25)  # sigma from the clean half
    clean = ~(pts[:, 2] - (fit.a * pts[:, 0] + fit.b * pts[:, 1] + fit.c) > 4.0)
    heights = fit.height(pts[clean])
    assert abs(float(np.median(heights))) < 0.1          # substrate sits at height 0


def test_fit_is_bit_identical_across_repeats():
    """The RANSAC failure mode, pinned (spec §3.2): a chain that cannot
    reprocess a frame to the same number twice is not a measurement chain."""
    pts, _ = _tilted_scene()
    first, second = PlaneSubstrate.fit(pts), PlaneSubstrate.fit(pts)
    assert (first.a, first.b, first.c, first.sigma_mm) \
        == (second.a, second.b, second.c, second.sigma_mm)


def test_derived_floor_is_clamped_both_ways():
    pts, _ = _tilted_scene(noise_mm=0.55)
    fit = PlaneSubstrate.fit(pts)
    assert 1.0 <= fit.floor_mm(3.0) <= 2.0
    assert fit.floor_mm(100.0) == 2.0        # ceiling: the k=4 cliff (spec §3.4)
    assert fit.floor_mm(0.0) == 1.0          # floor: never open to raw noise


def test_a_wall_is_refused_not_measured():
    rng = np.random.default_rng(3)
    n = 5000
    x = rng.uniform(-50, 50, n)
    z = rng.uniform(0, 100, n)
    pts = np.column_stack([x, 0.8 * z + rng.normal(0, 0.3, n), z])   # steep surface
    with pytest.raises(RuntimeError, match="substrate fit refused"):
        PlaneSubstrate.fit(pts)


def test_too_few_points_is_a_loud_error():
    with pytest.raises(RuntimeError, match="substrate fit needs"):
        PlaneSubstrate.fit(np.zeros((10, 3)))
```

- [ ] **Step 2: Run them to make sure they fail**

Run: `py -3.10 -m pytest tests/test_extrusion_substrate.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement `substrate.py`:**

```python
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


@dataclass(frozen=True)
class PlaneSubstrate:
    """z = a*x + b*y + c, fitted by deterministic one-sided IRLS.

    Positive residuals are down-weighted harder than negative ones (Tukey
    c+ = 2.0 vs c- = 4.685) because the deposit is the only thing that can sit
    ABOVE the surface. RANSAC was measured and rejected: Open3D 0.17 ignores
    its seed (a different plane per run) and one-sided IRLS beat it 0.064 mm
    to 0.191 mm mean |error| against ground truth (spec §3.2).
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
```

- [ ] **Step 4: Run the tests**

Run: `py -3.10 -m pytest tests/test_extrusion_substrate.py -v`
Expected: PASS, all five.

- [ ] **Step 5: Commit**

```bash
git add tasni/modules/extrusion/substrate.py tests/test_extrusion_substrate.py
git commit -m "feat(extrusion): PlaneSubstrate -- deterministic one-sided IRLS substrate reference"
```

---

### Task 5: `compactness_filter` — topology takes over the gate's safety role (spec §3.5)

**Files:**
- Modify: `tasni/modules/extrusion/substrate.py` (append)
- Test: `tests/test_extrusion_substrate.py` (append)

**Interfaces:**
- Produces:
  ```python
  def compactness_filter(points, *, mm_per_pixel: float, bead_mm: float,
                         min_length_beads: float, min_points: int,
                         counts: dict | None = None) -> np.ndarray
  ```
  Task 7 calls it between the height/radial ROI and `_filter_deposit`, with
  `mm_per_pixel=config.raster_mm_per_pixel`, `bead_mm=recipe.bead_diameter_mm`,
  `min_length_beads=config.deposit_min_length_beads`,
  `min_points=config.cluster_min_points`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_extrusion_substrate.py`):

```python
# ------------------------------------------ compactness: the gate's one real job

from tasni.modules.extrusion.substrate import compactness_filter


def _arc_points(radius_mm=41.0, span_deg=180.0, per_deg=6):
    angles = np.radians(np.linspace(0.0, span_deg, int(span_deg * per_deg)))
    return np.column_stack([radius_mm * np.cos(angles),
                            radius_mm * np.sin(angles),
                            np.full(len(angles), 3.0)])


def _patch_points(n, center=(60.0, 0.0)):
    rng = np.random.default_rng(11)
    xy = rng.uniform(-4.0, 4.0, size=(n, 2)) + np.asarray(center)
    return np.column_stack([xy, np.full(n, 3.0)])


def test_rejects_the_compact_patch_and_keeps_an_arc_of_equal_count():
    """The 22-point checker patch that exhausted the branch guard was COMPACT,
    not colourful -- an arc of the same pixel count is long and survives."""
    arc = _arc_points()
    patch = _patch_points(len(arc), center=(80.0, 0.0))
    counts = {}
    kept = compactness_filter(np.vstack([arc, patch]), mm_per_pixel=1.0,
                              bead_mm=9.0, min_length_beads=3.0,
                              min_points=10, counts=counts)
    assert counts["compactness_components"] == 2
    assert counts["compactness_kept_components"] == 1
    assert len(kept) == len(arc)
    assert np.allclose(np.sort(kept[:, 0]), np.sort(arc[:, 0]))


def test_fail_open_when_the_filter_would_starve_the_chain():
    """A thin or fragmented real ring must never be zeroed by topology alone:
    below min_points the cloud passes through untouched, recorded as bypassed."""
    patch = _patch_points(40)
    counts = {}
    kept = compactness_filter(patch, mm_per_pixel=1.0, bead_mm=9.0,
                              min_length_beads=3.0, min_points=10, counts=counts)
    assert counts["compactness_bypassed"] == 1
    assert len(kept) == len(patch)
```

- [ ] **Step 2: Run to make sure they fail**

Run: `py -3.10 -m pytest tests/test_extrusion_substrate.py -k compact -v`
Expected: FAIL — `compactness_filter` does not exist.

- [ ] **Step 3: Implement** (append to `substrate.py`; add `import cv2` at top):

```python
def compactness_filter(points, *, mm_per_pixel: float, bead_mm: float,
                       min_length_beads: float, min_points: int,
                       counts: dict | None = None) -> np.ndarray:
    """Drop connected components whose principal-axis extent is shorter than
    ``min_length_beads`` bead widths (spec §3.5).

    A deposit is a curve; contamination that clears the height floor (a speckle
    patch, a fixture corner, the 2026-08-29 checker patch) is compact. Occupancy
    raster at the chain's own pixel size, closed at half a bead width so one
    bead cannot self-fragment, 8-connected labels, per-component extent along
    the largest covariance eigenvector. FAIL-OPEN: if the survivors would be
    fewer than ``min_points`` the cloud passes untouched and the bypass is
    recorded -- topology alone must never starve a thin real ring.
    """
    pts = np.asarray(points, dtype=float)
    if counts is None:
        counts = {}
    if not len(pts):
        return pts
    xy = pts[:, :2]
    lo = xy.min(axis=0) - bead_mm
    size = np.ceil((xy.max(axis=0) + bead_mm - lo) / mm_per_pixel).astype(int) + 1
    if np.any(size > 4096):
        raise RuntimeError(f"compactness raster too large: {size[0]}x{size[1]}")
    pixels = np.rint((xy - lo) / mm_per_pixel).astype(int)
    mask = np.zeros((int(size[1]), int(size[0])), np.uint8)
    mask[pixels[:, 1], pixels[:, 0]] = 255
    close_px = max(1, int(round(bead_mm / (2.0 * mm_per_pixel))))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * close_px + 1, 2 * close_px + 1))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    total, labels = cv2.connectedComponents(closed, connectivity=8)
    min_extent_mm = float(min_length_beads) * float(bead_mm)
    keep_labels = []
    for label in range(1, total):
        ys, xs = np.nonzero(labels == label)
        coords = np.column_stack([xs, ys]).astype(float)
        coords -= coords.mean(axis=0)
        if len(coords) < 2:
            extent = 0.0
        else:
            _, vectors = np.linalg.eigh(np.cov(coords.T))
            projected = coords @ vectors[:, -1]
            extent = float(projected.max() - projected.min()) * mm_per_pixel
        if extent >= min_extent_mm:
            keep_labels.append(label)
    counts["compactness_components"] = int(total - 1)
    counts["compactness_kept_components"] = len(keep_labels)
    point_labels = labels[pixels[:, 1], pixels[:, 0]]
    keep = (np.isin(point_labels, keep_labels) if keep_labels
            else np.zeros(len(pts), bool))
    if int(keep.sum()) < int(min_points):
        counts["compactness_bypassed"] = 1
        return pts
    counts["compactness_bypassed"] = 0
    return pts[keep]
```

- [ ] **Step 4: Run the tests**

Run: `py -3.10 -m pytest tests/test_extrusion_substrate.py -v`
Expected: PASS, all seven.

- [ ] **Step 5: Commit**

```bash
git add tasni/modules/extrusion/substrate.py tests/test_extrusion_substrate.py
git commit -m "feat(extrusion): compactness filter -- the chroma gate's safety role, in geometry"
```

---

### Task 6: Delete `floor_profile` end-to-end (spec §3.6, §2.4)

Previous-layer referencing was measured to make the only stacked data WORSE
(completeness 0.62 → 0.50, spec §2.4); the parameter, its session plumbing, the 409
guard and the web UI's copy of the same gate all go. `Session.tops` recording STAYS
(the UI renders it; a future provider would be built from it).

**Files:**
- Modify: `tasni/modules/extrusion/processing.py` — `process_observation`
  (param `:669`, docstring `:681-686`, branch `:750-758`, the `ring_geometry` call
  `:877`), `ring_geometry` (`:330-356`), `measure_take` (drop the passthrough)
- Modify: `tasni/modules/extrusion/measure.py` — delete `Session.floor_profile`
  (`:134-140`) and the lookup at `:863`
- Modify: `tasni/modules/extrusion/module.py` — delete `allow_missing_floor` (`:93`)
  and the 409 guard (`:768-776`)
- Modify: `tasni/core/config.py` — retire `layer_floor_margin_mm` (`:868`): delete the
  field, add `"layer_floor_margin_mm"` to `RETIRED_EXTRUSION_CONFIG_KEYS`
- Modify: `tasni/modules/extrusion/paper_docx.py:198` — drop the
  `layer_floor_margin_mm` clause from the methods text (keep `.get` style so legacy
  archives still render)
- Modify: `tasni/webui/src/pages/Extrusion.tsx` — `floorReady` (`:997-998`) and
  `stepFloorReady` (`:1214-1217`)
- Test: `tests/test_extrusion_measure.py` (`observe` helper `:83-91`, tests at `:609`
  and `:1045`)

**Interfaces:**
- Consumes: nothing new. Produces: `process_observation`/`measure_take` without
  `floor_profile`; `ring_geometry(measured_xyz, cluster_xyz, center_xy, *,
  substrate=None, build_plane_z_mm=0.0, bins=36)` — the `substrate` slot is dormant
  until Task 7 fills it (this task always passes `None`, keeping today's
  `build_plane` reference).

- [ ] **Step 1: Delete in the backend.** Remove the `floor_profile` parameter and
branch from `process_observation` (`:669`, `:750-758` — `floor` stays
`{"source": "build_plane", ...}` unconditionally) and from `measure_take`; change
`ring_geometry`'s keyword from `floor_profile` to `substrate` (behaviour when `None`
identical; the non-None branch becomes `height = substrate.height(measured)` with
`reference_name = substrate.source` — dormant until Task 7). Delete
`Session.floor_profile` and `measure.py:863`'s lookup. Delete the `module.py` guard
and body field — the docstringed reason ("a stacked ring blends into the ring beneath
it") is superseded by spec §2.4's measurement, and say so in the commit message.
Retire `layer_floor_margin_mm` per the Files list. Trim `paper_docx.py:198`.

- [ ] **Step 2: Update the tests that pinned the old behaviour.**
  - `observe()` (`tests/test_extrusion_measure.py:83-91`): drop the `floor_profile`
    kwarg plumbing.
  - `test_floor_from_previous_layer_keeps_the_ring_below_out_of_the_measurement`
    (`:609`): delete it; in its place add a comment-anchor test asserting
    `process_observation` no longer accepts the kwarg:

```python
def test_the_previous_layer_floor_is_gone_by_design():
    """Spec §2.4: previous-layer referencing made the only stacked data WORSE
    (0.62 -> 0.50). Layers are measured against the substrate; the ROI ceiling
    accommodates the stack. Its return needs new evidence, not a revert."""
    import inspect
    from tasni.modules.extrusion.processing import process_observation
    assert "floor_profile" not in inspect.signature(process_observation).parameters
```

  - `test_repeat_takes_and_the_floor_from_the_previous_layer` (`:1045`): rename to
    `test_repeat_takes_share_one_session`, delete its floor assertions (`:1054`,
    `:1058`), keep the repeat-take assertions.
  - Any other `-k floor` hits in `tests/test_extrusion_measure.py` /
    `tests/test_extrusion.py`: update the same way — deletions must be justified by
    spec §2.4 in the diff, not silent.

- [ ] **Step 3: Update the web UI.** In `Extrusion.tsx` replace both computations with
constants and delete what depended on them being false:

```tsx
  // Layers are measured against the fitted substrate (2026-08-30 design);
  // measuring layer N no longer requires layer N-1's measured top.
  const floorReady = true;
```

(and `stepFloorReady = true;` at `:1214-1217`, keeping its comment removed). Then chase
both identifiers through the file: any disabled-state, tooltip or copy that told the
operator to measure the previous layer first goes too. Keep the `tops` **rendering**
(`:1348`, `:120`) untouched. If after this `floorReady`/`stepFloorReady` are literal
`true` everywhere they're used, inline and delete the variables.

- [ ] **Step 4: Verify**

Run: `py -3.10 -m pytest tests/test_extrusion_measure.py tests/test_extrusion.py tests/test_extrusion_processing.py -q`
then `cd tasni/webui && npm run build` (then back to repo root).
Expected: PASS / clean build. Also run
`py -3.10 -m pytest tests/test_extrusion_golden.py -q` — the golden numbers must not
move (the archived layer-2 reports were produced WITHOUT a floor; layer-1 never had
one).

- [ ] **Step 5: Commit**

```bash
git add tasni/modules/extrusion tasni/core/config.py tasni/webui/src/pages/Extrusion.tsx tests/test_extrusion_measure.py tests/test_extrusion.py tests/test_extrusion_processing.py
git commit -m "feat(extrusion): delete floor_profile end-to-end -- previous-layer referencing measured worse (spec 2.4)"
```

---

### Task 7: The swap — geometry replaces colour (spec §3.1–§3.6, §4)

The chroma gate, `deposit_floor_mm` and the `ColorRegistered` front end leave the
extrusion chain; the substrate + derived floor + compactness filter take over, in
`process_observation` AND `characterize_ring` together (characterization defines the
recipe the layers are judged against — the two must see the same cloud). The colour
frame keeps being captured and archived; it takes no part in any decision.

**Files:**
- Modify: `tasni/modules/extrusion/processing.py` — front ends of
  `process_observation` (`:706-758`) and `characterize_ring` (`:952-980`); delete
  `chroma_gate_mask` (`:64-127`) and `deposit_floor_mm` (`:48-61`); drop
  `color`/`K`/`dist` from both signatures and from `measure_take`; pass the substrate
  into `ring_geometry` at `:877`; add `report["substrate"]`
- Modify: `tasni/core/config.py` — add the four new fields; retire the five colour/floor
  fields into `RETIRED_EXTRUSION_CONFIG_KEYS`
- Modify: callers of `measure_take` — `measure.py`, `service.py` (×2), `figures.py`
  (×2): stop passing `color`/`K`/`dist`; in `figures.py` delete `_chroma_dist`
  (`:493`) and the `dist` slot of `reconstruct_take_inputs`
- Modify: `tasni/modules/extrusion/paper_docx.py:196-197` — methods text describes the
  substrate floor, not the constant/gate
- Modify: `tests/fixtures/extrusion/ring1/README.md`, `ring2/README.md` — re-scope
  from "gate abstains" to what the fixture now proves (pure-geometry segmentation)
- Test: `tests/test_extrusion_processing.py`, `tests/test_extrusion_measure.py` —
  replace the ~9 gate tests; `tests/test_extrusion_golden.py` reruns unchanged
- Test: `tests/test_extrusion_figures.py:800` — delete
  `test_the_method_figure_gates_through_the_lens_model_the_measurement_used` (its
  subject no longer exists)

**Interfaces:**
- Consumes: `PlaneSubstrate.fit`, `compactness_filter` (Tasks 4–5),
  `RETIRED_EXTRUSION_CONFIG_KEYS` (Task 3).
- Produces (final signatures):
  ```python
  def measure_take(*, depth, geometry, T_work_camera, plan, layer, config,
                   stages=None) -> ProcessingResult
  def process_observation(*, depth, geometry, T_work_camera, plan, layer, config,
                          stages=None, assemble_arcs=False) -> ProcessingResult
  def characterize_ring(*, depth, geometry, T_work_camera, search_center_mm,
                        work_frame, config, inspection_tool="Realsense",
                        print_tool="LongCalibTool") -> CharacterizationResult
  ```
  plus `report["substrate"]` (spec §4): `source`, `sigma_mm`, `tilt_deg`, `plane`,
  `inlier_fraction`, `floor_mm`, `plane_offset_at_center_mm`,
  `compactness` (the three counts), `separation_margin_mm`.

- [ ] **Step 1: Config first.** In `ExtrusionConfig`:
  - Delete `deposit_min_saturation` (`:836`), `deposit_min_chroma_fraction` (`:840`),
    `deposit_min_height_no_chroma_mm` (`:845`), `plane_distance_threshold_m` (`:817`),
    `deposit_min_height_mm` (`:828`) **and their comment blocks** (`:807-816`,
    `:829-844` — they document the design being deleted; the substrate module carries
    the new rationale). Add all five names to `RETIRED_EXTRUSION_CONFIG_KEYS`.
  - Add, where the deleted block sat:

```python
    # -- fitted-substrate segmentation (2026-08-30 design; modules/extrusion/substrate.py)
    # The deposit floor is DERIVED per frame: clamp(k * sigma, lo, hi) where sigma
    # is the substrate fit's own one-sided residual scale. k=3 reproduced the old
    # 1.5 mm constant on the 2026-08-30 archive (floors 1.55-1.74 mm); k=4 was
    # measured to fall off a cliff (completeness 0.358), hence the bound.
    substrate_sigma_k: float = Field(default=3.0, ge=1.0, le=3.5)
    # Clamp on the derived floor: ceiling under the measured k=4 cliff at 2.25 mm,
    # lower bound above raw sensor noise. A pathological fit can neither open the
    # floor to everything nor close it to nothing.
    substrate_floor_clamp_mm: list[float] = Field(
        default_factory=lambda: [1.0, 2.0], min_length=2, max_length=2)
    # Neighbourhood the plane is fitted in, about the plan centre. Spec §8: a wide
    # print that fills this disc starves the fit -- widen it rather than edit code.
    substrate_fit_radius_mm: float = Field(default=150.0, gt=0, le=1000)
    # Compactness filter: a surviving component must be at least this many bead
    # widths long along its principal axis. The 22-point checker patch of
    # 2026-08-29 dies here, in geometry, where the colour gate used to catch it.
    deposit_min_length_beads: float = Field(default=3.0, ge=0, le=50)
```

- [ ] **Step 2: Rewrite `process_observation`'s front end** (`:706-758` replacing the
gate/registration/floor code; imports: `from .substrate import PlaneSubstrate,
compactness_filter`):

```python
    mark = time.perf_counter()
    points, valid_depth = depth_to_work_points(depth, geometry, T_work_camera)
    counts["raw_depth_pixels"] = int(valid_depth)
    timings["backproject_ms"] = (time.perf_counter() - mark) * 1000
    keep("backprojected", points)
    setup, recipe = plan.setup, plan.recipe
    center_xy = np.array([setup.center_x_mm, setup.center_y_mm])
    radius = np.linalg.norm(points[:, :2] - center_xy, axis=1)
    # The substrate is refitted EVERY frame: a reused fit was measured to carry
    # pose-dependent depth bias straight into the numbers (radius sigma 0.107 ->
    # 0.234 mm; spec §3.4). The work frame supplies only the up-axis and the
    # search band -- the surface itself is measured, never assumed at Z=0.
    near = radius <= config.substrate_fit_radius_mm
    substrate = PlaneSubstrate.fit(points[near],
                                   clamp_mm=tuple(config.substrate_floor_clamp_mm))
    heights = substrate.height(points)
    min_h = substrate.floor_mm(config.substrate_sigma_k)
    max_h = layer.nominal_z_mm + recipe.bead_diameter_mm / 2 + config.deposit_height_margin_mm
    in_height = (heights >= min_h) & (heights <= max_h)
```

then keep the existing radial band / `roi_diag` structure, with these adjustments:
`height_band_mm` reports `[min_h, max_h]`, `observed_z_mm` reports percentiles of
`heights` (rename the key to `observed_height_mm` and update the one test that reads
it), and after `points = points[roi]` insert the compactness stage:

```python
    kept = compactness_filter(points, mm_per_pixel=config.raster_mm_per_pixel,
                              bead_mm=recipe.bead_diameter_mm,
                              min_length_beads=config.deposit_min_length_beads,
                              min_points=config.cluster_min_points, counts=counts)
    keep("compactness", kept)
    points = kept
```

The `floor` report dict becomes the substrate block (replacing
`{"source": "build_plane", ...}`):

```python
    sub_heights = heights[near]
    below = sub_heights[sub_heights < min_h]
    substrate_report = {**substrate.to_report(),
        "floor_mm": round(float(min_h), 3),
        "plane_offset_at_center_mm": round(float(substrate.plane_z(center_xy)), 3),
        "substrate_p99_mm": (round(float(np.percentile(below, 99)), 3)
                             if len(below) else None)}
```

and after the deposit cluster exists, complete spec §4's one derived number:

```python
    deposit_h = substrate.height(deposit)
    if substrate_report["substrate_p99_mm"] is not None:
        substrate_report["separation_margin_mm"] = round(
            float(np.median(deposit_h)) - substrate_report["substrate_p99_mm"], 3)
    else:
        substrate_report["separation_margin_mm"] = None
```

Wire `substrate_report` into the assembled `report` (`:897`) under `"substrate"`
(replacing the `"floor"` key — grep for consumers of `report["floor"]` and update
them), the compactness counts ride in `counts` already. Pass `substrate=substrate`
into the `ring_geometry` call at `:877` (its `build_plane_z_mm` argument becomes
dead — delete it from the signature Task 6 left).

- [ ] **Step 3: Rewrite `characterize_ring`'s front end** (`:952-980`) the same way:
`depth_to_work_points`, substrate fitted on `radial <= config.substrate_fit_radius_mm`
about `search_center_mm`, `min_z` → `substrate.floor_mm(config.substrate_sigma_k)`
applied to `substrate.height(points)`, the
`search_cylinder_above_floor_fraction` diagnostics computed on heights, and
`counts["deposit_floor_mm"]` renamed `counts["substrate_floor_mm"]` (update
`test_the_shape_gate_rejection_names_the_capture_that_produced_the_blobs` at
`tests/test_extrusion_measure.py:545` and
`test_characterize_records_what_the_search_cylinder_held_before_the_floor` at `:578`
accordingly — the latter's 2.5 mm expectation becomes the derived-floor value).
Drop `color`/`K`/`dist` from the signature and fix its callers (grep
`characterize_ring(` across `tasni/` — the service/module layer passes camera config
that now simply isn't passed).

- [ ] **Step 4: Delete the dead code.** `chroma_gate_mask`, `deposit_floor_mm`, the
`ColorRegistered` import in `processing.py`, `color`/`K`/`dist` in `measure_take` and
all its callers, `figures._chroma_dist` and `reconstruct_take_inputs`' `dist` slot,
and `TakeData.chroma_dist` if nothing else reads it (grep first). `figures.py` keeps
reading `color.png` for photographic panels — only the seam stops taking it.

- [ ] **Step 5a: The real crash frame first — measure, then pin.** The two headline
gate tests (`tests/test_extrusion_measure.py:254`, `:294`) run the REAL 2026-08-29
cell frame (`tests/fixtures/extrusion/ring1/ring1_take04_branchguard_20260829.npz`,
loaded by `_ring1_take04()` at `:238`) — the board patch welded near the ring's +X
flank that exhausted the branch guard. This fixture is the hard case for the new
chain and it was NOT part of the spec's offline validation (that used the 2026-08-30
archive). Note it is a pre-protocol-2 capture (1 mm depth words, `gf.aligned`
geometry, `voxel_size_m=0.002` pinned in the old tests — keep that override).

Run the new chain on it BEFORE writing assertions (a scratch script in the scratchpad
directory, printing `metrics`, `report["substrate"]`, the compactness counts, and
whether any cluster point sits at `x > 254 and r > 47`). Then pin per outcome:

  - **Ring valid, patch excluded** (by floor, compactness, radial trim, or spur
    pruning — the mechanism doesn't matter): write
    `test_the_board_patch_dies_by_geometry_not_colour` asserting exactly what the
    old `:254` test asserted (valid, `path_completeness >= 0.98`, gap < 5°, radius
    ≈ 42.2 ± 1.0, no cluster point at `x > 254` with `r > 47`, `r.max() < 54`),
    minus the `chroma_gate_applied` line.
  - **Frame refused** (branch guard or ROI starvation): pin the refusal with
    `pytest.raises`, docstringed as the honest outcome for a contaminated
    pre-protocol-2 frame — the 2026-08-29 handoff's own ruling is that the crash
    was the good outcome. Note in the fixture README that protocol-2 captures are
    the supported path.
  - **Ring valid WITH the patch included** (radius biased ~0.6–0.7 mm large, the
    silent-wrong case): **STOP THE TASK and report to the operator** — do not ship
    the swap, do not loosen anything. This is the one outcome the design cannot
    accept (spec §3.5's premise would be falsified on legacy data).

Also add the cause-locking twin (mirrors the deleted `:294` test): the same frame
with `deposit_min_length_beads=0.0` (compactness off) must NOT reach a clean valid
measurement that includes the patch — expect the branch-guard crash or an excluded
patch; whichever the run shows, pin it, so compactness's load-bearing role (or the
guard's) stays measured.

- [ ] **Step 5b: Delete the remaining gate tests and re-scope the fixtures.**
  - `tests/test_extrusion_processing.py`: the three gate tests at `:72`, `:94`, `:153`.
  - `tests/test_extrusion_measure.py`: `:462`, `:482`, `:500` (pure `chroma_gate_mask`
    unit tests), the `chroma` count keys inside `:545`/`:564-575` (that test keeps its
    shape-gate purpose with `substrate_floor_mm` instead), and the module-level
    `chroma_gate_mask, deposit_floor_mm` import at `:232`.
  - `tests/test_extrusion_figures.py:800` (the lens-model gate figure test — its
    subject no longer exists).
  - Update `tests/fixtures/extrusion/ring1/README.md` and `ring2/README.md`: the
    abstain contract is gone; ring1 now documents the geometry-vs-contamination pin
    of Step 5a, ring2 documents segmentation with no colour input at all.

  Add the derived-floor test (uses `observe()` and `scene_plan()` from the same file,
  and `syn.RingSpec`/`syn.flat` from `tests/extrusion_synthetic.py`):

```python
# ------------------------- geometric segmentation: the substrate replaces colour

def test_the_derived_floor_lands_where_the_constant_used_to():
    """clamp(k * sigma) must land in the old constant's neighbourhood on the
    synthetic plane (spec §3.4: 1.55-1.74 mm measured on the cell archive), and
    the report must carry the §4 health block."""
    plan = scene_plan(radius=40.0, bead=9.0, layer_height=6.0)
    out = observe(plan, 1, [syn.RingSpec(40.0, 9.0, CENTER, height_fn=syn.flat(6.0))])
    sub = out.report["substrate"]
    assert sub["source"] == "fitted_plane"
    assert sub["sigma_mm"] > 0.0
    assert 1.0 <= sub["floor_mm"] <= 2.0
    assert out.metrics.valid
```

  Expect collateral drift in this file: many synthetic tests previously ran at the
  abstained 2.5 mm floor and now run at the derived one (likely lower). A synthetic
  test that starts failing is telling you what the new floor admits — fix the CHAIN
  only if the admitted points are substrate (raise `noise` handling, check the fit),
  fix the TEST only if its assertion encoded the 2.5 mm constant.

- [ ] **Step 6: Verify — targeted suites, then the golden gate**

Run: `py -3.10 -c "import tasni.modules.extrusion.processing, tasni.modules.extrusion.figures"`
then `py -3.10 -m pytest tests/test_extrusion_processing.py tests/test_extrusion_measure.py tests/test_extrusion_substrate.py tests/test_extrusion_figures.py tests/test_extrusion_job.py tests/test_extrusion.py -q`
then `py -3.10 -m pytest tests/test_extrusion_golden.py -v`
Expected: all green. The golden layer-1 numbers move within acceptance (spec §2.1
measured: completeness 0.992–0.993, radius mean 40.98, σ 0.107 ≤ 0.15); layer-2 stays
invalid with completeness within 0.05 of archived. **If layer-2 comes back valid,
STOP — that is the §2.4 false positive; do not loosen the golden test.**

- [ ] **Step 7: Commit**

```bash
git add tasni tests
git commit -m "feat(extrusion): segmentation by geometry -- fitted substrate + derived floor + compactness replace the chroma gate"
```

---

### Task 8: Acceptance, docs, push

**Files:**
- Modify: `tests/test_extrusion_golden.py` (substrate-report assertions)
- Modify: `AGENTS.md` (the open-items section: the chroma-gate replacement is built),
  `docs/agent-debug-map.md` if it routes readers at the gate/floor (grep `chroma` and
  `deposit_floor` across `docs/` and fix what's now wrong — do not rewrite history
  docs like the 2026-08-30 handoff; add a one-line "superseded by" pointer at their
  top only where the spec hasn't already)

**Interfaces:** none new.

- [ ] **Step 1: Extend the golden harness with the §4 health block:**

```python
def test_substrate_report_is_present_and_sane_on_every_take():
    for name in LAYER1 + LAYER2:
        sub = _measure(name).report["substrate"]
        assert sub["source"] == "fitted_plane"
        assert 0.3 <= sub["sigma_mm"] <= 1.0, (name, sub)      # spec §3.3 band, loose
        assert 1.0 <= sub["floor_mm"] <= 2.0, (name, sub)
        assert 0.3 <= sub["tilt_deg"] <= 1.0 or sub["tilt_deg"] < 0.3, name
        assert sub["separation_margin_mm"] is None or sub["separation_margin_mm"] > 0, name
```

Tighten the numeric bands to what the run actually shows (record the observed values
in the test's docstring — these are the per-take baselines spec §3.4 defers to), then
re-run: `py -3.10 -m pytest tests/test_extrusion_golden.py -v`.

- [ ] **Step 2: Docs sweep.** `grep -rn "chroma\|deposit_floor\|floor_profile" AGENTS.md docs/agent-debug-map.md CLAUDE.md` — update live guidance (AGENTS.md open items, the debug map's extrusion row) to point at `substrate.py` and the spec; leave dated
handoffs/audits as history.

- [ ] **Step 3: Final verification + push**

Run: `py -3.10 -m pytest tests/test_extrusion_golden.py tests/test_extrusion_substrate.py -q`
and `cd tasni/webui && npm run build` one last time. Then:

```bash
git add -A
git commit -m "docs(extrusion): substrate segmentation shipped -- goldens tightened, guidance repointed"
git push
```

Report the pushed hashes. No Jetson deploy: nothing under `server/` changes in this
plan.

---

## Self-Review (done at planning time)

- **Spec coverage:** §3.1/§3.2/§3.3 → Task 4; §3.4 → Tasks 4+7 (config); §3.5 →
  Tasks 5+7; §3.6 → Tasks 3, 6, 7; §3.7 → Task 1; §4 → Tasks 7+8; §5 → Tasks 2, 4, 5,
  7, 8. Out-of-scope items (§7) deliberately have no tasks.
- **Known judgment calls:** `observed_z_mm` → `observed_height_mm` rename (Task 7)
  is a report-key change; grep for readers before renaming. Task 7 Step 5a's
  assertions are deliberately outcome-dependent: the 2026-08-29 crash-frame fixture
  was not in the spec's offline validation, so the step measures first and pins what
  it finds — with a hard stop on the one unacceptable outcome (patch silently
  included). That is evidence-first discipline, not a placeholder.
- **Type consistency:** `measure_take`'s transitional signature (Task 1) loses
  `color`/`K`/`dist`/`floor_profile` across Tasks 6–7; each task states the signature
  it leaves behind. `ring_geometry` gains `substrate=` in Task 6 (dormant) and its
  caller passes it in Task 7.
