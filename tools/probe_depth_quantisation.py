"""Audit R2 acceptance: depth word granularity in a centre patch, from the workstation.

    py -3.10 tools/probe_depth_quantisation.py [--patch 120]

Before protocol 2 (2026-08-29, arm parked): uint16 (720,1280) valid 0.999, unique 25,
min step 1 (mm). After: unit 0.1 mm, expect >= 200 unique values in the same patch.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tasni.core.camera import CameraClient  # noqa: E402
from tasni.core.config import load_config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch", type=int, default=120)
    args = ap.parse_args()
    frame = CameraClient(load_config().camera).grab(with_depth=True, timeout=20)
    d = frame.depth
    g = frame.geometry
    v = d[d > 0]
    print(f"{d.dtype} {d.shape} unit {g.depth_unit_mm} mm valid {v.size / d.size:.3f} "
          f"range {v.min() * g.depth_unit_mm:.1f}..{v.max() * g.depth_unit_mm:.1f} mm "
          f"temps {g.temps} preset {g.device.get('visual_preset')}")
    cy, cx = d.shape[0] // 2, d.shape[1] // 2
    r = args.patch // 2
    p = d[cy - r:cy + r, cx - r:cx + r]
    pv = np.unique(p[p > 0])
    print(f"centre {args.patch}x{args.patch}: unique {pv.size}, min step "
          f"{(np.diff(pv).min() if pv.size > 1 else 0) * g.depth_unit_mm:.2f} mm")
    return 0


if __name__ == "__main__":
    sys.exit(main())
