"""Scan module (#2): auto-scan a work platform → fused mesh + working frame.

Mirrors the calibration module's shape: a pure library (``reconstruct``, ``plane``)
with no RoboDK/socket/thread dependency, plus a ``service`` that orchestrates the
core services and a ``module`` that exposes the REST surface + UI hook.
"""
