"""tasni — a robotic-fabrication control platform built on RoboDK.

ONE external app that drives a RoboDK cell and hosts ALL robot workflows
(calibrate, scan, ArUco-to-plane, define-targets, 3D printing, ...) as pluggable
MODULES on a shared core (RoboDK connection, camera client, config, job runner).

Layout:
    tasni.core       shared services every module reuses
    tasni.modules    the workflow modules + the registry they plug into
    tasni.webapp     the FastAPI web shell that hosts the modules

The calibration module (``tasni.modules.calibration``) is module #1 and the
proof-of-pattern: a thin leaf on top of the core, with nothing scan- or
calibration-specific living in the core itself.
"""

# Pin the BLAS/OpenMP thread pools BEFORE anything can import numpy. This must be
# the first executable statement in the package: OpenBLAS reads these at DLL init,
# so setting them after numpy has loaded does nothing at all.
#
# Why: on 2026-09-01 the backend died with a Windows access violation for the eighth
# time in three days, and the Application-Error log named the module -
#
#     Faulting module: libopenblas64__v0.3.21-gcc_10_3_0.dll
#     Exception code:  0xc0000005          Fault offset: 0x11a321
#
# This process loads TWO different OpenBLAS runtimes: numpy 1.24.2 links
# ``openblas64_`` (the ILP64 build above) and scipy 1.10.1 ships its own separate
# 35 MB copy under ``scipy.libs/``. Each sizes its own thread pool to the machine,
# and the app calls into both from background threads (the job runner, the live
# preview loop). Two OpenBLAS runtimes multithreading against each other in one
# process is a documented way to get precisely this fault, and it explains why the
# crashes clustered in the extrusion/scan work rather than at rest.
#
# Single-threaded BLAS costs this workload almost nothing: the heavy lifting is
# open3d (its own threading, unaffected) and element-wise numpy, not large matrix
# products. ``setdefault`` rather than assignment, so an operator benchmarking on a
# quiet machine can still ask for more.
import os as _os

for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_var, "1")

# Pre-load onnxruntime (optional; used by the scan SAM boundary) BEFORE anything imports
# RoboDK's ``robolink`` — which pulls in PySide2/Qt. shiboken2 installs a global import
# hook, and if Qt loads first it breaks onnxruntime's native DLL init on Windows ("DLL
# initialization routine failed"), so SAM would silently fall back to colour forever. Every
# app entry point (``python -m tasni``, uvicorn ``tasni.webapp``) runs this package init
# before any ``tasni.core`` submodule imports robolink, so loading onnxruntime here fixes
# the order. Guarded: a core install without the ``[sam]`` extra just skips it.
try:  # noqa: SIM105
    import onnxruntime as _onnxruntime  # noqa: F401
except Exception:
    pass

__version__ = "0.1.0"
