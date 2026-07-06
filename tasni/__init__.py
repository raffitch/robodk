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
