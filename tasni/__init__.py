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

__version__ = "0.1.0"
