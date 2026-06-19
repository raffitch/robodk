"""Calibration module #1 — ChArUco eye-in-hand hand-eye calibration.

Refactor of the original ``macros/AutoCalibrate.py`` onto the tasni core, plus
the quality metrics that macro never reported (reprojection error in pixels and
held-out validation-pose error). Solver stays OpenCV ``calibrateHandEye`` TSAI.
"""
