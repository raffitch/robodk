"""Shared pytest setup.

Pre-load onnxruntime (if installed) before any test imports RoboDK's ``robolink``.
``robolink`` pulls in PySide2/Qt, whose shiboken2 import hook breaks onnxruntime's native
DLL init on Windows if Qt loads first — so without this the SAM boundary tests would skip
(onnxruntime "unavailable") and a fatal-exception dump would print mid-run. This mirrors
the same fix in ``tasni/__init__.py`` for the live app. Guarded: no onnxruntime -> no-op.
"""
try:  # noqa: SIM105
    import onnxruntime as _onnxruntime  # noqa: F401
except Exception:
    pass
