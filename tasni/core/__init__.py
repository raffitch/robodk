"""tasni.core — shared services reused by every workflow module.

Nothing in here is calibration- or scan-specific. Modules receive these
services through a :class:`~tasni.modules.base.ServiceContainer` rather than
importing ``robolink``/``socket`` directly, which is what lets new modules plug
in as pure leaves.
"""
