"""
rdk_session.py - Shared, ISOLATED RoboDK connection for the extract/sync scripts.

Why this exists: RoboDK's API attaches to ANY running RoboDK instance by default.
That means a script could reach into the station you have open in the GUI. To avoid
that entirely, we always spawn a private, headless instance:

    -NEWINSTANCE   force a brand-new RoboDK process (ignore your open window)
    -NOUI          run with no window (invisible, fast)
    -SKIPINI       don't read your personal RoboDK settings
    -EXIT_LAST_COM exit when the last API connection closes

Combined with quit_on_close=True, the instance disappears when the script ends, so
nothing is left running and your interactive RoboDK is never touched.
"""
from robodk.robolink import Robolink

ISOLATED_ARGS = ["-NEWINSTANCE", "-NOUI", "-SKIPINI", "-EXIT_LAST_COM"]


def connect():
    """Return a Robolink bound to a private headless RoboDK instance."""
    return Robolink(args=ISOLATED_ARGS, quit_on_close=True)
