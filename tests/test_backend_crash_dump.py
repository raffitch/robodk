"""The backend's native-crash instrument has to work when it is needed.

Seven hard crashes across 2026-08-30/31 (five sharing one ntdll heap access
violation, one BEX64) left no Python-level evidence at all: an access violation
unwinds no frames, so uvicorn's captured stdout simply stops mid-request. This
test is the proof that ``_arm_crash_dump`` turns that silence into a named
thread and a stack -- asserted against a REAL access violation in a subprocess,
because a mock of faulthandler would prove only that the mock was called.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap

CRASHER = textwrap.dedent(
    """
    import ctypes, os, sys, threading, time
    sys.path.insert(0, {repo!r})
    os.environ["TASNI_CRASH_LOG"] = {log!r}
    from tasni.__main__ import _arm_crash_dump

    _arm_crash_dump()

    def worker():
        time.sleep(0.1)
        ctypes.string_at(0)          # null deref: the shape of the real crash

    t = threading.Thread(target=worker, name="extrusion-measure", daemon=True)
    t.start()
    t.join(10)
    """
)


def test_a_native_access_violation_names_the_python_frame_that_caused_it(tmp_path):
    from tasni import __file__ as tasni_file
    from pathlib import Path

    repo = str(Path(tasni_file).resolve().parent.parent)
    log = tmp_path / "crash.log"
    script = tmp_path / "crasher.py"
    script.write_text(CRASHER.format(repo=repo, log=str(log)), encoding="utf-8")

    subprocess.run([sys.executable, str(script)], capture_output=True, timeout=180)

    assert log.is_file(), "the crash log was never created"
    text = log.read_text(encoding="utf-8", errors="replace")
    # The fault itself, so a silent stop is distinguishable from a clean exit.
    assert "access violation" in text.lower()
    # WHICH thread: the whole point. WER names a module; only this names the job.
    assert "extrusion-measure" in text or "in worker" in text
    # The startup banner, so successive crashes stay separable in one appended file.
    assert "tasni backend started" in text


def test_the_crash_log_is_appended_so_a_second_crash_keeps_the_first(tmp_path):
    """Crashes come in runs; the second is only interpretable beside the first."""
    from tasni import __file__ as tasni_file
    from pathlib import Path

    repo = str(Path(tasni_file).resolve().parent.parent)
    log = tmp_path / "crash.log"
    script = tmp_path / "crasher.py"
    script.write_text(CRASHER.format(repo=repo, log=str(log)), encoding="utf-8")

    for _ in range(2):
        subprocess.run([sys.executable, str(script)], capture_output=True, timeout=180)

    text = log.read_text(encoding="utf-8", errors="replace")
    assert text.count("tasni backend started") == 2
    assert text.lower().count("access violation") == 2
