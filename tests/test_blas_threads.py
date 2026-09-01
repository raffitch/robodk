"""The BLAS thread pools must be pinned before numpy is ever imported.

2026-09-01: the backend died with a Windows access violation for the eighth
time in three days. The Windows Application-Error log named the faulting module
exactly:

    Faulting module name: libopenblas64__v0.3.21-gcc_10_3_0.dll
    Exception code: 0xc0000005      Fault offset: 0x11a321

This process loads TWO different OpenBLAS runtimes: numpy 1.24.2 links
``openblas64_`` (the ILP64 build above), and scipy 1.10.1 ships its own
separate 35 MB ``scipy.libs/libopenblas-802f9ed1....dll``. Each spawns its own
thread pool sized to the machine, and the app calls into both from background
job threads (the job runner, the live-preview loop). Two OpenBLAS runtimes
multithreading against each other in one process is a known way to get exactly
this fault.

Pinning both pools to a single thread removes the contention. The variables are
read by OpenBLAS at DLL init, i.e. when numpy is first imported -- so setting
them afterwards does nothing at all, and that ordering is the only thing these
tests can meaningfully protect.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Every knob that can size a BLAS/OpenMP pool in this process. MKL is included
# for completeness: numpy could be swapped to an MKL build and the variable is
# inert when no MKL is present.
THREAD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")


def _run(code: str) -> str:
    out = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT),
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, f"subprocess failed:\n{out.stdout}\n{out.stderr}"
    return out.stdout.strip()


def test_importing_tasni_pins_every_blas_thread_pool():
    got = _run(
        "import tasni, os;"
        f"print(','.join(os.environ.get(v, 'UNSET') for v in {THREAD_VARS!r}))")
    assert got == "1,1,1", f"thread pools not pinned: {got}"


def test_the_pin_happens_before_numpy_is_imported():
    """The whole point: OpenBLAS reads these at DLL init.

    If numpy is already in sys.modules when tasni sets them, the pools are
    already sized and the setting is decoration. This asserts tasni's package
    init does not drag numpy in ahead of the pin.
    """
    got = _run(
        "import sys;"
        "assert 'numpy' not in sys.modules, 'numpy imported before tasni ran';"
        "import tasni, os;"
        "print(os.environ['OPENBLAS_NUM_THREADS'], 'numpy' in sys.modules)")
    pinned, _numpy_loaded = got.split(maxsplit=1)
    assert pinned == "1"


def test_an_operator_override_is_respected():
    """Pinned by default, not by force -- someone benchmarking on a quiet
    machine must still be able to ask for more threads."""
    code = ("import os;"
            "os.environ['OPENBLAS_NUM_THREADS'] = '4';"
            "import tasni;"
            "print(os.environ['OPENBLAS_NUM_THREADS'])")
    assert _run(code) == "4"


def test_numpy_still_works_after_the_pin():
    """A pinned pool must still compute -- this is the operation the scan's
    plane fit leans on."""
    got = _run(
        "import tasni, numpy as np;"
        "a = np.random.default_rng(0).normal(size=(300, 3));"
        "print(round(float(np.linalg.svd(a - a.mean(0), full_matrices=False)[1][0]), 3) > 0)")
    assert got == "True"
