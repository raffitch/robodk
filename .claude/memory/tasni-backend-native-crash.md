---
name: tasni-backend-native-crash
description: "SOLVED 2026-09-01 - the repeated Windows access violations were TWO OpenBLAS runtimes (numpy's openblas64_ + scipy's own copy) multithreading in one process. Fixed by pinning the thread pools in tasni/__init__.py (bcd05cd)."
metadata: 
  node_type: memory
  type: project
  originSessionId: 301796e3-2dbd-4a52-9fe9-f44aa33275d1
  modified: 2026-09-01T09:26:42.568Z
---

Eight hard crashes over 2026-08-30/31 and 2026-09-01, root cause unknown for
three days. **Solved 2026-09-01.**

**How it was found:** `%TEMP%\tasni-backend.crash.log` (faulthandler) was nearly
useless — 504 bytes, truncated mid-line, three interleaved "Windows fatal
exception" headers because several threads faulted at once. The answer came from
the **Windows Application-Error log**, which names the faulting DLL for every
crash:

```
Get-WinEvent -FilterHashtable @{LogName='Application'; StartTime=(Get-Date).AddHours(-6)} |
  Where-Object { $_.Id -eq 1000 }
```

```
Faulting module: libopenblas64__v0.3.21-gcc_10_3_0.dll
Exception code:  0xc0000005 (access violation)   offset 0x11a321
```

**Root cause:** the process loads **two different OpenBLAS runtimes** — numpy
1.24.2 links `openblas64_` (ILP64) and scipy 1.10.1 ships its own separate 35 MB
copy under `scipy.libs/`. Confirmed live with `threadpoolctl.threadpool_info()`:
both resident in one interpreter. Each sizes its own thread pool to the machine,
and the app calls into both from background threads (job runner, live-preview
loop). That is a documented way to get exactly this fault, and it explains why
the crashes clustered in extrusion/scan work rather than at rest.

**Fix (`bcd05cd`):** `OPENBLAS_NUM_THREADS` / `OMP_NUM_THREADS` /
`MKL_NUM_THREADS` set to "1" as the FIRST statement in `tasni/__init__.py`.

**How to apply:**
- It must stay the first statement there. OpenBLAS reads those at DLL init, so
  setting them after numpy loads does nothing. `py -3.10 -m tasni` runs the
  package init before `__main__`, so the real entry point is covered.
- `setdefault`, not assignment — an operator can still ask for more threads.
- Verify with `threadpoolctl.threadpool_info()`: both libraries must say
  `threads=1`.
- **For ANY future native crash, read the Windows Application-Error log first.**
  It names the module; the faulthandler dump does not survive a multi-thread
  fault. See [[windows-python-and-encoding-traps]].
- Confirmation is the crash not recurring; the mechanism is identified and the
  mitigation is standard, but it has not yet had a long clean run.
