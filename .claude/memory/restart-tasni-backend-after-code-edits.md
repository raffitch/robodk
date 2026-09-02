---
name: restart-tasni-backend-after-code-edits
description: "The Tasni backend caches imported modules, so editing tasni/ has no effect until restart — the app now reports its running build and warns when stale (554a379)"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: abd4bd3b-dd57-48ee-9017-c017072cddda
  modified: 2026-08-27T16:51:31.060Z
---

The Tasni app runs as a persistent `py -3.10 -m tasni --port 8000` process. Python caches
imported modules, so **editing `tasni/**.py` has no effect on a running app**. A cell test
after an edit exercises whatever code was on disk when the process started, and looks
exactly like a failed fix.

**This cost two full cell-test cycles on 2026-08-27** before it was caught, then a third
because I told the user to restart *while still pushing* — the process came up 2.5 minutes
before the fix hit disk.

**The app now detects this itself (`554a379`)** — `tasni/core/build_info.py` fingerprints
the packaged sources at import (= process start) and compares against the working tree:
- `GET /api/health` → `build` block with `loaded_sha` and `stale`.
- Extrusion run reports carry the same block; the old misleading field is renamed
  `git_commit_checked_out` (it reported `git rev-parse HEAD` at *report* time, so a report
  once claimed a commit the process had never loaded).
- Simulation and print jobs log `STALE CODE: ...` beside the mode banner.

**How to apply:** never ask for a cell test until the fix is committed AND the app has been
restarted afterwards. Verify with `Get-Process -Name python,py | Select Id,StartTime`
against `(Get-Item <file>).LastWriteTime`, or just read `build.stale` from `/api/health`.
Restarting is a state change on a live robot cell — the permission classifier blocks it, and
it should stay the user's call. It also clears in-memory state (the generated plan and its
approvals), so the user must re-generate coordinates afterwards.

Note `tasni.config.json` overrides code defaults, so reading a config value back from
`/api/health` does NOT prove fresh code. See [[extrusion-a4-wrist-flip-fix]].
