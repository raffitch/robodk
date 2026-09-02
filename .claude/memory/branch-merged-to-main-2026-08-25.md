---
name: branch-merged-to-main-2026-08-25
description: "2026-08-25: calibration-improvements MERGED to main (51849b1) and deleted locally; the Jetson now tracks main, not the feature branch."
metadata: 
  node_type: memory
  type: project
  originSessionId: 3372a85a-dae9-45ab-8cfc-d5be9e713794
  modified: 2026-08-25T10:38:08.094Z
---

On **2026-08-25** the long-lived `calibration-improvements` branch was merged into
`main` (merge commit `51849b1`, pushed) and **deleted locally**. `origin/calibration-improvements`
still exists on GitHub but is merged and frozen — do not keep committing to it.
Earlier memories ([[calibration-improvements-status]], [[scan-module-status]],
[[workframe-two-path-plan]]) that describe work "on branch `calibration-improvements`"
describe commits that are now all in `main`.

**The merge was a no-op in content:** the merged tree came out byte-identical to the
branch tip. `main`'s 6 branch-less commits were older duplicates the branch had already
superseded, so both conflicts (`server/server_unicast_syncronous.py`,
`tests/test_scan_overlay.py`) resolved to the branch side.

**Jetson consequence (the non-obvious part):** the Jetson follows whatever branch is
checked out in `/home/jetson/robodk`, and `tools/jetson_deploy.py deploy` pushes the
branch checked out on the *workstation*. It was re-pointed to `main` as part of this
merge and verified active + listening on 1024. If a future session creates a new
feature branch and runs `deploy`, the Jetson will switch to that branch — that is the
mechanism, not a bug.

**CLAUDE.md is stale on this point**: its working agreement still says "Commit + push
the working branch (today: `calibration-improvements`)". The user was told; they have
not asked for it to be edited.
