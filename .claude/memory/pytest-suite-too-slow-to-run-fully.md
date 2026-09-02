---
name: pytest-suite-too-slow-to-run-fully
description: "Don't run the full pytest suite in this repo - it's too slow; run focused tests only, or skip to syntax/import checks when asked"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 963be792-72bf-4fcf-ae9a-77dd06dcf629
  modified: 2026-08-13T07:42:33.893Z
---

In this repo, do NOT run the full `py -3.10 -m pytest -q` suite as a routine
verification step. The user interrupted it twice and said: "stop running tests for
now and just do the tasks, keep check minimum indispensable cause the tests are
taking too long."

**Why:** the suite is 400+ tests spanning heavy numpy/RANSAC/reconstruction paths,
and the user is watching for progress on the actual change, not for a green bar.
Blocking on a multi-minute suite after every edit stalls the work they can see.

**How to apply:** prefer, in order — (1) `py -3.10 -c "import ..."` plus
`ast.parse` for syntax/import sanity, which is seconds; (2) `pytest -k
"<specific names>" -p no:cacheprovider` on just the tests covering the change,
which typically runs in 2-4 s; (3) the frontend `npm run build`, which is a real
compile gate and IS worth running. Only run the whole suite if the user asks, or
before something irreversible. When you skip it, say so plainly in the summary
rather than implying the suite is green — see
[[verification-honesty-in-summaries]].

Note this cuts against the repo's own CLAUDE.md verification matrix, which lists
the full suite per task; the user's live instruction wins.
