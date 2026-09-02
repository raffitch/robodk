# Tasni

A robotic-fabrication control platform for a **KUKA KR150 R2700** cell with an Intel
**RealSense D435i** depth camera, built on the RoboDK Python API.

Rather than a pile of one-off macros, Tasni is a **module registry on a shared core** —
robot/RoboDK connection, camera client, config, job runner, live monitoring and logging
are provided once, and each workflow (calibration, scanning, extrusion/additive) plugs in
as a module. RoboDK stays the orchestrator; a Jetson Nano stays the camera server.

> **Status:** in active use on a real robot cell. Calibration and scanning are live;
> the extrusion module is producing measurements for a paper. Numbers quoted in the
> docs are measured on the physical cell, not simulated.

## Quick start

Requires **Python ≥ 3.10**, RoboDK installed at `C:\RoboDK`, and the cell powered up.

```powershell
pip install -e .            # core
pip install -e .[scan]      # + Open3D, needed for 3D scanning
.\start.ps1                 # dev: FastAPI backend + Vite hot-reload on :5173
.\start.ps1 prod            # build the UI and serve everything on :8000
```

`start.bat` is the cmd.exe equivalent, `start.sh` the Git Bash one. The app connects to
RoboDK in **`attach`** mode: it binds the already-running GUI, and if that GUI has no
station holding the robot it opens `Tasni.rdk` into it — so you are always driving the
real cell, never an empty station.

Check the cell is ready before a session:

```
py -3.10 tools/cell_health.py
```

## Layout

| Path | What it is |
|------|-----------|
| `tasni/` | The platform. FastAPI backend (`tasni/webapp`), React + Vite + TS frontend (`tasni/webui`), shared core, and the workflow modules. See [tasni/README.md](tasni/README.md). |
| `server/` | The **Jetson** camera server — runs on the Nano as a systemd service, streams RealSense colour + depth over TCP 1024. See [server/README.md](server/README.md). |
| `macros/` | Python extracted from the binary RoboDK station (the legacy embedded scripts). |
| `docs/` | Design docs, handoffs and measurement records. Start at [docs/agent-debug-map.md](docs/agent-debug-map.md). |
| `tools/` | Operational scripts — Jetson deploy/probe, cell health, characterization, figures. |
| `tests/` | Test suite. It is slow; run targeted subsets with `pytest -k`, not the whole thing. |
| `.claude/` | Assistant context and accumulated project memory — see [.claude/README.md](.claude/README.md). |

## The RoboDK station

`Tasni.rdk` is a ~117 MB **binary** station, so the Python embedded in it cannot be edited
directly. Two bridge scripts move that code in and out:

```
py -3.10 rdk_extract.py "Tasni.rdk"             # station -> macros/*.py (overwrites them)
py -3.10 rdk_sync.py "Tasni.rdk"                # macros/*.py -> Tasni.synced.rdk (safe)
py -3.10 rdk_sync.py "Tasni.rdk" --inplace      # overwrite the source station
```

Both connect through a **private headless RoboDK instance**, so running them never
reaches into a station you have open in the GUI. Keep a backup before using `--inplace`.

## The Jetson camera server

The Nano clones this repo to `~/robodk` and auto-pulls the checked-out branch every
~2 minutes via a systemd timer. It restarts the camera **only** when the pulled commit
actually touched `server/`, and defers the whole update while a client is mid-capture, so
running code never drifts from on-disk code.

```
py -3.10 tools/jetson_deploy.py status    # active? listening on 1024? timer? logs
py -3.10 tools/jetson_deploy.py deploy    # immediate pull + restart from the current branch
py -3.10 tools/jetson_deploy.py bootstrap # (re)install service + auto-pull, idempotent
```

Because this repo is **private**, the Nano cannot clone it anonymously the way it
could the old public one. It authenticates with a **read-only deploy key**
(`~/.ssh/tasni_deploy`, plus a `Host github-tasni` block in `~/.ssh/config`), scoped
to this repo alone — so a shop-floor device never holds an account-wide token in
plaintext. Add the public key under **Settings → Deploy keys** before bootstrapping a
fresh Nano. Note the auto-pull script exits 0 on *any* failure by design, so a
credential problem shows up as the camera silently running stale code — check
`jetson_deploy.py status` after changing anything about the remote.

Device details, firmware versions and known operational issues:
[docs/jetson-scanner.md](docs/jetson-scanner.md).
Credentials live in `secrets/jetson.env` — **git-ignored, never committed**.

## Working agreement

Every change is committed **and pushed**: the operator reviews progress from the pushed
history, and the Jetson deploys from it, so an unpushed commit is an invisible one.
Changes touching `server/` also need the Jetson deployed or restarted.

Full instructions for humans and AI assistants: [CLAUDE.md](CLAUDE.md) (project
specifics) and [AGENTS.md](AGENTS.md) (tool-agnostic entry point).

## Gotchas worth knowing before you start

- Use `py -3.10` on Windows — there is no bare `python` on PATH.
- Never round-trip source through PowerShell `Get-Content`/`Set-Content`; it silently
  mangles this repo's UTF-8.
- The backend caches imported modules, so **restart it after code edits** or you will
  test stale code on the real robot.
- Loading the 117 MB station takes a minute or two per script run. That is expected.
