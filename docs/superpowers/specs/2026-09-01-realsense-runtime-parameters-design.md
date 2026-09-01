# RealSense settings as runtime parameters — design

Opened 2026-09-01. Written because every camera-side A/B currently costs an edit to
`server/realsense-camera.service`, a `jetson_deploy.py bootstrap`, and a service restart —
per arm. A `smooth_delta` sweep of five values is five deploys, and the operator cannot
run one without an agent on the Jetson.

**Status: design only. No code has been written. The approval gate is unmet.**

---

## 1. Why this exists

The immediate driver is `docs/inspection-roll-probe-handoff.md` §3.1: a spatial-filter A/B
and, behind it, a `smooth_delta` sweep. Both are blocked on deploy friction rather than on
anything technical.

The deeper reason is that the deploy loop makes the *cheap* experiments cost the same as
the expensive ones. The camera has a large number of knobs nobody has ever turned (§2),
and the reason is not that they were considered and rejected — it is that turning one
costs a deploy, so none get tried.

**What this design does NOT do:** it does not make the camera's configuration mutable in
the sense that matters for reproducibility. Every setting stays declared in the unit file,
which remains the boot default and the reviewable diff. A runtime override is explicitly
temporary and dies on restart. That property is the whole safety argument (§4.2).

---

## 2. The inventory

Verified against the code on 2026-09-01, not from RealSense documentation.

### 2.1 Already parametrized (env var, restart)

`RS_VISUAL_PRESET`, `RS_LASER_POWER`, `RS_SPATIAL`, `RS_SPATIAL_SMOOTH_DELTA`. Plus
`RS_ASFOUND_DIR`, a path. That is all four functional knobs.

### 2.2 Host-side filter chain — the safe tier

Rebuilt from nothing by `setup_depth_filters()` on every start, so nothing here can
outlive the process.

| knob | today | scope |
|---|---|---|
| spatial on/off | env | in |
| spatial `smooth_delta` | env | in |
| spatial `magnitude`, `smooth_alpha`, `holes_fill` | SDK defaults, never touched | in |
| temporal `smooth_alpha`, `smooth_delta`, `persistency` | SDK defaults, never touched | in |
| threshold min/max | `RS_DEPTH_MIN_M=0.15`, `RS_DEPTH_MAX_M=1.5` module constants | **undecided — §6.4** |
| decimation | absent, deliberately (full-resolution scan data) | in, default off |
| hole filling | absent, deliberately (fabricates depth at surface edges) | in, default off |

Decimation and hole filling are included as *explicit* toggles defaulting off, so that
"we do not do this, on purpose" becomes a value in the greeting instead of an absence
someone has to infer.

### 2.3 Device options — the landmine tier

These live on the camera and survive a service restart. This is not theoretical: the
device once ran at 300 mW while the 2026-08-13 characterization assumed 150, because an
earlier build wrote it and "leave alone" preserved it.

Set today: `emitter_enabled`, `depth_units` (0.1 mm), plus laser power and visual preset
via env. Never touched at all: depth exposure and gain, colour exposure, white balance
(capability audit R11 — auto across every run to date).

### 2.4 Advanced mode — entirely unused

`get_depth_control()`: `textureCountThreshold`, `textureDifferenceThreshold`,
`deepSeaSecondPeakThreshold`, `deepSeaMedianThreshold`, `scoreThreshA/B`,
`lrAgreeThreshold`. `get_depth_table()`: `disparityShift`, `depthClampMin/Max`.

Worth stating plainly, because it changes what this work is *for*: those confidence
thresholds are exactly what decides whether a low-texture region yields a point or a hole.
That is the layer-2 dropout's precise symptom. If the roll probe returns SCENE-LOCKED,
this tier — not multi-view — is the next thing to try.

### 2.5 Structural — restart only, out of scope

Stream resolutions and fps (848×480 has never been A/B'd against 1280×720, audit R10), the
IR streams (enabled, never read, R6), the IMU (unused, R8). These require a pipeline
rebuild, which is a different operation from a filter swap and is not a "toggle".

### 2.6 A note on the capability audit

`docs/realsense-capability-audit-2026-08-29.md` is the inventory of record but is now
**stale in four places**: R2 (1 mm depth units), R3 (Jetson-side `align` discarding half
the depth field), R5 (hole filling) and R7 (720p colour) have all since been fixed in
code while the audit still lists them open. Fix the audit as part of this work; do not
re-chase those four.

---

## 3. The design

### 3.1 One command, on the existing protocol

The command loop is line-based and already ignores unknown commands. Add:

```
SET spatial=0 spatial_smooth_delta=8
```

answered by one JSON line: the ACHIEVED values after the change, in the same shape the
greeting's `filter_options` uses. `SET` with no arguments is a read.

Rejected alternatives: a REST sidecar on the Jetson (a second service to supervise, a
second thing to secure, and it would let a browser change the camera under a running
capture); and a config file the server watches (a file-watch race against a capture, and
it survives restarts, which is exactly the property §4.2 says not to have).

### 3.2 Scope: the safe tier only, for now

Ship §2.2 — the filter chain. Leave §2.3 and §2.4 pinned in the unit file.

This is the YAGNI cut, and it is also the risk cut: the filter chain cannot leave state on
the device, so the worst case of a bug is a bad capture that a restart fixes. Advanced-mode
knobs have the opposite property. §7 says what it would take to add them later.

### 3.3 Provenance is already solved — and is load-bearing

The greeting is sent per connection and, since `e06a2c5`, reports values read back off the
objects that actually run. Every take therefore records what it ran under, no matter who
set it. **This design is only safe because that shipped first.** A mutable setting without
per-connection achieved-value provenance would be a provenance disaster: two arms of an
A/B, indistinguishable on disk, with no error.

Consequence: every knob in §2.2 must appear in `filter_options`, not just the two that do
today. A knob that can be changed but not recorded must not ship.

### 3.4 A change ends in-flight sessions

`getFrames()` is called from both the streaming and burst paths against a module-global
chain. Swapping it mid-burst would fuse frames from two different chains into one median
with no error — and the existing fusion guard would not catch it, because it compares
*geometry*, and a filter swap does not change geometry.

Reuse the camera-generation mechanism that already exists for pipeline rebuilds: a `SET`
bumps the generation, and `_stale_greeting_close` ends any session greeted under the old
one. The client already handles that path by reconnecting and being greeted afresh.

### 3.5 Where the value lives

Three layers, in precedence order:

1. **Unit file** — the boot default, the reviewable diff, unchanged in role.
2. **Env at start** — what the unit file sets, read once as today.
3. **Runtime `SET`** — a temporary override, lost on restart.

A `SET` never writes to disk. There is deliberately no "persist this" verb: the way to
make a setting permanent is to edit the unit file and deploy, and that should stay
slightly inconvenient.

---

## 4. Risks

### 4.1 A forgotten override silently changes later takes

The mitigation is that it cannot survive a restart, and that every take records its own
achieved values. A sweep script should still restore explicitly rather than rely on it.

### 4.2 Why "dies on restart" is the central property

An override that persisted would recreate the 300 mW failure exactly: invisible device
state, no diff anywhere, and a dated characterization silently invalidated. The
inconvenience of re-applying a setting after a restart is the feature.

### 4.3 The auto-pull timer will wipe an override mid-experiment

The Jetson pulls every ~2 minutes and restarts the camera when `server/` changed. During a
sweep that is a live hazard. Options: have the sweep tool check the greeting between arms
(cheap, catches it after the fact), or pause the timer for the duration (surer, more
moving parts). **Open question, §6.**

### 4.4 Multi-client

The server is unicast, but nothing stops a second client connecting and issuing `SET`
under the first. The generation bump makes this loud rather than silent — the first
client's session ends. That is the right behaviour, but it should be a documented
behaviour rather than a surprise.

---

## 5. Testing

The host suite stubs `pyrealsense2` with a bare namespace, and `rs_config` takes `rs` as a
parameter precisely so it imports there. `tests/test_server_env.py` already grew a
`_FakeFilter` with a `get_option` SDK-defaults table for the `smooth_delta` work; extend
that rather than inventing a second fake.

Cases that matter, in TDD order:

1. `SET` with no arguments reads back the current chain without changing it.
2. `SET spatial=0` removes the filter, and the next greeting says so in BOTH `filters` and
   `filter_options`.
3. Every knob in §2.2 round-trips: set, read back, appears in the greeting.
4. An unknown key is rejected with an error line, not silently ignored — unknown *commands*
   are forgiving by design, but an unknown *setting* means the caller thinks it changed
   something it did not.
5. An out-of-range value is clamped and the clamped value reported (mirrors
   `_set_with_readback`).
6. A `SET` bumps the generation and ends a session greeted under the old one.
7. Restart discards a runtime override and returns to the unit file's value.
8. Every greeting already on disk still parses — the archive walk in
   `tests/test_depth_geometry.py` extends to the new fields.

---

## 6. Open questions

These need an answer before implementation. They are listed rather than silently decided.

1. **Auto-pull during a sweep (§4.3)** — check-between-arms, or pause the timer? Leaning
   check-between-arms: fewer moving parts, and it also catches an unrelated restart.
2. **Should `SET` be refused while a burst session is open**, rather than ending it? Ending
   it is safe but costs the operator a re-tour. Refusing is politer but adds a state
   machine. Leaning end-it, because it reuses machinery that already exists.
3. **Does the sweep want a driver tool** (`tools/rs_sweep.py`: set, capture, restore,
   repeat), or is a manual loop enough for the two or three sweeps actually planned?
   Leaning no tool until a second sweep is actually requested.
4. **Do the threshold min/max belong in the safe tier?** They are host-side and harmless
   mechanically, but they silently bound every measurement's valid range, so a forgotten
   override is more consequential than a filter parameter. Possibly worth excluding.

---

## 7. Out of scope, deliberately

- **Device options and advanced mode (§2.3, §2.4).** Adding them later requires: every
  knob recorded in the greeting; a documented restore-on-start so a runtime change cannot
  outlive the session; and the `load_json` trap handled — the committed as-found preset
  carries `depthunits=1000` and loading it would silently undo the 0.1 mm depth units.
- **Stream geometry (§2.5).** A different operation, needing a pipeline rebuild.
- **Any host-side setting.** `tasni.config.json` already has its own reload story.
- **Authentication on `SET`.** The camera is on the cell LAN with no auth on any other
  command either; adding it here alone would be theatre.

---

## 8. What this unblocks

Immediately: the §3.1 spatial A/B in the roll-probe handoff, without an agent driving the
Jetson between arms, and the `smooth_delta` sweep behind it.

Later, and more valuable: the depth-control confidence thresholds (§2.4) become a
half-hour experiment instead of a project — and if the roll probe comes back SCENE-LOCKED,
those thresholds are the most likely place the layer-2 dropout actually lives.
