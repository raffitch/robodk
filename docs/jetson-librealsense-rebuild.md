# Rebuilding librealsense on the Jetson as Release + CUDA + OpenMP (audit R1)

**Done: 2026-08-29.** This is the runbook as it was actually executed, with the measured
before/after numbers and the two places the original plan did not match the device.

Background: [realsense-capability-audit-2026-08-29.md](realsense-capability-audit-2026-08-29.md)
R1. The service was loading an **unoptimised debug build of 2.53.1** — no `-O` of any kind,
no CUDA, no OpenMP — which is why `align` + `spatial` "cost about a second per frame" and
why the architecture grew a colour-only fast path, burst capture and an H.264 preview to
work around it.

This task is **infra only, zero app code**. Its acceptance is that the **old** wire protocol
still serves *identically* afterwards — that is what makes it provable independently of the
protocol-2 change that follows it.

---

## Result

| | BEFORE (`build_py310`, 2.53.1) | AFTER (`build_cuda`, 2.55.1) |
|---|---|---|
| Idle service CPU, no client | **178.9 %** | **28.2 %** |
| `grab(with_depth=True)`, 10 grabs | **7688 ms** | **1055 ms** |
| `grab(with_depth=True)`, 5 grabs | 7551 ms | — |
| `grab(color_only=True)`, 5 grabs | **1273 ms** | **220 ms** |
| Depth path alone (depth grab − colour grab) | ~6.3 s/frame | ~0.8 s/frame |
| `CXX_FLAGS` | `-g`, **no `-O`**, no OpenMP, `gnu++11` | `-O3 -DNDEBUG -fopenmp`, `gnu++14` |
| CUDA | absent | `-DRS2_USE_CUDA`, nvcc 10.2.300 |

Measured from the workstation over the **LAN** path (10.12.171.70), not the Tailscale relay.

**Behaviour is unchanged, which is the point:** depth still arrives as `uint16 (720, 1280)`
aligned to a `(720, 1280, 3)` colour frame; valid fraction 0.9994 → 0.9995; distinct depth
values in a 120×120 centre patch 11 → 13 (both still 1 mm quantisation, because
`depth_units` has not changed yet — that is protocol 2's job, not this task's).

### Geometry sanity

`tools/characterize_distance.py`, two stops, against the 2026-08-13 baseline.

The comparison that matters is the **328 mm** stop, because it reproduces the baseline's
distance *and* its incidence (0.9° against 0.99°). Read it against the baseline's own
**same-distance repeatability**, not against a single row of it — that session happened to
capture the same 310 mm standoff twice, once in the main sweep and once in the incidence
sweep, and the two disagree wildly:

| | `plane_rms_mm` | `plane_max_mm` | `length_err_mm` |
|---|---|---|---|
| baseline @ 310 mm, main sweep | 0.934 | 24.654 | 0.003 |
| baseline @ 310 mm, incidence sweep @ 0.99° | 0.650 | 2.594 | 0.036 |
| **after the rebuild, @ 328 mm, 0.9°** | **0.786** | **5.164** | **0.023** |

Every metric falls *between* the baseline's own two captures, at 18 mm further out where
slightly more noise is expected. The cell's fixed-distance scatter was ±44 % in plane RMS
and ~10× in `plane_max`, so this is as close to "unchanged" as this rig can demonstrate.
`height_repeat=0.024 mm`, `normal_repeat=0.03°`, `length_spread=0.025 mm`, `coverage=100%`.

A second stop at a measured **449 mm** (incidence 0.8°) gave
`rms=1.255  plane_max=8.408  length_err=0.314  coverage=100%`. Interpolating the baseline
to 449 mm suggests ~1.05 / ~5.1, so that stop reads high — but it is inside the same-distance
scatter above, `plane_max` 8.408 is below the baseline's own worst (24.654), and
`length_err` 0.314 matches 0.316 @ 400 mm and 0.364 @ 498 mm almost exactly.

**`length_err_mm` is the metric that would expose a CUDA rounding change in the depth
geometry** — it is the only one referenced to a known physical length (the board's own
squares) rather than to a plane fit — and it is flat at both stops.

> The tool's `=== verdict === NO distance passed every budget criterion` line is **not** a
> result. `DEFAULT_BUDGET` is a placeholder (see the comment at
> `tools/characterize_distance.py:613`). At the 328 mm stop every criterion passes except
> `max_plane_max_mm=3.0` — which the 2026-08-13 baseline also fails at **all five** of its
> stops (4.085, 6.156, 6.277, 8.105, 24.654). Derive a real budget from
> `achieved_envelope` before gating anything on it.

> `coverage_frac` counts **inner corners detected**, not framing margin, so `coverage=100%`
> does not prove the board was uncropped. And the baseline's `incidence_range_deg` only ever
> reached `[0.99°, 9.14°]` — there is no 25° incidence reference to compare against.

Verified in the binary: 157 CUDA symbols, three GPU kernels compiled
(`cuda-align.cu.o`, `cuda-pointcloud.cu.o`, `cuda-conversion.cu.o`), and `libgomp.so.1`
linked. `cudart` does **not** appear in `ldd` because CMake links `libcudart_static.a`.

---

## Two things the plan did not know about the device

### 1. `/usr/local/cuda` points at a CUDA that cannot run here

```
$ /usr/local/cuda/bin/nvcc --version
/usr/local/cuda/bin/nvcc: /lib/aarch64-linux-gnu/libm.so.6: version `GLIBC_2.29' not found
```

The box is Ubuntu 18.04.6 / L4T R32.7.6 / **glibc 2.27**. A CUDA **12.2** toolkit is
installed and the `update-alternatives` symlink points at it, but its `nvcc` needs
glibc ≥ 2.29. **CUDA 10.2 is also installed and works** (`/usr/local/cuda-10.2/bin/nvcc`,
release 10.2, V10.2.300) — it is JetPack 4.6's own.

Build against 10.2 explicitly rather than repointing the system-wide alternative: the flag
is confined to this build, whereas flipping the alternative is a system-wide side effect on
a shared device.

### 2. `~/librealsense`'s working tree is dirty, and switching it would break the rollback

`git status --porcelain` in `~/librealsense` reports **3292 modified files** — all of them
file-**mode** changes only (`100644 → 100755`, 0 content change), an artefact of how the tree
was originally copied. That blocks `git checkout v2.55.1`. Worse, switching that tree to
2.55.1 would desynchronise `build_py310` — the documented rollback build — from its source.

So build 2.55.1 in a **linked git worktree** instead. `~/librealsense` is never touched.

---

## The build

```sh
# 1. Fetch the tag (the repo is already cloned; this only adds refs).
ssh -i ~/.ssh/jetson_robodk jetson@10.12.171.70 \
  'cd ~/librealsense && nohup git fetch --tags origin > /tmp/lrs_fetch.log 2>&1 & echo started'

# 2. A byte-clean worktree at the tag. Leaves ~/librealsense and build_py310 alone.
ssh ... 'cd ~/librealsense && git worktree add ~/librealsense-2.55.1 v2.55.1'
ssh ... 'cd ~/librealsense-2.55.1 && git describe --tags && git status --porcelain | wc -l'
#   -> v2.55.1 / 0
```

**Free the CPU and RAM first.** The Nano has 4 cores and 3963 MB with ~2 GB of zram and no
swapfile; the camera service alone idles at ~1.8 cores. Stopping it for the build fits `-j2`
comfortably (peak usage stayed ~2.4 GB, swap never touched) and avoids adding a privileged
`/swapfile` to a shared device. The cell's camera is dark for the duration.

```sh
py -3.10 tools/jetson_deploy.py stop
```

```sh
ssh ... 'export PATH=/usr/local/cuda-10.2/bin:$PATH
  export CUDACXX=/usr/local/cuda-10.2/bin/nvcc
  cd ~/librealsense-2.55.1 && mkdir -p build_cuda && cd build_cuda
  nohup cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_WITH_CUDA=true -DBUILD_WITH_OPENMP=true \
    -DBUILD_PYTHON_BINDINGS=true \
    -DPYTHON_EXECUTABLE=/home/jetson/EtherSenseServer/ethenv/bin/python \
    -DBUILD_EXAMPLES=false -DBUILD_GRAPHICAL_EXAMPLES=false -DFORCE_RSUSB_BACKEND=false \
    -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-10.2 > cmake.log 2>&1 &'
```

Check the cache before spending an hour on it:

```sh
ssh ... 'cd ~/librealsense-2.55.1/build_cuda && grep -E "^BUILD_WITH_CUDA|^BUILD_WITH_OPENMP|^CMAKE_BUILD_TYPE|^FORCE_RSUSB_BACKEND|^CUDA_NVCC_EXECUTABLE|^CUDA_VERSION" CMakeCache.txt'
```

Observed, and what to require:

```
BUILD_WITH_CUDA:BOOL=true
BUILD_WITH_OPENMP:BOOL=true
CMAKE_BUILD_TYPE:STRING=Release
FORCE_RSUSB_BACKEND:BOOL=false
CUDA_NVCC_EXECUTABLE:FILEPATH=/usr/local/cuda-10.2/bin/nvcc
CUDA_VERSION:STRING=10.2
```

```sh
ssh ... 'export PATH=/usr/local/cuda-10.2/bin:$PATH
  cd ~/librealsense-2.55.1/build_cuda && nohup make -j2 > build.log 2>&1 & echo started'
```

**~26 minutes** with the service stopped (18:33 → 19:00), 0 error lines.

> Watch for `make` to **exit**, not for `[100%]` in the log. `[100%] Linking CXX shared
> library ... pyrealsense2...so` is printed when the link *starts*; the artifact is 0 bytes
> for about another minute. Poll `pgrep -f 'make -j2'` instead.

Harmless configure warnings: `Could NOT find Udev ... using polling device-watcher`
(the `libudev` *headers* are absent; the runtime lib still resolves via libusb), and
`ADD_LIBRARY for library realsense2 without any source files`.

---

## Re-pointing the venv

```sh
SP=/home/jetson/EtherSenseServer/ethenv/lib/python3.10/site-packages
REL=/home/jetson/librealsense-2.55.1/build_cuda/Release

mkdir -p ~/pyrealsense2_rollback
cp -a $SP/pyrealsense2.so ~/pyrealsense2_rollback/
cp -a $REL/pyrealsense2.cpython-310-aarch64-linux-gnu.so* $SP/
cp -a $REL/pyrsutils.cpython-310-aarch64-linux-gnu.so*    $SP/
mv $SP/pyrealsense2.so ~/pyrealsense2_rollback/pyrealsense2.so.removed-from-site-packages
```

2.55 splits `pyrsutils` out of `pyrealsense2`; copy both. There is **no** need to copy
`librealsense2.so.2.55*` — the binding carries
`RUNPATH=/home/jetson/librealsense-2.55.1/build_cuda/Release`, so it resolves the library
from the build directory. That directory must therefore stay in place.

Verify:

```sh
/home/jetson/EtherSenseServer/ethenv/bin/python -c "import pyrealsense2 as rs; print(rs.__version__, rs.__file__)"
#   2.55.1 .../site-packages/pyrealsense2.cpython-310-aarch64-linux-gnu.so
ldd $SP/pyrealsense2.cpython-310-aarch64-linux-gnu.so | grep realsense
#   librealsense2.so.2.55 => /home/jetson/librealsense-2.55.1/build_cuda/Release/librealsense2.so.2.55
```

```sh
py -3.10 tools/jetson_deploy.py restart
py -3.10 tools/jetson_deploy.py status
```

The journal must show `visual_preset left as-is at 0`,
`laser_power -> requested 150, set 150, device reports 150`, `emitter_enabled ... 1`,
and no tracebacks.

---

## Rollback

**The obvious one-liner is wrong.** The new binding is cpython-ABI-tagged
(`pyrealsense2.cpython-310-aarch64-linux-gnu.so`) and Python's `EXTENSION_SUFFIXES` prefers
it over a plain `pyrealsense2.so`, so restoring the old file alone leaves the new build
still winning. The new files must be **removed**:

```sh
SP=/home/jetson/EtherSenseServer/ethenv/lib/python3.10/site-packages
rm $SP/pyrealsense2.cpython-310-*.so* $SP/pyrsutils.cpython-310-*.so*
cp -a ~/pyrealsense2_rollback/pyrealsense2.so $SP/
sudo systemctl restart realsense-camera
```

`~/librealsense/build_py310/` (2.53.1) and `~/pyrealsense2_rollback/` are both untouched by
this procedure, so that restores the exact previous state. Removing the 2.55.1 worktree
entirely, if ever wanted: `cd ~/librealsense && git worktree remove ~/librealsense-2.55.1`.
