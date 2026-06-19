# Best-Practices Review — Scanning Cell (early 2026)

Deep-research review of the fixed stack (Jetson Nano + D435i + KUKA via RoboDK) against
current robotics / 3D-scanning practice. 24 sources, 25 claims adversarially verified
(23 confirmed, 2 refuted). Hardware is fixed; RoboDK stays the orchestrator.

## TL;DR — do these first
1. **TSDF fusion** instead of voxel-downsample concatenation — biggest scan-quality win.
2. **Report calibration quality** (reprojection error + held-out validation poses); add a
   post-TSAI refinement. Software-only.
3. **RealSense High Accuracy preset** + Intel's disparity-domain filter order.
4. **Do NOT adopt ROS.** Keep RoboDK + bespoke transport (hardened).

---

## 1. ROS — verdict: do NOT adopt (hard ceiling) ✓ 3-0
The original Jetson Nano is permanently locked to **JetPack 4.6 / Ubuntu 18.04** — NVIDIA
officially stated **no JetPack 5** for Nano/TX1/TX2 (JetPack 4 EOL, 4.6.6 final, Nov 2024,
sustaining mode only). **ROS 1 Noetic** is **EOL since 2025-05-31** and targets Ubuntu
20.04; **ROS 2** needs 20.04/22.04. Neither has a native fit on 18.04. → **RoboDK remains
sole orchestrator.** This is a hardware ceiling, not a choice.
- Sources: [NVIDIA forum](https://forums.developer.nvidia.com/t/with-jetson-nano-dev-kits-running-until-2027-will-jetpack-5-be-available-for-jetson-nano/233655),
  [ROS Noetic EOL](https://discourse.openrobotics.org/t/ros-noetic-end-of-life-may-31-2025/43160)

## 2. Calibration — TSAI is fine; the gap is measurement
- **Keep TSAI** as the solver. ⚠️ The idea that OpenCV's **PARK is universally more robust
  was REFUTED (1-2)** — do **not** blindly switch solvers.
- **QUICK WIN: report quality.** On a real arm the true hand-eye transform is unknowable,
  so **reprojection error (px)** is the standard proxy — report it plus **held-out
  validation-pose errors**. The cell currently reports nothing. ✓ 3-0
- **QUICK WIN: post-solve refinement.** Add an iterative optimization minimizing
  reprojection error on top of TSAI. One study cut avg reprojection error **44.43%** vs
  Tsai-Lenz (medium confidence — single study, best-case baseline; expect less).
- Context: a ChArUco eye-in-hand method hit **0.4–0.6 mm**, but on a **5MP GigE** camera,
  not a D435i — **not transferable**. The **D435i's distance-squared depth noise is the
  real accuracy ceiling**, not the calibration method.
- Sources: [Sensors 2024 (PMC10780872)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10780872/),
  [OpenCV #24871](https://github.com/opencv/opencv/issues/24871)

## 3. Reconstruction — TSDF is the highest-leverage change ✓ 3-0
- **QUICK WIN: replace concatenation with Open3D TSDF integration**
  (`ScalableTSDFVolume` / `VoxelBlockGrid`). It takes exactly the inputs you already have
  (RGBD + intrinsics + **per-view robot pose as extrinsic** + depth_scale + depth_max) and
  acts as a **3D weighted-average filter** → smoother, denoised meshes. Runs on the
  existing workstation. *This is the single biggest scan-quality improvement available.*
- **STRUCTURAL (next): pose-graph / multiway registration** (colored-ICP edges +
  `GlobalOptimizationLevenbergMarquardt`) before fusion, to correct residual hand-eye /
  robot-pose error across the dome. Robot poses give a great initialization.
- **NKSR:** keep it OFF the Nano (correct — it's datacenter-GPU class, tens of GB VRAM).
  ⚠️ Claim that NKSR is **well-suited to multi-view RGBD fusion was REFUTED (0-3)** — it
  targets large/sparse/noisy clouds. For a clean fused TSDF volume, **marching-cubes or
  Poisson may suffice** — treat NKSR as optional, not central. (2024-25 methods NoKSR /
  SurfR / OffsetOPT now beat it, all still workstation-GPU.)
- Sources: [Open3D RGBD integration](https://www.open3d.org/docs/latest/tutorial/pipelines/rgbd_integration.html),
  [multiway registration](https://www.open3d.org/docs/latest/tutorial/pipelines/multiway_registration.html),
  [NKSR](https://huangjh-pub.github.io/publication/nksr/), [NoKSR](https://arxiv.org/html/2502.12534v1)

## 4. RealSense capture — two software-only wins ✓ 3-0
- **High Accuracy visual preset** — Intel explicitly recommends it for "Object Scanning…
  Robots" (high depth-confidence, fewer hallucinations). If too many holes, ease toward
  Medium Density (accuracy↔fill tradeoff).
- **Filter order in the disparity domain:**
  `Depth → Decimation → Depth2Disparity → Spatial → Temporal → Disparity2Depth → Hole-Filling`
  (noise grows with distance²; disparity space makes it uniform).
- Sources: [visual presets](https://dev.intelrealsense.com/docs/d400-series-visual-presets),
  [post-processing](https://dev.intelrealsense.com/docs/depth-post-processing)

## 5. Transport — harden, don't replace (medium confidence)
Keep the bespoke port-1024 TCP server; add robust length-prefixed framing, auto-reconnect,
heartbeat/health check, explicit error handling. Wholesale replacement (gRPC / RTSP /
ROS 2 image_transport) isn't worth it given the 18.04 lock + Py3.6/3.10 split. *(No primary
citation survived for bespoke-vs-gRPC; reasoned from the hardware constraints.)*

## 6. Ops / architecture
- **Tailscale** for the Nano on a private LAN/NAT: auth-keys enroll headless;
  built-in Tailscale SSH gives remote shell with no port-forwarding → enables single-repo
  deploy + health monitoring. ✓ 3-0 ([Tailscale CLI](https://tailscale.com/kb/1080/cli))
- **systemd over cron** for the server lifecycle (restart-on-crash + on-boot). *(standard
  practice; not separately citation-verified here.)*

---

## Refuted claims (do NOT act on)
- ❌ PARK is universally more robust than TSAI (1-2) → refine, don't swap solvers.
- ❌ NKSR is purpose-fit for multi-view RGBD fusion (0-3) → it targets large/sparse/noisy
  clouds; TSDF marching-cubes / Poisson may be enough.

## Open questions (need our own measurement)
1. **What dominates end-to-end error** — hand-eye calibration, robot absolute accuracy, or
   D435i depth noise? Unknown until we record held-out validation-pose errors. Determines
   whether the pose-graph pass (§3) is even worth adding.
2. **Workstation GPU VRAM / WSL CUDA path** — dictates which reconstruction backend (TSDF
   marching-cubes vs Poisson vs NKSR vs a 2025 successor) is viable. Specs not yet known.
3. **Is the TCP transport actually a bottleneck** in practice, or is hardening premature?
   No failure telemetry yet.
4. Would **containerizing** the camera server simplify the Py3.6/3.10 split on JetPack 4.6?

_Full machine-readable result: workflow run `wf_75068bdb-2b9`._
