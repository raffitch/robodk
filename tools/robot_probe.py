"""Probe the live RoboDK robot and stress-test the calibration pose generator.

Loads Tasni.rdk into a PRIVATE headless RoboDK (never touches your open GUI),
reads the robot's real joint limits / DOF / current pose, then replays
``generate_calibration_poses`` from the robot's *current* TCP pose and reports
how many candidates are actually reachable (the same IK gate service.py uses).

    python tools/robot_probe.py

Output is human-readable + a JSON blob at the end for tooling.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from tasni.core.config import RoboDKConfig, CalibrationConfig  # noqa: E402
from tasni.core.session import RdkSession  # noqa: E402
from tasni.core.rdk_io import RdkIO  # noqa: E402
from tasni.modules.calibration.poses import (  # noqa: E402
    generate_calibration_poses, select_diverse, viewing_angle_span)


def _mat_to_list(m) -> list[float]:
    try:
        return [float(x) for x in m.list()]
    except Exception:
        return [float(x) for x in np.asarray(m.Rows()).ravel()]


def main() -> None:
    cfg = RoboDKConfig(connection="isolated", station_path="Tasni.rdk")
    ccfg = CalibrationConfig()
    out: dict = {"robot": cfg.robot_name}

    print(f"Loading station (headless) — this takes a minute or two...")
    with RdkSession(cfg) as session:
        io = RdkIO(session)
        rdk = session.rdk
        import robolink

        robot = rdk.Item(cfg.robot_name, robolink.ITEM_TYPE_ROBOT)
        if not robot.Valid():
            print(f"!! robot {cfg.robot_name!r} not found in station")
            sys.exit(1)

        # --- robot identity + limits ---------------------------------------
        joints_now = _mat_to_list(robot.Joints())
        dof = len(joints_now)
        lower, upper, jtype = robot.JointLimits()
        lower, upper = _mat_to_list(lower), _mat_to_list(upper)
        out["dof"] = dof
        out["joint_limits_deg"] = [
            {"axis": f"A{i+1}", "min": round(lower[i], 1), "max": round(upper[i], 1),
             "range": round(upper[i] - lower[i], 1), "now": round(joints_now[i], 1)}
            for i in range(dof)]

        print(f"\n=== {cfg.robot_name} ===")
        print(f"DOF: {dof}")
        print(f"{'axis':<5}{'min':>9}{'max':>9}{'range':>9}{'current':>10}")
        for a in out["joint_limits_deg"]:
            print(f"{a['axis']:<5}{a['min']:>9}{a['max']:>9}{a['range']:>9}{a['now']:>10}")

        # --- active tool / frame + current TCP -----------------------------
        tool_T = io.use_camera_tool(cfg.camera_tool)
        out["camera_tool"] = cfg.camera_tool
        out["tool_mount_T"] = tool_T.tolist()
        seed_T = io.tcp_pose_T()
        out["seed_TCP_T"] = seed_T.tolist()
        pos = seed_T[:3, 3]
        print(f"\nActive tool: {cfg.camera_tool}")
        print(f"Seed TCP position (base frame, mm): "
              f"[{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}]")
        print(f"Seed reachable now: {io.is_reachable(seed_T)}")

        # --- replay the generator from the live seed -----------------------
        # EXACTLY as service.py: generate_calibration_poses(count=pose_count)
        # returns pose_count*oversample candidates (oversample=3 -> 45). Service
        # walks them in order and keeps the first `pose_count` reachable. The
        # spiral sampling density depends on count, so we must pass the real
        # pose_count to get a faithful sequence.
        candidates = generate_calibration_poses(
            seed_T, count=ccfg.pose_count, look_distance_mm=ccfg.ideal_distance_mm,
            cone_half_angle_deg=ccfg.cone_half_angle_deg,
            roll_max_deg=ccfg.roll_max_deg, distance_jitter=ccfg.distance_jitter)

        reach = [io.is_reachable(T) for T in candidates]
        reach_T = [T for T, r in zip(candidates, reach) if r]
        n_reach = len(reach_T)
        seed_fwd = seed_T[:3, 2]

        # OLD behaviour: first `pose_count` reachable (clusters at narrow cone).
        old_kept = reach_T[: ccfg.pose_count]
        _, old_max, old_mean = viewing_angle_span(old_kept, seed_fwd)
        # NEW behaviour: diverse spread (FPS) over reachable candidates.
        sel = select_diverse(reach_T, min(ccfg.pose_count, n_reach), seed_fwd=seed_fwd)
        new_kept = [reach_T[k] for k in sel]
        _, new_max, new_mean = viewing_angle_span(new_kept, seed_fwd)

        out["generator"] = {
            "pose_count_target": ccfg.pose_count,
            "candidates_generated": len(candidates),
            "candidates_reachable": n_reach,
            "reachable_pct": round(100 * n_reach / len(candidates), 1),
            "kept": len(new_kept),
            "holdout_count": ccfg.holdout_count,
            "train_after_holdout": max(0, len(new_kept) - ccfg.holdout_count),
            "cone_half_angle_deg": ccfg.cone_half_angle_deg,
            "first_n_cone_max_deg": round(old_max, 1),
            "first_n_cone_mean_deg": round(old_mean, 1),
            "diverse_cone_max_deg": round(new_max, 1),
            "diverse_cone_mean_deg": round(new_mean, 1),
        }

        print(f"\n=== generator replay from live seed ===")
        print(f"target pose_count          : {ccfg.pose_count}")
        print(f"candidates generated       : {len(candidates)} "
              f"(= pose_count x oversample 3)")
        print(f"candidates reachable       : {n_reach} "
              f"({out['generator']['reachable_pct']}%)")
        print(f"configured cone half-angle : {ccfg.cone_half_angle_deg:.0f} deg")
        print(f"OLD first-{ccfg.pose_count} reachable: effective cone "
              f"max {old_max:.0f} / mean {old_mean:.0f} deg")
        print(f"NEW diverse select        : effective cone "
              f"max {new_max:.0f} / mean {new_mean:.0f} deg")
        print(f"=> diversity gain          : +{new_max - old_max:.0f} deg max, "
              f"+{new_mean - old_mean:.0f} deg mean")
        print(f"kept                       : {len(new_kept)}")
        print(f"holdout                    : {ccfg.holdout_count}")
        print(f"=> train poses after holdout: "
              f"{out['generator']['train_after_holdout']}")
        if len(new_kept) < ccfg.pose_count:
            print(f"!! note: only {len(new_kept)} reachable < target "
                  f"{ccfg.pose_count} — seed is in a tight workspace region")
        if new_max < 0.5 * ccfg.cone_half_angle_deg:
            print(f"!! WARNING: effective cone < half configured — narrow diversity")

    print("\n--- JSON ---")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
