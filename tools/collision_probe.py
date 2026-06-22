"""Diagnose the calibration collision guard against the LIVE RoboDK station.

Run this with RoboDK open on the Tasni cell (and, ideally, the TasniCalib_*
targets already created) — it ATTACHES to that GUI, so you can see exactly what
the app sees:

    python tools/collision_probe.py

It reports, without changing your station on disk:
  * the robot, its DOF and the link ids the guard checks,
  * every flange-mounted tool/object the guard discovers (the camera, the
    spindle, …) — if your spindle is NOT listed, that is why its collision was
    never filtered,
  * for each TasniCalib_* target: whether its stored configuration collides once
    the tool<->arm pairs are enabled, and WHICH items collide (so a spindle-vs-arm
    overlap shows up as the spindle + the robot in the colliding set).

It restores the robot's joints, run mode and collision state when it finishes.
Motion is SIMULATE-only (no hardware moves).
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

from tasni.core.config import RoboDKConfig, CalibrationConfig  # noqa: E402
from tasni.core.session import RdkSession  # noqa: E402
from tasni.core.rdk_io import RdkIO  # noqa: E402
from tasni.core.geometry import invert_T  # noqa: E402

TARGET_PREFIX = "TasniCalib_"


def _names(items) -> list[str]:
    out = []
    for it in items:
        try:
            out.append(it.Name())
        except Exception:
            out.append("<?>")
    return out


def main() -> None:
    cfg = RoboDKConfig(connection="attach")     # bind the running GUI
    ccfg = CalibrationConfig()
    print("Attaching to the running RoboDK… (open the Tasni cell first)")

    with RdkSession(cfg) as session:
        io = RdkIO(session)
        rdk = session.rdk
        import robolink

        robot = rdk.Item(cfg.robot_name, robolink.ITEM_TYPE_ROBOT)
        if not robot.Valid():
            print(f"!! robot {cfg.robot_name!r} not found — is the Tasni station open?")
            sys.exit(1)

        dof = io.robot_dof()
        skip = ccfg.collision_skip_wrist_links
        last_link = max(0, dof - skip)
        print(f"\n=== robot ===\n{cfg.robot_name}  DOF={dof}")
        print(f"guard checks links 0..{last_link} (skip_trailing={skip} -> "
              f"skips the top {skip} link(s): wrist/flange)")

        # --- camera tool frame: is the offset real, or is "TCP" the flange? ---
        io.use_camera_tool(cfg.camera_tool)
        tool_off = np.asarray(io.get_tool_pose_T(cfg.camera_tool))   # flange->tool
        off_mm = float(np.linalg.norm(tool_off[:3, 3]))
        cam_tcp = io.tcp_pose_T()                                     # camera TCP, base
        flange = cam_tcp @ invert_T(tool_off)
        print(f"\n=== camera tool frame ===")
        print(f"{cfg.camera_tool} mount offset from flange: {off_mm:.1f} mm")
        print(f"  camera TCP (base mm): [{cam_tcp[0,3]:.1f}, {cam_tcp[1,3]:.1f}, {cam_tcp[2,3]:.1f}]")
        print(f"  flange     (base mm): [{flange[0,3]:.1f}, {flange[1,3]:.1f}, {flange[2,3]:.1f}]")
        if off_mm < 15.0:
            print("  !! offset is ~identity -> the camera 'TCP' sits AT the flange, so")
            print("     generated poses orbit the FLANGE, not the camera view. Set the")
            print(f"     {cfg.camera_tool} tool's approximate mount in RoboDK (or apply a")
            print("     prior calibration), then re-create targets. THIS is the cause.")
        else:
            print("  (camera TCP is genuinely offset from the flange -> poses are camera-relative)")

        # --- what the guard discovers --------------------------------------
        mounted = io.mounted_tool_items()
        print(f"\n=== mounted bodies discovered (guarded vs the arm) ===")
        for n in _names(mounted):
            print(f"  - {n}")
        if not mounted:
            print("  (NONE — no tool/object found on the robot; nothing can be guarded!)")
        all_tools = _names(rdk.ItemList(robolink.ITEM_TYPE_TOOL))
        print(f"all TOOL items in station: {all_tools}")

        # --- enable the pairs (as Create targets does) ---------------------
        prior_mode = io.current_run_mode()
        rdk.setRunMode(io.RUNMODE_SIMULATE)
        try:
            start_joints = robot.Joints()
        except Exception:
            start_joints = None
        io.set_collision_checking(True)
        guard = io.ensure_mounted_tool_collision_pairs(skip)
        print(f"\n=== guard enable result ===\n{guard}")

        # --- check each created target -------------------------------------
        targets = io.list_targets(TARGET_PREFIX)
        print(f"\n=== {len(targets)} {TARGET_PREFIX}* targets ===")
        if not targets:
            print("  (none created yet — create targets in the app, then re-run)")
        print("  (cam = where the Realsense TCP lands; flange = the wrist behind it.")
        print("   Correct = cam positions vary AROUND the board; flange is offset behind.)")
        for name in targets:
            j = io.target_joints(name)
            verdict, colliders, posns = "?", [], ""
            try:
                if j is not None:
                    robot.setJoints(j)
                else:
                    io.move_j(name)
                # Read where the CAMERA and the FLANGE actually are at this config —
                # independent of whatever tool is active in the GUI display.
                cam = io.tcp_pose_T()
                fl = cam @ invert_T(tool_off)
                posns = (f"  cam[{cam[0,3]:.0f},{cam[1,3]:.0f},{cam[2,3]:.0f}]"
                         f" flange[{fl[0,3]:.0f},{fl[1,3]:.0f},{fl[2,3]:.0f}]")
                n = rdk.Collisions()
                verdict = f"{int(n)} colliding pair(s)"
                if int(n) > 0:
                    colliders = _names(rdk.CollisionItems())
            except Exception as e:
                verdict = f"<error: {e}>"
            tag = "JOINT" if j is not None else "cartesian"
            extra = f"  <- {colliders}" if colliders else ""
            print(f"  {name} [{tag}]: {verdict}{posns}{extra}")

        # --- restore --------------------------------------------------------
        try:
            if start_joints is not None:
                robot.setJoints(start_joints)
        except Exception:
            pass
        io.set_collision_checking(False)
        io.set_run_mode_raw(prior_mode)
        print("\nRestored joints / run mode / collision checking. Done.")
        print("\nHints:")
        print("  * If your spindle is NOT in 'mounted bodies discovered', it isn't a")
        print("    TOOL or an object under the robot — tell me how it's mounted.")
        print("  * If a target shows 0 collisions here but visibly overlaps A4 in the")
        print("    GUI, the colliding link is above the guarded range — lower")
        print("    calibration.collision_skip_wrist_links (e.g. to 1) and re-run.")


if __name__ == "__main__":
    main()
