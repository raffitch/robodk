"""RoboDK item I/O — the only module that knows ``robolink``/``robomath``.

Wraps a :class:`~tasni.core.session.RdkSession` with the small set of cell
operations modules need: list targets, read poses, move the robot, set the run
mode, read/write tool poses. Poses cross the boundary as plain numpy 4x4
matrices (mm + rotation), so downstream module code never touches ``robomath``.
"""
from __future__ import annotations

import numpy as np

from .session import RdkSession


def pose_to_T(pose) -> np.ndarray:
    """RoboDK ``robomath.Mat`` -> numpy 4x4 homogeneous transform."""
    return np.array(pose.Rows(), dtype=float)


def T_to_pose(T: np.ndarray):
    """numpy 4x4 -> RoboDK ``robomath.Mat``."""
    import robodk.robomath as robomath

    return robomath.Mat(np.asarray(T, dtype=float).tolist())


class RdkIO:
    """Cell operations on top of an :class:`RdkSession`."""

    # RoboDK RUNMODE constants (avoids importing robolink just for the enum).
    RUNMODE_SIMULATE = 1
    RUNMODE_RUN_ROBOT = 6

    def __init__(self, session: RdkSession):
        self.session = session
        self._frame = None        # active reference frame item (set by use_tool_and_frame)

    @property
    def rdk(self):
        return self.session.rdk

    def robot(self):
        return self.rdk.Item(self.session.config.robot_name)

    # -- items / existence --------------------------------------------------
    def item_exists(self, name: str) -> bool:
        return self.rdk.Item(name).Valid()

    def use_tool_and_frame(self, tool_name: str, frame_of_target: str | None = None
                           ) -> np.ndarray:
        """Make ``tool_name`` the active tool (and, if given, adopt the reference
        frame that ``frame_of_target`` is defined in). Returns the tool's mounting
        pose (flange->tool) as numpy 4x4. Raises if the tool is missing."""
        import robolink

        tool = self.rdk.Item(tool_name, robolink.ITEM_TYPE_TOOL)
        if not tool.Valid():
            raise RuntimeError(f"tool {tool_name!r} not found in the station")
        robot = self.robot()
        robot.setPoseTool(tool)
        self._frame = None
        if frame_of_target:
            target = self.rdk.Item(frame_of_target, robolink.ITEM_TYPE_TARGET)
            if target.Valid():
                frame = target.Parent()
                if frame.Valid() and frame.Type() == robolink.ITEM_TYPE_FRAME:
                    robot.setPoseFrame(frame)
                    self._frame = frame
        return pose_to_T(tool.PoseTool())

    def use_camera_tool(self, tool_name: str) -> np.ndarray:
        """Activate ``tool_name`` and adopt the robot's base reference frame.

        With no taught NEUTRAL target, the live-gate seed is read from — and the
        generated calibration targets are written into — the robot's base frame,
        so seed pose, generated poses, IK checks and target creation all share one
        unambiguous frame. Returns the tool mounting pose (flange->tool) as 4x4."""
        import robolink

        tool = self.rdk.Item(tool_name, robolink.ITEM_TYPE_TOOL)
        if not tool.Valid():
            raise RuntimeError(f"tool {tool_name!r} not found in the station")
        robot = self.robot()
        robot.setPoseTool(tool)
        base = robot.Parent()       # the reference frame the robot is attached to
        if base.Valid() and base.Type() == robolink.ITEM_TYPE_FRAME:
            robot.setPoseFrame(base)
            self._frame = base
        else:
            self._frame = None      # AddTarget(None) / Pose() then use the base frame
        return pose_to_T(tool.PoseTool())

    # -- poses --------------------------------------------------------------
    def tcp_pose_T(self) -> np.ndarray:
        """Current TCP pose in the active reference frame (numpy 4x4)."""
        return pose_to_T(self.robot().Pose())

    def current_joints(self):
        """Current joint vector (RoboDK ``Mat``) — snapshot for a safe return."""
        return self.robot().Joints()

    def move_j_joints(self, joints) -> None:
        """MoveJ to a joint vector (avoids IK/elbow ambiguity of a cartesian move)."""
        self.robot().MoveJ(joints)

    def move_j_pose(self, T: np.ndarray) -> None:
        self.robot().MoveJ(T_to_pose(T))

    def is_reachable(self, T: np.ndarray) -> bool:
        """True if the robot has an IK solution for this TCP pose (active tool/frame)."""
        try:
            sol = self.robot().SolveIK(T_to_pose(T))
        except Exception:
            return False
        try:
            return len(list(sol)) >= 6 or np.asarray(sol.list()).size >= 6
        except Exception:
            return np.asarray(sol).size >= 6

    # -- temporary targets (auto pose generation) ---------------------------
    def add_target(self, name: str, T: np.ndarray):
        """Create a cartesian target at pose ``T`` in the active frame; return it."""
        frame = getattr(self, "_frame", None)
        target = self.rdk.AddTarget(name, frame, self.robot())
        target.setPose(T_to_pose(T))
        return target

    def delete_items(self, names: list[str]) -> None:
        for name in names:
            item = self.rdk.Item(name)
            if item.Valid():
                item.Delete()

    def apply_run_mode(self, mode: str | None = None) -> str:
        """Push the run mode (``"simulate"`` or ``"run_robot"``). ``mode`` overrides
        the configured default. Returns the mode that was applied."""
        mode = mode or self.session.config.run_mode
        self.rdk.setRunMode(self.RUNMODE_RUN_ROBOT if mode == "run_robot"
                            else self.RUNMODE_SIMULATE)
        return mode

    def current_run_mode(self) -> int:
        """Current RoboDK run mode (raw int) — captured before a dry run so the
        prior mode can be restored afterwards (a dry tour must never leave the
        station silently in RUN_ROBOT)."""
        return int(self.rdk.RunMode())

    def set_run_mode_raw(self, value: int) -> None:
        """Restore a previously captured raw run-mode value (see
        :meth:`current_run_mode`)."""
        self.rdk.setRunMode(int(value))

    def set_collision_checking(self, active: bool) -> bool:
        """Best-effort toggle of RoboDK collision checking. Returns True if the
        build accepted the call (collisions can be reported), False otherwise.
        Never raises — collision checking is an optional bonus on the dry tour."""
        try:
            import robolink

            flag = robolink.COLLISION_ON if active else robolink.COLLISION_OFF
            self.rdk.setCollisionActive(flag)
            return True
        except Exception:
            return False

    def collisions(self) -> int | None:
        """Number of colliding object pairs in the current (simulated) state, or
        ``None`` if this build/station can't check collisions. Best-effort; never
        raises so the dry tour degrades gracefully where collisions aren't set up."""
        try:
            return int(self.rdk.Collisions())
        except Exception:
            return None

    def list_targets(self, prefix: str | None = None) -> list[str]:
        """Sorted names of TARGET items, filtered by ``prefix``."""
        import robolink

        prefix = self.session.config.target_prefix if prefix is None else prefix
        items = self.rdk.ItemList(robolink.ITEM_TYPE_TARGET)
        return sorted(i.Name() for i in items if i.Name().startswith(prefix))

    def target_pose_T(self, name: str) -> np.ndarray:
        """Target pose as numpy 4x4 (robot flange/gripper in base frame)."""
        import robolink

        item = self.rdk.Item(name, robolink.ITEM_TYPE_TARGET)
        return pose_to_T(item.Pose())

    def move_j(self, name: str) -> None:
        import robolink

        target = self.rdk.Item(name, robolink.ITEM_TYPE_TARGET)
        self.robot().MoveJ(target)

    def list_tools(self) -> list[str]:
        import robolink

        items = self.rdk.ItemList(robolink.ITEM_TYPE_TOOL)
        return [i.Name() for i in items]

    def set_tool_pose(self, tool_name: str, T: np.ndarray) -> None:
        import robolink

        tool = self.rdk.Item(tool_name, robolink.ITEM_TYPE_TOOL)
        tool.setPoseTool(T_to_pose(T))

    def get_tool_pose_T(self, tool_name: str) -> np.ndarray:
        import robolink

        tool = self.rdk.Item(tool_name, robolink.ITEM_TYPE_TOOL)
        return pose_to_T(tool.PoseTool())
