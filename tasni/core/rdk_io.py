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

    @property
    def rdk(self):
        return self.session.rdk

    def robot(self):
        return self.rdk.Item(self.session.config.robot_name)

    def apply_run_mode(self, mode: str | None = None) -> str:
        """Push the run mode (``"simulate"`` or ``"run_robot"``). ``mode`` overrides
        the configured default. Returns the mode that was applied."""
        mode = mode or self.session.config.run_mode
        self.rdk.setRunMode(self.RUNMODE_RUN_ROBOT if mode == "run_robot"
                            else self.RUNMODE_SIMULATE)
        return mode

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
