"""RoboDK item I/O — the only module that knows ``robolink``/``robomath``.

Wraps a :class:`~tasni.core.session.RdkSession` with the small set of cell
operations modules need: list targets, read poses, move the robot, set the run
mode, read/write tool poses. Poses cross the boundary as plain numpy 4x4
matrices (mm + rotation), so downstream module code never touches ``robomath``.
"""
from __future__ import annotations

import numpy as np

from .geometry import invert_T
from .session import RdkSession


def pose_to_T(pose) -> np.ndarray:
    """RoboDK ``robomath.Mat`` -> numpy 4x4 homogeneous transform."""
    return np.array(pose.Rows(), dtype=float)


def T_to_pose(T: np.ndarray):
    """numpy 4x4 -> RoboDK ``robomath.Mat``."""
    import robodk.robomath as robomath

    return robomath.Mat(np.asarray(T, dtype=float).tolist())


def link_real_robot(rdk: "RdkIO", cfg) -> dict | None:
    """Best-effort connect the physical robot per a ``RoboDKConfig`` and summarise
    the link for the ``/connect`` response + UI. Returns ``None`` when
    ``connect_robot_on_connect`` is off (so the response field is simply absent).

    Shared by the calibration and scan connect endpoints so both link the
    controller the same way. Never raises (``connect_robot`` is best-effort)."""
    if not getattr(cfg, "connect_robot_on_connect", False):
        return None
    ready, msg = rdk.connect_robot(cfg.robot_ip, timeout_s=cfg.robot_connect_timeout_s)
    params = rdk.robot_connection_params()
    return {"connected": ready, "message": msg, "ip": params.get("ip", ""),
            "configured": bool(params.get("ip"))}


class RdkIO:
    """Cell operations on top of an :class:`RdkSession`."""

    # RoboDK RUNMODE constants (avoids importing robolink just for the enum).
    RUNMODE_SIMULATE = 1
    RUNMODE_RUN_ROBOT = 6
    # RoboDK real-robot driver connection status (robolink ROBOTCOM_*). The link is
    # "ready" (robot movable) only at ROBOTCOM_READY; negative values are
    # not-connected/disconnected/problems, positive values are working/waiting.
    ROBOTCOM_READY = 0

    def __init__(self, session: RdkSession):
        self.session = session
        self._frame = None        # active reference frame item (set by use_tool_and_frame)
        # Mounting pose (flange->tool, 4x4) of the tool last activated. We pass this
        # EXPLICITLY to SolveIK/SolveFK rather than trusting the robot's "active tool"
        # state — in an attach session setPoseTool doesn't reliably make a tool the
        # active TCP, so Pose()/SolveIK would silently fall back to the FLANGE and the
        # generated targets would be flange poses (the camera offset dropped). With
        # this every IK/pose query is anchored to the real camera TCP.
        self._tool_pose: np.ndarray | None = None
        # Pose of the ACTIVE reference frame w.r.t. the robot base, cached at
        # tool/frame activation. None = the active frame IS the base. SolveIK's
        # pose math is client-side against the robot base and ignores the
        # station's active frame, so _solve_ik multiplies this in first.
        self._frame_wrt_base_T: np.ndarray | None = None

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
        self._frame_wrt_base_T = None
        if frame_of_target:
            target = self.rdk.Item(frame_of_target, robolink.ITEM_TYPE_TARGET)
            if target.Valid():
                frame = target.Parent()
                if frame.Valid() and frame.Type() == robolink.ITEM_TYPE_FRAME:
                    robot.setPoseFrame(frame)
                    self._frame = frame
                    self._frame_wrt_base_T = self._frame_pose_wrt_base(frame)
        self._tool_pose = pose_to_T(tool.PoseTool())
        return self._tool_pose

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
            self._frame_wrt_base_T = None   # the active frame IS the base
        else:
            self._frame = None      # AddTarget(None) / Pose() then use the base frame
            self._frame_wrt_base_T = None
        self._tool_pose = pose_to_T(tool.PoseTool())
        return self._tool_pose

    # -- poses --------------------------------------------------------------
    def tcp_pose_T(self) -> np.ndarray:
        """Current TCP pose in the active reference frame (numpy 4x4)."""
        return pose_to_T(self.robot().Pose())

    def flange_pose_T(self) -> np.ndarray:
        """Current **flange** pose in the active reference frame, derived from the
        active TCP and its tool offset — so it is the flange regardless of which tool
        (camera, flange, spindle) RoboDK currently has active. ``Pose() @
        inv(PoseTool())`` = (base->TCP) @ (TCP->flange) = base->flange."""
        robot = self.robot()
        return pose_to_T(robot.Pose()) @ invert_T(pose_to_T(robot.PoseTool()))

    def camera_pose_T(self) -> np.ndarray:
        """Current **camera** (Realsense TCP) pose in the active frame, computed
        explicitly as ``flange @ camera_mount`` from the last-activated tool's
        mounting offset (:attr:`_tool_pose`). This is the camera's pose even when the
        active TCP is the flange — the whole point: the generated targets orbit (and
        IK is solved for) the camera, never the flange. Falls back to the raw active
        TCP if no camera tool has been activated yet."""
        if self._tool_pose is None:
            return self.tcp_pose_T()
        return self.flange_pose_T() @ self._tool_pose

    def current_joints(self):
        """Current joint vector (RoboDK ``Mat``) — snapshot for a safe return.

        Once :meth:`connect_robot` has linked the physical controller, RoboDK's
        driver mirrors the real arm into the model, so this is the live robot
        position (the seed Create-targets orbits, and the start the run returns to)."""
        return self.robot().Joints()

    # -- real-robot driver link --------------------------------------------
    def robot_connection_params(self) -> dict:
        """The real-controller connection params stored on the robot item — the
        IP/port RoboDK's "Connect robot" panel shows. ``{"ip": "", ...}`` when
        none is configured. Never raises."""
        try:
            ip, port, *_ = self.robot().ConnectionParams()
            return {"ip": str(ip or ""), "port": int(port)}
        except Exception:
            return {"ip": "", "port": 0}

    def robot_connected(self) -> tuple[bool, str]:
        """``(ready, message)`` for the link to the PHYSICAL robot controller.
        ``ready`` is True only when RoboDK reports ``ROBOTCOM_READY`` (driver
        linked, arm movable). Never raises — a build with no driver, or a transient
        socket error, simply reads as not-ready."""
        try:
            status, msg = self.robot().ConnectedState()
            return int(status) == self.ROBOTCOM_READY, str(msg or "")
        except Exception as e:
            return False, str(e)

    def connect_robot(self, ip: str = "", *, timeout_s: float = 10.0,
                      poll_s: float = 0.4) -> tuple[bool, str]:
        """Link RoboDK to the PHYSICAL robot via its driver and poll until the
        controller reports ready (or ``timeout_s`` elapses). ``ip`` blank uses the
        IP stored on the robot item (RoboDK's connection panel).

        Returns ``(ready, message)`` and **never raises** — the controller may be
        off, so the caller decides whether that's fatal (a real run) or merely
        informational (the connect status chip). Idempotent: if the link is already
        ready it returns immediately. No motion — this only establishes the driver
        link and starts position monitoring (which makes the model track the arm)."""
        import time

        robot = self.robot()
        ready, msg = self.robot_connected()
        if ready:
            return True, msg
        try:
            robot.Connect(ip, blocking=False)   # initiate; we poll ConnectedState
        except Exception as e:
            return False, f"driver connect failed: {e}"
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        while True:
            ready, msg = self.robot_connected()
            if ready or time.monotonic() >= deadline:
                return ready, msg
            time.sleep(max(0.05, float(poll_s)))

    def move_j_joints(self, joints) -> None:
        """MoveJ to a joint vector (avoids IK/elbow ambiguity of a cartesian move)."""
        self.robot().MoveJ(joints)

    def move_j_pose(self, T: np.ndarray) -> None:
        self.robot().MoveJ(T_to_pose(T))

    def _solve_ik(self, T: np.ndarray, seed=None):
        """Raw ``SolveIK`` for a pose, with the camera tool passed **explicitly**.

        ``T`` is the pose of the camera (the last-activated tool's TCP) in the
        **active reference frame**; it is converted to robot-base coordinates here
        (``_frame_wrt_base_T``) because SolveIK's client-side math is base-frame-only.
        Passing the tool mount (``_tool_pose``) as SolveIK's ``tool``
        argument makes the result place the CAMERA — not the flange — at ``T``,
        independent of whatever tool RoboDK thinks is active. ``seed`` (joints_approx)
        pins a deterministic IK branch. May raise; callers wrap as needed."""
        if self._frame_wrt_base_T is not None:
            T = self._frame_wrt_base_T @ np.asarray(T, dtype=float)
        robot = self.robot()
        pose = T_to_pose(T)
        tool = T_to_pose(self._tool_pose) if self._tool_pose is not None else None
        if seed is not None and tool is not None:
            return robot.SolveIK(pose, seed, tool)
        if seed is not None:
            return robot.SolveIK(pose, seed)
        if tool is not None:
            return robot.SolveIK(pose, None, tool)
        return robot.SolveIK(pose)

    def _solve_ik_all(self, T: np.ndarray):
        """Return every IK branch for an active-frame TCP pose, without motion."""
        if self._frame_wrt_base_T is not None:
            T = self._frame_wrt_base_T @ np.asarray(T, dtype=float)
        pose = T_to_pose(T)
        tool = T_to_pose(self._tool_pose) if self._tool_pose is not None else None
        return self.robot().SolveIK_All(pose, tool)

    def is_reachable(self, T: np.ndarray) -> bool:
        """True if the robot has an IK solution placing the **camera** at this pose
        (camera tool passed explicitly — not the flange)."""
        try:
            sol = self._solve_ik(T)
        except Exception:
            return False
        try:
            return len(list(sol)) >= 6 or np.asarray(sol.list()).size >= 6
        except Exception:
            return np.asarray(sol).size >= 6

    def solve_joints_for_pose(self, T: np.ndarray, seed=None):
        """A clean joint vector (RoboDK ``Mat``) that places the **camera** (the
        Realsense tool's TCP) at pose ``T``, or ``None`` if no IK solution exists.

        Used to lock a generated calibration target to a joint configuration so it
        reproduces the camera at the viewpoint when the target is selected/visited —
        the IK is solved with the camera tool passed explicitly (see :meth:`_solve_ik`),
        so the joints drive the camera, NOT the flange, to ``T``. A bare cartesian
        target instead stores only a TCP pose, which RoboDK drives the currently
        active tool to — the "flange visits the TCP" the operator sees.

        Anchors to ``seed`` (the gate/seed joints) for a deterministic IK branch.
        If the seeded solve finds no branch near the seed (a wrist-flipped cone-edge
        pose that nonetheless has *some* solution), it retries seedless from the seed
        config so a pose that is reachable at all still yields a config to lock.
        Never raises."""
        robot = self.robot()
        try:
            if seed is not None:
                sol = self._ik_to_joints(self._solve_ik(T, seed), seed)
                if sol is not None:
                    return sol
                try:
                    robot.setJoints(seed)   # deterministic 'nearest' for the retry
                except Exception:
                    pass
            return self._ik_to_joints(self._solve_ik(T), seed)
        except Exception:
            return None

    def solve_joints_on_neutral_branch(
            self, T: np.ndarray, neutral_joints, previous_joints=None,
            maximum_wrist_rotation_deg: float = 90.0):
        """Solve an extrusion pose without allowing an elbow/wrist branch flip.

        RoboDK's seeded ``SolveIK`` can still select the opposite wrist solution
        after a Curve Follow Project has changed the station's current simulated
        posture. Enumerate all solutions instead, retain the neutral robot
        configuration (front/rear, elbow and wrist-flip flags), bound axes 4 and 6
        relative to neutral, then choose the solution nearest both neutral and the
        preceding path point. This is read-only; no robot joints are set here.
        """
        import robodk.robomath as robomath

        neutral_values = self._joint_values(neutral_joints)
        if neutral_values is None or len(neutral_values) < 6:
            return None
        prior_reference = neutral_joints if previous_joints is None else previous_joints
        previous_values = self._joint_values(prior_reference)
        robot = self.robot()
        try:
            neutral_config = tuple(
                int(round(v)) for v in robot.JointsConfig(neutral_joints).list()[:3])
        except Exception:
            neutral_config = None

        candidates = []
        try:
            candidates = list(self._solve_ik_all(T))
        except Exception:
            pass
        if not candidates:
            fallback = self.solve_joints_for_pose(T, prior_reference)
            if fallback is not None:
                candidates = [fallback]

        accepted: list[tuple[float, object]] = []
        limit = float(maximum_wrist_rotation_deg)
        for candidate in candidates:
            joints = self._ik_to_joints(candidate, neutral_joints)
            if joints is None:
                continue
            joints = self._nearest_equivalent_joints(joints, neutral_joints)
            values = self._joint_values(joints)
            if values is None or len(values) < 6:
                continue
            deltas = [float(values[i] - neutral_values[i]) for i in range(6)]
            if abs(deltas[3]) > limit or abs(deltas[5]) > limit:
                continue
            if neutral_config is not None:
                try:
                    config = tuple(
                        int(round(v)) for v in robot.JointsConfig(joints).list()[:3])
                except Exception:
                    config = neutral_config
                if config != neutral_config:
                    continue
            continuity = 0.0
            if previous_values is not None and len(previous_values) >= 6:
                continuity = sum(
                    (((values[i] - previous_values[i] + 180.0) % 360.0) - 180.0) ** 2
                    for i in range(6))
            neutral_distance = sum(delta * delta for delta in deltas)
            accepted.append((neutral_distance + 0.25 * continuity, joints))
        if not accepted:
            return None
        accepted.sort(key=lambda item: item[0])
        # Re-materialise a plain Mat so no extra configuration columns can leak
        # from SolveIK_All into a program target.
        return robomath.Mat(self._joint_values(accepted[0][1]))

    # -- temporary targets (auto pose generation) ---------------------------
    def add_target(self, name: str, T: np.ndarray, joints=None):
        """Create a target at pose ``T`` in the active frame; return it.

        If ``joints`` (an opaque RoboDK joint vector from :meth:`screen_collisions`)
        is given, the target is stored as a **joint target** locked to that exact
        configuration — so the pose that was collision-checked is the one actually
        visited. Without it, a cartesian target is created and RoboDK may reach it in
        a different (possibly colliding) IK branch."""
        frame = getattr(self, "_frame", None)
        target = self.rdk.AddTarget(name, frame, self.robot())
        target.setPose(T_to_pose(T))
        if joints is not None:
            target.setAsJointTarget()
            target.setJoints(joints)
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

    def robot_dof(self) -> int:
        """Number of robot axes (joint vector length). Falls back to 6 — the
        cell's KUKA — if the live count can't be read. NB: count the joint values
        (``Joints().list()``); ``Mat.Rows()`` returns the row *lists*, not a count."""
        try:
            n = int(np.asarray(self.robot().Joints().list(), dtype=float).size)
            return n if n > 0 else 6
        except Exception:
            return 6

    def mounted_tool_items(self) -> list:
        """Every body that rides on the robot flange: all TOOL items in the station
        (the camera, the spindle, …) plus any OBJECT anywhere under the robot.

        Discovery uses ``ItemList(ITEM_TYPE_TOOL)`` rather than only ``robot.Childs()``
        so a tool is found *regardless of how it is parented* — a spindle that wasn't
        a direct child of the robot is exactly why its arm-collision pairs never got
        enabled and the spindle-into-A4 pose still got created. It then recursively
        walks the robot's descendants to also catch a spindle modelled as an OBJECT
        (attached to a tool or deeper in the subtree). Deduped by name; never raises.
        (Single-robot cell, so every tool rides this arm.)"""
        import robolink

        robot = self.robot()
        found: list = []
        seen: set = set()

        def add(it) -> None:
            try:
                key = it.Name()
            except Exception:
                key = id(it)
            if key not in seen:
                seen.add(key)
                found.append(it)

        # 1) every tool in the station — robust to parentage.
        try:
            for it in self.rdk.ItemList(robolink.ITEM_TYPE_TOOL):
                add(it)
        except Exception:
            pass

        # 2) objects anywhere in the robot's subtree (depth-limited guard).
        def walk(item, depth: int = 0) -> None:
            if depth > 6:
                return
            try:
                kids = item.Childs()
            except Exception:
                return
            for k in kids:
                try:
                    t = k.Type()
                except Exception:
                    continue
                if t == robolink.ITEM_TYPE_OBJECT:
                    add(k)
                if t in (robolink.ITEM_TYPE_OBJECT, robolink.ITEM_TYPE_TOOL):
                    walk(k, depth + 1)

        walk(robot)
        return found

    def ensure_mounted_tool_collision_pairs(self, skip_trailing: int = 2) -> dict:
        """Force-enable collision checking between every flange-mounted body
        (tools + their objects) and the robot's arm links.

        Why this is needed: RoboDK's default collision map EXCLUDES a tool from
        colliding with its own robot (the tool is the robot's child), so a spindle
        swinging into link 4 reports zero collisions and the pose sails through the
        generation filter / dry tour. Re-enabling those pairs is what lets
        :meth:`screen_collisions` and the dry tour actually catch a tool-vs-arm
        self-collision.

        Robot link ids: 0 = base, 1..dof = the moving links (``dof`` = the flange
        the tools bolt to). We enable the tool against links ``0..dof-skip_trailing``
        and skip the trailing ``skip_trailing`` links (the wrist + mounting flange
        the tool naturally sits against) so it isn't reported forever-colliding
        with its own mount. Idempotent, best-effort (modifies the live collision
        map only — never saved to the .rdk), and never raises. Returns a summary
        ``{"tools", "links", "pairs_enabled", "pairs_failed", "dof"}`` for the
        UI/log."""
        import robolink

        robot = self.robot()
        dof = self.robot_dof()
        last_link = max(0, dof - max(0, int(skip_trailing)))
        link_ids = list(range(0, last_link + 1))
        tools = self.mounted_tool_items()
        sibling = self.disable_mounted_body_collision_pairs(tools)
        enabled = failed = 0
        names: list[str] = []
        for it in tools:
            ok_any = False
            for lid in link_ids:
                try:
                    r = self.rdk.setCollisionActivePair(
                        robolink.COLLISION_ON, it, robot, 0, lid)
                    if int(r) == 1:
                        enabled += 1
                        ok_any = True
                    else:
                        failed += 1
                except Exception:
                    failed += 1
            if ok_any:
                try:
                    names.append(it.Name())
                except Exception:
                    pass
        return {"tools": names, "links": link_ids, "pairs_enabled": enabled,
                "pairs_failed": failed, "dof": dof,
                "sibling_pairs_disabled": sibling["pairs_disabled"]}

    def disable_mounted_body_collision_pairs(self, mounted: list | None = None) -> dict:
        """Disable collisions within the rigid flange-mounted assembly.

        Camera, spindle, tool frames and their model objects move as one rigid
        assembly. They may overlap by design and must never be tested against one
        another. Global collision activation can rebuild RoboDK's default map, so
        these exclusions are re-applied every time the calibration guard is enabled.
        """
        import robolink

        bodies = list(self.mounted_tool_items() if mounted is None else mounted)
        disabled = failed = 0
        for i, a in enumerate(bodies):
            for b in bodies[i + 1:]:
                try:
                    r = self.rdk.setCollisionActivePair(
                        robolink.COLLISION_OFF, a, b, 0, 0)
                    if int(r) == 1:
                        disabled += 1
                    else:
                        failed += 1
                except Exception:
                    failed += 1
        return {"pairs_disabled": disabled, "pairs_failed": failed}

    def ensure_obstacle_collision_pairs(self,
                                        ignore_names: list[str] | None = None) -> dict:
        """Force-enable collision pairs between every flange-mounted body (tools +
        their objects) and every static OBJECT in the station (the board pedestal,
        walls, cabinet, floor, …).

        Why: RoboDK's default map omits a tool↔object pair the same way it omits a
        tool↔own-robot pair, so a tool dipping into the board's pedestal reports zero
        collisions and the pose sails through. Re-enabling these lets
        :meth:`screen_collisions` and the dry tour see tool-vs-obstacle contact.
        Enabling *all* tool↔object pairs is safe under baseline-relative screening:
        a pair that's already in contact at the safe seed (e.g. the robot base
        overlapping the pedestal) is recorded in the baseline and subtracted out, so
        only NEW contact is ever acted on. Idempotent, best-effort, never raises.
        Returns ``{"objects", "pairs_enabled", "pairs_failed"}``."""
        import robolink

        tools = self.mounted_tool_items()
        self_pairs = self.disable_mounted_body_collision_pairs(tools)
        mounted_names = set()
        for it in tools:
            try:
                mounted_names.add(it.Name())
            except Exception:
                pass
        # Mounted model objects are returned by ItemList(OBJECT) too. Exclude them:
        # enabling tool↔mounted-object pairs would make the rigid flange assembly
        # collide with itself at every target.
        ignored = {str(n).casefold() for n in (ignore_names or [])}
        objects = []
        for ob in self.rdk.ItemList(robolink.ITEM_TYPE_OBJECT):
            try:
                if ob.Name() in mounted_names or ob.Name().casefold() in ignored:
                    continue
            except Exception:
                pass
            objects.append(ob)
        enabled = failed = 0
        names: list[str] = []
        for ob in objects:
            ok_any = False
            for it in tools:
                try:
                    r = self.rdk.setCollisionActivePair(
                        robolink.COLLISION_ON, it, ob, 0, 0)
                    if int(r) == 1:
                        enabled += 1
                        ok_any = True
                    else:
                        failed += 1
                except Exception:
                    failed += 1
            if ok_any:
                try:
                    names.append(ob.Name())
                except Exception:
                    pass
        return {"objects": names, "pairs_enabled": enabled, "pairs_failed": failed,
                "sibling_pairs_disabled": self_pairs["pairs_disabled"]}

    def disable_object_collision_pairs(self, names: list[str]) -> dict:
        """Disable ignored station objects against the robot and mounted assembly.

        Called after global collision checking is enabled because RoboDK rebuilds
        its default map at that point. Name matching is case-insensitive.
        """
        import robolink

        wanted = {str(n).casefold() for n in names if str(n).strip()}
        if not wanted:
            return {"objects": [], "pairs_disabled": 0, "pairs_failed": 0}
        objects = []
        for ob in self.rdk.ItemList(robolink.ITEM_TYPE_OBJECT):
            try:
                if ob.Name().casefold() in wanted:
                    objects.append(ob)
            except Exception:
                continue
        robot = self.robot()
        mounted = self.mounted_tool_items()
        disabled = failed = 0
        for ob in objects:
            # Disable against every robot link, including wrist/flange.
            for lid in range(self.robot_dof() + 1):
                try:
                    r = self.rdk.setCollisionActivePair(
                        robolink.COLLISION_OFF, ob, robot, 0, lid)
                    disabled += int(r) == 1
                    failed += int(r) != 1
                except Exception:
                    failed += 1
            for body in mounted:
                try:
                    r = self.rdk.setCollisionActivePair(
                        robolink.COLLISION_OFF, ob, body, 0, 0)
                    disabled += int(r) == 1
                    failed += int(r) != 1
                except Exception:
                    failed += 1
        return {"objects": [ob.Name() for ob in objects],
                "pairs_disabled": disabled, "pairs_failed": failed}

    @staticmethod
    def _parse_pair_endpoint(text: str) -> tuple[str, int]:
        """Parse ``Name[:Lid]`` as displayed by :meth:`collision_pairs`."""
        s = str(text).strip()
        if ":L" in s:
            name, lid = s.rsplit(":L", 1)
            try:
                return name.strip(), int(lid)
            except ValueError:
                return s, 0
        return s, 0

    def _item_by_name_any_type(self, name: str):
        """Best-effort RoboDK item lookup by name, independent of item type."""
        try:
            return self.rdk.Item(name)
        except Exception:
            return None

    def disable_collision_pair_strings(self, pairs: list[str] | None) -> dict:
        """Disable exact pair strings such as ``A:L1 ↔ B``.

        These strings are what the UI shows from RoboDK's current collision report.
        Applying them after collision checking is enabled lets an operator suppress
        station-model false positives without opening RoboDK's Collision Map.
        """
        import robolink

        disabled = failed = 0
        applied: list[str] = []
        for pair in pairs or []:
            text = str(pair).strip()
            if not text or "↔" not in text:
                continue
            left, right = [p.strip() for p in text.split("↔", 1)]
            n1, id1 = self._parse_pair_endpoint(left)
            n2, id2 = self._parse_pair_endpoint(right)
            it1, it2 = self._item_by_name_any_type(n1), self._item_by_name_any_type(n2)
            try:
                if it1 is None or it2 is None or not it1.Valid() or not it2.Valid():
                    failed += 1
                    continue
            except Exception:
                failed += 1
                continue
            ok = False
            # Try both orders. RoboDK usually accepts either, but this makes link IDs
            # robust when the pair was reported opposite to our lookup order.
            for a, b, aid, bid in ((it1, it2, id1, id2), (it2, it1, id2, id1)):
                try:
                    r = self.rdk.setCollisionActivePair(
                        robolink.COLLISION_OFF, a, b, aid, bid)
                    ok = ok or int(r) == 1
                except Exception:
                    pass
            if ok:
                disabled += 1
                applied.append(text)
            else:
                failed += 1
        return {"pairs": applied, "pairs_disabled": disabled, "pairs_failed": failed}

    # Joint step (deg) MoveJ_Test interpolates at while checking the approach — fine
    # enough to catch a thin self-collision, coarse enough to stay quick on the big
    # station. The destination config is always checked regardless of this.
    COLLISION_STEP_DEG = 8.0

    @staticmethod
    def _ik_to_joints(ik, seed_joints):
        """A clean joint vector (RoboDK ``Mat``) from a ``SolveIK`` result, or ``None``
        if it isn't a real solution.

        ``SolveIK`` returns an N-element ``Mat`` when reachable (N = DOF, occasionally
        +2 trailing config values) and a 1-element ``Mat([0])`` when not. We validate
        on the element COUNT — ``Mat.Cols()/Rows()`` return the row/column *lists*, not
        integer counts, so a ``Cols()==1`` style test is always False — and trim to the
        seed's DOF so the stored config and the ``MoveJ_Test`` start/end vectors are the
        same length."""
        import robodk.robomath as robomath

        try:
            vals = [float(v) for v in np.asarray(ik.list(), dtype=float).ravel()]
        except Exception:
            try:
                vals = [float(v) for v in np.asarray(ik, dtype=float).ravel()]
            except Exception:
                return None
        if len(vals) < 6:
            return None
        dof = None
        if seed_joints is not None:
            try:
                dof = int(np.asarray(seed_joints.list(), dtype=float).size)
            except Exception:
                dof = None
        if dof and 0 < dof <= len(vals):
            vals = vals[:dof]
        return robomath.Mat(vals)

    @staticmethod
    def _format_pair_keys(keys) -> list[str]:
        """Readable ``A:Ln ↔ B`` strings for a set of canonical pair keys."""
        out = []
        for k in keys or ():
            parts = [f"{name}:L{lid}" if lid else name
                     for (name, lid) in sorted(k, key=lambda x: (x[0], x[1]))]
            out.append(" ↔ ".join(parts))
        return sorted(out)

    def new_collisions_here(self, baseline):
        """At the current (already-positioned) sim config, return
        ``(has_new, readable_new_pairs)`` — collisions beyond ``baseline``. ``has_new``
        is ``None`` if collisions can't be evaluated. The resting-config companion to
        :meth:`path_new_collisions`."""
        keys = self.collision_pair_keys()
        if keys is None:
            return None, []
        new = keys - (baseline or set())
        return bool(new), self._format_pair_keys(new)

    def path_new_collision_details(self, j1, j2, baseline, samples: int = 6):
        """Does interpolating the joint move ``j1 -> j2`` introduce a colliding pair
        not present in ``baseline``? Samples the path (endpoints + interior points),
        reading the canonical pair-set at each. Returns ``True`` (a new collision
        appears), ``False`` (none beyond the baseline), or ``None`` (couldn't judge).

        ``j1`` may be ``None`` (then only the destination ``j2`` is checked). This is
        the swept, baseline-relative check both target generation (seed -> pose) and
        the dry tour (pose -> pose) use, so a pose that rests clear yet bumps an
        obstacle mid-move (the 8->9 transit) is caught."""
        import robodk.robomath as robomath

        robot = self.robot()
        try:
            b = np.asarray(j2.list(), dtype=float).ravel()
        except Exception:
            return None, []
        a = None
        if j1 is not None:
            try:
                aa = np.asarray(j1.list(), dtype=float).ravel()
                if aa.shape == b.shape:
                    a = aa
            except Exception:
                a = None
        samples = max(1, int(samples))
        fracs = ([1.0] if a is None or samples == 1
                 else [k / (samples - 1) for k in range(samples)])
        base = baseline or set()
        for f in fracs:
            cfg = b if a is None else a + (b - a) * f
            try:
                robot.setJoints(robomath.Mat([float(x) for x in cfg]))
            except Exception:
                return None, []
            keys = self.collision_pair_keys()
            if keys is None:
                return None, []
            new = keys - base
            if new:
                return True, self._format_pair_keys(new)
        return False, []

    def path_new_collisions(self, j1, j2, baseline, samples: int = 6):
        """Backward-compatible bool-only wrapper for swept new-collision checks."""
        collides, _pairs = self.path_new_collision_details(j1, j2, baseline, samples)
        return collides

    def screen_collisions(self, poses: list[np.ndarray], *,
                          guard_skip: int | None = None, obstacle_pairs: bool = False,
                          ignore_objects: list[str] | None = None,
                          ignore_pairs: list[str] | None = None,
                          baseline_relative: bool = True, path_samples: int = 6,
                          return_details: bool = False
                          ):
        """Test each TCP pose for collisions **in simulation** and record the exact
        joint configuration used.

        Returns ``(mask, checked, joints)``:

        * ``mask[i]``   True if pose ``i`` is collision-free, or its collision state
          couldn't be judged (unjudgeable poses are kept — this is a filter, the dry
          tour is the authoritative gate).
        * ``checked``   True iff collision checking was active for the sweep (False ⇒
          the build won't enable it ⇒ nothing dropped).
        * ``joints[i]`` the IK joint vector for pose ``i`` (opaque RoboDK ``Mat``), or
          ``None`` if no solution. Pass it to :meth:`add_target` so the *tested*
          configuration is the one stored — otherwise a cartesian target can be
          reached in a different IK branch that collides.

        ``guard_skip`` (when not None) force-enables the mounted-tool↔arm collision
        pairs, and ``obstacle_pairs`` the tool↔static-object pairs — both **after**
        turning collision checking on, because ``setCollisionActive(ON)`` rebuilds the
        default map (which excludes a tool↔own-robot and tool↔object), wiping any pair
        enabled earlier.

        Screening is **baseline-relative** (``baseline_relative``): a real cell reports
        constant collisions even at the safe seed pose (robot base on a pedestal, each
        tool against its mounting wrist, a parked axis clipping a wall). We record the
        colliding pair-set at the seed and reject a pose only if its swept path
        introduces a pair NOT in that baseline — so those artifacts are ignored and
        only genuine new contact drops the pose. ``path_samples`` interior+endpoint
        configs are checked along ``seed -> pose`` so a mid-move bump is caught too.
        (``baseline_relative=False`` restores the old total-count ``MoveJ_Test``.)

        Safety: forces SIMULATE, restores the seed joints and prior run mode, and
        **disables collision checking again afterwards** (leaving it on makes every
        later RoboDK call recompute collisions on the 117 MB station — the cause of
        slow/timing-out reconnects). Never raises."""
        robot = self.robot()
        try:
            seed_joints = robot.Joints()
        except Exception:
            seed_joints = None
        prior_mode = self.current_run_mode()
        self.rdk.setRunMode(self.RUNMODE_SIMULATE)
        on = self.set_collision_checking(True)
        if on and guard_skip is not None:
            self.ensure_mounted_tool_collision_pairs(guard_skip)
        if on and obstacle_pairs:
            self.ensure_obstacle_collision_pairs(ignore_objects)
        if on and ignore_objects:
            self.disable_object_collision_pairs(ignore_objects)
        ignored_pairs = (self.disable_collision_pair_strings(ignore_pairs)
                         if on and ignore_pairs else None)
        # Record the baseline collision pair-set at the SAFE seed config (the pose the
        # operator aimed from), so the constant modelling artifacts are subtracted out.
        baseline = set()
        if on and baseline_relative and seed_joints is not None:
            try:
                robot.setJoints(seed_joints)
            except Exception:
                pass
            bk = self.collision_pair_keys()
            baseline = bk if bk is not None else set()
        mask: list[bool] = []
        joints: list = []
        details: list[dict] = []
        try:
            for idx, T in enumerate(poses):
                sol = None
                collides: bool | None = None
                pairs: list[str] = []
                try:
                    # Anchor every solve to the seed config so the chosen IK branch is
                    # deterministic, and pass the camera tool EXPLICITLY so the joints
                    # place the camera (not the flange) at T — the config we lock +
                    # collision-check is the one the camera reaches the viewpoint in.
                    ik = self._solve_ik(T, seed_joints)
                    sol = self._ik_to_joints(ik, seed_joints)
                    if sol is not None and on:
                        if baseline_relative:
                            collides, pairs = self.path_new_collision_details(
                                seed_joints, sol, baseline, path_samples)
                        else:
                            j1 = seed_joints if seed_joints is not None else sol
                            ncol = robot.MoveJ_Test(j1, sol, self.COLLISION_STEP_DEG)
                            collides = int(ncol) > 0
                            if not collides:
                                collides = bool(self.collisions())
                            if collides:
                                pairs = self.collision_pairs(limit=20)
                except Exception:
                    sol, collides = sol, None
                joints.append(sol)
                mask.append(True if collides is None else not collides)
                details.append({
                    "index": idx,
                    "collides": collides,
                    "pairs": pairs,
                })
        finally:
            if seed_joints is not None:
                try:
                    robot.setJoints(seed_joints)
                except Exception:
                    pass
            self.set_run_mode_raw(prior_mode)
            self.set_collision_checking(False)   # don't leave the station heavy
        if return_details:
            return mask, on, joints, {
                "baseline_pairs": self._format_pair_keys(baseline),
                "ignored_pairs": ignored_pairs or {"pairs": [], "pairs_disabled": 0, "pairs_failed": 0},
                "poses": details,
            }
        return mask, on, joints

    def collision_status(self, *, ensure_pairs: bool = False,
                         skip_trailing: int = 2,
                         ignore_objects: list[str] | None = None,
                         ignore_pairs: list[str] | None = None) -> dict:
        """Best-effort snapshot of RoboDK collision checking at the **current** pose
        — no motion. Briefly enables checking to force a recompute, reads the count,
        then **disables it again** (so it doesn't slow every later call). Returns
        ``{"available": bool, "count": int | None}``; ``available`` False means this
        build/station can't evaluate collisions, so the generation filter drops
        nothing.

        When ``ensure_pairs`` is set, also force-enables the mounted-tool↔arm
        collision pairs first (see :meth:`ensure_mounted_tool_collision_pairs`) and
        adds ``guarded_tools`` / ``guarded_pairs`` so the UI chip can confirm the
        spindle/camera are actually being checked against the arm — RoboDK omits
        those pairs by default."""
        on = self.set_collision_checking(True)
        guard = (self.ensure_mounted_tool_collision_pairs(skip_trailing)
                 if on and ensure_pairs else None)
        ignored = (self.disable_object_collision_pairs(ignore_objects)
                   if on and ignore_objects else None)
        ignored_pairs = (self.disable_collision_pair_strings(ignore_pairs)
                         if on and ignore_pairs else None)
        n = self.collisions() if on else None
        pairs = self.collision_pairs(limit=50) if n else []
        self.set_collision_checking(False)
        out = {"available": n is not None, "count": n}
        if pairs:
            out["pairs"] = pairs
        if guard is not None:
            out["guarded_tools"] = guard["tools"]
            out["guarded_pairs"] = guard["pairs_enabled"]
        if ignored is not None:
            out["ignored_objects"] = ignored["objects"]
        if ignored_pairs is not None:
            out["ignored_pairs"] = ignored_pairs["pairs"]
            out["ignored_pair_count"] = ignored_pairs["pairs_disabled"]
        return out

    def collisions(self) -> int | None:
        """Number of colliding object pairs in the current (simulated) state, or
        ``None`` if this build/station can't check collisions. Best-effort; never
        raises so the dry tour degrades gracefully where collisions aren't set up."""
        try:
            return int(self.rdk.Collisions())
        except Exception:
            return None

    def collision_pairs(self, limit: int = 8) -> list[str]:
        """Best-effort names of currently colliding item pairs.

        RoboDK exposes pair details separately from the collision count. Returning
        names here gives the UI/logs enough signal to spot false positives such as
        an oversized "Wall" object colliding with a robot link.
        """
        try:
            pairs = self.rdk.CollisionPairs()
        except Exception:
            return []
        out: list[str] = []
        for p in pairs[:max(0, int(limit))]:
            try:
                item1, item2, id1, id2 = p
                n1 = item1.Name() if item1.Valid() else "<invalid>"
                n2 = item2.Name() if item2.Valid() else "<invalid>"
                l1 = f":L{id1}" if int(id1) else ""
                l2 = f":L{id2}" if int(id2) else ""
                out.append(f"{n1}{l1} ↔ {n2}{l2}")
            except Exception:
                continue
        return out

    def collision_pair_keys(self):
        """Canonical, order-independent set of the currently-colliding pairs (sim
        state), or ``None`` if collisions can't be evaluated on this build/station.

        Each key is a ``frozenset`` of ``(item_name, link_id)`` endpoints, so the
        same physical pair hashes identically no matter which order RoboDK reports it
        in. This is the unit of *baseline-relative* screening: a pose is unsafe only
        if its set contains a pair the safe seed's set does not (see
        :meth:`screen_collisions`). Best-effort; never raises."""
        try:
            n = int(self.rdk.Collisions())          # force a recompute of the map
        except Exception:
            return None
        if n <= 0:
            return set()
        try:
            pairs = self.rdk.CollisionPairs()
        except Exception:
            return set()
        keys = set()
        for p in pairs:
            try:
                it1, it2, id1, id2 = p
                n1 = it1.Name() if it1.Valid() else "<invalid>"
                n2 = it2.Name() if it2.Valid() else "<invalid>"
                keys.add(frozenset({(n1, int(id1)), (n2, int(id2))}))
            except Exception:
                continue
        return keys

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

    def target_joints(self, name: str):
        """The joint vector stored on a target (RoboDK ``Mat``), or ``None`` if it
        isn't a joint target / can't be read. Used to sweep the *exact* config the
        real run will visit when collision-checking the path between targets."""
        import robolink

        try:
            t = self.rdk.Item(name, robolink.ITEM_TYPE_TARGET)
            if not t.Valid():
                return None
            j = t.Joints()
            # Count the values (Mat.Cols()/Rows() return lists, not counts — the
            # gotcha that silently broke the collision sweep).
            if int(np.asarray(j.list(), dtype=float).size) >= 6:
                return j
        except Exception:
            pass
        return None

    def move_j_test(self, j1, j2, step_deg: float | None = None):
        """Collision-swept feasibility of the interpolated joint move ``j1 -> j2``.
        Returns the number of colliding object pairs along the path (0 = clear), or
        ``None`` if the inputs are bad / the build can't evaluate it. Requires global
        collision checking to be ON to actually report collisions."""
        if j1 is None or j2 is None:
            return None
        step = self.COLLISION_STEP_DEG if step_deg is None else step_deg
        try:
            return int(self.robot().MoveJ_Test(j1, j2, step))
        except Exception:
            return None

    def list_tools(self) -> list[str]:
        import robolink

        items = self.rdk.ItemList(robolink.ITEM_TYPE_TOOL)
        return sorted(i.Name() for i in items)

    def list_frames(self) -> list[str]:
        """Sorted reference-frame names available for a fabrication plan."""
        import robolink

        return sorted(i.Name() for i in self.rdk.ItemList(robolink.ITEM_TYPE_FRAME))

    def list_programs(self) -> list[str]:
        """Sorted standard-program names (Python macros excluded)."""
        import robolink

        return sorted(i.Name() for i in self.rdk.ItemList(robolink.ITEM_TYPE_PROGRAM))

    def program_instructions(self, name: str) -> list[str]:
        """Human-readable instruction texts for a standard program."""
        import robolink

        program = self.rdk.Item(name, robolink.ITEM_TYPE_PROGRAM)
        if not program.Valid():
            return []
        return [str(program.Instruction(index)[0])
                for index in range(program.InstructionCount())]

    def item_exists_as(self, name: str, kind: str) -> bool:
        """Exact-type existence check used for untrusted UI selections."""
        import robolink

        types = {
            "tool": robolink.ITEM_TYPE_TOOL,
            "frame": robolink.ITEM_TYPE_FRAME,
            "target": robolink.ITEM_TYPE_TARGET,
            "program": robolink.ITEM_TYPE_PROGRAM,
            "object": robolink.ITEM_TYPE_OBJECT,
        }
        if kind not in types or not name:
            return False
        return bool(self.rdk.Item(name, types[kind]).Valid())

    def use_named_tool_frame(self, tool_name: str, frame_name: str) -> np.ndarray:
        """Activate exact selected TOOL and FRAME items; return flange->TCP."""
        import robolink

        tool = self.rdk.Item(tool_name, robolink.ITEM_TYPE_TOOL)
        frame = self.rdk.Item(frame_name, robolink.ITEM_TYPE_FRAME)
        if not tool.Valid():
            raise RuntimeError(f"tool {tool_name!r} not found")
        if not frame.Valid():
            raise RuntimeError(f"frame {frame_name!r} not found")
        robot = self.robot()
        robot.setPoseTool(tool)
        robot.setPoseFrame(frame)
        self._tool_pose = pose_to_T(tool.PoseTool())
        self._frame = frame
        self._frame_wrt_base_T = self._frame_pose_wrt_base(frame)
        return self._tool_pose

    def _frame_pose_wrt_base(self, frame) -> np.ndarray:
        """Pose of ``frame`` w.r.t. the robot's base frame, via station-absolute
        poses — correct wherever either sits in the station tree."""
        import robolink

        base = self.robot().Parent()
        base_T = (pose_to_T(base.PoseAbs())
                  if base.Valid() and base.Type() == robolink.ITEM_TYPE_FRAME
                  else np.eye(4))
        return invert_T(base_T) @ pose_to_T(frame.PoseAbs())

    def current_tcp_xyzrpw(self, tool_name: str, frame_name: str) -> list[float]:
        """Read the selected tool TCP pose in the selected frame without motion."""
        import robodk.robomath as robomath

        self.use_named_tool_frame(tool_name, frame_name)
        return [float(value) for value in
                robomath.pose_2_xyzrpw(T_to_pose(self.tcp_pose_T()))]

    def extrusion_reachability_report(
            self, *, points_xyz: np.ndarray, orientation_rpy_deg,
            print_tool: str, work_frame: str, maximum_samples: int = 24,
            maximum_tool_axis_spin_deg: float = 90.0) -> dict:
        """Sample exact fixed-orientation path poses with IK, without robot motion.

        This is an early placement check, not a replacement for Curve Follow
        generation or the collision-enabled complete program dry run.
        """
        pts = np.asarray(points_xyz, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3 or not len(pts):
            raise ValueError("reachability path must be an Nx3 array")
        if not np.isfinite(pts).all():
            raise ValueError("reachability path contains non-finite coordinates")
        self.use_named_tool_frame(print_tool, work_frame)
        robot = self.robot()
        seed = robot.Joints()
        start_joints = seed
        orientation = self.xyzrpy_pose_T([0.0, 0.0, 0.0], orientation_rpy_deg)
        count = min(max(1, int(maximum_samples)), len(pts))
        indices = sorted(set(int(round(v)) for v in
                             np.linspace(0, len(pts) - 1, count)))
        samples: list[dict] = []
        for index in indices:
            target = orientation.copy()
            target[:3, 3] = pts[index]
            try:
                joints = self.solve_joints_on_neutral_branch(
                    target, start_joints, seed, maximum_tool_axis_spin_deg)
            except Exception:
                joints = None
            if joints is not None:
                joints = self._nearest_equivalent_joints(joints, seed)
            axis_4_rotation = self._joint_delta_deg(joints, start_joints, axis=3)
            axis_6_rotation = self._joint_delta_deg(joints, start_joints, axis=5)
            wrist_ok = (axis_4_rotation is not None and axis_6_rotation is not None
                        and abs(axis_4_rotation) <= maximum_tool_axis_spin_deg
                        and abs(axis_6_rotation) <= maximum_tool_axis_spin_deg)
            reachable = joints is not None
            acceptable = reachable and wrist_ok
            samples.append({
                "point_index": index,
                "xyz_mm": [float(v) for v in pts[index]],
                "reachable": acceptable,
                "ik_reachable": reachable,
                "axis_4_rotation_deg": axis_4_rotation,
                "tool_axis_spin_deg": axis_6_rotation,
                "reason": ("" if acceptable else
                           ("no IK solution on the neutral wrist branch within "
                            f"±{maximum_tool_axis_spin_deg:.1f} deg") if not reachable else
                           "wrist rotation exceeds the neutral limit"),
            })
            if acceptable:
                seed = joints
        failed = [sample for sample in samples if not sample["reachable"]]
        return {
            "all_reachable": not failed,
            "sample_count": len(samples),
            "reachable_count": len(samples) - len(failed),
            "first_unreachable": failed[0] if failed else None,
            "frame": work_frame,
            "tool": print_tool,
            "orientation_rpy_deg": [float(v) for v in orientation_rpy_deg],
            "maximum_tool_axis_spin_deg": float(maximum_tool_axis_spin_deg),
            "note": ("Sampled poses have IK solutions; full curve/collision dry run is still required."
                     if not failed else
                     "At least one sampled path pose has no solution on the neutral "
                     "robot configuration inside the axis-4/axis-6 rotation limit."),
        }

    @staticmethod
    def _joint_values(joints) -> list[float] | None:
        if joints is None:
            return None
        try:
            return [float(v) for v in np.asarray(joints.list(), dtype=float).ravel()]
        except Exception:
            try:
                return [float(v) for v in np.asarray(joints, dtype=float).ravel()]
            except Exception:
                return None

    @classmethod
    def _nearest_equivalent_joints(cls, joints, reference):
        """Return the same revolute configuration unwrapped nearest ``reference``."""
        import robodk.robomath as robomath

        values = cls._joint_values(joints)
        prior = cls._joint_values(reference)
        if values is None or prior is None or len(values) != len(prior):
            return joints
        # This cell's KUKA has six revolute robot axes. Leave any external axes
        # untouched; only normalize equivalent ±360-degree representations.
        for index in range(min(6, len(values))):
            values[index] = prior[index] + ((values[index] - prior[index] + 180.0) % 360.0 - 180.0)
        return robomath.Mat(values)

    @classmethod
    def _joint_delta_deg(cls, joints, reference, *, axis: int) -> float | None:
        values = cls._joint_values(joints)
        prior = cls._joint_values(reference)
        if values is None or prior is None or axis >= len(values) or axis >= len(prior):
            return None
        return float(values[axis] - prior[axis])

    @classmethod
    def _program_neutral_wrist_report(
            cls, program, neutral_joints, maximum_wrist_rotation_deg: float) -> dict:
        """Sample the interpolated program path and reject hidden wrist flips."""
        neutral = cls._joint_values(neutral_joints)
        if neutral is None or len(neutral) < 6:
            raise RuntimeError("neutral robot joints are unavailable for wrist validation")
        message, joint_list, status = program.InstructionListJoints(
            mm_step=10.0, deg_step=3.0, collision_check=0)
        if int(status) < 0:
            raise RuntimeError(
                "RoboDK could not sample the generated joint path: "
                + str(message or f"status {status}"))
        samples = list(joint_list)
        if not samples:
            raise RuntimeError("RoboDK returned no joint samples for the generated path")
        max_axis_4 = 0.0
        max_axis_6 = 0.0
        limit = float(maximum_wrist_rotation_deg)
        for sample_index, sample in enumerate(samples):
            values = [float(v) for v in sample]
            if len(values) < 6:
                raise RuntimeError("RoboDK returned an incomplete joint-path sample")
            axis_4 = ((values[3] - neutral[3] + 180.0) % 360.0) - 180.0
            axis_6 = ((values[5] - neutral[5] + 180.0) % 360.0) - 180.0
            max_axis_4 = max(max_axis_4, abs(axis_4))
            max_axis_6 = max(max_axis_6, abs(axis_6))
            if abs(axis_4) > limit or abs(axis_6) > limit:
                axis_name = "axis 4" if abs(axis_4) > limit else "axis 6"
                rotation = axis_4 if axis_name == "axis 4" else axis_6
                raise RuntimeError(
                    f"generated path sample {sample_index + 1} turns {axis_name} "
                    f"{rotation:.1f} deg from neutral; limit is ±{limit:.1f} deg. "
                    "The wrist-flipped path was blocked before simulation or robot motion.")
        return {
            "sample_count": len(samples),
            "maximum_axis_4_rotation_seen_deg": float(max_axis_4),
            "maximum_axis_6_rotation_seen_deg": float(max_axis_6),
        }

    @staticmethod
    def xyzrpy_pose_T(xyz_mm, rpy_deg) -> np.ndarray:
        """Build a RoboDK-convention XYZ+RPW pose as a numpy transform."""
        import robodk.robomath as robomath

        xyz = [float(v) for v in xyz_mm]
        rpy = [float(v) for v in rpy_deg]
        return pose_to_T(robomath.xyzrpw_2_pose(xyz + rpy))

    def ensure_mock_valve_programs(self, prefix: str = "TasniDry") -> tuple[str, str]:
        """Create comment-only dry-run valve programs (never touch I/O)."""
        import robolink

        names = (f"{prefix}AirOn", f"{prefix}AirOff")
        for name, state in zip(names, ("ON", "OFF")):
            old = self.rdk.Item(name, robolink.ITEM_TYPE_PROGRAM)
            if old.Valid():
                old.Delete()
            prog = self.rdk.AddProgram(name, self.robot())
            prog.RunInstruction(
                f"DRY_RUN mock valve {state}; physical outputs blocked",
                robolink.INSTRUCTION_COMMENT)
        return names

    def create_extrusion_layer_program(
            self, *, name: str, points_xyz: np.ndarray, orientation_rpy_deg,
            print_tool: str, work_frame: str, speed_mm_s: float,
            travel_speed_mm_s: float, rounding_mm: float,
            approach_clearance_mm: float, retreat_clearance_mm: float,
            air_on_program: str, air_off_program: str,
            maximum_tool_axis_spin_deg: float = 90.0) -> dict:
        """Build a disposable native RoboDK Curve Follow Project for one layer.

        Dense samples live inside one curve object (XYZ+IJK), not as station
        targets. RoboDK owns interpolation, approach/retract, preferred orientation,
        process/rapid speeds, blending, and path-boundary program events.
        """
        import robolink

        pts = np.asarray(points_xyz, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 3:
            raise ValueError("extrusion path must be a finite Nx3 array")
        if not np.isfinite(pts).all():
            raise ValueError("extrusion path contains non-finite coordinates")
        required = ((print_tool, "tool"), (work_frame, "frame"),
                    (air_on_program, "program"), (air_off_program, "program"))
        missing = [f"{kind} {value!r}" for value, kind in required
                   if not self.item_exists_as(value, kind)]
        if missing:
            raise RuntimeError("missing station item(s): " + ", ".join(missing))

        robot = self.robot()
        frame = self.rdk.Item(work_frame, robolink.ITEM_TYPE_FRAME)
        print_tcp = self.rdk.Item(print_tool, robolink.ITEM_TYPE_TOOL)
        # Cache the exact selected TCP/frame for deterministic seeded IK below.
        self.use_named_tool_frame(print_tool, work_frame)
        start_joints = robot.Joints()
        curve_name = name + "_Curve"
        project_name = name + "_Settings"
        for item_name, item_type in ((name, robolink.ITEM_TYPE_PROGRAM),
                                     (project_name, robolink.ITEM_TYPE_MACHINING),
                                     (curve_name, robolink.ITEM_TYPE_OBJECT)):
            old = self.rdk.Item(item_name, item_type)
            if old.Valid():
                old.Delete()

        orientation = self.xyzrpy_pose_T([0.0, 0.0, 0.0], orientation_rpy_deg)
        normal = orientation[:3, 2]
        curve_vertices = np.column_stack(
            (pts, np.repeat(normal.reshape(1, 3), len(pts), axis=0)))
        curve = self.rdk.AddCurve(
            curve_vertices.tolist(), projection_type=robolink.PROJECTION_NONE)
        if not curve.Valid():
            raise RuntimeError("RoboDK could not create the extrusion curve")
        curve.setName(curve_name)
        # setParent keeps the object's identity pose, so XYZ/IJK remain local to
        # the selected work frame instead of station/world coordinates.
        curve.setParent(frame)
        curve.setValue("Display", "LINEW=4 COLOR=#FF39D0BD")

        project = self.rdk.AddMachiningProject(project_name, robot)
        if not project.Valid():
            curve.Delete()
            raise RuntimeError("RoboDK could not create the Curve Follow Project")
        project.setPoseFrame(frame)
        project.setPoseTool(print_tcp)
        project.setJoints(start_joints)
        # Apply every deterministic path option before selecting the curve.
        # setMachiningParameters generates immediately and otherwise inherits the
        # station-wide CAM defaults for that first solve. In this cell those defaults
        # may enable the unrelated Positioner (TurntableActive) and orientation search,
        # making setup fail even when the exact fixed TCP poses passed our IK preflight.
        project.setParam("Machining", {
            "Algorithm": 0,
            "ApproachRetractAll": 1,
            "AutoUpdate": 0,
            "AvoidCollisions": 0,
            "FollowAngleOn": 0,
            "FollowRealignOn": 0,
            "FollowStepOn": 0,
            "JoinCurvesTol": 0.1,
            "PointApproach": float(approach_clearance_mm),
            "RapidApproachRetract": 1,
            "RotZ_Range": 0,
            "SpeedOperation": float(speed_mm_s),
            "SpeedRapid": float(travel_speed_mm_s),
            "TurntableActive": 0,
            "VisibleNormals": 1,
        })
        project.setParam("ProgEvents", {
            "CallPathStart": air_on_program,
            "CallPathStartOn": 1,
            "CallPathFinish": air_off_program,
            "CallPathFinishOn": 1,
            "RapidSpeed": float(travel_speed_mm_s),
            "Rounding": float(rounding_mm),
            "RoundingOn": 1 if rounding_mm > 0 else 0,
        })
        project.setParam("Approach", f"NTS {float(approach_clearance_mm):.6f} 0 0")
        project.setParam("Retract", f"NTS {float(retreat_clearance_mm):.6f} 0 0")
        # RoboDK's Curve Follow initializer can reject an otherwise reachable fixed
        # TCP pose depending on the selected tool's mounting transform. LongCalibTool
        # is one such case: its identity path-to-tool seed returns -5, while the
        # equivalent local-X-flipped seed produces the path. This pose is only an
        # internal generation seed. Every generated motion is pinned back to the exact
        # requested ``orientation`` below, then IK/collision validated, so the fallback
        # cannot silently change the commanded print orientation. Keep the curve normal
        # unchanged so approach/retract remain on the safe +normal side of the surface.
        flip_local_x = np.eye(4)
        flip_local_x[1, 1] = -1.0
        flip_local_x[2, 2] = -1.0
        generation_orientations = (orientation, orientation @ flip_local_x)
        setup_attempts: list[float] = []
        program = None
        setup_status = -1.0
        for generation_orientation in generation_orientations:
            project.setPose(T_to_pose(generation_orientation))
            candidate, candidate_status = project.setMachiningParameters(part=curve)
            setup_status = float(candidate_status)
            setup_attempts.append(setup_status)
            if candidate.Valid() and setup_status >= 0:
                program = candidate
                break
        if program is None:
            bounds_min = pts.min(axis=0)
            bounds_max = pts.max(axis=0)
            raise RuntimeError(
                "RoboDK found no feasible start/path for the Curve Follow Project "
                f"(statuses {setup_attempts}). Frame={work_frame!r}, tool={print_tool!r}, "
                f"XYZ bounds=[{bounds_min[0]:.1f}..{bounds_max[0]:.1f}, "
                f"{bounds_min[1]:.1f}..{bounds_max[1]:.1f}, "
                f"{bounds_min[2]:.1f}..{bounds_max[2]:.1f}] mm, "
                f"XYZRPW={[float(v) for v in orientation_rpy_deg]}. "
                f"The failed artifacts {curve_name!r} and {project_name!r} were kept "
                "for inspection; Reset removes them. Jog the selected TCP to the intended "
                "path start and seed the coordinates again.")
        program.setName(name)
        project.setParam("UpdatePath")
        generation = project.Update(robolink.COLLISION_OFF)
        if float(generation[3]) < 0:
            raise RuntimeError(
                "RoboDK Curve Follow Project generation failed: "
                + str(generation[4] or "unspecified path problem")
                + f". Failed artifacts {curve_name!r}, {project_name!r}, and any "
                  f"linked {name!r} program were kept; "
                  "Reset removes them.")
        program = project.getLink(robolink.ITEM_TYPE_PROGRAM)
        if not program.Valid():
            raise RuntimeError(
                "Curve Follow Project has no linked generated program. "
                f"Failed artifacts {curve_name!r} and {project_name!r} were kept; "
                "Reset removes them.")
        program.setName(name)
        # A Curve Follow program does not reliably retain joint vectors written
        # through setInstruction: on this station the next read/playback silently
        # re-solves the local-X fallback into the opposite axis-4 wrist branch.
        # Rebuild the linked program from named joint targets instead. The curve and
        # project remain visible/auditable, while every motion endpoint is now a real
        # target locked to the selected neutral IK branch.
        for instruction_index in reversed(range(program.InstructionCount())):
            program.InstructionDelete(instruction_index)
        program.setPoseFrame(frame)
        program.setPoseTool(print_tcp)
        program.setRounding(float(rounding_mm))
        prior_joints = start_joints
        maximum_axis_4_seen = 0.0
        maximum_spin_seen = 0.0
        target_names: list[str] = []

        def locked_target(label: str, xyz) -> object:
            nonlocal prior_joints, maximum_axis_4_seen, maximum_spin_seen
            target_T = orientation.copy()
            target_T[:3, 3] = np.asarray(xyz, dtype=float)
            solved = self.solve_joints_on_neutral_branch(
                target_T, start_joints, prior_joints,
                maximum_tool_axis_spin_deg)
            if solved is None:
                raise RuntimeError(
                    f"generated target {label!r} has no IK solution on the neutral "
                    "front/elbow/wrist branch within the axis-4/axis-6 rotation "
                    "limit; artifacts were kept for inspection")
            solved = self._nearest_equivalent_joints(solved, prior_joints)
            axis_4 = self._joint_delta_deg(solved, start_joints, axis=3)
            spin = self._joint_delta_deg(solved, start_joints, axis=5)
            if (axis_4 is None or spin is None
                    or abs(axis_4) > float(maximum_tool_axis_spin_deg)
                    or abs(spin) > float(maximum_tool_axis_spin_deg)):
                raise RuntimeError(
                    f"generated target {label!r} leaves the neutral wrist range "
                    f"(axis 4={axis_4 if axis_4 is not None else 'unknown'} deg, "
                    f"axis 6={spin if spin is not None else 'unknown'} deg); limit is "
                    f"±{float(maximum_tool_axis_spin_deg):.1f} deg. "
                    "Capture the current TCP orientation and regenerate the plan.")
            maximum_axis_4_seen = max(maximum_axis_4_seen, abs(axis_4))
            maximum_spin_seen = max(maximum_spin_seen, abs(spin))
            target_name = f"{name}_{label}"
            target = self.add_target(target_name, target_T, solved)
            target_names.append(target_name)
            prior_joints = solved
            return target

        approach_xyz = pts[0] + normal * float(approach_clearance_mm)
        retract_xyz = pts[-1] + normal * float(retreat_clearance_mm)
        approach_target = locked_target("Approach", approach_xyz)
        path_targets = [locked_target(f"P{index:04d}", xyz)
                        for index, xyz in enumerate(pts)]
        retract_target = locked_target("Retract", retract_xyz)

        program.setSpeed(float(travel_speed_mm_s))
        program.MoveJ(approach_target)
        program.MoveL(path_targets[0])
        program.RunInstruction(air_on_program, robolink.INSTRUCTION_CALL_PROGRAM)
        program.setSpeed(float(speed_mm_s))
        for target in path_targets[1:]:
            program.MoveL(target)
        program.RunInstruction(air_off_program, robolink.INSTRUCTION_CALL_PROGRAM)
        program.setSpeed(float(travel_speed_mm_s))
        program.MoveL(retract_target)
        trajectory = self._program_neutral_wrist_report(
            program, start_joints, maximum_tool_axis_spin_deg)
        artifacts = [curve_name, project_name, name, *target_names]
        return {
            "program": name, "curve": curve_name, "project": project_name,
            "artifacts": artifacts, "targets": target_names, "point_count": len(pts),
            "setup_status": float(setup_status),
            "setup_attempts": setup_attempts,
            "maximum_tool_axis_spin_deg": float(maximum_tool_axis_spin_deg),
            "maximum_axis_4_rotation_seen_deg": max(
                float(maximum_axis_4_seen),
                trajectory["maximum_axis_4_rotation_seen_deg"]),
            "maximum_tool_axis_spin_seen_deg": max(
                float(maximum_spin_seen),
                trajectory["maximum_axis_6_rotation_seen_deg"]),
            "joint_path_sample_count": trajectory["sample_count"],
            "project_generation_percent": float(generation[3]) * 100.0,
        }

    def cleanup_extrusion_artifacts(self, prefix: str = "TasniCylinder_") -> list[str]:
        """Delete stale generated programs/projects/curves owned by this module."""
        import robolink

        removed: list[str] = []
        for item_type in (robolink.ITEM_TYPE_PROGRAM, robolink.ITEM_TYPE_MACHINING,
                          robolink.ITEM_TYPE_TARGET, robolink.ITEM_TYPE_OBJECT):
            for item in list(self.rdk.ItemList(item_type)):
                item_name = item.Name()
                if item_name.startswith(prefix) and item.Valid():
                    item.Delete()
                    removed.append(item_name)
        return removed

    def create_inspection_target(self, *, name: str, T: np.ndarray,
                                 inspection_tool: str, work_frame: str) -> dict:
        """Create a derived inspection target at camera pose ``T`` (work frame).

        ``T`` places the **camera** — the inspection tool's TCP — not the flange:
        :meth:`solve_joints_for_pose` passes the tool mount to ``SolveIK``
        explicitly, so the joints put the lens at the requested viewpoint. The
        target is stored as a **joint** target locked to that solution, so the
        configuration RoboDK then collision-validates is the one actually visited
        (a cartesian target can be reached in a different, colliding IK branch).

        Returns ``{"created": False, "reason": ...}`` rather than raising when the
        pose has no IK solution: the caller is walking an ordered candidate list
        and an unreachable viewpoint is an expected outcome, not a fault.
        """
        import robolink

        self.use_named_tool_frame(inspection_tool, work_frame)
        old = self.rdk.Item(name, robolink.ITEM_TYPE_TARGET)
        if old.Valid():
            old.Delete()
        joints = self.solve_joints_for_pose(T, self.robot().Joints())
        if joints is None:
            return {"created": False, "target": name,
                    "reason": "no IK solution places the camera at this viewpoint"}
        self.add_target(name, np.asarray(T, dtype=float), joints)
        return {"created": True, "target": name, "reason": ""}

    def create_inspection_program(self, *, name: str, inspection_tool: str,
                                  inspection_target: str, speed_mm_s: float) -> dict:
        """Create the post-OFF inspection move as its own auditable program."""
        import robolink

        required = ((inspection_tool, "tool"), (inspection_target, "target"))
        missing = [f"{kind} {value!r}" for value, kind in required
                   if not self.item_exists_as(value, kind)]
        if missing:
            raise RuntimeError("missing station item(s): " + ", ".join(missing))
        robot = self.robot()
        tool = self.rdk.Item(inspection_tool, robolink.ITEM_TYPE_TOOL)
        target = self.rdk.Item(inspection_target, robolink.ITEM_TYPE_TARGET)
        old = self.rdk.Item(name, robolink.ITEM_TYPE_PROGRAM)
        if old.Valid():
            old.Delete()
        program = self.rdk.AddProgram(name, robot)
        parent = target.Parent()
        if parent.Valid() and parent.Type() == robolink.ITEM_TYPE_FRAME:
            program.setPoseFrame(parent)
        program.setPoseTool(tool)
        program.setSpeed(float(speed_mm_s))
        program.MoveJ(target)
        return {"program": name, "targets": [], "inspection_target": inspection_target}

    def update_program(self, name: str, *, collisions: bool = True) -> dict:
        """RoboDK program validation (instructions, time, distance, % feasible)."""
        import robolink

        program = self.rdk.Item(name, robolink.ITEM_TYPE_PROGRAM)
        if not program.Valid():
            raise RuntimeError(f"program {name!r} not found")
        result = program.Update(robolink.COLLISION_ON if collisions else robolink.COLLISION_OFF)
        return {"instructions_ok": int(result[0]), "time_s": float(result[1]),
                "distance_mm": float(result[2]), "percent_ok": float(result[3]) * 100.0,
                "problems": str(result[4] or "")}

    def simulation_speed(self) -> float:
        """Current RoboDK playback speed ratio (read only)."""
        return float(self.rdk.SimulationSpeed())

    def set_simulation_speed(self, ratio: float) -> None:
        """Set RoboDK playback speed; this never changes robot run mode or hardware."""
        self.rdk.setSimulationSpeed(float(ratio))

    def start_program(self, name: str, *, real_robot: bool) -> int:
        """Start a named program with an explicit simulator/robot run type."""
        import robolink

        program = self.rdk.Item(name, robolink.ITEM_TYPE_PROGRAM)
        if not program.Valid():
            raise RuntimeError(f"program {name!r} not found")
        program.setRunType(robolink.PROGRAM_RUN_ON_ROBOT if real_robot
                           else robolink.PROGRAM_RUN_ON_SIMULATOR)
        return int(program.RunCode())

    def program_busy(self, name: str) -> bool:
        import robolink

        return bool(self.rdk.Item(name, robolink.ITEM_TYPE_PROGRAM).Busy())

    def stop_program(self, name: str) -> None:
        import robolink

        program = self.rdk.Item(name, robolink.ITEM_TYPE_PROGRAM)
        if program.Valid():
            program.Stop()

    def run_station_program(self, name: str, *, real_robot: bool) -> None:
        """Run a station program to completion with explicit run type."""
        started = self.start_program(name, real_robot=real_robot)
        if started < 0:
            raise RuntimeError(f"program {name!r} could not start")
        while self.program_busy(name):
            import time
            time.sleep(0.05)

    def set_tool_pose(self, tool_name: str, T: np.ndarray) -> None:
        import robolink

        tool = self.rdk.Item(tool_name, robolink.ITEM_TYPE_TOOL)
        tool.setPoseTool(T_to_pose(T))

    def get_tool_pose_T(self, tool_name: str) -> np.ndarray:
        import robolink

        tool = self.rdk.Item(tool_name, robolink.ITEM_TYPE_TOOL)
        return pose_to_T(tool.PoseTool())

    def create_tool(self, tool_name: str, T: np.ndarray):
        """Add a tool named ``tool_name`` on the robot flange at mounting pose ``T``
        (flange->tool, numpy 4x4). Used to self-heal a deleted camera tool from a
        known-good offset so the *camera* — not the flange — is what the generated
        targets drive. Returns the new tool item."""
        return self.robot().AddTool(T_to_pose(T), tool_name)

    # -- scene geometry creation (scan module: frame / rectangle / mesh) ----
    def robot_base_frame(self):
        """The reference frame the robot is attached to (its parent), or ``None``.
        Scan results are computed in this frame — each view's camera pose comes from
        :meth:`camera_pose_T` in the active (base) frame — so new items default to it."""
        import robolink

        base = self.robot().Parent()
        if base.Valid() and base.Type() == robolink.ITEM_TYPE_FRAME:
            return base
        return None

    def add_frame(self, name: str, T: np.ndarray, parent=None):
        """Create a reference FRAME named ``name`` at pose ``T`` (numpy 4x4) relative
        to ``parent`` (default: the robot base frame). This is the *working frame* the
        scan derives from the table plane — the user then programs/jogs in it. Replaces
        any existing same-named frame so re-inserting a scan is idempotent. Returns the
        frame item."""
        import robolink

        parent = parent if parent is not None else self.robot_base_frame()
        existing = self.rdk.Item(name, robolink.ITEM_TYPE_FRAME)
        if existing.Valid():
            existing.Delete()
        frame = self.rdk.AddFrame(name, parent if parent is not None else 0)
        frame.setPose(T_to_pose(T))
        return frame

    def add_rectangle(self, name: str, corners_xyz: np.ndarray, parent=None,
                      color: list | None = None):
        """Create a flat quadrilateral OBJECT from 4 ``corners_xyz`` (4x3, ordered
        around the rectangle) as a visual work-surface reference. The corner
        coordinates are in ``parent`` (default: robot base frame), the frame the scan
        computes them in. Replaces any existing same-named object. Returns the object.

        Implemented as a tiny 2-triangle OBJ imported via ``AddFile`` rather than
        ``AddShape``: RoboDK's ``AddShape`` rejects a shape when an attach *parent* is
        given ("Invalid shape … 3xN") and returns an invalid item, whereas a parented
        ``AddFile`` is reliable (the same path the fused mesh uses)."""
        import os
        from tempfile import TemporaryDirectory

        import robolink

        c = np.asarray(corners_xyz, dtype=float).reshape(4, 3)
        existing = self.rdk.Item(name, robolink.ITEM_TYPE_OBJECT)
        if existing.Valid():
            existing.Delete()
        # 4 vertices, 2 triangles — both windings so the quad is visible from either
        # side (no back-face culling surprises).
        lines = ["# Tasni work-surface rectangle"]
        lines += [f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}" for p in c]
        lines += ["f 1 2 3", "f 1 3 4", "f 1 3 2", "f 1 4 3"]
        add_to = parent if parent is not None else self.robot_base_frame()
        with TemporaryDirectory(prefix="tasni_rect_") as td:
            path = os.path.join(td, "rect.obj")
            with open(path, "w", encoding="ascii") as fh:
                fh.write("\n".join(lines) + "\n")
            obj = self.rdk.AddFile(path, add_to if add_to is not None else 0)
        if obj.Valid():
            obj.setName(name)
            obj.setColor(color if color is not None else [0.0, 0.6, 1.0, 0.5])
        return obj

    def add_mesh_file(self, name: str, path: str, parent=None,
                      color: list | None = None):
        """Import a mesh file (``.obj``/``.ply``/...) as an OBJECT named ``name`` under
        ``parent`` (default: robot base frame) — the fused scan surface. Mirrors
        ``macros/3DScan.py:save_point_cloud`` (``AddFile`` + name + color). Replaces any
        existing same-named object. Returns the object item (invalid item if the import
        failed)."""
        import robolink

        existing = self.rdk.Item(name, robolink.ITEM_TYPE_OBJECT)
        if existing.Valid():
            existing.Delete()
        add_to = parent if parent is not None else self.robot_base_frame()
        item = self.rdk.AddFile(path, add_to if add_to is not None else 0)
        if item.Valid():
            item.setName(name)
            if color is not None:
                item.setColor(color)
        return item

    def add_keepout_box(self, name: str, board_pts_base: np.ndarray, *,
                        margin_mm: float, above_mm: float, depth_mm: float,
                        parent=None, color: list | None = None):
        """Create an axis-aligned box OBJECT spanning the board footprint + a lateral
        ``margin_mm``, from ``above_mm`` above the board's top down by ``depth_mm``
        toward the floor — a conservative stand-in for the physical platform the board
        rests on, so the collision screen can drop a pose that would graze it.

        ``board_pts_base`` is an ``(N, 3)`` array of board points in the base frame
        (the frame the box is parented to). Replaces any existing same-named object so
        re-creating it each generation is idempotent. Built as a 12-triangle OBJ via
        ``AddFile`` (the reliable parented-import path :meth:`add_rectangle` uses)."""
        import os
        from tempfile import TemporaryDirectory

        import robolink

        pts = np.asarray(board_pts_base, dtype=float).reshape(-1, 3)
        xlo, xhi = float(pts[:, 0].min()) - margin_mm, float(pts[:, 0].max()) + margin_mm
        ylo, yhi = float(pts[:, 1].min()) - margin_mm, float(pts[:, 1].max()) + margin_mm
        ztop = float(pts[:, 2].max()) + above_mm
        zbot = ztop - depth_mm
        verts = [(xlo, ylo, zbot), (xhi, ylo, zbot), (xhi, yhi, zbot), (xlo, yhi, zbot),
                 (xlo, ylo, ztop), (xhi, ylo, ztop), (xhi, yhi, ztop), (xlo, yhi, ztop)]
        faces = [(1, 2, 3), (1, 3, 4), (5, 8, 7), (5, 7, 6), (1, 5, 6), (1, 6, 2),
                 (2, 6, 7), (2, 7, 3), (3, 7, 8), (3, 8, 4), (4, 8, 5), (4, 5, 1)]
        existing = self.rdk.Item(name, robolink.ITEM_TYPE_OBJECT)
        if existing.Valid():
            existing.Delete()
        lines = ["# Tasni board keep-out (platform stand-in)"]
        lines += [f"v {x:.4f} {y:.4f} {z:.4f}" for (x, y, z) in verts]
        lines += [f"f {a} {b} {c}" for (a, b, c) in faces]
        add_to = parent if parent is not None else self.robot_base_frame()
        with TemporaryDirectory(prefix="tasni_keepout_") as td:
            path = os.path.join(td, "keepout.obj")
            with open(path, "w", encoding="ascii") as fh:
                fh.write("\n".join(lines) + "\n")
            obj = self.rdk.AddFile(path, add_to if add_to is not None else 0)
        if obj.Valid():
            obj.setName(name)
            obj.setColor(color if color is not None else [1.0, 0.25, 0.25, 0.28])
        return obj
