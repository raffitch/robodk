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


class RdkIO:
    """Cell operations on top of an :class:`RdkSession`."""

    # RoboDK RUNMODE constants (avoids importing robolink just for the enum).
    RUNMODE_SIMULATE = 1
    RUNMODE_RUN_ROBOT = 6

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
        else:
            self._frame = None      # AddTarget(None) / Pose() then use the base frame
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
        """Current joint vector (RoboDK ``Mat``) — snapshot for a safe return."""
        return self.robot().Joints()

    def move_j_joints(self, joints) -> None:
        """MoveJ to a joint vector (avoids IK/elbow ambiguity of a cartesian move)."""
        self.robot().MoveJ(joints)

    def move_j_pose(self, T: np.ndarray) -> None:
        self.robot().MoveJ(T_to_pose(T))

    def _solve_ik(self, T: np.ndarray, seed=None):
        """Raw ``SolveIK`` for a pose, with the camera tool passed **explicitly**.

        ``T`` is the pose of the camera (the last-activated tool's TCP) in the active
        reference frame; passing the tool mount (``_tool_pose``) as SolveIK's ``tool``
        argument makes the result place the CAMERA — not the flange — at ``T``,
        independent of whatever tool RoboDK thinks is active. ``seed`` (joints_approx)
        pins a deterministic IK branch. May raise; callers wrap as needed."""
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
                "pairs_failed": failed, "dof": dof}

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

    def screen_collisions(self, poses: list[np.ndarray], *,
                          guard_skip: int | None = None
                          ) -> tuple[list[bool], bool, list]:
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
        pairs **after** turning collision checking on — see the comment below; this is
        what actually catches the spindle↔arm self-collision and MUST happen here, not
        before, because ``setCollisionActive(ON)`` rebuilds the default map.

        Uses ``MoveJ_Test`` (interpolates the move and returns the number of colliding
        object pairs) from the robot's current ``seed`` joints to each candidate — so
        it catches the robot self-colliding with its own tooling (e.g. the spindle
        hitting link 4 in a wrist-flipped config). A resting-config ``Collisions()``
        backstop re-checks the endpoint in case the coarse joint step stepped over a
        thin contact. The spindle and every other flange body participate as long as
        their pair is enabled in the station's collision map.

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
        # Enable the mounted-tool<->arm collision pairs AFTER checking is on:
        # setCollisionActive(ON) rebuilds the collision map from RoboDK's defaults,
        # which EXCLUDE a tool from colliding with its own robot — so any pair enabled
        # while checking was off (e.g. by a caller before this) is wiped, and the
        # spindle<->A4 self-collision goes unseen (the target-12 collider that kept
        # slipping through). Re-applying here — the same order the dry tour uses —
        # keeps them active for the MoveJ_Test sweep below.
        if on and guard_skip is not None:
            self.ensure_mounted_tool_collision_pairs(guard_skip)
        mask: list[bool] = []
        joints: list = []
        try:
            for T in poses:
                sol = None
                collides: bool | None = None
                try:
                    # Anchor every solve to the seed config so the chosen IK branch
                    # is deterministic: without joints_approx, SolveIK returns the
                    # branch nearest the robot's CURRENT sim position, which drifts
                    # because MoveJ_Test below leaves the robot at each candidate.
                    # _solve_ik passes the camera tool EXPLICITLY, so the joints place
                    # the camera (not the flange) at T — the config we lock + collision
                    # check is the one the camera actually reaches the viewpoint in.
                    ik = self._solve_ik(T, seed_joints)
                    # SolveIK returns the joints as a Mat: reachable -> an N-element
                    # vector (N = DOF, sometimes +2 config values); unreachable -> a
                    # 1-element Mat([0]). CRITICAL: Mat.Cols()/Rows() return the row/
                    # column LISTS, not counts, so the old `ik.Cols()==1 and
                    # ik.Rows()>=6` was ALWAYS False — MoveJ_Test never ran and every
                    # pose survived as an un-collision-checked cartesian target (the
                    # spindle-into-A4 that slipped through). Discriminate on element
                    # count instead.
                    sol = self._ik_to_joints(ik, seed_joints)
                    if sol is not None and on:
                        j1 = seed_joints if seed_joints is not None else sol
                        ncol = robot.MoveJ_Test(j1, sol, self.COLLISION_STEP_DEG)
                        collides = int(ncol) > 0
                        if not collides:
                            # MoveJ_Test leaves the robot AT sol when the path is
                            # clear, so re-check the resting config directly — a
                            # coarse joint step can step over a thin endpoint contact.
                            rest = self.collisions()
                            collides = bool(rest)
                except Exception:
                    sol, collides = sol, None
                joints.append(sol)
                mask.append(True if collides is None else not collides)
        finally:
            if seed_joints is not None:
                try:
                    robot.setJoints(seed_joints)
                except Exception:
                    pass
            self.set_run_mode_raw(prior_mode)
            self.set_collision_checking(False)   # don't leave the station heavy
        return mask, on, joints

    def collision_status(self, *, ensure_pairs: bool = False,
                         skip_trailing: int = 2) -> dict:
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
        n = self.collisions() if on else None
        pairs = self.collision_pairs() if n else []
        self.set_collision_checking(False)
        out = {"available": n is not None, "count": n}
        if pairs:
            out["pairs"] = pairs
        if guard is not None:
            out["guarded_tools"] = guard["tools"]
            out["guarded_pairs"] = guard["pairs_enabled"]
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
        return [i.Name() for i in items]

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
