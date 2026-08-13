import math
import numpy as np
import pytest

from tasni.modules.scan.survey_contract import (
    MODE_COMPACT, PROVENANCE_BY_MODE, PROVENANCE_COMPACT,
    CaptureRecord, LockedWorkframeSurvey, RobotStateSnapshot,
    camera_calibration_id, capture_is_fresh, order_corners_clockwise, pose_delta,
    refresh_robot_state, robot_moved_since, workframe_from_rectangle,
)


def _T(x=0.0, y=0.0, z=0.0, rz_deg=0.0):
    T = np.eye(4)
    a = math.radians(rz_deg)
    T[:2, :2] = [[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]
    T[:3, 3] = [x, y, z]
    return T


def test_pose_delta_translation_and_rotation():
    trans, rot = pose_delta(_T(), _T(x=3.0, rz_deg=2.0))
    assert trans == pytest.approx(3.0)
    assert rot == pytest.approx(2.0, abs=1e-6)


class _FakeRdk:
    def __init__(self, joint_seq, T):
        self._seq = list(joint_seq)
        self._T = T
        self.calls = 0

    def current_joints(self):
        self.calls += 1
        return self._seq.pop(0) if len(self._seq) > 1 else self._seq[0]

    def camera_pose_T(self):
        return self._T


def test_refresh_robot_state_stationary_and_moving():
    still = refresh_robot_state(_FakeRdk([[0, 10, 20, 0, 30, 0]] * 2, _T()), sleep=lambda s: None)
    assert still.stationary is True
    assert still.camera_T_np().shape == (4, 4)
    moving = refresh_robot_state(
        _FakeRdk([[0, 10, 20, 0, 30, 0], [0, 10.5, 20, 0, 30, 0]], _T()), sleep=lambda s: None)
    assert moving.stationary is False


def test_refresh_robot_state_requires_pose():
    with pytest.raises(RuntimeError):
        refresh_robot_state(_FakeRdk([[0] * 6] * 2, None), sleep=lambda s: None)


class _MatJointsRdk:
    """Mimics the REAL RdkIO.current_joints() return shape (Defect 1a, Task 19):
    ``robodk.Item.Joints()`` returns ``robomath.Mat(values)`` built from a flat
    Python list. ``Mat.__init__`` treats a flat list as a single ROW and
    transposes it into a COLUMN vector (N rows x 1 col) -- so ``Mat.__len__``
    (which reports COLUMN count) reads 1, and ``np.asarray(mat, dtype=float)``
    ends up building a ``(1, N)`` array (one row holding all N joints) instead
    of the flat ``(N,)`` array every fake/test in this file used before. This
    class is the minimal real-shape stand-in: every other test here uses a
    plain Python list, which never triggers the bug because a flat list
    converts to ``(N,)`` directly.
    """
    def __init__(self, T):
        import robodk.robomath as robomath
        self._mat = robomath.Mat([0.0, 10.0, 20.0, 0.0, 30.0, 0.0])
        self._T = T

    def current_joints(self):
        return self._mat

    def camera_pose_T(self):
        return self._T


def test_refresh_robot_state_handles_robodk_mat_shaped_joints():
    """Defect 1a (Task 19): on real hardware, ``rdk.current_joints()`` returns a
    ``robomath.Mat`` column vector, not a flat list/array. Before the fix,
    ``np.asarray(mat, dtype=float)`` produced a ``(1, N)`` array; iterating it
    in ``tuple(float(v) for v in j1)`` handed ``float()`` the WHOLE N-element
    row in one call, raising exactly the operator's reported error: "only
    size-1 arrays can be converted to Python scalars". This is a hardware-free
    reproduction -- ``robomath.Mat`` is pure Python/pure math, no RoboDK
    connection required.
    """
    snap = refresh_robot_state(_MatJointsRdk(_T()), sleep=lambda s: None)
    assert snap.joints == pytest.approx((0.0, 10.0, 20.0, 0.0, 30.0, 0.0))
    assert snap.stationary is True
    assert snap.camera_T_np().shape == (4, 4)


def _snapshot(T=None):
    return RobotStateSnapshot(joints=(0.0,) * 6,
                              camera_T=tuple(map(tuple, (T if T is not None else _T()))),
                              fetched_at=100.0, stationary=True)


def test_robot_moved_since_tolerances_and_fail_open():
    snap = _snapshot()
    assert robot_moved_since(snap, _T(x=0.5), trans_tol_mm=1.2, rot_tol_deg=0.3) is False
    assert robot_moved_since(snap, _T(x=5.0), trans_tol_mm=1.2, rot_tol_deg=0.3) is True
    assert robot_moved_since(snap, None, trans_tol_mm=1.2, rot_tol_deg=0.3) is True


def _record(captured_at=100.0, stationary=True):
    snap = RobotStateSnapshot((0.0,) * 6, tuple(map(tuple, _T())), captured_at, stationary)
    return CaptureRecord(kind="center", robot=snap, measurement_ts=1.0,
                         captured_at=captured_at, n_frames=5, standoff_mm=350.0,
                         tilt_deg=1.0, valid_frac=0.9, plane_rms_mm=0.8,
                         plane_normal_base=(0, 0, 1), plane_point_base=(0, 0, 0))


def test_capture_freshness():
    assert capture_is_fresh(_record(100.0), now=105.0, max_age_s=10.0) is True
    assert capture_is_fresh(_record(100.0), now=120.0, max_age_s=10.0) is False
    assert capture_is_fresh(_record(100.0, stationary=False), now=101.0, max_age_s=10.0) is False


_RECT = np.array([[0, 0, 0], [1200, 0, 0], [1200, 800, 0], [0, 800, 0]], dtype=float)


def test_corner_ordering_is_deterministic_and_starts_near_base():
    ordered = order_corners_clockwise(_RECT, [0, 0, 1])
    shuffled = order_corners_clockwise(_RECT[[2, 0, 3, 1]], [0, 0, 1])
    assert np.allclose(ordered, shuffled)
    assert np.argmin(np.linalg.norm(ordered, axis=1)) == 0  # C1 nearest base
    # clockwise viewed along -Z: signed area in XY must be negative
    x, y = ordered[:, 0], ordered[:, 1]
    area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    assert area < 0


def test_workframe_convention_is_corner_origin_long_edge():
    # Task 1: the frame the survey publishes is now the one insertion already used --
    # origin at C1 (nearest the base), NOT the rectangle centre it used to return.
    T = workframe_from_rectangle(order_corners_clockwise(_RECT, [0, 0, 1]), [0, 0, -1])
    assert np.allclose(T[:3, 3], [0, 0, 0])               # origin = C1, nearest base
    assert abs(float(T[:3, 0] @ np.array([1.0, 0, 0]))) > 0.999  # X = the 1200 edge
    assert T[2, 2] > 0.99                                 # +Z up even if normal fed down
    assert abs(T[:3, 0] @ np.array([0, 0, 1])) < 1e-9     # X in plane
    assert np.allclose(np.cross(T[:3, 0], T[:3, 1]), T[:3, 2], atol=1e-9)  # right-handed


def _permutations_of(rect):
    import itertools
    return [rect[list(p)] for p in itertools.permutations(range(4))]


def test_workframe_is_identical_for_every_corner_ordering():
    """Task 1 acceptance: identical corners -> numerically identical frame.

    The acquisition paths (compact, five-position, reference, fused scan) each build
    their corner array in their own order. Order-independence here is what makes them
    agree, so this stands in for "the same rectangle through all result paths".
    """
    ref = workframe_from_rectangle(_RECT, [0, 0, 1])
    for perm in _permutations_of(_RECT):
        assert np.allclose(workframe_from_rectangle(perm, [0, 0, 1]), ref, atol=1e-12)


def test_near_square_tie_break_is_deterministic_and_noise_stable():
    """The historical bug source: on a near-square the longer-edge rule is noise.

    Inside EDGE_TIE_REL_TOL the length rule is abandoned for base-axis alignment, so
    a square measured slightly differently twice keeps the SAME frame instead of
    flipping 90 deg. Outside the band the plain longer-edge rule still decides.
    """
    square = np.array([[0, 0, 0], [500, 0, 0], [500, 500, 0], [0, 500, 0]], float)
    ref = workframe_from_rectangle(square, [0, 0, 1])
    # a hair longer in Y (0.04% -- inside the tie band) must NOT rotate the frame
    noisy = square.copy()
    noisy[2:, 1] = 500.2
    assert np.allclose(workframe_from_rectangle(noisy, [0, 0, 1]), ref, atol=1e-9)
    # ...and the choice is stable under corner reordering too
    for perm in _permutations_of(noisy):
        assert np.allclose(workframe_from_rectangle(perm, [0, 0, 1]), ref, atol=1e-9)
    # clearly oblong (10%) -> the longer-edge rule takes over and picks the Y edge
    oblong = square.copy()
    oblong[2:, 1] = 550.0
    assert abs(float(workframe_from_rectangle(oblong, [0, 0, 1])[:3, 0] @
                     np.array([0, 1.0, 0]))) > 0.999


def test_workframe_matches_the_fitted_plane_path():
    """The fused-scan path (plane.work_plane_from_points) must land on this frame."""
    from tasni.modules.scan.plane import work_plane_from_points
    xs, ys = np.meshgrid(np.linspace(200, 600, 60), np.linspace(100, 400, 45))
    pts = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(xs.size)])
    wp = work_plane_from_points(pts, distance=3.0, min_inlier_frac=0.5)
    assert np.allclose(wp.frame_T, workframe_from_rectangle(wp.corners, wp.normal),
                       atol=1e-9)


class _Cam:
    K = np.array([[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]])
    dist = np.zeros(5)


def test_calibration_id_stable_and_sensitive():
    a, b = camera_calibration_id(_Cam()), camera_calibration_id(_Cam())
    assert a == b and a.startswith("cam-") and len(a) == 16
    changed = _Cam()
    changed.K = _Cam.K + 1.0
    assert camera_calibration_id(changed) != a


def test_locked_survey_to_dict_roundtrips_geometry():
    snap = _snapshot()
    corners = order_corners_clockwise(_RECT, [0, 0, 1])
    survey = LockedWorkframeSurvey(
        mode=MODE_COMPACT, boundary_provenance=PROVENANCE_BY_MODE[MODE_COMPACT],
        captures=(_record(),), plane_normal_base=(0, 0, 1), plane_point_base=(600, 400, 0),
        corners_base=tuple(map(tuple, corners)), center_base=(600, 400, 0),
        frame_T_base=tuple(map(tuple, workframe_from_rectangle(corners, [0, 0, 1]))),
        size_mm=(1200.0, 800.0), quality={"plane_rms_mm": 0.8}, calibration_id="cam-abc",
        locked_robot=snap, locked_at=100.0)
    d = survey.to_dict()
    assert d["boundary_provenance"] == PROVENANCE_COMPACT
    assert np.asarray(d["corners_base"]).shape == (4, 3)
    assert survey.corners_np().shape == (4, 3) and survey.frame_np().shape == (4, 4)
