
import socket
import struct
import cv2
import io
import numpy as np
import lz4.frame as lz4f
from turbojpeg import TurboJPEG
import robolink
import robodk.robomath as robomath
import time
import logging
SERVER_IP_ADDRESS = '10.5.5.19'
PORT = 1024

#ROBOT CONNECTION
RDK = robolink.Robolink()
station_path = r"C:\Users\User\Desktop\RAFFI NO TOUCH\backuprobodk\230808_roboDrawq.rdk"
RDK.AddFile(station_path)
robot = RDK.Item('KUKA KR150 R2700')
success = robot.Connect()
status, status_msg = robot.ConnectedState()
RDK.setRunMode(6)

buffer = bytearray()
jpeg = TurboJPEG()


cv2.namedWindow("ColorMap", cv2.WINDOW_NORMAL)
HANDEYE_CHESS_SIZE = (5, 7)  # X/Y
HANDEYE_SQUARE_SIZE = 35.4  # mm
HANDEYE_MARKER_SIZE = 21.3  # mm




def receive_data(sock):
    # Initialize logging
    logging.basicConfig(level=logging.DEBUG)

    # First, receive the length of the intrinsics data.
    intrinsics_length_data = receive_exact(sock, 4)
    if not intrinsics_length_data:
        logging.error("Failed to receive intrinsics length.")
        return None, None, None, None, None

    intrinsics_length = struct.unpack('<I', intrinsics_length_data)[0]
    logging.info(f"Expected intrinsics data length: {intrinsics_length}")

    # Then, receive the intrinsics data itself.
    intrinsics_data = receive_exact(sock, intrinsics_length)
    if not intrinsics_data:
        logging.error("Failed to receive intrinsics data.")
        return None, None, None, None, None

    # Split and extract the intrinsics from the received data.
    half_length = intrinsics_length // 2
    mtx_buffer = io.BytesIO(intrinsics_data[:half_length])
    dist_buffer = io.BytesIO(intrinsics_data[half_length:])
    print(f"Received intrinsics data length: {len(intrinsics_data)}")
    try:
        color_mtx = np.load(mtx_buffer)
        dist_coeff = np.load(dist_buffer)
    except Exception as e:
        logging.error(f"Error loading numpy data: {e}")
        return None, None, None, None, None

    # Now, receive the header for frame data.
    header = receive_exact(sock, 16)
    if len(header) == 0:
        logging.error("Received empty header.")
        return None, None, None, None, None

    length_depth, length_color, timestamp = struct.unpack('<IId', header)
    logging.info(f"Depth data length: {length_depth}, Color data length: {length_color}, Timestamp: {timestamp}")

    # Receive the depth and color data.
    depth_data = receive_exact(sock, length_depth)
    color_data = receive_exact(sock, length_color)
    print(f"Received depth data length: {len(depth_data)}")
    print(f"Received color data length: {len(color_data)}")
    if not depth_data or not color_data:
        logging.error("Failed to receive frame data.")
        return None, None, None, None, None

    buffer = depth_data + color_data

    return buffer, length_depth, length_color, color_mtx, dist_coeff



def receive_exact(sock, n):
    data = bytearray()
    while len(data) < n:
        try:
            packet = sock.recv(n - len(data))
            if not packet:
                print("Connection closed by the server.")
                return None
        except socket.error as e:
            print(f"Socket error while receiving data: {e}")
            return None
        data.extend(packet)
    return data

def decompress_and_resize(buffer, length_depth, length_color):
    color_compressed = buffer[length_depth:length_depth + length_color]
    color_data = jpeg.decode(color_compressed)
    return color_data

def handle_frame(bigColor, corners, ids, gray,  target_name, trans_vec, rot_vec, color_mtx, dist_coeff):
    if bigColor is None:
        cv2.imshow("ColorMap", bigColor)
        cv2.waitKey(1)
        return
    draw_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    cv2.aruco.drawDetectedCornersCharuco(draw_img, corners, ids)
    bigColor = cv2.drawFrameAxes(draw_img, color_mtx, dist_coeff, rot_vec, trans_vec, max(1.5 * HANDEYE_SQUARE_SIZE, 5))
    cv2.putText(bigColor, f'{target_name}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
    cv2.imshow("ColorMap", bigColor)
    cv2.waitKey(1)


def process_markers(color_data, color_mtx, dist_coeff):
    gray = cv2.cvtColor(color_data, cv2.COLOR_BGR2GRAY)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_100)
    charuco_board = cv2.aruco.CharucoBoard((HANDEYE_CHESS_SIZE[0], HANDEYE_CHESS_SIZE[1]), HANDEYE_SQUARE_SIZE, HANDEYE_MARKER_SIZE, dictionary)
    marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray, dictionary)
    if marker_ids is None or len(marker_ids) < 1:
        raise Exception("No charucoboard found")
    retval, corners, ids = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, gray, charuco_board)
    if retval < 1 or len(ids) < 1:
        raise Exception("No charucoboard found")
    rot_vec = np.zeros((3, 1))
    trans_vec = np.zeros((3, 1))
    retval, rot_vec, trans_vec = cv2.aruco.estimatePoseCharucoBoard(corners, ids, charuco_board, color_mtx, dist_coeff, rot_vec, trans_vec)
    if not retval:
        raise Exception("Charucoboard pose not found")
    R_target2cam, _ = cv2.Rodrigues(rot_vec)

    return corners, ids, gray, R_target2cam, trans_vec, rot_vec
def get_all_target_names():
    all_items = RDK.ItemList(robolink.ITEM_TYPE_TARGET)
    return sorted([item.Name() for item in all_items if item.Name().startswith('Target')])

target_names = get_all_target_names()


def robot_poses():
    poses = {}
    for target_name in target_names:
        target_item = RDK.Item(target_name, robolink.ITEM_TYPE_TARGET)
        pose_target = target_item.Pose()
        poses[target_name] = pose_target
    return poses


def compute_transformation(R_target2cam_by_target, t_target2cam_by_target, robot_pose_matrices):
    R_target2cam_list = [R for _, R in sorted(R_target2cam_by_target.items())]
    t_target2cam_list = [t for _, t in sorted(t_target2cam_by_target.items())]

    R_gripper2base_list = []
    t_gripper2base_list = []
    for pose in robot_pose_matrices.values():
        R, t = pose_2_Rt(pose)
        R_gripper2base_list.append(R)
        t_gripper2base_list.append(t)

    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(R_gripper2base_list, t_gripper2base_list, R_target2cam_list, t_target2cam_list,method=cv2.CALIB_HAND_EYE_TSAI)
    return R_cam2gripper, t_cam2gripper

def pose_2_Rt(pose: robomath.Mat):
    pose_inv = pose.inv()
    R = np.array(pose_inv.Rot33())
    t = np.array(pose.Pos())
    return R, t

def Rt_2_pose(R, t):
    vx, vy, vz = R.tolist()
    cam_pose = robomath.eye(4)
    cam_pose.setPos([0, 0, 0])
    cam_pose.setVX(vx)
    cam_pose.setVY(vy)
    cam_pose.setVZ(vz)
    pose = cam_pose.inv()
    pose.setPos(t.tolist())
    return pose
def main():
    R_target2cam_by_target = {}
    t_target2cam_by_target = {}
    for target_name in target_names:
        robot.MoveJ(RDK.Item(target_name))
        time.sleep(1)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(10.0)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            try:
                s.connect((SERVER_IP_ADDRESS, PORT))
                s.sendall("RESOLUTION:1280x720".encode('utf-8'))  # Example for 1280x720 resolution
                buffer, length_depth, length_color, color_mtx, dist_coeff = receive_data(s)
                bigColor = decompress_and_resize(buffer, length_depth, length_color)
                if bigColor is not None:
                    corners, ids, gray, R_target2cam, t_target2cam, rot_vec = process_markers(bigColor, color_mtx, dist_coeff)
                    R_target2cam_by_target[target_name] = R_target2cam
                    t_target2cam_by_target[target_name] = t_target2cam
                    handle_frame(bigColor, corners, ids, gray,  target_name, t_target2cam, rot_vec, color_mtx, dist_coeff)

            except socket.timeout:
                print(f"Socket operation timed out for target {target_name}.")
            except socket.error as e:
                print(f"Socket error for target {target_name}: {e}")
            finally:
                s.shutdown(socket.SHUT_RDWR)
                s.close()
    R_cam2gripper, t_cam2gripper = compute_transformation(R_target2cam_by_target, t_target2cam_by_target, robot_poses())
    camera_pose = Rt_2_pose(R_cam2gripper, t_cam2gripper)
    tool = RDK.ItemUserPick('Select the "eye" tool to calibrate', RDK.ItemList(robolink.ITEM_TYPE_TOOL))
    tool.setPoseTool(camera_pose)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
    print("Calibration completed!")
    cv2.destroyAllWindows()

