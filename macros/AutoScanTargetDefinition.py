from robodk.robolink import *  # RoboDK API
from robodk.robomath import *  # Math toolbox
import math

# Initialize RoboDK
RDK = Robolink()

# Select the robot
robot = RDK.Item('KUKA KR150 R2700')
if not robot.Valid():
    raise Exception("No robot selected or found.")

# Interactive selection of reference frame and tool
reference_frame = RDK.ItemUserPick("Select a reference frame", ITEM_TYPE_FRAME)
if not reference_frame.Valid():
    raise Exception("No reference frame selected.")

tool = RDK.ItemUserPick("Select a tool", ITEM_TYPE_TOOL)
if not tool.Valid():
    raise Exception("No tool selected.")

# Set the robot's reference frame and tool
robot.setPoseFrame(reference_frame)
robot.setPoseTool(tool)

# Get the robot's current TCP pose (position and orientation)
current_pose = robot.Pose()  # TCP pose relative to the base frame
current_position = current_pose.Pos()  # Extract XYZ coordinates

# Parameters for generating targets
camera_distance = 800  # Radius of the dome in mm
num_azimuth = 6  # Number of points around the horizontal (360 degrees)
num_elevation = 2  # Number of vertical levels
bottom_angle = 65  # Angle from horizontal where bottom rim starts (degrees)
top_angle = 75  # Angle where top rim/point ends (degrees)
                # 90° = single point at top, lower values create a rim

# Helper function: Compute a pose aligned with the normal
def pose_from_normal(normal_vector, position):
    """
    Create a RoboDK pose aligned with the normal vector.
    The Z-axis of the pose will be aligned with the normal vector.
    """
    x, y, z = normal_vector

    # Ensure Z-axis (normal) is a unit vector
    normal = normalize3([x, y, z])

    # Compute a perpendicular vector for the X-axis
    arbitrary = [1, 0, 0] if abs(normal[0]) < 0.9 else [0, 1, 0]
    x_axis = normalize3([
        arbitrary[1] * normal[2] - arbitrary[2] * normal[1],
        arbitrary[2] * normal[0] - arbitrary[0] * normal[2],
        arbitrary[0] * normal[1] - arbitrary[1] * normal[0],
    ])

    # Compute the Y-axis as the cross product of Z (normal) and X
    y_axis = [
        normal[1] * x_axis[2] - normal[2] * x_axis[1],
        normal[2] * x_axis[0] - normal[0] * x_axis[2],
        normal[0] * x_axis[1] - normal[1] * x_axis[0],
    ]

    # Create the rotation matrix
    rotation_matrix = Mat([
        [x_axis[0], y_axis[0], normal[0], position[0]],
        [x_axis[1], y_axis[1], normal[1], position[1]],
        [x_axis[2], y_axis[2], normal[2], position[2]],
        [0, 0, 0, 1]
    ])

    return rotation_matrix

# Generate dome-shaped targets
capture_points = []

# Dome center is camera_distance below the TCP
dome_center = [
    current_position[0],
    current_position[1],
    current_position[2] - camera_distance  # Center is one radius below TCP
]

# Calculate elevation ranges based on angles
min_elevation = bottom_angle
max_elevation = top_angle
elevation_range = max_elevation - min_elevation

# Loop through elevation angles
for i in range(num_elevation):
    # Elevation from bottom_angle to top_angle
    elevation = min_elevation + (i * elevation_range / (num_elevation - 1))
    
    # Adjust number of azimuth points based on elevation
    # More points at lower elevations, fewer at higher elevations
    if elevation >= 70:
        current_num_azimuth = max(4, num_azimuth // 2)  # Minimum 4 points for high elevations
    else:
        current_num_azimuth = num_azimuth
    
    for j in range(current_num_azimuth):
        azimuth = j * (360 / current_num_azimuth)
        
        # Convert spherical coordinates to Cartesian
        x = camera_distance * math.cos(math.radians(elevation)) * math.cos(math.radians(azimuth))
        y = camera_distance * math.cos(math.radians(elevation)) * math.sin(math.radians(azimuth))
        z = camera_distance * math.sin(math.radians(elevation))
        
        # Compute the target position relative to the dome center
        target_position = [dome_center[0] + x, dome_center[1] + y, dome_center[2] + z]

        # Only add points that are below TCP height
        if target_position[2] <= current_position[2]:
            # Compute the normal vector (point towards dome center)
            normal_vector = normalize3([
                dome_center[0] - target_position[0],
                dome_center[1] - target_position[1],
                dome_center[2] - target_position[2]
            ])

            # Align the target pose to the normal
            target_pose = pose_from_normal(normal_vector, target_position)

            # Append the target pose
            capture_points.append(target_pose)

# If top_angle is 90, add a single point at the top
if abs(top_angle - 90) < 1:  # Allow for small floating-point differences
    top_position = [dome_center[0], dome_center[1], dome_center[2] + camera_distance]
    top_normal = normalize3([
        dome_center[0] - top_position[0],
        dome_center[1] - top_position[1],
        dome_center[2] - top_position[2]
    ])
    top_pose = pose_from_normal(top_normal, top_position)
    capture_points.append(top_pose)

# Create targets in RoboDK
for idx, target in enumerate(capture_points):
    target_name = f"CaptureTarget{idx + 1}"
    target_item = RDK.AddTarget(target_name, reference_frame)
    target_item.setPose(target)

print(f"Generated {len(capture_points)} unique capture targets in dome configuration.")
print(f"Dome spans from {bottom_angle}° to {top_angle}° elevation.")