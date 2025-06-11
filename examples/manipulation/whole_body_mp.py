#!/usr/bin/env python3
import rospy
import time
import numpy as np
import open3d as o3d
import os
import math
import tf.transformations
from fetch.fetch import Fetch

# Configuration for testing
TARGET_ENVIRONMENT = (
    "open_fridge"  # Options: workstation, table, open_kitchen, coffee_table, sofa
)
USE_POINTCLOUD = True  # Set to False to skip pointcloud loading
ENABLE_COLLISION_CHECKING = True  # Set to False to skip collision checking
TRAJECTORY_DURATION = 15.0  # Duration for executing whole body motion (seconds)

# List of all PCD files to load
PCD_FILES = [
    "coffee_table.ply",
    "open_kitchen.ply",
    "rls_2.ply",
    "sofa.ply",
    "table.ply",
    "wall.ply",
    "workstation.ply",
]

# Test case definitions for different environments
TEST_CASES = {
    "open_fridge": {
        "target_joints": [0.37, 0.29470240364379885,
            1.1211147828796386,
            -1.2186898401245116,
            1.876310651525879,
            1.202369851473999,
            -1.8404571460021972,
            1.7227418721292114],
        "position": [-4.296061810063319, -2.2113403585598902, 0.0],
        "orientation": [0.0, 0.0, -0.895797459720963, 0.4444624969077478]
    }
}


def load_pointcloud(pcd_path):
    """
    Load a point cloud from file.

    Args:
        pcd_path: Path to the point cloud file (.ply, .pcd)

    Returns:
        numpy.ndarray: Nx3 array of points or None if loading fails
    """
    rospy.loginfo(f"Loading point cloud from {pcd_path}...")

    try:
        # Check if file exists
        if not os.path.exists(pcd_path):
            rospy.logerr(f"Point cloud file not found: {pcd_path}")
            return None

        # Load the point cloud using Open3D
        try:
            pcd = o3d.io.read_point_cloud(pcd_path)
            points = np.asarray(pcd.points)

            # Check if point cloud is valid
            if len(points) == 0:
                rospy.logerr(f"Empty point cloud loaded from {pcd_path}")
                return None

            rospy.loginfo(f"Loaded {len(points)} points from {pcd_path}")
            return points

        except Exception as e:
            rospy.logerr(f"Failed to load point cloud with Open3D: {e}")

            # Fallback: Try to load as raw data
            rospy.loginfo("Attempting to load point cloud as raw data...")
            try:
                # Simple PLY parser for ASCII format (fallback)
                if pcd_path.lower().endswith(".ply"):
                    with open(pcd_path, "r") as f:
                        lines = f.readlines()

                    # Parse header
                    vertex_count = 0
                    data_start = 0
                    for i, line in enumerate(lines):
                        if "element vertex" in line:
                            vertex_count = int(line.split()[-1])
                        if "end_header" in line:
                            data_start = i + 1
                            break

                    # Parse points
                    points = []
                    for i in range(data_start, data_start + vertex_count):
                        if i < len(lines):
                            values = lines[i].split()
                            if len(values) >= 3:
                                points.append(
                                    [
                                        float(values[0]),
                                        float(values[1]),
                                        float(values[2]),
                                    ]
                                )

                    points = np.array(points)
                    if len(points) == 0:
                        rospy.logerr("Failed to extract points from PLY file")
                        return None

                    rospy.loginfo(f"Loaded {len(points)} points using fallback method")
                    return points
                else:
                    rospy.logerr(
                        f"Unsupported file format for fallback loading: {pcd_path}"
                    )
                    return None
            except Exception as e2:
                rospy.logerr(f"Fallback loading also failed: {e2}")
                return None

    except Exception as e:
        rospy.logerr(f"Error loading point cloud: {e}")
        import traceback

        rospy.logerr(traceback.format_exc())
        return None


def get_current_base_pose(robot):
    """
    Get the current base pose of the robot using TF.

    Args:
        robot: Fetch robot instance

    Returns:
        list: [x, y, theta] representing the current base pose
    """
    try:
        transform = robot.tf_buffer.lookup_transform(
            "map", "base_link", rospy.Time(0), rospy.Duration(5.0)
        )
        trans = transform.transform.translation
        rot = transform.transform.rotation

        # Convert quaternion to Euler angles (yaw)
        quaternion = [rot.x, rot.y, rot.z, rot.w]
        euler = tf.transformations.euler_from_quaternion(quaternion)
        yaw = euler[2]  # Get the yaw (rotation around z-axis)

        return [trans.x, trans.y, yaw]
    except Exception as e:
        rospy.logwarn(f"Could not get current base pose: {e}")
        # Return default values if transform lookup fails
        return [0.0, 0.0, 0.0]


def main():
    """
    Test script for Fetch robot whole body motion planning and execution.
    This script demonstrates:
    1. Loading point clouds for collision avoidance
    2. Planning whole body motion (arm + base)
    3. Executing synchronized arm and base movements
    """
    rospy.init_node("fetch_whole_body_test", anonymous=True)

    # Initialize the robot
    robot = Fetch()
    rospy.loginfo("Fetch robot initialized")

    # Use the selected test case
    if TARGET_ENVIRONMENT not in TEST_CASES:
        rospy.logerr(
            f"Unknown environment: {TARGET_ENVIRONMENT}. Using 'workstation' instead."
        )
        target_env = "workstation"
    else:
        target_env = TARGET_ENVIRONMENT

    test_case = TEST_CASES[target_env]
    rospy.loginfo(f"Selected environment: {target_env}")

    # Convert quaternion orientation to target_base format
    position = test_case["position"]
    orientation = test_case["orientation"]
    
    # Extract x, y from position
    x, y, z = position
    
    # Convert quaternion to yaw
    qx, qy, qz, qw = orientation
    _, _, theta = tf.transformations.euler_from_quaternion([qx, qy, qz, qw])
    theta=-1.8
    
    # Create target_base in the expected [x, y, theta] format
    target_base = [x, y, theta]

    # Update current base position from TF
    current_base = get_current_base_pose(robot)
    rospy.loginfo(
        f"Current base pose from TF: x={current_base[0]:.4f}, y={current_base[1]:.4f}, theta={current_base[2]:.4f}"
    )

    # Update the robot's internal base parameters
    robot.set_base_params(*current_base)

    # STEP 2: Load point clouds for collision avoidance (if enabled)
    if USE_POINTCLOUD:
        rospy.loginfo(
            "=== STEP 2: Loading point clouds for collision avoidance ==="
        )

        # Track total points and loading time
        total_points = 0
        pc_load_start_time = time.time()

        # We'll combine all point clouds into one for processing
        combined_points = None

        for pcd_file in PCD_FILES:
            # Path to point cloud
            pcd_path = f"resources/mp_collision_models/{pcd_file}"

            # Load point cloud data from file
            point_cloud_data = load_pointcloud(pcd_path)

            if point_cloud_data is not None:
                # Keep track of total points
                total_points += len(point_cloud_data)

                # Add to combined point cloud
                if combined_points is None:
                    combined_points = point_cloud_data
                else:
                    combined_points = np.vstack((combined_points, point_cloud_data))

        pc_load_time = time.time() - pc_load_start_time

        if combined_points is None or len(combined_points) == 0:
            rospy.logwarn(
                "Failed to load any point clouds. Proceeding without collision model."
            )
        else:
            rospy.loginfo(
                f"All point clouds loaded in {pc_load_time:.2f} seconds with {len(combined_points)} points"
            )

            if ENABLE_COLLISION_CHECKING:
                # Process the combined point cloud data
                pc_process_start_time = time.time()
                result = robot.add_pointcloud(combined_points)
                pc_process_time = time.time() - pc_process_start_time

                if result < 0:
                    rospy.logwarn(
                        "Failed to process point cloud. Proceeding without collision model."
                    )
                else:
                    rospy.loginfo(
                        f"Point cloud processed in {pc_process_time:.2f} seconds"
                    )
            else:
                rospy.loginfo(
                    "Collision checking disabled. Skipping point cloud processing."
                )
    else:
        rospy.loginfo("Point cloud loading disabled. Skipping...")

    # STEP 3: Plan and execute whole body motion
    rospy.loginfo("=== STEP 3: Planning and executing whole body motion ===")

    # Get target configuration
    target_joints = test_case["target_joints"]

    # Log target configuration
    rospy.loginfo("Target joint configuration:")
    for name, value in zip(robot.planning_joint_names, target_joints):
        rospy.loginfo(f"  {name}: {value:.4f}")

    rospy.loginfo(
        f"Target base configuration: x={target_base[0]:.4f}, y={target_base[1]:.4f}, theta={target_base[2]:.4f}"
    )

    # Get current joint positions
    for i in range(5):
        current_joints = robot.get_current_planning_joints()
        if current_joints is None:
            rospy.logerr(
                "Failed to get current joint positions. Retrying."
            )
            time.sleep(0.2)
        else:
            break
    if current_joints is None:
        rospy.logerr(
            "Failed to get current joint positions. Aborting motiong planning."
        )
        return

    rospy.loginfo("Current joint configuration:")
    for name, value in zip(robot.planning_joint_names, current_joints):
        rospy.loginfo(f"  {name}: {value:.4f}")

    # Start motion planning timer
    planning_start_time = time.time()

    # Plan whole body motion
    plan_result = robot.plan_whole_body_motion(
        current_joints, target_joints, current_base, target_base
    )

    planning_time = time.time() - planning_start_time

    if plan_result is None or not plan_result["success"]:
        rospy.logerr("Whole body motion planning failed!")
        return

    rospy.loginfo(
        f"Whole body motion planning completed in {planning_time:.2f} seconds"
    )

    # Log plan statistics
    stats = plan_result["stats"]
    rospy.loginfo("Planning statistics:")
    for key, value in stats.items():
        rospy.loginfo(f"  {key}: {value}")

    # Log motion plan details
    arm_path_len = len(plan_result["arm_path"]) if plan_result["arm_path"] else 0
    base_configs_len = (
        len(plan_result["base_configs"]) if plan_result["base_configs"] else 0
    )
    rospy.loginfo(f"Arm path length: {arm_path_len} waypoints")
    rospy.loginfo(f"Base path length: {base_configs_len} waypoints")

    # Sample and log a few waypoints for debugging
    if arm_path_len > 0 and base_configs_len > 0:
        sample_indices = [
            0,
            min(5, arm_path_len - 1),
            min(10, arm_path_len - 1),
            arm_path_len - 1,
        ]
        rospy.loginfo("Sample waypoints from the plan:")

        for idx in sample_indices:
            if idx < arm_path_len and idx < base_configs_len:
                arm_point = plan_result["arm_path"][idx]
                base_point = plan_result["base_configs"][idx]

                # Format the arm point for display
                if isinstance(arm_point, list):
                    arm_str = ", ".join([f"{val:.3f}" for val in arm_point])
                elif isinstance(arm_point, np.ndarray):
                    arm_str = ", ".join(
                        [f"{val:.3f}" for val in arm_point.tolist()]
                    )
                else:  # Assume it has a to_list method
                    arm_str = ", ".join(
                        [f"{val:.3f}" for val in arm_point.to_list()]
                    )

                # Format the base point
                base_str = f"[{base_point[0]:.3f}, {base_point[1]:.3f}, {base_point[2]:.3f}]"

                rospy.loginfo(f"  Waypoint {idx}:")
                rospy.loginfo(f"    Arm: [{arm_str}]")
                rospy.loginfo(f"    Base: {base_str}")

    # STEP 4: Execute the planned motion
    rospy.loginfo("=== STEP 4: Executing whole body motion ===")

    # Before execution, log expected motion duration
    rospy.loginfo(
        f"Executing motion with duration: {TRAJECTORY_DURATION:.2f} seconds"
    )

    # Execute the planned motion
    robot.execute_whole_body_motion(
        plan_result["arm_path"], plan_result["base_configs"], TRAJECTORY_DURATION
    )

    # STEP 5: Verify final pose
    rospy.loginfo("=== STEP 5: Verifying final pose ===")

    # Wait briefly for everything to settle
    rospy.sleep(0.5)

    # Get final joint positions
    final_joints = robot.get_current_planning_joints()
    final_base = get_current_base_pose(robot)

    if final_joints is not None:
        # Calculate joint error
        joint_errors = [
            abs(final - target)
            for final, target in zip(final_joints, target_joints)
        ]
        max_joint_error = max(joint_errors)
        avg_joint_error = sum(joint_errors) / len(joint_errors)

        rospy.loginfo("Final joint position errors:")
        for i, (name, error) in enumerate(
            zip(robot.planning_joint_names, joint_errors)
        ):
            rospy.loginfo(f"  {name}: {error:.4f} rad")

        rospy.loginfo(f"Maximum joint error: {max_joint_error:.4f} rad")
        rospy.loginfo(f"Average joint error: {avg_joint_error:.4f} rad")

    # Calculate base position error
    base_pos_error = math.sqrt(
        (final_base[0] - target_base[0]) ** 2
        + (final_base[1] - target_base[1]) ** 2
    )

    # Calculate base orientation error (accounting for angle wrapping)
    base_ori_error = abs(final_base[2] - target_base[2])
    if base_ori_error > math.pi:
        base_ori_error = 2 * math.pi - base_ori_error

    rospy.loginfo("Final base pose errors:")
    rospy.loginfo(f"  Position error: {base_pos_error:.4f} m")
    rospy.loginfo(f"  Orientation error: {base_ori_error:.4f} rad")

    # STEP 6: Summary
    rospy.loginfo("=== SUMMARY ===")
    rospy.loginfo(f"Environment: {target_env}")
    # rospy.loginfo(f"Initial pose time: {initial_time:.2f} seconds")
    rospy.loginfo(f"Planning time: {planning_time:.2f} seconds")

    rospy.loginfo("Test completed")


if __name__ == "__main__":
    main()
