#!/usr/bin/env python3
import rospy
import time
import numpy as np
import math
import tf.transformations
from geometry_msgs.msg import Pose
from fetch.fetch import Fetch

# Select which test case to run (0-4)
TEST_CASE_INDEX = 2  # Change this value to select a specific test case

# Configuration for testing
USE_POINTCLOUD = True  # Set to False to skip pointcloud loading
ENABLE_COLLISION_CHECKING = True  # Set to True to enable collision checking
TRAJECTORY_DURATION = 15.0  # Duration for executing whole body motion (seconds)

# Test cases for end effector poses: [x, y, z, qx, qy, qz, qw]
TEST_CASES = [
    [-3.66, -3.23, 1.27, 0.427, -0.726, -0.476, 0.254],
    [-3.66, -3.23, 1.27, 0.0, 1.0, 0.0, 0.0],
    [-2.75, -0.79, 1.26, 0.42737215, -0.7255706, -0.47561587, 0.25434207],
    [-2.75, -0.79, 1.26, 1.0, 0.0, 0.0, 0.0],
    [-2.75, -0.79, 1.26, 0.0, 1.0, 0.0, 0.0],
]

# Initial configuration to start from
INITIAL_CONFIG = {
    "joints": [0.1, 1.32, 1.4, -0.2, 1.72, 0, 1.66, 0],  # torso + 7 arm joints
    "base": [0.0, 0.0, 0.0],  # [x, y, theta]
}

# Point cloud files to load for collision checking
PCD_FILES = [
    "coffee_table.ply",
    "open_kitchen.ply",
    "rls_2.ply",
    "sofa.ply",
    "table.ply",
    "wall.ply",
    "workstation.ply",
]


def load_pointcloud(pcd_path):
    """
    Load a point cloud from file.

    Args:
        pcd_path: Path to the point cloud file (.ply, .pcd)

    Returns:
        numpy.ndarray: Nx3 array of points or None if loading fails
    """
    import os
    import open3d as o3d

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
            return None

    except Exception as e:
        rospy.logerr(f"Error loading point cloud: {e}")
        import traceback

        rospy.logerr(traceback.format_exc())
        return None


def load_all_pointclouds():
    """
    Load and combine all point clouds for collision checking.

    Returns:
        numpy.ndarray: Combined point cloud or None if loading fails
    """
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
        return None
    else:
        rospy.loginfo(
            f"All point clouds loaded in {pc_load_time:.2f} seconds with {len(combined_points)} points"
        )
        return combined_points


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


def create_pose_from_array(pose_array):
    """
    Create a ROS Pose message from a pose array.

    Args:
        pose_array: [x, y, z, qx, qy, qz, qw] array

    Returns:
        geometry_msgs.msg.Pose: ROS Pose message
    """
    pose = Pose()
    pose.position.x = pose_array[0]
    pose.position.y = pose_array[1]
    pose.position.z = pose_array[2]
    pose.orientation.x = pose_array[3]
    pose.orientation.y = pose_array[4]
    pose.orientation.z = pose_array[5]
    pose.orientation.w = pose_array[6]
    return pose


def verify_pose_accuracy(
    robot, target_pose, max_position_error=0.05, max_orientation_error=0.1
):
    """
    Verify the accuracy of the final end-effector pose.

    Args:
        robot: Fetch robot instance
        target_pose: Target Pose message
        max_position_error: Maximum allowable position error (meters)
        max_orientation_error: Maximum allowable orientation error (radians)

    Returns:
        tuple: (position_error, orientation_error, success)
    """
    # Get current configuration
    current_joints = robot.get_current_planning_joints()
    if current_joints is None:
        rospy.logerr("Failed to get current joint positions for pose verification")
        return None, None, False

    # Get current base position
    current_base = robot.get_base_params()

    # Get current end effector pose
    current_pose = robot.get_end_effector_pose(current_joints, current_base)
    if current_pose is None:
        rospy.logerr("Failed to get current end effector pose")
        return None, None, False

    # Calculate position error
    position_error = math.sqrt(
        (current_pose.position.x - target_pose.position.x) ** 2
        + (current_pose.position.y - target_pose.position.y) ** 2
        + (current_pose.position.z - target_pose.position.z) ** 2
    )

    # Calculate orientation error using quaternion distance
    # Dot product between the quaternions gives the cosine of half the rotation angle
    dot_product = (
        current_pose.orientation.x * target_pose.orientation.x
        + current_pose.orientation.y * target_pose.orientation.y
        + current_pose.orientation.z * target_pose.orientation.z
        + current_pose.orientation.w * target_pose.orientation.w
    )

    # Ensure dot product is within valid range for acos
    dot_product = max(min(dot_product, 1.0), -1.0)

    # Calculate the angle between the quaternions
    orientation_error = 2 * math.acos(abs(dot_product))

    # Check if within tolerance
    success = (
        position_error <= max_position_error
        and orientation_error <= max_orientation_error
    )

    return position_error, orientation_error, success


def run_ik_test(robot, test_case_idx, pose_array):
    """
    Run a single IK test case.

    Args:
        robot: Fetch robot instance
        test_case_idx: Index of the test case
        pose_array: [x, y, z, qx, qy, qz, qw] target pose array

    Returns:
        dict: Test results
    """
    # Create a ROS Pose message from the pose array
    target_pose = create_pose_from_array(pose_array)

    rospy.loginfo(f"\n=== Test Case {test_case_idx + 1} ===")
    rospy.loginfo(
        f"Target pose: position [{target_pose.position.x:.3f}, {target_pose.position.y:.3f}, {target_pose.position.z:.3f}]"
    )
    rospy.loginfo(
        f"Target pose: orientation [{target_pose.orientation.x:.3f}, {target_pose.orientation.y:.3f}, {target_pose.orientation.z:.3f}, {target_pose.orientation.w:.3f}]"
    )

    # Store test results
    results = {
        "test_case_idx": test_case_idx,
        "target_pose": pose_array,
        "ik_success": False,
        "planning_success": False,
        "execution_success": False,
        "final_success": False,
        "ik_time": 0.0,
        "planning_time": 0.0,
        "execution_time": 0.0,
        "position_error": None,
        "orientation_error": None,
    }

    # STEP 1: Solve whole-body IK
    rospy.loginfo("Solving whole-body IK...")
    ik_start_time = time.time()

    ik_solution = robot.solve_whole_body_ik(target_pose, max_attempts=100)

    ik_time = time.time() - ik_start_time
    results["ik_time"] = ik_time

    if ik_solution is None:
        rospy.logerr(f"Whole-body IK failed for test case {test_case_idx + 1}")
        rospy.loginfo("Moving to next test case...\n")
        return results

    results["ik_success"] = True

    # Extract goal configuration
    goal_base = ik_solution["base_config"]
    goal_joints = ik_solution["arm_config"]

    rospy.loginfo(
        f"IK solved in {ik_time:.2f} seconds, {ik_solution['attempts']} attempts"
    )
    rospy.loginfo(f"Goal base: {[round(v, 3) for v in goal_base]}")
    rospy.loginfo(f"Goal joints: {[round(v, 3) for v in goal_joints]}")

    # STEP 2: Plan whole-body motion
    rospy.loginfo("Planning whole-body motion...")

    # Get current joint positions
    current_joints = robot.get_current_planning_joints()
    if current_joints is None:
        rospy.logerr("Failed to get current joint positions. Aborting motion planning.")
        return results

    # Get current base position
    current_base = get_current_base_pose(robot)

    # Start motion planning timer
    planning_start_time = time.time()

    # Plan whole body motion
    plan_result = robot.plan_whole_body_motion(
        current_joints, goal_joints, current_base, goal_base
    )

    planning_time = time.time() - planning_start_time
    results["planning_time"] = planning_time

    if plan_result is None or not plan_result["success"]:
        rospy.logerr(
            f"Whole body motion planning failed for test case {test_case_idx + 1}"
        )
        rospy.loginfo("Moving to next test case...\n")
        return results

    results["planning_success"] = True

    rospy.loginfo(
        f"Whole body motion planning completed in {planning_time:.2f} seconds"
    )

    # Log plan statistics
    stats = plan_result["stats"]
    rospy.loginfo("Planning statistics:")
    for key, value in stats.items():
        rospy.loginfo(f"  {key}: {value}")

    # STEP 3: Execute the planned motion
    rospy.loginfo("Executing whole-body motion...")

    # Start execution timer
    execution_start_time = time.time()

    # Execute the planned motion
    execution_result = robot.execute_whole_body_motion(
        plan_result["arm_path"], plan_result["base_configs"]
    )

    execution_time = time.time() - execution_start_time
    results["execution_time"] = execution_time

    if not execution_result:
        rospy.logerr(
            f"Whole body motion execution failed for test case {test_case_idx + 1}"
        )
        rospy.loginfo("Moving to next test case...\n")
        return results

    results["execution_success"] = True

    rospy.loginfo(
        f"Whole body motion execution completed in {execution_time:.2f} seconds"
    )

    # STEP 4: Verify final pose
    rospy.loginfo("Verifying final pose...")

    # Wait briefly for everything to settle
    rospy.sleep(0.5)

    # Verify pose accuracy
    position_error, orientation_error, success = verify_pose_accuracy(
        robot, target_pose
    )

    if position_error is not None and orientation_error is not None:
        results["position_error"] = position_error
        results["orientation_error"] = orientation_error
        results["final_success"] = success

        rospy.loginfo(f"Position error: {position_error:.4f} m")
        rospy.loginfo(f"Orientation error: {orientation_error:.4f} rad")

        if success:
            rospy.loginfo("Final pose verification: SUCCESS")
        else:
            rospy.loginfo("Final pose verification: FAILED")
    else:
        rospy.logwarn("Could not verify final pose accuracy")

    rospy.loginfo(f"Test case {test_case_idx + 1} completed\n")
    return results


def move_to_initial_pose(robot):
    """
    Move robot to the initial pose.

    Args:
        robot: Fetch robot instance

    Returns:
        bool: True if successful, False otherwise
    """
    rospy.loginfo("=== Moving to initial pose ===")
    initial_joints = INITIAL_CONFIG["joints"]

    # Log initial joint configuration
    rospy.loginfo("Initial joint configuration:")
    for name, value in zip(robot.planning_joint_names, initial_joints):
        rospy.loginfo(f"  {name}: {value:.4f}")

    # Execute motion to initial pose
    initial_start_time = time.time()
    initial_result = robot.send_joint_values(initial_joints)
    initial_time = time.time() - initial_start_time

    if initial_result is not None:
        rospy.loginfo(f"Initial pose achieved in {initial_time:.2f} seconds")

        # Wait for things to settle
        rospy.sleep(1.0)

        # Update current base position from TF
        current_base = get_current_base_pose(robot)
        rospy.loginfo(
            f"Current base pose from TF: x={current_base[0]:.4f}, y={current_base[1]:.4f}, theta={current_base[2]:.4f}"
        )

        # Update the robot's internal base parameters
        robot.set_base_params(*current_base)
        return True
    else:
        rospy.logwarn("Failed to reach initial pose.")
        return False


def setup_collision_environment(robot):
    """
    Set up collision environment for motion planning.

    Args:
        robot: Fetch robot instance

    Returns:
        bool: True if successful, False otherwise
    """
    if not USE_POINTCLOUD:
        rospy.loginfo("Point cloud loading disabled. Skipping...")
        return True

    rospy.loginfo("=== Loading point clouds for collision avoidance ===")

    # Load point clouds
    combined_points = load_all_pointclouds()

    if combined_points is None or len(combined_points) == 0:
        rospy.logwarn(
            "Failed to load point clouds. Proceeding without collision model."
        )
        return False

    if ENABLE_COLLISION_CHECKING:
        # Process the combined point cloud data
        pc_process_start_time = time.time()
        result = robot.add_pointcloud(combined_points)
        pc_process_time = time.time() - pc_process_start_time

        if result < 0:
            rospy.logwarn(
                "Failed to process point cloud. Proceeding without collision model."
            )
            return False
        else:
            rospy.loginfo(f"Point cloud processed in {pc_process_time:.2f} seconds")
            return True
    else:
        rospy.loginfo("Collision checking disabled. Skipping point cloud processing.")
        return True


def main():
    """
    Main function for Fetch whole body IK and motion planning test.
    """
    rospy.init_node("fetch_whole_body_ik_test", anonymous=True)

    # Initialize the robot
    robot = Fetch()
    rospy.loginfo("Fetch robot initialized")

    # Validate test case index
    if TEST_CASE_INDEX < 0 or TEST_CASE_INDEX >= len(TEST_CASES):
        rospy.logerr(
            f"Invalid TEST_CASE_INDEX: {TEST_CASE_INDEX}. Must be between 0 and {len(TEST_CASES)-1}."
        )
        return

    try:
        # STEP 1: Set up collision environment
        setup_collision_environment(robot)
        
        # STEP 2: Move to initial pose
        if not move_to_initial_pose(robot):
            rospy.logwarn(
                "Failed to reach initial pose. Proceeding with test anyway..."
            )

        # STEP 3: Run the selected IK test
        test_case = TEST_CASES[TEST_CASE_INDEX]
        rospy.loginfo(f"\n=== Running IK Test Case {TEST_CASE_INDEX} ===")
        rospy.loginfo(
            f"Target pose: [{test_case[0]:.3f}, {test_case[1]:.3f}, {test_case[2]:.3f}, {test_case[3]:.3f}, {test_case[4]:.3f}, {test_case[5]:.3f}, {test_case[6]:.3f}]"
        )

        # Run the single test case
        result = run_ik_test(robot, TEST_CASE_INDEX, test_case)

        # STEP 4: Output test results
        rospy.loginfo("\n=== Test Results ===")
        rospy.loginfo(f"Test Case {TEST_CASE_INDEX}:")
        rospy.loginfo(
            f"  Target Position: [{result['target_pose'][0]:.3f}, {result['target_pose'][1]:.3f}, {result['target_pose'][2]:.3f}]"
        )
        rospy.loginfo(
            f"  Target Orientation: [{result['target_pose'][3]:.3f}, {result['target_pose'][4]:.3f}, {result['target_pose'][5]:.3f}, {result['target_pose'][6]:.3f}]"
        )
        rospy.loginfo(
            f"  IK Success: {result['ik_success']} ({result['ik_time']:.2f}s)"
        )
        rospy.loginfo(
            f"  Planning Success: {result['planning_success']} ({result['planning_time']:.2f}s)"
        )
        rospy.loginfo(
            f"  Execution Success: {result['execution_success']} ({result['execution_time']:.2f}s)"
        )

        if result["position_error"] is not None:
            rospy.loginfo(f"  Position Error: {result['position_error']:.4f}m")
            rospy.loginfo(f"  Orientation Error: {result['orientation_error']:.4f}rad")
            rospy.loginfo(f"  Final Success: {result['final_success']}")

        # Calculate total time
        total_time = (
            result["ik_time"] + result["planning_time"] + result["execution_time"]
        )
        rospy.loginfo(f"  Total Time: {total_time:.2f}s")

        rospy.loginfo("\nTest completed")

    except rospy.ROSInterruptException:
        rospy.logerr("Program interrupted!")
    except Exception as e:
        rospy.logerr(f"An error occurred: {e}")
        import traceback

        rospy.logerr(traceback.format_exc())


if __name__ == "__main__":
    main()
