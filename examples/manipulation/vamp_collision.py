#!/usr/bin/env python3
import rospy
import time
import numpy as np
import open3d as o3d
import os
import math
from fetch.fetch import Fetch

target_test = "workstation"

# List of all PCD files to load
pcd_files = [
    "coffee_table.ply",
    "open_kitchen.ply",
    "rls2.ply",
    "sofa.ply",
    "table.ply",
    "wall.ply",
    "workstation.ply",
]

test_cases = {
    "workstation": {
        "initial_target_joints": [0.1, 1.32, 1.4, -0.2, 1.72, 0, 1.66, 0],
        "position": [-2.7782226013936215, 0.15580237473099096, 0.0],
        "orientation": [0.0, 0.0, -0.6538142154802868, 0.7566551206698445],
    },
    "table": {
        "initial_target_joints": [0.1, 1.32, 1.4, -0.2, 1.72, 0, 1.66, 0],
        "position": [-1.4870152105885424, 1.3377760483626338, 0.0],
        "orientation": [0.0, 0.0, 0.9999908525888619, 0.004277234924690651],
    },
    "open_kitchen": {
        "initial_target_joints": [0.1, 1.32, 1.4, -0.2, 1.72, 0, 1.66, 0],
        "position": [-1.6795810749673257, -2.3876239483526103, 0.0],
        "orientation": [0.0, 0.0, -0.7056717966949518, 0.7085388594490203],
    },
    "coffee_table": {
        "initial_target_joints": [0.1, 1.32, 1.4, -0.2, 1.72, 0, 1.66, 0],
        "position": [3.8899313571135536, 0.9125618816816462, 0.0],
        "orientation": [0.0, 0.0, -0.71349853502434, 0.7006567208827164],
    },
    "sofa": {
        "initial_target_joints": [0.1, 1.32, 1.4, -0.2, 1.72, 0, 1.66, 0],
        "position": [2.098556770716819, 1.1490546885248516, 0.0],
        "orientation": [0.0, 0.0, -0.690454382206587, 0.7233759369039866],
    },
}


def load_pointcloud(pcd_path):
    """
    Load a point cloud from file.

    Args:
        pcd_path: Path to the point cloud file (.ply, .pcd)

    Returns:
        numpy.ndarray: Nx3 array of points or None if loading fails
    """
    print(f"Loading point cloud from {pcd_path}...")

    try:
        # Check if file exists
        if not os.path.exists(pcd_path):
            print(f"Point cloud file not found: {pcd_path}")
            return None

        # Load the point cloud using Open3D
        try:
            pcd = o3d.io.read_point_cloud(pcd_path)
            points = np.asarray(pcd.points)

            # Check if point cloud is valid
            if len(points) == 0:
                print(f"Empty point cloud loaded from {pcd_path}")
                return None

            print(f"Loaded {len(points)} points from {pcd_path}")
            return points

        except Exception as e:
            print(f"Failed to load point cloud with Open3D: {e}")

            # Fallback: Try to load as raw data
            print("Attempting to load point cloud as raw data...")
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
                        print("Failed to extract points from PLY file")
                        return None

                    print(f"Loaded {len(points)} points using fallback method")
                    return points
                else:
                    print(f"Unsupported file format for fallback loading: {pcd_path}")
                    return None
            except Exception as e2:
                print(f"Fallback loading also failed: {e2}")
                return None

    except Exception as e:
        print(f"Error loading point cloud: {e}")
        import traceback

        traceback.print_exc()
        return None


def main():
    """
    Test script for Fetch robot collision avoidance.
    Modified to load multiple point clouds for a more complete environment model and update the robot's base parameters
    using the current ROS transform before motion planning.
    """
    # Initialize the robot
    robot = Fetch()

    try:
        # STEP 0: First go to a good starting pose for further motion planning
        print("\n=== Step 0: Moving to a good starting pose ===")
        initial_target_joints = test_cases[target_test]["initial_target_joints"]

        print("Initial target joint configuration:")
        for name, value in zip(robot.planning_joint_names, initial_target_joints):
            print(f" {name}: {value}")

        # Start motion planning timer for initial pose
        initial_planning_start_time = time.time()

        # Execute motion to initial pose
        initial_result = robot.send_joint_values(initial_target_joints)

        # Calculate planning and execution time for initial pose
        initial_motion_time = time.time() - initial_planning_start_time

        if initial_result is not None:
            print(f"Initial motion completed in {initial_motion_time:.2f} seconds")
        else:
            print("Initial motion planning or execution failed!")
            # Continue anyway, as the robot might still be in a workable position

        # STEP 1: Navigate to the specified pose
        print("\n=== Step 1: Navigating to target position ===")
        # Target pose from your example
        position = test_cases[target_test]["position"]
        orientation = test_cases[target_test]["orientation"]

        print(f"Target position: {position}")
        print(f"Target orientation: {orientation}")

        # Start navigation timer
        nav_start_time = time.time()

        # Send navigation command
        robot.send_target_position(position, orientation)

        # Calculate navigation time
        nav_time = time.time() - nav_start_time
        print(f"Navigation completed in {nav_time:.2f} seconds")

        # STEP 2: Load all point clouds for collision checking
        print("\n=== Step 2: Loading multiple point clouds for collision avoidance ===")

        # Track total points and loading time
        total_points = 0
        pc_load_start_time = time.time()

        # We'll combine all point clouds into one for processing
        combined_points = None

        for pcd_file in pcd_files:
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
            print(
                "Failed to load any point clouds. Proceeding without collision model."
            )
            pc_time = pc_load_time
        else:
            print(
                f"All point clouds loaded in {pc_load_time:.2f} seconds with a total of {len(combined_points)} points"
            )

            # Process the combined point cloud data
            pc_process_start_time = time.time()
            result = robot.add_pointcloud(combined_points)
            pc_process_time = time.time() - pc_process_start_time

            if result < 0:
                print(
                    "Failed to process combined point cloud. Proceeding without collision model."
                )
                pc_time = pc_load_time
            else:
                print(
                    f"Combined point cloud processed in {pc_process_time:.2f} seconds"
                )
                pc_time = pc_load_time + pc_process_time

        # NEW: Update robot base parameters using current transform
        # Before motion planning, read the transform from "map" to "base_link" and update base_x, base_y, and base_theta.
        try:
            transform = robot.tf_buffer.lookup_transform(
                "map", "base_link", rospy.Time(0), rospy.Duration(5.0)
            )
            trans = transform.transform.translation
            rot = transform.transform.rotation

            # Compute yaw from quaternion
            yaw = math.atan2(
                2.0 * (rot.w * rot.z + rot.x * rot.y), 1.0 - 2.0 * (rot.y**2 + rot.z**2)
            )
            rospy.loginfo(
                f"Current robot base transform: x={trans.x}, y={trans.y}, theta={yaw}"
            )
            # Update the robot's base parameters
            robot.set_base_params(yaw, trans.x, trans.y)
        except Exception as e:
            rospy.logwarn(f"Could not update base parameters: {e}")

        # STEP 3: Plan and execute the arm motion with collision avoidance
        print(
            "\n=== Step 3: Planning and executing arm motion with collision avoidance ==="
        )

        # Target arm joint configuration with torso included as first joint
        torso_height = 0.37
        target_joints = [
            torso_height,  # torso_lift
            0.80,  # shoulder_pan
            -0.40,  # shoulder_lift
            -1.5,  # upperarm_roll
            1.5,  # elbow_flex
            1.0,  # forearm_roll
            -0.0,  # wrist_flex
            2.169129759130249,  # wrist_roll
        ]

        print("Target joint configuration:")
        for name, value in zip(robot.planning_joint_names, target_joints):
            print(f" {name}: {value}")

        # Start motion planning timer
        planning_start_time = time.time()

        # Execute motion with collision avoidance
        result = robot.send_joint_values(target_joints)

        # Calculate planning and execution time
        motion_time = time.time() - planning_start_time

        if result is not None:
            print(
                f"Motion planning and execution completed in {motion_time:.2f} seconds"
            )

            # Final status
            print("\n=== Task Summary ===")
            print(f"Initial motion time: {initial_motion_time:.2f} seconds")
            print(f"Navigation time: {nav_time:.2f} seconds")
            print(f"Point cloud processing time: {pc_time:.2f} seconds")
            print(f"Motion planning and execution time: {motion_time:.2f} seconds")
            print(
                f"Total time: {initial_motion_time + nav_time + pc_time + motion_time:.2f} seconds"
            )
            print(
                f"Total points loaded from {len(pcd_files)} point clouds: {total_points}"
            )
            print("\nTask completed successfully!")
        else:
            print("Motion planning or execution failed!")
            print("Attempting to return to initial position...")
            # Try to return to the initial position if the motion fails
            robot.send_joint_values(initial_target_joints)

    except rospy.ROSInterruptException:
        print("Program interrupted!")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
