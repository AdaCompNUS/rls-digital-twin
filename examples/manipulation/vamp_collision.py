#!/usr/bin/env python3
import rospy
import time
import numpy as np
import open3d as o3d
import os
from fetch.fetch import Fetch


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
    Modified to load the point cloud outside the Fetch class.
    """
    # Initialize the robot
    robot = Fetch()

    try:
        # STEP 0: First go to a good starting pose for further motion planning
        print("\n=== Step 0: Moving to a good starting pose ===")
        initial_target_joints = [0.37, 1.32, 1.4, -0.2, 1.72, 0, 1.66, 0]

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
        position = [-2.7782226013936215, 0.15580237473099096, 0.0]
        orientation = [0.0, 0.0, -0.6538142154802868, 0.7566551206698445]
        print(f"Target position: {position}")
        print(f"Target orientation: {orientation}")

        # Start navigation timer
        nav_start_time = time.time()

        # Send navigation command
        robot.send_target_position(position, orientation)

        # Calculate navigation time
        nav_time = time.time() - nav_start_time
        print(f"Navigation completed in {nav_time:.2f} seconds")

        # STEP 2: Load the point cloud for collision checking
        print(
            "\n=== Step 2: Loading workstation point cloud for collision avoidance ==="
        )
        # Path to point cloud
        pcd_path = "mp_collision_models/workstation.ply"

        # Load point cloud data from file - this is now done in the test script
        pc_load_start_time = time.time()
        point_cloud_data = load_pointcloud(pcd_path)
        pc_load_time = time.time() - pc_load_start_time

        if point_cloud_data is None:
            print("Failed to load point cloud. Proceeding without collision model.")
            pc_time = 0
        else:
            print(
                f"Point cloud loaded in {pc_load_time:.2f} seconds with {len(point_cloud_data)} points"
            )

            # Process the loaded point cloud data
            pc_process_start_time = time.time()
            result = robot.add_pointcloud(
                point_cloud_data, frame_id="map", filter_radius=0.02, filter_cull=True
            )
            pc_process_time = time.time() - pc_process_start_time

            if result < 0:
                print(
                    "Failed to process point cloud. Proceeding without collision model."
                )
                pc_time = pc_load_time
            else:
                print(f"Point cloud processed in {pc_process_time:.2f} seconds")
                pc_time = pc_load_time + pc_process_time

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
