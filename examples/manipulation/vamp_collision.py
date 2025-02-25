#!/usr/bin/env python3
import rospy
import time
from fetch.fetch import Fetch


def main():
    # Initialize the robot
    robot = Fetch()
    try:
        # First, navigate to the specified pose
        # print("=== Step 1: Navigating to target position ===")
        # # Target pose from your example
        # position = [-1.4662278308455918, 2.438868977769083, 0.0]
        # orientation = [0.0, 0.0, 0.9998942789433433, 0.014540664921926247]
        # print(f"Target position: {position}")
        # print(f"Target orientation: {orientation}")

        # # Start navigation timer
        # nav_start_time = time.time()

        # # Send navigation command
        # robot.send_target_position(position, orientation)

        # # Calculate navigation time
        # nav_time = time.time() - nav_start_time
        # print(f"Navigation completed in {nav_time:.2f} seconds")

        # After navigation, load the point cloud for collision checking
        # print("\n=== Step 2: Loading workstation point cloud for collision avoidance ===")
        # # Path to point cloud (adjust as needed)
        # pcd_path = "mp_collision_models/workstation.ply"

        # # Load and process point cloud, timing the operation
        # pc_start_time = time.time()
        # result = robot.add_pointcloud(pcd_path, frame_id="world")
        # pc_time = time.time() - pc_start_time

        # if result is None:
        #     print("Failed to load point cloud. Proceeding without collision model.")
        # else:
        #     print(f"Point cloud processed in {pc_time:.2f} seconds")

        # Now plan and execute the arm motion
        print("\n=== Step 3: Planning and executing arm motion ===")

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
            # print(f"Navigation time: {nav_time:.2f} seconds")
            # print(f"Point cloud processing time: {pc_time:.2f} seconds")
            print(f"Motion planning and execution time: {motion_time:.2f} seconds")
            # print(f"Total time: {nav_time + pc_time + motion_time:.2f} seconds")
            print("\nTask completed successfully!")
        else:
            print("Motion planning or execution failed!")

    except rospy.ROSInterruptException:
        print("Program interrupted!")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
