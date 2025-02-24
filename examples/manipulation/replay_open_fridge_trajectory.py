import rospy
import numpy as np
from fetch.fetch import Fetch


def replay_trajectory(robot, trajectory, duration=0.1):
    """
    Replay a sequence of end effector poses.

    Args:
        robot: Fetch robot instance
        trajectory: Nx7 array of [x, y, z, qx, qy, qz, qw] poses
        duration: Time to wait between poses (seconds)
    """
    for pose in trajectory:
        # Extract position and orientation
        position = pose[:3]
        orientation = pose[3:7]

        # Send the pose command
        robot.send_end_effector_pose(position, orientation)

        # Wait for specified duration
        rospy.sleep(duration)


def main():
    # Initialize the robot
    robot = Fetch()

    try:
        # First set the torso height for fridge opening
        print("Setting torso height...")
        torso_height = 0.37  # From prepare_conf["open_fridge"]["torso"]
        robot.set_torso_height(torso_height)
        rospy.sleep(2)  # Wait for torso movement

        # Move to fridge-opening arm configuration
        print("Moving to initial arm position...")
        initial_joints = [
            0.29470240364379885,  # shoulder_pan
            1.1211147828796386,  # shoulder_lift
            -1.2186898401245116,  # upperarm_roll
            1.876310651525879,  # elbow_flex
            1.202369851473999,  # forearm_roll
            -1.8404571460021972,  # wrist_flex
            1.7227418721292114,  # wrist_roll
        ]
        robot.send_joint_values(initial_joints)
        rospy.sleep(3)  # Wait for movement to complete

        # Open gripper to prepare for manipulation
        print("Opening gripper...")
        robot.control_gripper(1.0)  # Fully open
        rospy.sleep(1)

        # Load the trajectory from file
        print("\nLoading trajectory data...")
        try:
            trajectory = np.load("examples/trajectory/abs_ee_pose.npy")
            print(f"Loaded trajectory with {len(trajectory)} waypoints")
        except Exception as e:
            print(f"Error loading trajectory file: {e}")
            return

        # Execute the trajectory
        print("\nExecuting trajectory...")
        replay_trajectory(robot, trajectory)

        print("\nTrajectory execution complete!")

    except rospy.ROSInterruptException:
        print("Program interrupted!")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
