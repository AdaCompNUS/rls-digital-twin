import rospy
from fetch.fetch import Fetch


def main():
    # Initialize the robot
    robot = Fetch()

    try:
        # Example 1: Move to a position in front
        print("Moving 1 meter forward...")
        position = [1.0, 0.0, 0.0]  # 1 meter forward
        orientation = [0.0, 0.0, 0.0, 1.0]  # No rotation
        robot.send_target_position(position, orientation)

        rospy.sleep(2)  # Wait between movements

        # Example 2: Move to a position with rotation
        print("Moving to the right with 90-degree rotation...")
        position = [1.0, -1.0, 0.0]  # 1m forward, 1m right
        orientation = [0.0, 0.0, -0.707, 0.707]  # 90 degrees rotation
        robot.send_target_position(position, orientation)

        print("\nNavigation complete!")

    except rospy.ROSInterruptException:
        print("Program interrupted!")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
