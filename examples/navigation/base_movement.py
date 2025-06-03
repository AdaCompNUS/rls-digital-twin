import sys
import tty
import termios
import rospy
from fetch.fetch import Fetch


def get_key():
    """Get a single keyboard press without requiring Enter."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch


def print_instructions():
    """Print keyboard control instructions."""
    print("\nFetch Base Movement Example")
    print("-------------------------")
    print("Use the following keys to control the robot:")
    print("w - Move forward")
    print("s - Move backward")
    print("a - Turn left")
    print("d - Turn right")
    print("q - Quit")
    print("Any other key - Stop")
    print("-------------------------")


def main():
    # Initialize the Fetch robot
    robot = Fetch()
    print_instructions()

    try:
        while not rospy.is_shutdown():
            key = get_key()

            if key == "w":
                print("Moving forward...")
                robot.move_base(1.5, 0.0)  # 50% forward speed
            elif key == "s":
                print("Moving backward...")
                robot.move_base(-1.5, 0.0)  # 50% backward speed
            elif key == "a":
                print("Turning left...")
                robot.move_base(0.0, 1.5)  # 50% turn left speed
            elif key == "d":
                print("Turning right...")
                robot.move_base(0.0, -1.5)  # 50% turn right speed
            elif key == "q":
                print("\nQuitting...")
                break
            else:
                robot.stop_base()

            robot.rate.sleep()

    except rospy.ROSInterruptException:
        pass
    finally:
        # Ensure the robot stops when the program ends
        robot.stop_base()


if __name__ == "__main__":
    main()
