#!/usr/bin/env python3
import rospy
from fetch.fetch import Fetch


def generate_constraints(robot):
    """Generate random geometric constraints"""
    # Random sphere
    sphere_pos = [0.4, 0, 0.6]
    sphere_radius = 0.1
    robot.add_sphere(sphere_pos, sphere_radius)
    print(f"Added sphere at {sphere_pos} with radius {sphere_radius}")

    # Random box
    box_pos = [0, 0.8, 0.4]
    box_dims = [0.1, 0.2, 0.3]
    box_orientation = [0, 0, 1]  # No rotation for simplicity
    robot.add_box(box_pos, box_dims, box_orientation)
    print(f"Added box at {box_pos} with dimensions {box_dims}")

    # Random cylinder
    cylinder_pos = [-0.5, 0.3, 0.6]
    cylinder_radius = 0.06
    cylinder_height = 0.03
    cylinder_orientation = [0, 0, 1]  # No rotation for simplicity
    robot.add_cylinder(
        cylinder_pos, cylinder_radius, cylinder_height, cylinder_orientation
    )
    print(
        f"Added cylinder at {cylinder_pos} with radius {cylinder_radius} and height {cylinder_height}"
    )


def main():
    # Initialize the robot
    robot = Fetch()

    try:
        # Add random constraints
        print("Adding random geometric constraints...")
        generate_constraints(robot)

        # Target configuration (8-DOF)
        # [torso, shoulder_pan, shoulder_lift, upperarm_roll, elbow_flex, forearm_roll, wrist_flex, wrist_roll]
        target_joints = [
            0.37,  # torso_lift
            0.29470240364379885,  # shoulder_pan
            1.1211147828796386,  # shoulder_lift
            -1.2186898401245116,  # upperarm_roll
            1.876310651525879,  # elbow_flex
            1.202369851473999,  # forearm_roll
            -1.8404571460021972,  # wrist_flex
            1.7227418721292114,  # wrist_roll
        ]

        print("\nTesting VAMP motion planning...")
        print(f"Moving to target configuration: {target_joints}")

        result = robot.send_joint_values(target_joints)

        if result is not None:
            print("Motion successfully executed!")
        else:
            print("Motion planning or execution failed!")

        rospy.sleep(1)
        print("Motion planning test completed!")

    except rospy.ROSInterruptException:
        print("Program interrupted!")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
