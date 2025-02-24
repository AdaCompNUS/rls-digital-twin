#!/usr/bin/env python3

import rospy
from fetch.fetch import Fetch

def main():
    # Initialize the robot
    robot = Fetch()
    try:
        # Target configuration (8-DOF) 
        # [torso, shoulder_pan, shoulder_lift, upperarm_roll, elbow_flex, forearm_roll, wrist_flex, wrist_roll]
        target_joints = [
            0.37,                 # torso_lift
            0.29470240364379885,  # shoulder_pan
            1.1211147828796386,   # shoulder_lift
            -1.2186898401245116,  # upperarm_roll
            1.876310651525879,    # elbow_flex
            1.202369851473999,    # forearm_roll
            -1.8404571460021972,  # wrist_flex
            1.7227418721292114    # wrist_roll
        ]
        
        print("Testing VAMP motion planning...")
        print(f"Moving to target configuration: {target_joints}")
        robot.send_joint_values(target_joints)
        rospy.sleep(1)

        print("Motion planning test completed!")

    except rospy.ROSInterruptException:
        print("Program interrupted!")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()