import rospy
from fetch.fetch import Fetch

def main():
    # Initialize the robot
    robot = Fetch()
    
    try:
        # First, set the torso height
        print("Setting torso height...")
        torso_height = 0.37  # From prepare_conf["open_fridge"]["torso"]
        robot.set_torso_height(torso_height)
        rospy.sleep(1)  # Short wait to ensure torso movement is complete
        
        # Then move to target configuration
        print("Moving arm to target position...")
        target_joints = [
            0.29470240364379885,  # shoulder_pan
            1.1211147828796386,   # shoulder_lift
            -1.2186898401245116,  # upperarm_roll
            1.876310651525879,    # elbow_flex
            1.202369851473999,    # forearm_roll
            -1.8404571460021972,  # wrist_flex
            1.7227418721292114,   # wrist_roll
        ]
        robot.send_joint_values(target_joints)
        rospy.sleep(1)  # Wait for arm movement to complete
        
        # Open gripper to prepare for grasping
        print("Close gripper")
        robot.control_gripper(0.0)
        rospy.sleep(1)
        print("Open gripper...")
        robot.control_gripper(1.0)  # Fully open
        rospy.sleep(1)
        
        print("\nMovements complete! The robot is ready to open the fridge.")
        
    except rospy.ROSInterruptException:
        print("Program interrupted!")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    main()