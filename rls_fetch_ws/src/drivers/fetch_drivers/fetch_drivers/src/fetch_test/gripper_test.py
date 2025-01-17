import fetch_api
import rospy

rospy.init_node('gripper_test')

gripper = fetch_api.Gripper()

gripper.open()
raw_input("press anything to continue")
gripper.close(opening=0.05)
raw_input("press anything to continue")
gripper.close(opening=0.02)
raw_input("press anything to continue")
gripper.close()