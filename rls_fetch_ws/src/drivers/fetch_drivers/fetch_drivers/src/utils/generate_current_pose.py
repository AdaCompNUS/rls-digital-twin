import rospy
from moveit_commander import RobotCommander, MoveGroupCommander, PlanningSceneInterface, roscpp_initialize, roscpp_shutdown
###import service type
import sys
from std_msgs.msg import String


def get_current_pose():
    '''
    move arm to predefined location to give kinect a clear view of the shelf
    '''
    pose = arm.get_current_pose()
    print pose
    return

if __name__=='__main__':
    roscpp_initialize(sys.argv)
    rospy.init_node('get_joint_config')

    ###############------Robot Planner Initialization-----#############3
    global arm
    arm = MoveGroupCommander("arm")
    get_current_pose()
