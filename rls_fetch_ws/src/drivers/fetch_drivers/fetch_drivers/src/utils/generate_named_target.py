import rospy
from moveit_commander import RobotCommander, MoveGroupCommander, PlanningSceneInterface, roscpp_initialize, roscpp_shutdown
###import service type
import sys
from std_msgs.msg import String


def get_joint_state():
    '''
    move arm to predefined location to give kinect a clear view of the shelf
    '''
    bin_name = raw_input("name of the current joint configuration ")
    joints = arm.get_current_joint_values()
    print joints
    print '<group_state name="' + bin_name + '" group="arm">'
    print '\t<joint name="shoulder_pan_joint" value="' + str(joints[0]) + '" />'
    print '\t<joint name="shoulder_lift_joint" value="' + str(joints[1]) + '" />'
    print '\t<joint name="upperarm_roll_joint" value="' + str(joints[2]) + '" />'
    print '\t<joint name="elbow_flex_joint" value="' + str(joints[3]) + '" />'
    print '\t<joint name="forearm_roll_joint" value="' + str(joints[4]) + '" />'
    print '\t<joint name="wrist_flex_joint" value="' + str(joints[5]) + '" />'
    print '\t<joint name="wrist_roll_joint" value="' + str(joints[6]) + '" />'
    print '</group_state>'

    return

if __name__=='__main__':
    roscpp_initialize(sys.argv)
    rospy.init_node('get_joint_config')

    ###############------Robot Planner Initialization-----#############3
    global arm
    arm = MoveGroupCommander("arm")
    get_joint_state()
