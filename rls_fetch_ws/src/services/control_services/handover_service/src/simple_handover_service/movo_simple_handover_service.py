#!/usr/bin/env python

'''
TODO

P1:
*. Ready pick place pos dynamic


P2:
1. Side pick and place
2. Obstacle avoidance including the object being manipulated
3. Extract pnp_simple
4. Assumption of base_link frame being horizontal
5. Orientate grasp along short side of object
6. tf time synchronization!!
*. Decouple from MOVO
*. Move this into movo_onboard so that there is no need to edit moveit_python in add_table_as_obstacle()
'''


import rospy

from rls_control_msgs.srv import *
from movo_robot.movo_robot import MovoRobot
from movo_robot.movo_joint_position_config import *

right_hand_over_joints = [-1.5487714519477231, -0.7, -0.04603966366085288, -1, -0.012279303154137722, 0.5, -0.785398]

class SimpleHandoverService(object):
    def __init__(self):
        self.robot = MovoRobot(use_ros=True)

        ## advertise service
        rospy.Service('rls_control_services/movo/simple_handover', SimpleHandOver, self._simple_hand_over)
        rospy.loginfo('SimpleHandoverService is Ready!')

    def _simple_hand_over(self, req):
        res = False
        if req.action == SimpleHandOverRequest.RESET:
            res = self.robot.arm_move_to_joint(home_right_joints, arm=req.arm_name)
        elif req.action == SimpleHandOverRequest.HAND_OVER:
            res = self.robot.arm_move_to_joint(right_hand_over_joints, arm=req.arm_name)
            rospy.sleep(1)
            self.robot.gripper_position(MovoRobot.GRIPPER_OPENED_POS)
            rospy.sleep(2)
            res = self.robot.arm_move_to_joint(home_right_joints, arm=req.arm_name)

        return SimpleHandOverResponse(res)

if __name__ == "__main__":
    ## init node
    rospy.init_node('simple_handover')

    try:
        handover_service = SimpleHandoverService()
        rospy.spin()
    except Exception as e:
        rospy.logerr(e)
        rospy.logerr("Failed to init SimpleHandoverService")
