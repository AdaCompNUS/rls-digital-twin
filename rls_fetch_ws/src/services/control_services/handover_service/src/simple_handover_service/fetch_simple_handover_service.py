#!/usr/bin/env python


import rospy

from rls_control_msgs.srv import *
from fetch_robot.fetch_robot import FetchRobot
from fetch_robot.fetch_joint_position_config import *

right_hand_over_joints = [1.3205522228271485, 1.399532370159912, -1.5, 1.719844644293213, 0, 0.9, 0]

class SimpleHandoverService(object):
    def __init__(self):
        self.robot = FetchRobot()

        ## advertise service
        rospy.Service('rls_control_services/fetch/simple_handover', SimpleHandOver, self._simple_hand_over)
        rospy.loginfo('SimpleHandoverService is Ready!')

    def _simple_hand_over(self, req):
        res = False
        if req.action == SimpleHandOverRequest.RESET:
            res = self.robot.arm_move_to_joint(home_joints)
        elif req.action == SimpleHandOverRequest.HAND_OVER:
            result = self.robot.arm_move_to_joint(right_hand_over_joints)
            rospy.sleep(2)
            self.robot.gripper_position(FetchRobot.GRIPPER_OPENED_POS)
            rospy.sleep(2)
            res = self.robot.arm_move_to_joint(home_joints)

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
