#! /usr/bin/env python

import fetch_api
import rospy
from moveit_commander import MoveGroupCommander
from geometry_msgs.msg import PoseStamped
from copy import deepcopy
from robot_config import *


def wait_for_time():
    """Wait for simulated time to begin.
    """
    while rospy.Time().now().to_sec() == 0:
        pass


def main():
    rospy.init_node('arm_demo')
    wait_for_time()
    argv = rospy.myargv()

    torso = fetch_api.Torso()
    arm = fetch_api.Arm()

    ## Test torso
    torso.set_height(fetch_api.Torso.MAX_HEIGHT)

    arm.move_to_joint_goal(HIDE_LEFT_JOINTS, replan=True)
    rospy.sleep(1)
    arm.move_to_joint_goal(HIDE_RIGHT_JOINTS, replan=True)
    rospy.sleep(1)
    arm.move_to_joint_goal(HOLD_RIGHT_JOINTS, replan=True)
    rospy.sleep(1)
    arm.move_to_joint_goal(HOLD_LEFT_JOINTS, replan=True)
    rospy.sleep(1)
    arm.move_to_joint_goal(HIDE_LEFT_JOINTS, replan=True)
    rospy.sleep(1)
    arm.move_to_joint_goal(HOME_JOINTS, replan=True)
    rospy.sleep(1)

    torso.set_height(fetch_api.Torso.MIN_HEIGHT)
    arm.clean_exit()

if __name__ == '__main__':
    main()
