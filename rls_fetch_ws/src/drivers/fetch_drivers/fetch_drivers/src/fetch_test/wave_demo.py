#! /usr/bin/env python

import fetch_api
import rospy
from moveit_commander import MoveGroupCommander
from geometry_msgs.msg import PoseStamped
from copy import deepcopy
from robot_config import *
import math


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
    arm = fetch_api.ArmV2()
    head = fetch_api.Head()
    gripper = fetch_api.Gripper()

    # # reset to home
    # arm.move_to_joint(home_joints, allowed_planning_time=2.0)
    # torso.set_height(0.1, duration=1.0)

    # rospy.sleep(4.0)

    ## raise torso
    gripper.open()
    torso.set_height(fetch_api.Torso.MAX_HEIGHT, duration=1.0)

    # move head
    head.pan_tilt(0, math.radians(45), 0.5)
    head.pan_tilt(0, math.radians(-45), 0.5)
    head.pan_tilt(math.radians(-45), 0, 0.5)
    head.pan_tilt(math.radians(45), 0, 0.5)
    head.pan_tilt(0, 0, 0.5)

    # wave
    arm.set_max_velocity_scaling_factor(scale = 1.0)
    arm.move_to_joint(wave_left_joints, replan=True, allowed_planning_time=1.0)
    rospy.sleep(0.2)

    for i in range(2):
        if rospy.is_shutdown():
            break
        arm.move_to_joint(wave_left_joints, replan=True, allowed_planning_time=0.1)
        # arm.move_to_joints_direct(WAVE_LEFT_JOINTS)
        #rospy.sleep(1)

        arm.move_to_joint(wave_right_joints, replan=True, allowed_planning_time=0.1)
        # arm.move_to_joints_direct(WAVE_RIGHT_JOINTS)

    arm.move_to_joint(wave_left_joints, replan=True, allowed_planning_time=0.1)
    # arm.move_to_joints_direct(WAVE_LEFT_JOINTS)
    #rospy.sleep(1)
    #arm.move_to_joint_goal(HOME_JOINTS, replan=True)
    #rospy.sleep(1)
    #torso.set_height(fetch_api.Torso.MIN_HEIGHT)
    #arm.clean_exit()

if __name__ == '__main__':
    main()
