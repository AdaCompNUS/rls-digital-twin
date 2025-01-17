#! /usr/bin/env python

import fetch_api
import rospy
from moveit_commander import MoveGroupCommander
from geometry_msgs.msg import PoseStamped
from copy import deepcopy
from twist_command_test import Armm


def wait_for_time():
    """Wait for simulated time to begin.
    """
    while rospy.Time().now().to_sec() == 0:
        pass


def main():
    rospy.init_node('arm_demo')
    wait_for_time()
    argv = rospy.myargv()

    #torso = fetch_api.Torso()
    #arm = fetch_api.Arm()
    armm = Armm()
    #group = MoveGroupCommander('arm')


    ## Test torso
    #torso.set_height(fetch_api.Torso.MAX_HEIGHT)

    ## Test named target
    #arm.move_to_named_target(group, 'pre_push')
    #rospy.sleep(1)

    ### Test pose target
    #pose = PoseStamped()
    #pose.header.frame_id = 'base_link'
    #pose.pose.position.x = 0.65285
    #pose.pose.position.y = 0.41251
    #pose.pose.position.z = 1.1005
    #pose.pose.orientation.x = -0.063
    #pose.pose.orientation.y = 0.700
    #pose.pose.orientation.z = 0.0644
    #pose.pose.orientation.w = 0.7081

    #arm.move_to_pose(pose, replan=True, execution_timeout=10)
    #rospy.sleep(1)
    while not rospy.is_shutdown() and not armm.valid_pose_received:
        continue

    for i in range(2):
        for j in range(2):
            armm.move_by_z(dz = -0.1)
            armm.move_by_z(dz = 0.1)
            armm.move_by_xy(dx=0.15, dy = -0.1)
        for k in range(2):
            armm.move_by_z(dz = -0.1)
            armm.move_by_z(dz = 0.1)
            armm.move_by_xy(dx=-0.15, dy = 0.1)



    #armm.move_by_xy(dx = 0.05)
    #rospy.sleep(0.5)
    #armm.move_by_xy(dy = -0.05)
    #rospy.sleep(0.5)
    #armm.move_by_z(dz = -0.05)
    #rospy.sleep(0.5)
    #armm.move_by_xy(dx = 0.05)
    #rospy.sleep(0.5)
    #armm.move_by_xy(dy = -0.05)
    #rospy.sleep(0.5)
    #armm.move_by_z(dz = -0.05)
    #rospy.sleep(0.5)
    #armm.move_by_z(dz = 0.05)
    #rospy.sleep(0.5)
    #armm.move_by_xy(dy = 0.05)
    #rospy.sleep(0.5)
    #armm.move_by_xy(dx = -0.05)
    #rospy.sleep(0.5)
    #armm.move_by_z(dz = 0.05)
    #rospy.sleep(0.5)
    #armm.move_by_xy(dy = 0.05)
    #rospy.sleep(0.5)
    #armm.move_by_xy(dx = -0.05)
    #rospy.sleep(0.5)
    #armm.move_by_xy(dx = 0.1)
    #rospy.sleep(0.5)
    #armm.move_by_xy(dy = -0.1)
    #rospy.sleep(0.5)
    #armm.move_by_z(dz = -0.1)
    #rospy.sleep(0.5)
    #armm.move_by_z(dz = 0.1)
    #rospy.sleep(0.5)
    #armm.move_by_xy(dy = 0.1)
    #rospy.sleep(0.5)
    #armm.move_by_xy(dx = -0.1)
    #rospy.sleep(0.5)
    #armm.move_by_xy(dx = 0.08, dy = 0.08)
    #rospy.sleep(0.5)
    #armm.move_by_xy(dx = -0.08, dy = -0.08)
    #rospy.sleep(0.5)
    #armm.move_by_xy(dx = 0.05, dy = -0.05)
    #rospy.sleep(0.5)
    #armm.move_by_xy(dx =- 0.05, dy = 0.05)


    ## Tuck
    print 'tuck'
    #arm.move_to_named_target(group, 'tuck')
    #torso.set_height(fetch_api.Torso.MIN_HEIGHT)

if __name__ == '__main__':
    main()
