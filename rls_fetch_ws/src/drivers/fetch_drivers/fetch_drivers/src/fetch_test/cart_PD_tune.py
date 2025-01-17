#! /usr/bin/env python

import fetch_api
import rospy
from moveit_commander import MoveGroupCommander
from geometry_msgs.msg import PoseStamped
from copy import deepcopy


def wait_for_time():
    """Wait for simulated time to begin.
    """
    while rospy.Time().now().to_sec() == 0:
        pass


def main():
    rospy.init_node('arm_demo')
    wait_for_time()
    argv = rospy.myargv()

    arm = fetch_api.Arm()

    while not rospy.is_shutdown() and not arm.is_pose_received():
        continue


    ## Test Cartesian Straight line path
    #for i in range(20):
    #    z = arm.pose.pose.position.z
    #    arm.move_by_z(dz = -0.2)
    #    print arm.pose.pose.position.z - z
    #    z = arm.pose.pose.position.z

    #    rospy.sleep(1)
    #    arm.move_by_z(dz = 0.2)
    #    print arm.pose.pose.position.z - z

    for i in range(20):
        x = arm.pose.pose.position.x
        y = arm.pose.pose.position.y
        arm.move_by_xy(dx=0.15, dy=0.15)
        rospy.sleep(1)
        print arm.pose.pose.position.x - x
        print arm.pose.pose.position.y - y
        x = arm.pose.pose.position.x
        y = arm.pose.pose.position.y

        arm.move_by_xy(dx=-0.15, dy=-0.15)
        rospy.sleep(1)
        print arm.pose.pose.position.x - x
        print arm.pose.pose.position.y - y

    print '--------------------'
    arm.clean_exit()

if __name__ == '__main__':
    main()
