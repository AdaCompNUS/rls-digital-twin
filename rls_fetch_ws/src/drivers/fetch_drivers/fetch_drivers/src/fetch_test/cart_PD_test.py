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

    pose = PoseStamped()
    pose.header.frame_id = 'base_link'
    pose.pose.position.x = 0.4285
    pose.pose.position.y = 0.41251
    pose.pose.position.z = 1.1005
    pose.pose.orientation.x = -0.063
    pose.pose.orientation.y = 0.700
    pose.pose.orientation.z = 0.0644
    pose.pose.orientation.w = 0.7081
    arm.move_to_pose(pose, replan=True, execution_timeout=10)

    ## Test Cartesian Straight line path
    for i in range(10):
        result, pose_init, pose_final = arm.move_in_cartesian(dz=-0.15)
        rospy.sleep(1)
        print pose_final.pose.position.z - pose_init.pose.position.z
        result, pose_init, pose_final = arm.move_in_cartesian(dz=0.15)
        rospy.sleep(1)
        print pose_final.pose.position.z - pose_init.pose.position.z

    for i in range(10):
        result, pose_init, pose_final = arm.move_in_cartesian(dx=0.15, dy=0.15)
        rospy.sleep(1)
        print pose_final.pose.position.x - pose_init.pose.position.x
        print pose_final.pose.position.y - pose_init.pose.position.y

        result, pose_init, pose_final = arm.move_in_cartesian(dx=-0.15, dy=-0.15)
        rospy.sleep(1)
        print pose_final.pose.position.x - pose_init.pose.position.x
        print pose_final.pose.position.y - pose_init.pose.position.y


if __name__ == '__main__':
    main()
