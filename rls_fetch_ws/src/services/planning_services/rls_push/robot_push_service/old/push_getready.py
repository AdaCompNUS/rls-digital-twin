'''
Main program to push objects with real robot
'''

__author__='Li Juekun'
__date__ = '2018/01/12'

import numpy as np
import os
### ROS modules
import rospy

## Perception
import fetch_api
from robot_config import *
from sensor_msgs.msg import Image, PointCloud2, JointState


''' Push Controller '''
class PushController:
    def __init__(self):
        self.arm = fetch_api.Arm()
        self.torso = fetch_api.Torso()
        self.joint_angle = []
        self.joint_sub = rospy.Subscriber('/joint_states', JointState, self.joint_callback)

    def joint_callback(self, data):
        if len(data.position) > 8:
            self.joint_angle = list(data.position[6:])

    def _test(self):
        self.torso.set_height(fetch_api.Torso.MAX_HEIGHT)
        rospy.sleep(1)
        self.arm.move_to_joint_goal(HIDE_LEFT_JOINTS, replan=True)
        rospy.sleep(1)
        return
        self.arm.move_to_joint_goal(LEFT_GRASP_JOINTS, replan=True)
        rospy.sleep(1)
        print self.joint_angle

        self.joint_angle[5] = 0.8279026104675293
        HAND_OVER_JOINTS = [(joint_names[i], self.joint_angle[i]) for i in range(7)]
        self.arm.move_to_joint_goal(HAND_OVER_JOINTS, replan=True)

        self.arm.move_in_cartesian(dx=0.1)
        ## test hand over
        return


if __name__=='__main__':
    try:
        rospy.init_node('push_test', log_level=rospy.INFO)
        con = PushController()
        con._test()
    except rospy.ROSInterruptException:
        print 'program interrupted before completion'

