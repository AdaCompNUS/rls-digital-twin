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

    head = fetch_api.Head()

    head.pan_tilt(0, math.pi/3, 1)
    raw_input('wait')

    head.look_at('map', -8.1, -0.6, 0.8)

if __name__ == '__main__':
    main()
