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

    base = fetch_api.Base()
    raw_input('wait')

    base.go_forward(0.2, 0.1)
    raw_input('wait')
    base.go_forward(-0.2, 0.05)


if __name__ == '__main__':
    main()
