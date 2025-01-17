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

    ## raise to its max height
    torso.set_height(fetch_api.Torso.MAX_HEIGHT)
    ## lower by 20cm
    torso.move_by(-0.2)
    ## raise by 10cm
    torso.move_by(0.1)
    ## lower to its min height
    torso.set_height(fetch_api.Torso.MIN_HEIGHT)



if __name__ == '__main__':
    main()
