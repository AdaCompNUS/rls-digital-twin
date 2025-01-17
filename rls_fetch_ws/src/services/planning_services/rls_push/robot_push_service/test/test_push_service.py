#!/usr/bin/env python

import numpy as np
import rospy
import math

from rls_push_msgs.srv import *

rospy.init_node("test_push_service")
push_client = rospy.ServiceProxy("rls_control_services/push", RobotPush)

def init_goal():
    x_range = np.array([0.2, 0.1])
    y_range = np.array([-0.15, -0.2])
    x = x_range[0] - np.random.random() * (x_range[0] - x_range[1])
    x = x * np.random.choice([-1, 1], 1, p=[0.5, 0.5])[0]
    y = y_range[0] - np.random.random() * (y_range[0] - y_range[1])

    theta_range = [np.pi / 2, np.pi / 6]  # 90 to 30
    theta = theta_range[0] - np.random.random() * (theta_range[0] - theta_range[1])
    theta = theta * np.random.choice([-1, 1], 1, p=[0.5, 0.5])[0]  # 90 to 30 or -90 to -30

    # goal_x goal_y goal_z goal_theta
    goal_xyz_theta = np.array([x, y, theta])

    print("xy_theta", goal_xyz_theta)

    # upload goal to server
    return goal_xyz_theta

goal = init_goal()
req = RobotPushRequest()
req.request = RobotPushRequest.PUSH_WXY
req.dx = goal[1]
req.dy = -goal[0]
req.dw = goal[2] / math.pi * 180
push_client(req)