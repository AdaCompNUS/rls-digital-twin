#!/usr/bin/env python

import math
from geometry_msgs.msg import PoseStamped
from moveit_msgs.msg import MoveItErrorCodes
import rospy

from rls_driver_msgs.srv import *

from movo_robot_config import *
from movo_speaker_google.srv import MovoSpeakGoogle

# Setting
SIMULATION = False

# Constants
APPROACH_DZ = 0.12
W = 128.0  # !!!! Important to make it float to prevent integer division becomes zeros
H = 106.0


class MovoPushController():
    def __init__(self):
        self._arm_client = rospy.ServiceProxy('/rls_drivers/movo/arm', MovoArmAction)
        self._voice_client = rospy.ServiceProxy(
            'rls_contro_servicesl/movo/movo_speaker_google', MovoSpeakGoogle)

    def set_table_height(self, table_height):
        self._table_height = table_height

    def move_arm_to_pose(self, pose, plan_only=False):
        req = MovoArmActionRequest()
        req.action = MovoArmActionRequest.MOVE_TO_POSE
        req.pose_goal = pose
        req.arm = MovoArmActionRequest.RIGHT_ARM

        if SIMULATION:
            req.plan_only = True
            resp = self._arm_client(req)
        else:
            req.plan_only = plan_only
            resp = self._arm_client(req)

        if not resp.success:
            return False

        return True

    def move_arm_in_cartesian(self, dx=0.0, dy=0.0, dz=0.0):
        req = MovoArmActionRequest()
        req.action = MovoArmActionRequest.MOVE_IN_CARTESIAN
        req.cartesian_goal = [dx, dy, dz]
        req.arm = MovoArmActionRequest.RIGHT_ARM
        req.plan_only = False

        if not SIMULATION:
            resp = self._arm_client(req)
            result = resp.success
        else:
            print("MovoPushController/move_arm_in_cartesian: simulation set to true, skip")
            result = True

        return result

    def move_arm_to_home(self):
        req = MovoArmActionRequest()
        req.action = MovoArmActionRequest.MOVE_TO_JOINT
        req.joint_goal = right_ready_pick_joints
        req.arm = MovoArmActionRequest.RIGHT_ARM
        req.plan_only = False

        resp = self._arm_client(req)

        if not resp.success:
            return False

        req.action = MovoArmActionRequest.MOVE_TO_JOINT
        req.joint_goal = home_left_joints
        req.arm = MovoArmActionRequest.LEFT_ARM
        req.plan_only = False
        resp = self._arm_client(req)

        return resp.success

    def grasp_from_side(self, side):
        # TODO
        pass

    def say(self, text):
        resp = self._voice_client(text)
        return resp.success
