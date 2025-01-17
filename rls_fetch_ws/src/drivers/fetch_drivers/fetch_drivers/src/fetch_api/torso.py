#!/usr/bin/env python

import actionlib
import control_msgs.msg
import rospy
import trajectory_msgs.msg
from sensor_msgs.msg import JointState

ACTION_SERVER = 'torso_controller/follow_joint_trajectory'
TORSO_JOINT_NAME = 'torso_lift_joint'
TIME_FROM_START = 5  # How many seconds it should take to set the torso height.


class Torso(object):
    """Torso controls the robot's torso height.
    """
    MIN_HEIGHT = 0.0
    MAX_HEIGHT = 0.4

    def __init__(self):
        self._client = actionlib.SimpleActionClient(
            ACTION_SERVER, control_msgs.msg.FollowJointTrajectoryAction)
        self._client.wait_for_server(rospy.Duration(10))
        self.torso_value = -10
        self.rate = rospy.Rate(50)
        self.joint_sub = rospy.Subscriber('/joint_states', JointState, self.read_torso)

    def read_torso(self, data):
        if not data == None:
            if TORSO_JOINT_NAME in data.name:
                idx = data.name.index(TORSO_JOINT_NAME)
                self.torso_value = data.position[idx]

    def move_by(self, distance):
        """incrementally move torso up and down"""
        ## get current torso value
        while self.torso_value < 0 and not rospy.is_shutdown():
            self.rate.sleep()
        height = self.torso_value + distance
        self.set_height(height)


    def set_height(self, height, duration=5):
        """Sets the torso height.

        This will always take ~5 seconds to execute.

        Args:
            height: The height, in meters, to set the torso to. Values range
                from Torso.MIN_HEIGHT (0.0) to Torso.MAX_HEIGHT(0.4).
        """
        height = min(height, 0.4)
        height = max(height, 0.0)

        goal = control_msgs.msg.FollowJointTrajectoryGoal()
        goal.trajectory.joint_names.append(TORSO_JOINT_NAME)
        point = trajectory_msgs.msg.JointTrajectoryPoint()
        point.positions.append(height)
        point.time_from_start = rospy.Duration(duration)
        goal.trajectory.points.append(point)
        self._client.send_goal(goal)
        self._client.wait_for_result(rospy.Duration(10))
