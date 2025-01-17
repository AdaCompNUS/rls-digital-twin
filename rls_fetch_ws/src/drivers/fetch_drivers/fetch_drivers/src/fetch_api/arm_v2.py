#!/usr/bin/env python

import actionlib
import control_msgs.msg
import geometry_msgs.msg
import trajectory_msgs.msg
import math
import moveit_commander
import rospy
import tf
import moveit_msgs
from moveit_msgs.msg import MoveItErrorCodes, MoveGroupAction
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest
from tf.listener import TransformListener
from geometry_msgs.msg import TwistStamped, Pose, PoseStamped
from sensor_msgs.msg import JointState
from math import sqrt, pow, sin, cos, atan2
import sys

from rls_driver_msgs.srv import CartesianGoal, CartesianGoalRequest

from .arm_joints import ArmJoints


# ------- Settings --------
ARM_GROUP_NAME = 'arm'
JOINT_ACTION_SERVER = 'arm_controller/follow_joint_trajectory'
MOVE_GROUP_ACTION_SERVER = 'move_group'
TIME_FROM_START = 5
CARTESIAN_SERVER = '/rls_drivers/fetch/cartesian_service'
DEFAULT_ARM_VELOCITY_SCALE = 0.4
PLANNER_ID = "RRTstarkConfigDefault"

# ------- Code --------
def moveit_error_string(val):
    """Returns a string associated with a MoveItErrorCode.

    Args:
        val: The val field from moveit_msgs/MoveItErrorCodes.msg

    Returns: The string associated with the error value, 'UNKNOWN_ERROR_CODE'
        if the value is invalid.
    """
    if val == MoveItErrorCodes.SUCCESS:
        return 'SUCCESS'
    elif val == MoveItErrorCodes.FAILURE:
        return 'FAILURE'
    elif val == MoveItErrorCodes.PLANNING_FAILED:
        return 'PLANNING_FAILED'
    elif val == MoveItErrorCodes.INVALID_MOTION_PLAN:
        return 'INVALID_MOTION_PLAN'
    elif val == MoveItErrorCodes.MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE:
        return 'MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE'
    elif val == MoveItErrorCodes.CONTROL_FAILED:
        return 'CONTROL_FAILED'
    elif val == MoveItErrorCodes.UNABLE_TO_AQUIRE_SENSOR_DATA:
        return 'UNABLE_TO_AQUIRE_SENSOR_DATA'
    elif val == MoveItErrorCodes.TIMED_OUT:
        return 'TIMED_OUT'
    elif val == MoveItErrorCodes.PREEMPTED:
        return 'PREEMPTED'
    elif val == MoveItErrorCodes.START_STATE_IN_COLLISION:
        return 'START_STATE_IN_COLLISION'
    elif val == MoveItErrorCodes.START_STATE_VIOLATES_PATH_CONSTRAINTS:
        return 'START_STATE_VIOLATES_PATH_CONSTRAINTS'
    elif val == MoveItErrorCodes.GOAL_IN_COLLISION:
        return 'GOAL_IN_COLLISION'
    elif val == MoveItErrorCodes.GOAL_VIOLATES_PATH_CONSTRAINTS:
        return 'GOAL_VIOLATES_PATH_CONSTRAINTS'
    elif val == MoveItErrorCodes.GOAL_CONSTRAINTS_VIOLATED:
        return 'GOAL_CONSTRAINTS_VIOLATED'
    elif val == MoveItErrorCodes.INVALID_GROUP_NAME:
        return 'INVALID_GROUP_NAME'
    elif val == MoveItErrorCodes.INVALID_GOAL_CONSTRAINTS:
        return 'INVALID_GOAL_CONSTRAINTS'
    elif val == MoveItErrorCodes.INVALID_ROBOT_STATE:
        return 'INVALID_ROBOT_STATE'
    elif val == MoveItErrorCodes.INVALID_LINK_NAME:
        return 'INVALID_LINK_NAME'
    elif val == MoveItErrorCodes.INVALID_OBJECT_NAME:
        return 'INVALID_OBJECT_NAME'
    elif val == MoveItErrorCodes.FRAME_TRANSFORM_FAILURE:
        return 'FRAME_TRANSFORM_FAILURE'
    elif val == MoveItErrorCodes.COLLISION_CHECKING_UNAVAILABLE:
        return 'COLLISION_CHECKING_UNAVAILABLE'
    elif val == MoveItErrorCodes.ROBOT_STATE_STALE:
        return 'ROBOT_STATE_STALE'
    elif val == MoveItErrorCodes.SENSOR_INFO_STALE:
        return 'SENSOR_INFO_STALE'
    elif val == MoveItErrorCodes.NO_IK_SOLUTION:
        return 'NO_IK_SOLUTION'
    else:
        return 'UNKNOWN_ERROR_CODE'


class ArmV2(object):
    """Arm controls the robot's arm.

    Joint space control:
        joints = ArmJoints()
        # Fill out joint states
        arm = fetch_api.Arm()
        arm.move_to_joints(joints)
    """

    def __init__(self):
        # moveit arm controller
        moveit_commander.roscpp_initialize(sys.argv)
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.move_group_arm = moveit_commander.MoveGroupCommander(ARM_GROUP_NAME)
        self.display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path',
                                               moveit_msgs.msg.DisplayTrajectory,
                                               queue_size=20)

        # default parameters
        self.move_group_arm.set_max_velocity_scaling_factor(DEFAULT_ARM_VELOCITY_SCALE) # to reduce fetch arm movement speed
        self.move_group_arm.set_max_acceleration_scaling_factor(1.0)
        self.move_group_arm.set_planner_id("RRTstarkConfigDefault") # to produce optimal trajectory

        self._joint_client = actionlib.SimpleActionClient(
            JOINT_ACTION_SERVER, control_msgs.msg.FollowJointTrajectoryAction)

        # cartesian controller client
        self._cartesian_client = rospy.ServiceProxy(CARTESIAN_SERVER, CartesianGoal)
        rospy.loginfo("waiting for CARTESIAN_SERVER in arm.py...")
        rospy.wait_for_service(CARTESIAN_SERVER)
        rospy.loginfo("CARTESIAN_SERVER in arm.py is up!")

    def set_max_velocity_scaling_factor(self, scale=DEFAULT_ARM_VELOCITY_SCALE):
        self.move_group_arm.set_max_velocity_scaling_factor(scale)
        return True

    def move_to_joint(self,
                      joints,
                      allowed_planning_time=10.0,
                      execution_timeout=15.0,
                      num_planning_attempts=200,
                      plan_only=False,
                      replan=False,
                      replan_attempts=5,
                      tolerance=0.004,
                      return_plan=False):

        self.move_group_arm.set_num_planning_attempts(num_planning_attempts)
        self.move_group_arm.set_planning_time(allowed_planning_time)
        self.move_group_arm.set_goal_joint_tolerance(tolerance)

        if plan_only:
            plan = self.move_group_arm.plan(joints)

            # visualize plan
            display_trajectory = moveit_msgs.msg.DisplayTrajectory()
            display_trajectory.trajectory_start = self.robot.get_current_state()
            display_trajectory.trajectory.append(plan)
            self.display_trajectory_publisher.publish(display_trajectory)

            if return_plan:
                return plan
            else:
                return 'SUCCESS' if len(plan.joint_trajectory.points)>0 else "FAIL"
        else:
            res = self.move_group_arm.go(joints, wait=True)

            # Calling ``stop()`` ensures that there is no residual movement
            self.move_group_arm.stop()
            return moveit_error_string(res)

    def move_to_pose(self,
                     pose_stamped,
                     allowed_planning_time=10.0,
                     execution_timeout=15.0,
                     num_planning_attempts=200,
                     orientation_constraint=None,
                     plan_only=False,
                     replan=False,
                     replan_attempts=5,
                     tolerance=0.01,
                     return_plan=False,
                     frame_id=None):
        self.move_group_arm.set_num_planning_attempts(num_planning_attempts)
        self.move_group_arm.set_planning_time(allowed_planning_time)
        self.move_group_arm.set_goal_joint_tolerance(tolerance)

        if frame_id is not None:
            self.move_group_arm.set_pose_reference_frame(frame_id)

        if plan_only:
            self.move_group_arm.set_pose_target(pose_stamped)
            plan = self.move_group_arm.plan()

            # visualize plan
            display_trajectory = moveit_msgs.msg.DisplayTrajectory()
            display_trajectory.trajectory_start = self.robot.get_current_state()
            display_trajectory.trajectory.append(plan)
            self.display_trajectory_publisher.publish(display_trajectory)

            if return_plan:
                return plan

            return 'SUCCESS' if len(plan.joint_trajectory.points)>0 else "FAIL"
        else:
            self.move_group_arm.set_pose_target(pose_stamped)
            res = self.move_group_arm.go(wait=True)

            # Calling ``stop()`` ensures that there is no residual movement
            self.move_group_arm.stop()

            return moveit_error_string(res)

    def move_to_poses(self,
                     pose_stamped,
                     allowed_planning_time=10.0,
                     execution_timeout=15.0,
                     num_planning_attempts=200,
                     orientation_constraint=None,
                     plan_only=False,
                     replan=False,
                     replan_attempts=5,
                     tolerance=0.01,
                     return_plan=False,
                     frame_id='base_link'):
        self.move_group_arm.set_num_planning_attempts(num_planning_attempts)
        self.move_group_arm.set_planning_time(allowed_planning_time)
        self.move_group_arm.set_goal_joint_tolerance(tolerance)
        self.move_group_arm.set_pose_reference_frame(frame_id)

        if plan_only:
            # print("pose_stamped:", pose_stamped, type(pose_stamped))
            self.move_group_arm.set_pose_targets(pose_stamped)
            plan = self.move_group_arm.plan()

            # visualize plan
            display_trajectory = moveit_msgs.msg.DisplayTrajectory()
            display_trajectory.trajectory_start = self.robot.get_current_state()
            display_trajectory.trajectory.append(plan)
            self.display_trajectory_publisher.publish(display_trajectory)

            if return_plan:
                return plan

            return 'SUCCESS' if len(plan.joint_trajectory.points)>0 else "FAIL"
        else:
            self.move_group_arm.set_pose_targets(pose_stamped)
            res = self.move_group_arm.go(wait=True)

            # Calling ``stop()`` ensures that there is no residual movement
            self.move_group_arm.stop()

            return moveit_error_string(res)

    def move_to_joints_direct(self, joint_state):
        goal = control_msgs.msg.FollowJointTrajectoryGoal()
        goal.trajectory.joint_names.extend(ArmJoints.names())
        point = trajectory_msgs.msg.JointTrajectoryPoint()
        point.positions.extend(joint_state)
        point.time_from_start = rospy.Duration(TIME_FROM_START)
        goal.trajectory.points.append(point)
        self._joint_client.send_goal(goal)
        self._joint_client.wait_for_result(rospy.Duration(10))

    def execute_plan(self, plan):
        res = self.move_group_arm.execute(plan)
        print(res)

    def move_in_cartesian(self, dx=0, dy=0, dz=0, frame_id="base_link"):
        req = CartesianGoalRequest()

        req.action = CartesianGoalRequest.MOVE_BY_DISTANCE
        req.frame_id = frame_id
        req.dx = dx
        req.dy = dy
        req.dz = dz

        try:
            rospy.wait_for_service(CARTESIAN_SERVER)
            res = self._cartesian_client(req)
        except (rospy.ServiceException, rospy.ROSException) as e:
            rospy.logerr('Service call failed')
            return False, None, None

        return res.executed, res.pose_init, res.pose_final

    def get_current_pose(self):
        return self.move_group_arm.get_current_pose()