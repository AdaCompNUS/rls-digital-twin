#!/usr/bin/env python

import actionlib
import control_msgs.msg
import geometry_msgs.msg
import trajectory_msgs.msg
import math
import moveit_commander
import rospy
import tf

from .arm_joints import ArmJoints
from .moveit_goal_builder import MoveItGoalBuilder
from moveit_msgs.msg import MoveItErrorCodes, MoveGroupAction
from moveit_msgs.srv import GetPositionIK, GetPositionIKRequest
from tf.listener import TransformListener

from geometry_msgs.msg import TwistStamped, Pose, PoseStamped
from sensor_msgs.msg import JointState
from math import sqrt, pow, sin, cos, atan2

from fetch_drivers.srv import *

import traceback

ARM_GROUP_NAME = 'arm'
JOINT_ACTION_SERVER = 'arm_controller/follow_joint_trajectory'
MOVE_GROUP_ACTION_SERVER = 'move_group'
TIME_FROM_START = 5

CARTESIAN_SERVER = 'fetch_cartesian'

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


class Arm(object):
    """Arm controls the robot's arm.

    Joint space control:
        joints = ArmJoints()
        # Fill out joint states
        arm = fetch_api.Arm()
        arm.move_to_joints(joints)
    """

    def __init__(self):
        self._joint_client = actionlib.SimpleActionClient(
            JOINT_ACTION_SERVER, control_msgs.msg.FollowJointTrajectoryAction)
        self._joint_client.wait_for_server(rospy.Duration(10))
        self._move_group_client = actionlib.SimpleActionClient(
            MOVE_GROUP_ACTION_SERVER, MoveGroupAction)
        self._move_group_client.wait_for_server(rospy.Duration(10))
        self._compute_ik = rospy.ServiceProxy('compute_ik', GetPositionIK)
        self._tf_listener = TransformListener()

        ## cartesian controller client
        # self._cartesian_client = rospy.ServiceProxy(CARTESIAN_SERVER, CartesianGoal)
        # rospy.loginfo("waiting for CARTESIAN_SERVER in arm.py...")
        # rospy.wait_for_service(CARTESIAN_SERVER)
        # rospy.loginfo("CARTESIAN_SERVER in arm.py is up!")

        ## cartesian controller parameters
        #self.P_gain = 10
        #self.D_gain = 0
        #self.max_vel = 0.06
        #self.pose = PoseStamped()
        #self.pose_sub = rospy.Subscriber('tool_pose', PoseStamped, self.pose_callback)
        #self.vel_pub = rospy.Publisher('/arm_controller/cartesian_twist/command', TwistStamped)
        #self.tolerance = 0.005
        #self.valid_pose_received = False
        #self.rate = rospy.Rate(50)

    #def clean_exit(self):
    #    self.pose_sub.unregister()

    def move_in_cartesian(self, dx=0, dy=0, dz=0):
        req = CartesianGoalRequest()
        req.dx = dx
        req.dy = dy
        req.dz = dz

        res = CartesianGoalResponse()
        try:
            rospy.wait_for_service(CARTESIAN_SERVER)
            res = self._cartesian_client(req)
        except (rospy.ServiceException, rospy.ROSException), e:
            rospy.logerr('Service call failed')
            return False, None, None

        return res.executed, res.pose_init, res.pose_final


    def move_to_joints(self, joint_state):
        goal = control_msgs.msg.FollowJointTrajectoryGoal()
        goal.trajectory.joint_names.extend(ArmJoints.names())
        point = trajectory_msgs.msg.JointTrajectoryPoint()
        point.positions.extend(joint_state.values())
        point.time_from_start = rospy.Duration(TIME_FROM_START)
        goal.trajectory.points.append(point)
        self._joint_client.send_goal(goal)
        self._joint_client.wait_for_result(rospy.Duration(10))

    def move_to_joint_goal(self,
                           joints,
                           allowed_planning_time=0.5,
                           execution_timeout=15.0,
                           group_name='arm',
                           num_planning_attempts=10,
                           plan_only=False,
                           replan=False,
                           replan_attempts=5,
                           tolerance=0.01):
        """Moves the end-effector to a pose, using motion planning.

        Args:
            joints: A list of (name, value) for the arm joints.
            allowed_planning_time: float. The maximum duration to wait for a
                planning result.
            execution_timeout: float. The maximum duration to wait for an arm
                motion to execute (or for planning to fail completely), in
                seconds.
            group_name: string. Either 'arm' or 'arm_with_torso'.
            num_planning_attempts: int. The number of times to compute the same
                plan. The shortest path is ultimately used. For random
                planners, this can help get shorter, less weird paths.
            plan_only: bool. If True, then this method does not execute the
                plan on the robot. Useful for determining whether this is
                likely to succeed.
            replan: bool. If True, then if an execution fails (while the arm is
                moving), then come up with a new plan and execute it.
            replan_attempts: int. How many times to replan if the execution
                fails.
            tolerance: float. The goal tolerance, in meters.

        Returns:
            string describing the error if an error occurred, else None.
        """
        goal_builder = MoveItGoalBuilder()
        goal_builder.set_joint_goal(*zip(*joints))
        goal_builder.allowed_planning_time = allowed_planning_time
        goal_builder.num_planning_attempts = num_planning_attempts
        goal_builder.plan_only = plan_only
        #goal_builder.planner_id = 'BITstarConfigDefault'
        goal_builder.planner_id = 'RRTstarkConfigDefault'
        goal_builder.replan = replan
        goal_builder.replan_attempts = replan_attempts
        goal_builder.tolerance = tolerance
        goal = goal_builder.build()
        if goal is not None:
            self._move_group_client.send_goal(goal)
            success = self._move_group_client.wait_for_result(
                rospy.Duration(execution_timeout))
            if not success:
                return moveit_error_string(MoveItErrorCodes.TIMED_OUT)
            result = self._move_group_client.get_result()

        if result:
            if result.error_code.val == MoveItErrorCodes.SUCCESS:
                return None
            else:
                return moveit_error_string(result.error_code.val)
        else:
            return moveit_error_string(MoveItErrorCodes.TIMED_OUT)

    def move_to_pose(self,
                     pose_stamped,
                     allowed_planning_time=1.0,
                     execution_timeout=15.0,
                     group_name='arm',
                     num_planning_attempts=50,
                     orientation_constraint=None,
                     plan_only=False,
                     replan=False,
                     replan_attempts=5,
                     tolerance=0.01):
        """Moves the end-effector to a pose, using motion planning.

        Args:
            pose: geometry_msgs/PoseStamped. The goal pose for the gripper.
            allowed_planning_time: float. The maximum duration to wait for a
                planning result.
            execution_timeout: float. The maximum duration to wait for an arm
                motion to execute (or for planning to fail completely), in
                seconds.
            group_name: string. Either 'arm' or 'arm_with_torso'.
            num_planning_attempts: int. The number of times to compute the same
                plan. The shortest path is ultimately used. For random
                planners, this can help get shorter, less weird paths.
            orientation_constraint: moveit_msgs/OrientationConstraint. An
                orientation constraint for the entire path.
            plan_only: bool. If True, then this method does not execute the
                plan on the robot. Useful for determining whether this is
                likely to succeed.
            replan: bool. If True, then if an execution fails (while the arm is
                moving), then come up with a new plan and execute it.
            replan_attempts: int. How many times to replan if the execution
                fails.
            tolerance: float. The goal tolerance, in meters.

        Returns:
            string describing the error if an error occurred, else None.
        """
        goal_builder = MoveItGoalBuilder()
        goal_builder.set_pose_goal(pose_stamped)
        if orientation_constraint is not None:
            goal_builder.add_path_orientation_constraint(orientation_constraint)
        goal_builder.allowed_planning_time = allowed_planning_time
        goal_builder.num_planning_attempts = num_planning_attempts
        goal_builder.plan_only = plan_only
        goal_builder.replan = replan
        goal_builder.replan_attempts = replan_attempts
        goal_builder.tolerance = tolerance
        goal = goal_builder.build()
        if goal is not None:
            self._move_group_client.send_goal(goal)
            self._move_group_client.wait_for_result(
                rospy.Duration(execution_timeout))
            result = self._move_group_client.get_result()

        if result:
            if result.error_code.val == MoveItErrorCodes.SUCCESS:
                return None
            else:
                return moveit_error_string(result.error_code.val)
        else:
            return moveit_error_string(MoveItErrorCodes.TIMED_OUT)


    def move_to_named_target(self, group, named_target):
        '''
        move the arm to a predefined joint configuration
        '''
        plan = None
        group.set_planner_id('BITstarConfigDefault')
        group.set_num_planning_attempts(10)
        while not rospy.is_shutdown():
            try:
                group.clear_pose_targets()
                group.set_named_target(named_target)
                print "==== Waiting while setting joint value target"

                try:
                    plan = group.plan()
                except Exception, e:
                    print traceback.format_exc()
                    continue
                if len(plan.joint_trajectory.points) == 0:
                    print "trajectory is empty"
                    continue
                break
            except Exception, err:
                print 'No IK solution is found'
                print traceback.format_exc()
                continue

        result = group.execute(plan, wait=True)
        if not result:
            return moveit_error_string(MoveItErrorCodes.INVALID_MOTION_PLAN)
        else:
            return None



    def straight_move_to_pose(self,
                              group,
                              pose_stamped,
                              ee_step=0.025,
                              jump_threshold=2.0,
                              avoid_collisions=True):
        """Moves the end-effector to a pose in a straight line.

        Args:
          group: moveit_commander.MoveGroupCommander. The planning group for
            the arm.
          pose_stamped: geometry_msgs/PoseStamped. The goal pose for the
            gripper.
          ee_step: float. The distance in meters to interpolate the path.
          jump_threshold: float. The maximum allowable distance in the arm's
            configuration space allowed between two poses in the path. Used to
            prevent "jumps" in the IK solution.
          avoid_collisions: bool. Whether to check for obstacles or not.

        Returns:
            string describing the error if an error occurred, else None.
        """
        # Transform pose into planning frame
        group.set_planner_id('BITstarConfigDefault')
        group.set_num_planning_attempts(10)
        self._tf_listener.waitForTransform(pose_stamped.header.frame_id,
                                           group.get_planning_frame(),
                                           rospy.Time.now(),
                                           rospy.Duration(1.0))
        try:
            pose_transformed = self._tf_listener.transformPose(
                group.get_planning_frame(), pose_stamped)
        except (tf.LookupException, tf.ConnectivityException):
            rospy.logerr('Unable to transform pose from frame {} to {}'.format(
                pose_stamped.header.frame_id, group.get_planning_frame()))
            return moveit_error_string(
                MoveItErrorCodes.FRAME_TRANSFORM_FAILURE)

        # Compute path
        plan, fraction = group.compute_cartesian_path(
            [group.get_current_pose().pose,
             pose_transformed.pose], ee_step, jump_threshold, avoid_collisions)
        if fraction < 1 and fraction > 0:
            rospy.logerr(
                'Only able to compute {}% of the path'.format(fraction * 100))
        if fraction == 0:
            return moveit_error_string(MoveItErrorCodes.PLANNING_FAILED)

        # Execute path
        result = group.execute(plan, wait=True)
        if not result:
            return moveit_error_string(MoveItErrorCodes.INVALID_MOTION_PLAN)
        else:
            return None

    def check_pose(self,
                   pose_stamped,
                   allowed_planning_time=10.0,
                   group_name='arm',
                   tolerance=0.01):
        return self.move_to_pose(
            pose_stamped,
            allowed_planning_time=allowed_planning_time,
            group_name=group_name,
            tolerance=tolerance,
            plan_only=True)

    def compute_ik(self, pose_stamped, timeout=rospy.Duration(5)):
        """Computes inverse kinematics for the given pose.

        Note: if you are interested in returning the IK solutions, we have
            shown how to access them.

        Args:
            pose_stamped: geometry_msgs/PoseStamped.
            timeout: rospy.Duration. How long to wait before giving up on the
                IK solution.

        Returns: A list of (name, value) for the arm joints if the IK solution
            was found, False otherwise.
        """
        request = GetPositionIKRequest()
        request.ik_request.pose_stamped = pose_stamped
        request.ik_request.group_name = 'arm'
        request.ik_request.timeout = timeout
        response = self._compute_ik(request)
        error_str = moveit_error_string(response.error_code.val)
        success = error_str == 'SUCCESS'
        if not success:
            return False
        joint_state = response.solution.joint_state
        joints = []
        for name, position in zip(joint_state.name, joint_state.position):
            if name in ArmJoints.names():
                joints.append((name, position))
        return joints

    def cancel_all_goals(self):
        self._move_group_client.cancel_all_goals()
        self._joint_client.cancel_all_goals()

#    def pose_callback(self, data):
#        ''' get gripper link pose with respect to base_link'''
#        self.valid_pose_received = True
#        self.pose = data
#
#    def is_pose_received(self):
#        return self.valid_pose_received
#
#    def move_by_xy(self, dx=0, dy=0):
#        ''' move the gripper link in xy plan in a straight line'''
#        vel_msg = TwistStamped()
#        vel_msg.header.frame_id = 'base_link'
#        goal_x = self.pose.pose.position.x + dx
#        goal_y = self.pose.pose.position.y + dy
#        dx_prev = dx
#        dy_prev = dy
#
#        while not rospy.is_shutdown():
#            dxx = goal_x - self.pose.pose.position.x
#            dyy = goal_y - self.pose.pose.position.y
#            err = sqrt(pow(dxx, 2) + pow(dyy, 2))
#
#            ## PD Control
#            vel_x_PD = self.P_gain * cos(atan2(dyy, dxx)) * abs(dxx) + self.D_gain * (dxx - dx_prev)
#            vel_y_PD = self.P_gain * sin(atan2(dyy, dxx)) * abs(dyy) + self.D_gain * (dyy - dy_prev)
#
#            dx_prev = dxx
#            dy_prev = dyy
#
#            if abs(vel_x_PD) > abs(self.max_vel * cos(atan2(dyy, dxx))):
#                vel_x = self.max_vel * cos(atan2(dyy, dxx))
#                #print 'X: constant'
#            else:
#                vel_x = vel_x_PD
#                #print 'X: PD'
#
#            if abs(vel_y_PD) > abs(self.max_vel * sin(atan2(dyy, dxx))):
#                vel_y = self.max_vel * sin(atan2(dyy, dxx))
#                #print 'Y: constant'
#            else:
#                vel_y = vel_y_PD
#                #print 'Y: PD'
#
#            if err < self.tolerance:
#                break
#            vel_msg.twist.linear.x = vel_x
#            vel_msg.twist.linear.y = vel_y
#            self.vel_pub.publish(vel_msg)
#            self.rate.sleep()
#
#        vel_msg.twist.linear.x = 0
#        vel_msg.twist.linear.y = 0
#        self.vel_pub.publish(vel_msg)
#        rospy.sleep(0.5)
#
#    def move_by_z(self, dz=0):
#        ''' move the gripper link in z plan in a straight line'''
#        vel_msg = TwistStamped()
#        vel_msg.header.frame_id = 'base_link'
#        goal_z = self.pose.pose.position.z + dz
#        dz_prev = dz
#
#        while not rospy.is_shutdown():
#            dzz = goal_z - self.pose.pose.position.z
#            vel_z_PD = self.P_gain * dzz + self.D_gain * (dzz - dz_prev)
#            dz_prev = dzz
#            if abs(dzz) < self.tolerance:
#                break
#            if abs(vel_z_PD) > abs(self.max_vel):
#                #print 'Z: constant'
#                if dzz > 0:
#                    vel_z = self.max_vel
#                else:
#                    vel_z = - self.max_vel
#            else:
#                #print 'Z: PD'
#                vel_z = vel_z_PD
#
#            vel_msg.twist.linear.z = vel_z
#            self.vel_pub.publish(vel_msg)
#            self.rate.sleep()
#
#        vel_msg.twist.linear.z = 0
#        self.vel_pub.publish(vel_msg)
#        rospy.sleep(0.5)
#





