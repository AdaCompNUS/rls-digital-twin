#!/usr/bin/env python

import rospy
import tf

import fetch_api
from fetch_robot_config import *

# Constants
end_eff_frame = 'gripper_link'
gripper_to_wrist = 0.16645
fetch_gripper_height = 0.81  # for lower box eg. bowl, sugar box
# fetch_gripper_height = 0.85 ##for higher box eg. stick/straw
z_diff = 0.10
gripper_to_leftout_y = -0.09


class FetchPushController():
    def __init__(self):
        self.arm = fetch_api.ArmV2()
        # self.arm.set_max_velocity_scaling_factor(0.45)
        self.arm.set_max_velocity_scaling_factor(0.9)
        self.gripper = fetch_api.Gripper()

    def set_table_height(self, table_height):
        self._table_height = table_height

    def move_arm_to_pose(self, pose, plan_only=False):
        res = self.arm.move_to_pose(pose, allowed_planning_time=2.0, plan_only = plan_only)
        return res == "SUCCESS"

    def move_arm_in_cartesian(self, dx=0.0, dy=0.0, dz=0.0):
        self.arm.move_in_cartesian(dx, dy, dz)
        return True

    def move_arm_to_home(self):
        res = self.arm.move_to_joint(hide_left_joints, plan_only=False, allowed_planning_time=2)
        self.arm.set_max_velocity_scaling_factor()
        if res != "SUCCESS":
            rospy.loginfo("SimplePnp: fail to move to ready pick pose")
            return False

        self.gripper.close()
        return True

    def say(self, text):
        print("dummy say")
        return True

    # def get_dz(self):
    #     time = 0
    #     trans = None
    #     qat = None
    #     z_curr = 0
    #     iteration = 15
    #     for i in range(iteration):
    #         while not rospy.is_shutdown():
    #             try:
    #                 time = self._vision_detector.tl.getLatestCommonTime(
    #                     self.robot_frame, end_eff_frame)
    #                 (trans, qat) = self._vision_detector.tl.lookupTransform(
    #                     self.robot_frame, end_eff_frame, time)
    #                 z_curr += trans[2]
    #                 break
    #             except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
    #                 print 'try again'
    #                 continue

    #     # take an average with the hope it will be more accurate
    #     z_curr = z_curr / iteration
    #     diff = fetch_gripper_height - z_curr
    #     print diff
    #     if abs(diff) > 0.11:
    #         print 'cannot move too much!'
    #         # if self.left:
    #         #     self.arm.move_to_joint_goal(
    #         #         HIDE_LEFT_JOINTS, replan=True, execution_timeout=20.0)
    #         # else:
    #         #     self.arm.move_to_joint_goal(
    #         #         HIDE_RIGHT_JOINTS, replan=True, execution_timeout=20.0)
    #         # self.arm.move_to_joint(
    #         #     READY_PICK_R_JOINTS, group_name='right_arm')
    #         return None, None, None

    def grasp_from_side(self, side):
        rospy.loginfo("grasp_from_side, skip")
        return

        if side == 'left':
            # self.arm.move_to_joint_goal(HIDE_LEFT_JOINTS, replan=True)
            rospy.sleep(1)
            # self.arm.move_to_joint_goal(LEFT_GRASP_JOINTS, replan=True)
            rospy.sleep(1)
        # self.gripper.open()

        self.update_tf()
        points = self.get_points()

        x_avg = 0
        y_avg = 0
        z_avg = 0
        count = 0
        for p in points:
            if -0.13 < p.y < 0:
                count += 1
                x_avg += p.x
                y_avg += p.y
                z_avg += p.z

        x_avg = x_avg / count
        y_avg = y_avg / count
        z_avg = z_avg / count

        center = PointStamped()
        center.header.frame_id = self._robot_vision_detector.world_frame
        center.point.x = x_avg
        # center.point.y = y_avg
        center.point.y = gripper_to_leftout_y
        center.point.z = z_avg

        for i in range(100):
            self.goal_pub.publish(center)

        # transform wrt to base_link
        time = 0
        center_transform = PointStamped()
        while not rospy.is_shutdown():
            try:
                time = self._robot_vision_detector.tl.getLatestCommonTime(
                    self.robot_frame, self._robot_vision_detector.world_frame)
                center.header.stamp = time
                center_transform = self._robot_vision_detector.tl.transformPoint(
                    self.robot_frame, center)
                break
            except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue
        # get current gripper link pose
        time = 0
        trans = None
        rot = None
        while not rospy.is_shutdown():
            try:
                time = self._robot_vision_detector.tl.getLatestCommonTime(
                    self.robot_frame, end_eff_frame)
                (trans, qat) = self._robot_vision_detector.tl.lookupTransform(
                    self.robot_frame, end_eff_frame, time)
                break
            except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                continue

        x_diff = center_transform.point.x - trans[0]
        # y_diff = center_transform.point.y - trans[1] + 0.015
        y_diff = center_transform.point.y - trans[1]
        z_diff = center_transform.point.z - trans[2] - 0.02
        print x_diff, y_diff, z_diff
        # raw_input('wait')

        # self.arm.move_by_z(dz=z_diff)
        # self.torso.set_height(fetch_api.Torso.MAX_HEIGHT + z_diff)
        # self.torso.move_by(z_diff)
        rospy.sleep(1)
        # raw_input('wait')
        # self.arm.move_in_cartesian(dx=x_diff, dy=y_diff)
        # self.arm.move_by_xy(dx=x_diff, dy=y_diff)
        # rospy.sleep(1)
        # raw_input('wait')

        # self.gripper.close(50)
        # raw_input('wait')
        # self.torso.set_height(self.torso.MAX_HEIGHT)

        # rotate the wrist forward 0.8279026104675293 (6th joint)
        # self.joint_angle[5] = 0.8279026104675293
        # HAND_OVER_JOINTS = [(right_joint_names[i], self.joint_angle[i])
        #                     for i in range(7)]
        # self.arm.move_to_joint_goal(HAND_OVER_JOINTS, replan=True)

        # self.arm.move_in_cartesian(dz=0.2)
        # self.arm.move_in_cartesian(dx=0.10)
        # self.arm.move_by_z(dz=0.2)
        # rospy.sleep(1)
        # if side == 'left':
        #    self.arm.move_to_joint_goal(HOLD_LEFT_JOINTS, replan=True)
        # rospy.sleep(1)
