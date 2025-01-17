#! /usr/bin/env python

import actionlib
import copy
import geometry_msgs.msg
import math
import nav_msgs.msg
import numpy as np
import rospy
import tf.transformations as tft
import time
from threading import Lock

class Base(object):
    """Base controls the mobile base portion of the Fetch robot.

    Sample usage:
        base = fetch_api.Base()

        base.go_forward(0.1)
        base.turn(30 * math.pi / 180)

        while CONDITION:
            base.move(0.2, 0)
        base.stop()
    """

    def __init__(self):
        self._publisher = rospy.Publisher(
            'cmd_vel', geometry_msgs.msg.Twist, queue_size=1)
        self.odom_sub = rospy.Subscriber(
            'odom',
            nav_msgs.msg.Odometry,
            callback=self._odom_callback,
            queue_size=1)
        self._lock = Lock()
        self.odom = None

    def move(self, linear_speed, angular_speed):
        """Moves the base instantaneously at given linear and angular speeds.

        "Instantaneously" means that this method must be called continuously in
        a loop for the robot to move.

        Args:
            linear_speed: The forward/backward speed, in meters/second. A
                positive value means the robot should move forward.
            angular_speed: The rotation speed, in radians/second. A positive
                value means the robot should rotate clockwise.
        """
        twist = geometry_msgs.msg.Twist()
        twist.linear.x = linear_speed
        twist.angular.z = angular_speed
        # print(twist)
        self._publisher.publish(twist)

    def go_forward(self, distance, speed=0.1):
        """Moves the robot a certain distance.

        It's recommended that the robot move slowly. If the robot moves too
        quickly, it may overshoot the target. Note also that this method does
        not know if the robot's path is perturbed (e.g., by teleop). It stops
        once the distance traveled is equal to the given distance.

        You cannot use this method to move less than 1 cm.

        Args:
            distance: The distance, in meters, to rotate. A positive value
                means forward, negative means backward.
            max_speed: The maximum speed to travel, in meters/second.
        """
        while self.odom is None:
            rospy.sleep(0.1)
        start = copy.deepcopy(self.odom)
        rate = rospy.Rate(25)
        distance_from_start = self._linear_distance(start, self.odom)
        while distance_from_start < math.fabs(distance):
            distance_from_start = self._linear_distance(start, self.odom)
            if distance_from_start >= math.fabs(distance):
                return
            direction = -1 if distance < 0 else 1
            self.move(direction * speed, 0)
            rate.sleep()

    def turn(self, angular_distance, max_speed=3.0, timeout = 5):
        """Rotates the robot a certain angle.

        This illustrates how to turn the robot by checking that the X-axis of
        the robot matches that of the goal.

        Args:
            angular_distance: The angle, in radians, to rotate. A positive
                value rotates counter-clockwise.
            speed: The maximum angular speed to rotate, in radians/second.
        """
        while self.odom is None:
            rospy.sleep(0.1)

        rate = rospy.Rate(25)

        direction = -1 if angular_distance < 0 else 1

        with self._lock:
            odom = copy.deepcopy(self.odom)

        current_yaw = self._yaw_from_quaternion(odom.orientation)
        goal_yaw = current_yaw + angular_distance
        if goal_yaw < 0:
            goal_yaw += 2 * math.pi
        tmp = int(goal_yaw / math.pi / 2)
        goal_yaw = goal_yaw - tmp * math.pi * 2

        # goal_x_axis = np.array([math.cos(goal_yaw), math.sin(goal_yaw), 0])
        # x_axis = self._x_axis_from_quaternion(odom.orientation)

        # # slow down speed if angular distance is small
        # speed = min(speed, direction * angular_distance * speed)
        # print("speed: {}".format(speed))

        # while not np.allclose(x_axis, goal_x_axis, atol=0.01):
        #     self.move(0, direction * speed)
        #     with self._lock:
        #         odom = copy.deepcopy(self.odom)

        #     x_axis = self._x_axis_from_quaternion(odom.orientation)
        #     rate.sleep()
        #     cur_time = time.time()
        #     if cur_time - start_time > timeout:
        #         rospy.logerr("turn failed")
        #         break

        if direction == 1:
            # rotate anticlockwise, current yaw has to be smaller than goal_yaw
            if current_yaw > goal_yaw:
                current_yaw -= 2 * math.pi
        else:
            # rotate clockwise, current yaw has to be larger than goal_yaw
            if current_yaw < goal_yaw:
                current_yaw += 2 * math.pi

        initial_yaw = current_yaw
        start_time = time.time()
        error = goal_yaw - current_yaw
        initial_error = error
        error_sum = 0
        while math.fabs(error) > 0.05: # about 8 degree error
            cur_speed = error * 1.2 + error_sum * 0.5 # PI control
            if cur_speed > max_speed:
                cur_speed = max_speed
            if cur_speed < -max_speed:
                cur_speed = -max_speed
            # print(cur_speed)
            self.move(0, cur_speed)
            rate.sleep()
            self.move(0, 0)

            prev_error = error
            with self._lock:
                odom = copy.deepcopy(self.odom)
            current_yaw = self._yaw_from_quaternion(odom.orientation)
            if direction == 1:
                # rotate anticlockwise, current yaw has to be smaller than goal_yaw
                if current_yaw > goal_yaw:
                    current_yaw -= 2 * math.pi
            else:
                # rotate clockwise, current yaw has to be larger than goal_yaw
                if current_yaw < goal_yaw:
                    current_yaw += 2 * math.pi

            # print(current_yaw)
            error = goal_yaw - current_yaw
            # print(error)
            # error_sum += error

            if math.fabs(error - initial_error) < initial_error * 0.1:
                error_sum += error
            else:
                error_sum = 0

            # print(error_sum)
            cur_time = time.time()
            if cur_time - start_time > timeout:
                rospy.logerr("[FetchBase]: turn failed, goal_yaw: {}, current_yaw: {}".format(goal_yaw, current_yaw))
                break

        rospy.loginfo("[FetchBase]: turn finished: from {} to {}".format(math.degrees(initial_yaw), math.degrees(current_yaw)))

    def turn_alternate(self, angular_distance, speed=1.0):
        """Rotates the robot a certain angle.

        This illustrates how to turn the robot using yaw angles and careful
        accounting.

        Args:
            angular_distance: The angle, in radians, to rotate. A positive
                value rotates counter-clockwise.
            speed: The maximum angular speed to rotate, in radians/second.
        """
        while self.odom is None:
            rospy.sleep(0.1)
        direction = -1 if angular_distance < 0 else 1

        current_coord = self._yaw_from_quaternion(self.odom.orientation) % (
            2 * math.pi)
        end_coord = (current_coord + angular_distance) % (2 * math.pi)
        rate = rospy.Rate(25)

        while True:
            current_coord = self._yaw_from_quaternion(
                self.odom.orientation) % (2 * math.pi)
            remaining = (direction *
                         (end_coord - current_coord)) % (2 * math.pi)
            if remaining < 0.01:
                return
            speed = max(0.25, min(1, remaining))
            self.move(0, direction * speed)
            rate.sleep()

    def stop(self):
        """Stops the mobile base from moving.
        """
        self.move(0, 0)

    def _odom_callback(self, msg):
        with self._lock:
            self.odom = msg.pose.pose

    @staticmethod
    def _linear_distance(pose1, pose2):
        pos1 = pose1.position
        pos2 = pose2.position
        dx = pos1.x - pos2.x
        dy = pos1.y - pos2.y
        dz = pos1.z - pos2.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    @staticmethod
    def _x_axis_from_quaternion(q):
        m = tft.quaternion_matrix([q.x, q.y, q.z, q.w])
        return m[:3, 0]  # First column

    @staticmethod
    def _yaw_from_quaternion(q):
        m = tft.quaternion_matrix([q.x, q.y, q.z, q.w])
        yaw = math.atan2(m[1, 0], m[0, 0])
        if yaw < 0:
            yaw += 2 * math.pi

        return yaw

    @staticmethod
    def _angular_distance(pose1, pose2):
        q1 = pose1.orientation
        q2 = pose2.orientation
        y1 = Base._yaw_from_quaternion(q1)
        y2 = Base._yaw_from_quaternion(q2)
        return math.fabs(y1 - y2) % (2 * math.pi)
