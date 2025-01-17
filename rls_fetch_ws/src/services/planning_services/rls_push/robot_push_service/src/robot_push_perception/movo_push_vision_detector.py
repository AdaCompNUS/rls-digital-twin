#!/usr/bin/env python

import rospy
import cv2
import numpy as np
import math
from geometry_msgs.msg import PoseStamped, Pose, Point, PointStamped

from tf import transformations
from rls_perception_msgs.srv import FetchImage

from robot_push_vision_detector import RobotPushVisionDetector
from process_img import *

# Settings

# Constants
W = 128
H = 106
Z_OFFSET = 0.1
MOVO_GRIPPER_LENGTH = 0.12
APPROACH_DZ = Z_OFFSET - 0.02

SEG_SERVICE_NAME = 'tabletop_segmentation'


class MovoPushVisionDetector(RobotPushVisionDetector):
    def __init__(self):
        super(MovoPushVisionDetector, self).__init__()
        # tabletop segmentation setting
        self.world_frame = 'base_link'
        self.table_for_seg = None
        self._table_height = 0
        print('segmentation service is ready')

        # setup image fetch client
        self.img_client = rospy.ServiceProxy(
            'rls_perception_services/movo/fetch_image', FetchImage)
        print('MovoPushVisionDetector initialized')

    ''' Segmentation '''

    def segment(self):
        '''
        @Override
        perform segmentation
        '''
        # get point cloud markers
        # self.active_sub.request_pcl()
        result = self.seg_client(self.table_for_seg, "base_link")
        markers = result.markers
        if result.table is not None:
            print("MovoPushVisionDetector/segment: new table remembered")
            self._table_for_seg = result.table
        # self.active_sub.unregister_pcl()
        return markers

    def get_current_rgb(self):
        '''
        retrieve current rgb image from fetch onboard server
        '''
        img_msg = self.img_client(int(1)).color
        cv_img = self.bridge.imgmsg_to_cv2(img_msg, 'bgr8')
        return cv_img

    def set_table_height(self, table_height):
        '''
        @override
        '''
        self._table_height = table_height

    def pixel_to_pose(self, best_start, best_end, xc, yc):
        '''
        Convert best_start, best_end in image frame to pose in robot base_link frame
        @best_start, xy coordinate in image frame
        @best_end, xy coordinate in image frame
        @xc, center of points of objects in robot base_link frame
        @yc, center of points of objects in robot base_link frame
        '''

        x0 = - (best_start[1] - H / 2.0) * self.meters_per_pixel + xc
        y0 = - (best_start[0] - W / 2.0) * self.meters_per_pixel + yc
        x1 = - (best_end[1] - H / 2.0) * self.meters_per_pixel + xc
        y1 = - (best_end[0] - W / 2.0) * self.meters_per_pixel + yc

        start_pose = PoseStamped()
        end_pose = PoseStamped()

        # get orientation aroud z axis
        angle_z = math.atan2(x1 - x0, y1 - y0)  # in rads

        # rpy,
        # rotate around y axis to ensure gripper is facing down
        # rotate around z axis to ensure the gripper is approaching object in the right angle
        q = transformations.quaternion_from_euler(0, math.pi / 2, angle_z)

        # calculate start_pose and end_pose
        start_pose.header.frame_id = self.world_frame
        start_pose.pose.position.x = x0
        start_pose.pose.position.y = y0
        start_pose.pose.position.z = self._table_height + Z_OFFSET + MOVO_GRIPPER_LENGTH
        start_pose.pose.orientation.x = q[0]
        start_pose.pose.orientation.y = q[1]
        start_pose.pose.orientation.z = q[2]
        start_pose.pose.orientation.w = q[3]

        end_pose.header.frame_id = self.world_frame
        end_pose.pose.position.x = x1
        end_pose.pose.position.y = y1
        end_pose.pose.position.z = self._table_height + Z_OFFSET + MOVO_GRIPPER_LENGTH
        end_pose.pose.orientation.x = q[0]
        end_pose.pose.orientation.y = q[1]
        end_pose.pose.orientation.z = q[2]
        end_pose.pose.orientation.w = q[3]

        # start_pose.pose.position.z += self.pre_height
        dx = end_pose.pose.position.x - start_pose.pose.position.x
        dy = end_pose.pose.position.y - start_pose.pose.position.y
        # print 'action in meters: dx, dy'
        # print dx, dy

        # while not rospy.is_shutdown():
        # for i in range(100):
        #     # while not rospy.is_shutdown():
        #     self.p1_pub.publish(start_pose_robot)
        #     self.p2_pub.publish(end_pose_robot)

        return start_pose, dx, dy

    def get_dz(self):
        '''
        Get the approach and depart height
        '''
        return APPROACH_DZ


def test_input():
    # generate random points
    P = []
    num = 1000
    x_range = [0, 0.2]
    y_range = [0, 0.3]
    x_values = np.random.uniform(x_range[0], x_range[1], num)
    y_values = np.random.uniform(y_range[0], y_range[1], num)
    for i in range(num):
        p = Point()
        p.x = x_values[i]
        p.y = y_values[i]
        P.append(p)

    xc, yc = get_point_center(P)
    img = point_to_image(P, xc, yc)
    img_con = fill_contour(img.copy())
    cv2.imshow('mask', img_con)
    cv2.waitKey(0)


def test_full():
    S = MovoPushVisionDetector()
    points = S.segment().points
    img_inp, xc, yc = S.generate_input_image(points)
    cv2.imshow('mask', img_inp)
    cv2.waitKey(0)


if __name__ == '__main__':
    rospy.init_node('input_test', log_level=rospy.INFO)
    test_full()
