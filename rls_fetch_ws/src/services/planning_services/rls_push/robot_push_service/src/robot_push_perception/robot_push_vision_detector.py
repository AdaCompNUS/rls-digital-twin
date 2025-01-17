
#!/usr/bin/env python

import numpy as np
import os
import rospy
from sensor_msgs.msg import Image, PointCloud2
import json
import math
import argparse
from cv_bridge import CvBridge
import imutils
from process_img import *
import tf
import cv2
from shape_msgs.msg import MeshTriangle
from geometry_msgs.msg import PoseStamped, Pose, Point, PointStamped
from cv_bridge import CvBridge
from rls_perception_msgs.srv import *


# Constants
W = 128
H = 106

SEG_SERVICE_NAME = 'tabletop_segmentation'


class RobotPushVisionDetector(object):
    ''' Base class for robot push vision detector '''

    def __init__(self):
        # self.active_sub = ActiveSubscriber()
        # only allow object to occupy at most half of image plane; the maximum dimension of an object is 30cm
        self.meters_per_pixel = 0.3 / (min(H, W) / 2.5)
        self.tl = tf.TransformListener()

        # tabletop segmentation setting
        self.world_frame = None  # to be overrided

        # services
        rospy.wait_for_service(SEG_SERVICE_NAME)
        self.seg_client = rospy.ServiceProxy(
            SEG_SERVICE_NAME, TabletopSegmentation)
        print('segmentation service is ready')

        # setup image fetch client
        self.mask_t = None
        self.bridge = CvBridge()

    def transform_img(self, img, w, x, y):

        # perform rotation and translation
        rotated = imutils.rotate(img, w)
        translated = imutils.translate(rotated, x, y)

        return translated

    ''' Input Creation '''

    def generate_input_image(self, points):
        """
        generate Push-Net input image given a set of points representing the segmented object
        """
        print("RobotPushVisionDetector/generate_input_image")

        # find positional center
        (xc, yc) = self.get_point_center(points)

        # map to an image
        img_mapped = self.point_to_image(points, xc, yc)

        # find contour and fill up the region within contour
        img_filled = self.fill_contour(img_mapped)

        return img_filled, xc, yc

    def generate_next_image(self, points, xc, yc):
        """
        generate Push-Net next image given a set of points representing the segmented object and its previsou image's center
        """

        # map to an image
        img_mapped = self.point_to_image(points, xc, yc)

        # find contour and fill up the region within contour
        img_filled = self.fill_contour(img_mapped)

        return img_filled

    def get_point_center(self, points):
        x_avg = 0
        y_avg = 0
        num = len(points)
        for p in points:
            x_avg += p.x
            y_avg += p.y
        return x_avg / num, y_avg / num

    def point_to_image(self, points, xc, yc):
        ''' convert points of real scale into 2d image mask of size 128 x 106
            the object will be centered in the image plane'''

        # assume image x axis is aligned with real negative y
        # assume image y axis is aligned with real negative x
        mask = np.zeros((H, W))
        center_x = W / 2
        center_y = H / 2
        for p in points:
            # real scale x, y position
            xr = p.x
            yr = p.y
            delta_xr = xr - xc
            delta_yr = yr - yc
            delta_yp = - int(delta_xr / self.meters_per_pixel)
            delta_xp = - int(delta_yr / self.meters_per_pixel)

            mask[center_y + delta_yp, center_x + delta_xp] = 255

        return mask.astype(np.uint8)

    def fill_contour(self, img):
        # dialate img
        dilate_size = 7  # adjustable
        erode_size = 5
        kernel_dilate = np.ones((dilate_size, dilate_size), np.uint8)
        kernel_erode = np.ones((erode_size, erode_size), np.uint8)
        # img_close = cv2.morphologyEx(img.copy(), cv2.MORPH_CLOSE, kernel)
        img_dilate = cv2.morphologyEx(
            img.copy(), cv2.MORPH_DILATE, kernel_dilate)
        img_erode = cv2.morphologyEx(
            img_dilate.copy(), cv2.MORPH_ERODE, kernel_erode)
        contours, _ = cv2.findContours(
            img_erode.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[-2:]

        imgs = np.zeros((H, W, 3))
        cv2.drawContours(imgs, contours, -1, (255, 255, 255), -1)
        return imgs.copy()[:, :, 0]

    ''' Segmentation '''

    def segment(self):
        '''
            perform segmentation
        '''
        print("RobotPushVisionDetector/segment: dummy!!!")
        return

    def project_3D2D(self, points):
        # RGB camera parameters
        fx = 541.787
        fy = 538.148
        x0 = 321.601
        y0 = 232.013

        pixels = []
        P = PointStamped()
        P.header.frame_id = 'left_out'
        P_news = []
        for p in points:
            P.point.x = p.x
            P.point.y = p.y
            P.point.z = p.z

            P_new = self.tl.transformPoint('/head_camera_rgb_optical_frame', P)

            # get pixel coordinates
            px = x0 + P_new.point.x / P_new.point.z * fx
            py = y0 + P_new.point.y / P_new.point.z * fy
            pixels.append([int(px), int(py)])

        return pixels

    def get_current_rgb(self):
        '''retrieve current rgb image from fetch onboard server'''
        print("RobotPushVisionDetector/get_current_rgb: dummy!!!")
        return None

    def get_goal_mask(self, points, gw, gx, gy, mode='convex', type='wxy'):
        '''prepare the goal mask'''
        pixels = self.project_3D2D(points)
        h, w, d = 480, 640, 3
        if type == 'wxy' or type == 'w':
            hull = None
            if mode == 'convex':
                hull = cv2.convexHull(np.array(pixels))
            else:
                binary_mask = np.zeros((h, w), dtype=np.uint8)
                for p in pixels:
                    binary_mask[p[1], p[0]] = 255

                dilate_size = 18  # adjustable
                erode_size = 2
                kernel_dilate = np.ones((dilate_size, dilate_size), np.uint8)
                kernel_erode = np.ones((erode_size, erode_size), np.uint8)
                img_dilate = cv2.morphologyEx(
                    binary_mask.copy(), cv2.MORPH_DILATE, kernel_dilate)
                img_erode = cv2.morphologyEx(
                    img_dilate.copy(), cv2.MORPH_ERODE, kernel_erode)
                cnts, _ = cv2.findContours(
                    img_erode.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                hull = cv2.approxPolyDP(np.array(cnts[0]), 0.01, True)

            mask = np.zeros((h, w, d))
            cv2.drawContours(mask, [hull], 0, (255, 255, 255), -1)
            self.mask_t = generate_goal_img_real(mask.copy(), gw, gx, gy)
        else:  # xy goal
            x_avg = 0
            y_avg = 0
            num = len(pixels)
            for p in pixels:
                x_avg += p[0]
                y_avg += p[1]

            x_avg = 1.0 * x_avg / num
            y_avg = 1.0 * y_avg / num
            x_goal = int(x_avg + gx)
            y_goal = int(y_avg + gy)
            self.mask_t = np.zeros((h, w, d))
            cv2.circle(self.mask_t, (x_goal, y_goal), 10, (255, 0, 0), -1)

    def add_goal_to_img(self, img):
        '''overlay goal mask onto the current rgb image'''
        overlay = img.copy()
        y, x = np.where(self.mask_t[:, :, 0] > 1)
        overlay[y, x] = np.array([255, 0, 0])
        cv2.addWeighted(img, 0.5, overlay, 0.5, 0, overlay)
        return overlay

    def get_table_height(self):
        print("RobotPushVisionDetector/get_table_height: dummy")
        return 0

    def set_table_height(self, table_height):
        print("RobotPushVisionDetector/set_table_height: dummy")
        return True

    def get_dz(self):
        print("RobotPushVisionDetector/get_dz: dummy")
        return 0
