'''
Main program to push objects with real robot
'''

__author__='Li Juekun'
__date__ = '2018/01/12'

## pytorch mods ##
import cv2

import numpy as np
import os
### ROS modules
import rospy
from sensor_msgs.msg import Image, PointCloud2
import json
import math
from dart_ros.srv import *
import argparse
from cv_bridge import CvBridge
import imutils
from images2gif import writeGif
import tf
from geometry_msgs.msg import PoseStamped, PointStamped
from transformations import *

import time

## Perception
from push_vision.push_vision_seg import Sensor
from push_vision.process_img import *
from push_vision.find_marker import MarkerFinder
import config as args
import fetch_api
from moveit_commander import MoveGroupCommander
from robot_config import *
from std_msgs.msg import Bool, Float64


W = 128.0 ##!!!! Important to make it float to prevent integer division becomes zeros
H = 106.0
MODE = 'xy'
#MODE = 'w'
#MODE = 'wxy'
pose_thresh = 0.04
fetch_end_eff = 'gripper_link'
gripper_to_wrist = 0.16645
fetch_gripper_height = 0.81 ##for lower box eg. bowl, sugar box
#fetch_gripper_height = 0.85 ##for higher box eg. stick/straw
z_diff = 0.10
gripper_to_leftout_y = -0.09
#LEFT = True


''' Push Controller '''
class PushController:
    def __init__(self):
        self.sensor = Sensor()
        self.makerfinder = MarkerFinder()
        self.arm = fetch_api.Arm()
        self.torso = fetch_api.Torso()
        self.head = fetch_api.Head()
        self.gripper = fetch_api.Gripper()

        self.p1_pub = rospy.Publisher('start_pose', PoseStamped, queue_size=10)
        self.p2_pub = rospy.Publisher('end_pose', PoseStamped, queue_size=10)
        self.goal_pub = rospy.Publisher('goal_pose', PointStamped, queue_size=10)
        self.bridge = CvBridge()
        self.img_action_pub = rospy.Publisher('img_act', Image, queue_size=10)

        self.update_tf_pub = rospy.Publisher('/update_tf', Bool, queue_size=10)
        self.update_static_pub = rospy.Publisher('/update_static', Float64, queue_size=10)

        self.angle_range = [30, 60]
        self.x_range = [-0.15, 0.2]
        self.y_range = [-0.05, -0.03]
        #self.step_left = 10
        self.step_left = 3
        ## goal region
        self.z_offset = 0.03 + gripper_to_wrist ## height of wrist roll link above the table plane
        self.pre_height = 0.1 ## prepush height
        self.num_exp = 10
        self.max_step = 15
        self.left = True
        self.dx = 0
        self.dy = 0
        self.dw = 0

        self.f = Float64(0)
        self.robot_frame = 'base_link'
        print 'Server is UP!!!'

    def _test(self):
        #self.torso.set_height(fetch_api.Torso.MAX_HEIGHT)
        #rospy.sleep(1)
        #self.arm.move_to_joint_goal(HIDE_LEFT_JOINTS, replan=True)
        #rospy.sleep(1)
        #return
        #self.arm.move_to_joint_goal(LEFT_GRASP_JOINTS, replan=True)
        #rospy.sleep(1)
        #self.arm.move_to_joint_goal(HIDE_LEFT_JOINTS, replan=True)
        #rospy.sleep(1)
        #return
        ###self.gripper.open()
        #self.gripper.close(50)
        ####self.head.look_at('plane_frame', -0.30, 0, 0.45)
        ####rospy.sleep(1)
        #self.arm.move_to_joint_goal(LEFT_GRASP_JOINTS, replan=True)
        #rospy.sleep(1)
        #while True and not rospy.is_shutdown():
        #    raw_input('start')
        #    points = self.sensor.segment().points
        #    print len(points)
        #    while (len(points) > 200 or len(points) < 20) and not rospy.is_shutdown():
        #        points = self.sensor.segment().points
        #    if self.is_graspable(points):
        #        self.grasp_from_side('left')

        #self.arm.move_to_joint_goal(HIDE_LEFT_JOINTS, replan=True)
        self.makerfinder.find_marker()
        print 'marker localized !!!'
        self.gripper.open()
        rospy.sleep(1)
        self.gripper.close(50)
        self.run()
        #self.grasp_from_side('left')

    def init_system(self):
        ## init robot
        self.torso.set_height(fetch_api.Torso.MAX_HEIGHT)
        self.arm.move_to_named_target(self.group, 'pre_push')
        rospy.sleep(1)

        raw_input('system is ready! put down an object please')
        self.obj_name = raw_input('object name: ')

        for i in range(100):
            self.exp_idx = i

            start_time = time.time()
            self.run()
            ## reset hidden state
            self.pred.reset_model()

            dec = raw_input('continue exp? y/n?')
            if dec == 'n':
                break

    def get_points(self):
        ## segmentation
        points = None
        for i in range(3):
            points = self.sensor.segment().points

        while not rospy.is_shutdown():
            ## make sure table is not incorrectly segmented as object
            for i in range(5):
                if (len(points) > 200 or len(points) < 20):
                    points = self.sensor.segment().points
                else:
                    break
            if len(points) > 200:
                print '********************** move table UP'
                self.f.data += 0.005
            elif len(points) < 20:
                print '********************** move table DOWN'
                self.f.data -= 0.005
            else:
                break

            #print 'table should be move by: ', self.f.data
            for i in range(10):
                self.update_static_pub.publish(self.f)

        return points

    def get_current_mask(self, points):

        ## transformation
        mask, xc, yc = self.sensor.generate_input_image(points)
        ## mask is already centered
        return mask, xc, yc

    def get_target_mask(self, mask, dw, dx, dy):
        return self.sensor.transform_img(mask, dw, dx, dy)

    def sample_action_xy(self, img, dx, dy):
        center = get_center(img.copy()) #(yc ,xc)
        angle = math.atan2(dy, dx)
        ddx = Dtran * pixel_metric_ratio * math.cos(angle)
        ddy = Dtran * pixel_metric_ratio * math.sin(angle)
        #print ddx, ddy

        best_start = [-1, -1]
        best_end = [-1, -1]

        end_x = center[1]
        end_y = center[0]


        while not rospy.is_shutdown():
            end_x -= ddx
            end_y -= ddy
            if img[int(end_y), int(end_x)] == 0:
                break

        best_start[0] = int(end_x - 3*ddx)
        best_start[1] = int(end_y - 3*ddy)
        best_end[0] = int(end_x + ddx)
        best_end[1] = int(end_y + ddy)

        return best_start, best_end

    def sample_action_w(self, img, dw):
        center = get_center(img.copy())
        (yidx, xidx) = np.where(img > 1)
        ymin = np.min(yidx)
        ymax = np.max(yidx)
        xmin = np.min(xidx)
        xmax = np.max(xidx)

        num_pt = len(xidx)

        horizontal = ((xmax-xmin) > (ymax-ymin))
        best_start = [-1, -1]
        best_end = [-1, -1]
        idx = 0
        action_len = (xmax-xmin+ymax-ymin)/2.0

        if dw > 0: ## ccw
            if horizontal: #push right part up
                while not rospy.is_shutdown():
                    idx = np.random.choice(num_pt)
                    if xidx[idx] > (center[1] * 0.2 + xmax * 0.8):
                        break
                best_end[0] = xidx[idx]
                best_end[1] = yidx[idx]

                while not rospy.is_shutdown():
                    l = np.random.uniform(0, action_len, 1)[0]
                    best_start[0] = best_end[0]
                    best_start[1] = best_end[1] + int(l)

                    if best_start[0] >= 512 or best_start[1] >=424:
                        continue
                    if best_start[0] < 0 or best_start[1] < 0:
                        continue


                    if img[best_start[1], best_start[0]] == 0:
                        break
            else: # push bottom part right
                while not rospy.is_shutdown():
                    idx = np.random.choice(num_pt)
                    if yidx[idx] > (center[0] * 0.2 + ymax * 0.8):
                        break
                best_end[0] = xidx[idx]
                best_end[1] = yidx[idx]

                while not rospy.is_shutdown():
                    l = np.random.uniform(0, action_len, 1)[0]
                    best_start[0] = best_end[0] - int(l)
                    best_start[1] = best_end[1]
                    if best_start[0] >= 512 or best_start[1] >=424:
                        continue
                    if best_start[0] < 0 or best_start[1] < 0:
                        continue

                    if img[best_start[1], best_start[0]] == 0:
                        break

        else: ## cw
            if horizontal: #push left part up
                while not rospy.is_shutdown():
                    idx = np.random.choice(num_pt)
                    if xidx[idx] < (center[1] * 0.2 + xmin * 0.8):
                        break
                best_end[0] = xidx[idx]
                best_end[1] = yidx[idx]
                while not rospy.is_shutdown():
                    l = np.random.uniform(0, action_len, 1)[0]
                    best_start[0] = best_end[0]
                    best_start[1] = best_end[1] + int(l)
                    if best_start[0] >= 512 or best_start[1] >=424:
                        continue
                    if best_start[0] < 0 or best_start[1] < 0:
                        continue

                    if img[best_start[1], best_start[0]] == 0:
                        break
            else: # push bottom part left
                while not rospy.is_shutdown():
                    idx = np.random.choice(num_pt)
                    if yidx[idx] > (center[0] * 0.2 + ymax * 0.8):
                        break
                best_end[0] = xidx[idx]
                best_end[1] = yidx[idx]
                while not rospy.is_shutdown():
                    l = np.random.uniform(0, action_len, 1)[0]
                    best_start[0] = best_end[0] + int(l)
                    best_start[1] = best_end[1]
                    if best_start[0] >= 512 or best_start[1] >=424:
                        continue
                    if best_start[0] < 0 or best_start[1] < 0:
                        continue

                    if img[best_start[1], best_start[0]] == 0:
                        break

        dxx = best_end[0] - best_start[0]
        dyy = best_end[1] - best_start[1]
        angle = math.atan2(dyy, dxx)
        next_x = best_start[0]
        next_y = best_start[1]
        while not rospy.is_shutdown():
            next_x += 1 * math.cos(angle)
            next_y += 1 * math.sin(angle)
            if img[int(next_y), int(next_x)] > 1:
                break
        R = math.sqrt((next_x - center[1]) * (next_x - center[1]) + (next_y - center[0]) * (next_y - center[0]))

        dist = R * 5 / 180 * np.pi
        best_end[0] = int(next_x + dist * math.cos(angle))
        best_end[1] = int(next_y + dist * math.sin(angle))


        return best_start, best_end


    def is_success(self, tran, w, mode):
        if mode == 'wxy':
            return (abs(tran[0]) * self.sensor.meters_per_pixel <= pose_thresh) and (abs(tran[1]) * self.sensor.meters_per_pixel <= pose_thresh) and (abs(w) <= 10)
        elif mode == 'xy':
            return (abs(tran[0]) <= pose_thresh) and (abs(tran[1]) <= pose_thresh)
        elif mode == 'w':
            return (abs(w) <= 10)

    def update_tf(self):
        for i in range(20):
            self.update_tf_pub.publish(Bool())

    def run(self):
        global MODE
        self.curr_step = 0
        self.step_left = 0
        # Step 1: get object mask
        bs = args.batch_size


        self.update_tf()

        points = self.get_points()
        curr_mask, xc, yc = self.get_current_mask(points)

        ###################
        ''' Define Goal '''
        ###################
        ''' Push by certain positional amount'''
        MODE = 'xy'
        gx = 0
        gy = -0.4
        gw = 0
        xm = gx + xc
        ym = gy + yc


        print '++++ Goal dw, dx, dy ++++++++'
        print gw, gx, gy
        print '++++++++++++'
        if gy > 0:
            self.left = False
        else:
            self.left = True
        if self.left:
            self.arm.move_to_joint_goal(HIDE_LEFT_JOINTS, replan=True)
        else:
            self.arm.move_to_joint_goal(HIDE_RIGHT_JOINTS, replan=True)
        rospy.sleep(1)
        goal_point = PointStamped()
        goal_point.header.frame_id = self.sensor.world_frame
        goal_point.point.x = xm
        goal_point.point.y = ym
        goal_point.point.z = 0.05

        for i in range(100):
            self.goal_pub.publish(goal_point)

        # Step 3: generate goal mask
        goal_mask = self.get_target_mask(curr_mask, gw, int(-gy / self.sensor.meters_per_pixel), int(-gx / self.sensor.meters_per_pixel))


        # Step 4: loop over until goal is reached
        while True and not rospy.is_shutdown():
           # Step 5: generate next target mask
           self.update_tf()
           points = self.get_points()

           img_in_curr, xc, yc = self.get_current_mask(points)
           img_in_curr = img_in_curr.astype(np.uint8)

           _, img_in_curr = cv2.threshold(img_in_curr.copy(), 30, 255, cv2.THRESH_BINARY)

           ## difference in metric scale
           error_x = xm - xc
           error_y = ym - yc
           error_w = 0
           if MODE == 'wxy' or MODE == 'w':
               error_w = get_goal_w(img_in_curr)
           print 'error is '
           print error_x, error_y, error_w

           if self.is_success([error_x, error_y], error_w, MODE):
               print 'Success :)'
               break
           if self.curr_step > self.max_step:
               print 'Failure :)'
               break

           if abs(error_y) < 0.1:
               self.dx = int(-error_y / self.sensor.meters_per_pixel)
           else:
               self.dx = -error_y/abs(error_y) * 0.1 / self.sensor.meters_per_pixel

           if abs(error_x) < 0.1:
               self.dy = int(-error_x / self.sensor.meters_per_pixel)
           else:
               self.dy = -error_x/abs(error_x) * 0.1 / self.sensor.meters_per_pixel

           if abs(error_w) < 30: ## degree
               self.dw = error_w
           else:
               self.dw = error_w / abs(error_w) * 30
           ## don't neet to be perfectly aligned
           if abs(error_w) < 10:
               self.dw = 0

           if MODE =='w':
               self.dx = 0
               self.dy = 0

           # Step 6: sample actions
           if abs(self.dw) < 10:
               best_start, best_end = self.sample_action_xy(img_in_curr.copy(), dx, dy)
           else:
               best_start, best_end = self.sample_action_w(img_in_curr.copy(), dw)


           best_start, best_end = self.execute_push(best_start, best_end, xc, yc, img_in_curr.copy())
           if best_start == None or best_end == None:
               print 'sample action again'


    def draw_action(self, img, start, end, offset=[0,0], last=False):
        (yy, xx) = np.where(img>0)
        img_3d = np.zeros((int(H), int(W), 3))
        img_3d[yy, xx] = np.array([255,255,255])
        #print img_3d.shape
        sx = int(start[0] + offset[1])
        sy = int(start[1] + offset[0])
        ex = int(end[0] + offset[1])
        ey = int(end[1] + offset[0])
        #print sx, sy, ex, ey
        cv2.line(img_3d, (sx, sy), (ex, ey), (0,0,255), 3)
        img_3d = img_3d.astype(np.uint8)

        ## publish image
        ros_img = self.bridge.cv2_to_imgmsg(img_3d, encoding='bgr8')
        for i in range(100):
            self.img_action_pub.publish(ros_img)

        #cv2.imshow('action', img_3d)
        #if last:
        #    cv2.waitKey(0)
        #else:
        #    cv2.waitKey(10)

    def pixel_to_pose(self, best_start, best_end, xc, yc):
           x0 = - (best_start[1] - H / 2.0) * self.sensor.meters_per_pixel + xc
           y0 = - (best_start[0] - W / 2.0) * self.sensor.meters_per_pixel + yc
           x1 = - (best_end[1] - H / 2.0) * self.sensor.meters_per_pixel + xc
           y1 = - (best_end[0] - W / 2.0) * self.sensor.meters_per_pixel + yc

           start_pose = PoseStamped()
           end_pose = PoseStamped()
           ## get orientation aroud z axis
           angle_z = math.atan2(y1 - y0, x1 - x0) ## in rads

           q = quaternion_from_euler(0, math.pi / 2, angle_z) ## need to double check

           start_pose.header.frame_id = self.sensor.world_frame
           start_pose.pose.position.x = x0
           start_pose.pose.position.y = y0
           start_pose.pose.position.z = self.z_offset
           start_pose.pose.orientation.x = q[0]
           start_pose.pose.orientation.y = q[1]
           start_pose.pose.orientation.z = q[2]
           start_pose.pose.orientation.w = q[3]

           end_pose.header.frame_id = self.sensor.world_frame
           end_pose.pose.position.x = x1
           end_pose.pose.position.y = y1
           end_pose.pose.position.z = self.z_offset
           end_pose.pose.orientation.x = q[0]
           end_pose.pose.orientation.y = q[1]
           end_pose.pose.orientation.z = q[2]
           end_pose.pose.orientation.w = q[3]

           ## transform to robot frame
           time = 0
           while not rospy.is_shutdown():
               try:
                   time = self.sensor.tl.getLatestCommonTime(self.robot_frame,
                           self.sensor.world_frame)
                   break
               except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                   continue
           start_pose.header.stamp = time
           end_pose.header.stamp = time

           start_pose_robot = self.sensor.tl.transformPose(self.robot_frame, start_pose)
           end_pose_robot = self.sensor.tl.transformPose(self.robot_frame, end_pose)

           start_pose_robot.pose.position.z += self.pre_height
           dx = end_pose_robot.pose.position.x - start_pose_robot.pose.position.x
           dy = end_pose_robot.pose.position.y - start_pose_robot.pose.position.y
           #print 'action in meters: dx, dy'
           #print dx, dy

           #while not rospy.is_shutdown():
           for i in range(100):
           #while not rospy.is_shutdown():
               self.p1_pub.publish(start_pose_robot)
               self.p2_pub.publish(end_pose_robot)

           return start_pose_robot, dx, dy

    def tune_action(self, start, end, img):
        ''' fine tune action to see if it might cause overshooting '''
        xes = int(end[0] - start[0])
        yes = int(end[1] - start[1])

        angle = math.atan2(xes, yes)
        step_size = 5 # pixels
        xs = step_size * math.sin(angle)
        ys = step_size * math.cos(angle)
        next_x = start[0]
        next_y = start[1]

        max_count = 20
        count = 0

        while True and not rospy.is_shutdown():
            next_x += xs
            next_y += ys
            next_x = min(next_x, W-1)
            next_x = max(next_x, 0)
            next_y = min(next_y, H-1)
            next_y = max(next_y, 0)

            if img[int(next_y), int(next_x)] == 255:
                break
            count += 1
            if count > max_count:
                print 'not able to find contact point'
                return start, end


        action_inside = math.sqrt((next_x - end[0]) * (next_x - end[0]) + (next_y - end[1]) * (next_y - end[1]))
        ideal_len = math.sqrt((self.dx * self.dx + self.dy * self.dy)) * 0.8
        print 'ideal length is: ', ideal_len

        if action_inside < ideal_len:
            print 'it is ok'
            #end[0] = int(end[0] * 0.7 + (next_x + math.sqrt(ideal_len) * math.sin(angle)) * 0.3)
            #end[1] = int(end[1] * 0.7 + (next_y + math.sqrt(ideal_len) * math.cos(angle)) * 0.3)

            return start, end
        else:
            ## reduce action length to avoid overshooting
            print 'reduce action length'
            end[0] = int(next_x + ideal_len * math.sin(angle))
            end[1] = int(next_y + ideal_len * math.cos(angle))
            return start, end

    def execute_push(self, best_start, best_end, xc, yc, img):
           self.draw_action(img.copy(), best_start, best_end, last=True)
           rospy.sleep(1)
           best_start, best_end = self.tune_action(best_start, best_end, img)
           self.draw_action(img.copy(), best_start, best_end, last=True)

           start_pose_robot, dx, dy = self.pixel_to_pose(best_start, best_end, xc, yc)
           result = self.arm.move_to_pose(start_pose_robot, replan=False, execution_timeout=10)
           print result
           if not result == None:
               print 'ERROR: cannot plan a path'
               return None, None
           rospy.sleep(1)

           ## check height to ensure the robot won't touch the table surface
           time = 0
           trans = None
           qat = None
           z_curr = 0
           iter = 15
           for i in range(iter):
               while not rospy.is_shutdown():
                   try:
                       time = self.sensor.tl.getLatestCommonTime(self.robot_frame, fetch_end_eff)
                       (trans, qat) = self.sensor.tl.lookupTransform(self.robot_frame, fetch_end_eff, time)
                       z_curr += trans[2]
                       break
                   except (tf.Exception,tf.LookupException,tf.ConnectivityException, tf.ExtrapolationException):
                       print 'try again'
                       continue
           # take an average with the hope it will be more accurate
           z_curr = z_curr / iter
           diff = fetch_gripper_height - z_curr
           self.arm.move_in_cartesian(dz=diff)
           self.arm.move_in_cartesian(dx=dx, dy=dy)
           self.arm.move_in_cartesian(dz=-2*diff)

           if self.left:
               self.arm.move_to_joint_goal(HIDE_LEFT_JOINTS, replan=True)
           else:
               self.arm.move_to_joint_goal(HIDE_RIGHT_JOINTS, replan=True)
           rospy.sleep(1)

           return best_start, best_end


if __name__=='__main__':
    try:
        rospy.init_node('push_test', log_level=rospy.INFO)
        con = PushController()
        con._test()
        #con.init_system()
        #rospy.spin()
    except rospy.ROSInterruptException:
        print 'program interrupted before completion'

