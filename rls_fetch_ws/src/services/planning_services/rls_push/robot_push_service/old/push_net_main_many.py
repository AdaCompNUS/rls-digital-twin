#!/usr/bin/env python
'''
Main program to push objects with real robot
'''

__author__ = 'Li Juekun'
__date__ = '2018/01/12'

## pytorch mods ##
from model_tc import *
import torch.nn as nn
import torch
from torch.autograd import Variable
from rls_push_msgs.srv import *
from std_msgs.msg import Bool, Float64, String
from robot_config import *
import fetch_api
import config as args
from push_vision.find_marker import MarkerFinder
from push_vision.process_img import *
from push_vision.push_vision_seg import Sensor
import time
from transformations import *
from geometry_msgs.msg import PoseStamped, PointStamped
import tf
from cv_bridge import CvBridge
import argparse
import math
import json
from sensor_msgs.msg import Image, PointCloud2, JointState
import rospy
import os
import numpy as np

import cv2

# ROS modules


# Perception


W = 128.0  # !!!! Important to make it float to prevent integer division becomes zeros
H = 106.0
MODE = 'wxy'
METHOD = 'simcom'
pose_thresh = 0.04
fetch_end_eff = 'gripper_link'
gripper_to_wrist = 0.16645
fetch_gripper_height = 0.81  # for lower box eg. bowl, sugar box
# fetch_gripper_height = 0.85 ##for higher box eg. stick/straw
z_diff = 0.10
gripper_to_leftout_y = -0.09
view_pixel = []
MAX_PT = 150


def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


'''deep neural network predictor'''


class Predictor:
    def __init__(self):
        self.bs = args.batch_size
        model_path = 'model'
        best_model_name = 'model_best_' + args.arch[METHOD] + '.pth.tar'
        self.model_path = os.path.join(model_path, best_model_name)
        self.model = self.build_model()
        self.load_model()

    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_path)['state_dict'])
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()

    def build_model(self):
        if METHOD == 'simcom':
            return COM_net_sim(self.bs)
        elif METHOD == 'sim':
            return COM_net_sim_only(self.bs)
        elif METHOD == 'nomem':
            return COM_net_nomem(self.bs)

    def reset_model(self):
        self.model.hidden = self.model.init_hidden()

    def update(self, start, end, img_curr, img_goal):
        bs = self.bs
        A1 = []
        I1 = []
        Ig = []
        for i in range(bs):
            a1 = [[start[0]/W, start[1]/H, end[0]/W, end[1]/H]]
            i1 = [img_curr]
            ig = [img_goal]
            A1.append(a1)
            I1.append(i1)
            Ig.append(ig)

        A1 = torch.from_numpy(np.array(A1)).float()
        I1 = torch.from_numpy(np.array(I1)).float().div(255)
        Ig = torch.from_numpy(np.array(Ig)).float().div(255)

        A1 = to_var(A1)
        I1 = to_var(I1)
        Ig = to_var(Ig)

        if METHOD == 'simcom':
            sim_out, com_out = self.model(
                A1, I1, A1, Ig, [1 for i in range(bs)], bs)
        elif METHOD == 'sim':
            sim_out = self.model(A1, I1, A1, Ig, [1 for i in range(bs)], bs)
        elif METHOD == 'nomem':
            sim_out = self.model(A1, I1, A1, Ig, [1 for i in range(bs)], bs)

    def select_action(self, img_curr, img_goal, actions):
        bs = self.bs
        A1 = []
        I1 = []
        Ig = []

        for i in range(bs):
            a1 = [[actions[4*i]/W, actions[4*i+1]/H,
                   actions[4*i+2]/W, actions[4*i+3]/H]]
            i1 = [img_curr]
            ig = [img_goal]
            A1.append(a1)
            I1.append(i1)
            Ig.append(ig)

        A1 = torch.from_numpy(np.array(A1)).float()
        I1 = torch.from_numpy(np.array(I1)).float().div(255)
        Ig = torch.from_numpy(np.array(Ig)).float().div(255)

        A1 = to_var(A1)
        I1 = to_var(I1)
        Ig = to_var(Ig)

        sim_out = None
        com_out = None
        if METHOD == 'simcom':
            sim_out, com_out = self.model(
                A1, I1, A1, Ig, [1 for j in range(bs)], bs)
        elif METHOD == 'sim':
            sim_out = self.model(A1, I1, A1, Ig, [1 for j in range(bs)], bs)
        elif METHOD == 'nomem':
            sim_out = self.model(A1, I1, A1, Ig, [1 for j in range(bs)], bs)

        sim_np = sim_out.data.cpu().data.numpy()

        if MODE == 'wxy':
            sim_sum = np.sum(sim_np, 1)  # measure (w ,x, y)
        elif MODE == 'xy':
            sim_sum = np.sum(sim_np[:, 1:], 1)  # measure (x, y)
        else:
            sim_sum = np.sum(sim_np, 1)  # measure (w ,x, y)
            # sim_sum = sim_np[:, 0] # measure (w)
        action_value = []
        for ii in range(len(sim_sum)):
            s = [actions[4 * ii], actions[4 * ii + 1]]
            e = [actions[4 * ii + 2], actions[4 * ii + 3]]
            action_value.append([[s, e], sim_sum[ii]])

        return action_value


''' Push Controller '''


class PushController:
    def __init__(self):
        # self.sensor = Sensor()
        # self.makerfinder = MarkerFinder()

        self.p1_pub = rospy.Publisher('start_pose', PoseStamped, queue_size=10)
        self.p2_pub = rospy.Publisher('end_pose', PoseStamped, queue_size=10)
        self.goal_pub = rospy.Publisher(
            'goal_pose', PointStamped, queue_size=10)
        self.bridge = CvBridge()
        self.img_action_pub = rospy.Publisher('img_act', Image, queue_size=10)

        # update tf and static tf clients
        self.update_tf_client = rospy.ServiceProxy('update_tf', UpdateTF)
        self.update_static_client = rospy.ServiceProxy(
            'update_static', UpdateStatic)
        print 'ok'

        self.angle_range = [30, 60]
        self.x_range = [-0.15, 0.2]
        self.y_range = [-0.05, -0.03]
        self.step_left = 3
        # goal region
        # height of wrist roll link above the table plane
        self.z_offset = 0.03 + gripper_to_wrist
        self.pre_height = 0.1  # prepush height
        self.num_exp = 10
        self.max_step = 10
        self.left = True
        self.dx = 0
        self.dy = 0
        self.dw = 0

        self.f = Float64(0)
        self.robot_frame = 'base_link'
        self.pred = Predictor()

        self.arm = fetch_api.Arm()
        self.torso = fetch_api.Torso()
        self.gripper = fetch_api.Gripper()
        self.head = fetch_api.Head()

        self.joint_angle = []
        self.joint_sub = rospy.Subscriber(
            '/joint_states', JointState, self.joint_callback)
        self.push_srv = rospy.Service(
            'fetch_push', FetchPush, self.push_callback)
        self.last_value = -0.005
        print 'Server is UP!!!'
        # self.initialize()

    def initialize(self):

        self.torso.set_height(fetch_api.Torso.MAX_HEIGHT)
        self.arm.move_to_joint_goal(
            HIDE_LEFT_JOINTS, replan=True, execution_timeout=20.0)

    def joint_callback(self, data):
        if len(data.position) > 8:
            self.joint_angle = list(data.position[6:])

    def push_callback(self, req):
        # Find marker first
        if req.request < 0:
            self.makerfinder.find_marker()
            print 'marker localized !!!'
        else:
            # self.makerfinder.find_marker()
            # print 'marker localized !!!'
            self.gripper.close(50)
            self.run(req)

        return FetchPushResponse(Bool())

    def _test(self):
        # Find marker first
        self.makerfinder.find_marker()
        print 'marker localized !!!'
        self.gripper.open()
        rospy.sleep(1)
        self.gripper.close(50)
        self.run()
        # self.grasp_from_side('left')

    def get_points(self):
        req = UpdateStaticRequest()
        # req.dz.data = -0.005
        req.dz.data = self.last_value
        self.update_static_client(req)
        # segmentation
        points = None
        for i in range(3):
            points = self.sensor.segment().points

        while not rospy.is_shutdown():
            # make sure table is not incorrectly segmented as object
            for i in range(5):
                if (len(points) > MAX_PT or len(points) < 20):
                    points = self.sensor.segment().points
                else:
                    break
            if len(points) > MAX_PT:
                print '********************** move table UP'
                req.dz.data += 0.005
            elif len(points) < 20:
                print '********************** move table DOWN'
                req.dz.data -= 0.005
            else:
                break

            print 'table should be move by: ', req.dz.data
            self.last_value = req.dz.data
            self.update_static_client(req)

        return points

    def get_current_mask(self, points):

        # transformation
        mask, xc, yc = self.sensor.generate_input_image(points)
        # mask is already centered
        return mask, xc, yc

    def get_target_mask(self, mask, dw, dx, dy):
        return self.sensor.transform_img(mask, dw, dx, dy)

    def sample_action_new(self, img, num_actions):
        s = 0.9
        safe_margin = 1.4
        out_margin = 2.0

        img_inner = cv2.resize(img.copy(), (0, 0), fx=s,
                               fy=s, interpolation=cv2.INTER_AREA)
        h, w = img_inner.shape
        img_end = np.zeros((int(H), int(W)))
        img_end[(int(H)-h)/2:(int(H)+h)/2, (int(W)-w) /
                2:(int(W)+w)/2] = img_inner.copy()
        (inside_y, inside_x) = np.where(img_end.copy() > 0)

        # sample start push point outside a safe margin of object
        img_outer1 = cv2.resize(
            img.copy(), (0, 0), fx=safe_margin, fy=safe_margin, interpolation=cv2.INTER_CUBIC)
        h, w = img_outer1.shape
        img_start_safe = np.zeros((int(H), int(W)))
        img_start_safe = img_outer1.copy()[(
            h-int(H))/2:(h+int(H))/2, (w-int(W))/2:(w+int(W))/2]

        img_outer2 = cv2.resize(
            img.copy(), (0, 0), fx=out_margin, fy=out_margin, interpolation=cv2.INTER_CUBIC)
        h, w = img_outer2.shape
        img_start_out = np.zeros((int(H), int(W)))
        img_start_out = img_outer2.copy()[(
            h-int(H))/2:(h+int(H))/2, (w-int(W))/2:(w+int(W))/2]

        img_start = img_start_out.copy() - img_start_safe.copy()
        (outside_y, outside_x) = np.where(img_start.copy() > 100)

        # cv2.imshow('ori', img)
        # cv2.waitKey(0)
        # cv2.imshow('small', img_end)
        # cv2.waitKey(0)
        # cv2.imshow('safe', img_start_safe)
        # cv2.waitKey(0)
        # cv2.imshow('out', img_start_out)
        # cv2.waitKey(0)
        # cv2.imshow('start', img_start)
        # cv2.waitKey(0)

        num_inside = len(inside_x)
        num_outside = len(outside_x)

        actions = []
        for i in range(num_actions):
            start_x = 0
            start_y = 0
            end_x = 0
            end_y = 0
            while True:
                # sample an inside point
                inside_idx = np.random.choice(num_inside)
                # sample an outside point
                outside_idx = np.random.choice(num_outside)
                end_x = int(inside_x[inside_idx])
                end_y = int(inside_y[inside_idx])
                start_x = int(outside_x[outside_idx])
                start_y = int(outside_y[outside_idx])

                if start_x < 0 or start_x >= W or start_y < 0 or start_y >= H:
                    print 'out of bound'
                    continue
                if img[start_y, start_x] == 0:
                    break
                else:
                    # print img[start_y, start_x]
                    continue

            actions.append(start_x)
            actions.append(start_y)
            actions.append(end_x)
            actions.append(end_y)
        return actions

    def is_success(self, tran, w, mode):
        if mode == 'wxy':
            return (abs(tran[0]) <= pose_thresh) and (abs(tran[1]) <= pose_thresh) and (abs(w) <= 10)
            # return (abs(tran[0]) * self.sensor.meters_per_pixel <= pose_thresh) and (abs(tran[1]) * self.sensor.meters_per_pixel <= pose_thresh) and (abs(w) <= 10)
        elif mode == 'xy':
            return (abs(tran[0]) <= pose_thresh) and (abs(tran[1]) <= pose_thresh)
        elif mode == 'w':
            return (abs(w) <= 10)

    def update_tf(self):
        self.update_tf_client(Bool())

    def run(self, req=None):
        direction = req.request
        distance = req.distance

        global MODE
        self.curr_step = 0
        self.step_left = 0
        # Step 1: get object mask
        bs = args.batch_size

        self.update_tf()

        points = self.get_points()
        curr_mask, xc, yc = self.get_current_mask(points)
        gw = 0
        gx = 0
        gy = 0
        ###################
        ''' Define Goal '''
        ###################
        if direction == 0:
            ''' Push to Edge'''
            MODE = 'wxy'
            xm = 0.28  # cannot to far away from the robot, otherwise the arm will collide with table
            ym = -0.13 - 0.02  # edge y position NOTE: -0.02 make sure COM is away from table edge for2cm
            curr_mask = curr_mask.astype(np.uint8)
            gw = get_goal_w(curr_mask.copy())
            gx = xm - xc
            gy = ym - yc
        elif direction == 1:
            ''' Push to Left '''
            MODE = 'xy'
            gw = 0
            gx = 0
            gy = abs(distance)
            xm = gx + xc
            ym = gy + yc
        elif direction == 2:
            ''' Push to Right '''
            MODE = 'xy'
            gw = 0
            gx = 0
            gy = - abs(distance)
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
            self.arm.move_to_joint_goal(
                HIDE_LEFT_JOINTS, replan=True, execution_timeout=20.0)
        else:
            self.arm.move_to_joint_goal(
                HIDE_RIGHT_JOINTS, replan=True, execution_timeout=20.0)

        rospy.sleep(1)
        goal_point = PointStamped()
        goal_point.header.frame_id = self.sensor.world_frame
        goal_point.point.x = xm
        goal_point.point.y = ym
        goal_point.point.z = 0.05

        for i in range(100):
            self.goal_pub.publish(goal_point)

        # Step 3: generate goal mask
        goal_mask = self.get_target_mask(
            curr_mask, gw, int(-gy / self.sensor.meters_per_pixel), int(-gx / self.sensor.meters_per_pixel))

        # Step 4: loop over until goal is reached
        while True and not rospy.is_shutdown():
            # Step 5: generate next target mask
            # in case camera moves relative to the marker
            self.update_tf()
            points = self.get_points()

            img_in_curr, xc, yc = self.get_current_mask(points)
            img_in_curr = img_in_curr.astype(np.uint8)

            _, img_in_curr = cv2.threshold(
                img_in_curr.copy(), 30, 255, cv2.THRESH_BINARY)

            # update error
            error_x = 0
            error_y = 0
            error_w = 0

            if direction == 0:
                error_x = xm - xc
                error_y = ym - yc
                error_w = get_goal_w(img_in_curr)
                if self.is_graspable(points):
                    print 'Success :)'
                    break
            else:
                error_x = 0  # don't care about x position
                error_y = ym - yc
                error_w = 0

                if self.is_success([error_x, error_y], error_w, MODE):
                    print 'Success :)'
                    break

            print 'error is '
            print error_x, error_y, error_w

            if abs(error_y) < 0.1:
                self.dx = int(-error_y / self.sensor.meters_per_pixel)
            else:
                self.dx = -error_y/abs(error_y) * 0.1 / \
                    self.sensor.meters_per_pixel

            if abs(error_x) < 0.1:
                self.dy = int(-error_x / self.sensor.meters_per_pixel)
            else:
                self.dy = -error_x/abs(error_x) * 0.1 / \
                    self.sensor.meters_per_pixel

            if abs(error_w) < 30:  # degree
                self.dw = error_w
            else:
                self.dw = error_w / abs(error_w) * 30
            # don't neet to be perfectly aligned
            if abs(error_w) < 10:
                self.dw = 0
                MODE = 'xy'
            else:
                MODE = 'wxy'

            if MODE == 'xy':
                self.dw = 0
            elif MODE == 'w':
                self.dx = 0
                self.dy = 0

            img_in_next = self.get_target_mask(
                img_in_curr, self.dw, self.dx, self.dy)

            # Step 6: sample actions
            actions = self.sample_action_new(img_in_curr.copy(), 1000)
            print 'action sampled'

            # Step 7: select actions
            num_action = len(actions) / 4
            num_action_batch = num_action / bs
            min_sim_score = 1000

            best_start = None
            best_end = None
            best_sim = None
            best_com = None

            action_batch = []
            hidden = self.pred.model.hidden  # keep hidden state the same for all action batches

            action_value_pairs = []

            for i in range(num_action_batch):
                self.pred.model.hidden = hidden
                action = actions[4*i*bs: 4*(i+1)*bs]
                action_value = self.pred.select_action(
                    img_in_curr, img_in_next, action)
                action_value_pairs = action_value_pairs + action_value

            # sort action based on sim score
            action_value_pairs.sort(key=lambda x: x[1])

            # execute action
            best_start, best_end = self.execute_push(
                action_value_pairs, xc, yc, img_in_curr.copy())
            if best_start == None or best_end == None:
                print 'sample action again'
            else:
                self.pred.update(best_start, best_end,
                                 img_in_curr, img_in_next)

        if direction == 0:
            self.grasp_from_side('left')
        # self.arm.move_to_joint_goal(HIDE_LEFT_JOINTS, replan=True)

    def is_graspable(self, points):
        y_avg = 0
        count = 0
        for p in points:
            if p.y > -0.13:
                y_avg += p.y
                count += 1
        if count == 0:
            print 'no part is outside edge'
            return False
        y_avg = y_avg / count

        distance_from_table = y_avg + 0.13
        print 'distance from table is ', distance_from_table
        # if distance_from_table > 0.025:
        if distance_from_table > 0.021:
            return True
        else:
            return False

    def grasp_from_side(self, side):
        if side == 'left':
            self.arm.move_to_joint_goal(HIDE_LEFT_JOINTS, replan=True)
            rospy.sleep(1)
            self.arm.move_to_joint_goal(LEFT_GRASP_JOINTS, replan=True)
            rospy.sleep(1)
        self.gripper.open()

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
        center.header.frame_id = self.sensor.world_frame
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
                time = self.sensor.tl.getLatestCommonTime(
                    self.robot_frame, self.sensor.world_frame)
                center.header.stamp = time
                center_transform = self.sensor.tl.transformPoint(
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
                time = self.sensor.tl.getLatestCommonTime(
                    self.robot_frame, fetch_end_eff)
                (trans, qat) = self.sensor.tl.lookupTransform(
                    self.robot_frame, fetch_end_eff, time)
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
        self.torso.move_by(z_diff)
        rospy.sleep(1)
        # raw_input('wait')
        self.arm.move_in_cartesian(dx=x_diff, dy=y_diff)
        # self.arm.move_by_xy(dx=x_diff, dy=y_diff)
        # rospy.sleep(1)
        # raw_input('wait')
        self.gripper.close(50)
        # raw_input('wait')
        self.torso.set_height(fetch_api.Torso.MAX_HEIGHT)

        # rotate the wrist forward 0.8279026104675293 (6th joint)
        self.joint_angle[5] = 0.8279026104675293
        HAND_OVER_JOINTS = [(joint_names[i], self.joint_angle[i])
                            for i in range(7)]
        self.arm.move_to_joint_goal(HAND_OVER_JOINTS, replan=True)

        self.arm.move_in_cartesian(dz=0.2)
        self.arm.move_in_cartesian(dx=0.10)
        # self.arm.move_by_z(dz=0.2)
        # rospy.sleep(1)
        # if side == 'left':
        #    self.arm.move_to_joint_goal(HOLD_LEFT_JOINTS, replan=True)
        # rospy.sleep(1)

    def draw_action(self, img, start, end, offset=[0, 0], last=False):
        (yy, xx) = np.where(img > 0)
        img_3d = np.zeros((int(H), int(W), 3))
        img_3d[yy, xx] = np.array([255, 255, 255])
        # print img_3d.shape
        sx = int(start[0] + offset[1])
        sy = int(start[1] + offset[0])
        ex = int(end[0] + offset[1])
        ey = int(end[1] + offset[0])
        # print sx, sy, ex, ey
        cv2.line(img_3d, (sx, sy), (ex, ey), (0, 0, 255), 3)
        img_3d = img_3d.astype(np.uint8)

        # publish image
        ros_img = self.bridge.cv2_to_imgmsg(img_3d, encoding='bgr8')
        for i in range(100):
            self.img_action_pub.publish(ros_img)

        # cv2.imshow('action', img_3d)
        # if last:
        #    cv2.waitKey(0)
        # else:
        #    cv2.waitKey(10)

    def pixel_to_pose(self, best_start, best_end, xc, yc):
        x0 = - (best_start[1] - H / 2.0) * self.sensor.meters_per_pixel + xc
        y0 = - (best_start[0] - W / 2.0) * self.sensor.meters_per_pixel + yc
        x1 = - (best_end[1] - H / 2.0) * self.sensor.meters_per_pixel + xc
        y1 = - (best_end[0] - W / 2.0) * self.sensor.meters_per_pixel + yc

        start_pose = PoseStamped()
        end_pose = PoseStamped()
        # get orientation aroud z axis
        angle_z = math.atan2(y1 - y0, x1 - x0)  # in rads

        q = quaternion_from_euler(
            0, math.pi / 2, angle_z)  # need to double check

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

        # transform to robot frame
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

            start_pose_robot = self.sensor.tl.transformPose(
                self.robot_frame, start_pose)
            end_pose_robot = self.sensor.tl.transformPose(
                self.robot_frame, end_pose)

            start_pose_robot.pose.position.z += self.pre_height
            dx = end_pose_robot.pose.position.x - start_pose_robot.pose.position.x
            dy = end_pose_robot.pose.position.y - start_pose_robot.pose.position.y
            # print 'action in meters: dx, dy'
            # print dx, dy

            # while not rospy.is_shutdown():
            for i in range(100):
                # while not rospy.is_shutdown():
                self.p1_pub.publish(start_pose_robot)
                self.p2_pub.publish(end_pose_robot)

            return start_pose_robot, dx, dy

    def tune_action(self, start, end, img):
        ''' fine tune action to see if it might cause overshooting '''
        xes = int(end[0] - start[0])
        yes = int(end[1] - start[1])

        angle = math.atan2(xes, yes)
        step_size = 5  # pixels
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

        action_inside = math.sqrt(
            (next_x - end[0]) * (next_x - end[0]) + (next_y - end[1]) * (next_y - end[1]))
        ideal_len = math.sqrt((self.dx * self.dx + self.dy * self.dy)) * 0.8
        print 'ideal length is: ', ideal_len

        if action_inside < ideal_len:
            print 'it is ok'
            # end[0] = int(end[0] * 0.7 + (next_x + math.sqrt(ideal_len) * math.sin(angle)) * 0.3)
            # end[1] = int(end[1] * 0.7 + (next_y + math.sqrt(ideal_len) * math.cos(angle)) * 0.3)

            return start, end
        else:
            # reduce action length to avoid overshooting
            print 'reduce action length'
            end[0] = int(next_x + ideal_len * math.sin(angle))
            end[1] = int(next_y + ideal_len * math.cos(angle))
            return start, end

    def execute_push(self, sorted_actions, xc, yc, img):
        pack = sorted_actions.pop(0)
        best_start = pack[0][0]
        best_end = pack[0][1]
        self.draw_action(img.copy(), best_start, best_end, last=True)

        # rospy.sleep(1)

        best_start, best_end = self.tune_action(best_start, best_end, img)
        self.draw_action(img.copy(), best_start, best_end, last=True)

        start_pose_robot, dx, dy = self.pixel_to_pose(
            best_start, best_end, xc, yc)
        result = self.arm.move_to_pose(
            start_pose_robot, replan=False, execution_timeout=20)
        # print result
        while not result == None and not rospy.is_shutdown():
            print 'ERROR: cannot plan a path'
            if len(sorted_actions) == 0:
                print 'all actions have been exhausted!'
                return None, None
            pack = sorted_actions.pop(0)
            best_start = pack[0][0]
            best_end = pack[0][1]
            best_start, best_end = self.tune_action(
                best_start, best_end, img)

            self.draw_action(img.copy(), best_start, best_end, last=True)
            start_pose_robot, dx, dy = self.pixel_to_pose(
                best_start, best_end, xc, yc)
            result = self.arm.move_to_pose(
                start_pose_robot, replan=False, execution_timeout=20)

            # check height to ensure the robot won't touch the table surface
            time = 0
            trans = None
            qat = None
            z_curr = 0
            iter = 15
            for i in range(iter):
                while not rospy.is_shutdown():
                    try:
                        time = self.sensor.tl.getLatestCommonTime(
                            self.robot_frame, fetch_end_eff)
                        (trans, qat) = self.sensor.tl.lookupTransform(
                            self.robot_frame, fetch_end_eff, time)
                        z_curr += trans[2]
                        break
                    except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                        print 'try again'
                        continue

            # take an average with the hope it will be more accurate
            z_curr = z_curr / iter
            diff = fetch_gripper_height - z_curr
            print diff
            if abs(diff) > 0.11:
                print 'cannot move too much!'
                if self.left:
                    self.arm.move_to_joint_goal(
                        HIDE_LEFT_JOINTS, replan=True, execution_timeout=20.0)
                else:
                    self.arm.move_to_joint_goal(
                        HIDE_RIGHT_JOINTS, replan=True, execution_timeout=20.0)
                return None, None, None

            self.arm.move_in_cartesian(dz=diff)
            self.arm.move_in_cartesian(dx=dx, dy=dy)
            self.arm.move_in_cartesian(dz=-diff)

            if self.left:
                self.arm.move_to_joint_goal(
                    HIDE_LEFT_JOINTS, replan=True, execution_timeout=20)
            else:
                self.arm.move_to_joint_goal(
                    HIDE_RIGHT_JOINTS, replan=True, execution_timeout=20)

            return best_start, best_end


if __name__ == '__main__':
    try:
        rospy.init_node('push_test', log_level=rospy.INFO)
        con = PushController()
        # con._test()
        # con.init_system()
        rospy.spin()
    except rospy.ROSInterruptException:
        print 'program interrupted before completion'
