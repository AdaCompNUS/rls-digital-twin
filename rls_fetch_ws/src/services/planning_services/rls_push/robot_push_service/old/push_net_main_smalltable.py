'''
Main program to push objects with real robot
'''

__author__='Li Juekun'
__date__ = '2018/01/12'

## pytorch mods ##
import torch, cv2
import torch.nn as nn
from model_tc import *
from torch.autograd import Variable

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
import config as args
import fetch_api
from moveit_commander import MoveGroupCommander
from robot_config_smalltable import *


W = 128.0 ##!!!! Important to make it float to prevent integer division becomes zeros
H = 106.0
MODE = 'xy'
pose_thresh = 0.04
fetch_end_eff = 'gripper_link'
gripper_to_wrist = 0.16645
fetch_gripper_height = 0.54 ##TODO: to be measured

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


'''deep neural network predictor'''
class Predictor:
    def __init__(self):
        self.bs = args.batch_size
        self.model_path = args.best_model_path
        self.model = self.build_model()
        self.load_model()

    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_path)['state_dict'])
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()

    def build_model(self):
        return COM_net_sim(self.bs)

    def reset_model(self):
        self.model.hidden = self.model.init_hidden()

    def update(self, start, end, img_curr, img_goal):
        bs = args.batch_size
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

        sim_out, com_out = self.model(A1, I1, A1, Ig, [1 for i in range(bs)], bs)

    def select_action(self, img_curr, img_goal, actions):
        bs = args.batch_size
        A1 = []
        I1 = []
        Ig = []
        for i in range(bs):
            a1 = [[actions[4*i]/W, actions[4*i+1]/H, actions[4*i+2]/W, actions[4*i+3]/H]]
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

        sim_out, com_out = self.model(A1, I1, A1, Ig, [1 for i in range(bs)], bs)

        sim_np = sim_out.data.cpu().data.numpy()
        com_np = com_out.data.cpu().data.numpy()

        if MODE == 'wxy':
            sim_sum = np.sum(sim_np, 1) # measure (w ,x, y)
        elif MODE == 'xy':
            sim_sum = np.sum(sim_np[:,1:], 1) # measure (x, y)
        else:
            sim_sum = sim_np[:, 0] # measure (w)
        min_idx = np.argmin(sim_sum)
        best_sim_score = np.min(sim_sum)

        start = [actions[4*min_idx], actions[4*min_idx+1]]
        end = [actions[4*min_idx+2], actions[4*min_idx+3]]

        sim_measure = list(sim_np[min_idx,:])
        com_pred = [com_np[min_idx, 0] * W, com_np[min_idx, 1] * H]


        return start, end, sim_measure, com_pred, best_sim_score


''' Push Controller '''
class PushController:
    def __init__(self):
        self.pred = Predictor()
        self.sensor = Sensor()
        self.arm = fetch_api.Arm()
        self.torso = fetch_api.Torso()
        self.head = fetch_api.Head()

        self.p1_pub = rospy.Publisher('start_pose', PoseStamped, queue_size=10)
        self.p2_pub = rospy.Publisher('end_pose', PoseStamped, queue_size=10)
        self.goal_pub = rospy.Publisher('goal_pose', PointStamped, queue_size=10)

        self.angle_range = [30, 60]
        self.x_range = [-0.15, 0.2]
        self.y_range = [-0.05, -0.03]
        #self.step_left = 10
        self.step_left = 3
        ## goal region
        self.z_offset = 0.03 + gripper_to_wrist ## height of wrist roll link above the table plane
        self.pre_height = 0.08 ## prepush height
        self.num_exp = 10
        self.max_step = 15

        self.robot_frame = 'base_link'
        print 'Server is UP!!!'

    def _test(self):
        self.torso.set_height(fetch_api.Torso.MAX_HEIGHT)
        rospy.sleep(1)
        self.arm.move_to_joint_goal(HIDE_LEFT_JOINTS, replan=True)
        rospy.sleep(1)
        self.head.look_at('plane_frame', -0.18, 0, 0.18)
        rospy.sleep(1)

        self.run()

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

    def get_current_mask(self):
        ## segmentation
        points = self.sensor.segment().points
        ## transformation
        mask, xc, yc = self.sensor.generate_input_image(points)
        return mask, xc, yc

    def get_target_mask(self, mask, dw, dx, dy):
        return self.sensor.transform_img(mask, dw, dx, dy)


    def sample_action(self, img, num_actions, Dw, Dx, Dy):

        if MODE == 'w':
            s = 0.8
        else:
            s = 0.5
        #img_center, center_offset = center_img(img.copy())
        #h, w = img_center.shape
        #img_large = np.zeros((H, W))
        #img_large[(H-h)/2:(H+h)/2, (W-w)/2:(W+w)/2] = img_small.copy()
        #img_shift = transform_img(img_large.copy(), 0, center_offset[1], center_offset[0])


        #(inside_y, inside_x) = np.where(img_shift>0)

        ## get the center coordinate of mask
        (inside_y, inside_x) = np.where(img.copy()>0)
        mu_y = np.mean(inside_y)
        mu_x = np.mean(inside_x)

        ## find min area rectangle estimation
        img = img.astype(np.uint8)
        contours, _ = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        rect = cv2.minAreaRect(cnt)

        #print rect[1]

        ## find out longest and shortest dimension of the mask to deterine scale of an action
        num_inside = len(inside_x)
        max_pixel_len = max(rect[1]) / 1.
        min_pixel_len = min(rect[1]) / 1.

        #print '======='
        #print rect
        #print max_pixel_len, min_pixel_len

        if min_pixel_len < 1 or max_pixel_len < 10:
            min_pixel_len = 30
            max_pixel_len = 60
        actions = []
        for i in range(num_actions):
            ## sample an inside point
            inside_idx = np.random.choice(num_inside)
            #print inside_idx
            ## rotation
            if MODE == 'w':
                if Dw > 0:
                    ## select from the right
                    while not rospy.is_shutdown():
                        inside_idx = np.random.choice(num_inside)
                        if inside_x[inside_idx] < mu_x:
                            continue
                        else:
                            break
                else:
                    ## select from the left
                    while not rospy.is_shutdown():
                        inside_idx = np.random.choice(num_inside)
                        if inside_x[inside_idx] >= mu_x:
                            continue
                        else:
                            break

            ## sample an outside point
            target_x = 0
            target_y = 0
            end_x = 0
            end_y = 0
            while True:
                length = np.random.uniform(min_pixel_len, max_pixel_len) * 0.8
                ## for long object
                #length = min_pixel_len
                #print length
                #angle = np.random.uniform(0, 2 * np.pi)

                dir = 0
                if MODE == 'xy' or MODE == 'wxy':
                    #angle = np.random.uniform(-np.pi/4, np.pi/4)
                    angle = np.random.uniform(0, 2 * np.pi)
                    dir = math.atan2(Dx, Dy)
                    dx = length * np.sin(dir + np.pi + angle)
                    dy = length * np.cos(dir + np.pi + angle)
                elif MODE == 'w':
                    #angle = np.random.uniform(-np.pi/6, np.pi/6)
                    angle = np.random.uniform(0, 2 * np.pi)
                    dx = length * np.sin(angle)
                    dy = length * np.cos(angle)

                target_x = int(inside_x[inside_idx] + dx * 1.2)
                target_y = int(inside_y[inside_idx] + dy * 1.2)
                end_x = int(inside_x[inside_idx] - dx/6)
                end_y = int(inside_y[inside_idx] - dy/6)

                ## avoid action pointing to robot itself, it is hard to execute
                direction = math.atan2(target_x - end_x, target_y - end_y) / np.pi * 180
                #if direction > 80 or direction < -80:
                if MODE == 'w':
                    if max_pixel_len > 100:
                        if Dw > 0:
                            if direction > 60 or direction < 0:
                                continue
                        else:
                            if direction < -60 or direction > 0:
                                continue
                    else:
                        if direction > 100 or direction < -100:
                                continue

                if target_x < 0 or target_x >= W or target_y < 0 or target_y >= H:
                    #print 'out of bound'
                    continue
                if img[target_y, target_x] == 0:
                    break

            actions.append(target_x)
            actions.append(target_y)
            actions.append(end_x)
            actions.append(end_y)

        #print 'len of action', len(actions)
        return actions



    def is_success(self, tran, w, mode):
        if mode == 'wxy':
            return (abs(tran[0]) * self.sensor.meters_per_pixel <= pose_thresh) and (abs(tran[1]) * self.sensor.meters_per_pixel <= pose_thresh) and (abs(w) <= 10)
        elif mode == 'xy':
            return (abs(tran[0]) <= pose_thresh) and (abs(tran[1]) <= pose_thresh)
        elif mode == 'w':
            return (abs(w) <= 10)


    def run(self):
        self.curr_step = 0
        self.step_left = 0
        # Step 1: get object mask
        bs = args.batch_size

        # Step 2: random sample goal pose
        gw = np.random.uniform(self.angle_range[0], self.angle_range[1], 1)[0]
        gx = np.random.uniform(self.x_range[0], self.x_range[1], 1)[0]
        gy = np.random.uniform(self.y_range[0], self.y_range[1], 1)[0]

        if abs(gw) < 10:
            gw = gw / abs(gw) * 20
        if abs(gx) < 0.1:
            gx = gx / abs(gx) * 0.15
        gw = 0
        gx = 0
        gy = -0.35
        print '++++ Goal dw, dx, dy ++++++++'
        print gw, gx, gy
        print '++++++++++++'
        #gx = 0.2

        curr_mask, xc, yc = self.get_current_mask()
        ## get object goal pose
        xm = xc + gx
        ym = yc + gy
        goal_point = PointStamped()
        goal_point.header.frame_id = self.sensor.world_frame
        goal_point.point.x = xm
        goal_point.point.y = ym
        goal_point.point.z = 0.05

        for i in range(100):
            self.goal_pub.publish(goal_point)

        # Step 3: generate goal mask
        goal_mask = self.get_target_mask(curr_mask, gw, int(-gy / self.sensor.meters_per_pixel), int(-gx / self.sensor.meters_per_pixel))


        #cv2.imshow('init mask', curr_mask)
        #cv2.waitKey(0)
        #cv2.imshow('goal mask', goal_mask)
        #cv2.waitKey(0)


        # Step 4: loop over until goal is reached
        while True and not rospy.is_shutdown():
           # Step 5: generate next target mask
           img_in_curr, xc, yc = self.get_current_mask()
           error_x = xm - xc
           error_y = ym - yc
           diff_xy, diff_w = get_img_transform(img_in_curr.copy(), goal_mask.copy())


           #print 'diff xy'
           ## for symmetric object
           if abs(diff_w) > 90:
               if diff_w > 0:
                   diff_w -= 180
               else:
                   diff_w += 180
           #print diff_xy, diff_w

           if self.is_success([error_x, error_y], diff_w, MODE):
               print 'Success :)'
               break
           if self.curr_step > self.max_step:
               print 'Failure :)'
               break

           if self.step_left == 0:
               if abs(error_y) < 0.1:
                   dx = int(-error_y / self.sensor.meters_per_pixel)
               else:
                   dx = -error_y/abs(error_y) * 0.1 / self.sensor.meters_per_pixel

               if abs(error_x) < 0.1:
                   dy = int(-error_x / self.sensor.meters_per_pixel)
               else:
                   dy = -error_x/abs(error_x) * 0.1 / self.sensor.meters_per_pixel
               dw = diff_w

           else:
               dy = diff_xy[0] / self.step_left
               dx = diff_xy[1] / self.step_left
               dw = diff_w / self.step_left
               self.step_left -= 1

           if abs(diff_w) > 15:
               dw = dw / abs(dw) * 10



           if MODE == 'xy':
               dw = 0
           elif MODE =='w':
               dx = 0
               dy = 0
           print 'next goal in pixel: dw, dx, dy'
           print dw, dx, dy

           img_in_next = self.get_target_mask(img_in_curr, dw, dx, dy)

           #cv2.imshow('curr mask', img_in_curr)
           #cv2.waitKey(0)
           #cv2.imshow('next mask', img_in_next)
           #cv2.waitKey(0)

           # Step 6: sample actions
           actions = self.sample_action(img_in_curr.copy(), 1000, dw, dx, dy)

           # Step 7: select actions
           num_action = len(actions) / 4
           num_action_batch = num_action / bs
           min_sim_score = 1000

           best_start = None
           best_end = None
           best_sim = None
           best_com = None

           action_batch = []
           hidden = self.pred.model.hidden ## keep hidden state the same for all action batches

           for i in range(num_action_batch):
               self.pred.model.hidden = hidden
               action = actions[4*i*bs: 4*(i+1)*bs]

               start, end, sim, com, sim_score = self.pred.select_action(img_in_curr, img_in_next, action)
               if sim_score < min_sim_score:
                   min_sim_score = sim_score
                   best_start = start
                   best_end = end
                   best_sim = sim
                   best_com = com

            #print best_start, best_end, min_sim_score
           self.pred.update(best_start, best_end, img_in_curr, img_in_next)


           #for a in range(num_action):
           #    start = [actions[4*a], actions[4*a+1]]
           #    end = [actions[4*a+2], actions[4*a+3]]
           #    self.draw_action(img_in_curr.copy(), start, end)


           self.draw_action(img_in_curr.copy(), best_start, best_end, last=True)

           ## Step 8: execute the best action
           #ans = raw_input('execute? y/n')
           #if ans == 'n':
           #    continue
           xr_start = - (best_start[1] - H / 2.0) * self.sensor.meters_per_pixel + xc
           yr_start = - (best_start[0] - W / 2.0) * self.sensor.meters_per_pixel + yc
           xr_end = - (best_end[1] - H / 2.0) * self.sensor.meters_per_pixel + xc
           yr_end = - (best_end[0] - W / 2.0) * self.sensor.meters_per_pixel + yc

           #### execute action
           self.execute_push(xr_start, yr_start, xr_end, yr_end)

        self.arm.move_to_joint_goal(HOME_JOINTS, replan=True)
        rospy.sleep(1)
        self.torso.set_height(fetch_api.Torso.MIN_HEIGHT)
        rospy.sleep(2)
        self.arm.clean_exit()


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
        #self.save_img(img_3d, 'mask_'+str(self.curr_step))
        #img_3d = img_3d.astype(int)

        cv2.imshow('action', img_3d)
        #if last:
        #    cv2.waitKey(0)
        #else:
        #    cv2.waitKey(10)


    def execute_push(self, x0, y0, x1, y1):
           start_pose = PoseStamped()
           end_pose = PoseStamped()
           ## get orientation aroud z axis
           angle_z = math.atan2(y1 - y0, x1 - x0) ## in rads

           #q = quaternion_from_euler(math.pi, 0, -math.pi/2 - angle_z)
           #q = quaternion_from_euler(math.pi, 0, - math.pi / 2 * 3 - angle_z)
           #q = quaternion_from_euler(math.pi, 0, angle_z - (math.pi / 2 * 3)) ## thanks to Aseem
           #q = quaternion_from_euler(math.pi, 0, angle_z - math.pi) ## thanks to Aseem
           #q = quaternion_from_euler(math.pi, 0,  angle_z - (math.pi)) ## thanks to Aseem

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


           #while not rospy.is_shutdown():
           #    self.p1_pub.publish(start_pose)
           #    self.p2_pub.publish(end_pose)

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
           #start_pose_robot.pose.position.z += (-gripper_to_wrist)
           #end_pose_robot.pose.position.z += (-gripper_to_wrist)
           dx = end_pose_robot.pose.position.x - start_pose_robot.pose.position.x
           dy = end_pose_robot.pose.position.y - start_pose_robot.pose.position.y
           print 'action in meters: dx, dy'
           print dx, dy

           #while not rospy.is_shutdown():
           for i in range(100):
           #while not rospy.is_shutdown():
               self.p1_pub.publish(start_pose_robot)
               self.p2_pub.publish(end_pose_robot)
           #return

           ## go to pre-push pose
           #raw_input('move to the pose')
           self.arm.move_to_pose(start_pose_robot, replan=True, execution_timeout=10)
           #raw_input('go on')

           ## check height to ensure the robot won't touch the table surface
           time = 0
           trans = None
           qat = None
           while not rospy.is_shutdown():
               try:
                   time = self.sensor.tl.getLatestCommonTime(self.robot_frame, fetch_end_eff)
                   (trans, qat) = self.sensor.tl.lookupTransform(self.robot_frame, fetch_end_eff, time)
                   break
               except (tf.Exception,tf.LookupException,tf.ConnectivityException, tf.ExtrapolationException):
                   print 'try again'
                   continue
           z_curr = trans[2]
           diff = fetch_gripper_height - z_curr
           print diff
           #raw_input('go on')
           self.arm.move_by_z(dz=diff)
           #raw_input('go on')
           self.arm.move_by_xy(dx=dx, dy=dy)
           #raw_input('go on')
           self.arm.move_by_z(dz=-3*diff)
           self.arm.move_to_joint_goal(HIDE_LEFT_JOINTS, replan=True)


if __name__=='__main__':
    try:
        rospy.init_node('push_test', log_level=rospy.INFO)
        con = PushController()
        con._test()
        #con.init_system()
        #rospy.spin()
    except rospy.ROSInterruptException:
        print 'program interrupted before completion'

