'''
Main program to push objects with real robot
'''

__author__='Li Juekun'
__date__ = '2018/01/12'

## debug segfault
#import faulthandler
#faulthandler.enable()
## pytorch mods ##
import torch
import torch.nn as nn
import cv2
from model_tc import *
from torch.autograd import Variable

import numpy as np
import os
### ROS modules
import rospy
from sensor_msgs.msg import Image, PointCloud2
import json
import math
import argparse
from cv_bridge import CvBridge
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
from robot_config import *
from std_msgs.msg import Bool, Float64, String
from rls_push_msgs.srv import *



W = 128.0 ##!!!! Important to make it float to prevent integer division becomes zeros
H = 106.0
#OBJ_TYPE = 'convex'
OBJ_TYPE = 'concave'
#MODE = 'xy'
#MODE = 'w'
MODE = 'wxy'
#METHOD = 'simcom'
#METHOD = 'sim'
METHOD = 'nomem'
pose_thresh = 0.04
fetch_end_eff = 'gripper_link'
gripper_to_wrist = 0.16645
fetch_gripper_height = 0.81 ##for lower box eg. bowl, sugar box
#fetch_gripper_height = 0.85 ##for higher box eg. stick/straw
z_diff = 0.10
gripper_to_leftout_y = -0.09
#LEFT = True
view_pixel = [98.86, 66.91]

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
            sim_out, com_out = self.model(A1, I1, A1, Ig, [1 for i in range(bs)], bs)
        elif METHOD == 'sim':
            sim_out = self.model(A1, I1, A1, Ig, [1 for i in range(bs)], bs)
        elif METHOD == 'nomem':
            sim_out = self.model(A1, I1, A1, Ig, [1 for i in range(bs)], bs)

    def select_action(self, img_curr, img_goal, actions):
        bs = self.bs
        A1 = []
        I1 = []
        Ig = []
        #print 'select'
        #print bs, H, W

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

        #com_np = None
        sim_out = None
        com_out = None
        #st_time = time.time()
        if METHOD == 'simcom':
            sim_out, com_out = self.model(A1, I1, A1, Ig, [1 for j in range(bs)], bs)
            #com_np = com_out.data.cpu().data.numpy()
        elif METHOD == 'sim':
            sim_out = self.model(A1, I1, A1, Ig, [1 for j in range(bs)], bs)
        elif METHOD == 'nomem':
            sim_out = self.model(A1, I1, A1, Ig, [1 for j in range(bs)], bs)
        #print time.time() - st_time

        sim_np = sim_out.data.cpu().data.numpy()

        if MODE == 'wxy':
            sim_sum = np.sum(sim_np, 1) # measure (w ,x, y)
        elif MODE == 'xy':
            sim_sum = np.sum(sim_np[:,1:], 1) # measure (x, y)
        else:
            sim_sum = np.sum(sim_np, 1) # measure (w ,x, y)
            #sim_sum = sim_np[:, 0] # measure (w)
        action_value = []
        for ii in range(len(sim_sum)):
            s = [actions[4 * ii], actions[4 * ii + 1]]
            e = [actions[4 * ii + 2], actions[4 * ii + 3]]
            action_value.append([[s, e], sim_sum[ii]])

        return action_value
        #min_idx = np.argmin(sim_sum)
        #best_sim_score = np.min(sim_sum)

        #start = [actions[4*min_idx], actions[4*min_idx+1]]
        #end = [actions[4*min_idx+2], actions[4*min_idx+3]]

        #sim_measure = list(sim_np[min_idx,:])
        #com_pred = [com_np[min_idx, 0] * W, com_np[min_idx, 1] * H]
        #return start, end, sim_measure, com_pred, best_sim_score


''' Push Controller '''
class PushController:
    def __init__(self):
        rospy.init_node('push_net_main', log_level=rospy.INFO)
        self.sensor = Sensor()
        self.markerfinder = MarkerFinder(target_pixel=view_pixel)
        self.arm = fetch_api.Arm()
        self.torso = fetch_api.Torso()
        self.head = fetch_api.Head()
        self.gripper = fetch_api.Gripper()

        self.p1_pub = rospy.Publisher('start_pose', PoseStamped, queue_size=10)
        self.p2_pub = rospy.Publisher('end_pose', PoseStamped, queue_size=10)
        self.goal_pub = rospy.Publisher('goal_pose', PointStamped, queue_size=10)
        self.bridge = CvBridge()
        self.img_action_pub = rospy.Publisher('img_act', Image, queue_size=10)

        ## update tf and static tf clients
        self.update_tf_client = rospy.ServiceProxy('update_tf', UpdateTF)
        self.update_static_client = rospy.ServiceProxy('update_static', UpdateStatic)
        print 'ok'

        self.angle_range = [30, 60]
        self.x_range = [-0.15, 0.2]
        self.y_range = [-0.05, -0.03]
        #self.step_left = 10
        self.step_left = 3
        ## goal region
        self.z_offset = 0.03 + gripper_to_wrist ## height of wrist roll link above the table plane
        self.pre_height = 0.1 ## prepush height
        self.num_exp = 10
        self.max_step = 10
        self.left = True
        self.dx = 0
        self.dy = 0
        self.dw = 0
        #self.img_path = '/home/ljk/robot_exp/code/fetch_ws/src/rls_push/rls_push/src/exp_img/'
        self.img_path = '/home/ljk/robot_exp/code/fetch_ws/src/rls_push/rls_push/src/rss_real_result/'
        self.exp_config = [
                            [-30, 0.0, -0.2],
                            [30, 0.0, -0.4],
                            [-45, 0.05, -0.2],
                            [45, 0.05, -0.4],
                            [-60, -0.05, -0.4],
                            [60, 0.0, 0.2],
                            [-75, 0.0, 0.4],
                            [75, 0.05, 0.2],
                            [-90, 0.05, 0.4],
                            [90, -0.05, 0.4]
                            ]
        #self.object_xy = ['strawbox', 'cup', 'glassbox', 'clamp', 'lion', 'bowl', 'wire']
        #self.object_wxy = ['sugar', 'mustard', 'gift', 'banana', 'joystick', 'drill']
        self.object_xy = ['weighted_sugar']
        self.object_wxy = ['weighted_sugar']
        self.symmetry = {'longbar':True, 'sugar':True, 'mustard':False, 'gift':True, 'banana':False, 'joystick':False, 'drill':False, 'weighted_sugar':True}
        #self.object_wxy = ['sugar', 'mustard', 'gift', 'banana', 'joystick', 'drill']
        #self.symmetry = {'longbar':True, 'sugar':True, 'mustard':False, 'gift':True, 'banana':False, 'joystick':False, 'drill':False}
        #self.object_wxy = ['sugar', 'mustard', 'gift', 'banana', 'joystick', 'drill']
        self.json_filename = ''

        #self.video_client = rospy.ServiceProxy('video_record_srv', RecordVideo)

        self.f = Float64(0)
        self.robot_frame = 'base_link'
        self.exp_dict = {}
        self.rate = rospy.Rate(50)
        self.pred = Predictor()
        print 'Server is UP!!!'
        self._test()

    def _test(self):
        '''
        self.torso.set_height(fetch_api.Torso.MAX_HEIGHT)
        rospy.sleep(1)
        self.arm.move_to_joint_goal(HIDE_LEFT_JOINTS, replan=True)
        rospy.sleep(1)
        return
        '''
        global fetch_gripper_height
        self.markerfinder.find_marker()
        print 'marker localized !!!'
        self.gripper.open()
        rospy.sleep(1)
        self.gripper.close(50)

        num_exp = 10
        objects = None

        if MODE == 'xy':
            objects = self.object_xy
        else:
            objects = self.object_wxy

        #for i in range(num_obj):
        start = 0
        for obj in objects:
            self.obj_name = obj
            self.json_filename = 'rss_real_result/'+self.obj_name + '_' + MODE + '_' + METHOD + '.json'
            if os.path.exists(self.json_filename):
                f = open(self.json_filename)
                self.exp_dict = json.load(f)
                f.close()
                start = len(self.exp_dict.keys())
                if start > 9: ## skip to next object
                    continue
            else:
                self.exp_dict = {}
                start = 0

            print 'next object: ', obj
            raw_input('system is ready! put down an object please')
            if obj == 'strawbox':
                fetch_gripper_height = 0.85 ##for higher box eg. stick/straw
            elif obj == 'clamp':
                fetch_gripper_height = 0.805
            else:
                fetch_gripper_height = 0.81

            for j in range(start, num_exp):
                self.exp_idx = j
                self.exp_dict[j] = {}
                self.img_name = self.obj_name + '_exp_' + str(j)
                start_time = time.time()
                print self.exp_config[j]
                if j == 5:
                    print '************ put on the right'

                raw_input('reset the object please ...')
                self.run()
                self.exp_dict[j]['duration'] = round(time.time() - start_time, 4)
                self.save_dict()
                return
                if not METHOD == 'nomem':
                    self.pred.reset_model()

    def save_dict(self):
        with open(self.json_filename, 'w') as fp:
            json.dump(self.exp_dict, fp, sort_keys=True, indent=4)

    def get_points(self):
        req = UpdateStaticRequest()
        req.dz.data -= 0.01
        self.update_static_client(req)
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
                req.dz.data += 0.005
            elif len(points) < 20:
                print '********************** move table DOWN'
                req.dz.data -= 0.005
            else:
                break

            print 'table should be move by: ', req.dz.data
            self.update_static_client(req)

        return points

    def get_current_mask(self, points):

        ## transformation
        mask, xc, yc = self.sensor.generate_input_image(points)
        ## mask is already centered
        return mask, xc, yc

    def get_target_mask(self, mask, dw, dx, dy):
        return self.sensor.transform_img(mask, dw, dx, dy)


    def sample_action_new(self, img, num_actions):
        s = 0.9
        safe_margin = 1.4
        out_margin = 2.0

        img_inner = cv2.resize(img.copy(), (0,0), fx=s, fy=s, interpolation=cv2.INTER_AREA)
        h, w = img_inner.shape
        img_end = np.zeros((int(H), int(W)))
        img_end[(int(H)-h)/2:(int(H)+h)/2, (int(W)-w)/2:(int(W)+w)/2] = img_inner.copy()
        (inside_y, inside_x) = np.where(img_end.copy()>0)

        ## sample start push point outside a safe margin of object
        img_outer1 = cv2.resize(img.copy(), (0,0), fx=safe_margin, fy=safe_margin, interpolation=cv2.INTER_CUBIC)
        h, w = img_outer1.shape
        img_start_safe = np.zeros((int(H), int(W)))
        img_start_safe = img_outer1.copy()[(h-int(H))/2:(h+int(H))/2, (w-int(W))/2:(w+int(W))/2]

        img_outer2 = cv2.resize(img.copy(), (0,0), fx=out_margin, fy=out_margin, interpolation=cv2.INTER_CUBIC)
        h, w = img_outer2.shape
        img_start_out = np.zeros((int(H), int(W)))
        img_start_out = img_outer2.copy()[(h-int(H))/2:(h+int(H))/2, (w-int(W))/2:(w+int(W))/2]

        img_start = img_start_out.copy() - img_start_safe.copy()
        (outside_y, outside_x) = np.where(img_start.copy()>100)

        num_inside = len(inside_x)
        num_outside = len(outside_x)

        actions = []
        for i in range(num_actions):
            start_x = 0
            start_y = 0
            end_x = 0
            end_y = 0
            while not rospy.is_shutdown():
                ## sample an inside point
                inside_idx = np.random.choice(num_inside)
                ## sample an outside point
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
                    #print img[start_y, start_x]
                    continue

            actions.append(start_x)
            actions.append(start_y)
            actions.append(end_x)
            actions.append(end_y)
        return actions

    def is_success(self, tran, w, mode):
        if mode == 'wxy':
            return (abs(tran[0]) <= pose_thresh) and (abs(tran[1]) <= pose_thresh) and (abs(w) <= 10)
        elif mode == 'xy':
            return (abs(tran[0]) <= pose_thresh) and (abs(tran[1]) <= pose_thresh)
        elif mode == 'w':
            return (abs(w) <= 10)

    def update_tf(self):
        self.update_tf_client(Bool())

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
        gw = self.exp_config[self.exp_idx][0]
        gx = self.exp_config[self.exp_idx][1]
        gy = self.exp_config[self.exp_idx][2]
        #MODE = 'xy'
        #gx = 0.0
        #gy = 0.4
        #gw = -60
        xm = gx + xc
        ym = gy + yc


        print '++++ Goal dw, dx, dy ++++++++'
        print gw, gx, gy
        print '++++++++++++'
        self.exp_dict[self.exp_idx]['goal'] = [gw, gx, gy]
        if gy > 0:
            self.left = False
        else:
            self.left = True
        if self.left:
            self.arm.move_to_joint_goal(HIDE_LEFT_JOINTS, replan=True, execution_timeout=20.0)
        else:
            self.arm.move_to_joint_goal(HIDE_RIGHT_JOINTS, replan=True, execution_timeout=20.0)
        #rospy.sleep(1)
        goal_point = PointStamped()
        goal_point.header.frame_id = self.sensor.world_frame
        goal_point.point.x = xm
        goal_point.point.y = ym
        goal_point.point.z = 0.05

        for i in range(100):
            self.goal_pub.publish(goal_point)

        # Step 3: generate goal mask
        goal_mask = self.get_target_mask(curr_mask, gw, int(-gy / self.sensor.meters_per_pixel), int(-gx / self.sensor.meters_per_pixel))

        '''
        ## prepare for goal overlay
        pt = self.get_points()
        move_pt = PointStamped()
        center_pt = PointStamped()
        move_pt.header.frame_id = 'left_out'
        move_pt.point.x = xm
        move_pt.point.y = ym
        move_pt.point.z = 0
        center_pt.header.frame_id = 'left_out'
        center_pt.point.x = xc
        center_pt.point.y = yc
        center_pt.point.z = 0
        cam_pt = self.sensor.tl.transformPoint('/head_camera_rgb_optical_frame', move_pt)
        cam_pt_c = self.sensor.tl.transformPoint('/head_camera_rgb_optical_frame', center_pt)
        fx = 541.787
        fy = 538.148
        x0 = 321.601
        y0 = 232.013
        gpx = (cam_pt.point.x / cam_pt.point.z - cam_pt_c.point.x / cam_pt_c.point.z) * fx
        gpy = (cam_pt.point.y / cam_pt.point.z - cam_pt_c.point.y / cam_pt_c.point.z) * fy

        #self.sensor.get_goal_mask(pt, gw, int(-gy / self.sensor.meters_per_pixel), int(-gx / self.sensor.meters_per_pixel), mode=OBJ_TYPE, type=MODE)
        #self.sensor.get_goal_mask(pt, gw, int(gpx), int(gpy), mode=OBJ_TYPE, type=MODE)

        # start video recording
        goal_img = self.sensor.mask_t
        goal_img = goal_img.astype(np.uint8)
        video_req = RecordVideoRequest()
        video_req.start = Bool(1)
        video_req.img_goal = self.bridge.cv2_to_imgmsg(goal_img, 'bgr8')
        video_req.file_name = String(self.img_name)
        self.video_client(video_req)
        '''


        # Step 4: loop over until goal is reached
        while not rospy.is_shutdown():
           print '################ current step: ', self.curr_step

           ''' overlay goal image '''
           #cv_img = self.sensor.get_current_rgb()
           #overlay = self.sensor.add_goal_to_img(cv_img.copy())
           #cv2.imwrite(self.img_path + self.img_name+'_step_'+str(self.curr_step)+'.jpg', overlay)

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
               #error_w = get_goal_w(img_in_curr)
               diff_tran, error_w = get_img_transform(img_in_curr, goal_mask, symmetric=self.symmetry[self.obj_name])
           print 'error is '
           print error_x, error_y, error_w

           if self.is_success([error_x, error_y], error_w, MODE):
               print 'Success :)'
               self.exp_dict[self.exp_idx]['success'] = True
               break
           if self.curr_step > self.max_step:
               print 'Failure :)'
               self.exp_dict[self.exp_idx]['success'] = False
               break
           self.exp_dict[self.exp_idx][self.curr_step] = {}

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
           print self.dx, self.dy, self.dw

           self.exp_dict[self.exp_idx][self.curr_step]['goal_diff'] = [round(self.dw, 4), round(self.dx, 4), round(self.dy, 4)]

           #print 'get next image'

           img_in_next = self.get_target_mask(img_in_curr, self.dw, self.dx, self.dy)

           # Step 6: sample actions

           #print 'sampling actions'
           actions = self.sample_action_new(img_in_curr.copy(), 1000)
           #print 'actions are sampled!'


           # Step 7: select actions
           if bs == 0:
               print '!!!!!! something seriously wrong!!!!!'
           num_action = len(actions) / 4
           num_action_batch = num_action / bs
           min_sim_score = 1000

           best_start = None
           best_end = None
           best_sim = None
           best_com = None

           action_batch = []
           hidden = None
           if not METHOD == 'nomem':
               hidden = self.pred.model.hidden ## keep hidden state the same for all action batches

           action_value_pairs = []

           for i in range(num_action_batch):
               if not hidden == None:
                   self.pred.model.hidden = hidden
               action = actions[4*i*bs: 4*(i+1)*bs]
               action_value = self.pred.select_action(img_in_curr, img_in_next, action)
               action_value_pairs = action_value_pairs + action_value

           ## sort action based on sim score
           action_value_pairs.sort(key=lambda x : x[1])

           #print 'going to execute action!'
           #### execute action
           best_start, best_end, img_action = self.execute_push(action_value_pairs, xc, yc, img_in_curr.copy())
           if best_start == None or best_end == None:
               print 'sample action again'
           else:
               self.pred.update(best_start, best_end, img_in_curr, img_in_next)
               self.exp_dict[self.exp_idx][self.curr_step]['action'] = [best_start, best_end]
               #cv2.imwrite(self.img_path + self.img_name+'_step_'+str(self.curr_step)+'_mask.jpg', img_action)
               self.curr_step += 1

        ''' release current video recording '''
        #video_req.start = Bool(0)
        #self.video_client(video_req)
        #rospy.sleep(1)


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

        return img_3d

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

           for i in range(100):
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

        while not rospy.is_shutdown():
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
        #print 'ideal length is: ', ideal_len

        if action_inside < ideal_len:
            #print 'it is ok'
            #end[0] = int(end[0] * 0.7 + (next_x + math.sqrt(ideal_len) * math.sin(angle)) * 0.3)
            #end[1] = int(end[1] * 0.7 + (next_y + math.sqrt(ideal_len) * math.cos(angle)) * 0.3)

            return start, end
        else:
            ## reduce action length to avoid overshooting
            #print 'reduce action length'
            end[0] = int(next_x + ideal_len * math.sin(angle))
            end[1] = int(next_y + ideal_len * math.cos(angle))
            return start, end

    def execute_push(self, sorted_actions, xc, yc, img):
           pack = sorted_actions.pop(0)
           best_start = pack[0][0]
           best_end = pack[0][1]
           #print 'draw action'
           self.draw_action(img.copy(), best_start, best_end, last=True)
           #rospy.sleep(1)
           if MODE == 'xy' or MODE == 'wxy':
               best_start, best_end = self.tune_action(best_start, best_end, img)
           #print 'draw again'
           img_ = self.draw_action(img.copy(), best_start, best_end, last=True)
           #print 'convert pixel to pose'

           start_pose_robot, dx, dy = self.pixel_to_pose(best_start, best_end, xc, yc)
           #print 'plan a path'
           result = self.arm.move_to_pose(start_pose_robot, replan=False, execution_timeout=20)
           #print result
           while not result == None and not rospy.is_shutdown():
               print 'ERROR: cannot plan a path'
               if len(sorted_actions) == 0:
                   print 'all actions have been exhausted!'
                   return None, None
               pack = sorted_actions.pop(0)
               best_start = pack[0][0]
               best_end = pack[0][1]
               if MODE == 'xy' or MODE == 'wxy':
                   best_start, best_end = self.tune_action(best_start, best_end, img)

               img_ = self.draw_action(img.copy(), best_start, best_end, last=True)
               start_pose_robot, dx, dy = self.pixel_to_pose(best_start, best_end, xc, yc)
               result = self.arm.move_to_pose(start_pose_robot, replan=False, execution_timeout=20)

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
           print diff
           if abs(diff) > 0.11:
               print 'cannot move too much!'
               if self.left:
                   self.arm.move_to_joint_goal(HIDE_LEFT_JOINTS, replan=True, execution_timeout=20.0)
               else:
                   self.arm.move_to_joint_goal(HIDE_RIGHT_JOINTS, replan=True, execution_timeout=20.0)
               return None, None, None

           self.arm.move_in_cartesian(dz=diff)
           self.arm.move_in_cartesian(dx=dx, dy=dy)
           self.arm.move_in_cartesian(dz=-diff)

           if self.left:
               self.arm.move_to_joint_goal(HIDE_LEFT_JOINTS, replan=True, execution_timeout=20.0)
           else:
               self.arm.move_to_joint_goal(HIDE_RIGHT_JOINTS, replan=True, execution_timeout=20.0)
           #rospy.sleep(1)

           #print 'action done'

           return best_start, best_end, img_


if __name__=='__main__':
    try:
        con = PushController()
        #con._test()
        #con.init_system()
        #rospy.spin()
    except rospy.ROSInterruptException:
        print 'program interrupted before completion'

