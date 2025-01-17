#!/usr/bin/env python
'''
Main program to push objects with real robot
'''

from std_msgs.msg import Bool, Float64, String
import time
from geometry_msgs.msg import PoseStamped, PointStamped
import tf
import argparse
import math
import json
from sensor_msgs.msg import Image, PointCloud2, JointState
import rospy
import os
import numpy as np
import cv2
from cv_bridge import CvBridge
from moveit_python import PlanningSceneInterface

from push_net_msgs.srv import *
from rls_push_msgs.srv import *
from rls_perception_msgs.msg import *
from rls_perception_msgs.srv import *

from robot_push_perception.process_img import *
from robot_push_perception import robot_push_vision_detector_factory, process_img
from robot_push_control import robot_push_controller_factory

# Settings
DEBUG = True

ASK_BEFORE_EXECUTE = False
PUBLISH_GOAL_POINT = False

# Constants
pose_thresh = 0.04
GOAL_W_TOLERANCE = 15

view_pixel = []
MAX_PT = 1000
PLANNING_SCENE_SYNC_WAIT_TIME = 2
READY_POSE_HEAD_TILT = -42.0
TABLE_EXTRA_HEIGHT = 0.01

MAX_DIST_X = 0.3
MAX_DIST_Y = 0.3

MAX_PUSH_STEP = 25

# TODO!!!
MODE = 'wxy'
METHOD = 'simcom'
# BATCH_SIZE = 25
BATCH_SIZE = 25
W = 128.0  # !!!! Important to make it float to prevent integer division becomes zeros
H = 106.0


def dbg_print(text):
    if DEBUG:
        print(text)


class RobotPushService:
    ''' Robot Push Service '''

    def __init__(self, robot_name="Fetch"):
        # self.makerfinder = MarkerFinder()

        # Table Segmentor
        rospy.loginfo(
            "RobotPushService: Waiting for table pointcloud segmentation service rls_perception_services/segment_table ...")
        rospy.wait_for_service('rls_perception_services/segment_table')
        self.table_segmentor_srv = rospy.ServiceProxy('rls_perception_services/segment_table', TableSegmentation)
        rospy.loginfo("Service found!")

        # push_net_client
        rospy.loginfo("RobotPushService: Waiting for rls_planning_service/push_net ...")
        rospy.wait_for_service('/rls_planning_service/push_net')
        self._push_net_client = rospy.ServiceProxy("rls_planning_service/push_net", PushNet)

        # subscribe joint state
        # self.joint_sub = rospy.Subscriber('/joint_states', JointState, self.joint_callback)

        # get robot push controller and vision detector
        self._robot_vision_detector = robot_push_vision_detector_factory.get_robot_push_vision_detector(
            robot_name)
        self._robot_push_controller = robot_push_controller_factory.get_robot_push_controller(
            robot_name)

        # moveit planning scene
        self.scene = PlanningSceneInterface("base_link")

        # debug publisher
        self.p1_pub = rospy.Publisher('start_pose', PoseStamped, queue_size=10)
        self.p2_pub = rospy.Publisher('end_pose', PoseStamped, queue_size=10)
        self.goal_pub = rospy.Publisher('goal_pose', PointStamped, queue_size=10)
        self.img_action_pub = rospy.Publisher('img_act', Image, queue_size=10)
        self._img_debug_pub = rospy.Publisher('img_debug', Image, queue_size=2)

        # other attributes
        self.step_left = 3
        self.num_exp = 10
        self.max_step = MAX_PUSH_STEP
        self.left = True
        self.dist_x = 0
        self.dist_y = 0
        self.dist_w = 0
        self._cv_bridge = CvBridge()
        self.f = Float64(0)
        self.robot_frame = 'base_link'
        self.joint_angle = []
        self.last_value = -0.005
        self._robot = robot_name
        self._ready = False

        # establish the service
        self.push_srv = rospy.Service('rls_control_services/push', RobotPush, self.push_callback)

        rospy.loginfo('RobotPushService is UP!!!')

    def get_ready(self):
        # check whether it is already in ready state
        if self._ready:
            rospy.loginfo("RobotPushService/get_ready: already ready")
            return True

        # Get ready to push
        rospy.loginfo("RobotPushService/get_ready")

        # clear existing obstacles
        self._clear_obstacles(all_obj=True)

        # 1. Raise torso
        # self.torso.set_height(self.torso.MAX_HEIGHT)
        # self.gripper_right.close()
        # self.head.pan_tilt(pan=0, tilt=READY_POSE_HEAD_TILT * math.pi / 180, duration=2)
        # rospy.sleep(2)  # wait for it to finish

        # Detect table
        res = self._set_table_as_obstacle()
        if not res:
            rospy.logerr("RobotPushService/get_ready failed")
            raise RuntimeError("RobotPushService/get_ready failed")
            return False

        # Set table height to sub-modules
        self._robot_push_controller.set_table_height(self._table_height)
        self._robot_vision_detector.set_table_height(self._table_height)

        # Move to ready pick pose first
        res = self._robot_push_controller.move_arm_to_home()
        if not res:
            rospy.logerr("RobotPushService/get_ready failed!!")
            return False

        # Move closer to table if needed
        # !!!!This is dangerous if you do not detect table edge!!
        # res = self._move_closer_to_table(dist=DIST_TO_MOVE_TO_TABLE)
        # if not res:
        #     rospy.logerr("simple_pnp/get_ready failed")
        #     return False

        self._ready = True
        rospy.loginfo("RobotPushService/get_ready finished!")
        return True

    # def joint_callback(self, data):
    #     if len(data.position) > 8:
    #         self.joint_angle = list(data.position[6:])

    def push_callback(self, req):
        if not self._ready:
            rospy.logerr("RobotPushService/push_callback: not ready yet!")
            return RobotPushResponse(False)

        res = False
        # Find marker first
        if req.request < 0:
            # self.makerfinder.find_marker()
            # print 'marker localized !!!'
            pass
        else:
            # self.makerfinder.find_marker()
            # print 'marker localized !!!'
            # self.gripper.close()
            res = self.run(req)

        if res:
            self._robot_push_controller.say("I have finished pushing")
        else:
            self._robot_push_controller.say("Sorry, I can not push this")

        return RobotPushResponse(res)

    def _test(self):
        # Find marker first
        # self.makerfinder.find_marker()
        # print 'marker localized !!!'
        # self.gripper.open()
        rospy.sleep(1)
        # self.gripper.close(50)
        self.run()
        # self.grasp_from_side('left')

    def get_points(self):
        dbg_print("RobotPushService/GetPoints")

        req = UpdateStaticRequest()
        # req.dz.data = -0.005
        req.dz.data = self.last_value
        # self.update_static_client(req)
        # segmentation
        points = None
        for i in range(3):
            points = self._robot_vision_detector.segment().points

        '''
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
            # self.update_static_client(req)
        '''
        return points

    def get_current_mask(self, points):
        dbg_print("RobotPushService/get_current_mask")
        # transformation
        mask, cur_x, cur_y = self._robot_vision_detector.generate_input_image(points)
        # mask is already centered
        return mask, cur_x, cur_y

    def get_target_mask(self, mask, dist_w, dist_x, dist_y):
        return self._robot_vision_detector.transform_img(mask, dist_w, dist_x, dist_y)

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
                    print('out of bound')
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
            return (abs(tran[0]) <= pose_thresh) and (abs(tran[1]) <= pose_thresh) and (abs(w) <= GOAL_W_TOLERANCE)
            # return (abs(tran[0]) * self.sensor.meters_per_pixel <= pose_thresh) and (abs(tran[1]) * self.sensor.meters_per_pixel <= pose_thresh) and (abs(w) <= 10)
        elif mode == 'xy':
            return (abs(tran[0]) <= pose_thresh) and (abs(tran[1]) <= pose_thresh)
        elif mode == 'w':
            return (abs(w) <= GOAL_W_TOLERANCE)

        return False

    def update_tf(self):
        dbg_print("RobotPushService/update_tf")
        # self.update_tf_client(Bool())

    def run(self, req=None):
        dbg_print("RobotPushService/run")

        global MODE
        self.curr_step = 0
        self.step_left = 0
        # Step 1: get object mask
        bs = BATCH_SIZE

        self.update_tf()

        # Get points corresponding to object on the table
        points = self.get_points()

        # Generate input image to pushnet from point
        curr_mask, cur_x, cur_y = self.get_current_mask(points)
        cv2.imshow("cur_img", curr_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


        goal_w = 0  # angle at goal
        dist_x = 0  # distance to move in x
        dist_y = 0  # distance to move in y

        ###################
        ''' Define Goal '''
        ###################
        if req.request == RobotPushRequest.PUSH_TO_EDGE:
            ''' Push to Edge'''
            rospy.logerr("RobotPushService: PUSH_TO_EDGE not supported, skip")
            return False

            MODE = 'wxy'
            goal_x = 0.28  # cannot to far away from the robot, otherwise the arm will collide with table
            goal_y = -0.13 - 0.02  # edge y position NOTE: -0.02 make sure COM is away from table edge for2cm
            curr_mask = curr_mask.astype(np.uint8)
            goal_w = get_goal_w(curr_mask.copy())
            dist_x = goal_x - cur_x
            dist_y = goal_y - cur_y

        elif req.request == RobotPushRequest.PUSH_XY:
            if req.dx > MAX_DIST_X or req.dy > MAX_DIST_Y:
                rospy.logerr(
                    "RobotPushService, dx {} and dy {} is too far away, skip".format(req.dx, req.dy))
                return False

            MODE = 'xy'
            dist_w = 0
            dist_x = req.dx
            dist_y = req.dy
            goal_x = dist_x + cur_x
            goal_y = dist_y + cur_y

        elif req.request == RobotPushRequest.PUSH_WXY:
            if req.dx > MAX_DIST_X or req.dy > MAX_DIST_Y:
                rospy.logerr(
                    "RobotPushService, dx {} and dy {} is too far away, skip".format(req.dx, req.dy))

            MODE = 'wxy'
            dist_w = req.dw
            dist_x = req.dx
            dist_y = req.dy
            goal_x = dist_x + cur_x
            goal_y = dist_y + cur_y

        else:
            rospy.logerr("RobotPushService: request {} not recognized".format(req.request))
            return False

        print('++++ Goal dist_w, dist_x, dist_y ++++++++')
        print("{} {} {}".format(dist_w, dist_x, dist_y))
        print('++++++++++++')
        if dist_y > 0:
            self.left = False
        else:
            self.left = True

        # if self.left:
        #     # self.arm.move_to_joint_goal(
        #     #     HIDE_LEFT_JOINTS,    replan=True, execution_timeout=20.0)
        # else:
        #     # self.arm.move_to_joint_goal(
        #     #     HIDE_RIGHT_JOINTS, replan=True, execution_timeout=20.0)

        # self.arm.move_to_joint(READY_PICK_R_JOINTS, group_name='right_arm')

        if PUBLISH_GOAL_POINT:
            goal_point = PointStamped()
            goal_point.header.frame_id = self._robot_vision_detector.world_frame
            goal_point.point.x = goal_x
            goal_point.point.y = goal_y
            goal_point.point.z = self._table_height + 0.05

            self.goal_pub.publish(goal_point)
            rospy.sleep(0.1)

        # Step 3: generate goal mask
        goal_mask = self.get_target_mask(curr_mask, dist_w, int(-dist_y / self._robot_vision_detector.meters_per_pixel),
                                         int(-dist_x / self._robot_vision_detector.meters_per_pixel))
        cv2.imshow("goal_img", goal_mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        num_step = 0
        # Step 4: loop over until goal is reached
        while not rospy.is_shutdown():
            # Step 5: generate next target mask
            # in case camera moves relative to the marker
            self.update_tf()
            points = self.get_points()

            # convert current mask to cv2 img
            img_in_curr, cur_x, cur_y = self.get_current_mask(points)
            img_in_curr = img_in_curr.astype(np.uint8)

            _, img_in_curr = cv2.threshold(img_in_curr.copy(), 30, 255, cv2.THRESH_BINARY)

            # publish dbg img
            cur_img_msg = self._cv_bridge.cv2_to_imgmsg(curr_mask)
            self._img_debug_pub.publish(cur_img_msg)
            rospy.loginfo("RobotPushService/Run: current img published")

            # update error
            error_x = 0
            error_y = 0
            error_w = 0

            if req.request == RobotPushRequest.PUSH_TO_EDGE:
                # error_x = goal_x - cur_x
                # error_y = goal_y - cur_y
                # error_w = get_goal_w(img_in_curr)
                # if self.is_graspable(points):
                #     print('Success :)')
                #     break
                break

            elif req.request == RobotPushRequest.PUSH_XY:
                error_x = goal_x - cur_x
                error_y = goal_y - cur_y
                error_w = 0

                if self.is_success([error_x, error_y], error_w, MODE):
                    print('Success :)')
                    break

            elif req.request == RobotPushRequest.PUSH_WXY:
                error_x = goal_x - cur_x
                error_y = goal_y - cur_y
                diff_tran, error_w = process_img.get_img_transform(img_in_curr, goal_mask)

                if self.is_success([error_x, error_y], error_w, MODE):
                    print('Success :)')
                    rospy.logwarn("final error is {} {} {}".format(error_x, error_y, error_w))
                    break

            else:
                rospy.logerr("RobotPushService: request {} not recognized".format(req.request))
                return False

            # increase step cnt
            self.curr_step += 1
            if self.curr_step > self.max_step:
                print('maximum step {} reached Failure :('.format(self.max_step))
                break

            rospy.logwarn("error is {} {} {}".format(error_x, error_y, error_w))

            # calculate next push distances
            if abs(error_y) < 0.1:
                self.dist_x = int(-error_y / self._robot_vision_detector.meters_per_pixel)
            else:
                self.dist_x = -error_y/abs(error_y) * 0.1 / \
                    self._robot_vision_detector.meters_per_pixel

            if abs(error_x) < 0.1:
                self.dist_y = int(-error_x / self._robot_vision_detector.meters_per_pixel)
            else:
                self.dist_y = -error_x/abs(error_x) * 0.1 / \
                    self._robot_vision_detector.meters_per_pixel

            if abs(error_w) < 30:  # degree
                self.dist_w = error_w
            else:
                self.dist_w = error_w / abs(error_w) * 30

            # don't neet to be perfectly aligned
            # if abs(error_w) < 10:
            #     self.dist_w = 0
            #     MODE = 'xy'
            # else:
            #     MODE = 'wxy'

            if MODE == 'xy':
                self.dist_w = 0
            elif MODE == 'w':
                self.dist_x = 0
                self.dist_y = 0

            img_in_next = self.get_target_mask(img_in_curr, self.dist_w, self.dist_x, self.dist_y)

            # Step 6: sample actions
            actions = self.sample_action_new(img_in_curr.copy(), 1000)
            print('action sampled')

            # Step 7: select actions
            num_action = len(actions) / 4
            num_action_batch = num_action / bs
            min_sim_score = 1000

            best_start = None
            best_end = None
            best_sim = None
            best_com = None

            action_batch = []
            # hidden = self.pred.model.hidden  # keep hidden state the same for all action batches

            action_value_pairs = []

            for i in range(num_action_batch):
                # self.pred.model.hidden = hidden
                action = actions[4*i*bs: 4*(i+1)*bs]
                # action_value = self.pred.select_action(
                #     img_in_curr, img_in_next, action)
                push_net_req = PushNetRequest()
                push_net_req.Action = PushNetRequest.SelectAction
                push_net_req.image_cur = self._cv_bridge.cv2_to_imgmsg(img_in_curr)
                push_net_req.image_goal = self._cv_bridge.cv2_to_imgmsg(img_in_next)
                push_net_req.action = action

                resp = self._push_net_client(push_net_req)
                action_value = json.loads(resp.action_value)

                action_value_pairs = action_value_pairs + action_value

            # sort action based on sim score
            action_value_pairs.sort(key=lambda x: x[1])

            # execute action
            best_start, best_end = self.execute_push(action_value_pairs, cur_x, cur_y, img_in_curr.copy())
            if best_start == None or best_end == None:
                print('sample action again')
            else:
                dbg_print("best_start: {}".format(best_start))
                dbg_print("best_end: {}".format(best_end))

                push_net_req = PushNetRequest()
                push_net_req.Action = PushNetRequest.Update
                push_net_req.best_start = best_start
                push_net_req.best_end = best_end
                push_net_req.image_cur = self._cv_bridge.cv2_to_imgmsg(img_in_curr)
                push_net_req.image_goal = self._cv_bridge.cv2_to_imgmsg(img_in_next)

                self._push_net_client(push_net_req)

            # to_continue = raw_input("to_continue?")
            # if to_continue == 'q':
            #     break
            num_step += 1
            rospy.logwarn("RobotPush step : {}".format(num_step))

        if req.request == RobotPushRequest.PUSH_TO_EDGE:
            self._robot_push_controller.grasp_from_side('left')
        # self.arm.move_to_joint_goal(HIDE_LEFT_JOINTS, replan=True)

        rospy.loginfo("RobotPush takes {} steps!!!".format(num_step))

        return True

    def is_graspable(self, points):
        y_avg = 0
        count = 0
        for p in points:
            if p.y > -0.13:
                y_avg += p.y
                count += 1
        if count == 0:
            print('no part is outside edge')
            return False
        y_avg = y_avg / count

        distance_from_table = y_avg + 0.13
        print('distance from table is {}'.format(distance_from_table))
        # if distance_from_table > 0.025:
        if distance_from_table > 0.021:
            return True
        else:
            return False

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
        ros_img = self._cv_bridge.cv2_to_imgmsg(img_3d, encoding='bgr8')
        for i in range(100):
            self.img_action_pub.publish(ros_img)

        # cv2.imshow('action', img_3d)
        # if last:
        #    cv2.waitKey(0)
        # else:
        #    cv2.waitKey(10)

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
                print('not able to find contact point')
                return start, end

        action_inside = math.sqrt(
            (next_x - end[0]) * (next_x - end[0]) + (next_y - end[1]) * (next_y - end[1]))
        ideal_len = math.sqrt((self.dist_x * self.dist_x + self.dist_y * self.dist_y)) * 0.8
        print('ideal length is: {}'.format(ideal_len))

        if action_inside < ideal_len:
            print('it is ok')
            # end[0] = int(end[0] * 0.7 + (next_x + math.sqrt(ideal_len) * math.sin(angle)) * 0.3)
            # end[1] = int(end[1] * 0.7 + (next_y + math.sqrt(ideal_len) * math.cos(angle)) * 0.3)

            return start, end
        else:
            # reduce action length to avoid overshooting
            print('reduce action length')
            end[0] = int(next_x + ideal_len * math.sin(angle))
            end[1] = int(next_y + ideal_len * math.cos(angle))
            return start, end

    def execute_push(self, sorted_actions, cur_x, cur_y, img):
        rospy.loginfo("RobotPushService/execute_push")

        next_best_action = sorted_actions.pop(0)
        best_start = next_best_action[0][0]
        best_end = next_best_action[0][1]
        self.draw_action(img.copy(), best_start, best_end, last=True)

        # rospy.sleep(1)

        best_start, best_end = self.tune_action(best_start, best_end, img)
        self.draw_action(img.copy(), best_start, best_end, last=True)

        # calculate start pose from action
        push_start_pose, dist_x, dist_y = self._robot_vision_detector.pixel_to_pose(best_start, best_end, cur_x, cur_y)

        rospy.loginfo("RobotPushService/execute_push: push_start_pose: {}".format(push_start_pose))

        # plan arm to push start pose
        result = self._robot_push_controller.move_arm_to_pose(push_start_pose, plan_only=True)
        if ASK_BEFORE_EXECUTE:
            execute = raw_input('move to push_start_pose, execute?')
            if execute != 'y':
                rospy.loginfo("RobotPushService/execute_push: fail to move to push_start_pose")
                return best_start, best_end

        while not result and not rospy.is_shutdown():
            rospy.logwarn('RobotPushService: cannot plan a path with this action')
            if len(sorted_actions) == 0:
                rospy.logwarn('RobotPushService: all actions have been exhausted!')
                return None, None

            next_best_action = sorted_actions.pop(0)
            best_start = next_best_action[0][0]
            best_end = next_best_action[0][1]
            best_start, best_end = self.tune_action(best_start, best_end, img)

            self.draw_action(img.copy(), best_start, best_end, last=True)
            push_start_pose, dist_x, dist_y = self._robot_vision_detector.pixel_to_pose(
                best_start, best_end, cur_x, cur_y)

            rospy.loginfo("RobotPushService/execute_push: push_start_pose: {}".format(push_start_pose))

            result = self._robot_push_controller.move_arm_to_pose(push_start_pose, plan_only=True)
            if ASK_BEFORE_EXECUTE:
                execute = raw_input('move to push_start_pose, execute?')
                if execute != 'y':
                    rospy.loginfo("RobotPushService/execute_push: fail to move to push_start_pose")
                    return best_start, best_end

        # move arm to push start pose
        result = self._robot_push_controller.move_arm_to_pose(push_start_pose, plan_only=False)

        # get approach and depart height and ensure the robot won't touch the table surface
        approach_height = self._robot_vision_detector.get_dz()

        rospy.loginfo(
            "RobotPushService/execute_push: dist_x {}, dist_y {}, app_height: {}".format(dist_x, dist_y, approach_height))
        if ASK_BEFORE_EXECUTE:
            execute = raw_input('execute?')
            if execute != 'y':
                rospy.loginfo("RobotPushService/execute_push: fail to execute")
                return best_start, best_end

        # push
        self._robot_push_controller.move_arm_in_cartesian(dz=-approach_height)
        self._robot_push_controller.move_arm_in_cartesian(dx=dist_x, dy=dist_y)
        self._robot_push_controller.move_arm_in_cartesian(dz=approach_height)

        self._robot_push_controller.move_arm_to_home()
        return best_start, best_end

    def _set_table_as_obstacle(self):
        '''
        Add table to planning scene as an obstacle
        @return, boolean, return true if successful, return false otherwise.
        '''

        # call table segmentor to get table marker
        rospy.loginfo("Calling table segmentor")
        resp = self.table_segmentor_srv(0)  # segment table
        table_msg = resp.marker
        # rospy.loginfo("table_marker msg arrived: {}".format(table_msg))

        # TODO !!!
        # This involves editting thrid party library. move this module into MOVO to fix this.
        # the reason is that movo time is different from predator's time.
        # self.scene.addBoxWithTime('table', table_msg.scale.x,
        #                           table_msg.scale.y,
        #                           table_msg.scale.z + TABLE_EXTRA_HEIGHT / 2,  # increase table height a little bit for safety
        #                           table_msg.pose.position.x,
        #                           table_msg.pose.position.y,
        #                           table_msg.pose.position.z + TABLE_EXTRA_HEIGHT / 2,  # increase table height a little bit for safety
        #                           table_msg.header.stamp)  # this forces the time to be the same. if not, sync will fail
        self.scene.addBox('table', table_msg.scale.x,
                            table_msg.scale.y,
                            table_msg.scale.z + TABLE_EXTRA_HEIGHT / 2,  # increase table height a little bit for safety
                            table_msg.pose.position.x,
                            table_msg.pose.position.y,
                            table_msg.pose.position.z + TABLE_EXTRA_HEIGHT / 2)  # increase table height a little bit for safety
        self._table_height = table_msg.pose.position.z + table_msg.scale.z / 2
        self._table_dist = table_msg.pose.position.x - table_msg.scale.x / 2
        rospy.loginfo("table_height: {}, table_dist {}".format(
            self._table_height, self._table_dist))
        self.scene.waitForSync(max_time=PLANNING_SCENE_SYNC_WAIT_TIME)

        return True

    def _clear_obstacles(self, all_obj=False):
        '''
        Remove collision objects
        @param all_obj, boolean, if true, removes table as well
        '''

        for name in self.scene.getKnownAttachedObjects():
            self.scene.removeAttachedObject(name)
        for name in self.scene.getKnownCollisionObjects():
            if name == 'table':
                if all_obj:
                    self.scene.removeCollisionObject(name)
                else:
                    rospy.loginfo('RobotPushService: removing objects, ignoring table')
                    continue
            self.scene.removeCollisionObject(name)

        self.scene.waitForSync(max_time=PLANNING_SCENE_SYNC_WAIT_TIME)


if __name__ == '__main__':
    try:
        rospy.init_node('push_test', log_level=rospy.INFO)
        robot_push_service = RobotPushService()
        robot_push_service.get_ready()
        # con._test()
        # con.init_system()
        rospy.spin()
    except rospy.ROSInterruptException:
        print('program interrupted before completion')
