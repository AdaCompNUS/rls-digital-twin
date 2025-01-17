#!/usr/bin/env python

import rospy
from cv_bridge import CvBridge
import json

from push_net.push_net import PushNetModel
from push_net_msgs.srv import *

# Settings
DEBUG = False


def dbg_print(text):
    if DEBUG:
        print(text)


class PushNetService():
    '''
    ROS Wrapper for PushNet
    '''

    def __init__(self):
        self._push_net = PushNetModel()
        self._srv = rospy.Service(
            'rls_planning_service/push_net', PushNet, self._handle_req)
        self._cv_bridge = CvBridge()
        rospy.loginfo("PushNetService initialized")

    def _handle_req(self, req):
        '''
        Handle Request
        '''
        resp = PushNetResponse()

        rospy.loginfo("PushNetService/_handle_req")
        if req.Action == PushNetRequest.SelectAction:
            action_value = self._select_action(
                req.image_cur, req.image_goal, req.action)

            dbg_print(action_value)

            resp.action_value = json.dumps(action_value)
        elif req.Action == PushNetRequest.Update:
            self._update(req.best_start, req.best_end,
                         req.image_cur, req.image_goal)

        return resp

    def _select_action(self, img_cur_msg, img_goal_msg, action):
        '''
        Ask PushNet to output action values for each action
        @param img_cur_msg, sensor_msgs/Image, the current image
        @param img_goal_msg, sensor_msgs/Image, the goal image
        @oaram action, list, the list of actions. Each action is represented by 4 items in list. [start_x, start_y, goal_x, goal_y]
        '''

        # keep hidden state the same for all action batches
        hidden = self._push_net.model.hidden
        img_cur = self._cv_bridge.imgmsg_to_cv2(img_cur_msg)
        img_goal = self._cv_bridge.imgmsg_to_cv2(img_goal_msg)
        action_value = self._push_net.select_action(img_cur, img_goal, action)
        self._push_net.model.hidden = hidden

        return action_value

    def _update(self, start, end, img_cur_msg, img_goal_msg):
        '''
        Update the most recent push to PushNet. This will make PushNet's LSTM condition more on the real push actions
        @param start, the push start position in image frame
        @param end, the push end position in image frame
        @param img_cur_msg, sensor_msgs/Image, the current image
        @param img_goal_msg, sensor_msgs/Image, the goal image
        '''

        img_cur = self._cv_bridge.imgmsg_to_cv2(img_cur_msg)
        img_goal = self._cv_bridge.imgmsg_to_cv2(img_goal_msg)

        self._push_net.update(start, end, img_cur, img_goal)


if __name__ == '__main__':
    try:
        rospy.init_node('push_net_service', log_level=rospy.INFO)
        push_net_srv = PushNetService()
        rospy.spin()
    except rospy.ROSInterruptException:
        print 'program interrupted before completion'
