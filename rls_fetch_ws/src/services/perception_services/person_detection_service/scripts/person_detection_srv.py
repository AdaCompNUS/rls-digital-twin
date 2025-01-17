#!/usr/bin/env python

import rospy
import json
from std_msgs.msg import String

import person_detector.person_detector as person_detector
from rls_perception_msgs.srv import *


class PersonDetectionService(object):
    def __init__(self):
        self._started = False
        self._to_remember_new_face = False
        self._img_listener = {}
        self._img_listener["rgb"] = self.get_rgb_image
        self._img_listener["depth"] = self.get_depth_image

        ## Person detector
        self._person_detector = person_detector.PersonDetector(self._img_listener, self._on_person_detected)

    def get_rgb_image(self):
        return self._robot.camera_get_rgb()

    def get_depth_image(self):
        return self._robot.camera_get_depth()

    def _on_person_detected(self, person_list):
        person_list_string = json.dumps(person_list)
        self._pub.publish(person_list_string)

    def _handle_req(self, req):
        resp = PersonDetectionResponse()

        res = True
        if req.action == PersonDetectionRequest.START_DETECTION:
            if self._started:
                rospy.loginfo("PersonDetectionService/_handle_req: alr started, skip")
            else:
                self._person_detector.start_detection()
                self._started = True

        elif req.action == PersonDetectionRequest.STOP_DETECTION:
            if not self._started:
                rospy.loginfo("PersonDetectionService/_handle_req: alr stopped, skip")
            else:
                self._person_detector.stop_detection()
                self._started = False

        elif req.action == PersonDetectionRequest.REMEMBER_NEW_FACE:
            rospy.loginfo("PersonDetectionService/_handle_req: remember new face")
            res = self._person_detector.remember_new_face(req.person_name)

        else:
            rospy.logerr("PersonDetectionService/_handle_req: invalid req.action")
            res = False

        resp.success = res
        return resp

if __name__ == '__main__':
    rospy.init_node('person_detection_srv')
    person_detection_srv = PersonDetectionService()
    rospy.spin()