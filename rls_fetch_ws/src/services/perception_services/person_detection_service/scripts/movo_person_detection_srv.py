#!/usr/bin/env python
import rospy
from std_msgs.msg import String

from rls_perception_msgs.srv import *
from person_detection_srv import PersonDetectionService
from movo_robot.movo_robot import MovoRobot

class MovoPersonDetectionService(PersonDetectionService):
    def __init__(self):
        self._robot = MovoRobot(use_ros=True)
        PersonDetectionService.__init__(self)
        ## Expose service and publisher
        self._service = rospy.Service('rls_perception_services/movo/person_detection', PersonDetection, self._handle_req)
        self._pub = rospy.Publisher('rls_perception_services/movo/person_detection_res', String, queue_size = 0) # infinite queue_size
        rospy.loginfo("MovoPersonDetectionService: inited")

if __name__ == '__main__':
    rospy.init_node('movo_person_detection_srv')
    person_detection_srv = MovoPersonDetectionService()
    rospy.spin()