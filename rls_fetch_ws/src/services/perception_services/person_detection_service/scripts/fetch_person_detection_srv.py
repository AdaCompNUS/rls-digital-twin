#!/usr/bin/env python
import rospy
from std_msgs.msg import String

from rls_perception_msgs.srv import *
from person_detection_srv import PersonDetectionService
from fetch_robot.fetch_robot import FetchRobot

class FetchPersonDetectionService(PersonDetectionService):
    def __init__(self):
        self._robot = FetchRobot()
        PersonDetectionService.__init__(self)
        ## Expose service and publisher
        self._service = rospy.Service('rls_perception_services/fetch/person_detection', PersonDetection, self._handle_req)
        self._pub = rospy.Publisher('rls_perception_services/fetch/person_detection_res', String, queue_size = 0) # infinite queue_size
        rospy.loginfo("FetchPersonDetectionService: inited")

if __name__ == '__main__':
    rospy.init_node('fetch_person_detection_srv')
    person_detection_srv = FetchPersonDetectionService()
    rospy.spin()