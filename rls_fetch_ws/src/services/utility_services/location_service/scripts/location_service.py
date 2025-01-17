#!/usr/bin/env python

import json
import os
import tf
import rospy
from geometry_msgs.msg import Pose

from location_service.srv import *
import location_srv


SET = 0
GET = 1
FORGET = 2

class LocationServiceROS:
    def __init__(self):
        self._location_srv = location_srv.LocationService()

        self._service = rospy.Service('/rls_robot_process/location_service', LocationSrv, self._handle_location)
        rospy.loginfo("LocationService: Init success")

    def _handle_location(self, req):
        resp = LocationSrvResponse()
        if req.action == SET:
            resp.success = self.location_srv.set_location(req.name, req.pose, save = req.save)
        elif req.action == GET:
            pose = self._location_srv.get_location(req.name)
            if pose is None:
                resp.success = False
            else:
                resp.success = True
                resp.pose = pose
        elif req.action == FORGET:
            resp.success = self._location_srv.forget_location(req.name, save = req.save)
        return resp

if __name__ == '__main__':
    rospy.init_node("location_service")
    loc_service = LocationServiceROS()
    rospy.spin()