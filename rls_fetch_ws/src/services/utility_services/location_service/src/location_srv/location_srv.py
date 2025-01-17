#!/usr/bin/env python

import json
import os
from geometry_msgs.msg import Pose
from location_service.srv import *

# Debug
DEBUG = True

def dbg_print(text):
    if DEBUG:
        print(text)


class LocationService:
    def __init__(self, robot_name):
        my_path = os.path.abspath(os.path.dirname(__file__))
        self._file_path = os.path.join(my_path, "known_locations_{}.txt".format(robot_name))
        with open(self._file_path) as json_file:
            self._known_locations = json.load(json_file)

        print("LocationService: Init success")

    def set_location(self, location_name, location_pos, save = False):
        '''
        @param location_name, string
        @param location_pose, geometry_msgs.msg.Pose.
        '''
        if location_pos is None:
            print("LocationService/set_location: location_pos is None!!")
            return False

        if location_name not in self._known_locations:
            self._known_locations[location_name] = {}

        self._known_locations[location_name]['x'] = location_pos.position.x
        self._known_locations[location_name]['y'] = location_pos.position.y
        # robot_orientation = [location_pos.orientation.x, location_pos.orientation.y, location_pos.orientation.z, location_pos.orientation.w]
        # theta = tf.transformations.euler_from_quaternion(robot_orientation)[2]
        # self._known_locations[location_name]['theta'] = theta
        self._known_locations[location_name]['quat'] = {}
        self._known_locations[location_name]['quat']["x"] = location_pos.orientation.x
        self._known_locations[location_name]['quat']["y"] = location_pos.orientation.y
        self._known_locations[location_name]['quat']["z"] = location_pos.orientation.z
        self._known_locations[location_name]['quat']["w"] = location_pos.orientation.w

        if DEBUG:
            print("LocationService: Added {} with location {}, {}".format(location_name, location_pos.position.x, location_pos.position.y))

        if save:
            rospy.loginfo("LocationSrv: saving!")
            with open(self._file_path, "w") as json_file:
                json.dump(self._known_locations, json_file)
                json_file.close()

            with open(self._file_path) as json_file:
                self._known_locations = json.load(json_file)

        return True

    def forget_location(self, location_name, save = False):
        '''
        Forget position of a location
        @ return, return True if successful, return False otherwise
        '''

        if location_name in self._known_locations:
            del self._known_locations[location_name]

        if save:
            rospy.loginfo("LocationSrv: saving!")
            with open(self._file_path, "w") as json_file:
                json.dump(self._known_locations, json_file)
                json_file.close()

            with open(self._file_path) as json_file:
                self._known_locations = json.load(json_file)

        return True

    def get_location(self, location_name):
        '''
        Get posiiton of a location
        @return, return pose if successful, return None if not successful.
        '''

        print("LocationService: Getting location of name '{}'".format(location_name))
        if location_name in self._known_locations:
            pose = Pose()
            pose.position.x = self._known_locations[location_name]['x']
            pose.position.y = self._known_locations[location_name]['y']
            pose.position.z = 0
            #rot = tf.transformations.quaternion_from_euler(0, 0, self._known_locations[location_name]['theta'])
            pose.orientation.x = self._known_locations[location_name]['quat']['x']
            pose.orientation.y = self._known_locations[location_name]['quat']['y']
            pose.orientation.z = self._known_locations[location_name]['quat']['z']
            pose.orientation.w = self._known_locations[location_name]['quat']['w']

            return pose
        else:
            print("LocationService: No location of name '{}'".format(location_name))
            return None

if __name__ == '__main__':
    loc_service = LocationService()