#!/usr/bin/env python

import rospy
from std_msgs.msg import Int32

from fetch_robot.fetch_robot import FetchRobot

RESOLUTION = 60

rospy.init_node("test", anonymous=True)

robot = FetchRobot()

def on_dir_received(msg):
    dir = msg.data + 90
    if dir > 360:
        dir -= 360
    print(dir)

    # resolution
    tmp = (dir + RESOLUTION / 2)
    if tmp > 360:
        tmp -= 360
    dir = tmp / RESOLUTION * RESOLUTION
    print(dir)

    clockwise = False
    if dir > 180:
        clockwise = True
        dir = 360 - dir

    print(dir, clockwise)
    robot.base_turn(dir, clockwise=clockwise, speed=0.5)
    rospy.loginfo("turn finished")

while not rospy.is_shutdown():
    msg = rospy.wait_for_message("rls_perception_service/audio_localization", Int32)
    on_dir_received(msg)

rospy.spin()


