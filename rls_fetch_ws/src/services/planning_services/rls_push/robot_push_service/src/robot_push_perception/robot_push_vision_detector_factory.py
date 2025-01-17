#!/usr/bin/env python

from movo_push_vision_detector import MovoPushVisionDetector
from fetch_push_vision_detector import FetchPushVisionDetector


def get_robot_push_vision_detector(robot='Movo'):
    robot_push_vision_detector = None
    if robot == 'Movo':
        robot_push_vision_detector = MovoPushVisionDetector()
    elif robot == 'Fetch':
        robot_push_vision_detector = FetchPushVisionDetector()
    return robot_push_vision_detector
