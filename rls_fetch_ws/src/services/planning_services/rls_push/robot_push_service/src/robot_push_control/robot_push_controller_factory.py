#!/usr/bin/env python

def get_robot_push_controller(robot="Movo"):
    robot_push_controller = None
    if robot == 'Movo':
        from movo_push_controller import MovoPushController
        robot_push_controller = MovoPushController()
    elif robot == 'Fetch':
        from fetch_push_controller import FetchPushController
        robot_push_controller = FetchPushController()
    return robot_push_controller
