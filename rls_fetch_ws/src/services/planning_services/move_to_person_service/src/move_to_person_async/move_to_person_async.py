#!/usr/bin/env python

import rospy
import threading
import tf
import math

class MoveToPersonThread(threading.Thread):
    def __init__(self, person_obs, robot, done_cb, move_to_initial_location=True):
        threading.Thread.__init__(self)
        self._event = threading.Event()
        self._done_cb = done_cb
        self._person_obs = person_obs
        self._robot = robot
        self._cancel = False
        self._lock = threading.Lock()
        self._detected_cnt = 0
        self._person_location_camera = None
        self.cfg = {}

    def set_target(self, person_name, person_location):
        self._person_name = person_name
        self._person_location = person_location

    def set_cfg(self, cfg):
        self.cfg = cfg
        rospy.loginfo("MoveToPerson: cfg: {}".format(self.cfg))

    def run(self):
        self._cancel = False
        rospy.loginfo("MoveToPerson: thread start")
        try:
            if self.cfg["move_to_initial_location"]:
                success = self._move_to_initial_location()
                if not success:
                    raise RuntimeError("MoveToPerson: failed to move to initial location")

            self._person_obs.register_camera_callback(self.person_camera_callback)
            person_detected = self._search()
            if person_detected:
                lost = self._follow()
                self._done_cb(lost)
            else:
                self._done_cb(True)
        except Exception as e:
            rospy.logerr(e)

        self._person_obs.unregister_camera_callback(self.person_camera_callback)
        rospy.loginfo("MoveToPerson thread exit")

    def cancel(self):
        self._cancel = True
        # self._action_clients["nav_to"](cancel_goal = True, move_to_home = False, goal = None)
        self._robot.base_nav(None)

    def person_camera_callback(self, person_name, person_location):
        rospy.loginfo("MoveToPersonThread: person_camera_callback")
        if person_name == self._person_name:
            self._lock.acquire()
            if self._person_location_camera is None:
                self._person_location_camera = person_location
            else:
                # LPF
                self._person_location_camera.position.x = person_location.position.x
                self._person_location_camera.position.y = person_location.position.y
                self._person_location_camera.position.z = self.cfg["alpha"] * self._person_location_camera.position.z + (1 - self.cfg["alpha"]) * person_location.position.z

            self._detected_cnt += 1
            #rospy.loginfo("person_detected {}".format(self._detected_cnt))
            self._lock.release()

            if self._detected_cnt == 2:
                self._detected_cnt = 0
                self._person_detected = True
                self._event.set()

        else:
            rospy.loginfo(
                "MoveToPersonThread: person_camera_callback: but name is different {} vs {}".format(person_name, self._person_name))

    def _move_to_initial_location(self):
        self._person_obs.stop_detection()
        rospy.loginfo("MoveToPerson: start to move to initial location")
        # resp = self._action_clients["nav_to"](cancel_goal = False, move_to_home = False, goal = self._person_location)
        resp = self._robot.base_nav(self._person_location)
        self._person_obs.start_detection()
        return resp

    def _search(self):

        self._person_detected = False
        self._event.clear()
        self._event.wait(timeout = self.cfg["search_wait_duration"])

        self._lock.acquire()
        person_detected = self._person_detected
        self._lock.release()

        if not person_detected:
            self._person_obs.stop_detection()
            # rospy.sleep(6.0)
            rospy.loginfo("MoveToPersonThread: trying to see left")
            # self._action_clients["basic_move"](action = 5, data = 45)
            self._robot.base_turn(angular_distance = 60, clockwise = True)
            #rospy.sleep(1.5)
            self._person_detected = False
            self._person_obs.start_detection()
            #rospy.loginfo("MoveToPersonThread: seeing right")
            self._event.clear()
            self._event.wait(timeout = self.cfg["search_wait_duration"])
        else:
            return True

        self._lock.acquire()
        person_detected = self._person_detected
        self._lock.release()

        if not person_detected:
            self._person_obs.stop_detection()
            rospy.loginfo("MoveToPersonThread: trying to see right")
            # self._action_clients["basic_move"](action = 4, data = 45)
            # self._action_clients["basic_move"](action = 4, data = 45)
            self._robot.base_turn(angular_distance = 40, clockwise = False)
            # self._robot.base_turn(angular_distance = 45, clockwise = True)
            #rospy.sleep(1.5)
            self._person_detected = False
            self._person_obs.start_detection()
            self._event.clear()
            self._event.wait(timeout = self.cfg["search_wait_duration"])
        else:
            return True

        self._lock.acquire()
        person_detected = self._person_detected
        self._lock.release()

        if not person_detected:
            rospy.logwarn("MoveToPersonThread: can't find person")
            return False
        else:
            return True

    def _search_by_rotate(self):

        self._person_detected = False
        self._event.clear()
        self._event.wait(timeout = self.cfg["search_wait_duration"])

        self._lock.acquire()
        person_detected = self._person_detected
        self._lock.release()

        cnt = 0
        while not person_detected and cnt < 8:
            self._person_obs.stop_detection()
            rospy.loginfo("MoveToPersonThread: rotating")
            # self._action_clients["basic_move"](action = 4, data = 45)
            self._robot.base_turn(angular_distance = 30, clockwise = False)
            self._person_detected = False
            self._person_obs.start_detection()
            self._event.clear()
            self._event.wait(timeout = self.cfg["search_wait_duration"])

            self._lock.acquire()
            person_detected = self._person_detected
            self._lock.release()

            cnt += 1

        if not person_detected:
            rospy.logwarn("MoveToPersonThread: can't find person")
            return False
        else:
            return True

    def _follow(self):
        rospy.loginfo("MoveToPerson: following camera")

        self._person_detected = False
        self._event.clear()
        self._event.wait(timeout = 2.0)

        lost_cnt = 0
        lost = False
        followed = False
        follow_cnt = 0
        while not self._cancel:
            self._lock.acquire()
            person_detected = self._person_detected
            self._lock.release()
            if not person_detected:
                rospy.logwarn("MoveToPerson: person lost")
                lost_cnt = lost_cnt + 1
                if (lost_cnt > 5):
                    lost = True
                    break

            else:
                lost_cnt = 0
                lost = False
                # self._robot_goal = self._robot_pose_listener.get_robot_pose()
                # robot_pose_time = self._robot_pose_listener.get_robot_pose_time()
                self._cur_robot_state = self._robot.base_get_state()
                self._lock.acquire()
                person_location_camera = self._person_location_camera
                self._lock.release()

                # compute diff
                robot_orientation = [self._cur_robot_state.orientation.x, self._cur_robot_state.orientation.y, self._cur_robot_state.orientation.z, self._cur_robot_state.orientation.w]
                theta = tf.transformations.euler_from_quaternion(robot_orientation)[2]
                diff_pixel = person_location_camera.position.x - self.cfg["center_x"]
                diff_theta = self.cfg["heading_p"] * diff_pixel
                rospy.loginfo("diff_pixel {}, diff_theta : {}".format(diff_pixel, diff_theta))

                # calcualte dist
                diff_dist = person_location_camera.position.z - self.cfg["desired_dist"]

                # if abs(diff_pixel) >= self.cfg["pixel_diff_threshold"] and math.fabs(diff_theta) >= self.cfg["theta_diff_threshold"]:
                if abs(diff_pixel) >= self.cfg["pixel_diff_threshold"]:
                    follow_cnt = 0
                    # heading control

                    degree = math.degrees(diff_theta)
                    new_theta = theta + diff_theta
                    rospy.loginfo("x {}, diff_pixel {}, theta {}, new_theta {}".format(person_location_camera.position.x, diff_pixel, theta, new_theta))

                    self._person_obs.stop_detection()
                    '''
                    resp = self._move_client(cancel_goal = False, move_to_home = False, goal = self._robot_goal)
                    '''
                    if diff_theta < 0:
                        rospy.loginfo("rotating clockwise")
                        # resp = self._action_clients["basic_move"](action=MovoBasicMoveActionRequest.ROTATE_CLOCKWISE, data=math.fabs(diff_theta) / math.pi * 180)
                        self._robot.base_turn(angular_distance = degree, clockwise = True)
                    elif diff_theta > 0:
                        rospy.loginfo("rotating anticlocwise")
                        # resp = self._action_clients["basic_move"](action=MovoBasicMoveActionRequest.ROTATE_ANTICLOCKWISE, data=diff_theta / math.pi * 180)
                        self._robot.base_turn(angular_distance = degree, clockwise = False)
                    rospy.sleep(1.0)
                    self._person_obs.start_detection()

                elif self.cfg["approach_person"] and math.fabs(diff_dist) > self.cfg["dist_diff_threshold"]:
                    # distance control

                    # calculate target dist
                    diff_dist = diff_dist * self.cfg["dist_p"]
                    # clamp. Important for safety
                    if diff_dist > self.cfg["clamp_dist_p"]:
                        diff_dist = self.cfg["clamp_dist_p"]
                    if diff_dist < self.cfg["clamp_dist_n"]:
                        diff_dist = self.cfg["clamp_dist_n"]

                    #self._robot_goal.position.x = self._robot_goal.position.x + self._person_location_camera.position.z * math.cos(new_theta)
                    #self._robot_goal.position.y = self._robot_goal.position.y + self._person_location_camera.position.z * math.sin(new_theta)
                    rospy.loginfo("camera_z {}, diff_dist {}".format(person_location_camera.position.z, diff_dist))

                    #rot = tf.transformations.quaternion_from_euler(0, 0, new_theta)
                    #self._robot_goal.orientation.x = rot[0]
                    #self._robot_goal.orientation.y = rot[1]
                    #self._robot_goal.orientation.z = rot[2]
                    #self._robot_goal.orientation.w = rot[3]

                    self._person_obs.stop_detection()
                    if diff_dist < 0:
                        # resp = self._action_clients["basic_move"](action = 1, data = math.fabs(diff_dist))
                        self._robot.base_go_straight(distance = - math.fabs(diff_dist))

                    elif diff_dist > 0:
                        # resp = self._action_clients["basic_move"](action = 0, data = diff_dist)
                        self._robot.base_go_straight(distance = diff_dist)
                    rospy.sleep(1.0)
                    self._person_obs.start_detection()

                else:
                    diff_theta = 0
                    diff_dist = 0
                    if not followed:
                        follow_cnt += 1
                        if follow_cnt > 1:
                            followed = True
                            # self._action_clients["speak"].say("{}, How can I help you?".format(self._person_name))
                            rospy.sleep(0.5)
                            break

            self._lock.acquire()
            self._person_detected = False
            self._lock.release()
            self._event.clear()
            self._event.wait(timeout = 2.0)

        return lost

class MoveToPersonAsync():
    def __init__(self, person_obs, robot, done_cb):
        self._person_obs = person_obs
        self._robot = robot
        self._done_cb = done_cb
        self._move_to_person_thread = None

        self.cfg = {
            "move_to_initial_location" : True,
            "approach_person" : True,
            "desired_dist" : 1.0,
            "pixel_diff_threshold": 100,
            "theta_diff_threshold" : 0.2,
            "dist_diff_threshold" : 0.2,
            "center_x" : 480,
            "heading_p" : -math.pi / (180.0 * 20),
            "dist_p" : 1,
            "clamp_dist_p" : 0.3,
            "clamp_dist_n" : -0.3,
            "alpha" : 0.2,
            "search_wait_duration" : 12.0
        }

    def configure(self, cfg):
        self.cfg.update(cfg)

    def set_target(self, person_name, person_location):
        self._person_name = person_name
        self._person_location = person_location

    def start(self):
        if self._move_to_person_thread is not None:
            rospy.loginfo("MoveToPersonAsync: thread already started")
            return

        self._move_to_person_thread = MoveToPersonThread(self._person_obs, self._robot, self._on_thread_finish)
        self._move_to_person_thread.set_cfg(self.cfg)
        self._move_to_person_thread.set_target(self._person_name, self._person_location)
        self._move_to_person_thread.start()

    def cancel(self):
        if self._move_to_person_thread is not None:
            self._move_to_person_thread.cancel()
            self._move_to_person_thread.join()
            self._move_to_person_thread = None

    def _on_thread_finish(self, lost):
        self._move_to_person_thread = None
        self._done_cb(lost)




'''
    Legacy code:  use PCL


    def _calculate_distance(self, pose1, pose2):
        dist = math.sqrt((pose1.position.x - pose2.position.x)**2 + (pose1.position.y - pose2.position.y)**2)
        return dist

    def person_obs_callback(self, person_name, person_location):
        if person_name == self._person_name:
            dist = self._calculate_distance(person_location, self._person_location)
            if dist < self._threshold:
                self._person_location = person_location
                self._person_detected = True
                self._event.set()
            else:
                dbg_print("dist {} > threshold, skipping observation".format(dist))

    def _move_near_initial_location(self):
        rospy.loginfo("MovetoPerson: start to move near initial location")
        self._move_near_cb(self._person_location, distance = 3.0)

    def _follow(self):
        self._person_obs.register_callback(self.person_obs_callback)

        self._person_detected = False
        self._event.wait(timeout = 2.0) # wait two second
        while not self._cancel:
            if not self._person_detected:
                rospy.logwarn("MoveToPerson: person lost, abort")
                break

            self._move_near_cb(self._person_location, distance = 1.5)
            self._person_detected = False
            self._event.wait(timeout = 2.0) # wait two second

        self._person_obs.unregister_callback(self.person_obs_callback)
'''