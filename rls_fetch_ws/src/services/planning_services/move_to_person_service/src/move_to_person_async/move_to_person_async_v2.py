#!/usr/bin/env python

import rospy
import threading
import tf
import math
import json
import numpy as np
import soundfile as sf
import time
from speech_recognition_msgs.srv import SpeechRecognition

from rls_perception_msgs.srv import *
from rls_perception_msgs.msg import *

class MoveToPersonThread(threading.Thread):
    def __init__(self, person_obs, audio_obs, speaker_recog_srv, audio_local_srv, robot, robot_speaker, status_cb):
        threading.Thread.__init__(self)
        self._event = threading.Event()
        self._status_cb = status_cb
        self._person_obs = person_obs
        self._audio_obs = audio_obs
        self._speaker_recog_srv = speaker_recog_srv
        self._audio_local_srv = audio_local_srv
        self._robot = robot
        self._robot_speaker = robot_speaker
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
            success = self._move_to_initial_location()
            if not success:
                raise RuntimeError("MoveToPerson: failed to move to initial location")
            self._person_obs.register_camera_callback(self.person_camera_callback)
            person_detected = self._detect_person()

            cnt = 0
            while not person_detected and cnt < self.cfg['max_audio_trials']:
                self._localize_person_audio()
                person_detected = self._detect_person()
                cnt += 1

            if person_detected:
                lost = self._follow()
                self._status_cb(lost)
            else:
                self._status_cb(True)
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

    def _localize_person_audio(self, timeout=10):
        self._robot_speaker.say("{}, where are you?".format(self._person_name))

        start_time = time.time()
        angle = None
        while time.time() - start_time <= timeout:
            try:
                audio = self._audio_obs()
                rt_value = json.loads(audio.result.transcript[0])
                rt_value = np.array(rt_value)
                quantity = int(audio.result.confidence[0])
                rt_value = np.reshape(rt_value, (quantity, 2730, 4))
                print(rt_value.shape)
                current_data = rt_value.astype(np.float64)
                current_data = current_data / math.pow(2, 15)
            except Exception as e:
                print(e)
                continue

            # speaker recog first
            speaker_recog_audio = current_data.reshape(1, -1, 4)
            speaker_recog_audio = np.mean(speaker_recog_audio, axis=-1)
            audio_string = json.dumps(speaker_recog_audio.tolist())
            req = SpeakerRecognitionRequest()
            req.action = SpeakerRecognitionRequest.RECOGNIZE
            req.type = SpeakerRecognitionRequest.AUDIO
            req.audio_string = audio_string
            resp = self._speaker_recog_srv(req)
            if resp.speaker != self._person_name or resp.score < self.cfg["speaker_recog_threshold"]:
                print("Not {}'s voice!".format(self._person_name))
                continue

            print("Recognized {}'s voice with score {}".format(self._person_name, resp.score))
            audio_string = json.dumps(current_data.tolist())
            req = AudioLocalizationRequest()
            req.audio_string = audio_string
            req.quantity = quantity
            resp = self._audio_local_srv(req)
            # print(resp.angle)

            # if resp.angle == last_angle:
            #     continue

            angle = resp.angle
            break

        if angle is not None:
            angle_to_rotate = angle - 180
            if angle_to_rotate < 0:
                self._robot.base_turn(-angle_to_rotate, clockwise=True)
                print(-angle_to_rotate, "clockwise=True")
            else:
                self._robot.base_turn(angle_to_rotate, clockwise=False)
                print(angle_to_rotate, "clockwise=False")
        # last_angle = resp.angle

    def _move_to_initial_location(self):
        self._person_obs.stop_detection()
        rospy.loginfo("MoveToPerson: start to move to initial location")
        # resp = self._action_clients["nav_to"](cancel_goal = False, move_to_home = False, goal = self._person_location)
        resp = self._robot.base_nav(self._person_location)
        self._person_obs.start_detection()
        return resp

    def _detect_person(self):
        self._person_detected = False
        self._event.clear()
        self._event.wait(timeout = self.cfg["search_wait_duration"])

        self._lock.acquire()
        person_detected = self._person_detected
        self._lock.release()

        if not person_detected:
            rospy.logwarn("MoveToPersonThread: can't find person at initial position")
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


                if abs(diff_pixel) >= self.cfg["pixel_diff_threshold"] and math.fabs(diff_theta) >= self.cfg["theta_diff_threshold"]:
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

                elif math.fabs(diff_dist) > self.cfg["dist_diff_threshold"]:
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

class MoveToPersonAsyncV2():
    def __init__(self, person_obs, robot, robot_speaker, status_cb):
        self._person_obs = person_obs
        self._robot = robot
        self._robot_speaker = robot_speaker
        self._status_cb = status_cb
        self._move_to_person_thread = None

        self.cfg = {
            "desired_dist" : 1.2,
            "pixel_diff_threshold": 80,
            "theta_diff_threshold" : 0.1,
            "dist_diff_threshold" : 0.2,
            "center_x" : 480,
            "heading_p" : -math.pi / (180.0 * 20),
            "dist_p" : 1,
            "clamp_dist_p" : 0.3,
            "clamp_dist_n" : -0.3,
            "alpha" : 0.2,
            "search_wait_duration" : 5.0,
            "speaker_recog_threshold" : -1.05,
            "max_audio_trials" : 2
        }

        self.audio_service = rospy.ServiceProxy('/respeaker/get_trans_audio', SpeechRecognition)
        self.speaker_recog_service = rospy.ServiceProxy('rls_perception_services/speaker_recognition_service', SpeakerRecognition)
        self.audio_local_service = rospy.ServiceProxy('rls_perception_services/audio_localization_service', AudioLocalization)

    def configure(self, cfg):
        self.cfg.update(cfg)

    def set_target(self, person_name, person_location):
        self._person_name = person_name
        self._person_location = person_location

    def start(self):
        if self._move_to_person_thread is not None:
            rospy.loginfo("MoveToPersonAsync: thread already started")
            return

        self._move_to_person_thread = MoveToPersonThread(self._person_obs, self.audio_service, self.speaker_recog_service, self.audio_local_service,
                                                            self._robot, self._robot_speaker, self.thread_status_cb)
        self._move_to_person_thread.set_cfg(self.cfg)
        self._move_to_person_thread.set_target(self._person_name, self._person_location)
        self._move_to_person_thread.start()

    def cancel(self):
        if self._move_to_person_thread is not None:
            self._move_to_person_thread.cancel()
            self._move_to_person_thread.join()
            self._move_to_person_thread = None

    def thread_status_cb(self, lost):
        self._move_to_person_thread = None
        self._status_cb(lost)
