#!/usr/bin/env python

import rospy
import json
import cv2
import numpy as np
import base64
import threading
import time
import requests
import json
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import *
from geometry_msgs.msg import Pose
from face_recognition_service.srv import *
from rls_perception_msgs.srv import *
import math

## Defines
PUBLISH_DBG_IMG = True

class PublishDbgImgThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self._cancel = False
        self._lock = threading.Lock()
        # self._pub = rospy.Publisher("rls_perception_services/person_detector/debug", Image, queue_size = 2)
        self._pub = rospy.Publisher("/rls_robot_app/ingress_demo/image_view", Image, queue_size=1)
        self._rate = rospy.Rate(0.5)  # 0.5hz
        self.daemon = True
        self._img = None
        self._to_pub = False
        self._bridge = CvBridge()

    def set_image(self, image, faces):
        face_locations = []
        face_names = []

        if len(faces) > 0:
            self._to_pub = True
        else:
            self._to_pub = False
            return

        for face in faces:
            face_locations.append(face["location"])
            face_names.append(face["id"])

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):

            # Draw a box around the face
            cv2.rectangle(image, (left - 10 , top - 10), (right + 10, bottom + 10), (0, 0, 255), 2)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(image, name, (left + 6, top - 15), font, 1, (255, 255, 255), 2)

        # Display the resulting image
        image_msg = self._bridge.cv2_to_imgmsg(image, encoding="bgr8")
        with self._lock:
            self._img = image_msg

    def run(self):
        to_cancel = self._cancel
        while not to_cancel:
            with self._lock:
                img = self._img
                to_cancel = self._cancel

            if img is not None and self._to_pub:
                self._pub.publish(img)

            self._rate.sleep()

class PersonDetectionThread(threading.Thread):
    def __init__(self, image_listener, facerec_client, detected_cb, publish_dbg_img_thread = None):
        threading.Thread.__init__(self)
        self._image_listener = image_listener
        self._facerec_client = facerec_client
        self._detected_cb = detected_cb
        #self.tf = TransformListener()
        #self._publish = pub_callback
        #self._event = event
        self.bridge = CvBridge()
        self._lock = threading.Lock()
        self._canceled = False
        self.daemon = True

        if PUBLISH_DBG_IMG:
            self._pub_dbg_img_thread = publish_dbg_img_thread

    def run(self):
        rospy.loginfo("PersonDetectionThread: thread start")
        self._lock.acquire()
        canceled = self._canceled
        self._lock.release()

        while not canceled:
            # req = FetchImageRequest()
            # req.request = FetchImageRequest.COLOUR_AND_DEPTH
            # start_time = rospy.get_time()
            # resp = self._image_listener(req)
            # end_time = rospy.get_time()
            # # rospy.loginfo("PersonDetectionThread: request image takes {} seconds".format(end_time - start_time))
            # if not resp.success:
            #     rospy.sleep(0.1)
            #     continue

            # image = resp.color
            # depth_image = resp.depth_image
            image = self._image_listener["rgb"]()
            depth_image = self._image_listener["depth"]()
            person_list = self.detect_person_faceonly(image, depth_image)
            # self.detect_person(image)
            if len(person_list) > 0:
                # rospy.loginfo("{} time: {}".format(person_list, image_time))
                rospy.loginfo("{}".format(person_list))
                self._detected_cb(person_list)

            self._lock.acquire()
            canceled = self._canceled
            self._lock.release()

        rospy.loginfo("PersonDetectionThread: thread exit")

    def cancel(self):
        self._lock.acquire()
        self._canceled = True
        self._lock.release()

    def _get_eye_avg(self, eyes):
        eye1 = eyes[0]
        eye2 = eyes[1]
        eye1_x = eye1[0]
        eye1_y = eye1[1]
        eye2_x = eye2[0]
        eye2_y = eye2[1]
        x = int(eye1_x + eye2_x / 2)
        y = int(eye1_y + eye2_y / 2)
        return (x, y)

    def _cal_score(self, nose1, nose2, dist):
        x1 = nose1[0]
        y1 = nose1[1]
        x2 = nose2[0]
        y2 = nose2[1]
        return math.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2)) * dist

    def _find_identity(self, faces, pose):
        min_identity = {"id" : "Unknown"}

        if len(faces) == 0:
            return min_identity, float("inf")

        pose_nose = None
        pose_leye = None
        pose_reye = None

        for part in pose["body_parts"]:
            if part["part_name"] == "Nose":
                pose_nose = (part["x"], part["y"])
            elif part["part_name"] == "LEye":
                pose_leye = (part["x"], part["y"])
            elif part["part_name"] == "REye":
                pose_reye = (part["x"], part["y"])
        # rospy.loginfo("PersonDetectionService: waiting for /rls_robot_obs/movo/fetch_image")

        for face in faces:
            face_nose = face["nose_tip"][0]
            face_leye = self._get_eye_avg(face["left_eye"])
            face_reye = self._get_eye_avg(face["right_eye"])
            cur_score = self._cal_score(face_nose, pose_nose, face["dist"])
            if cur_score < min_score:
                min_score = cur_score
                min_identity = face

        return min_identity, min_score

    def detect_person_faceonly(self, image, depth_image):
        people_list = []
        #input validity check
        if image is None or depth_image is None:
            rospy.logerr("detect_person_faceonly: input invalid")
            return people_list

        ## translate depth image
        # depth_image = self.bridge.imgmsg_to_cv2(depth_image, desired_encoding="passthrough")

        ## translate image
        # image_cv2 = self.bridge.imgmsg_to_cv2(image, "bgr8")
        image_cv2 = image
        retval, buffer = cv2.imencode('.jpg', image_cv2)
        image = base64.b64encode(buffer)
        #print(image_np.shape)

        req = RecognizeFaceRequest()
        req.action = RecognizeFaceRequest.RECOGNIZE_FACE
        req.image = image
        # start_time = time.time()
        #face_rec_response = requests.post("http://192.168.0.32:8080/encode", data={'data': image})
        face_rec_response = self._facerec_client(req)
        # end_time = time.time()
        # rospy.loginfo("face recognition takes {} seconds".format(end_time - start_time))
        faces = json.loads(face_rec_response.result)["faces"]

        if PUBLISH_DBG_IMG:
            self._pub_dbg_img_thread.set_image(image_cv2, faces)

        for face in faces:
            if face["dist"] > 0.5:
                continue

            face_nose = face["nose_tip"][0]

            if depth_image is not None:
                dist = np.nanmean(depth_image[face_nose[1] - 10: face_nose[1] + 10, face_nose[0] - 10: face_nose[0] + 10]) / 1000
                # rospy.loginfo("depth_image sample value: {} {} {} {}".format(depth_image[face_nose[1], face_nose[0]], depth_image[face_nose[1] - 5, face_nose[0] + 5], depth_image[face_nose[1] - 5, face_nose[0] - 5], depth_image[face_nose[1] + 5, face_nose[0]+ 5]))
                #resp = self._pcl_retriever(face_nose[0], face_nose[1])
            else:
                rospy.logwarn("depth image None, skip")
                dist = 0

            person = {"name": face["id"], "score" : face["dist"],
                "camera_pixel": {"x": face_nose[0], "y": face_nose[1], "z": dist}}
            people_list.append(person)

        return people_list

class PersonDetector:
    def __init__(self, image_listener, detected_cb):
        self._detected_cb = detected_cb
        self._lock = threading.Lock()
        self._image_listener = image_listener
        self._camera_callbacks = []
        self._person_detection_thread = None
        self.bridge = CvBridge()
        rospy.loginfo("Waiting for face recognition service")
        rospy.wait_for_service('rls_perception_services/face_recognition')
        self._facerec_client = rospy.ServiceProxy('rls_perception_services/face_recognition', RecognizeFace)
        # self.start_detection()
        if PUBLISH_DBG_IMG:
            self._pub_dbg_img_thread = PublishDbgImgThread()
            self._pub_dbg_img_thread.start()

    def start_detection(self):
        if self._person_detection_thread is not None:
            return
        else:
            self._person_detection_thread = PersonDetectionThread(self._image_listener, self._facerec_client, self._on_person_detected, self._pub_dbg_img_thread)
            self._person_detection_thread.start()

    def stop_detection(self):
        if self._person_detection_thread is not None and self._person_detection_thread.is_alive():
            self._person_detection_thread.cancel()
            self._person_detection_thread.join()
            self._person_detection_thread = None
        else:
            return

    def remember_new_face(self, person_name):
        to_resume_detection = False
        if self._person_detection_thread is not None and self._person_detection_thread.is_alive():
            rospy.loginfo("PersonDetector/remember_new_face: stop detection temporarily")
            self.stop_detection()
            to_resume_detection = True

        rospy.loginfo("PersonDetector/remember_new_face")
        # req = FetchImageRequest()
        # req.request = FetchImageRequest.COLOUR_ONLY
        # start_time = rospy.get_time()
        # resp = self._image_listener(req)
        # end_time = rospy.get_time()
        # # rospy.loginfo("PersonDetectionThread: request image takes {} seconds".format(end_time - start_time))
        # if not resp.success:
        #     rospy.logerr("PersonDetector/remember_new_face: request img failed")
        #     ## resume detection before quit
        #     if to_resume_detection:
        #         self.start_detection()
        #     return False

        ## translate image
        image = self._image_listener["rgb"]()
        # image = self.bridge.imgmsg_to_cv2(resp.color, "bgr8")
        retval, buffer = cv2.imencode('.jpg', image)
        image = base64.b64encode(buffer)

        req = RecognizeFaceRequest()
        req.action = RecognizeFaceRequest.REMEMBER_NEW_FACE
        req.person_name = person_name
        req.image = image
        resp = self._facerec_client(req)
        res = resp.success
        if res:
            rospy.loginfo("PersonDetector/remember_new_face: new face remembered")
        else:
            rospy.logerr("PersonDetector/remember_new_face: remember new face failed")

        ## resume detection before quit
        if to_resume_detection:
            self.start_detection()

        return res

    def _on_person_detected(self, people_list):
        self._detected_cb(people_list)

        '''
        for p in people_list:
            if p["name"].lower() not in PERSON_LIST:
                continue

            pose = Pose()
            pose.position.x = p["camera_pixel"]["x"]
            pose.position.y = p["camera_pixel"]["y"]
            pose.position.z = p["camera_pixel"]["z"]

            for callback in self._camera_callbacks:
                callback(p['name'].lower(), pose)
        '''


if __name__ == '__main__':
    rospy.init_node('person_detector')
    person_detector = PersonDetector()
    person_detector.start_detection()
    rospy.spin()