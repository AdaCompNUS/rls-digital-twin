#!/usr/bin/env python

import rospy
from std_msgs.msg import String
import os.path as osp
import json

from rls_perception_msgs.srv import *
from speaker_recognition.speaker_recognizer import SpeakerRecognizer

class SpeakerRecognitionService():
    def __init__(self):
        self.recognizer = SpeakerRecognizer()

        ## Expose service and publisher
        self._service = rospy.Service('rls_perception_services/speaker_recognition_service', SpeakerRecognition, self._handle_req)
        rospy.loginfo("SpeakerRecognitionService: inited")

    def _handle_req(self, req):
        resp = SpeakerRecognitionResponse()

        if req.action == SpeakerRecognitionRequest.RECOGNIZE:
            if req.type == SpeakerRecognitionRequest.AUDIO:
                audio_feat = json.loads(req.audio_string)
                speaker, score = self.recognizer.recognize_audio(audio_feat)
            elif req.type == SpeakerRecognitionRequest.FILE:
                speaker, score = self.recognizer.recognize_file(req.filename)
            resp.speaker = speaker
            resp.score = score
        elif req.action == SpeakerRecognitionRequest.REMEMBER_NEW_SPEAKER:
            filename = req.filename
            res = self.recognizer.remember_new_speaker(filename)
            if not res:
                rospy.logerr("SpeakerRecognitionService: remember_new_speaker failed!!!")
            resp.success = res

        return resp

if __name__ == '__main__':
    rospy.init_node('speaker_recognition_service')
    person_detection_srv = SpeakerRecognitionService()
    rospy.spin()