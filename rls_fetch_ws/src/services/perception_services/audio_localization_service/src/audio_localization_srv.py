#!/usr/bin/env python

import rospy
from std_msgs.msg import String
import json
import numpy as np

from rls_perception_msgs.srv import *
from audio_localization.audio_localizer import AudioLocalizer

class AudioLocalizationService():
    def __init__(self):
        self.localizer = AudioLocalizer()

        ## Expose service and publisher
        self._service = rospy.Service('rls_perception_services/audio_localization_service', AudioLocalization, self._handle_req)
        rospy.loginfo("AudioLocalizationService: inited")

    def _handle_req(self, req):
        audio_data = np.array(json.loads(req.audio_string))
        quantity = req.quantity
        result = self.localizer.localize_audio(audio_data, quantity)
        if(result["changeFlag"]):
            print(result["location"])

        return AudioLocalizationResponse(int(result["location"]))

if __name__ == '__main__':
    rospy.init_node('AudioLocalizationService')
    person_detection_srv = AudioLocalizationService()
    rospy.spin()