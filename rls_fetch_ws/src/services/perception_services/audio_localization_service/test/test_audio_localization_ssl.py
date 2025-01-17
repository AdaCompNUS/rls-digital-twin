#!/usr/bin/env python

import rospy
import numpy as np
import math
from speech_recognition_msgs.srv import SpeechRecognition

from audio_localization.audio_localizer import AudioLocalizer

def get_audio_data():
    try:
        audio = respeaker_audio_client()
        rt_value = np.frombuffer(audio.result.transcript[0].encode('utf-8'), dtype=np.int16)
        quantity = int(audio.result.confidence[0])
        print(rt_value.shape)
        rt_value = np.reshape(rt_value, (quantity, 2730, 4))
        current_data = rt_value.astype(np.float64)
        current_data = current_data / math.pow(2, 15)
        # print(current_data)
        return current_data, quantity
    except rospy.ServiceException as e:
        print("Service call failed: {}".format(e))

respeaker_audio_client = rospy.ServiceProxy('/respeaker/get_trans_audio', SpeechRecognition)
localizer = AudioLocalizer()

while not rospy.is_shutdown():
    audio_data, quantity = get_audio_data()
    res = localizer.localize(audio_data, quantity)
    print(res)