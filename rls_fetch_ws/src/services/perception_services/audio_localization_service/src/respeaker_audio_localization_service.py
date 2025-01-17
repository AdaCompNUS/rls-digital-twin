#!/usr/bin/env python

import usb.core
import usb.util
import time
import rospy
from std_msgs.msg import Int32

from tuning import Tuning

class AudioLocalizationService():
    def __init__(self):
        self._rate = 2 #hz

        self._pub = rospy.Publisher("rls_perception_service/audio_localization", Int32, queue_size=10)
        dev = usb.core.find(idVendor=0x2886, idProduct=0x0018)
        self._mic_tuning = Tuning(dev)
        self._timer = rospy.Timer(rospy.Duration(1.0 / self._rate), self.on_timer)

    def on_timer(self, event):
        try:
            is_voice = self._mic_tuning.is_voice()
            if is_voice:
                dir = self._mic_tuning.direction
                self._pub.publish(int(dir))
            else:
                rospy.logdebug("not voice")
        except Exception as e:
            rospy.logwarn(e)

if __name__ == '__main__':
    rospy.init_node("AudioLocalizationService")
    service = AudioLocalizationService()
    rospy.spin()
