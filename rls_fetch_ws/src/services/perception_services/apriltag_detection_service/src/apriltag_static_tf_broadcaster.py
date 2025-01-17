#!/usr/bin/env python

import rospy
import tf

class AprilTagStaticTfBroadcaster():
    def __init__(self):
        self.tf_listener = tf.TransformListener()
        self.tf_br = tf.TransformBroadcaster()
        self.pos = (0, 0, 0)
        self.quat = (0, 0, 0, 1)
        self.rate = rospy.Rate(10)
        self.camera_frame = rospy.get_param('apriltag_camera_frame')

    def run(self):
        rospy.loginfo('AprilTagStaticTfBroadcaster, running')
        while not rospy.is_shutdown():
            now = rospy.Time.now()
            try:
                self.tf_listener.waitForTransform('/base_link', '/my_tag', now, rospy.Duration(1))
                self.pos, self.quat = self.tf_listener.lookupTransform('/base_link', '/my_tag', now)
            except Exception as e:
                rospy.logwarn('AprilTagStaticTfBroadcaster, tf not received')

            # rospy.loginfo('AprilTagStaticTfBroadcaster, tf sent')
            self.tf_br.sendTransform(self.pos, self.quat, now, '/my_tag_static', '/base_link')
            self.rate.sleep()

if __name__ == '__main__':
    rospy.init_node('apriltag_static_tf_br')
    apriltag_static_tf_br = AprilTagStaticTfBroadcaster()
    apriltag_static_tf_br.run()
