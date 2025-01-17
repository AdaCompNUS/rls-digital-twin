#!/usr/bin/env python
import rospy
import tf
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, Float64
from active_subscriber import ActiveSubscriber
from rls_push_msgs.srv import *


class RobotPlaneTF(object):
    def __init__(self):
        self.listened = False
        self.br = tf.TransformBroadcaster()
        self.pose = PoseStamped()
        self.rate = rospy.Rate(50)
        self.trans = (0, 0, 0)
        self.qat = (0, 0, 0, 1)
        self.pose_sub = rospy.Subscriber('/aruco_simple/plane_frame', PoseStamped, self.callback)
        self.update_tf_ser = rospy.Service('update_tf', UpdateTF, self.update_transform)
        self.static_ser = rospy.Service('update_static', UpdateStatic, self.update_table)
        #self.update_sub = rospy.Subscriber('/update_tf', Bool, self.update_transform)
        #self.static_sub = rospy.Subscriber('/update_static', Float64, self.update_table)

        ## define static transform
        self.static_tf = [['plane', 'left_out', (-0.7, 0.035, -0.24), (-0.707, 0, 0, 0.707)],
                          ['plane', 'left_edge', (-0.7, 0.02, -0.10), (-0.707, 0, 0, 0.707)],
                          ['left_edge', 'right_edge', (0, -1.0, 0), (0, 0, 0, 1)],
                          ['left_out', 'left_in', (0.6, 0, 0), (0, 0, 0, 1)],
                          ['left_out', 'right_out', (0, -1.3, 0), (0, 0, 0, 1)],
                          ['left_out', 'right_in', (0.6, -1.3, 0), (0, 0, 0, 1)]
                          ]

        print 'tf updater is running'
        self.sendTransform()

    def callback(self, data):
        if not data == None:
            self.pose = data
            #self.listened = True

    def update_table(self, req):
        dz = req.dz.data
        self.static_tf = [['plane', 'left_out', (-0.7, 0.035+dz, -0.24), (-0.707, 0, 0, 0.707)],
                          ['plane', 'left_edge', (-0.7, 0.02, -0.10), (-0.707, 0, 0, 0.707)],
                          ['left_edge', 'right_edge', (0, -1.0, 0), (0, 0, 0, 1)],
                          ['left_out', 'left_in', (0.6, 0, 0), (0, 0, 0, 1)],
                          ['left_out', 'right_out', (0, -1.3, 0), (0, 0, 0, 1)],
                          ['left_out', 'right_in', (0.6, -1.3, 0), (0, 0, 0, 1)]
                          ]
        return UpdateStaticResponse(Bool())



    def update_transform(self, req):
        print 'update transform'
        ### update when relative pose between camera and marker changes
        #self.listened = False
        #while not self.listened:
        #    self.rate.sleep()
        pose = self.pose.pose
        self.trans = (pose.position.x, pose.position.y, pose.position.z)
        self.qat = (pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
        print 'tf updated'
        return UpdateTFResponse(Bool())

    def sendTransform(self):
        while not rospy.is_shutdown():
            for stf in self.static_tf:
                self.br.sendTransform(stf[2], stf[3], rospy.Time.now(), stf[1], stf[0])

            self.br.sendTransform(self.trans, self.qat, rospy.Time.now(), 'plane', 'head_camera_rgb_optical_frame')
            self.rate.sleep()


if __name__=='__main__':
    rospy.init_node('static')
    rt = RobotPlaneTF()









