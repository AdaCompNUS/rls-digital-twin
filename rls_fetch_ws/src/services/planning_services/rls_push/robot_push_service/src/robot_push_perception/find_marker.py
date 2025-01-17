import rospy
from fetch_api import Head
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import JointState
import math


class MarkerFinder(object):
    def __init__(self, target_pixel=[285, 65]):
        self.head = Head()
        self.ready = False
        self.received = False
        self.pixel_pos = None
        self.current_pan = 0
        self.current_tilt = 0
        self.inc = 5.0 / 180.0 * math.pi
        self.target_pixel = target_pixel
        self.threshold = 8 ## goal region within 10 pixels distance
        #self.start_sub()
        self.rate = rospy.Rate(0.25)

    def move(self, x=0, y=0):
        ''' move the head so that the image shift in x and/or y direction
            0: no movement
            1: positive direction
            -1: negative direction
        '''
        pan = x / 180.0 * math.pi + self.current_pan
        tilt = - y / 180.0 * math.pi + self.current_tilt
        self.head.pan_tilt(pan, tilt, 0.1)
        #self.head.pan_tilt(0, tilt, 0.1)

    def joint_callback(self, data):
        if len(data.position) > 8:
            self.current_pan = data.position[4]
            self.current_tilt = data.position[5]

    def callback(self, data):
        if self.ready:
            self.pixel_pos = data.point
            self.received = True

    def start_sub(self):
        self.joint_sub = rospy.Subscriber('/joint_states', JointState, self.joint_callback)
        self.pixel_sub = rospy.Subscriber('/aruco_simple/pixel', PointStamped, self.callback)

    def close_sub(self):
        self.joint_sub.unregister()
        self.pixel_sub.unregister()

    def find_marker(self):
        self.start_sub()
        self.ready = True
        print 'current pan is', self.current_pan
        self.head.pan_tilt(0, 0.95, 2) ## preset the head tilt angle
        rospy.sleep(2)

        ### marker is not in view
        tilt_inc = 0
        angle = 20.0
        while not self.received and not rospy.is_shutdown():
            tilt_inc -= angle / 180.0 * math.pi
            #self.head.pan_tilt(self.current_pan, Head.MAX_TILT + tilt_inc, 2.5)
            self.head.pan_tilt(self.current_pan, self.current_tilt + tilt_inc, 2.5)
            self.rate.sleep()

        ### move to the right viewpoint
        x_dir = 0
        y_dir = 0
        while not rospy.is_shutdown():
            curr_x = self.pixel_pos.x
            curr_y = self.pixel_pos.y
            diff_x = self.target_pixel[0] - curr_x
            diff_y = self.target_pixel[1] - curr_y

            print curr_x, curr_y
            print diff_x, diff_y
            if abs(diff_x) < self.threshold and abs(diff_y) < self.threshold:
                break

            x_scale = min(abs(diff_x) / 20.0, 5)
            x_scale = max(x_scale, 2)
            y_scale = min(abs(diff_y) / 20.0, 5)
            y_scale = max(y_scale, 2)

            #print 'scales are'
            #print x_scale, y_scale

            if abs(diff_x) < self.threshold:
                x_dir = 0
            elif diff_x > 0:
                x_dir = 1
            else:
                x_dir = -1

            if abs(diff_y) < self.threshold:
                y_dir = 0
            elif diff_y > 0:
                y_dir = 1
            else:
                y_dir= -1

            self.move(x_dir * x_scale, y_dir * y_scale)
            self.rate.sleep()

        self.ready = False
        self.received = False
        self.close_sub()


if __name__=='__main__':
    rospy.init_node('test')
    mf = MarkerFinder()
    mf.find_marker()
    raw_input('dfd')

