import rospy
from geometry_msgs.msg import TwistStamped, Pose
from sensor_msgs.msg import JointState
import tf
from tf.listener import TransformListener
from math import sqrt, pow, sin, cos, atan2

class Armm(object):

    def __init__(self):
        self._tf_listener = TransformListener()
        self.P_gain = 5
        self.D_gain = 10
        self.max_vel = 0.1
        self.pose = Pose()
        self.pose_sub = rospy.Subscriber('joint_states', JointState, self.tool_pose_callback)
        self.vel_pub = rospy.Publisher('/arm_controller/cartesian_twist/command', TwistStamped)
        self.tolerance = 0.005
        self.valid_pose_received = False
        self.rate = rospy.Rate(50)

    def tool_pose_callback(self, data):

        time = 0
        trans = None
        rot = None
        while not rospy.is_shutdown():
            try:
                time = self._tf_listener.getLatestCommonTime('base_link', 'gripper_link')
                trans, rot = self._tf_listener.lookupTransform('base_link', 'gripper_link', time)
                break
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                continue
                #rospy.logerr(e)
            self.rate.sleep()

        self.valid_pose_received = True
        self.pose.position.x = round(trans[0], 4)
        self.pose.position.y = round(trans[1], 4)
        self.pose.position.z = round(trans[2], 4)

    def move_by_xy(self, dx=0, dy=0):
        print '======================='
        vel_msg = TwistStamped()
        vel_msg.header.frame_id = 'base_link'
        goal_x = self.pose.position.x + dx
        goal_y = self.pose.position.y + dy
        dx_prev = dx
        dy_prev = dy

        while not rospy.is_shutdown():
            dxx = goal_x - self.pose.position.x
            dyy = goal_y - self.pose.position.y
            err = sqrt(pow(dxx, 2) + pow(dyy, 2))
            #print dxx, dyy

            ## PD Control
            vel_x_PD = self.P_gain * cos(atan2(dyy, dxx)) * abs(dxx) + self.D_gain * (dxx - dx_prev)
            vel_y_PD = self.P_gain * sin(atan2(dyy, dxx)) * abs(dyy) + self.D_gain * (dyy - dy_prev)

            dx_prev = dxx
            dy_prev = dyy

            if abs(vel_x_PD) > abs(self.max_vel * cos(atan2(dyy, dxx))):
                vel_x = self.max_vel * cos(atan2(dyy, dxx))
                print 'X: constant'
            else:
                vel_x = vel_x_PD
                print 'X: PD'

            if abs(vel_y_PD) > abs(self.max_vel * sin(atan2(dyy, dxx))):
                vel_y = self.max_vel * sin(atan2(dyy, dxx))
                print 'Y: constant'
            else:
                vel_y = vel_y_PD
                print 'Y: PD'

            if err < self.tolerance:
                break
            vel_msg.twist.linear.x = vel_x
            vel_msg.twist.linear.y = vel_y
            self.vel_pub.publish(vel_msg)
            self.rate.sleep()

        vel_msg.twist.linear.x = 0
        vel_msg.twist.linear.y = 0
        self.vel_pub.publish(vel_msg)
        rospy.sleep(0.5)

    def move_by_z(self, dz=0):
        print '======================='
        vel_msg = TwistStamped()
        vel_msg.header.frame_id = 'base_link'
        goal_z = self.pose.position.z + dz
        dz_prev = dz

        while not rospy.is_shutdown():
            dzz = goal_z - self.pose.position.z
            vel_z_PD = self.P_gain * dzz + self.D_gain * (dzz - dz_prev)
            dz_prev = dzz
            #print dzz
            if abs(dzz) < self.tolerance:
                break
            if abs(vel_z_PD) > abs(self.max_vel):
                print 'Z: constant'
                if dzz > 0:
                    vel_z = self.max_vel
                else:
                    vel_z = - self.max_vel
            else:
                print 'Z: PD'
                vel_z = vel_z_PD

            vel_msg.twist.linear.z = vel_z
            self.vel_pub.publish(vel_msg)
            self.rate.sleep()

        vel_msg.twist.linear.z = 0
        self.vel_pub.publish(vel_msg)
        rospy.sleep(0.5)










