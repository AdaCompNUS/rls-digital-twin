import rospy
from sensor_msgs.msg import JointState
from tf.listener import TransformListener
from geometry_msgs.msg import PoseStamped


class ToolPose(object):
    def __init__(self):
        self.pose = PoseStamped()
        self._tf_listener = TransformListener()
        self.pose_pub = rospy.Publisher('tool_pose', PoseStamped, queue_size=10)
        self.joint_sub = rospy.Subscriber('joint_states', JointState, self.tool_pose_callback)

    def tool_pose_callback(self, data):
        ''' get gripper link pose with respect to base_link'''
        time = 0
        trans = None
        rot = None
        self.pose.header.frame_id = 'base_link'
        while not rospy.is_shutdown():
            try:
                time = self._tf_listener.getLatestCommonTime('base_link', 'wrist_roll_link')
                trans, rot = self._tf_listener.lookupTransform('base_link', 'wrist_roll_link', time)
                break
            except (tf.Exception, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                continue

        self.pose.pose.position.x = round(trans[0], 4)
        self.pose.pose.position.y = round(trans[1], 4)
        self.pose.pose.position.z = round(trans[2], 4)
        self.pose.pose.orientation.x = round(rot[0], 4)
        self.pose.pose.orientation.y = round(rot[1], 4)
        self.pose.pose.orientation.z = round(rot[2], 4)
        self.pose.pose.orientation.w = round(rot[3], 4)
        self.pose_pub.publish(self.pose)


if __name__=='__main__':
    rospy.init_node('tool_pose')
    ToolPose()
    rospy.spin()

