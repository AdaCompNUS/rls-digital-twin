import rospy
from std_msgs.msg import ColorRGBA
from std_msgs.msg import Int16

rospy.init_node('Process_server')
Color_pub = rospy.Publisher('status_led', ColorRGBA, queue_size=10)

rate = rospy.Rate(1)
t = 0
while 1:
    color = ColorRGBA()
    color.r = 2
    Color_pub.publish(color)
    rate.sleep()


