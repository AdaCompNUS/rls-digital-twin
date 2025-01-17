#!/usr/bin/env python
import rospy
from sensor_msgs.msg import PointCloud2

def relay_pointcloud(data):
    if not data == None:
        #print 'received'
        pcl_pub.publish(data)


if __name__=='__main__':
    rospy.init_node('relay_pcl')
    pcl_sub = rospy.Subscriber('/head_camera/depth_registered/points', PointCloud2, relay_pointcloud)
    pcl_pub = rospy.Publisher('pointcloud_local', PointCloud2, queue_size=5)
    rospy.spin()
