#! /usr/bin/env python

import fetch_api
import rospy
from moveit_commander import MoveGroupCommander
from geometry_msgs.msg import PoseStamped
from copy import deepcopy
from robot_config import *
import threading

def wait_for_time():
    """Wait for simulated time to begin.
    """
    while rospy.Time().now().to_sec() == 0:
        pass


def main():
    rospy.init_node('arm_demo')
    pose_thread = PoseThread()
    pose_thread.start()

    # wait_for_time()
    argv = rospy.myargv()

    torso = fetch_api.Torso()
    arm = fetch_api.Arm()
    # group = MoveGroupCommander('arm')

    ## Test torso
    torso.set_height(fetch_api.Torso.MAX_HEIGHT)
    rospy.sleep(1)

    ## Test named target
    #arm.move_to_named_target(group, 'pre_push')
    # rospy.sleep(1)

    ## Test pose target
    pose = PoseStamped()
    # pose.header.frame_id = 'base_link'
    # pose.pose.position.x = 0.4285
    # pose.pose.position.y = 0.41251
    # pose.pose.position.z = 1.1005
    # pose.pose.orientation.x = -0.063
    # pose.pose.orientation.y = 0.700
    # pose.pose.orientation.z = 0.0644
    # pose.pose.orientation.w = 0.7081
    pose.header.frame_id = 'my_tag'
    pose.pose.position.x = 0
    pose.pose.position.y = 0
    pose.pose.position.z = 0.2
    pose.pose.orientation.x = 0
    pose.pose.orientation.y = 0
    pose.pose.orientation.z = 0
    pose.pose.orientation.w = 1

    # while not rospy.is_shutdown():
    pose_thread.set_pose(pose)
    
    raw_input('wait')
    arm.move_to_pose(pose, plan_only=True, replan=True, execution_timeout=10)
    rospy.sleep(1)
    # arm.clean_exit()    pose_pub.publish(pose)

    # Test Cartesian Straight line path
    # while not rospy.is_shutdown() and not arm.is_pose_received():
    #    continue

    # for j in range(2):
    #     arm.move_in_cartesian(dz = -0.1)
    #     arm.move_in_cartesian(dz = 0.1)
    #     arm.move_in_cartesian(dx=0.15, dy = -0.1)
    # for k in range(2):
    #     arm.move_in_cartesian(dz = -0.1)
    #     arm.move_in_cartesian(dz = 0.1)
    #     arm.move_in_cartesian(dx=-0.15, dy = 0.1)

    # print('tuck')
    # arm.move_to_joint_goal(HOME_JOINTS, replan=True)
    # rospy.sleep(1)
    # torso.set_height(fetch_api.Torso.MIN_HEIGHT)

    pose_thread.stop()
    pose_thread.join()
    print('exit')

class PoseThread(threading.Thread):
    def __init__(self):
        super(PoseThread, self).__init__()
        self.pose_pub = rospy.Publisher('arm_goal_pose', PoseStamped, queue_size=2)
        self.pose = PoseStamped()
        self.to_exit = False
        self.lock = threading.Lock()
    
    def run(self):
        to_exit = False
        while not to_exit:
            self.lock.acquire()
            self.pose_pub.publish(self.pose)
            to_exit = self.to_exit
            self.lock.release()
            rospy.sleep(0.1)
    
    def set_pose(self, pose):
        with self.lock:
            self.pose = pose
    
    def stop(self):
        with self.lock:
            self.to_exit = True

if __name__ == '__main__':
    main()
