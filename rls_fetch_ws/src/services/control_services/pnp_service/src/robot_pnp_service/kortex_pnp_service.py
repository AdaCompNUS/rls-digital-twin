#!/usr/bin/env python

import rospy
from moveit_python import PlanningSceneInterface
import math
from moveit_msgs.msg import Grasp
import tf.transformations as T
import tf2_ros
from geometry_msgs.msg import TransformStamped

from kortex_robot.kortex_robot import KortexRobot
from rls_perception_msgs.srv import *
from rls_control_msgs.srv import *
from robot_pnp_service import PnpService
from kortex_robot.kortex_joint_position_config import *

''' --------- Settings -----------'''
ASK_BEFORE_EXECUTE = False
MOVE_CLOSER_TO_TABLE = False
VIS_GRASP = True

''' --------- Constant ----------'''
BASE_FRAME_NAME = 'base_link'
GRIPPER_LENGTH = 0.2  # metres
_SIDE_PICK_TILT = 15  # degrees
GRIPPER_CLOSE_OPENING = 0.05  # between 0 and 0.16
PLANNING_SCENE_SYNC_WAIT_TIME = 1  # second
BUFFER_HEIGHT = 0.01
PLACE_BUFFER_HEIGHT = 0.2  # meter
APPROACH_HEIGTH = 0.1
DEPART_HEIGHT = 0.1
TOP_PLACE_APPROACH_DIST = 0.05
READY_POSE_HEAD_TILT = 42
READY_POSE_FINAL_HEAD_TILT = 50
CLOSE_GRIPPER_EFFORT = 70

DIST_TO_MOVE_TO_TABLE = 0.15  # meter

''' ---------- Defines ----------- '''
FAIL_TO_MOVE_TO_READY_POSE = 1

''' ---------- Code -------------- '''

class KortexPnpService(PnpService):
    def __init__(self):
        PnpService.__init__(self)

        self.robot = KortexRobot()
        if self.robot is None:
            raise RuntimeError("PNPServer: Failed to init PNPServer")

        # planning scene
        self._planning_frame = 'base_link'
        self._scene = PlanningSceneInterface('base_link')

        # config
        self._depart_height = DEPART_HEIGHT
        self._depart_height_min = 0.15
        self._depart_height_max = 0.20
        self._top_grasp_approach_height = APPROACH_HEIGTH
        self._plan_attempt = 10
        self._sync_wait_time = PLANNING_SCENE_SYNC_WAIT_TIME
        self._top_place_approach_dist = TOP_PLACE_APPROACH_DIST
        self._table_height = 0

        # advertise service
        rospy.Service('rls_control_services/kortex/pnp', PickPlace, self._handle_pnp_req)
        rospy.loginfo('PickNPlace Service is Ready!')

    def _get_ready(self, req):
        '''
        Enter pick_n_place pose.
        @return, boolean, return true if successful, return false otherwise.
        '''

        # check whether it is already in ready state
        if self._ready:
            rospy.loginfo("kortex_pnp_service/get_ready: already ready")
            return True

        # move arm to ready_pick_pose
        res = self._move_to_ready_pick_pose()
        if not res:
            rospy.logerr("kortex_pnp_service/get_ready: failed")
            return False

        self._ready = True
        rospy.loginfo('kortex_pnp_service: get_ready succeeded')
        return True

    def _clear_obstacles(self, all_obj=False):
        '''
        Remove collision objects
        @param all_obj, boolean, if true, removes table as well
        '''
        self._scene.remove_world_object()

    def _move_to_ready_pick_pose(self):
        '''
        Move arm back to ready_pick pose
        @return, boolean, return true if successful, return false otherwise.
        '''

        # move arm to ready pick pose
        if ASK_BEFORE_EXECUTE:
            self.robot.arm_move_to_joint(ready_pick_joints, plan_only=True, allowed_planning_time=3)
            #Need to expose allowed_planning_time
            execute = raw_input('move to ready pick pose, execute?')
            if execute != 'y':
                rospy.loginfo("SimplePnp: fail to move to ready pick pose")
                return False

        res = self.robot.arm_move_to_joint(ready_pick_joints, plan_only=False, allowed_planning_time=3)
        if not res:
            rospy.loginfo("SimplePnp: fail to move to ready pick pose")
            return False

        return True

    def _calculate_top_place_pose(self, target_pose):
        '''
        Calculate top place pose. Overide parent class because quat is different
        '''
        # TODO frame transformation

        # add approach height
        grasp_pose = target_pose
        grasp_pose.header.frame_id = "base_link"
        if grasp_pose.pose.position.z <= self._table_height:
            rospy.loginfo("PnpService: Gripper might hit table! Setting z to {} from {}".format(self._table_height, grasp_pose.pose.position.z))
            grasp_pose.pose.position.z = self._table_height

        grasp_pose.pose.position.z += self._top_place_approach_dist + self.robot.GRIPPER_LENGTH
        quat = T.quaternion_from_euler(0, math.pi, 0, 'rxyz') # rotate by y to make it facing downwards
        grasp_pose.pose.orientation.x = quat[0]
        grasp_pose.pose.orientation.y = quat[1]
        grasp_pose.pose.orientation.z = quat[2]
        grasp_pose.pose.orientation.w = quat[3]

    def _calculate_top_pick_pose(self, object3d):
        '''
        Calculate top pick pose. Overide parent class because quat is different
        '''

        obj_primitive = object3d.primitive
        obj_pose = object3d.primitive_pose
        assert obj_primitive.type == SolidPrimitive.BOX # only box is supported

        obj_z = obj_primitive.dimensions[SolidPrimitive.BOX_Z]
        obj_x = obj_primitive.dimensions[SolidPrimitive.BOX_X]
        obj_y = obj_primitive.dimensions[SolidPrimitive.BOX_Y]

        grasp = PoseStamped()
        grasp.header.frame_id = self._planning_frame

        grasp.pose.position.x = obj_pose.position.x
        grasp.pose.position.y = obj_pose.position.y
        grasp.pose.position.z = obj_pose.position.z + obj_z / 2 - 0.05

        # extract euler angle, only rotation around z matters for top pick
        euler_angles = T.euler_from_quaternion([obj_pose.orientation.x, obj_pose.orientation.y, obj_pose.orientation.z, obj_pose.orientation.w])
        angle_z = euler_angles[2]

        # always grasp from shorter side of the box
        if obj_x > obj_y:
            # apply additional 90 degree around z axis
            angle_z += math.pi / 2

        # point gripper downwards
        # here use intrinsic euler angle. Basically rotate around rotated y axis by 90 degrees
        grasp_orientation = T.quaternion_from_euler(angle_z, math.pi, 0, 'rzyx') # TODO verify
        grasp.pose.orientation.x = grasp_orientation[0]
        grasp.pose.orientation.y = grasp_orientation[1]
        grasp.pose.orientation.z = grasp_orientation[2]
        grasp.pose.orientation.w = grasp_orientation[3]

        # protect against hitting table
        if grasp.pose.position.z <= self._table_height:
            rospy.loginfo("PnpService: Gripper might hit table! Setting z to {} from {}".format(self._table_height, grasp.pose.position.z))
            grasp.pose.position.z = self._table_height

        # add approach height
        grasp.pose.position.z += self._top_grasp_approach_height + self.robot.GRIPPER_LENGTH

        return grasp, self._top_grasp_approach_height

if __name__ == "__main__":
    # init node
    rospy.init_node('kortex_pnp_service')

    try:
        pnp_server = KortexPnpService()
        # pnp_server.get_ready()
        rospy.spin()
        #pnp_server = PNPServer()
    except Exception as e:
        rospy.logerr(e)
        rospy.logerr("Failed to init pnp_server")