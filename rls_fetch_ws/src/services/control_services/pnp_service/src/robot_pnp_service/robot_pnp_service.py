import rospy
from rospy.client import get_param
from tf import transformations as T
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import PoseStamped, TransformStamped
from rls_perception_msgs.msg import Object3D
import math
import tf2_ros

from rls_control_msgs.srv import PickPlaceRequest, PickPlaceResponse

''' --------- Settings -----------'''
ASK_BEFORE_EXECUTE = False
VIS_GRASP = True

class PnpService(object):
    def __init__(self):
        self._ready = False

        # attributes to be instantiated by sub class
        self._planning_frame = None
        self._top_grasp_approach_height = None
        self._side_grasp_approach_height = None

    def set_planning_scene(self, scene):
        self._scene = scene

    def _handle_pnp_req(self, req):
        '''
        Handle client request
        @param req, SimplePnp, refer to srv definition
        @return, SimplePnpResponse, refer to srv definition
        '''

        res = False
        if req.action == PickPlaceRequest.GET_READY:
            res = self._get_ready(req)
        elif req.action == PickPlaceRequest.EXIT_READY_POSE:
            res = self._exit_ready_pose()
        elif req.action == PickPlaceRequest.MOVE_ARM_TO_HOME:
            if self._ready:
                res = self._move_arm_to_home()
            else:
                rospy.logerr("PnpService, not ready!!")
        elif req.action == PickPlaceRequest.PICK:
            if self._ready:
                res = self._pick(req)
            else:
                rospy.logerr("PnpService, not ready!!")
        elif req.action == PickPlaceRequest.PLACE:
            if self._ready:
                res = self._place(req)
            else:
                rospy.logerr("PnpService, not ready!!")
        elif req.action == PickPlaceRequest.POINT:
            if self._ready:
                res = self._point(req)
            else:
                rospy.logerr("PnpService, not ready!!")
        elif req.action == PickPlaceRequest.EXECUTE_GRASP:
            if self._ready:
                res = self._execute_grasp(req)
            else:
                rospy.logerr("PnpService, not ready!!")

        return PickPlaceResponse(res)

    def _execute_grasp(self, req):
        rospy.loginfo('pnp_service/_execute_grasp')

        grasp = req.grasp
        grasp_pose = grasp.grasp_pose
        gripper_opening = req.gripper_opening
        pre_grasp_approach = grasp.pre_grasp_approach
        post_grasp_retreat = grasp.post_grasp_retreat

        res = self._move_arm_to_pose(grasp_pose)
        if not res:
            return False


        if gripper_opening > 0:
            res = self._set_gripper_opening(gripper_opening)
        else:
            res = self._open_gripper()
        if not res:
            return False

        dx = pre_grasp_approach.desired_distance * pre_grasp_approach.direction.vector.x
        dy = pre_grasp_approach.desired_distance * pre_grasp_approach.direction.vector.y
        dz = pre_grasp_approach.desired_distance * pre_grasp_approach.direction.vector.z

        self._visualize_grasp_finger(grasp_pose, gripper_opening, dx, dy, dz)

        res = self._move_arm_in_cartesian(dx=dx, dy=dy, dz=dz)
        if not res:
            return False

        res = self._close_gripper()
        if not res:
            return False

        dx = post_grasp_retreat.desired_distance * post_grasp_retreat.direction.vector.x
        dy = post_grasp_retreat.desired_distance * post_grasp_retreat.direction.vector.y
        dz = post_grasp_retreat.desired_distance * post_grasp_retreat.direction.vector.z
        res = self._move_arm_in_cartesian(dx=dx, dy=dy, dz=dz)
        if not res:
            return False

        # res = self._move_arm_to_home()
        return res

    def _pick(self, req):
        if req.pick_type == PickPlaceRequest.TOP_PICK:
            grasp_pose, approach_dist = self._calculate_top_pick_pose(req.object)
            res = self._top_pick(grasp_pose, approach_dist)
        elif req.pick_type == PickPlaceRequest.SIDE_PICK:
            grasp_pose, top_approach_dist, side_approach_dist = self._calculate_side_pick_pose(req.object)
            res = self._side_pick(grasp_pose, top_approach_dist, side_approach_dist)
        elif req.pick_type == PickPlaceRequest.SIXDOF_PICK:
            grasp_pose, approach_dist = self._calculate_6dof_pick_pose(req.object)
            res = self._6dof_pick(grasp_pose, approach_dist)

        return res

    def _point(self, req):
        grasp_pose, _ = self._calculate_top_pick_pose(req.object)
        res = self._top_point(grasp_pose)

        return res

    def _place(self, req):
        if req.place_type == PickPlaceRequest.DROP:
            grasp_pose, approach_dist = self._calculate_top_place_pose(req.target_pose)
            res = self._drop_place(grasp_pose)
        elif req.place_type == PickPlaceRequest.TOP_PLACE:
            grasp_pose, approach_dist = self._calculate_top_place_pose(req.target_pose)
            res = self._top_place(grasp_pose, approach_dist)

        return res

    def _get_ready(self):
        '''
        Enter pick_n_place pose.
        @return, boolean, return true if successful, return false otherwise.
        '''
        pass

    def _exit_ready_pose(self):
        '''
        Exit pick_n_place pose.
        @return, boolean, return true if successful, return false otherwise.
        '''
        pass

    def _visualize_grasp_finger(self, pose, gripper_opening=0, dx=0, dy=0, dz=0):
        if not VIS_GRASP:
            return

        br = tf2_ros.StaticTransformBroadcaster()

        left_finger_t = TransformStamped()
        left_finger_t.header.stamp = rospy.Time.now()
        left_finger_t.header.frame_id = 'grasp_pose'
        left_finger_t.child_frame_id = 'left_finger'
        left_finger_t.transform.translation.x = self.robot.GRIPPER_LENGTH - dz
        left_finger_t.transform.translation.y = -gripper_opening / 2 + dx
        left_finger_t.transform.translation.z = -dy
        left_finger_t.transform.rotation.w = 1.0
        br.sendTransform(left_finger_t)  # broadcast transform so can see in rviz

        right_finger_t = TransformStamped()
        right_finger_t.header.stamp = rospy.Time.now()
        right_finger_t.header.frame_id = 'grasp_pose'
        right_finger_t.child_frame_id = 'right_finger'
        right_finger_t.transform.translation.x = self.robot.GRIPPER_LENGTH - dz
        right_finger_t.transform.translation.y = gripper_opening / 2 + dx
        right_finger_t.transform.translation.z = -dy
        right_finger_t.transform.rotation.w = 1.0
        br.sendTransform(right_finger_t)  # broadcast transform so can see in rviz


    def _move_closer_to_table(self, table_dist):
        pass

    def _move_away_from_table(self, dist=0):
        pass

    def _clear_obstacles(self, all_obj=False):
        pass

    def _move_arm_to_home(self):
        '''
        Move arm back to home
        @return, boolean, return true if successful, return false otherwise.
        '''

        rospy.loginfo("PnpService/_move_arm_to_home")

        # here, home pose is the ready_pick_pose()
        return self._move_to_ready_pick_pose()

    def _move_arm_to_pose(self, pose):
        if VIS_GRASP:
            self._visualize_grasp(pose)

        if ASK_BEFORE_EXECUTE:
            res = self.robot.arm_move_to_pose(pose, plan_only=True, allowed_planning_time=3)
            if not res:
                rospy.logerr("move_arm_to_pose failed")
                return False

            to_execute = raw_input('execute?')
            if to_execute != 'y':
                rospy.logerr("move_arm_to_pose failed")
                return False

        # move arm to pose
        res = self.robot.arm_move_to_pose(pose, plan_only=False, allowed_planning_time=3)
        if not res:
            rospy.logerr("_move_arm_to_pose failed 4")
            return False

        return res

    def _move_arm_in_cartesian(self, dx=0, dy=0, dz=0):
        if ASK_BEFORE_EXECUTE:
            rospy.loginfo('move in x, y, z  by {} {} {}'.format(dx, dy, dz))
            to_execute = raw_input('ok?')
            if to_execute != 'y':
                rospy.logerr("_top_pick failed 5")
                return False

        res = self.robot.arm_move_in_cartesian(dx, dy, dz)
        if not res:
            rospy.logerr('move_arm_in_cartesian failed')
        return res

    def _close_gripper(self):
        '''
        provider
        '''
        self.robot.gripper_position(position = self.robot.GRIPPER_CLOSED_POS) # close with minimum effort
        return True

    def _open_gripper(self):
        '''
        provider
        '''
        self.robot.gripper_position(position = self.robot.GRIPPER_OPENED_POS)
        return True

    def _set_gripper_opening(self, opening):
        if opening > self.robot.GRIPPER_OPENED_POS:
            opening = self.robot.GRIPPER_OPENED_POS
        self.robot.gripper_position(position=opening)
        return True

    def _calculate_top_place_pose(self, target_pose):
        # TODO frame transformation

        # add approach height
        grasp_pose = target_pose
        grasp_pose.header.frame_id = "base_link"
        if grasp_pose.pose.position.z <= self._table_height:
            rospy.loginfo("PnpService: Gripper might hit table! Setting z to {} from {}".format(self._table_height, grasp_pose.pose.position.z))
            grasp_pose.pose.position.z = self._table_height

        grasp_pose.pose.position.z += self._top_place_approach_dist + self.robot.GRIPPER_LENGTH + 0.01
        quat = T.quaternion_from_euler(0, math.pi / 2, 0, 'rxyz') # rotate by y to make it facing downwards
        grasp_pose.pose.orientation.x = quat[0]
        grasp_pose.pose.orientation.y = quat[1]
        grasp_pose.pose.orientation.z = quat[2]
        grasp_pose.pose.orientation.w = quat[3]

        return grasp_pose, self._top_place_approach_dist

    def _calculate_top_pick_pose(self, object3d):
        '''
        pick according to grasp_box defined in camera frame
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
        grasp_orientation = T.quaternion_from_euler(0, math.pi / 2, angle_z, 'rxyz')
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

    def _calculate_side_pick_pose(self, object3d):
        '''
        pick according to grasp_box defined in camera frame
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
        grasp.pose.position.z = obj_pose.position.z + self._top_grasp_approach_height

        # point gripper downwards
        # here use intrinsic euler angle. Basically rotate around rotated y axis by 90 degrees
        grasp.pose.orientation.x = 0.0
        grasp.pose.orientation.y = 0.0
        grasp.pose.orientation.z = 0.0
        grasp.pose.orientation.w = 1.0

        # protect against hitting table
        if grasp.pose.position.z <= self._table_height:
            rospy.loginfo("PnpService: Gripper might hit table! Setting z to {} from {}".format(self._table_height, grasp.pose.position.z))
            grasp.pose.position.z = self._table_height

        # add approach height
        grasp.pose.position.x -= self._side_grasp_approach_height + self.robot.GRIPPER_LENGTH

        return grasp, self._top_grasp_approach_height, self._side_grasp_approach_height

    def _top_point(self, grasp_pose):
        rospy.loginfo("PnpService/_top_point: grasp_pose: {}".format(grasp_pose))
        res = self._close_gripper()
        if not res:
            return False
        res = self._move_arm_to_pose(grasp_pose)

        return res

    def _top_pick(self, grasp_pose, approach_dist):
        rospy.loginfo("PnpService/_top_pick: grasp_pose: {}, approach dist: {}".format(grasp_pose, approach_dist))

        res = self._move_arm_to_pose(grasp_pose)
        if not res:
            return False
        res = self._open_gripper()
        if not res:
            return False
        res = self._move_arm_in_cartesian(dz = -approach_dist)
        if not res:
            return False
        rospy.sleep(1) # to prevent possible timing issue
        res = self._close_gripper()
        if not res:
            return False
        res = self._move_arm_in_cartesian(dz = approach_dist)
        if not res:
            return False
        res = self._move_arm_to_home()
        if not res:
            return False

        return res

    def _side_pick(self, grasp_pose, top_approach_dist, side_approach_dist):
        rospy.loginfo("PnpService/_side_pick: grasp_pose: {}, top_approach dist: {}, side_approach_dist: {}".format(grasp_pose, top_approach_dist, side_approach_dist))

        res = self._move_arm_to_pose(grasp_pose)
        if not res:
            return False
        res = self._open_gripper()
        if not res:
            return False
        res = self._move_arm_in_cartesian(dz = -top_approach_dist)
        if not res:
            return False
        res = self._move_arm_in_cartesian(dx = side_approach_dist)
        if not res:
            return False
        rospy.sleep(1) # to prevent possible timing issue
        res = self._close_gripper()
        if not res:
            return False
        res = self._move_arm_in_cartesian(dz = top_approach_dist)
        if not res:
            return False
        res = self._move_arm_to_home()
        if not res:
            return False

        return res

    def _drop_place(self, drop_pose):
        rospy.loginfo("PnpService/_drop_place: pick_pose: {}".format(drop_pose))

        res = self._move_arm_to_pose(drop_pose)
        if not res:
            return False

        rospy.sleep(0.5)
        res = self._open_gripper()
        if not res:
            return False

        res = self._move_arm_to_home()
        if not res:
            return False

        return res

    def _top_place(self, place_pose, approach_dist):
        res = self._move_arm_to_pose(place_pose)
        if not res:
            return False

        res = self._move_arm_in_cartesian(dz = -approach_dist)
        if not res:
            return False

        res = self._open_gripper()
        if not res:
            return False

        res = self._move_arm_in_cartesian(dz = approach_dist)

        return res

    def _visualize_grasp(self, pose, wait=True):
        '''
        visualize grasp pose in rviz
        '''
        # print(pose)
        br = tf2_ros.StaticTransformBroadcaster()
        grasp_t = TransformStamped()
        grasp_t.header.stamp = rospy.Time.now()
        grasp_t.header.frame_id = 'base_link'
        grasp_t.child_frame_id = 'grasp_pose'
        grasp_t.transform.translation = pose.pose.position
        grasp_t.transform.rotation = pose.pose.orientation
        br.sendTransform(grasp_t)  # broadcast transform so can see in rviz

        self._visualize_grasp_finger(pose)
        return True