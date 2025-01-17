#!/usr/bin/env python

import rospy
import tf.transformations as T
import tf2_ros
import math
from cv_bridge import CvBridge
from PIL import Image
import cv2
from geometry_msgs.msg import Pose, PoseStamped, TransformStamped

from bbox_segmentation_service.srv import *
from rls_perception_msgs.srv import *
from movo_handover_service.srv import *
from movo_action_clients_v2 import movo_action_clients_v2
from ingress_srv import ingress_srv
from rls_control_msgs.srv import *


import robot_config

ASK_BEFORE_EXECUTE = False
APPROACH_HEIGTH = 0.1
BUFFER_HEIGHT = 0.01
IMAGE_CROP_X = 200
IMAGE_CROP_Y = 190
IMAGE_CROP_X_END = 760
IMAGE_CROP_Y_END = 530

class HandoverService(object):
    def __init__(self):
        self._ready = False

        # use ros client to control
        # NOTE: this is due to multimaster
        rospy.wait_for_service('rls_control_services/movo/torso')
        self._movo_torso_client = rospy.ServiceProxy('rls_control_services/movo/torso', MovoTorsoAction)
        self._movo_arm_client = rospy.ServiceProxy('rls_control_services/movo/arm', MovoArmAction)
        self._movo_gripper_client = rospy.ServiceProxy('rls_control_services/movo/gripper', MovoGripperAction)
        self._movo_head_client = rospy.ServiceProxy('rls_control_services/movo/head', MovoHeadAction)

        # # arm controller
        # self.arm = movo_action_clients.mac_arm
        # self.arm_name = 'right_arm'

        # # torso controller
        # self.torso = movo_action_clients.mac_torso
        # self.gripper_right = movo_action_clients.mac_right_gripper
        # self.head = movo_action_clients.mac_head

        ## arm, torso, gripper are all necessary, raise error if initialization fails
        # if self.arm is None or self.gripper_right is None or self.head is None:
        #     raise RuntimeError("HandoverService: Failed to init HandoverService")

        # dependent services
        self._image_srv = rospy.ServiceProxy(
            '/rls_perception_services/movo/fetch_image', FetchImage)
        self._segmentor_srv = rospy.ServiceProxy(
            '/rls_perception_services/bbox_segmentor', BBoxSegmentation)
        self._ingress_client = ingress_srv.Ingress()

        # attributes
        self._br = CvBridge()

        ## advertise service
        rospy.Service('rls_control_services/movo/handover', Handover, self._hand_over)
        rospy.loginfo('HandoverService is Ready!')

    def get_img(self):
        # get camera image
        rospy.loginfo("HandoverService: waiting for image")
        resp = self._image_srv(FetchImageRequest.COLOUR_ONLY)
        img_msg = resp.color
        if img_msg is None:
            rospy.logerr("HandoverService: no image!")
            return IngressObjectManipulator.FAIL
        rospy.loginfo("HandoverService: received image")
        img_cv = self._br.imgmsg_to_cv2(img_msg)
        print(img_cv.shape)
        img_cv = img_cv[IMAGE_CROP_Y:IMAGE_CROP_Y_END, IMAGE_CROP_X:IMAGE_CROP_X_END]

        img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img)
        im_pil.show()

        return img_cv

    def _detect_hand_pose(self):
        rospy.loginfo("HandoverService: _detect_hand_pose")

        # get an image
        img_cv = self.get_img()

        boxes, top_idx, context_idxs, captions = self._ingress_client.ground(img_cv, "hand", is_rel_query=False)  # context_idxs are indexes of other 'relevant' bboxes
        if len(context_idxs) == 0 or captions is None:
            rospy.logerr("HandoverService: hand not detected")
            return None

        # segment out human hand pos
        req = BBoxSegmentationRequest()
        box = boxes[top_idx]
        req.x = (box[0] + IMAGE_CROP_X)
        req.y = (box[1] + IMAGE_CROP_Y)
        req.width = box[2]
        req.height = box[3]
        rospy.loginfo("HandoverService: min_z = {}".format(req.min_z))
        rospy.loginfo("waiting for segmentation service result")
        resp = self._segmentor_srv(req)

        rospy.loginfo("HandoverService: num_of_pc = {}".format(resp.num_points))
        if resp.num_points < 5:
            rospy.logerr("HandoverService: segment hand failed")
            return None

        hand_pose = PoseStamped()
        hand_pose.header.frame_id = 'base_link'
        hand_pose.pose.position.x = resp.centroid.x
        hand_pose.pose.position.y = resp.centroid.y
        hand_pose.pose.position.z = resp.centroid.z + resp.object_height # highest point
        hand_pose.pose.orientation.w = 1.0

        return hand_pose

    def get_ready(self):
        rospy.loginfo("get_ready")
        # raise torso
        self._movo_torso_client(0.2)

        # lower head
        req = MovoHeadActionRequest()
        req.action = MovoHeadActionRequest.PAN_TILT
        req.tilt = -30.0
        self._movo_head_client(req)

    def _hand_over(self, req):
        res = False

        self.get_ready()
        human_hand_pos = self._detect_hand_pose()
        if human_hand_pos is None:
            rospy.logerr("HandoverService: detect_hand_pose failed")
            return HandoverResponse(res)

        res = self._place_on_human_hand(human_hand_pos)
        if not res:
            rospy.logerr("HandoverService: place_on_human_hand failed")

        return HandoverResponse(res)

    def _visualize_grasp(self, pose, wait=True):
        '''
        visualize grasp pose in rviz
        '''
        br = tf2_ros.StaticTransformBroadcaster()
        grasp_t = TransformStamped()
        grasp_t.header.stamp = rospy.Time.now()
        grasp_t.header.frame_id = 'base_link'
        grasp_t.child_frame_id = 'grasp_pose'
        #objPose_pickFromSide = ros_numpy.msgify(Pose, pose_num)
        grasp_t.transform.translation = pose.pose.position
        grasp_t.transform.rotation = pose.pose.orientation
        br.sendTransform(grasp_t)  # broadcast transform so can see in rviz
        if ASK_BEFORE_EXECUTE:
            to_execute = raw_input('visualize grasp pose, execute?')
            if to_execute != 'y':
                return False
        return True

    def _get_preplace_pos(self, grasp):
        # Ensure gripper is facing down
        # TODO!!! this assumes that base_link frame is horizontal
        pre_place_pose = PoseStamped()
        pre_place_pose.header.frame_id = 'base_link'
        pre_place_pose.pose.position = grasp.pose.position
        pre_place_pose.pose.position.z += APPROACH_HEIGTH + BUFFER_HEIGHT
        q = T.quaternion_from_euler(0, math.radians(90), 0)
        pre_place_pose.pose.orientation.x = q[0]
        pre_place_pose.pose.orientation.y = q[1]
        pre_place_pose.pose.orientation.z = q[2]
        pre_place_pose.pose.orientation.w = q[3]

        return pre_place_pose

    def _movo_arm_to_pose(self, pose):
        req = MovoArmActionRequest()
        req.action = MovoArmActionRequest.MOVE_TO_POSE
        req.arm = MovoArmActionRequest.RIGHT_ARM
        req.pose_goal = pose
        self._movo_arm_client(req)

    def _movo_arm_to_joint(self, joint):
        req = MovoArmActionRequest()
        req.action = MovoArmActionRequest.MOVE_TO_JOINT
        req.arm = MovoArmActionRequest.RIGHT_ARM
        req.joint_goal = joint
        self._movo_arm_client(req)

    def _move_arm_in_cartesian(self, dx, dy, dz):
        req = MovoArmActionRequest()
        req.action = MovoArmActionRequest.MOVE_IN_CARTESIAN
        req.arm = MovoArmActionRequest.RIGHT_ARM
        req.cartesian_goal = [dx, dy, dz]
        self._movo_arm_client(req)

    def _place_on_human_hand(self, grasp):
        '''
        place object from top
        @param, grasp, geometry_msgs/Pose, the grasp pose
        @param point_only, if true, the robot will simply move arm to grasp pose without placing the object.
        @return, boolean, return true if successful, return false otherwise.
        '''
        # get pre_place_pose
        pre_place_pose = self._get_preplace_pos(grasp)

        # visualize grasp
        res = self._visualize_grasp(pre_place_pose)
        if not res:
            rospy.logerr("_place_on_human_hand failed 1")
            return False

        replan = False

        # move to grasp pose
        res = self._movo_arm_to_pose(pre_place_pose)
        if res != 'SUCCESS':
            rospy.logerr("_place_on_human_hand failed 4")
            return

        # Lower arm to pick
        approach_ht = APPROACH_HEIGTH
        print('move in z by {}'.format(-approach_ht))
        res = self._move_arm_in_cartesian(dx = 0, dy = 0, dz=-approach_ht)  # move arm down to object
        if not res[0]:
            rospy.logerr("_place_on_human_hand failed 6")
            return False

        # open gripper
        req = MovoGripperActionRequest()
        req.action = MovoGripperActionRequest.GRIPPER_OPEN
        self._movo_gripper_client(req)

        # Move arm up
        depart_ht = APPROACH_HEIGTH
        res = self._move_arm_in_cartesian(dx = 0, dy = 0, dz=depart_ht)  # move arm up
        if not res[0]:
            rospy.logerr("_place_on_human_hand failed 7")
            return False

        # Move arm back
        res = self._movo_arm_to_joint(robot_config.right_ready_joints)
        if res != 'SUCCESS':
            rospy.logerr("_place_on_human_hand failed 8")
            return False

        # self._detach_object(self.obj, remove=True)
        return True


if __name__ == "__main__":
    ## init node
    rospy.init_node('handover_service')

    try:
        handover_service = HandoverService()
        rospy.spin()
    except Exception as e:
        rospy.logerr(e)
        rospy.logerr("Failed to init SimpleHandoverService")
