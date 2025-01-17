#!/usr/bin/env python

from scipy.spatial.transform import Rotation as R
import rospy
import sys

from rls_control_msgs.srv import *
from rls_perception_msgs.msg import Object3DV2

rospy.init_node('pnp_get_ready')

rospy.wait_for_service('/rls_control_services/fetch/pnp')
pnp_client = rospy.ServiceProxy('/rls_control_services/fetch/pnp', PickPlace)

# get ready, this will raise robot torso, detect table, move robot's arm to ready pick position
user_input = raw_input('press y to continue')
if user_input != 'y':
    sys.exit()

req = PickPlaceRequest()
req.action = PickPlaceRequest.GET_READY
req.torso_height = 0.4
resp = pnp_client(req)
if not resp.success:
    print('get ready failed, abort')
    sys.exit()

# top_pick
user_input = raw_input('press y to continue')
if user_input != 'y':
    sys.exit()

req = PickPlaceRequest()
req.action = PickPlaceRequest.PICK
req.pick_type = PickPlaceRequest.TOP_PICK
object = Object3DV2()
object.primitive.type = 1 # box
object.primitive.dimensions = [0.1, 0.1, 0.1]
object.primitive_pose.position.x = 0.5
object.primitive_pose.position.y = 0
object.primitive_pose.position.z = 1.0
object.primitive_pose.orientation.w = 1
req.object = object
resp = pnp_client(req)
if not resp.success:
    print('pick failed, abort')
    sys.exit()

# drop place
user_input = raw_input('press y to continue')
if user_input != 'y':
    sys.exit()
req = PickPlaceRequest()
req.action = PickPlaceRequest.PLACE
req.pick_type = PickPlaceRequest.DROP
req.target_pose.header.frame_id = 'base_link'
req.target_pose.pose.position.x = 0.5
req.target_pose.pose.position.y = 0
req.target_pose.pose.position.z = 1.0
req.target_pose.pose.orientation.w = 1
req.object = object
resp = pnp_client(req)
if not resp.success:
    print('drop place failed, abort')
    sys.exit()

# exit pose
req = PickPlaceRequest()
req.action = PickPlaceRequest.EXIT_READY_POSE
resp = pnp_client(req)
if not resp.success:
    print('exit failed, abort')
    sys.exit()


