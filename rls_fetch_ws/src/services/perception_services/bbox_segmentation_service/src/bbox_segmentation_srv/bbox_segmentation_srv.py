#! /usr/bin/env python

import rospy
import actionlib
import pickle
import copy
import os
import sys
from geometry_msgs.msg import PoseStamped, Pose, Point, PointStamped
from sensor_msgs.msg import Image, PointCloud2, PointField
import tf
import sensor_msgs.point_cloud2 as pcl2
import math
from shape_msgs.msg import SolidPrimitive
import numpy as np

from rls_perception_msgs.srv import BBoxSegmentation, BBoxSegmentationResponse
from rls_perception_msgs.msg import Object3DV2

## Constants
DEBUG_PC_PUB = True
BUFFER_MIN_Z  = 0.003
# MIN_HEIGHT = (0.027+0.003) #0.043

class BBoxSegmentor:

    def __init__(self, pc_topic_name, bbox_seg_srv_name):
        self._pc_topic_name = pc_topic_name
        self._seg_pub = rospy.Publisher('bbox_segmentor/segmented_pc_debug', PointCloud2, queue_size=2)
        self._centroid_pub = rospy.Publisher('bbox_segmentor/segmented_centroid', PointStamped, queue_size=2)
        self._bbox_srv = rospy.Service(bbox_seg_srv_name, BBoxSegmentation, self.segment)

        self._tl = tf.TransformListener()
        rospy.loginfo("bbox_segmentor, using {}".format(self._pc_topic_name))
        rospy.loginfo("bbox_segmentor inited")

    '''
    def pc_sub(self, msg):
        # rospy.loginfo("BBoxSegmentor: PointCloud msg received")
        self._raw_pc = msg
    '''

    def clean_nan_inf(self, val):
        if np.isnan(val) or np.isinf(val):
            return 0.0
        else:
            return val

    def segment(self, req):
        try:
            rospy.loginfo("Waiting for point cloud")
            raw_pc = rospy.wait_for_message(self._pc_topic_name , PointCloud2, timeout=10.0) # at most 10 seconds
        except Exception as e:
            rospy.logerr(e)
            return BBoxSegmentationResponse()

        if raw_pc == None:
            print("No PointCloud Received")
            return BBoxSegmentationResponse()

        print(req)

        # calculate thresholds
        min_z = req.min_z
        rospy.loginfo("min_z {}".format(min_z))

        # cloud properties
        cloud_width = raw_pc.width
        cloud_height = raw_pc.height
        row_step = raw_pc.row_step
        point_step = raw_pc.point_step
        x_offset = raw_pc.fields[0].offset
        y_offset = raw_pc.fields[1].offset
        z_offset = raw_pc.fields[2].offset

        # bbox parameters
        box_x0 = int(req.x)
        box_y0 = int(req.y)
        box_width = int(req.width)
        box_height = int(req.height)
        box_angle = req.angle

        # build uv array for segmentation
        uvs = []
        for x in range(box_width):
            for y in range(box_height):
                # convert to image coordinates
                # Note that this works when box_angle is 0.
                delta_x = int(x * math.cos(box_angle) - y * math.sin(box_angle))
                delta_y = int(x * math.sin(box_angle) + y * math.cos(box_angle))

                uvs.append([box_x0 + delta_x, box_y0 + delta_y])

        # find common time for transforming points
        # raw_pc = copy.deepcopy(self._raw_pc)
        # print(uvs)

        # segment
        segmented_points = []
        points = pcl2.read_points(raw_pc, field_names=('x', 'y', 'z'), skip_nans=True, uvs=uvs)
        rospy.loginfo('bbox_segmentor, segment raw pc completed!')

        if req.transform_to_reference_frame:
            reference_frame = req.reference_frame

            # find reference frame transformation
            cnt = 0
            while not rospy.is_shutdown() and cnt < 100:
                try:
                    time = self._tl.getLatestCommonTime(reference_frame, raw_pc.header.frame_id)
                    trans, rot = self._tl.lookupTransform(reference_frame, raw_pc.header.frame_id, time)
                    transform_mat44 = np.dot(tf.transformations.translation_matrix(
                        trans), tf.transformations.quaternion_matrix(rot))
                    break
                except:
                    print "Waiting for tf between %s and %s" % (raw_pc.header.frame_id, reference_frame)
                    cnt += 1
                    continue

            if cnt >= 100:
                rospy.logerr("bbox_segmentor: wait for frame transformation timed out, segmentation failed")
                return

            for p in points:
                xyz = tuple(np.dot(transform_mat44, np.array([p[0], p[1], p[2], 1.0])))[:3]
                point = Point(*xyz)
                segmented_points.append([point.x, point.y, point.z])
        else:
            reference_frame = raw_pc.header.frame_id
            for p in points:
                segmented_points.append([p[0], p[1], p[2]])

        rospy.loginfo('got {} points'.format(len(segmented_points)))

        # Apply threshold (cutoff the table points)
        segmented_points = [p for p in segmented_points if p[2] > min_z]
        rospy.loginfo('after threshold, got {} points'.format(len(segmented_points)))

        # create segmented PC2
        header = raw_pc.header
        header.frame_id = reference_frame
        segmented_pc = pcl2.create_cloud_xyz32(header,segmented_points)

        # fields = [PointField('x', 0, PointField.FLOAT32, 1),
        #           PointField('y', 4, PointField.FLOAT32, 1),
        #           PointField('z', 8, PointField.FLOAT32, 1)]
        # segmented_pc = pcl2.create_cloud(header, fields, segmented_points)

        # compute centroid
        x_sum = 0.
        y_sum = 0.
        z_sum = 0.
        for point in segmented_points:
            x_sum += point[0]
            y_sum += point[1]
            z_sum += point[2]

        centroid = Point()
        num_points = len(segmented_points)
        if num_points > 0:
            centroid.x = x_sum / len(segmented_points)
            centroid.y = y_sum / len(segmented_points)
            centroid.z = z_sum / len(segmented_points)

        if DEBUG_PC_PUB:
            self._seg_pub.publish(segmented_pc)

            point_stamped = PointStamped()
            point_stamped.header = reference_frame
            point_stamped.point = centroid
            self._centroid_pub.publish(point_stamped)

        # compute object properties
        if num_points > 0:
            segmented_points = np.array(segmented_points)
            # print(segmented_points.shape)

            x = segmented_points[:, 0]
            y = segmented_points[:, 1]
            z = segmented_points[:, 2]

            top_x = np.sort(x)[-10:]
            top_y = np.sort(y)[-10:]
            top_z = np.sort(z)[-10:]

            btm_x = np.sort(x)[:10]
            btm_y = np.sort(y)[:10]
            btm_z = np.sort(z)[:10]

            print(top_x, top_y, top_z)
            print(btm_x, btm_y, btm_z)

            obj_z = top_z.mean() - btm_z.mean()
            obj_y = top_y.mean() - btm_y.mean()
            obj_x = top_x.mean() - btm_x.mean()

            cx = (top_x.mean() + btm_x.mean()) / 2
            cy = (top_y.mean() + btm_y.mean()) / 2
            cz = (top_z.mean() + btm_z.mean()) / 2

            # max_z = segmented_points[:, 2].max()
            # min_z = segmented_points[:, 2].min()
            # max_y = segmented_points[:, 1].max()
            # min_y = segmented_points[:, 1].min()
            # max_x = segmented_points[:, 0].max()
            # min_x = segmented_points[:, 0].min()

            # obj_z = max_z - min_z
            # obj_y = max_y - min_y
            # obj_x = max_x - min_x
        else:
            obj_z = -1.0
            obj_y = -1.0
            obj_x = -1.0
            cx = -1.0
            cy = -1.0
            cz = -1.0

        print("Num Points: %d, obj_x: %f, obj_y: %f, obj_z: %f" % (num_points, obj_x, obj_y, obj_z))

        # construct response
        # wrap object in a AABB wrt reference frame
        object3d = Object3DV2()
        object3d.header.frame_id = reference_frame
        obj_bbox = SolidPrimitive()
        obj_bbox.type = SolidPrimitive.BOX
        obj_bbox.dimensions = [obj_x, obj_y, obj_z]
        object3d.primitive = obj_bbox
        obj_pose = Pose()
        # obj_pose.position.x = centroid.x
        # obj_pose.position.y = centroid.y
        # obj_pose.position.z = centroid.z
        obj_pose.position.x = cx
        obj_pose.position.y = cy
        obj_pose.position.z = cz
        obj_pose.orientation.w = 1.0
        object3d.primitive_pose = obj_pose

        print("cx: %f, cy: %f, cz: %f" % (obj_pose.position.x, obj_pose.position.y, obj_pose.position.z))

        return BBoxSegmentationResponse(segmented_pc, object3d, num_points)


if __name__ == '__main__':

    try:
        rospy.init_node('bbox_segmentor', anonymous=True)
        pc_topic_name = rospy.get_param('~pc_topic')
        srv_namespace = rospy.get_param('~srv_namespace')
        bbox_segmentor = BBoxSegmentor(pc_topic_name, 'rls_perception_services/{}/bbox_pc_segmention_service'.format(srv_namespace))
        rospy.spin()

    except rospy.ROSInterruptException:
        pass
