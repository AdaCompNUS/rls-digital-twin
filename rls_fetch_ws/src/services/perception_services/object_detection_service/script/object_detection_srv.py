#!/usr/bin/env python3
import sys 
sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")
sys.path.append("/opt/ros/kinetic/lib/python2.7/dist-packages")

import rospy

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from rls_perception_msgs.srv import *
from rls_perception_msgs.msg import Object2D

# --------- SETTINGS ------------
VISUALIZE = False

class ObjectDetectionService():
    def __init__(self):
        # init Detectron2
        self._cfg = get_cfg()
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        self._cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
        self._cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        self._cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
        self._predictor = DefaultPredictor(self._cfg)

        self._class_names = MetadataCatalog.get(self._cfg.DATASETS.TRAIN[0]).get("thing_classes", None)

        # init ros service
        self._service = rospy.Service('rls_perception_services/object_detection_srv', DetectObjects, self._detect_objects)
        rospy.loginfo("object_detection_srv inited")
    
    def _imgmsg_to_cv2(self, img_msg):
        dtype, n_channels = np.uint8, 3 # hard code
        dtype = np.dtype(dtype)
        dtype = dtype.newbyteorder('>' if img_msg.is_bigendian else '<')
        im = np.ndarray(shape=(img_msg.height, img_msg.width, n_channels),
                        dtype=dtype, buffer=img_msg.data)
        return im

    def _detect_objects(self, req):
        if req.image.encoding != '8UC3':
            rospy.logerr("object_detection_srv, image encoding not supported!!")
            res = DetectObjectsResponse()
            return res

        img_cv2 = self._imgmsg_to_cv2(req.image)
        outputs = self._predictor(img_cv2)
        # look at the outputs. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification
        print(outputs["instances"].pred_classes)
        print(outputs["instances"].pred_boxes)
        self._visualize(img_cv2, outputs)

        res = DetectObjectsResponse()
        pred_classes = outputs["instances"].pred_classes.cpu().numpy().tolist()
        pred_bboxes = outputs["instances"].pred_boxes.tensor
        pred_bboxes =  pred_bboxes.cpu().numpy().tolist()
        pred_masks = outputs["instances"].pred_masks.cpu()
        pred_masks = pred_masks.view(pred_masks.size()[0], -1)
        pred_masks =  pred_masks.numpy().tolist()
        object_list = []
        for i in range(len(pred_classes)):
            object2d = Object2D()
            object2d.class_name = self._class_names[pred_classes[i]]
            object2d.prob = outputs["instances"].scores[i].item()
            object2d.bbox = pred_bboxes[i]
            object2d.mask = pred_masks[i]
            object_list.append(object2d)
        res.objects = object_list

        return res

    def _visualize(self, im, outputs):
        if not VISUALIZE:
            return

        # We can use `Visualizer` to draw the predictions on the image.
        v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(self._cfg.DATASETS.TRAIN[0]), scale=1.2)

        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imshow("img", out.get_image()[:, :, ::-1])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    rospy.init_node('object_detection_service')
    object_detection_service = ObjectDetectionService()
    rospy.spin()

