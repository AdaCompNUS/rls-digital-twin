import cv_bridge
import rospy
import cv2
from cv_bridge import CvBridge

from rls_perception_msgs.srv import *

obj_det_client = rospy.ServiceProxy("rls_perception_services/object_detection_srv", DetectObjects)

im = cv2.imread("./test.jpg")

req = DetectObjectsRequest()
req.image = CvBridge().cv2_to_imgmsg(im)
print(req.image.encoding)

resp = obj_det_client(req)
objects = resp.objects

draw_img = im.copy()
for object in objects:
    x1 = int(object.bbox[0])
    y1 = int(object.bbox[1])
    x2 = int(object.bbox[2])
    y2 = int(object.bbox[3])

    cv2.rectangle(draw_img, (x1, y1), (x2, y2), (0, 0, 255), 5)

cv2.imshow("img", draw_img)
cv2.waitKey(0)
cv2.destroyAllWindows(0)