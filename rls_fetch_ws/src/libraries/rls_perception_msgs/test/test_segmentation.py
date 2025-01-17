import rospy
from perception_msgs.srv import MaskSegmentation, MaskSegmentationRequest
from perception_msgs.msg import MaskIndice
from moveit_python import PlanningSceneInterface

class Segment(object):
    def __init__(self):
        rospy.wait_for_service('segment_objects')
        self.segment_client = rospy.ServiceProxy('segment_objects', MaskSegmentation)
        self.scene = PlanningSceneInterface("base_link")

        raw_input('ready')
        self.run()

    def setTableAsObstacle(self):
        table_msg = rospy.wait_for_message('table_marker', Marker)
        self.scene.addBox('table', table_msg.scale.x,
                                   table_msg.scale.y,
                                   table_msg.scale.z,
                                   table_msg.pose.position.x,
                                   table_msg.pose.position.y,
                                   table_msg.pose.position.z)

    def run(self):
        mask1 = MaskIndice()
        mask2 = MaskIndice()

        for i in range(40, 200):
            for j in range(40, 80):
                (mask1.mask).append(640*j+i)

        for i in range(40, 200):
            for j in range(150, 190):
                (mask2.mask).append(640*j+i)

        req = MaskSegmentationRequest()
        req.masks.append(mask1)
        req.masks.append(mask2)

        res = self.segment_client(req)
        self.setTableAsObstacle()


if __name__=='__main__':
    rospy.init_node('seg_test')
    s = Segment()
    rospy.spin()

