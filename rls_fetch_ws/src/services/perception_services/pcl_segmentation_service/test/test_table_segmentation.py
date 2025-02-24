import rospy
from moveit_python import PlanningSceneInterface

from rls_perception_msgs.srv import *

class TestTableSeg():
    def __init__(self):
        # Table Segmentor
        rospy.loginfo("Segmentor: Waiting for table pointcloud segmentation service /segment_table ...")
        rospy.wait_for_service('/segment_table')
        self.table_segmentor_srv = rospy.ServiceProxy('/segment_table', TableSegmentation)
        rospy.loginfo("Service found!")

        self.scene = PlanningSceneInterface("base_link")

    def set_table_as_obstacle(self):
        '''
        Add table to planning scene as an obstacle
        @return, boolean, return true if successful, return false otherwise.
        '''

        # call table segmentor to get table marker
        rospy.loginfo("Calling table segmentor")
        resp = self.table_segmentor_srv(0)
        table_msg = resp.marker
        # rospy.loginfo("table_marker msg arrived: {}".format(table_msg))

        # add box
        self.scene.addBox('table', table_msg.scale.x,
                                  table_msg.scale.y,
                                  table_msg.scale.z,
                                  table_msg.pose.position.x,
                                  table_msg.pose.position.y,
                                  table_msg.pose.position.z)
        self._table_height = table_msg.pose.position.z + table_msg.scale.z / 2
        self._table_dist = table_msg.pose.position.x - table_msg.scale.x / 2
        rospy.loginfo("table_height: {}, table_dist {}".format(self._table_height, self._table_dist))
        self.scene.waitForSync()

        return True

if __name__ == "__main__":
    rospy.init_node('test_table_seg')
    test_table_seg = TestTableSeg()
    test_table_seg.set_table_as_obstacle()


