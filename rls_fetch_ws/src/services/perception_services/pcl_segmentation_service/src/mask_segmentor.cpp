#include <vector>
#include <string>
#include <iostream>

#include "perception/crop.h"
#include "perception/object_recognizer.h"
#include "perception/segmentation.h"
#include "rls_perception_msgs/ObjectFeatures.h"

#include "rls_perception_msgs/MaskSegmentation.h"
#include "rls_perception_msgs/MaskIndice.h"
#include "rls_perception_msgs/Object3D.h"

#include "ros/ros.h"
#include "sensor_msgs/PointCloud2.h"
#include "tf/transform_listener.h"
#include "visualization_msgs/Marker.h"
#include "visualization_msgs/MarkerArray.h"
#include "geometry_msgs/PoseStamped.h"
#include "geometry_msgs/PoseArray.h"
#include "pcl_ros/transforms.h"
#include "pcl_conversions/pcl_conversions.h"
#include <grasping_msgs/FindGraspableObjectsAction.h>
#include <actionlib/server/simple_action_server.h>

#include "pcl/filters/extract_indices.h"
#include "pcl/filters/voxel_grid.h"

typedef pcl::PointXYZRGB PointC;
typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloudC;

namespace pcl_segmentation_service
{
    void apply_mask(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
                    pcl::PointIndices::Ptr mask_indices,
                    const ros::Publisher &mask_pub_)
    {

        PointCloudC::Ptr cloud_out(new PointCloudC());
        pcl::ExtractIndices<PointC> extract;
        extract.setInputCloud(cloud);
        extract.setIndices(mask_indices);
        extract.setNegative(false);
        extract.filter(*cloud_out);

        sensor_msgs::PointCloud2 mask_pcl_out;
        pcl::toROSMsg(*cloud_out, mask_pcl_out);
        mask_pub_.publish(mask_pcl_out);
    }

    class MaskSegmentor
    {
        typedef actionlib::SimpleActionServer<grasping_msgs::FindGraspableObjectsAction> server_t;
        typedef pcl::PointXYZRGB PointC;
        typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloudC;

        ros::NodeHandle nh_;

        bool debug_;

        tf::TransformListener listener_;
        std::string world_frame_;
        //std::string camera_topic = "head_camera/depth_registered/points";
        std::string camera_topic = "/kinect2/qhd/points";

        bool find_objects_;
        std::vector<grasping_msgs::Object> objects_;
        std::vector<grasping_msgs::Object> supports_;

        ros::Subscriber cloud_sub;
        ros::Publisher crop_pub;
        ros::Publisher table_pub;
        ros::Publisher above_table_pub;
        ros::Publisher marker_pub; //for overlaying box on table/obstacles in rviz
        ros::Publisher objPose_pub;
        ros::Publisher objMarker_pub;
        ros::Subscriber segment_sub;

        ros::Publisher mask_pub_;
        ros::ServiceServer mask_srv_;
        ros::Publisher table_marker_pub;

        tf::TransformListener tf_listener;

        boost::shared_ptr<server_t> server_;

        pcl::ModelCoefficients::Ptr coeff_;
        bool first_time;
        pcl::VoxelGrid<PointC> grid_;

    public:
        //this is the constructor that takes in nodehandle as arg
        MaskSegmentor(ros::NodeHandle n) 
            : nh_(n)
        {
            first_time = true;

            // get camera topic from ROS parameter server 
            n.getParam("cemera_topic", camera_topic)

            coeff_ = pcl::ModelCoefficients::Ptr(new pcl::ModelCoefficients());
            // Advertise a service to get segmented pcls from image masks
            mask_srv_ = nh_.advertiseService("mask_segmentor", &MaskSegmentor::segmentCallback, this);

            //use "true" for last parameter so that topics are latched
            crop_pub = nh_.advertise<sensor_msgs::PointCloud2>("cropped_cloud", 1, true);
            table_pub = nh_.advertise<sensor_msgs::PointCloud2>("table_cloud", 1, true);
            above_table_pub = nh_.advertise<sensor_msgs::PointCloud2>("above_table_cloud", 1, true);

            mask_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("mask", 1, true);

            marker_pub = nh_.advertise<visualization_msgs::Marker>("visualization_marker", 1, true);
            table_marker_pub = nh_.advertise<visualization_msgs::Marker>("table_marker", 1, true);

            objPose_pub = nh_.advertise<geometry_msgs::PoseArray>("objPose_arraymarker", 1, true);
            objMarker_pub = nh_.advertise<visualization_msgs::MarkerArray>("objMarker_arraymarker", 1, true);

            // Start thread for action server
            //server_->start();
            //grid_.setLeafSize(0.02, 0.02, 0.02);
            grid_.setLeafSize(0.01, 0.01, 0.01);
            grid_.setDownsampleAllData(true);

            ROS_INFO("mask_segmentor has been initialized");
            //test();
        }

    private:
        bool segmentCallback(rls_perception_msgs::MaskSegmentation::Request &request,
                             rls_perception_msgs::MaskSegmentation::Response &response)
        {
            //assign a list of masks
            std::vector<rls_perception_msgs::MaskIndice> masks = request.masks;

            // get raw point cloud
            sensor_msgs::PointCloud2ConstPtr orig_cloud = ros::topic::waitForMessage<sensor_msgs::PointCloud2>(
                camera_topic, ros::Duration(3));

            //read point cloud multiple times due to weird intermittent bug where previous image is received
            // so it goes to locations from previous trial - could be due to network latency
            ros::Time t = ros::Time::now();
            do
            {
                orig_cloud = ros::topic::waitForMessage<sensor_msgs::PointCloud2>(camera_topic, ros::Duration(9.0));
                ros::Duration(1 / 5.0).sleep();
                if (ros::Time::now() - t > ros::Duration(15.0)) //wait for at most 15s
                {
                    //ROS_ERROR("timeout waiting for pointcloud head_camera/depth_registered/points.");
                    ROS_ERROR("timeout waiting for pointcloud kinect2/qhd/points.");
                    break;
                }
            } while (orig_cloud == NULL);

            if (request.transform_to_reference_frame) {
                // transform pointcloud to base_link
                tf_listener.waitForTransform("base_link", orig_cloud->header.frame_id,
                                            ros::Time(0), ros::Duration(2.0));
                tf::StampedTransform transform;
                try
                {
                    tf_listener.lookupTransform("base_link", orig_cloud->header.frame_id,
                                                ros::Time(0), transform);
                }
                catch (tf::LookupException &e)
                {
                    std::cerr << e.what() << std::endl;
                }
                catch (tf::ExtrapolationException &e)
                {
                    std::cerr << e.what() << std::endl;
                }

                sensor_msgs::PointCloud2 cloud_out;
                pcl_ros::transformPointCloud("base_link", transform, *orig_cloud, cloud_out);

                // convert ros msg to pcl
                PointCloudC::Ptr cloud(new PointCloudC());
                pcl::fromROSMsg(cloud_out, *cloud);
            }

            //loop over all masks
            //std::cout<<masks.size()<<std::endl;
            geometry_msgs::PoseArray pose_arr;
            pose_arr.header.frame_id = "base_link";

            for (int i = 0; i < masks.size(); i++)
            {
                //std::cout<< "Next Segment ..." <<std::endl;
                //std::cin.get();

                pcl::PointIndices::Ptr mask_indices(new pcl::PointIndices());

                // prepare mask as indices
                //std::cout<<"before push indice "<<std::endl;
                //for (int j=0; j<(masks[i].mask).size(); j++)
                for (int j = 0; j < (masks[i].mask).size(); j++)
                    mask_indices->indices.push_back((masks[i].mask)[j]);

                //std::cout<<"size of mask  "<<mask_indices->indices.size()<<std::endl;

                // filter cloud
                PointCloudC::Ptr cloud_seg(new PointCloudC());
                pcl::ExtractIndices<PointC> extract;
                extract.setInputCloud(cloud);
                //extract.setInputCloud(cloud_down);
                extract.setIndices(mask_indices);
                extract.setNegative(false);
                extract.filter(*cloud_seg);

                //pcl::PointCloud<PointC>::Ptr cloud_down (new pcl::PointCloud<PointC>);
                //grid_.setInputCloud(cloud_seg);
                //grid_.filter(*cloud_down);

                //std::cout<<" size of downsampled cloud is "<< cloud_seg->width * cloud_seg->height<<std::endl;
                // get object's pose and 3dbbox
                //grasping_msgs::Object object;
                rls_perception_msgs::Object3D object;
                object = segmenter.get_object(cloud_seg, i, coeff_);
                //object = segmenter.get_object(cloud_down, i, coeff_);

                // add stamp and frame
                object.header.stamp = ros::Time::now();
                object.header.frame_id = cloud_out.header.frame_id;

                response.objects.push_back(object);

                //publish mask pointcloud
                sensor_msgs::PointCloud2 mask_pcl_out;
                pcl::toROSMsg(*cloud_seg, mask_pcl_out);
                mask_pub_.publish(mask_pcl_out);

                Object obj;
                obj.pose = object.primitive_poses[0];
                obj.dimensions.x = object.primitives[0].dimensions[0];
                obj.dimensions.y = object.primitives[0].dimensions[1];
                obj.dimensions.z = object.primitives[0].dimensions[2];

                //std::cout<<object.primitives[0].dimensions[0]<<std::endl;
                //std::cout<<object.primitives[0].dimensions[1]<<std::endl;
                //std::cout<<object.primitives[0].dimensions[2]<<std::endl;

                // publish 3d bbox marker
                visualization_msgs::Marker object_marker;
                object_marker.ns = "objects";
                object_marker.id = i;
                object_marker.header.frame_id = "base_link";
                object_marker.header.stamp = ros::Time::now(); //added so that labels in rviz will refresh
                object_marker.type = visualization_msgs::Marker::CUBE;
                object_marker.pose = obj.pose;
                object_marker.scale = obj.dimensions;
                object_marker.color.g = 1;
                object_marker.color.a = 0.3;
                object_marker.lifetime = ros::Duration(15);
                marker_pub.publish(object_marker);

                pose_arr.poses.push_back(object.primitive_poses[0]);
            }

            objPose_pub.publish(pose_arr);

            return true;
        }
} // namespace perception

// this program is a node
int main(int argc, char **argv)
{
    ros::init(argc, argv, "perception_demo");
    ros::NodeHandle nh;

    perception::PerceptionMask perceive_object(nh); //will call constructor
    ros::spin();
    return 0;
}