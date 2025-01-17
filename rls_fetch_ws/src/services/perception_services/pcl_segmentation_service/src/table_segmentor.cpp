#include <vector>
#include <string>
#include <iostream>

// #include "perception/crop.h"
// #include "perception/object_recognizer.h"
// #include "perception/segmentation.h"
#include "rls_perception_msgs/ObjectFeatures.h"

#include "rls_perception_msgs/TableSegmentation.h"
#include "rls_perception_msgs/MaskIndice.h"
#include "rls_perception_msgs/Object3D.h"

#include "ros/ros.h"
#include "sensor_msgs/PointCloud2.h"
#include "tf/transform_listener.h"
#include "visualization_msgs/Marker.h"
#include "visualization_msgs/MarkerArray.h"
#include "geometry_msgs/PoseStamped.h"
#include "geometry_msgs/PoseArray.h"
#include "pcl/PointIndices.h"
#include "pcl/common/angles.h"
#include "pcl/common/common.h"
#include "pcl/common/pca.h"
#include "pcl/filters/extract_indices.h"
#include "pcl/point_cloud.h"
#include "pcl/point_types.h"
#include "pcl/sample_consensus/method_types.h"
#include "pcl/sample_consensus/model_types.h"
#include "pcl/segmentation/extract_clusters.h"
#include "pcl/segmentation/sac_segmentation.h"
#include "pcl_conversions/pcl_conversions.h"
#include "pcl_ros/transforms.h"
#include <grasping_msgs/FindGraspableObjectsAction.h>
#include <actionlib/server/simple_action_server.h>

#include "pcl/filters/extract_indices.h"
#include "pcl/filters/voxel_grid.h"

#define MAX_DISTANCE_TO_CAMERA (4.0)

typedef pcl::PointXYZRGB PointC;
typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloudC;

namespace perception
{
void apply_mask(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
                           pcl::PointIndices::Ptr mask_indices,
                           const ros::Publisher& mask_pub_){

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

typedef enum TableSegmentorAction {
    GET_NEW_RESULT,
    USE_EXISTING_RESULT,
    
    NUM_OF_ACTION
} TableSegmentorActionType;

class TableSegmentor
{
    typedef actionlib::SimpleActionServer<grasping_msgs::FindGraspableObjectsAction> server_t;
    typedef pcl::PointXYZRGB PointC;
    typedef pcl::PointCloud<pcl::PointXYZRGB> PointCloudC;

    ros::NodeHandle nh_;

    bool debug_;

    tf::TransformListener listener_;
    std::string world_frame_;
    std::string camera_topic;
    // std::string camera_topic = "/head_camera/depth_registered/points";
    // std::string camera_topic = "/kinect2/qhd/points";

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

    visualization_msgs::Marker mTableMarker;

public:
    //this is the constructor that takes in nodehandle as arg
    TableSegmentor(ros::NodeHandle n) : nh_(n)
    {
        first_time = true;
        coeff_ = pcl::ModelCoefficients::Ptr(new pcl::ModelCoefficients());
        // Advertise an action for perception + planning
        //server_.reset(new server_t(nh_, "find_objects",
        //                           boost::bind(&PerceptionMask::executeCallback, this, _1),
        //                          false));
        // Advertise a service to get segmented pcls from image masks
        std::string service_name;
        if (ros::param::get("~service_name", service_name)) {
            mask_srv_ = nh_.advertiseService(service_name, &TableSegmentor::SegmentCallback, this);
        } 
        else {
            ROS_ERROR("service_name param not found!!");
        }

        if (!ros::param::get("~cloud_in", camera_topic)) {
            ROS_ERROR("cloud_in param not found!!");
        }
        // camera_topic = nh_.resolveName("cloud_in");

        // Start thread for action server
        //server_->start();
        //grid_.setLeafSize(0.02, 0.02, 0.02);
        grid_.setLeafSize(0.01, 0.01, 0.01);
        grid_.setDownsampleAllData(true);

        ROS_INFO("table_segmentor has been initialized");
        //test();
    }

private:
    bool SegmentCallback(rls_perception_msgs::TableSegmentation::Request &request,
                       rls_perception_msgs::TableSegmentation::Response &response)
    {
        // input validity check
        if (request.action >= NUM_OF_ACTION) {
            ROS_INFO("table_segmentor: ignore action");
            response.marker = mTableMarker;
            return false;
        }
        if (request.action == USE_EXISTING_RESULT) {
            if (first_time) {
                ROS_WARN("table_segmentor: requested to get existing result but first_time is true");
            } else {
                ROS_INFO("table_segmentor: returning existing result");
                response.marker = mTableMarker;
                return true;
            }
        }

        // get raw point cloud
        sensor_msgs::PointCloud2ConstPtr orig_cloud = ros::topic::waitForMessage<sensor_msgs::PointCloud2>(
                camera_topic, ros::Duration(3)); 
          
        //read point cloud multiple times due to weird intermittent bug where previous image is received 
        // so it goes to locations from previous trial - could be due to network latency
        ros::Time t = ros::Time::now();
        do
        {
            orig_cloud = ros::topic::waitForMessage<sensor_msgs::PointCloud2>(camera_topic, ros::Duration(9.0)); 
            ros::Duration(1/5.0).sleep();
            if ( ros::Time::now() - t > ros::Duration(15.0) ) //wait for at most 15s
            {
              //ROS_ERROR("timeout waiting for pointcloud head_camera/depth_registered/points.");
              ROS_ERROR("timeout waiting for pointcloud kinect2/qhd/points.");
              break;
            }
        } while(orig_cloud == NULL);

        // transform pointcloud to base_link
        tf_listener.waitForTransform("base_link", orig_cloud->header.frame_id,
                                   ros::Time(0), ros::Duration(2.0));
        tf::StampedTransform transform;
        try {
            tf_listener.lookupTransform("base_link", orig_cloud->header.frame_id,
                                        ros::Time(0), transform);
        } catch (tf::LookupException& e) {
            std::cerr << e.what() << std::endl;
        } catch (tf::ExtrapolationException& e) {
            std::cerr << e.what() << std::endl;
        }

        sensor_msgs::PointCloud2 cloud_out;
        pcl_ros::transformPointCloud("base_link", transform, *orig_cloud, cloud_out);

        // convert ros msg to pcl
        PointCloudC::Ptr cloud(new PointCloudC());
        pcl::fromROSMsg(cloud_out, *cloud);


          //std::cout<<"before downsample "<<std::endl;
          //int scale = 2;
          //PointCloudC::Ptr cloud_down(new PointCloudC());
          //cloud_down->width = int(cloud->width / scale);
          //cloud_down->height = int(cloud->height / scale);
          //cloud_down->points.resize(cloud_down->height * cloud_down->width);

          //std::cout<<cloud->height<<", "<<cloud->width<<std::endl;
        

          // direction of push back point is from left to right along x axis in image plane then top to bottom along y axis
    //      for (int ii=0; ii<cloud->height; ii+=2)
    //             for (int jj=0; jj<cloud->width; jj+=2)
    //      cloud_down->at(int(jj/scale), int(ii/scale)) = cloud->at(jj, ii);

          //cloud_down->width = cloud->width / scale;
          //cloud_down->height = cloud->height / scale;
        
            //std::cout<<"after downsample "<<cloud_down->height<<", "<<cloud_down->width<<std::endl;
          //for (int ii=0; ii<cloud_down->height * cloud_down->width; ii++)
         //         std::cout<<"after downsample "<<cloud_down->at(ii)<<std::endl;
          // get table info
        // perception::Segmenter segmenter(table_pub, above_table_pub, marker_pub, objPose_pub, objMarker_pub, table_marker_pub);
          //pcl::ModelCoefficients::Ptr coeff(new pcl::ModelCoefficients());
          
        ROS_INFO("First time, segmenting table!");
        pcl::PointIndices::Ptr table_inliers(new pcl::PointIndices());
        // visualization_msgs::Marker table_marker;
        GetSurface(cloud, table_inliers, coeff_, mTableMarker);
        //segmenter.get_surface(cloud_down, table_inliers, coeff_);
        response.marker = mTableMarker;
        
        first_time = false;
        return true;
    }


    bool GetSurface(PointCloudC::Ptr cloud_in, pcl::PointIndices::Ptr indices,
                    pcl::ModelCoefficients::Ptr coeff, visualization_msgs::Marker& table_marker) 
    {
        PointCloudC::Ptr cloud(new PointCloudC());
        std::vector<int> index;
        pcl::removeNaNFromPointCloud(*cloud_in, *cloud, index);
        //std::cout<<"get_surface function shall only be called ONCE ..."<<std::endl;
        pcl::PointIndices indices_internal; 
        pcl::SACSegmentation<PointC> seg;
        seg.setOptimizeCoefficients(true);

        // Search for a plane perpendicular to some axis that is specified below
        seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        // Set the distance in metres to the plane for a point to be an inlier
        seg.setDistanceThreshold(0.002);
        seg.setInputCloud(cloud);

        // Make sure the plane is perpendicular to Z-axis, 10 deg tolerance
        Eigen::Vector3f axis;
        axis << 0, 0, 1;
        seg.setAxis(axis);
        seg.setEpsAngle(pcl::deg2rad(20.0));

        seg.segment(indices_internal, *coeff);
        //
        // Build custom indices that ignores points above the plane.
        double distance_above_plane;
        ros::param::param("distance_above_plane", distance_above_plane, 0.0); //in metres
        for (size_t i = 0; i < cloud->size(); ++i) { 
            const PointC& pt = cloud->points[i];
            float val = coeff->values[0] * pt.x + coeff->values[1] * pt.y +
                coeff->values[2] * pt.z + coeff->values[3];

            float dist = pt.x * pt.x + pt.y * pt.y + pt.z * pt.z;

            if (val <= distance_above_plane && dist <= MAX_DISTANCE_TO_CAMERA && pt.z >= 0) {
                indices->indices.push_back(i); //indices needs to be empty before entering for loop
            }
        }

        if (indices->indices.size() == 0) {
            ROS_ERROR("Unable to find surface.");
            return false;
        }

        // extract table cloud
        PointCloudC::Ptr cloud_table(new PointCloudC());
        pcl::ExtractIndices<PointC> extract;
        extract.setInputCloud(cloud);
        extract.setIndices(indices);
        extract.setNegative(false);
        extract.filter(*cloud_table);

        // publish table marker
        // visualization_msgs::Marker table_marker;
        table_marker.ns = "table";
        table_marker.header.frame_id = "base_link";
        table_marker.type = visualization_msgs::Marker::CUBE;
        GetAxisAlignedBoundingBox(cloud_table, &table_marker.pose, &table_marker.scale);

        table_marker.color.r = 1;
        table_marker.color.a = 0.8;
        // table_marker_pub_.publish(table_marker);
        // std::cout<<"table published!"<<std::endl;
        return true;
    }

    // Computes the axis-aligned bounding box of a point cloud.
    //
    // Args:
    //  cloud: The point cloud
    //  pose: The output pose. Because this is axis-aligned, the orientation is just
    //    the identity. The position refers to the center of the box.
    //  dimensions: The output dimensions, in meters.
    void GetAxisAlignedBoundingBox(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud,
                                   geometry_msgs::Pose* pose,
                                   geometry_msgs::Vector3* dimensions) 
    {
        Eigen::Vector4f min_pt, max_pt;
        pcl::getMinMax3D(*cloud, min_pt, max_pt);

        pose->position.x = (max_pt.x() + min_pt.x()) / 2;
        pose->position.y = (max_pt.y() + min_pt.y()) / 2;
        pose->position.z = (max_pt.z()) / 2; // here min_z is set to be 0 so that table box touches the ground
        pose->orientation.w = 1;

        dimensions->x = max_pt.x() - min_pt.x();
        dimensions->y = max_pt.y() - min_pt.y();
        dimensions->z = max_pt.z(); // here min_z is set to be 0 so that table box touches the ground
    }

};
}//perception namespace

// this program is a node
int main(int argc, char** argv) 
{
    ros::init(argc, argv, "table_segmentor");
    ros::NodeHandle nh;

    perception::TableSegmentor table_segmentor(nh);//will call constructor
    ros::spin();
    return 0;
}
