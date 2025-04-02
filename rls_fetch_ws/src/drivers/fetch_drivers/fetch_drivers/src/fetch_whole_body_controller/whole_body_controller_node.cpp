#include "whole_body_controller.h"
#include <ros/ros.h>

int main(int argc, char** argv)
{
    // Initialize ROS
    ros::init(argc, argv, "fetch_whole_body_controller");
    
    // Create node handle
    ros::NodeHandle nh;
    
    ROS_INFO("Starting Fetch Whole Body Controller node");
    
    // Create controller
    fetch_drivers::WholeBodyController controller(nh);
    
    // Spin to process callbacks
    ros::spin();
    
    return 0;
} 