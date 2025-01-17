cmake_minimum_required(VERSION 3.0.2)
project(low_level_planning)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  roscpp
  pluginlib
  geometry_msgs
  tf
  robot_controllers_interface
  robot_controllers
)

## Declare a catkin package
catkin_package(
  CATKIN_DEPENDS roscpp pluginlib geometry_msgs tf robot_controllers_interface robot_controllers
)

## Specify additional locations of header files
include_directories(
  ${catkin_INCLUDE_DIRS}
)
