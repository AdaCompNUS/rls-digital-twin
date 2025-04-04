cmake_minimum_required(VERSION 2.8.3)
project(fetch_drivers)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  geometry_msgs
  message_generation
  tf2_ros
  tf2
  tf2_geometry_msgs
  sensor_msgs
  nav_msgs
)

catkin_python_setup()

# add_service_files(
#     FILES
#     CartesianGoal.srv
# )

generate_messages(
    DEPENDENCIES
    std_msgs
    geometry_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES fetch_api
  CATKIN_DEPENDS roscpp rospy
  std_msgs
  geometry_msgs
  message_generation
  tf2_ros
  tf2
  tf2_geometry_msgs
  sensor_msgs
  nav_msgs
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

## Declare a C++ library
# add_library(fetch_api
#   src/${PROJECT_NAME}/fetch_api.cpp
# )

## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
# add_dependencies(fetch_api ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

# C++ executable for whole body controller has been replaced by Python implementation

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
install(PROGRAMS
  src/fetch_whole_body_controller/whole_body_controller_mpc.py
  scripts/whole_body_controller_mpc
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Mark executables and/or libraries for installation
# C++ executable has been removed in favor of Python implementation

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
install(FILES
  launch/whole_body_controller.launch
  launch/whole_body_controller_mpc.launch
  config/whole_body_controller.yaml
  config/whole_body_controller_mpc.yaml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_fetch_api.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
