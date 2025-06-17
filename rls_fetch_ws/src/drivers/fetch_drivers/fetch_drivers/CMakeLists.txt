cmake_minimum_required(VERSION 2.8.3)
project(fetch_drivers)

## Find catkin packages
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

## Generate messages, services, etc.
generate_messages(
    DEPENDENCIES
    std_msgs
    geometry_msgs
)

## Catkin Package
catkin_package(
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

## Build
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

# NOTE: C++ library/executables are commented out as we are using Python implementations.

#############
## Install ##
#############

# Mark executable Python scripts for installation.
# Assumes your scripts are in the 'scripts' directory of your package.
install(PROGRAMS
  # The original MPC controller script
  scripts/whole_body_controller_mpc
  # ADDED: The new Chained Form controller script
  scripts/whole_body_controller_chained
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)


# Mark launch files, config files, etc. for installation.
install(DIRECTORY
  launch/
  config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

# The above install(DIRECTORY ...) is a modern, cleaner way to install all files
# from a directory. If you prefer the old file-by-file method, use the block below.
# I have updated it to include all necessary files.
#
# install(FILES
#   # --- NEW & RECOMMENDED LAUNCH FILE ---
#   launch/start_whole_body_controller.launch
#
#   # --- CONFIG FILES FOR BOTH CONTROLLERS ---
#   config/whole_body_controller_mpc.yaml
#   config/whole_body_controller_chained.yaml # ADDED
#
#   # --- OLD/REDUNDANT FILES (can be removed if no longer needed) ---
#   # launch/whole_body_controller_mpc.launch
#   # config/whole_body_controller.yaml
#   # launch/whole_body_controller.launch
#
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )


#############
## Testing ##
#############
# Testing targets are commented out for this example.