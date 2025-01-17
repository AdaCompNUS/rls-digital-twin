# rls_robot_ws
The unified workspace for robot application in Robot Living Studio

# Structure Overview:
The source code is organized into four layers. Libraries -> Drivers -> Services -> Apps

## Libraries Layer:
Library layer contains libraries that all modules can import and use

## Driver Layer:
Driver layer contains bringup code for robots in RLS. One example would be map_server for Navigation in RLS

## Service Layer:
Service layer contains services or primitive capabilities implemented for robot. It is further divided into **perception services**, **control services**, **planning services** and **utility services**

## App Layer
App layer contains entry point for different demos in RLS. For exmaple, one such app would be ingress_demo_app

# Build and run Instruction
## Pre-requisite
Ubuntu18.04. For ubuntu18.04, you can only interact with Fetch robot currently.

## Setup submodules
This repo contains multiple submodules. If you need any of them, please follow the submodule readme to install respective dependencies.

## Build Steps
Step 1: Download this repo
Step 4. Install dependent packages for this repo. Try rosdep.
Step 5. Git init and update desired submodules. You can also leave them all uninitialized.
Step 6. catkin build

# Demo Apps
## Robi the Butler Dmeo App
Follow instruction [here](src/apps/ingress_demo_app/Readme.md) to launch Robi the Butler Demo App

