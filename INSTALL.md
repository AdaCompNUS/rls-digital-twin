# Installation Guide

This guide provides comprehensive instructions for setting up the development environment for the Digital Twin project. Native Ubuntu 20.04 setup is strongly recommended for the best performance. If you are using macOS or Windows, a virtual machine setup is included as a section below.

## Native Ubuntu 20.04 Installation (Recommended)

### Prerequisites

Ensure you have the following installed or configured:

- Ubuntu 20.04 LTS Desktop
- At least 50GB free disk space
- Minimum 8GB RAM (16GB recommended)
- CPU with virtualization support enabled

### Environment Setup

1. **Update and Install Required Packages**:

   ```bash
   sudo apt update && sudo apt install -y build-essential cmake curl ffmpeg git gnupg2 \
       libatlas-base-dev libboost-filesystem-dev libboost-program-options-dev \
       libboost-system-dev libboost-test-dev libeigen3-dev libflann-dev \
       libfreeimage-dev libgflags-dev libglew-dev libgoogle-glog-dev \
       libmetis-dev libprotobuf-dev libqt5opengl5-dev libsqlite3-dev \
       libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libgtk-3-dev \
       nano protobuf-compiler python-is-python3 python3-pip qtbase5-dev \
       software-properties-common sudo unzip vim-tiny wget libhdf5-dev
   ```

2. **Install ROS Noetic**:

   ```bash
   sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
   curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
   sudo apt update
   sudo apt install -y ros-noetic-desktop-full
   ```

3. **Initialize `rosdep`**:

   ```bash
   sudo apt install -y python3-rosdep
   sudo rosdep init
   rosdep update
   ```

4. **Setup ROS Environment**:

   ```bash
   echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
   source ~/.bashrc
   ```

5. **Install Additional ROS Packages**:

   ```bash
   sudo apt install -y ros-noetic-ros-numpy \
       ros-noetic-moveit \
       ros-noetic-ros-control \
       ros-noetic-ros-controllers \
       ros-noetic-rviz-imu-plugin
   ```

6. **Clone and Setup Fetch Robot Repositories**:

   ```bash
   mkdir -p ~/catkin_ws/src
   cd ~/catkin_ws/src
   git clone https://github.com/RobotIL-rls/fetch_ros.git
   git clone https://github.com/RobotIL-rls/fetch_gazebo.git
   cd fetch_gazebo
   git checkout gazebo11
   cd ~/catkin_ws/src
   git clone https://github.com/RobotIL-rls/robot_controllers.git
   ```

7. **Build the Catkin Workspace**:

   ```bash
   cd ~/catkin_ws
   rosdep install --from-paths src --ignore-src -r -y
   catkin_make
   echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
   source ~/.bashrc
   ```

8. **Clone and Setup the Digital Twin Repository**:

   ```bash
   # In anyplace you like
   git clone https://github.com/AdaCompNUS/rls-digital-twin.git
   cd rls-digital-twin
   ```

9. **Download Required Models and Meshes**:

   The meshes and models are stored in two locations:

   a. From nBox (requires NUS account):
   [Download Link](https://nusu-my.sharepoint.com/:u:/g/personal/tianrun_nus_edu_sg/EYZGSOMYA59Phdb_twMSETkB09yb540NqVwjMgn9cR0jmQ)

   b. From Dropbox (requires NUS account):
   [Download Link](https://www.dropbox.com/scl/fo/t07x7b0d9kts21gyi2d8y/AA9Aq2I6uk70rfKt75SZD1M?rlkey=bphh5xz01je5tdcokymkq33k8&st=xxydhhtu&dl=0)

   After downloading, place both `models` folders in:

   ```bash
   {workspace}/rls-digital-twin/rls_fetch_ws/src/apps/low_level_planning
   ```

10. **Install Python Dependencies**:

   ```bash
   python3 -m pip install --upgrade pip setuptools
   python3 -m pip install -e .
   ```

11. **Build the ROS Digital Twin Workspace**:

   ```bash
   cd {workspace}/rls-digital-twin/rls_fetch_ws
   rosdep install --from-paths src --ignore-src -r -y
   catkin_make
   echo "source $(pwd)/devel/setup.bash" >> ~/.bashrc
   source ~/.bashrc
   ```

## Virtual Machine Setup (For macOS and Windows Users)

### Prerequisites

- [VirtualBox](https://www.virtualbox.org/wiki/Downloads)
- [Ubuntu 20.04 LTS Desktop ISO](https://releases.ubuntu.com/20.04/)

### VM Configuration

1. **Create a Virtual Machine**:
   - Name: "RLS_Development"
   - Type: "Linux"
   - Version: "Ubuntu (64-bit)"
   - Memory size: 8192 MB (or half your system RAM)
   - Virtual hard disk: 50GB (VDI, dynamically allocated)

2. **Configure VM Settings**:
   - System > Processor: Assign at least 2 CPU cores
   - Display > Video Memory: 128 MB, Enable 3D Acceleration
   - Network > Adapter 1: NAT

3. **Install Ubuntu 20.04**:
   - Start the VM and select your Ubuntu ISO
   - Follow the installation wizard
   - Choose "Minimal installation" and "Install third-party software"

4. **Follow Native Installation Instructions**:
   After setting up Ubuntu, proceed with the steps in the **Native Ubuntu 20.04 Installation** section above.

## Running the Environment

If you encounter any issues, check the following:

- All ROS environment variables are properly set
- All workspaces are properly sourced
- Required models and meshes are in the correct locations
- The ROS do not work well with conda. Please disable your conda environment before installing

For additional help, consult the project's issue tracker or contact the [maintainers](tianrun@nus.edu.sg).
