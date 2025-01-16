# Installation Guide

This guide provides instructions for setting up the development environment for the Digital Twin project. Native Ubuntu 20.04 setup is strongly recommended for the best performance. If you are using macOS or Windows, a virtual machine setup is included as a section below.

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
       software-properties-common sudo unzip vim-tiny wget
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

5. **Clone and Setup the Digital Twin Repository**:

   ```bash
   cd ~
   git clone https://github.com/AdaCompNUS/rls-digital-twin.git
   cd rls-digital-twin
   ```

6. **Downloading Meshes and Models**

   The meshes and models required for the project are stored on nBox. Please download them manually using a NUS account via the following link:

   [Download Link](https://nusu-my.sharepoint.com/:u:/g/personal/tianrun_nus_edu_sg/EYZGSOMYA59Phdb_twMSETkB09yb540NqVwjMgn9cR0jmQ)

   After downloading, place the `models` folder into the following directory:

   ```bash
   {workspace}/rls-digital-twin/rls_fetch_ws/src/apps/low_level_planning
   ```

7. **Install Python Dependencies**:

   ```bash
   python3 -m pip install --upgrade pip setuptools
   pip install -e .
   ```

8. **Build the ROS Workspace**:

   ```bash
   cd ~/rls-digital-twin/rls_fetch_ws
   rosdep install --from-paths src --ignore-src -r -y
   catkin_make
   echo "source $(pwd)/devel/setup.bash" >> ~/.bashrc
   source ~/.bashrc
   ```

9. **Download Required Models**:

   Download models manually from Dropbox:

   [Download Link](https://www.dropbox.com/scl/fo/t07x7b0d9kts21gyi2d8y/AA9Aq2I6uk70rfKt75SZD1M?rlkey=bphh5xz01je5tdcokymkq33k8&st=xxydhhtu&dl=0)

   Place the `models` folder in:

   ```bash
   {workspace}/rls-digital-twin/rls_fetch_ws/src/apps/low_level_planning
   ```

---

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

---

## Running the Environment

Once the setup is complete, follow the instructions in the main README to start using the digital twin environment.
