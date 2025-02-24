ARG OS_VERSION=20.04
ARG USER_ID=1000
# Define base image.
FROM ubuntu:${OS_VERSION}
ARG OS_VERSION
ARG USER_ID

# Set environment variables.
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Singapore
ENV QT_QUICK_BACKEND=software

# Install required apt packages and clear cache afterwards.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    ffmpeg \
    git \
    gnupg2 \
    libatlas-base-dev \
    libboost-filesystem-dev \
    libboost-program-options-dev \
    libboost-system-dev \
    libboost-test-dev \
    libhdf5-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libgflags-dev \
    libglew-dev \
    libgoogle-glog-dev \
    libmetis-dev \
    libprotobuf-dev \
    libqt5opengl5-dev \
    libsqlite3-dev \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libgtk-3-dev \
    nano \
    protobuf-compiler \
    python-is-python3 \
    python3-pip \
    qtbase5-dev \
    software-properties-common \
    sudo \
    unzip \
    vim-tiny \
    wget && \
    rm -rf /var/lib/apt/lists/*

# Install Git LFS
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get update && \
    apt-get install -y git-lfs && \
    rm -rf /var/lib/apt/lists/*

RUN add-apt-repository ppa:kisak/kisak-mesa && \
    apt-get update

# Install ROS Noetic and Gazebo
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' && \
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add - && \
    apt-get update && \
    apt-get install -y ros-noetic-desktop-full && \
    rm -rf /var/lib/apt/lists/*

# Initialize rosdep
RUN apt-get update && apt-get install -y python3-rosdep

RUN rosdep init && \
    rosdep update

# Setup ROS environment
RUN echo "source /opt/ros/noetic/setup.bash" >> /etc/bash.bashrc
ENV ROS_PACKAGE_PATH=/opt/ros/noetic/share

# Install additional packages for ros
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ros-noetic-ros-numpy \
    ros-noetic-moveit \
    ros-noetic-ros-control \
    ros-noetic-ros-controllers \
    ros-noetic-rviz-imu-plugin

# Create non-root user and setup environment.
RUN useradd -m -d /home/user -g root -G sudo -u ${USER_ID} user
RUN usermod -aG sudo user

# Set user password
RUN echo "user:user" | chpasswd

# Ensure sudo group users are not asked for a password when using sudo command by amending sudoers file
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Switch to new user and workdir.
USER ${USER_ID}
WORKDIR /home/user

# Add local user binary folder to PATH variable.
ENV PATH="${PATH}:/home/user/.local/bin"
SHELL ["/bin/bash", "-c"]

# Upgrade pip and install necessary Python packages.
RUN python3 -m pip install --upgrade pip setuptools

# Clone Fetch robot repositories and checkout to gazebo11 branch
RUN mkdir -p ~/catkin_ws/src && \
    cd ~/catkin_ws/src && \
    git clone https://github.com/RobotIL-rls/fetch_ros.git && \
    git clone https://github.com/RobotIL-rls/fetch_gazebo.git && \
    cd fetch_gazebo && \
    git checkout gazebo11 && \
    cd ~/catkin_ws/src && \
    git clone https://github.com/RobotIL-rls/robot_controllers.git

# Install dependencies and build the workspace
RUN source /opt/ros/noetic/setup.bash && \
    cd ~/catkin_ws && \
    rosdep update && \
    rosdep install --from-paths src --ignore-src -r -y && \
    catkin_make

# Source the workspace setup
RUN echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc

# Change working directory
WORKDIR /workspace

# Initialize Git LFS for the user
RUN git lfs install

# Bash as default entrypoint.
CMD /bin/bash -l
