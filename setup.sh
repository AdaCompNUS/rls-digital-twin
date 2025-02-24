#!/bin/bash

CONDA_ENV_NAME="arobot"
if conda env list | grep -q "^$CONDA_ENV_NAME "; then
    echo "Environment $CONDA_ENV_NAME exists."
else
    echo "Environment $CONDA_ENV_NAME does not exist."
    conda create -n $CONDA_ENV_NAME python=3.8
fi

CONDA_HOME=$(conda info --base)
source $CONDA_HOME/bin/activate $CONDA_ENV_NAME
export PYTHONPATH=$CONDA_PREFIX/lib/python3.8/site-packages:$PYTHONPATH
export PYTHON_EXECUTABLE=$(which python)


# manually install pip
cd $HOME/Downloads
curl -O https://bootstrap.pypa.io/get-pip.py
python3 get-pip.py
echo "-- python and pip path:"
echo $(which python)
echo $(which pip)


sudo apt update && sudo apt install -y build-essential cmake curl ffmpeg git gnupg2 \
    libatlas-base-dev libboost-filesystem-dev libboost-program-options-dev \
    libboost-system-dev libboost-test-dev libeigen3-dev libflann-dev \
    libfreeimage-dev libgflags-dev libglew-dev libgoogle-glog-dev \
    libmetis-dev libprotobuf-dev libqt5opengl5-dev libsqlite3-dev \
    libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libgtk-3-dev \
    nano protobuf-compiler python-is-python3 python3-pip qtbase5-dev \
    software-properties-common sudo unzip vim-tiny wget libhdf5-dev

sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
sudo apt update
sudo apt install -y ros-noetic-desktop-full

sudo apt install -y python3-rosdep
sudo rosdep init
rosdep update

echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
source ~/.bashrc

sudo apt install -y ros-noetic-ros-numpy \
    ros-noetic-moveit \
    ros-noetic-ros-control \
    ros-noetic-ros-controllers \
    ros-noetic-rviz-imu-plugin

source /opt/ros/noetic/setup.bash
source $CONDA_HOME/bin/activate $CONDA_ENV_NAME
export PYTHON_EXECUTABLE=$(which python)

sudo apt-get update
sudo apt-get install python3-empy
pip3 install empy==3.3.4
pip3 install catkin_pkg rospkg defusedxml

mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src
git clone https://github.com/RobotIL-rls/fetch_ros.git
git clone https://github.com/RobotIL-rls/fetch_gazebo.git
cd fetch_gazebo
git checkout gazebo11
cd ~/catkin_ws/src
git clone https://github.com/RobotIL-rls/robot_controllers.git

cd ~/catkin_ws
rosdep install --from-paths src --ignore-src -r -y
catkin_make
echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
source ~/.bashrc
source $CONDA_HOME/bin/activate $CONDA_ENV_NAME
export PYTHON_EXECUTABLE=$(which python)

# In anyplace you like
cd ~/catkin_ws/src
git clone https://github.com/AdaCompNUS/rls-digital-twin/
cd rls-digital-twin
pip install gdown
gdown "https://drive.google.com/uc?export=download&id=1kKSVDvgZIFIveBKQbKgqr0dnM7DFvy-a"
unzip models.zip
mkdir -p rls_fetch_ws/src/apps/low_level_planning
mv models rls_fetch_ws/src/apps/low_level_planning

python3 -m pip install --upgrade pip setuptools
python3 -m pip install -e .

cd $HOME/catkin_ws/src/rls-digital-twin/rls_fetch_ws
rosdep install --from-paths src --ignore-src -r -y
catkin_make
echo "source $(pwd)/devel/setup.bash" >> ~/.bashrc
source ~/.bashrc


echo 'export CONDA_ENV_NAME=arobot' >> ~/.bashrc
echo 'CONDA_HOME=$(conda info --base)' >> ~/.bashrc
echo 'conda activate $CONDA_ENV_NAME' >> ~/.bashrc
echo 'export PYTHON_EXECUTABLE=$(which python)' >> ~/.bashrc
echo 'export PYTHONPATH=$CONDA_PREFIX/lib/python3.8/site-packages:$PYTHONPATH' >> ~/.bashrc
