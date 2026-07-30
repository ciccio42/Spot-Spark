FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute
ENV ROS_DISTRO=humble

# Locale
RUN apt update && apt install -y locales \
 && locale-gen en_US en_US.UTF-8 \
 && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8

# Install utilities and OpenGL libraries
RUN apt update && apt install -y \
    curl gnupg2 lsb-release software-properties-common \
    mesa-utils libgl1 libegl1 libglx0 x11-apps \
    && rm -rf /var/lib/apt/lists/*

# ROS 2 repository and install
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg

RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
    > /etc/apt/sources.list.d/ros2.list

RUN apt update && apt install -y ros-$ROS_DISTRO-desktop \
    && rm -rf /var/lib/apt/lists/*

# ROS environment
SHELL ["/bin/bash", "-c"]
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /root/.bashrc

WORKDIR /root
CMD ["bash"]