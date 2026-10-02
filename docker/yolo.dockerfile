# Inherits from the 'ros2' image built earlier
FROM ros2

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies (git-lfs added)
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-opencv \
    ros-${ROS_DISTRO}-cv-bridge \
    git \
    git-lfs \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 1. Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip

# 2. Install PyTorch with CUDA support from the PyTorch wheel index
RUN python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --break-system-packages --no-cache-dir --ignore-installed

# 3. Install Ultralytics and NumPy from the standard PyPI
RUN pip3 install --no-cache-dir \
    ultralytics

RUN pip install --break-system-packages \
    tensorboard \
    gdown \
    huggingface_hub

RUN pip install --break-system-packages cython 'setuptools<71' 
RUN pip install --break-system-packages 'numpy==1.26.4'
RUN pip install --break-system-packages --no-build-isolation \
    https://github.com/KaiyangZhou/deep-person-reid/archive/refs/heads/master.zip

# Create the workspace dedicated to YOLO
RUN mkdir -p /home/yolo_ws/src
WORKDIR /home/yolo_ws

# Download the model from Hugging Face via git-lfs (avoids a Python decompression bug)
RUN git lfs install && git clone https://huggingface.co/kaiyangzhou/osnet /home/yolo_ws/osnet

# Build, source the workspace and finally downgrade/fix numpy
RUN /bin/bash -c "source /opt/ros/${ROS_DISTRO}/setup.bash && \
    colcon build --symlink-install --packages-ignore champ_gazebo && \
    source install/setup.bash && \
    pip install --break-system-packages 'numpy<2'"

# Make sure both ROS 2 and the built workspace are sourced automatically
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /root/.bashrc && \
    echo "source /home/yolo_ws/install/setup.bash" >> /root/.bashrc

CMD ["bash"]