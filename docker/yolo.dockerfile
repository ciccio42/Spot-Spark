# Eredita dall'immagine 'ros2' che hai compilato prima
FROM ros2

ENV DEBIAN_FRONTEND=noninteractive

# Installazione dipendenze di sistema (incluso ros-humble-cv-bridge via apt)
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-opencv \
    ros-${ROS_DISTRO}-cv-bridge \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 1. Aggiorna pip
RUN pip3 install --no-cache-dir --upgrade pip

# 2. Installa PyTorch con supporto CUDA da PyTorch Wheel Index
RUN python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --break-system-packages --no-cache-dir --ignore-installed


# 3. Installa Ultralytics e NumPy dal PyPI standard
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

# Creazione del workspace dedicato a YOLO
RUN mkdir -p /home/yolo_ws/src
WORKDIR /home/yolo_ws



RUN hf download kaiyangzhou/osnet --local-dir /home/yolo_ws/osnet

# Assicura il source di ROS 2
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /root/.bashrc

CMD ["bash"]