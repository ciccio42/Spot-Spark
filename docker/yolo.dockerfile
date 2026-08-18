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
RUN pip3 install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Installa Ultralytics e NumPy dal PyPI standard
RUN pip3 install --no-cache-dir \
    ultralytics \
    numpy

# Creazione del workspace dedicato a YOLO
RUN mkdir -p /home/yolo_ws/src
WORKDIR /home/yolo_ws

# Assicura il source di ROS 2
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /root/.bashrc

CMD ["bash"]