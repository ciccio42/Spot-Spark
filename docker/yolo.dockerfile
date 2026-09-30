# Eredita dall'immagine 'ros2' che hai compilato prima
FROM ros2

ENV DEBIAN_FRONTEND=noninteractive

# Installazione dipendenze di sistema (aggiunto git-lfs)
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-opencv \
    ros-${ROS_DISTRO}-cv-bridge \
    git \
    git-lfs \
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

# Download del modello da Hugging Face tramite git-lfs (evita bug di decompressione Python)
RUN git lfs install && git clone https://huggingface.co/kaiyangzhou/osnet /home/yolo_ws/osnet

# Esegue il build, il source del workspace e infine il downgrade/fix di numpy
RUN /bin/bash -c "source /opt/ros/${ROS_DISTRO}/setup.bash && \
    colcon build --symlink-install --packages-ignore champ_gazebo && \
    source install/setup.bash && \
    pip install --break-system-packages 'numpy<2'"

# Assicura il source automatico sia di ROS 2 che del workspace compilato
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /root/.bashrc && \
    echo "source /home/yolo_ws/install/setup.bash" >> /root/.bashrc

CMD ["bash"]