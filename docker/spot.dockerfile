FROM gazebo

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt install -y \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN wget https://github.com/boston-dynamics/spot-sdk/archive/refs/tags/v5.0.1.tar.gz -O spot-sdk.tar.gz \
    && tar -xzf spot-sdk.tar.gz \
    && mv spot-sdk-5.0.1 spot-sdk \
    && rm spot-sdk.tar.gz

RUN pip3 install --no-cache-dir --upgrade pip \
    && pip3 install --no-cache-dir \
    bosdyn-client==5.0.1 \
    bosdyn-mission==5.0.1 \
    bosdyn-choreography-client==5.0.1 \
    bosdyn-api==5.0.1 \
    bosdyn-core==5.0.1

RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> /root/.bashrc

WORKDIR /root
CMD ["bash"]