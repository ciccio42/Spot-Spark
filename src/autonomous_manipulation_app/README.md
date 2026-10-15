# Autonomous Manipulation APP

# Ros2 packages
1. Autonomous-Manipulation FSM
2. yolo-detection
3. manipulation

# Dependencies
1. spot_motion
2. spot

# Docker files
```bash
export DEMO_PATH=/home/mivia/Desktop/SPOT/pick_pallina
# Contaier for SPOT with navigation
cd ${DEMO_PATH}/Spot-Spark/docker
docker build -t spot_autonomous_manipulation . -f spot.dockerfile
# Container for yolo detection
docker build -t yolo_autonomous_manipulation . -f yolo.dockerfile
```

# Run Containers
```bash
# Autonomous Manipulation
export DISPLAY=:2
xhost +local:docker    
docker run -it --rm \
  --name spot-autonomous-manipulation-container \
  --gpus all \
  --privileged \
  --net=host \
  --ipc=host \
  -e DISPLAY=$DISPLAY \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $DEMO_PATH/Spot-Spark/src/autonomous_manipulation_app:/home/spot_ws/src/autonomous_manipulation_app \
  -v $DEMO_PATH/Spot-Spark/src/spot_motion:/home/spot_ws/src/spot_motion \
  -v $DEMO_PATH/Spot-Spark/src/spot:/home/spot_ws/src/spot \
  spot_autonomous_manipulation

# Yolo Container
export DISPLAY=:2
xhost +local:docker            
docker run -it --rm \
  --name yolo-autonomous-manipulation-container \
  --gpus all \
  --privileged \
  --net=host \
  --ipc=host \
  -e DISPLAY=$DISPLAY \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $DEMO_PATH/Spot-Spark/src/autonomous_manipulation_app:/home/yolo_ws/src/autonomous_manipulation_app \
  -v $DEMO_PATH/Spot-Spark/src/spot_motion:/home/spot_ws/src/spot_motion \
  -v $DEMO_PATH/Spot-Spark/src/spot:/home/spot_ws/src/spot \
  yolo_autonomous_manipulation
```


# Run App
```bash
# 1. Lanunch SPOT-Driver
export ROS_DISTRO=humble
pip install aiortc
apt-get update && apt-get install -y ros-${ROS_DISTRO}-image-transport-plugins
ros2 launch spot_driver spot_driver.launch.py   \
    config_file:=/home/spot_ws/src/spot/config_spot/spot.yaml \
    launch_rviz:=True

# 2. Laumch Spot-Moveit
docker exec -it spot-autonomous-manipulation-container bash

cd /home/spot_ws
colcon build --packages-select spot_motion --symlink-install
source install/setup.bash

# Launcher for coast-map
ros2 launch spot_motion depth_launch.py

# Launcher for navigation
docker exec -it spot-autonomous-manipulation-container bash

source install/setup.bash
ros2 launch spot_motion navigation_launch.py


# Launcher detector
colcon build --packages-select detection_service_msg yolo_detector --symlink-install
source install/setup.bash
ros2 launch yolo_detector detector_launch.py

# Launch the fsm
docker exec -it spot-autonomous-manipulation-container bash
cd /home/spot_ws
colcon build --packages-select detection_service_msg manipulation_interfaces autonomous_manipulation_fsm --symlink-install
source install/setup.bash

ros2 launch autonomous_manipulation_fsm fsm_launch.py

```

# Useful commands
```bash
# create ros2 package
# c++ package
ros2 pkg create --build-type ament_cmake <package_name>
# python package
ros2 pkg create --build-type ament_python <package_name>

# build packages
colcon build --symlink-install
colcon build --symlink-install --packages-select my_package 
```