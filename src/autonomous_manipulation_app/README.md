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
vncserver :2 -geometry 1920x1080 -depth 24 -localhost no
export DEMO_PATH_PICK_PALLINA=/home/mivia/Desktop/SPOT/pick_pallina
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
  -v $DEMO_PATH_PICK_PALLINA/Spot-Spark/src/autonomous_manipulation_app:/home/spot_ws/src/autonomous_manipulation_app \
  -v $DEMO_PATH_PICK_PALLINA/Spot-Spark/src/spot_motion:/home/spot_ws/src/spot_motion \
  -v $DEMO_PATH_PICK_PALLINA/Spot-Spark/src/spot:/home/spot_ws/src/spot \
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
  -v $DEMO_PATH_PICK_PALLINA/Spot-Spark/src/autonomous_manipulation_app:/home/yolo_ws/src/autonomous_manipulation_app \
  -v $DEMO_PATH_PICK_PALLINA/Spot-Spark/src/spot_motion:/home/spot_ws/src/spot_motion \
  -v $DEMO_PATH_PICK_PALLINA/Spot-Spark/src/spot:/home/spot_ws/src/spot \
  yolo_autonomous_manipulation
```


# Run App
```bash
# 1. Lanunch SPOT-Driver
colcon build --symlink-install
source install/setup.bash
ros2 launch spot_driver spot_driver.launch.py   \
    config_file:=/home/spot_ws/src/spot/config_spot/spot.yaml \
    launch_rviz:=True

# Launch for manipulation
docker exec -it spot-autonomous-manipulation-container bash

cd /home/spot_ws
colcon build --packages-select manipulation_interfaces manipulation --symlink-install
source install/setup.bash
ros2 run manipulation manipulation_node

# Launcher for navigation
docker exec -it spot-autonomous-manipulation-container bash

cd /home/spot_ws
colcon build --packages-select spot_motion --symlink-install
source install/setup.bash
ros2 launch spot_motion navigation_launch.py

#  Laumch Spot-Moveit
docker exec -it spot-autonomous-manipulation-container bash

cd /home/spot_ws
colcon build --packages-select spot_motion --symlink-install
source install/setup.bash

# Launcher for coast-map
ros2 launch spot_motion depth_launch.py


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
# Kill switch
Run on the HOST, in its own terminal, before starting the robot. Ctrl+C stops the robot
(cancels every Nav2 / arm / manipulation goal and calls `/stop`) and kills the app in both
containers; the Spot driver keeps running so the robot stays controllable. It is NOT an E-Stop.
```bash
cd ${DEMO_PATH}/Spot-Spark/src/autonomous_manipulation_app
./kill_switch.sh          # Ctrl+C: stop the robot + kill the app
./kill_switch.sh --all    # Ctrl+C: also kill the Spot driver
```
