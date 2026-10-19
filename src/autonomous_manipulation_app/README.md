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

# with the click (default): before every motion the robot waits for the operator's OK,
# a left click anywhere on the yolo_detector window (Enter in this terminal works too);
# if YOLO finds nothing, the target can be clicked
ros2 launch autonomous_manipulation_fsm fsm_launch.py

# without the click: YOLO only, the robot navigates to the detection without asking
# (pick, place and return home are still confirmed with Enter; if YOLO finds nothing -> ERROR)
ros2 launch autonomous_manipulation_fsm fsm_launch.py use_click:=false

# debug: clicks published by the detector window and requests of the FSM (no robot motion needed)
ros2 topic echo /clicked_pixel
ros2 topic echo /click_request
```
The detector (`ros2 launch yolo_detector detector_launch.py`) is launched the same way in both cases.
With the click it needs its window: `show_image: true` in `yolo_detector/config/detector_params.yaml`
and `DISPLAY=:2` in the yolo container.

# Operator OK by click
Before the robot moves, the FSM waits for the operator's OK and shows on the detector window a yellow
banner `CLICK = OK: <what>`. A left click anywhere on the window gives the OK (Enter in the FSM terminal
and `ros2 service call /fsm_node/start std_srvs/srv/Trigger` work too). Clicks made while no banner is
shown are ignored. The OK is asked for:
- `start the cycle` (IDLE): the arm moves to the detection pose
- `navigate to the target (... m away)`: after the far detection, before Nav2 is started; check the green
  TARGET box on the window before clicking
- `start picking`, `move the arm to the safe position and place the object`, `go back to the initial
  position and sit`

To stop instead of confirming: Ctrl+C on the FSM, or the kill switch.

Parameters in `autonomous_manipulation_fsm/config/fsm_params.yaml`:
- `confirm_by_click: true`: a click on the window gives the OK (false: only Enter / service)
- `confirm_navigation: true`: OK before the navigation to the detected target (false: navigates directly)

## Fallback (YOLO finds nothing)
If YOLO does not detect the target within `detection_timeout`, the banner shows
`no detection: click the ...` instead of going to ERROR:
- `click_fallback_navigation: true`: the robot walks towards the clicked point (3D from the hand depth,
  stops `standoff_distance` from it, facing it); this click is already the OK for the navigation
- `click_fallback_pick: true`: the clicked pixel is grasped by the Manipulate action (after the
  `start picking` OK)

Without a click for `click_timeout` seconds the FSM goes to ERROR.

`use_click:=false` sets `confirm_by_click`, `confirm_navigation` and `click_fallback_*` to false.

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
