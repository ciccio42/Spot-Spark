# Running the application

[← Back to README](../README.md)

This page launches the full **person / robot following** pipeline on the real Spot. It assumes the images are built and the model weights are in place ([Installation](installation.md)).

- [Overview of the terminals](#overview-of-the-terminals)
- [Start the containers](#start-the-containers)
- [Terminal 1: Spot driver](#terminal-1-spot-driver-spot-container)
- [Terminal 2: Detector](#terminal-2-detector-yolo-container)
- [Terminal 3: Tracking FSM](#terminal-3-tracking-fsm-spot-container)
- [Terminal 4: 3D pose estimation](#terminal-4-3d-pose-estimation-spot-container)
- [Terminals 5–7: Motion with Nav2](#terminals-57-motion-with-nav2-spot-container)
- [Alternative: motion through the Spot SDK](#alternative-motion-through-the-spot-sdk)
- [Visualisation and monitoring](#visualisation-and-monitoring)
- [Stopping](#stopping)

---

## Overview of the terminals

| # | Container | Command | Role |
|---|---|---|---|
| 1 | `spot` | `ros2 launch spot_driver spot_driver.launch.py ...` | Connects to Spot; publishes cameras, depth, TF, odom; accepts `/cmd_vel` |
| 2 | `yolo` | `ros2 run detector_package detector_node ...` | Serves the `detect` service (YOLOE + OSNet embeddings) |
| 3 | `spot` | `ros2 run demo_package tracking_fsm` | Finds, locks and tracks the target; publishes `target_info` |
| 4 | `spot` | `ros2 run spot_motion pose_3d_estimation` | `target_info` → `target_3d` (3D point in the camera frame) |
| 5 | `spot` | `ros2 launch spot_motion depth_launch.py` | Depth images → point clouds for the Nav2 costmaps |
| 6 | `spot` | `ros2 launch spot_motion navigation_launch.py` | Nav2 servers, lifecycle manager, costmap refresher |
| 7 | `spot` | `ros2 run spot_motion nav2_bridge` | `target_3d` → Nav2 goals and in-place rotation |

The perception chain (1–4) does not move the robot. It is safe to start it alone first and check the debug image before you start the motion terminals (5–7).

> **Safety.** Keep the E-Stop within reach whenever motion nodes run. Start with a large empty area and conservative speeds (see [Configuration](configuration.md#motion)).

---

## Start the containers

Allow containers to use the X server (once per host session):

```bash
xhost +local:docker   # or: xhost +local:root
```

**`spot` container** (first terminal):

```bash
docker run -it --rm \
  --name spot-container \
  --gpus all \
  --privileged \
  --net=host \
  --ipc=host \
  -e DISPLAY=$DISPLAY \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $SPOT_SPARK/src/trackerApp:/home/spot_ws/src/codice_carmine \
  spot
```

**`yolo` container** (second terminal):

```bash
docker run -it --rm \
  --name yolo-container \
  --runtime=nvidia \
  --gpus all \
  --privileged \
  --net=host \
  --ipc=host \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v $SPOT_SPARK/src/trackerApp:/home/yolo_ws/src \
  spot-yolo
```

Open more shells in a running container with:

```bash
docker exec -it spot-container bash
docker exec -it yolo-container bash
```

In **every** shell of the `spot` container, source the workspace:

```bash
cd /home/spot_ws && source install/setup.bash
```

The first time, build the workspaces as described in [Installation → Build the ROS 2 workspaces](installation.md#5-build-the-ros-2-workspaces).

---

## Terminal 1: Spot driver (spot container)

```bash
ros2 launch spot_driver spot_driver.launch.py \
  config_file:=/home/spot_ws/src/codice_carmine/spot/config_spot/spot.yaml \
  launch_rviz:=True
```

With the provided `spot.yaml` the driver claims the lease, powers on and stands Spot (`auto_claim`, `auto_power_on`, `auto_stand` all `True`).

Check the topics the pipeline needs:

```bash
ros2 topic list | grep -E "camera/hand|depth/hand|depth/front|odom"
ros2 topic hz /camera/hand/camera_info
```

`tracking_fsm` subscribes to the **compressed** hand image on `camera/hand/compressed`. If the driver does not publish it, re-publish it from the raw stream in an extra shell:

```bash
ros2 run image_transport republish raw compressed \
  -r in:=/camera/hand/image \
  -r out:=/camera/hand/compressed
```

---

## Terminal 2: Detector (yolo container)

```bash
cd /home/yolo_ws && source install/setup.bash
cd /home/yolo_ws/src            # working dir where mobileclip_blt.ts is cached
ros2 run detector_package detector_node --ros-args \
  -p model_path:=/home/yolo_ws/src/codice_carmine/yolo/model/yoloe-11s-seg.pt
```

The node logs `Pronto a ricevere richieste.` ("ready to receive requests") once the model is loaded. Check the service from any container:

```bash
ros2 service list | grep detect      # → /detect
```

Optional parameters (see [Configuration → detector_node](configuration.md#detector_node-ros-parameters)): `conf_threshold`, `imgsz`, `use_tracker`, `tracker_config`, `reid_model_path`.

---

## Terminal 3: Tracking FSM (spot container)

```bash
ros2 run demo_package tracking_fsm
```

On start-up the node:

1. Connects to Spot through the SDK (only to drive the status **LEDs**). If this fails, the FSM still runs and the LEDs stay unchanged.
2. Stays in **INIT** until the hand camera intrinsics arrive and the `detect` service is available.
3. Moves to **WAITING_TRIGGER** and prints:

   ```text
   >>> Premi INVIO in questo terminale per avviare SEARCH...
   ```

4. **Press Enter in this terminal** to start **SEARCH**. Stand alone in front of the arm camera, **1.5–3.5 m** away (the ROI range). After 10 stable frames the target is locked and the state becomes **TRACKING**.

If the target is lost for more than 30 frames the FSM enters **RECOVERY**. If the target is not found again within 15 s, it goes back to **WAITING_TRIGGER** and you need to press Enter again. See [Architecture → Tracking state machine](architecture.md#tracking-state-machine).

---

## Terminal 4: 3D pose estimation (spot container)

```bash
ros2 run spot_motion pose_3d_estimation
```

Converts each `target_info` (box + depth) into `target_3d` (3D point in the hand camera optical frame + yaw). Check it:

```bash
ros2 topic echo /target_3d
```

---

## Terminals 5–7: Motion with Nav2 (spot container)

This is the default motion back-end. Nav2 runs reactively in the `odom` frame, with no map and no AMCL.

**Terminal 5: depth to point clouds** (feeds the costmaps):

```bash
ros2 launch spot_motion depth_launch.py
```

This creates `/depth/{frontleft,frontright,hand}/points`.

**Terminal 6: Nav2 stack:**

```bash
ros2 launch spot_motion navigation_launch.py
```

This starts `planner_server`, `controller_server` (Regulated Pure Pursuit), `behavior_server`, `bt_navigator` (with `bt/follow_point_spot.xml`), `velocity_smoother`, `lifecycle_manager` and `costmap_refresher`, all configured by `config/nav2_params_spot_real.yaml`.

> `costmap_refresher` does not start because of a wrong entry point. See [Troubleshooting → Known issues](troubleshooting.md#known-issues). Nav2 works without it, because the behaviour tree also clears the local costmap every 2 s.

**Terminal 7: Nav2 bridge:**

```bash
ros2 run spot_motion nav2_bridge
# example with parameters:
ros2 run spot_motion nav2_bridge --ros-args -p target_distance:=2.0 -p rot_gain:=1.0
```

Behaviour:
- **NAVIGATE**: the target is farther than `target_distance + enter_margin` (default 2.9 m). Nav2 drives to a point `target_distance` away from it, facing it, and the goal is updated live via `/goal_update`.
- **HOLD**: the target is inside the distance band (or too close). Spot never backs up. It only rotates in place to keep the target centred in the camera.
- If no target arrives for 5 s, the current navigation is cancelled.

Velocity commands go `controller_server → cmd_vel_nav → velocity_smoother → /cmd_vel → spot_driver`.

---

## Alternative: motion through the Spot SDK

`spot_motion` (the node `spot_motion.spot_motion`) skips Nav2 and `cmd_vel`. It sends `synchro_trajectory_command_in_body_frame` commands straight through the Boston Dynamics SDK, with a Kalman filter and short-term extrapolation. It has **no obstacle avoidance**.

Because this node takes the robot **lease** itself, start the driver with `auto_claim`, `auto_power_on` and `auto_stand` set to `False` in `spot.yaml`. Then run terminals 1–4 as above, and **instead of** terminals 5–7:

```bash
ros2 run spot_motion spot_motion
```

Set `DRY_RUN = True` at the top of `spot_motion/spot_motion.py` to log the commands without sending them. That is a good first test.

---

## Visualisation and monitoring

**Tracking debug image** (ROI rectangle, all boxes with labels, FSM state):

```bash
ros2 run rqt_image_view rqt_image_view /person_follow/hand_debug/compressed
```

The debug image is only rendered while something subscribes to it, so it costs nothing when no viewer is open.

**RViz with the following layout** (Nav2 costmaps, plans, filtered target, stand-off goal):

```bash
rviz2 -d /home/spot_ws/src/codice_carmine/spot_nav_2.rviz
```

Useful debug topics from `nav2_bridge`: `/nav2_bridge/filtered_target` and `/nav2_bridge/standoff_goal`.

**LED feedback on the robot** (if the SDK connection works):

| FSM state | Spot LED behaviour |
|---|---|
| INIT / WAITING_TRIGGER | robot default |
| SEARCH | `internal_autonomous_operation` (white, pulsing) |
| TRACKING | `internal_wait_for_entity` (green, pulsing) |
| RECOVERY | `internal_autonomous_navigation` (green, blinking) |

To list the behaviours available on your robot:

```bash
export BOSDYN_CLIENT_USERNAME=<user>
export BOSDYN_CLIENT_PASSWORD=<password>
python3 /home/spot_ws/src/codice_carmine/color.py            # list
python3 /home/spot_ws/src/codice_carmine/color.py NAME 5     # try NAME for 5 s
```

---

## Stopping

1. Stop the motion nodes first (`Ctrl+C` in terminals 7 → 5). `spot_motion` sends a stop command on exit.
2. Stop `tracking_fsm` (this also stops the LED behaviour) and the detector.
3. Stop the driver last. If needed, sit the robot down from the tablet / controller.
4. Leave the containers with `exit`. They are started with `--rm`, so they are deleted, while the mounted source and the build folders stay on the host.

Want to develop without the robot? See [Testing with rosbags →](rosbag_testing.md)
