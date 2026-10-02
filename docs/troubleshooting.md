# Troubleshooting

[← Back to README](../README.md)

- [Known issues](#known-issues)
- [Common problems](#common-problems)
- [Useful commands](#useful-commands)
- [Docker housekeeping](#docker-housekeeping)

---

## Known issues

These issues are in the current code. Each one comes with a workaround.

| # | Issue | Effect | Workaround / fix |
|---|---|---|---|
| 1 | `spot_motion/setup.py` declares `costmap_refresher = spot_motion.costmap_refresh:main`, but the module is `costmap_refresher.py`. | `ros2 run spot_motion costmap_refresher` and the `costmap_refresher` node in `navigation_launch.py` fail with `ModuleNotFoundError`. | Change the entry point to `spot_motion.costmap_refresher:main` and rebuild `spot_motion`. Nav2 still works without it: the BT clears the local costmap every 2 s. |
| 2 | `spot_motion/setup.py` declares `nav2_backup = spot_motion.nav2_backup:main`, but `nav2_backup` has no `.py` extension. | `ros2 run spot_motion nav2_backup` fails. | It is an older version of the bridge kept for reference. Rename it to `nav2_backup.py` if you need it. |
| 3 | `demo_package/setup.py` declares `nav2_bridge = demo_package.nav2_bridge:main`, but the module does not exist. | `ros2 run demo_package nav2_bridge` fails. | Use `ros2 run spot_motion nav2_bridge`. |
| 4 | `detector_node` defaults to `model_path=/models/yoloe-11s-seg.pt`, but nothing is mounted at `/models`. | The node fails to load the model. | Pass `-p model_path:=...` (see [Running](running.md#terminal-2-detector-yolo-container)) or add `-v <host_dir>:/models` to `docker run`. |
| 5 | With `TRACKING_METHOD = "botsort_hsv"`, `target_info` is never published. | The FSM tracks but the robot does not move. | Use `embedding_only`, or add `_publish_target_info()` calls to the `botsort_hsv` handlers. |
| 6 | `nav2_bridge.py` and `nav2_bridge_v2.py` are identical. | None | Remove one, or use `v2` for experiments. |
| 7 | Credentials are hard-coded and committed. | Security risk | See [Configuration → Robot connection](configuration.md#robot-connection). |

---

## Common problems

### `Servizio 'detect' non ancora disponibile` / FSM stuck in INIT

`tracking_fsm` cannot see the detector service.

- Is `detector_node` running and has it printed that it is ready? Loading the model takes a few seconds.
- Do both containers use `--net=host` and the same `ROS_DOMAIN_ID`? Check with `ros2 service list` in the spot container.
- The FSM also waits for `/camera/hand/camera_info`: `ros2 topic hz /camera/hand/camera_info`.

### SEARCH never locks

The debug image shows why each box was rejected:

| Label in the debug image | Meaning | What to do |
|---|---|---|
| `(confidenza ... troppo bassa)` | YOLOE score below `MIN_DETECTION_CONFIDENCE` | Improve lighting, or lower the threshold |
| `(depth n/d)` | No depth image received yet | Check `/depth/hand/image` |
| `(depth invalida)` | Too few valid depth pixels at the box centre | Target too far or reflective. ToF depth is noisy beyond about 4 m. |
| `(fuori range)` | Distance outside 1.5–3.5 m | Move closer or farther, or change `CONE_MIN_RANGE` / `CONE_MAX_RANGE` |
| log says `serve esattamente 1` | More than one valid target in the ROI | Only one target may be in the cone while locking |
| log says `nessun embedding ricevuto` | Detector runs without ReID | Set `reid_model_path` on the detector |

### Detector crashes on start-up with numpy errors

torchreid and cv_bridge need **numpy < 2**. Inside the yolo container:

```bash
pip install --break-system-packages "numpy<2"
```

### `mobileclip_blt.ts` download fails

The first `set_classes()` call downloads the MobileCLIP text encoder. Give the container internet access once, or copy an existing `mobileclip_blt.ts` into the node's working directory.

### High latency / low frame rate

- Each frame makes one **synchronous** detection call, so the FSM rate is bounded by the detector latency. The `[timing]` logs of both nodes show decode / inference / embedding times.
- Lower `imgsz`, use a lighter ReID model (`reid_model_name:=osnet_x0_25` with matching weights), or narrow `CONE_FOV_DEG` to shrink the crop.
- Close `rqt_image_view` when you do not need it: the debug image is only rendered while someone subscribes.

### Point clouds at about 1.5 Hz

`depth_launch.py` already sets `approximate_sync: True`, which fixes the timestamp mismatch of the Spot wrapper (it raises the rate above 10 Hz). If you write your own launch file, keep that parameter.

### Nav2 nodes start with default parameters

The lifecycle manager node name must match the top-level `lifecycle_manager:` key in `nav2_params_spot_real.yaml`. Do not rename it to `lifecycle_manager_navigation`.

### `TF non disponibile (... -> odom)` in `nav2_bridge`

The driver is not publishing TF, or the camera frame in `CameraInfo` is not connected to `odom`. Dump the tree:

```bash
ros2 run tf2_tools view_frames      # creates frames_<date>.gv / .pdf
```

Compare with the reference snapshots `src/trackerApp/frames_*.pdf`.

### Robot does not move with the SDK back-end

- `Robot in e-stop`: an E-Stop client must be active (tablet or E-Stop app).
- Lease errors: the driver holds the lease. Set `auto_claim: False` in `spot.yaml`.
- Check that `DRY_RUN = False` in `spot_motion.py`.

### GUI apps do not open (`cannot open display`)

Run `xhost +local:docker` (or `+local:root`) on the host and check that `-e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:rw` are passed to `docker run`.

---

## Useful commands

```bash
# Drive the robot manually at 10 Hz (driver must hold the lease)
ros2 topic pub -r 10 /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.3, y: 0.0, z: 0.0}, angular: {z: 0.0}}"

# Move the arm to a pose (joint trajectory controller, simulation)
ros2 topic pub --once /arm_controller/joint_trajectory trajectory_msgs/msg/JointTrajectory \
"{joint_names: [arm_sh0, arm_sh1, arm_el0, arm_el1, arm_wr0, arm_wr1, arm_f1x],
  points: [{positions: [0.0, -3.1, 3.1, 0.0, 0.0, 0.0, 0.0], time_from_start: {sec: 3}}]}"

# Inspect topics / frequencies
ros2 topic list
ros2 topic hz /camera/hand/compressed
ros2 node list

# Rebuild a single package
colcon build --symlink-install --packages-select spot_motion

# Kill a stuck simulation
pkill -9 -f "ign gazebo"; pkill -9 -f parameter_bridge; pkill -9 -f quadruped_controller; \
pkill -9 -f state_estimation; pkill -9 -f robot_state_publisher; pkill -9 -f ekf_node
```

---

## Docker housekeeping

```bash
# Remove all build cache
docker builder prune --all

# Remove dangling images (left behind by rebuilds)
docker image prune -f

# Show disk usage of images, containers and cache
docker system df

# List the project images
docker images | grep -E "ros2|nav2|gazebo|spot"
```
