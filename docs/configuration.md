# Configuration

[← Back to README](../README.md)

The project uses two kinds of configuration:

- **ROS 2 parameters**: change them at launch time with `--ros-args -p name:=value`.
- **In-code constants**: module-level constants at the top of a Python file. Edit the file and re-run the node. With `colcon build --symlink-install`, Python edits apply without rebuilding.

Paths below are relative to `src/trackerApp/`.

- [Robot connection](#robot-connection)
- [Spot driver (`spot.yaml`)](#spot-driver-spotyaml)
- [`detector_node` ROS parameters](#detector_node-ros-parameters)
- [Tracker YAML files](#tracker-yaml-files)
- [ROI and depth range (`common.py`)](#roi-and-depth-range-commonpy)
- [Tracking FSM (`tracking_fsm.py`)](#tracking-fsm-tracking_fsmpy)
- [Motion](#motion)
- [Nav2 parameters and behaviour tree](#nav2-parameters-and-behaviour-tree)

---

## Robot connection

The robot address and credentials are currently set in **three places**. Keep them in sync:

| File | Used by | Keys |
|---|---|---|
| `spot/config_spot/spot.yaml` | `spot_driver` | `hostname`, `username`, `password` |
| `spot/demo_package/demo_package/tracking_fsm.py` → `main()` | LED control | `robot_ip`, `SPOT_USERNAME`, `SPOT_PASSWORD` |
| `spot_motion/spot_motion/spot_motion.py` | SDK motion back-end | `SPOT_HOSTNAME`, `SPOT_USERNAME`, `SPOT_PASSWORD` |

> **Recommended:** do not commit credentials. Read them from the environment, as `color.py` already does:
>
> ```python
> import os
> SPOT_USERNAME = os.environ["BOSDYN_CLIENT_USERNAME"]
> SPOT_PASSWORD = os.environ["BOSDYN_CLIENT_PASSWORD"]
> ```
>
> and pass them to the container with `-e BOSDYN_CLIENT_USERNAME -e BOSDYN_CLIENT_PASSWORD`.

---

## Spot driver (`spot.yaml`)

`spot/config_spot/spot.yaml` is passed to `spot_driver.launch.py` with `config_file:=`. Main entries:

| Parameter | Value | Meaning |
|---|---|---|
| `hostname` | `192.168.80.3` | Robot IP |
| `auto_claim` / `auto_power_on` / `auto_stand` | `True` | Driver takes the lease, powers on and stands. Set all three to **`False`** when you use the SDK motion back-end (`spot_motion`). |
| `robot_state_rate` | `50.0` Hz | Joint states, TF, odometry |
| `image_rate` | `15.0` Hz | Camera images (Spot rarely goes beyond 15 Hz) |
| `preferred_odom_frame` / `tf_root` | `odom` | Root of the TF tree. Nav2 and `nav2_bridge` also work in `odom`. |
| `cmd_duration` | `0.7` s | How long a `/cmd_vel` command stays valid. Raise it if Spot stutters. |
| `rgb_cameras` | `True` | Set to `False` on robots with greyscale body cameras |
| `cameras_used` | (commented) | Uncomment to publish only some cameras |
| `gripperless` | `False` | Set to `True` if the arm has no gripper |

---

## `detector_node` ROS parameters

Set with `ros2 run detector_package detector_node --ros-args -p <name>:=<value>`.

| Parameter | Default | Description |
|---|---|---|
| `model_path` | `/models/yoloe-11s-seg.pt` | Local path of the YOLOE weights. Nothing is downloaded at run time. |
| `default_classes` | `[person, quadruped, quadruped animal, quadruped robot, robotic dog, four-legged robot, dog, robot, umanoid robot]` | Text prompts used when a request has no `target_classes` |
| `conf_threshold` | `0.35` | YOLOE confidence threshold |
| `imgsz` | `640` | Inference image size |
| `service_name` | `detect` | Name of the `Detect` service |
| `use_tracker` | `False` | `True` → `model.track()` with persistent `track_id`. `False` → `model.predict()`, `track_id = -1`. |
| `tracker_config` | `/home/yolo_ws/src/yolo/detector_package/oc_sort.yaml` | Tracker YAML (only used with `use_tracker:=true`) |
| `reid_model_name` | `osnet_x1_0` | torchreid architecture (`osnet_x0_25` is lighter) |
| `reid_model_path` | `/home/yolo_ws/osnet/osnet_x1_0_imagenet.pth` | ReID weights. An empty string disables ReID, so `embedding` is always empty. **`embedding_only` tracking needs it.** |

> `tracking_fsm` sends its own class list in each request (`TARGET_CLASSES`), so `default_classes` only matters for other clients.

---

## Tracker YAML files

Only used when `use_tracker:=true` (needed by the `botsort_hsv` method). Both live in `yolo/detector_package/`:

| File | Tracker | Notable settings |
|---|---|---|
| `oc_sort.yaml` (default) | Deep OC-SORT | `with_reid: true`, `appearance_thresh: 0.9`, `track_buffer: 30`, `gmc_method: sparseOptFlow` |
| `botsort.yaml` | BoT-SORT | `with_reid: True`, `appearance_thresh: 0.8`, `model: yolo11n-cls.pt` |

---

## ROI and depth range (`common.py`)

`spot/demo_package/demo_package/common.py`: the single place that defines the region of interest.

| Constant | Default | Meaning |
|---|---|---|
| `CONE_FOV_DEG` | `35.0` | Horizontal field of view of the crop, in degrees |
| `CONE_MIN_RANGE` | `1.5` m | Closer targets are ignored. `spot_motion` also imports this value. |
| `CONE_MAX_RANGE` | `3.5` m | Farther targets are ignored |
| `CROP_TOP_MARGIN_FRAC` | `0` | Fraction of the image height cut from the top |
| `CROP_BOTTOM_MARGIN_FRAC` | `0` | Fraction of the image height cut from the bottom |

`box_center_depth()` arguments (`patch_frac=0.2`, `min_valid_pixels=3`) control how depth is sampled inside a box.

---

## Tracking FSM (`tracking_fsm.py`)

`spot/demo_package/demo_package/tracking_fsm.py`. All values are constants at the top of the file.

### Method and I/O

| Constant | Default | Meaning |
|---|---|---|
| `TRACKING_METHOD` | `"embedding_only"` | `"embedding_only"` or `"botsort_hsv"` (see [Architecture](architecture.md#re-identification-strategies)) |
| `HAND_RGB_TOPIC` | `camera/hand/compressed` | RGB input topic |
| `HAND_RGB_COMPRESSED` | `True` | `False` to subscribe to a raw `sensor_msgs/Image` instead |
| `HAND_CAMERA_INFO_TOPIC` | `/camera/hand/camera_info` | Intrinsics |
| `HAND_DEPTH_TOPIC` | `/depth/hand/image` | ToF depth |
| `TARGET_POSE_TOPIC` | `target_info` | Output topic |
| `DETECT_SERVICE` | `detect` | Detector service name |
| `TARGET_CLASSES` | people / quadrupeds / robots | Prompts sent to YOLOE |
| `DEBUG_IMAGE_TOPIC` | `/person_follow/hand_debug/compressed` | Annotated debug image |

### Locking, tracking and recovery

| Constant | Default | Meaning |
|---|---|---|
| `MIN_DETECTION_CONFIDENCE` | `0.10` | Minimum YOLOE score accepted by the FSM |
| `STABILITY_FRAMES_REQUIRED` | `10` | Consecutive stable frames before locking in SEARCH |
| `TRACKING_GRACE_FRAMES` | `30` | Missed frames in TRACKING before RECOVERY |
| `RECOVERY_TIMEOUT_SEC` | `15.0` | Wall-clock time in RECOVERY before giving up |
| `REACQUISITION_STABILITY_FRAMES` | `3` | Consecutive wins needed to confirm a re-acquired target |
| `REACQUISITION_PX_TOLERANCE` | `60.0` px | Max movement between those wins to count as "the same" candidate |
| `TARGET_DISTANCE_THRESHOLD_FRAC` | `0.30` | Max jump between frames, as a fraction of the image diagonal |

### Re-identification scoring

| Constant | Default | Meaning |
|---|---|---|
| `REID_SIMILARITY_THRESHOLD` | `0.60` | Minimum appearance similarity to be a candidate |
| `REID_COMBINE` | `0.40` | Minimum combined score of the winning candidate |
| `REID_EMA_ALPHA` | `0.3` | Update rate of the reference embedding |
| `W_SIMILARITY` / `W_POSITION` | `0.7` / `0.3` | Combined score = W_SIMILARITY·sim − W_POSITION·(pixel jump / threshold) |
| `W_COSINE` / `W_EUCLIDEAN` / `W_MAGNITUDE` | `0.5` / `1.0` / `0.8` | Weights inside `rich_neural_embedding_similarity` |
| `EUCLIDEAN_SCALE` / `MAGNITUDE_SCALE` | `10.0` / `10.0` | Normalisation of the Euclidean and magnitude terms |

### LEDs

| Constant | Default | Meaning |
|---|---|---|
| `STATI_LED` | see [Running](running.md#visualisation-and-monitoring) | FSM state → name of an existing A/V behaviour on the robot (`None` = robot default). Robot software 5.0.1 cannot create new behaviours. Use `color.py` to list them. |
| `LED_DURATION_SEC` | `5.0` | Lifetime of each `run_behavior`. LEDs reset by themselves if the node dies. |
| `LED_REFRESH_SEC` | `2.0` | Refresh period (must be < `LED_DURATION_SEC`) |

---

## Motion

### `nav2_bridge` ROS parameters

| Parameter | Default | Meaning |
|---|---|---|
| `target_distance` | `2.5` m | Distance kept between body and target |
| `enter_margin` | `0.40` m | Start walking above `target_distance + enter_margin` |
| `exit_margin` | `0.15` m | Stop walking below `target_distance + exit_margin` |
| `rot_gain` | `1.2` | Rotation gain in HOLD (rad/s per rad of error) |
| `max_rot_vel` | `0.5` rad/s | Rotation limit (matches the velocity smoother) |
| `rot_deadband_deg` | `4.0`° | No rotation below this error |
| `rot_timeout` | `1.0` s | No rotation on data older than this |
| `center_on_camera` | `True` | Centre the target in the camera image (`False` = relative to the body) |
| `base_frame` | `body` | Robot base frame |
| `cmd_vel_topic` | `cmd_vel_nav` | Topic for rotation commands (input of the velocity smoother) |
| `behavior_tree` | `''` | Override the BT. Empty = the default set by the launch file. |

In-code constants: `GLOBAL_FRAME = 'odom'`, `KF_PROCESS_VAR = KF_MEASUREMENT_VAR = 0.05`, `KF_RESET_GAP_SEC = 1.0`, `MAX_TARGET_LOSS_SEC = 5.0`, `CANCEL_FORCE_SEC = 2.0`.

### `spot_motion` (SDK back-end) constants

`spot_motion/spot_motion/spot_motion.py`:

| Constant | Default | Meaning |
|---|---|---|
| `DRY_RUN` | `False` | `True` = compute and log, never send commands |
| `TARGET_DISTANCE` | `2.5` m | Distance to keep |
| `DISTANCE_TOLERANCE` | `0.15` m | Dead-band before walking |
| `MAX_LINEAR_VEL` / `MAX_ANGULAR_VEL` | `0.6` m/s / `0.5` rad/s | Velocity limits |
| `COMMAND_DURATION` | `3.0` s | Safety timeout of each command |
| `KF_PROCESS_VAR` / `KF_MEASUREMENT_VAR` | `0.05` / `0.05` | Kalman filter tuning |
| `KF_RESET_GAP_SEC` | `2.0` s | Re-initialise the filter after a gap longer than this |
| `EXTRAPOLATION_TIMER_PERIOD` | `0.2` s | Extrapolation timer period |
| `MAX_EXTRAPOLATION_SEC` | `2.0` s | Stop extrapolating after this long without data |
| `HAND_CAMERA_IMAGE_SOURCE` / `WRIST_FRAME_NAME` | `hand_color_image` / `arm0.link_wr1` | Frames used for `body_T_camera` |

### `costmap_refresher` ROS parameters

| Parameter | Default | Meaning |
|---|---|---|
| `period` | `2.0` s | Time between two clears |
| `services` | `['/local_costmap/clear_entirely_local_costmap']` | Clear services to call |

---

## Nav2 parameters and behaviour tree

| File | Purpose |
|---|---|
| `spot_motion/config/nav2_params_spot_real.yaml` | **Real robot.** Loaded by `navigation_launch.py`. Changes for the real robot are marked `[REAL]`. |
| `spot_motion/config/nav2_params_spot_sim.yaml` | Configuration tested in Gazebo simulation |
| `spot_motion/bt/follow_point_spot.xml` | Behaviour tree for dynamic goal following |

Key values in `nav2_params_spot_real.yaml`:

| Section | Setting | Value |
|---|---|---|
| all | `global_frame` / `robot_base_frame` | `odom` / `body` |
| `controller_server` | controller | Regulated Pure Pursuit, `desired_linear_vel: 0.6`, `lookahead_dist: 0.9` (0.6–1.2), collision detection on |
| `controller_server` | goal checker | `xy_goal_tolerance: 0.30`, `yaw_goal_tolerance: 3.14` (heading is handled by `nav2_bridge`) |
| `controller_server` | progress checker | `required_movement_radius: 0.3` within `movement_time_allowance: 20.0` s |
| `local_costmap` | size / resolution / rate | 6 × 6 m, 0.05 m, 5 Hz |
| `local_costmap` | sources | `/depth/frontleft/points`, `/depth/frontright/points` (obstacles 0.5–2.0 m high, up to 2.5 m away), plus clearing-only sources and `/depth/hand/points` |
| `local_costmap` | footprint | arrow-shaped polygon, about 1.65 × 0.70 m. The narrow "nose" (up to x = 1.05 m) covers the arm stretched forward with the hand camera. Measure it with `ros2 run tf2_ros tf2_echo body hand` and update both costmaps. |
| `local_costmap` | inflation | `inflation_radius: 1.0` |
| `global_costmap` | size / resolution / rate | 16 × 16 m, 0.05 m, 2 Hz |
| `planner_server` | planner | NavFn |
| `velocity_smoother` | `max_velocity` | `[0.6, 0.0, 0.5]` |

> **Lifecycle manager name.** In `navigation_launch.py` the lifecycle manager node **must** be called `lifecycle_manager`, the same as the top-level key in the YAML. ROS 2 matches parameters by node name. With a different name the node would silently start with default parameters.

> **BT path.** `$(find-pkg-share ...)` is not resolved inside a parameters YAML. `navigation_launch.py` therefore computes the absolute path of `follow_point_spot.xml` and passes it as `default_nav_to_pose_bt_xml`.

To tune the behaviour tree, edit `bt/follow_point_spot.xml`:
- `RateController hz="0.5"` around `PeriodicClearLocal`: local costmap clear rate. Use `0.33` to be more conservative or `1.0` to be more aggressive.
- `RateController hz="3.0"`: replanning rate.
- `RecoveryNode number_of_retries="10"`: retries before the goal is aborted.
