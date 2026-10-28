# ROS 2 interfaces

[← Back to README](../README.md)

The project defines two interface packages. Both are `ament_cmake` packages built with `rosidl_default_generators`.

- [`detector_interfaces`](#detector_interfaces): contract between `tracking_fsm` (spot container) and `detector_node` (yolo container)
- [`demo_interfaces`](#demo_interfaces): messages exchanged inside the spot container

---

## `detector_interfaces`

Source: `src/trackerApp/yolo/detector_interfaces/`. It must be built in **both** containers.

### `srv/Detect.srv`

Synchronous detection request. The image may be the full frame or a crop: the service handles both the same way. ROI logic and coordinate translation are the caller's job.

```text
sensor_msgs/Image image          # BGR8 image to analyse
string[]  target_classes         # text prompts; empty → detector's default_classes
bool      reset_tracker false    # reset the tracker state before this request
---
BoxDetection[] detections
```

Set `reset_tracker` to `true` when you start a new tracking session, so that old `track_id`s do not come back. `tracking_fsm` does this automatically after RECOVERY times out.

### `msg/BoxDetection.msg`

```text
float32 x1          # top-left x  (pixels of the image in the request, xyxy)
float32 y1          # top-left y
float32 x2          # bottom-right x
float32 y2          # bottom-right y
float32 score       # YOLOE confidence
string  class_name  # matched text prompt
int32   track_id    # tracker ID; -1 if not tracked or tracking disabled
float32[] embedding # OSNet appearance vector; EMPTY if ReID is not configured
```

Callers must handle an empty `embedding`: `demo_package.common.embedding_from_msg()` returns `None` in that case.

---

## `demo_interfaces`

Source: `src/trackerApp/demo_interfaces/`.

### `msg/TargetInfoMessage.msg`

Published by `tracking_fsm` on `target_info`, **only on frames where the target is confirmed**.

```text
std_msgs/Header header       # stamp and frame of the source RGB image
float32[4] bounding_box      # target box, xyxy, in full-image pixels
float32 depth_m              # median ToF depth at the box centre (metres)
string camera_rgb_topic      # RGB topic the box refers to
string camera_depth_topic    # depth topic the distance comes from
```

### `msg/TargetPose3D.msg`

Published by `pose_3d_estimation` on `target_3d`.

```text
std_msgs/Header header       # frame_id = hand camera optical frame (from CameraInfo)
geometry_msgs/Point position # target centre in the optical frame (x right, y down, z forward)
float32 yaw                  # horizontal bearing of the target, atan2(u − cx, fx), radians
```

> `srv/TargetInfoMessage.srv` and `srv/TargetPose3D.srv` are empty placeholder files. They are not listed in `CMakeLists.txt` and are not generated.

---

## Inspecting the interfaces

```bash
ros2 interface show detector_interfaces/srv/Detect
ros2 interface show demo_interfaces/msg/TargetPose3D

ros2 topic echo /target_info
ros2 topic echo /target_3d
```
