# Installation

[← Back to README](../README.md)

This guide sets up the host machine (NVIDIA DGX Spark), builds the Docker images and prepares the ROS 2 workspaces.

- [1. Host prerequisites](#1-host-prerequisites)
- [2. Clone the repository](#2-clone-the-repository)
- [3. Build the Docker images](#3-build-the-docker-images)
- [4. Model weights](#4-model-weights)
- [5. Build the ROS 2 workspaces](#5-build-the-ros-2-workspaces)
- [6. Network setup for Spot](#6-network-setup-for-spot)

---

## 1. Host prerequisites

| Requirement | Notes |
|---|---|
| NVIDIA DGX Spark (or another `aarch64` machine with an NVIDIA GPU) | The `spot` image installs `spot_ros2` with `--arm64`. On `x86_64`, change that flag in `docker/spot.dockerfile`. |
| NVIDIA driver | Check with `nvidia-smi`. |
| Docker Engine | Your user should be in the `docker` group. |
| [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) | Required for `--gpus all` / `--runtime=nvidia`. |
| X11 server | Needed for RViz, rqt and Gazebo GUIs. On a headless Spark, use a VNC session or X forwarding. |
| Disk space | About 40 GB for the full image chain (`spot-yolo` ≈ 13 GB, `spot` ≈ 9 GB). |

Check that containers can see the GPU:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-runtime-ubuntu22.04 nvidia-smi
```

---

## 2. Clone the repository

```bash
mkdir -p ~/Desktop/SPOT && cd ~/Desktop/SPOT
git clone <repository-url> Spot-Spark
cd Spot-Spark
export SPOT_SPARK=$PWD      # used by the commands in these docs
```

> The `docker run` commands in these docs mount `$SPOT_SPARK/src/trackerApp`. If you skip the `export`, replace `$SPOT_SPARK` with the absolute path of your clone.

---

## 3. Build the Docker images

The images form a **chain**: each Dockerfile starts `FROM` the previous one by its local tag. Build them **in this order and with exactly these tags**, from inside `docker/` (the build context must be `docker/` because `spot.dockerfile` copies `src/spot_launch_helpers.py`).

```text
nvidia/cuda:12.4.1-runtime-ubuntu22.04
└── ros2        (ros2_humble.dockerfile)   ROS 2 Humble desktop, colcon, robot_localization
    ├── nav2    (nav2.dockerfile)          navigation2, nav2_bringup, turtlebot3
    │   └── gazebo  (gazebo.dockerfile)    Ignition Fortress, ros_gz, ros2_control
    │       └── spot    (spot.dockerfile)  Spot SDK 5.0.1, bosdyn-* 5.0.1, spot_ros2 (built)
    └── spot-yolo (yolo.dockerfile)        PyTorch (cu128), Ultralytics, torchreid, OSNet weights
```

```bash
cd $SPOT_SPARK/docker

docker build -t ros2      . -f ros2_humble.dockerfile
docker build -t nav2      . -f nav2.dockerfile
docker build -t gazebo    . -f gazebo.dockerfile
docker build -t spot      . -f spot.dockerfile
docker build -t spot-yolo . -f yolo.dockerfile
```

### What each image contains

**`ros2`**: base image with CUDA 12.4 runtime, Ubuntu 22.04, `ros-humble-desktop` (RViz, rqt, cv_bridge, image_transport…), `python3-colcon-common-extensions`, `robot_localization`, and OpenGL libraries for GUI apps.

**`nav2`**: adds `navigation2`, `nav2_bringup` and the TurtleBot3 packages (useful for testing Nav2 on its own).

**`gazebo`**: adds Ignition Gazebo Fortress, `ros_gz`, `gz_ros2_control` and `ros2_controllers` for simulation.

**`spot`**: the robot-side image:
- Spot SDK `v5.0.1` sources in `/opt/spot-sdk`, plus the `bosdyn-client`, `bosdyn-mission`, `bosdyn-choreography-client`, `bosdyn-api` and `bosdyn-core` Python packages (5.0.1).
- The [`ciccio42/spot_ros2`](https://github.com/ciccio42/spot_ros2) fork, cloned to `/home/spot_ws/src/spot_ros2`, installed with `install_spot_ros2.sh --arm64` and built with colcon.
- `docker/src/spot_launch_helpers.py` overwrites `spot_common/launch/spot_launch_helpers.py` in the driver before the build.

**`spot-yolo`**: the perception image (built on `ros2`, not on `spot`):
- PyTorch + torchvision from the CUDA 12.8 wheel index.
- `ultralytics` (YOLOE), `tensorboard`, `gdown`, `huggingface_hub`.
- [`torchreid`](https://github.com/KaiyangZhou/deep-person-reid) (deep-person-reid), with `numpy==1.26.4` / `numpy<2` for compatibility.
- OSNet weights cloned from Hugging Face (`kaiyangzhou/osnet`) to `/home/yolo_ws/osnet` with git-lfs.
- An empty workspace at `/home/yolo_ws`. The source code is mounted at run time.

> **Rebuilding.** Changing a lower image (for example `ros2`) does **not** rebuild the images above it. Rebuild every image above it in order. See [Troubleshooting → Docker housekeeping](troubleshooting.md#docker-housekeeping) to free space after many rebuilds.

---

## 4. Model weights

| Model | Used by | Default path in the container | Provided by |
|---|---|---|---|
| YOLOE `yoloe-11s-seg.pt` | `detector_node` (parameter `model_path`) | `/models/yoloe-11s-seg.pt` | **You.** It is not in git (`*pt` is in `.gitignore`). |
| OSNet `osnet_x1_0_imagenet.pth` | `detector_node` (parameter `reid_model_path`) | `/home/yolo_ws/osnet/osnet_x1_0_imagenet.pth` | Baked into the `spot-yolo` image |
| MobileCLIP text encoder `mobileclip_blt.ts` | YOLOE `set_classes()` (turns the class names into text embeddings) | downloaded by Ultralytics on first use, into the working directory | Ultralytics |
| `yolo11n-cls.pt` | BoT-SORT / Deep OC-SORT ReID (only with `use_tracker:=true`) | downloaded by Ultralytics on first use | Ultralytics |

> The first start of `detector_node` needs **internet access** to download the MobileCLIP text encoder. Start the node once from the same working directory (for example `/home/yolo_ws/src`) so the downloaded `.ts` file is reused next time. `*.ts` is git-ignored.

Download the YOLOE weights once and put them **inside the mounted source tree**, so the container can read them:

```bash
mkdir -p $SPOT_SPARK/src/trackerApp/codice_carmine/yolo/model
cd $SPOT_SPARK/src/trackerApp/codice_carmine/yolo/model
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yoloe-11s-seg.pt
```

Inside the `yolo` container this file is at `/home/yolo_ws/src/codice_carmine/yolo/model/yoloe-11s-seg.pt`. Pass that path to the node with `-p model_path:=...` (see [Running](running.md#terminal-2-detector-yolo-container)), or mount a host folder at `/models` instead.

---

## 5. Build the ROS 2 workspaces

The custom packages live in `src/trackerApp/` and are **mounted** into the containers, not copied, so you can edit code on the host and rebuild inside the container. Each container has its own workspace:

| Container | Host folder | Mount point | Workspace |
|---|---|---|---|
| `spot` | `src/trackerApp` | `/home/spot_ws/src/codice_carmine` | `/home/spot_ws` |
| `spot-yolo` | `src/trackerApp` | `/home/yolo_ws/src` | `/home/yolo_ws` |

### Spot workspace (in the `spot` container)

```bash
cd /home/spot_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install \
  --packages-select demo_interfaces detector_interfaces demo_package spot_motion
source install/setup.bash
```

`demo_package` needs `detector_interfaces` (it calls the `Detect` service), so build it here too.

### YOLO workspace (in the `spot-yolo` container)

```bash
cd /home/yolo_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-up-to detector_package
source install/setup.bash
pip install --break-system-packages "numpy<2"   # only if colcon pulled numpy 2.x
```

> With `--symlink-install`, edits to Python files apply on the next `ros2 run`. You only need to rebuild after changing `setup.py`, `package.xml`, launch/config files that are installed, or `.msg`/`.srv` files.

The `build/`, `install/` and `log/` folders end up inside the mounted tree (`src/trackerApp/` for the yolo workspace). They are git-ignored, but the yolo and spot builds are separate. Do not mix them.

---

## 6. Network setup for Spot

- Connect the Spark to Spot, either to the robot's Wi-Fi access point or via Ethernet / payload port. The default robot address in this repository is `192.168.80.3` (Spot's Wi-Fi AP address).
- Check that the robot answers: `ping 192.168.80.3`.
- Set the robot address and credentials in `src/trackerApp/spot/config_spot/spot.yaml` and in the two Python nodes that use the SDK directly. See [Configuration → Robot connection](configuration.md#robot-connection).
- Every container runs with `--net=host`. If you run several ROS 2 systems on the same network, give each its own `ROS_DOMAIN_ID` and pass the **same** value to all containers of one setup (`-e ROS_DOMAIN_ID=<n>`).

Next step: [Running the application →](running.md)
