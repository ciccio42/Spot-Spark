# ROS2 Docker

**Build**
```bash
docker build -t ros2 . -f ros2_humble.dockerfile
``` 

**Run**
```bash
xhost +local:docker
docker run -it --rm \
  --gpus all \
  --privileged \
  --net=host \
  -e DISPLAY=$DISPLAY \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  ros2
``` 