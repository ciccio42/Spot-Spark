# ROS2 Docker

**Build**
```bash
docker build -t gazebo . -f gazebo.dockerfile
```

**Run**
```bash
xhost +local:docker
docker run -it --rm \
  --name gazebo-container \
  --gpus all \
  --privileged \
  --net=host \
  -e DISPLAY=$DISPLAY \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  gazebo
```