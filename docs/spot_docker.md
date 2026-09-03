# ROS2 Docker

**Build**
```bash
docker build -t spot . -f spot.dockerfile
```

**Run**
```bash
xhost +local:docker
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
  -v /home/mivia/Desktop/SPOT/Spot-Spark/src/trackerApp:/home/spot_ws/src/codice_carmine \
  spot
```