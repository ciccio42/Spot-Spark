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
  -e DISPLAY=$DISPLAY \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v /home/mivia/Desktop/SPOT/Spot-Spark/src/provola:/home/spot_ws/src/prova_codice \
  spot
```