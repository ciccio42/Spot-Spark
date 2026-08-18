# YOLO ROS2 Docker

**Build**
```bash
docker build -t spot-yolo . -f yolo.dockerfile
```


**Run**
```bash
xhost +local:root

docker run -it --rm \
  --name yolo-container \
  --gpus all \
  --privileged \
  --net=host \
  --ipc=host \
  -e DISPLAY=$DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v /home/mivia/Desktop/SPOT/Spot-Spark/src/trackerApp:/home/yolo_ws/src \
  spot-yolo
```