# RUN SPOT

**RUN**
```bash
docker run -it --rm \
  --name spot-container \
  --gpus all \
  --privileged \
  --net=host \
  -e DISPLAY=$DISPLAY \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v /home/mivia/Desktop/SPOT/Spot-Spark/src/codice_carmine:/home/spot_ws/src/codice_carmine \
  spot

  source install/setup.bash

  ros2 launch spot_driver spot_driver.launch.py config_file:=/home/spot_ws/src/codice_carmine/spot/config_spot/spot.yaml   launch_rviz:=True
```