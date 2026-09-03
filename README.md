# Spot-Spark
This repository contains all the docker files and configurations for running Spot demos on Nvidia-Spark

# Docs
The folder **docs** contains Markdown files reporting the instructions and commands for running containers for each specific application. Specifically:

* [ROS2-Docker](docs/ros2_docker.md), contains the instruction to build and run ros2 docker container 

# Dependencies
```bash
```


# Docker utils

## Clean docker build cache
docker builder prune --all

## Clean dandling image
docker image prune -f