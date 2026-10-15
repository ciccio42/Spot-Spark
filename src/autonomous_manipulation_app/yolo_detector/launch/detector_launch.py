#!/usr/bin/env python3
"""
detector_launch.py (yolo_detector)

Starts detector_node with the parameters in config/detector_params.yaml.

Arguments:
  params_file     YAML file with the node parameters (default: the one installed with the package)
  checkpoint      override of 'checkpoint' (file in <share>/yolo_detector/yolo_checkpoint)
  device          override of 'device' ("cuda:0", "cpu", ...)
  camera_name     override of 'camera_name' (CompressedImage topic)
  log_level       ROS log level (debug, info, warn, error)

The overrides are applied only when non-empty, otherwise the YAML values are used:
  ros2 launch yolo_detector detector_launch.py device:=cpu
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

OVERRIDES = ['checkpoint', 'device', 'camera_name']


def launch_node(context):
    params = [LaunchConfiguration('params_file').perform(context)]
    overrides = {name: LaunchConfiguration(name).perform(context) for name in OVERRIDES}
    overrides = {name: value for name, value in overrides.items() if value}
    if overrides:
        params.append(overrides)

    return [Node(
        package='yolo_detector',
        executable='detector_node',
        # must match the top-level key in the YAML, otherwise its parameters are ignored
        name='detection_node',
        output='screen',
        emulate_tty=True,
        parameters=params,
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level').perform(context)],
    )]


def generate_launch_description():
    default_params = os.path.join(
        get_package_share_directory('yolo_detector'), 'config', 'detector_params.yaml')

    return LaunchDescription([
        DeclareLaunchArgument('params_file', default_value=default_params,
                              description='YAML file with the detector parameters'),
        *[DeclareLaunchArgument(name, default_value='',
                                description=f"Override of '{name}' (empty = value from params_file)")
          for name in OVERRIDES],
        DeclareLaunchArgument('log_level', default_value='info', description='ROS log level'),
        OpaqueFunction(function=launch_node),
    ])
