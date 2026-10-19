#!/usr/bin/env python3
"""
fsm_launch.py (autonomous_manipulation_fsm)

Starts fsm_node with the parameters in config/fsm_params.yaml.

Press Enter in the launch terminal to start the FSM (the node reads /dev/tty, since ros2 launch
does not forward its stdin), or from another terminal:
  ros2 service call /fsm_node/start std_srvs/srv/Trigger

Arguments:
  params_file     YAML file with the node parameters (default: the one installed with the package)
  use_click       true (default): click_* parameters from params_file (by default the operator confirms
                  every detection with a click on the yolo_detector window before the navigation);
                  false: no click at all (click_confirm_* and click_fallback_* set to false)
  log_level       ROS log level (debug, info, warn, error)

Example:
  ros2 launch autonomous_manipulation_fsm fsm_launch.py use_click:=false log_level:=debug
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# parameters disabled by use_click:=false
CLICK_PARAMS = ['click_confirm_navigation', 'click_confirm_pick',
                'click_fallback_navigation', 'click_fallback_pick']


def launch_node(context):
    params = [LaunchConfiguration('params_file').perform(context)]
    use_click = LaunchConfiguration('use_click').perform(context).strip().lower()
    if use_click not in ('true', 'false'):
        raise RuntimeError(f"use_click must be 'true' or 'false', got '{use_click}'")
    if use_click == 'false':
        params.append({name: False for name in CLICK_PARAMS})

    return [Node(
        package='autonomous_manipulation_fsm',
        executable='fsm_node',
        # must match the top-level key in the YAML, otherwise its parameters are ignored
        name='fsm_node',
        output='screen',
        emulate_tty=True,
        parameters=params,
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level').perform(context)],
        # on Ctrl+C the node resets the arm before exiting (up to shutdown_timeout):
        # give it time before launch escalates to SIGTERM/SIGKILL (default 5 s + 5 s)
        sigterm_timeout='15',
        sigkill_timeout='5',
    )]


def generate_launch_description():
    default_params = os.path.join(
        get_package_share_directory('autonomous_manipulation_fsm'), 'config', 'fsm_params.yaml')

    return LaunchDescription([
        DeclareLaunchArgument('params_file', default_value=default_params,
                              description='YAML file with the FSM parameters'),
        DeclareLaunchArgument('use_click', default_value='true',
                              description='false: no click confirmation/fallback on the yolo_detector window'),
        DeclareLaunchArgument('log_level', default_value='info', description='ROS log level'),
        OpaqueFunction(function=launch_node),
    ])
