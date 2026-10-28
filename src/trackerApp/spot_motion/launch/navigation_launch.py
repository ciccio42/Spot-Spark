#!/usr/bin/env python3
"""
navigation_launch.py (spot_motion)

Brings up the five Nav2 servers used in this project (no map_server,
no AMCL — reactive navigation on odom, a decision taken months ago) plus the
lifecycle_manager that starts/manages them, all pointing to the same
parameters file: config/nav2_params_spot_real.yaml.

CAREFUL with the name of the lifecycle_manager node below: it MUST match
exactly the top-level 'lifecycle_manager:' key inside the YAML file
(not 'lifecycle_manager_navigation', the most common convention in the
nav2_bringup tutorials) — ROS 2 matches the parameters of a YAML file to a
node by NAME, not by position; a different name here would make the node
load the default parameters, silently ignoring the autostart/node_names
written in the YAML, without any visible error.

Prerequisite: the nodes of depth_to_pointcloud_launch.py must already be
running (or launched together, see note at the bottom) — otherwise the two
costmaps never receive data from the obstacle_layers.
"""
import os

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    package_dir = get_package_share_directory('spot_motion')
    params_file = os.path.join(package_dir, 'config', 'nav2_params_spot_real.yaml')

    # $(find-pkg-share ...) inside the YAML is NEVER resolved: that syntax
    # is only understood by the launch system (.launch.xml files or
    # Substitutions in launch.py), not by loading a parameters file directly
    # via Node(parameters=[...]) — the node would read it as a literal
    # string (verified: that is exactly the error we had).
    # We compute the real path here, in Python, and pass it as an
    # override — it silently overrides the string value in the YAML.
    bt_xml_path = os.path.join(package_dir, 'bt', 'follow_point_spot.xml')

    lifecycle_nodes = [
        'controller_server', 'planner_server', 'behavior_server',
        'bt_navigator', 'velocity_smoother',
    ]

    return LaunchDescription([
        # Node(
        #     package='nav2_controller',
        #     executable='controller_server',
        #     name='controller_server',
        #     output='screen',
        #     parameters=[params_file],
        # ),
        Node(
            package='nav2_planner',
            executable='planner_server',
            name='planner_server',
            output='screen',
            parameters=[params_file],
        ),
        Node(
            package='nav2_behaviors',
            executable='behavior_server',
            name='behavior_server',
            output='screen',
            parameters=[params_file],
        ),
        Node(
            package='nav2_bt_navigator',
            executable='bt_navigator',
            name='bt_navigator',
            output='screen',
            parameters=[params_file, {'default_nav_to_pose_bt_xml': bt_xml_path}],
        ),
        # Node(
        #     package='nav2_velocity_smoother',
        #     executable='velocity_smoother',
        #     name='velocity_smoother',
        #     output='screen',
        #     parameters=[params_file],
        # ),
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager',  # <-- must match the key in the YAML, see note above
            output='screen',
            parameters=[params_file],
        ),
        
        Node(
            package='nav2_controller', 
            executable='controller_server',
            name='controller_server', output='screen',
            parameters=[params_file],
            remappings=[('cmd_vel', 'cmd_vel_nav')]),

        Node(
            package='nav2_velocity_smoother', 
            executable='velocity_smoother',
            name='velocity_smoother', output='screen',
            parameters=[params_file],
            remappings=[('cmd_vel', 'cmd_vel_nav'),
                     ('cmd_vel_smoothed', 'cmd_vel')]),
        
        Node(
            package='spot_motion',
            executable='costmap_refresher',
            name='costmap_refresher',
            output='screen',
            parameters=[{'period': 2.0}],
        ),
    ])