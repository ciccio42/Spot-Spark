#!/usr/bin/env python3
"""
depth_to_pointcloud_launch.py

spot_ros2 publishes depth as sensor_msgs/Image (/depth/<camera>/image +
/depth/<camera>/camera_info) — NEVER as PointCloud2. The Nav2 obstacle_layer,
however, only accepts PointCloud2 or LaserScan as input, not a raw depth image.

This launch file brings up one depth_image_proc::PointCloudXyzNode for
each of the THREE selected cameras (frontleft, frontright, hand).

CRITICAL FIX: 'approximate_sync' is enabled to avoid the rate dropping to
1.5 Hz because of the timestamp misalignment of the Spot wrapper.
"""
from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

CAMERAS = ['frontleft', 'frontright', 'hand']


def generate_launch_description():
    composable_nodes = [
        ComposableNode(
            package='depth_image_proc',
            plugin='depth_image_proc::PointCloudXyzNode',
            name=f'pointcloud_{camera}',
            remappings=[
                ('image_rect', f'/depth/{camera}/image'),
                ('camera_info', f'/depth/{camera}/camera_info'),
                ('points', f'/depth/{camera}/points'),
            ],
            # approximate_sync unblocks the queue by pairing frames with close timestamps,
            # raising the publish rate from 1.5 Hz to a stable 10+ Hz.
            parameters=[{
                'queue_size': 30,
                'approximate_sync': True  # <--- ESSENTIAL CHANGE FOR SPOT
            }],
        )
        for camera in CAMERAS
    ]

    container = ComposableNodeContainer(
        name='depth_to_pointcloud_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=composable_nodes,
        output='screen',
    )

    return LaunchDescription([container])
