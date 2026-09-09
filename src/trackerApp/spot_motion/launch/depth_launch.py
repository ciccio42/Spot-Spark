#!/usr/bin/env python3
"""
depth_to_pointcloud_launch.py

spot_ros2 pubblica la depth come sensor_msgs/Image (/depth/<camera>/image +
/depth/<camera>/camera_info) — MAI come PointCloud2. L'obstacle_layer di Nav2
pero' accetta solo PointCloud2 o LaserScan in input, non un'immagine depth grezza.

Questo launch file porta su un nodo depth_image_proc::PointCloudXyzNode per
ciascuna delle TRE camere scelte (frontleft, frontright, hand).

CORREZIONE CRITICA: Abilitato 'approximate_sync' per evitare il crollo della
frequenza a 1.5Hz causato dal disallineamento dei timestamp del wrapper dello Spot.
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
            #approximate_sync sblocca la coda associando frame con timestamp vicini,
            #portando la pubblicazione da 1.5Hz a oltre 10Hz stabili.
            parameters=[{
                'queue_size': 30,
                'approximate_sync': True  # <--- MODIFICA FONDAMENTALE PER SPOT
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
