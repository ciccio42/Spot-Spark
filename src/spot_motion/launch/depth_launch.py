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

POINTCLOUD -> LASERSCAN (front cameras only)
  The obstacle_layer filters the obstacle height in its global frame (odom).
  Spot's odom drifts in z: after a while the floor rises above
  min_obstacle_height and the whole costmap fills with obstacles.
  Each front cloud is therefore turned into a LaserScan in flat_body (moves with
  the robot, always horizontal), cutting the height band THERE:
      /depth/<camera>/points -> /scan/<camera>
  The floor is always ~0.5 m below flat_body, whatever the odom drift.
  The costmaps use /scan/<camera> (see config/nav2_params_spot_real.yaml).

  Height band (min_height / max_height, metres in flat_body):
    defaults 0.0 / 1.5 = ~0.5 m / ~2.0 m above the floor, the same band as the
    old min/max_obstacle_height in odom. MEASURE the floor height first:
        ros2 run tf2_ros tf2_echo flat_body gpe      (z ~ -0.5)
    floor + 0.20 m as min_height also sees lower obstacles (boxes, chairs).

  Scan sector (angle_min / angle_max, rad, in flat_body):
    the beams with no return are 'inf' and CLEAR the costmap (inf_is_valid in the
    YAML): the sector must stay inside what the two front cameras see, otherwise
    it clears obstacles on the sides that no camera is looking at.
    Default +-0.7 rad (+-40 deg). Check in RViz that /scan/frontleft and
    /scan/frontright cover the floor in front of the robot.

  Example:
    ros2 launch spot_motion depth_launch.py min_height:=-0.30 max_height:=1.5
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

CAMERAS = ['frontleft', 'frontright', 'hand']
SCAN_CAMERAS = ['frontleft', 'frontright']

# launch argument -> default value
SCAN_ARGS = {
    'scan_frame': 'flat_body',
    'min_height': '0.0',
    'max_height': '1.5',
    'angle_min': '-0.7',
    'angle_max': '0.7',
    'range_min': '0.3',
    'range_max': '4.0',
}


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

    composable_nodes += [
        ComposableNode(
            package='pointcloud_to_laserscan',
            plugin='pointcloud_to_laserscan::PointCloudToLaserScanNode',
            name=f'scan_{camera}',
            remappings=[
                ('cloud_in', f'/depth/{camera}/points'),
                ('scan', f'/scan/{camera}'),
            ],
            parameters=[{
                'target_frame': LaunchConfiguration('scan_frame'),
                # the clouds arrive late (~1.5-3 Hz): wait for the TF at their stamp
                'transform_tolerance': 0.5,
                'min_height': LaunchConfiguration('min_height'),
                'max_height': LaunchConfiguration('max_height'),
                'angle_min': LaunchConfiguration('angle_min'),
                'angle_max': LaunchConfiguration('angle_max'),
                'angle_increment': 0.0087,  # 0.5 deg
                'scan_time': 0.3,
                'range_min': LaunchConfiguration('range_min'),
                'range_max': LaunchConfiguration('range_max'),
                # no return = inf: the costmap clears along the whole beam (inf_is_valid)
                'use_inf': True,
            }],
        )
        for camera in SCAN_CAMERAS
    ]

    container = ComposableNodeContainer(
        name='depth_to_pointcloud_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=composable_nodes,
        output='screen',
    )

    return LaunchDescription([
        *[DeclareLaunchArgument(name, default_value=value) for name, value in SCAN_ARGS.items()],
        container,
    ])
