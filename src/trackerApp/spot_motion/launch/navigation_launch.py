#!/usr/bin/env python3
"""
navigation_launch.py (spot_motion)

Porta su i cinque server Nav2 usati in questo progetto (nessun map_server,
nessun AMCL — navigazione reattiva su odom, decisione presa mesi fa) piu' il
lifecycle_manager che li avvia/gestisce, tutti puntati allo stesso file
parametri: config/nav2_params_spot_real.yaml.

ATTENZIONE al nome del nodo lifecycle_manager qui sotto: DEVE combaciare
esattamente con la chiave top-level 'lifecycle_manager:' dentro il file YAML
(non 'lifecycle_manager_navigation', la convenzione piu' comune nei tutorial
nav2_bringup) — ROS2 associa i parametri di un file YAML al nodo per NOME,
non per posizione; un nome diverso qui farebbe caricare il nodo con i
parametri di default, ignorando silenziosamente autostart/node_names scritti
nel YAML, senza nessun errore visibile.

Prerequisito: i nodi di depth_to_pointcloud_launch.py devono essere gia' in
esecuzione (o lanciati insieme, vedi nota in fondo) — altrimenti i due
costmap non ricevono mai dati dagli obstacle_layer.
"""
import os

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    package_dir = get_package_share_directory('spot_motion')
    params_file = os.path.join(package_dir, 'config', 'nav2_params_spot_real.yaml')

    # $(find-pkg-share ...) dentro il YAML non viene MAI risolto: quella
    # sintassi la capisce solo il sistema di lancio (file .launch.xml o
    # Substitution in launch.py), non il caricamento diretto di un file
    # di parametri via Node(parameters=[...]) — il nodo lo leggerebbe come
    # stringa letterale (verificato: e' esattamente l'errore avuto).
    # Calcoliamo il percorso vero qui, in Python, e lo passiamo come
    # override — sovrascrive silenziosamente il valore-stringa nel YAML.
    bt_xml_path = os.path.join(package_dir, 'bt', 'follow_point_spot.xml')

    lifecycle_nodes = [
        'controller_server', 'planner_server', 'behavior_server',
        'bt_navigator', 'velocity_smoother',
    ]

    return LaunchDescription([
        Node(
            package='nav2_controller',
            executable='controller_server',
            name='controller_server',
            output='screen',
            parameters=[params_file],
        ),
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
        Node(
            package='nav2_velocity_smoother',
            executable='velocity_smoother',
            name='velocity_smoother',
            output='screen',
            parameters=[params_file],
        ),
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager',  # <-- deve combaciare con la chiave nel YAML, vedi nota sopra
            output='screen',
            parameters=[params_file],
        ),
    ])