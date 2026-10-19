#!/usr/bin/env python3
"""
app_launch.py (autonomous_manipulation_fsm)

Starts the whole autonomous manipulation app EXCEPT the detection, in the spot container:
  1. Spot driver            (spot_driver/spot_driver.launch.py)          -- launch_driver:=false to skip it
  2. depth -> clouds/scans  (spot_motion/depth_launch.py)
  3. Nav2                   (spot_motion/navigation_launch.py)
  4. manipulation_node      (manipulation)
  5. fsm_node               (autonomous_manipulation_fsm)

The detection runs in the yolo container:
  ros2 launch yolo_detector detector_launch.py

separate_terminals:=false (default): everything in this terminal, as included launch files.
  The FSM reads Enter from this terminal (/dev/tty) and also accepts
    ros2 service call /fsm_node/start std_srvs/srv/Trigger
  Ctrl+C stops EVERYTHING at once, the driver included: the driver may shut down before the FSM has
  stowed the arm, and on shutdown the driver sits the robot. To keep the robot under control on exit,
  run the driver in its own terminal and start this launch with launch_driver:=false.

separate_terminals:=true: ONE Tilix window split in panes, one per component (driver | Depth | Nav2
  on top, Manipulation | FSM below). Needs tilix + dbus-x11 + a UTF-8 locale in the container and a
  reachable X display (e.g. DISPLAY=:2 of the VNC server). The layout is written to
  /tmp/autonomous_manipulation_tilix_session.json and loaded with tilix --session.
  - press Enter for the FSM in the "FSM" pane;
  - stop the components one by one with Ctrl+C inside their panes, FSM first and driver last, so
    the FSM can still stow the arm through the driver;
  - each pane stays open after its process exits (to read the log): press Enter to close it;
  - Ctrl+C on this launch closes the whole window (the nodes get SIGHUP, the FSM still tries
    to stow the arm, but the driver is closed at the same time).

Examples:
  ros2 launch autonomous_manipulation_fsm app_launch.py
  ros2 launch autonomous_manipulation_fsm app_launch.py separate_terminals:=true
  ros2 launch autonomous_manipulation_fsm app_launch.py launch_driver:=false launch_rviz:=False
"""
import json
import os
import uuid

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def include(package, launch_file, arguments=None):
    path = os.path.join(get_package_share_directory(package), 'launch', launch_file)
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(path),
        launch_arguments=(arguments or {}).items(),
    )


# Tilix default profile (always present): every pane of the session uses it
TILIX_DEFAULT_PROFILE = '2b7c4080-0ddd-46c5-8f23-563fd3ba789d'
SESSION_FILE = '/tmp/autonomous_manipulation_tilix_session.json'


def tilix_terminal(title, command, workspace):
    """Tilix session pane running `command` (a single command, e.g. ros2 launch or a node executable)
    in a shell with ROS 2 and the workspace sourced. The pane stays open after the command exits.

    Shutdown: when the window/pane is closed the terminal sends SIGHUP. ros2 launch does not handle
    SIGHUP (it would die at once leaving its nodes running), so the command is started with SIGHUP
    ignored and the shell turns SIGHUP (and Ctrl+C / SIGTERM) into SIGINT for the command, then waits
    for it: ros2 launch and the nodes shut down cleanly."""
    # one command per line ("cmd &; ..." would be a bash syntax error)
    script = '\n'.join([
        f"printf '\\033]0;{title}\\007'",
        f"source /opt/ros/humble/setup.bash && source {workspace}/install/setup.bash "
        f"|| {{ read -p 'cannot source the workspace, press Enter'; exit 1; }}",
        # the command must not receive the SIGHUP of the closed terminal: it gets SIGINT from the trap
        "trap '' HUP",
        # a background job of a non-interactive bash ignores SIGINT (ros2 launch would then ignore the
        # SIGINT of the trap): restore it in the subshell, then exec the command
        f"( trap - INT QUIT; exec {command} ) &",
        "child=$!",
        "trap 'kill -INT $child 2>/dev/null; wait $child' HUP INT TERM",
        "wait $child",
        "trap - HUP INT TERM",
        "echo",
        f"read -p '[{title}] exited: press Enter to close this pane'",
    ])
    return {
        'type': 'Terminal',
        'uuid': str(uuid.uuid4()),
        'profile': TILIX_DEFAULT_PROFILE,
        'directory': workspace,
        'overrideCommand': f'bash -c "{script}"',
        'readOnly': False,
        'synchronizedInput': False,
        # required by Tilix to load the session; the real size comes from the window
        'width': 400,
        'height': 300,
    }


def node_executable(package, executable):
    """Shell expression of the path of a ROS 2 executable (what ros2 run would start)."""
    return f"$(ros2 pkg prefix {package})/lib/{package}/{executable}"


def tilix_split(children, orientation):
    """Nested Tilix panes with equal sizes. orientation 0 = side by side, 1 = stacked."""
    if len(children) == 1:
        return children[0]
    return {
        'type': 'Paned',
        'orientation': orientation,
        # divider: ratio = fraction of the space given to the first child (1/n -> equal panes);
        # position (pixels) is only a fallback, Tilix uses ratio when present
        'ratio': 1.0 / len(children),
        'position': 100,
        'child1': children[0],
        'child2': tilix_split(children[1:], orientation),
    }


def tilix_session(rows, workspace, width=1920, height=1080, theme='Adwaita:dark'):
    """ONE Tilix window: one row of side by side panes per element of rows, rows stacked.
    rows: list of lists of (title, command). Returns the ExecuteProcess opening it.
    width, height: size of the (maximized) window, in pixels: Tilix splits the panes in proportion
    to this size, so it must match the screen (e.g. 1920x1080 of the VNC server) for equal panes.
    theme: GTK theme of the window (GTK_THEME), 'Adwaita:dark' = dark; empty = system default.
    dbus-launch: Tilix needs a D-Bus session bus, which the container does not have.
    --new-process: independent Tilix process (no single-instance D-Bus handshake).
    """
    session = {
        'type': 'Session',
        'uuid': str(uuid.uuid4()),
        'name': 'autonomous manipulation',
        'version': '1.0',
        'synchronizedInput': False,
        'width': width,
        'height': height,
        'child': tilix_split([tilix_split([tilix_terminal(t, c, workspace) for t, c in row], 0)
                              for row in rows if row], 1),
    }
    with open(SESSION_FILE, 'w') as f:
        json.dump(session, f, indent=2)
    # private D-Bus session for Tilix; on Ctrl+C (SIGINT from ros2 launch) or SIGTERM Tilix is
    # closed with SIGTERM (a background job ignores SIGINT), the panes get SIGHUP and stop their
    # commands (see tilix_terminal); the bus daemon is killed at the end (no orphan dbus-daemon)
    wrapper = '\n'.join([
        'eval "$(dbus-launch --sh-syntax)"',
        f"tilix --new-process --maximize --session {SESSION_FILE} &",
        'tilix_pid=$!',
        'trap \'kill -TERM $tilix_pid 2>/dev/null; wait $tilix_pid\' INT TERM',
        'wait $tilix_pid',
        'kill $DBUS_SESSION_BUS_PID 2>/dev/null',
    ])
    return ExecuteProcess(
        cmd=['bash', '-c', wrapper],
        additional_env=dict(
            {'GTK_THEME': theme} if theme else {},
            # no accessibility bus / desktop modules of the host in the container (only warnings)
            NO_AT_BRIDGE='1',
            GTK_MODULES='',
        ),
        name='tilix',
        output='screen',
    )


def launch_setup(context):
    get = lambda name: context.launch_configurations[name]  # noqa: E731
    launch_driver = get('launch_driver').lower() in ('true', '1', 'yes')
    fsm_args = f"--ros-args --params-file {get('fsm_params_file')} --log-level {get('log_level')}"

    if get('separate_terminals').lower() in ('true', '1', 'yes'):
        # one Tilix window: driver | depth | Nav2 on top, manipulation | FSM below
        top = []
        if launch_driver:
            top.append(('Spot driver', f"ros2 launch spot_driver spot_driver.launch.py "
                                       f"config_file:={get('config_file')} launch_rviz:={get('launch_rviz')}"))
        top += [
            ('Depth', 'ros2 launch spot_motion depth_launch.py'),
            ('Nav2', 'ros2 launch spot_motion navigation_launch.py'),
        ]
        # nodes started as executables, not with ros2 run: the SIGINT of the pane reaches the node
        # itself (ros2 run would be in between); the FSM reads Enter directly from its pane
        bottom = [
            ('Manipulation', node_executable('manipulation', 'manipulation_node')),
            ('FSM', f"{node_executable('autonomous_manipulation_fsm', 'fsm_node')} {fsm_args}"),
        ]
        return [tilix_session([top, bottom], get('workspace'),
                              int(get('window_width')), int(get('window_height')), get('tilix_theme'))]

    actions = []
    if launch_driver:
        actions.append(include('spot_driver', 'spot_driver.launch.py',
                               {'config_file': get('config_file'), 'launch_rviz': get('launch_rviz')}))
    actions += [
        # depth images -> point clouds -> laser scans for the costmaps
        include('spot_motion', 'depth_launch.py'),
        include('spot_motion', 'navigation_launch.py'),
        # arm / grasp action servers (MoveArmToPose, Manipulate)
        Node(package='manipulation', executable='manipulation_node', name='manipulation_node',
             output='screen', emulate_tty=True),
        include('autonomous_manipulation_fsm', 'fsm_launch.py',
                {'params_file': get('fsm_params_file'), 'log_level': get('log_level')}),
    ]
    return actions


def generate_launch_description():
    default_fsm_params = os.path.join(
        get_package_share_directory('autonomous_manipulation_fsm'), 'config', 'fsm_params.yaml')

    return LaunchDescription([
        DeclareLaunchArgument('separate_terminals', default_value='true',
                              description='true: one Tilix window with one pane per component'),
        DeclareLaunchArgument('workspace', default_value='/home/spot_ws',
                              description='Workspace sourced in the Tilix panes'),
        DeclareLaunchArgument('window_width', default_value='1920',
                              description='Screen width (px) for the Tilix panes (VNC geometry)'),
        DeclareLaunchArgument('window_height', default_value='1080',
                              description='Screen height (px) for the Tilix panes (VNC geometry)'),
        DeclareLaunchArgument('tilix_theme', default_value='Adwaita:dark',
                              description="GTK theme of the Tilix window ('' = system default)"),
        DeclareLaunchArgument('launch_driver', default_value='true',
                              description='Start the Spot driver (false if it already runs elsewhere)'),
        DeclareLaunchArgument('config_file', default_value='/home/spot_ws/src/spot/config_spot/spot.yaml',
                              description='Spot driver configuration'),
        DeclareLaunchArgument('launch_rviz', default_value='True', description='RViz of the Spot driver'),
        DeclareLaunchArgument('fsm_params_file', default_value=default_fsm_params,
                              description='YAML file with the FSM parameters'),
        DeclareLaunchArgument('log_level', default_value='info', description='FSM log level'),
        OpaqueFunction(function=launch_setup),
    ])
