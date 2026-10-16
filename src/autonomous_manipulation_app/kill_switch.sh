#!/usr/bin/env bash
# kill_switch.sh — emergency stop of the autonomous manipulation app. Run it on the HOST.
#
#   ./kill_switch.sh          Ctrl+C: stops the robot and kills the app (the Spot driver keeps running,
#                             so the robot stays controllable: sit it from the tablet or with /sit)
#   ./kill_switch.sh --all    Ctrl+C: same, then also kills the Spot driver
#
# On Ctrl+C:
#   1. stop the robot: cancel every Nav2 / arm / manipulation goal and call the driver /stop service
#   2. SIGINT to the app in both containers (fsm_node stows the arm on its own shutdown)
#   3. after GRACE_SEC, SIGKILL to whatever is still alive
# A second Ctrl+C during the shutdown skips the wait and kills everything immediately.
#
# This is NOT a replacement for the E-Stop: if the robot is in danger, use the E-Stop.
#
# Container names (override with env vars if yours differ):
#   SPOT_CONTAINER=spot-autonomous-manipulation-container
#   YOLO_CONTAINER=yolo-autonomous-manipulation-container

SPOT_CONTAINER=${SPOT_CONTAINER:-spot-autonomous-manipulation-container}
YOLO_CONTAINER=${YOLO_CONTAINER:-yolo-autonomous-manipulation-container}
GRACE_SEC=${GRACE_SEC:-12}   # fsm_node needs up to ~8 s to stow the arm on shutdown

KILL_DRIVER=false
[[ "$1" == "--all" ]] && KILL_DRIVER=true

# process patterns (pkill -f). The first letter in brackets keeps the pattern from matching the
# command line of the shell running pkill itself.
SPOT_APP='[f]sm_node|[f]sm_launch|[m]anipulation_node|[s]ave_arm_pose|[n]avigation_launch|[p]lanner_server|[c]ontroller_server|[b]t_navigator|[b]ehavior_server|[v]elocity_smoother|[l]ifecycle_manager|[c]ostmap_refresher|[n]av2_bridge|[d]epth_launch|[d]epth_to_pointcloud_container'
YOLO_APP='[d]etector_node|[d]etector_launch'
SPOT_DRIVER='[s]pot_driver'

# action servers whose goals are cancelled on stop
ACTIONS=(/navigate_to_pose /manipulation_custom /move_arm_to_pose /manipulation /robot_command)

running() { [[ "$(docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null)" == "true" ]]; }

# ros_exec <container> <workspace> <command>: runs a ROS 2 command inside a container
ros_exec() {
    docker exec "$1" bash -c "source /opt/ros/humble/setup.bash; \
        [ -f $2/install/setup.bash ] && source $2/install/setup.bash; $3" >/dev/null 2>&1
}

# signal <container> <SIGNAL> <pattern>
signal() { running "$1" && docker exec "$1" pkill "-$2" -f "$3" >/dev/null 2>&1; }

# alive <container> <pattern>: true if some matching process is still running
alive() { running "$1" && docker exec "$1" pgrep -f "$2" >/dev/null 2>&1; }

stop_robot() {
    echo "[kill_switch] Stopping the robot: cancelling all goals and calling /stop"
    running "$SPOT_CONTAINER" || { echo "[kill_switch] $SPOT_CONTAINER is not running"; return; }
    local cmds=""
    for action in "${ACTIONS[@]}"; do
        # empty CancelGoal request (zero goal id and stamp) = cancel ALL goals of the action server
        cmds+="timeout 3 ros2 service call ${action}/_action/cancel_goal action_msgs/srv/CancelGoal '{}' & "
    done
    cmds+="timeout 3 ros2 service call /stop std_srvs/srv/Trigger & wait"
    ros_exec "$SPOT_CONTAINER" /home/spot_ws "$cmds"
}

kill_now() {
    echo "[kill_switch] SIGKILL to everything still running"
    signal "$SPOT_CONTAINER" KILL "$SPOT_APP"
    signal "$YOLO_CONTAINER" KILL "$YOLO_APP"
    $KILL_DRIVER && signal "$SPOT_CONTAINER" KILL "$SPOT_DRIVER"
    exit 1
}

shutdown_all() {
    # a second Ctrl+C (or SIGTERM) from now on skips the grace period
    trap kill_now INT TERM

    stop_robot

    echo "[kill_switch] SIGINT to the app nodes (fsm_node stows the arm while exiting)"
    signal "$SPOT_CONTAINER" INT "$SPOT_APP"
    signal "$YOLO_CONTAINER" INT "$YOLO_APP"

    local waited=0
    while (( waited < GRACE_SEC )); do
        alive "$SPOT_CONTAINER" "$SPOT_APP" || alive "$YOLO_CONTAINER" "$YOLO_APP" || break
        sleep 1
        (( waited++ ))
    done
    if alive "$SPOT_CONTAINER" "$SPOT_APP" || alive "$YOLO_CONTAINER" "$YOLO_APP"; then
        echo "[kill_switch] Still running after ${GRACE_SEC} s, killing"
        signal "$SPOT_CONTAINER" KILL "$SPOT_APP"
        signal "$YOLO_CONTAINER" KILL "$YOLO_APP"
    fi

    if $KILL_DRIVER; then
        echo "[kill_switch] Stopping the Spot driver"
        signal "$SPOT_CONTAINER" INT "$SPOT_DRIVER"
        sleep 5
        alive "$SPOT_CONTAINER" "$SPOT_DRIVER" && signal "$SPOT_CONTAINER" KILL "$SPOT_DRIVER"
    else
        echo "[kill_switch] Spot driver left running: the robot is still standing."
        echo "[kill_switch] Sit it with the tablet or: docker exec -it $SPOT_CONTAINER bash -c" \
             "'source /opt/ros/humble/setup.bash && ros2 service call /sit std_srvs/srv/Trigger'"
    fi
    echo "[kill_switch] Done"
    exit 0
}

for c in "$SPOT_CONTAINER" "$YOLO_CONTAINER"; do
    running "$c" || echo "[kill_switch] Warning: container $c is not running"
done

trap shutdown_all INT TERM
echo "[kill_switch] Armed. Press Ctrl+C to stop the robot and kill the app$($KILL_DRIVER && echo ' and the driver')."
echo "[kill_switch] (This is not an E-Stop: in danger, use the E-Stop.)"
while true; do
    sleep 1
done
