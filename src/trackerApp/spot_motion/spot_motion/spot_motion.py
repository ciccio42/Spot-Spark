#!/usr/bin/env python3
"""
motion_command_node.py (spot_motion)

Node that receives TargetPose3D (RAW target position — not an already
scaled goal — in the camera OPTICAL frame, published by Pose3DEstimationNode)
and commands Spot to:

  1. ALWAYS stay facing the target (rotation, on every message received).
  2. Walk to converge to TARGET_DISTANCE, only moving closer (never
     backwards) and only beyond a DISTANCE_TOLERANCE error.

FRAME CHAIN — two different transforms, taken from two different sources:
  - body -> wrist (mechanical): LIVE, from kinematic_state.transforms_snapshot
    (robot_state_client) on every message.
  - wrist -> camera (optical): FIXED, taken ONCE at start-up from an
    ImageResponse.shot.transforms_snapshot (via ImageClient).
  The two are composed: body_tform_camera = body_tform_wrist * wrist_tform_camera.

KALMAN FILTER (constant velocity, 2 independent axes x/y): smooths the
target position in the ODOM frame (fixed — NEVER the body frame, which moves
with the robot). The filter is NOT the extrapolation — the extrapolation
(see below) READS the filter state, it never modifies it outside of a real
measurement received in _on_target_pose.

EXTRAPOLATION (_on_timer): runs on a timer INDEPENDENT of message
arrival — if no fresh TargetPose3D arrives, it projects forward the last
known position/velocity of the filter (without touching the persistent
state) and still sends a command, so Spot does not stop abruptly for a
short gap. Beyond MAX_EXTRAPOLATION_SEC since the last real measurement, it
stops guessing — letting COMMAND_DURATION stop Spot. NO change to the
tracking_fsm state: it relies only on the real time elapsed since the last
real measurement, not on the SEARCH/TRACKING/RECOVERY state.

CONCURRENCY — important, the cause of a hang observed in a previous
version: _on_target_pose and _on_timer BOTH make a real network call
(get_robot_state()). If they ran on the same thread (the default
single-threaded executor of rclpy.spin()), one would block the other every
time they happen close in time. Exactly the same problem — and the same
solution — already adopted in tracking_fsm.py: two separate callback groups
+ a MultiThreadedExecutor (see main()), so the two calls can run on
different threads without waiting for each other.

Uses RobotCommandBuilder.synchro_trajectory_command_in_body_frame(): it takes
a goal RELATIVE to the body (dx, dy, dyaw) + a snapshot of the transforms,
and converts it itself into the non-moving world frame (odom, hard-coded).

SPEED: limited via MobilityParams.vel_limit (MAX_LINEAR_VEL/MAX_ANGULAR_VEL
below) — pattern confirmed by the official Boston Dynamics example
spot_detect_and_follow.py.

Talks directly to the bosdyn SDK — bypasses Nav2 and spot_ros2's cmd_vel.
spot_driver (if running in parallel) must be launched with
auto_claim/auto_power_on/auto_stand set to false, so it does not compete for
the lease.

CONFIGURATION: constants below, not command-line arguments.
WARNING — plain-text SPOT_USERNAME/SPOT_PASSWORD are a TEMPORARY exception
for the initial test: they must be moved to environment variables
(BOSDYN_CLIENT_USERNAME/PASSWORD) before any commit.
"""
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor

import bosdyn.client
import bosdyn.client.util
from bosdyn.api import geometry_pb2
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import (
    BODY_FRAME_NAME, ODOM_FRAME_NAME, get_a_tform_b, get_se2_a_tform_b)
from bosdyn.client.image import ImageClient
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient, blocking_stand
from bosdyn.client.robot_state import RobotStateClient

from demo_interfaces.msg import TargetPose3D
from demo_package.common import CONE_MIN_RANGE

# ============================================================
# Configuration — edit here, not from the command line.
# ============================================================
SPOT_HOSTNAME = '192.168.80.3'  # <-- put the real IP of your Spot
SPOT_USERNAME = 'admin'         # <-- ONLY for the initial test, see note above
SPOT_PASSWORD = 'prb4e3wparqx'  #     to be moved to an env var before any commit
COMMAND_DURATION = 3.0          # seconds — end_time_secs safety net
DRY_RUN = False                 # True: compute and log WITHOUT ever sending commands to the robot
HAND_CAMERA_IMAGE_SOURCE = 'hand_color_image'
WRIST_FRAME_NAME = 'arm0.link_wr1'  # verified: present in both snapshots
TARGET_DISTANCE = 2.5           # metres — distance Spot always tries to keep
DISTANCE_TOLERANCE = 0.15       # metres — below this error, stay still (rotation only)
MAX_LINEAR_VEL = 0.6   # m/s — well below Spot's hardware maximum; lower it if you need
                         #       it even slower, the command never "goes faster" than this
MAX_ANGULAR_VEL = 0.5  # rad/s — same principle for rotation
KF_PROCESS_VAR = 0.05       # PROCESS noise — how much we expect the true velocity of the
                              # target to change (higher = filter more reactive to changes of
                              # direction/curves, but smooths noise less)
KF_MEASUREMENT_VAR = 0.05   # MEASUREMENT noise — how much we trust a single raw bx,by
                              # (higher = smooths more, but reacts more slowly to real changes)
KF_RESET_GAP_SEC = 2.0      # if more than this has passed since the last update, the filter is
                              # RE-INITIALISED on the new measurement instead of merging it with a
                              # state that is by now too old (e.g. after a long RECOVERY)
EXTRAPOLATION_TIMER_PERIOD = 0.2  # seconds between two checks when no fresh data
                                    # arrives — shorter than the typical interval between two
                                    # TargetPose3D (~0.3-0.6s), so gaps are noticed quickly
MAX_EXTRAPOLATION_SEC = 2.0       # beyond this time WITHOUT real data, stop walking
                                    # "blind" — no new command, let COMMAND_DURATION
                                    # stop Spot as the final safety net.
# ============================================================


def _se2_transform_point(a_tform_b, x, y):
    """Transforms a point (x,y) from frame b to frame a. SE2Pose has no
    direct transform_point (unlike Quat/SE3Pose, verified) — we get it by
    composing with .mult(), the only confirmed composition method: a point
    is an SE2Pose with angle=0, and the result of the composition inherits
    its transformed position."""
    result = a_tform_b.mult(math_helpers.SE2Pose(x, y, 0.0))
    return result.x, result.y


class _ConstantVelocityKalman1D:
    """1D Kalman filter, constant-velocity model: state [pos, vel].
    Used twice (independent x and y axes) to smooth the target position in
    the ODOM frame (fixed in the world — NEVER the body frame, which moves
    with the robot: estimating a velocity on coordinates that already move
    by themselves would mix the target motion with the robot motion)."""

    def __init__(self, process_var, measurement_var):
        self.pos = 0.0
        self.vel = 0.0
        self.P = [[1e3, 0.0], [0.0, 1e3]]  # high initial covariance: we do not trust anything yet
        self.q = process_var
        self.r = measurement_var

    def reset(self, pos):
        self.pos = pos
        self.vel = 0.0
        self.P = [[1e3, 0.0], [0.0, 1e3]]

    def predict(self, dt):
        self.pos = self.pos + self.vel * dt
        p00, p01, p10, p11 = self.P[0][0], self.P[0][1], self.P[1][0], self.P[1][1]
        self.P = [
            [p00 + dt * (p10 + p01) + dt * dt * p11 + self.q * dt, p01 + dt * p11],
            [p10 + dt * p11, p11 + self.q * dt],
        ]

    def update(self, measurement):
        innovation = measurement - self.pos
        s = self.P[0][0] + self.r
        k_pos = self.P[0][0] / s
        k_vel = self.P[1][0] / s
        self.pos = self.pos + k_pos * innovation
        self.vel = self.vel + k_vel * innovation
        p00, p01 = self.P[0][0], self.P[0][1]
        self.P = [
            [self.P[0][0] - k_pos * p00, self.P[0][1] - k_pos * p01],
            [self.P[1][0] - k_vel * p00, self.P[1][1] - k_vel * p01],
        ]


class MotionCommandNode(Node):
    def __init__(self, robot_state_client, robot_command_client, wrist_tform_camera):
        super().__init__('motion_command_node')
        self.robot_state_client = robot_state_client
        self.robot_command_client = robot_command_client
        self._wrist_tform_camera = wrist_tform_camera  # fixed, computed once in main()
        self._command_duration = COMMAND_DURATION
        self._dry_run = DRY_RUN
        self._mobility_params = spot_command_pb2.MobilityParams(
            vel_limit=geometry_pb2.SE2VelocityLimit(
                max_vel=geometry_pb2.SE2Velocity(
                    linear=geometry_pb2.Vec2(x=MAX_LINEAR_VEL, y=MAX_LINEAR_VEL),
                    angular=MAX_ANGULAR_VEL)))
        self._kf_x = _ConstantVelocityKalman1D(KF_PROCESS_VAR, KF_MEASUREMENT_VAR)
        self._kf_y = _ConstantVelocityKalman1D(KF_PROCESS_VAR, KF_MEASUREMENT_VAR)
        self._kf_last_update_time = None

        # Two separate callback groups: without this, _on_target_pose and
        # _on_timer would run on the same thread (rclpy default) and would
        # block each other every time both make a network call close in
        # time — see the note at the top of the file.
        pose_group = MutuallyExclusiveCallbackGroup()
        timer_group = MutuallyExclusiveCallbackGroup()

        self.create_subscription(TargetPose3D, 'target_3d', self._on_target_pose, 1,
                                  callback_group=pose_group)
        self.create_timer(EXTRAPOLATION_TIMER_PERIOD, self._on_timer,
                           callback_group=timer_group)

        self.get_logger().info(
            f"Pronto. command_duration={self._command_duration:.1f}s, "
            f"target_distance={TARGET_DISTANCE:.2f}m (+/-{DISTANCE_TOLERANCE:.2f}m), "
            f"vel_limit=({MAX_LINEAR_VEL:.2f}m/s, {MAX_ANGULAR_VEL:.2f}rad/s), frame mondo=odom, "
            f"max_extrapolation={MAX_EXTRAPOLATION_SEC:.1f}s. dry_run={self._dry_run} — "
            f"'NESSUN comando verra\\' inviato al robot' if self._dry_run else 'comandi ATTIVI'.")

    def _on_target_pose(self, msg: TargetPose3D):
        transforms = self.robot_state_client.get_robot_state().kinematic_state.transforms_snapshot

        body_tform_wrist = get_a_tform_b(transforms, BODY_FRAME_NAME, WRIST_FRAME_NAME)
        body_tform_camera = body_tform_wrist * self._wrist_tform_camera

        bx, by, bz = body_tform_camera.transform_point(
            msg.position.x, msg.position.y, msg.position.z)

        # --- Kalman filter, in the ODOM frame (fixed), not on bx,by directly ---
        odom_tform_body = get_se2_a_tform_b(transforms, ODOM_FRAME_NAME, BODY_FRAME_NAME)
        world_x, world_y = _se2_transform_point(odom_tform_body, bx, by)

        now = time.monotonic()
        if self._kf_last_update_time is None or (now - self._kf_last_update_time) > KF_RESET_GAP_SEC:
            self._kf_x.reset(world_x)
            self._kf_y.reset(world_y)
        else:
            dt = now - self._kf_last_update_time
            self._kf_x.predict(dt)
            self._kf_y.predict(dt)
            self._kf_x.update(world_x)
            self._kf_y.update(world_y)
        self._kf_last_update_time = now

        body_tform_odom = odom_tform_body.inverse()
        bx, by = _se2_transform_point(body_tform_odom, self._kf_x.pos, self._kf_y.pos)
        # --- end of filter ---

        self._build_and_send_command(bx, by, transforms, source_tag="reale")

    def _on_timer(self):
        """Runs every EXTRAPOLATION_TIMER_PERIOD, independently of the
        messages. Reads the filter state (without ever modifying it) to
        project the position forward during a short gap."""
        if self._kf_last_update_time is None:
            return  # no real data received yet

        elapsed = time.monotonic() - self._kf_last_update_time
        if elapsed < EXTRAPOLATION_TIMER_PERIOD or elapsed > MAX_EXTRAPOLATION_SEC:
            return

        extrapolated_world_x = self._kf_x.pos + self._kf_x.vel * elapsed
        extrapolated_world_y = self._kf_y.pos + self._kf_y.vel * elapsed

        transforms = self.robot_state_client.get_robot_state().kinematic_state.transforms_snapshot
        odom_tform_body = get_se2_a_tform_b(transforms, ODOM_FRAME_NAME, BODY_FRAME_NAME)
        body_tform_odom = odom_tform_body.inverse()
        bx, by = _se2_transform_point(body_tform_odom, extrapolated_world_x, extrapolated_world_y)

        self._build_and_send_command(bx, by, transforms, source_tag="estrapolato")

    def _build_and_send_command(self, bx, by, transforms, source_tag):
        goal_heading_rt_body = math.atan2(by, bx)
        r_horizontal = math.sqrt(bx * bx + by * by)

        if r_horizontal > TARGET_DISTANCE + DISTANCE_TOLERANCE:
            zone = "si avvicina"
            scale = (r_horizontal - TARGET_DISTANCE) / r_horizontal
            dx, dy = bx * scale, by * scale
        else:
            zone = "a distanza (fermo)"
            dx, dy = 0.0, 0.0

        mode_tag = f"[SIMULAZIONE-{source_tag}]" if self._dry_run else f"[INVIATO-{source_tag}]"
        self.get_logger().info(
            f"{mode_tag} zona={zone} bx={bx:.2f} by={by:.2f} "
            f"r_orizz={r_horizontal:.2f}m -> dx={dx:.2f}m dy={dy:.2f}m "
            f"yaw={math.degrees(goal_heading_rt_body):.1f}deg", throttle_duration_sec=0.5)

        if self._dry_run:
            return

        robot_cmd = RobotCommandBuilder.synchro_trajectory_command_in_body_frame(
            goal_x_rt_body=dx, goal_y_rt_body=dy, goal_heading_rt_body=goal_heading_rt_body,
            frame_tree_snapshot=transforms, params=self._mobility_params)

        self.robot_command_client.robot_command(
            lease=None, command=robot_cmd,
            end_time_secs=time.time() + self._command_duration)


def main():
    rclpy.init()

    sdk = bosdyn.client.create_standard_sdk('SpotMotionClient')
    robot = sdk.create_robot(SPOT_HOSTNAME)
    robot.authenticate(SPOT_USERNAME, SPOT_PASSWORD)
    robot.time_sync.wait_for_sync()

    assert not robot.is_estopped(), (
        "Robot in e-stop. Serve un client e-stop attivo prima di poter comandare movimento.")

    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    robot_command_client = robot.ensure_client(RobotCommandClient.default_service_name)
    image_client = robot.ensure_client(ImageClient.default_service_name)

    image_responses = image_client.get_image_from_sources([HAND_CAMERA_IMAGE_SOURCE])
    camera_snapshot = image_responses[0].shot.transforms_snapshot
    camera_frame_name = image_responses[0].shot.frame_name_image_sensor
    wrist_tform_camera = get_a_tform_b(camera_snapshot, WRIST_FRAME_NAME, camera_frame_name)
    assert wrist_tform_camera is not None, (
        f"get_a_tform_b non ha trovato un percorso tra '{WRIST_FRAME_NAME}' e "
        f"'{camera_frame_name}' nello snapshot immagine.")

    with LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        robot.power_on()
        blocking_stand(robot_command_client)

        node = MotionCommandNode(robot_state_client, robot_command_client, wrist_tform_camera)
        # MultiThreadedExecutor, not rclpy.spin(node): _on_target_pose and
        # _on_timer are in separate callback groups precisely so they can run
        # on different threads — with the default single-threaded executor,
        # the two callback groups would be useless, they would still be
        # queued one after the other.
        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(node)
        try:
            executor.spin()
        except KeyboardInterrupt:
            pass
        finally:
            robot_command_client.robot_command(RobotCommandBuilder.stop_command())
            node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()