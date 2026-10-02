#!/usr/bin/env python3
"""
nav2_bridge.py (spot_motion) — person following with Nav2 + centring.

From TargetPose3D (WHERE the person is) to WHAT the robot must do.
Restores, on top of Nav2, the policy of the old SDK node (spot_motion.py):
"translate only if the person leaves the distance band, always rotate to
keep them centred".

MODES
  NAVIGATE  the person is far (d > target_distance + enter_margin):
            Nav2 goal at target_distance FROM the person, on the
            robot->person line, facing them. Updated via
            /goal_update (GoalUpdater in the BT), as before.
  HOLD      the person is inside the band (or too close — never reverse):
            no Nav2 goal; the bridge rotates the robot in place to bring
            the person back to the centre of the image (= of the ROI).
  Hysteresis: enter NAVIGATE above target_distance + enter_margin and go
  back to HOLD below target_distance + exit_margin (or when the goal is
  reached), so as not to keep switching between the two modes.

CENTRING (HOLD)
  The error is computed NOW, not at the time of the shot: the filtered
  position of the person (in odom, fixed) is brought back into the camera
  frame with the current TF. So it accounts for the rotation the robot has
  already made, the perception latency and the arm misalignment.
      bearing = atan2(x_cam, z_cam)   (optical frame: x right, z forward)
      w = -rot_gain * bearing         (ROS: positive rotation = counter-clockwise)
  With center_on_camera:=false it centres relative to the body instead of
  the camera.

ROTATION COMMAND -> cmd_vel_nav (not /cmd_vel)
  It is the input of the velocity_smoother, the same topic the Nav2
  controller writes to. This way only one node (the smoother) writes on
  /cmd_vel, the rotation is acceleration-limited like the navigation, and
  there are no two conflicting sources. The bridge only rotates when no
  NavigateToPose is active.

BEHAVIOR TREE
  The goal is already at a distance from the person: use a BT WITHOUT
  truncation (follow_point_spot.xml, TruncatePath distance="0.0" = the
  bt_navigator default). With follow_person_spot.xml the distance would be
  added twice. That is why behavior_tree is empty by default.

TARGET LOSS
  - rotation: stops if the last data is older than rot_timeout
    (no turning on stale data);
  - navigation: cancelled after MAX_TARGET_LOSS_SEC, as before.

DEBUG TOPICS (RViz)
  ~/filtered_target  filtered position of the person (odom)
  ~/standoff_goal    point at target_distance from the person (Nav2 goal)
"""
import math
import time

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

import tf2_ros
from tf2_geometry_msgs import do_transform_pose_stamped

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, Twist
from nav2_msgs.action import NavigateToPose

from demo_interfaces.msg import TargetPose3D

# ============================================================
GLOBAL_FRAME = 'odom'       # must match global_frame in the Nav2 parameters
KF_PROCESS_VAR = 0.05
KF_MEASUREMENT_VAR = 0.05
KF_RESET_GAP_SEC = 1.0
MAX_TARGET_LOSS_SEC = 5.0
CANCEL_FORCE_SEC = 2.0      # if the cancellation result does not arrive, unblock anyway
# ============================================================

NAVIGATE = 'NAVIGATE'
HOLD = 'HOLD'


class _ConstantVelocityKalman1D:
    """Unchanged from the previous version."""

    def __init__(self, process_var, measurement_var):
        self.pos = 0.0
        self.vel = 0.0
        self.P = [[1e3, 0.0], [0.0, 1e3]]
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


def _yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def _quat_from_yaw(yaw):
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


class Nav2Bridge(Node):
    def __init__(self):
        super().__init__('nav2_bridge')

        # --- parameters ---
        self.declare_parameter('target_distance', 2.5)   # m, body -> person (like the SDK TARGET_DISTANCE)
        self.declare_parameter('enter_margin', 0.40)     # m beyond target_distance to start walking
        self.declare_parameter('exit_margin', 0.15)      # m beyond target_distance to stop
        self.declare_parameter('rot_gain', 1.2)          # rad/s per rad of error
        self.declare_parameter('max_rot_vel', 0.5)       # rad/s (= angular limit of the velocity_smoother)
        self.declare_parameter('rot_deadband_deg', 4.0)  # no rotation below this error
        self.declare_parameter('rot_timeout', 1.0)       # s: beyond this, no rotation on stale data
        self.declare_parameter('center_on_camera', True) # centre in the image (ROI) or relative to the body
        self.declare_parameter('base_frame', 'body')
        self.declare_parameter('cmd_vel_topic', 'cmd_vel_nav')
        self.declare_parameter('behavior_tree', '')      # empty = default BT (no truncation)

        gp = lambda n: self.get_parameter(n).value
        self.D = float(gp('target_distance'))
        self.enter_margin = float(gp('enter_margin'))
        self.exit_margin = float(gp('exit_margin'))
        self.rot_gain = float(gp('rot_gain'))
        self.max_rot = float(gp('max_rot_vel'))
        self.deadband = math.radians(float(gp('rot_deadband_deg')))
        self.rot_timeout = float(gp('rot_timeout'))
        self.center_on_camera = bool(gp('center_on_camera'))
        self.base_frame = gp('base_frame')
        self._behavior_tree = gp('behavior_tree')

        # --- state ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self._kf_x = _ConstantVelocityKalman1D(KF_PROCESS_VAR, KF_MEASUREMENT_VAR)
        self._kf_y = _ConstantVelocityKalman1D(KF_PROCESS_VAR, KF_MEASUREMENT_VAR)
        self._kf_last_update_time = None
        self._camera_frame = None

        self.mode = HOLD
        self._goal_active = False
        self._goal_handle = None
        self._cancel_requested_at = None
        self._was_rotating = False

        # --- I/O ---
        self.goal_update_pub = self.create_publisher(PoseStamped, 'goal_update', 5)
        self.debug_target_pub = self.create_publisher(PoseStamped, '~/filtered_target', 5)
        self.debug_goal_pub = self.create_publisher(PoseStamped, '~/standoff_goal', 5)
        self.cmd_pub = self.create_publisher(Twist, gp('cmd_vel_topic'), 10)
        self._nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        self.create_subscription(TargetPose3D, 'target_3d', self._on_target_pose, 5)
        self.create_timer(1.0, self._check_target_loss)
        self.create_timer(0.05, self._rotation_loop)  # 20 Hz

        self.get_logger().info(
            f"nav2_bridge pronto: distanza={self.D:.2f} m (cammina oltre {self.D + self.enter_margin:.2f}, "
            f"si ferma sotto {self.D + self.exit_margin:.2f}), centraggio su "
            f"{'camera (ROI)' if self.center_on_camera else self.base_frame}, "
            f"rotazione su '{gp('cmd_vel_topic')}', "
            f"behavior_tree={self._behavior_tree or '(default del bt_navigator)'}.")

    # ------------------------------------------------------------------
    # Perception -> filtered position of the person in odom
    # ------------------------------------------------------------------
    def _on_target_pose(self, msg: TargetPose3D):
        try:
            transform = self.tf_buffer.lookup_transform(GLOBAL_FRAME, msg.header.frame_id, rclpy.time.Time())
        except tf2_ros.TransformException as ex:
            self.get_logger().warn(f"TF non disponibile ({msg.header.frame_id}->{GLOBAL_FRAME}): {ex}",
                                   throttle_duration_sec=2.0)
            return
        self._camera_frame = msg.header.frame_id

        raw_pose = PoseStamped()
        raw_pose.header = msg.header
        raw_pose.pose.position = msg.position
        raw_pose.pose.orientation.w = 1.0
        world = do_transform_pose_stamped(raw_pose, transform)

        now = time.monotonic()
        if self._kf_last_update_time is None or (now - self._kf_last_update_time) > KF_RESET_GAP_SEC:
            self._kf_x.reset(world.pose.position.x)
            self._kf_y.reset(world.pose.position.y)
        else:
            dt = now - self._kf_last_update_time
            self._kf_x.predict(dt)
            self._kf_y.predict(dt)
            self._kf_x.update(world.pose.position.x)
            self._kf_y.update(world.pose.position.y)
        self._kf_last_update_time = now

        robot = self._robot_pose()
        if robot is None:
            return
        rx, ry, _ = robot
        px, py = self._kf_x.pos, self._kf_y.pos
        dx, dy = px - rx, py - ry
        d = math.hypot(dx, dy)
        heading = math.atan2(dy, dx)  # robot -> person direction, in odom

        self.debug_target_pub.publish(self._pose(px, py, heading))
        self._decide(d, heading, px, py)

    # ------------------------------------------------------------------
    # Distance policy (with hysteresis)
    # ------------------------------------------------------------------
    def _decide(self, d, heading, px, py):
        if d < 1e-3:
            return
        far = d > self.D + self.enter_margin
        near = d < self.D + self.exit_margin

        # goal at distance D from the person, on the robot->person line, facing them
        gx = px - self.D * math.cos(heading)
        gy = py - self.D * math.sin(heading)
        goal = self._pose(gx, gy, heading)

        if self.mode == HOLD and far:
            self.mode = NAVIGATE
            self.get_logger().info(f"Persona a {d:.2f} m -> NAVIGATE")
        elif self.mode == NAVIGATE and near:
            self.mode = HOLD
            self.get_logger().info(f"Persona a {d:.2f} m -> HOLD (stop e centraggio)")
            self._cancel_navigation()
            return

        if self.mode == NAVIGATE:
            self.debug_goal_pub.publish(goal)
            if not self._goal_active:
                self._send_initial_goal(goal)
            elif self._cancel_requested_at is None:
                self.goal_update_pub.publish(goal)

    # ------------------------------------------------------------------
    # Centring: in-place rotation in HOLD
    # ------------------------------------------------------------------
    def _rotation_loop(self):
        # Force unblocking if the cancellation confirmation does not arrive
        if self._cancel_requested_at is not None and time.monotonic() - self._cancel_requested_at > CANCEL_FORCE_SEC:
            self.get_logger().warn("Cancellazione non confermata: considero la navigazione terminata.")
            self._goal_active = False
            self._goal_handle = None
            self._cancel_requested_at = None

        fresh = (self._kf_last_update_time is not None
                 and time.monotonic() - self._kf_last_update_time < self.rot_timeout)
        can_rotate = self.mode == HOLD and not self._goal_active and fresh

        if not can_rotate:
            if self._was_rotating:
                self.cmd_pub.publish(Twist())  # an explicit stop when rotation ends
                self._was_rotating = False
            return

        err = self._centering_error()
        if err is None:
            return
        w = 0.0 if abs(err) < self.deadband else _clamp(self.rot_gain * err, -self.max_rot, self.max_rot)

        cmd = Twist()
        cmd.angular.z = w
        self.cmd_pub.publish(cmd)
        self._was_rotating = True
        self.get_logger().info(f"HOLD: errore={math.degrees(err):+.1f} deg  w={w:+.2f} rad/s",
                               throttle_duration_sec=1.0)

    def _centering_error(self):
        """Angular error (rad, positive = the person is on the LEFT, so rotate
        counter-clockwise), computed with the CURRENT TF."""
        px, py = self._kf_x.pos, self._kf_y.pos
        if self.center_on_camera and self._camera_frame:
            try:
                tf_cam = self.tf_buffer.lookup_transform(self._camera_frame, GLOBAL_FRAME, rclpy.time.Time())
            except tf2_ros.TransformException:
                return None
            p = do_transform_pose_stamped(self._pose(px, py, 0.0), tf_cam).pose.position
            # optical frame: x right, z forward -> positive bearing = to the right
            return -math.atan2(p.x, p.z)
        robot = self._robot_pose()
        if robot is None:
            return None
        rx, ry, ryaw = robot
        desired = math.atan2(py - ry, px - rx)
        return math.atan2(math.sin(desired - ryaw), math.cos(desired - ryaw))

    # ------------------------------------------------------------------
    # Nav2
    # ------------------------------------------------------------------
    def _send_initial_goal(self, goal: PoseStamped):
        if not self._nav_client.wait_for_server(timeout_sec=0.5):
            self.get_logger().warn("Action navigate_to_pose non disponibile.", throttle_duration_sec=5.0)
            return
        req = NavigateToPose.Goal()
        req.pose = goal
        if self._behavior_tree:
            req.behavior_tree = self._behavior_tree
        self._goal_active = True
        self._nav_client.send_goal_async(req).add_done_callback(self._on_goal_response)
        self.get_logger().info(f"NavigateToPose inviata: ({goal.pose.position.x:.2f}, {goal.pose.position.y:.2f})")

    def _on_goal_response(self, future):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().warn("Goal NavigateToPose rifiutato.")
            self._goal_active = False
            return
        self._goal_handle = handle
        # If we switched to HOLD in the meantime, cancel right away
        if self.mode == HOLD:
            self._cancel_navigation()
        handle.get_result_async().add_done_callback(self._on_goal_result)

    def _on_goal_result(self, future):
        status = future.result().status
        names = {GoalStatus.STATUS_SUCCEEDED: 'SUCCEEDED', GoalStatus.STATUS_CANCELED: 'CANCELED',
                 GoalStatus.STATUS_ABORTED: 'ABORTED'}
        self.get_logger().info(f"Navigazione terminata: {names.get(status, status)}")
        self._goal_active = False
        self._goal_handle = None
        self._cancel_requested_at = None
        if status == GoalStatus.STATUS_SUCCEEDED and self.mode == NAVIGATE:
            self.mode = HOLD  # reached the distance: centring from here on

    def _cancel_navigation(self):
        if self._goal_handle is not None and self._cancel_requested_at is None:
            self._goal_handle.cancel_goal_async()
            self._cancel_requested_at = time.monotonic()

    def _check_target_loss(self):
        if self._kf_last_update_time is None:
            return
        elapsed = time.monotonic() - self._kf_last_update_time
        if elapsed > MAX_TARGET_LOSS_SEC and self._goal_active:
            self.get_logger().warn(f"Target perso da {elapsed:.1f}s: cancello la navigazione.")
            self._cancel_navigation()
            self.mode = HOLD

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def _robot_pose(self):
        try:
            t = self.tf_buffer.lookup_transform(GLOBAL_FRAME, self.base_frame, rclpy.time.Time())
        except tf2_ros.TransformException as ex:
            self.get_logger().warn(f"Posa robot non disponibile ({GLOBAL_FRAME}->{self.base_frame}): {ex}",
                                   throttle_duration_sec=2.0)
            return None
        tr = t.transform.translation
        return tr.x, tr.y, _yaw_from_quat(t.transform.rotation)

    def _pose(self, x, y, yaw):
        p = PoseStamped()
        p.header.stamp = self.get_clock().now().to_msg()
        p.header.frame_id = GLOBAL_FRAME
        p.pose.position.x = x
        p.pose.position.y = y
        qx, qy, qz, qw = _quat_from_yaw(yaw)
        p.pose.orientation.x, p.pose.orientation.y = qx, qy
        p.pose.orientation.z, p.pose.orientation.w = qz, qw
        return p


def main():
    rclpy.init()
    node = Nav2Bridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()