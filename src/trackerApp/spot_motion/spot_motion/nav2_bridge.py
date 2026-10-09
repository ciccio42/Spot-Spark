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

TARGET LOSS — BLIND FOLLOWING (single limit: BLIND_FOLLOW_SEC)
  tracking_fsm publishes ONLY real measurements. When they stop arriving,
  this node keeps moving for at most BLIND_FOLLOW_SEC after the last one:
  - for the first BLIND_PREDICT_SEC it follows the PREDICTED position
    (constant-velocity Kalman filter in odom): on a straight path it is
    accurate and it bridges short detection holes;
  - after that the point is FROZEN: a constant-velocity model cannot know
    that the person turned a corner, and extrapolating further only moves
    the goal away from them, straight into the wall. The robot keeps going
    towards the frozen point, i.e. towards where the person disappeared,
    stopping BLIND_STANDOFF_M from it (closer than the normal following
    distance, so that it reaches the corner and can see around it);
  - NAVIGATE: an already active navigation is updated. A NEW navigation is
    never started on the moving prediction; towards the FROZEN point it may
    start (BLIND_START_NAV), also from HOLD: otherwise a robot standing still
    when the person turned a corner would never get to see around it.
    HOLD: the centring keeps rotating towards the (predicted, then frozen) point.
  Beyond BLIND_FOLLOW_SEC: the navigation is cancelled and the final rotation
  (LOOK, below) starts. When the person is measured again: if the gap was
  longer than BLIND_PREDICT_SEC the filter restarts CLEANLY from the new
  measurement (KF_RESET_GAP_SEC), so a wrong prediction never leaves a
  spurious velocity behind; otherwise it is a normal update.

FINAL ROTATION AFTER BLIND FOLLOWING (LOOK)
  When blind following ends the robot would stop facing wherever the path
  left it — after a corner, usually the wall. Instead, for at most
  LOOK_TIMEOUT_SEC it rotates IN PLACE (never translates) so that the camera
  looks where the person probably went, and tracking_fsm can re-acquire them.
  Any new measurement ends it at once (normal behaviour resumes).
    LOOK_AT = "turn_side" (default): from the last SEEN position, along the
      direction the person was walking, turned by LOOK_TURN_DEG towards the
      side they were turning to (sign of the turn rate estimated from the
      last measurements). If they were not turning (|turn rate| below
      LOOK_TURN_MIN_DEG_S): straight along their direction of walking.
      Even when the person disappears right at the start of a turn, the
      estimated turn rate is small but its SIGN still says left or right.
    LOOK_AT = "last_seen": towards the last seen position.
    LOOK_AT = "predicted": towards the predicted (frozen) position.
    LOOK_AT = None: no final rotation.

TF AT THE TIME OF THE SHOT
  The person position is brought into odom with the TF at msg.header.stamp
  (the time the frame was captured), not the latest one: with 150-300 ms of
  perception latency and the robot rotating, the latest TF would place the
  person a few degrees off. If the TF at that time is not available, the
  latest one is used (and it is reported once in the log).

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
BLIND_PREDICT_SEC = 1.0     # blind: the prediction is used for at most this long, then the point is FROZEN
BLIND_FOLLOW_SEC = 4.0      # blind: total time moving without measurements, then navigation stops + LOOK
BLIND_STANDOFF_M = 1.5      # blind, frozen point: stop this far from it (normal following uses target_distance)
BLIND_MIN_FROM_SEEN_M = 1.0 # blind: the goal is never closer than this to where the person was last SEEN (they
                            # may have stopped right there while not being detected: the frozen point would then
                            # be beyond them, and BLIND_STANDOFF_M alone would bring the robot too close)
BLIND_START_NAV = True      # blind, frozen point: a NEW navigation towards it may start (also from HOLD).
                            # The frozen point is at most BLIND_PREDICT_SEC beyond a real measurement, i.e.
                            # "where the person disappeared", not a long extrapolation. Without this, a robot
                            # that was standing in HOLD when the person turned a corner never moves, and the
                            # corner keeps hiding them. During the first BLIND_PREDICT_SEC (moving prediction)
                            # a new navigation is never started. False = previous behaviour.
BLIND_GOAL_RATE_HZ = 5.0    # rate of the goal updates / loss checks while following blind
KF_RESET_GAP_SEC = BLIND_PREDICT_SEC  # measured again after a longer gap: the filter restarts from the new
                                      # measurement (no spurious velocity from a wrong prediction)
CANCEL_FORCE_SEC = 2.0      # if the cancellation result does not arrive, unblock anyway
LOOK_AT = "turn_side"       # final rotation: "turn_side", "last_seen", "predicted" or None (off)
LOOK_TIMEOUT_SEC = 4.0      # at most this long rotating (0.5 rad/s -> ~115 deg)
LOOK_MIN_DIST = 0.5         # m: closer than this the direction is meaningless -> no final rotation
LOOK_TURN_DEG = 60.0        # "turn_side": look this far off the walking direction, towards the turn
LOOK_TURN_MIN_DEG_S = 10.0  # "turn_side": below this turn rate the person is considered walking straight
LOOK_DIST_M = 2.0           # "turn_side": distance of the look point from the last seen position
HEADING_MIN_SPEED = 0.3     # m/s: below this the walking direction is not reliable
HEADING_WINDOW_SEC = 0.8    # turn rate = change of walking direction over this window
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
        self.declare_parameter('target_distance', 2.8)   # m, body -> person (like the SDK TARGET_DISTANCE)
        self.declare_parameter('enter_margin', 0.40)     # m beyond target_distance to start walking
        self.declare_parameter('exit_margin', 0.15)      # m beyond target_distance to stop
        self.declare_parameter('rot_gain', 1.2)          # rad/s per rad of error
        self.declare_parameter('max_rot_vel', 0.5)       # rad/s (= angular limit of the velocity_smoother)
        self.declare_parameter('rot_deadband_deg', 4.0)  # no rotation below this error
        self.declare_parameter('rot_timeout', BLIND_FOLLOW_SEC)  # s: beyond this, no rotation (predicted data included)
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
        self._kf_last_update_time = None   # time.monotonic() of the last REAL measurement
        self._kf_last_predict_time = None  # time.monotonic() the filter state is propagated to
        self._camera_frame = None
        self._warned_stamp_tf = False
        self._blind_announced = False
        self._last_seen = None             # filtered position at the last REAL measurement (odom)
        self._look_target = None           # point to rotate towards after blind following (odom)
        self._look_until = None            # time.monotonic() at which the final rotation gives up
        self._look_started = False         # final rotation already started for this loss
        self._heading_hist = []            # (time, walking direction) at the last measurements
        self._walk_heading = None          # walking direction at the last measurement (rad, odom)
        self._turn_rate = 0.0              # rad/s, estimated from the change of walking direction
        self._frozen_announced = False

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
        self.create_timer(1.0 / BLIND_GOAL_RATE_HZ, self._blind_follow_loop)
        self.create_timer(0.05, self._rotation_loop)  # 20 Hz

        self.get_logger().info(
            f"nav2_bridge pronto: distanza={self.D:.2f} m (cammina oltre {self.D + self.enter_margin:.2f}, "
            f"si ferma sotto {self.D + self.exit_margin:.2f}), centraggio su "
            f"{'camera (ROI)' if self.center_on_camera else self.base_frame}, "
            f"rotazione su '{gp('cmd_vel_topic')}', "
            f"behavior_tree={self._behavior_tree or '(default del bt_navigator)'}, "
            f"inseguimento alla cieca max {BLIND_FOLLOW_SEC:.1f} s.")

    # ------------------------------------------------------------------
    # Perception -> filtered position of the person in odom
    # ------------------------------------------------------------------
    def _lookup_at_stamp(self, target, source, stamp_msg):
        """TF at the time of the shot; falls back to the latest one if not available."""
        try:
            return self.tf_buffer.lookup_transform(target, source, rclpy.time.Time.from_msg(stamp_msg))
        except tf2_ros.TransformException as ex:
            if not self._warned_stamp_tf:
                self._warned_stamp_tf = True
                self.get_logger().warn(
                    f"TF all'istante dello scatto non disponibile ({ex}) — uso la piu' recente. "
                    f"Se succede sempre, controlla che robot e container abbiano l'orologio sincronizzato.")
        return self.tf_buffer.lookup_transform(target, source, rclpy.time.Time())

    def _on_target_pose(self, msg: TargetPose3D):
        try:
            transform = self._lookup_at_stamp(GLOBAL_FRAME, msg.header.frame_id, msg.header.stamp)
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
            self._heading_hist, self._walk_heading, self._turn_rate = [], None, 0.0  # old direction: meaningless
        else:
            self._advance_filter_to(now)  # the timer may already have propagated part of the gap
            self._kf_x.update(world.pose.position.x)
            self._kf_y.update(world.pose.position.y)
        self._kf_last_update_time = now
        self._kf_last_predict_time = now
        self._last_seen = (self._kf_x.pos, self._kf_y.pos)
        self._update_heading(now)
        self._frozen_announced = False
        if self._look_target is not None:
            self.get_logger().info("Target di nuovo misurato: fine rotazione finale.")
        self._look_target = None
        self._look_started = False
        if self._blind_announced:
            self.get_logger().info("Target di nuovo misurato: fine inseguimento alla cieca.")
            self._blind_announced = False

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
    def _decide(self, d, heading, px, py, predicted=False, standoff=None, may_start=False):
        """predicted=True: called on the PREDICTED / frozen position while blind. An
        active navigation is updated; a NEW one starts only if may_start (frozen point).
        standoff: distance to keep from the point (default: target_distance)."""
        if d < 1e-3:
            return
        D = self.D if standoff is None else standoff
        far = d > D + self.enter_margin
        near = d < D + self.exit_margin

        # goal at distance D from the person, on the robot->person line, facing them
        gx = px - D * math.cos(heading)
        gy = py - D * math.sin(heading)
        goal = self._pose(gx, gy, heading)

        if self.mode == HOLD and far and (not predicted or may_start):
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

        # Final rotation after blind following: only once the navigation has stopped.
        if not fresh and self._look_target is not None and not self._goal_active:
            self._look_step()
            return

        if not can_rotate:
            if self._was_rotating:
                self.cmd_pub.publish(Twist())  # an explicit stop when rotation ends
                self._was_rotating = False
            return

        self._advance_filter_to(time.monotonic())  # centre on the CURRENT (predicted) position
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

    def _start_look(self, now):
        """Chooses the point to look at when blind following ends (see LOOK_AT)."""
        if LOOK_AT is None:
            return
        if LOOK_AT == "turn_side":
            target = self._turn_side_point()
        elif LOOK_AT == "last_seen":
            target = self._last_seen
        else:
            target = (self._kf_x.pos, self._kf_y.pos)
        robot = self._robot_pose()
        if target is None or robot is None:
            return
        if math.hypot(target[0] - robot[0], target[1] - robot[1]) < LOOK_MIN_DIST:
            return
        self._look_target = target
        self._look_until = now + LOOK_TIMEOUT_SEC
        what = {"turn_side": f"il lato della svolta (turn rate {math.degrees(self._turn_rate):+.0f} deg/s)",
                "last_seen": "dove la persona e' stata vista l'ultima volta"}.get(
                    LOOK_AT, "dove la persona dovrebbe essere secondo la predizione")
        self.get_logger().info(f"Rotazione finale verso {what}: ({target[0]:.2f}, {target[1]:.2f}), "
                               f"max {LOOK_TIMEOUT_SEC:.1f} s.")

    def _blind_standoff(self, rx, ry, px, py, d, heading):
        """Distance to keep from the frozen point: BLIND_STANDOFF_M, increased (goal moved back
        along the robot->point line) until the goal is at least BLIND_MIN_FROM_SEEN_M from the
        last SEEN position, never beyond the robot itself."""
        standoff = BLIND_STANDOFF_M
        if self._last_seen is None:
            return standoff
        lx, ly = self._last_seen
        while standoff < d:
            gx, gy = px - standoff * math.cos(heading), py - standoff * math.sin(heading)
            if math.hypot(gx - lx, gy - ly) >= BLIND_MIN_FROM_SEEN_M:
                break
            standoff += 0.05
        return min(standoff, d)

    def _update_heading(self, now):
        """Walking direction (from the filtered velocity) and turn rate, at every measurement."""
        vx, vy = self._kf_x.vel, self._kf_y.vel
        if math.hypot(vx, vy) < HEADING_MIN_SPEED:
            return
        h = math.atan2(vy, vx)
        self._walk_heading = h
        self._heading_hist.append((now, h))
        self._heading_hist = [(t, a) for t, a in self._heading_hist if now - t <= HEADING_WINDOW_SEC]
        t0, h0 = self._heading_hist[0]
        if now - t0 > 1e-3:
            self._turn_rate = math.atan2(math.sin(h - h0), math.cos(h - h0)) / (now - t0)

    def _turn_side_point(self):
        """Point to look at for LOOK_AT = "turn_side" (see the docstring)."""
        if self._last_seen is None:
            return None
        if self._walk_heading is None:
            return self._last_seen
        direction = self._walk_heading
        if abs(self._turn_rate) >= math.radians(LOOK_TURN_MIN_DEG_S):
            direction += math.copysign(math.radians(LOOK_TURN_DEG), self._turn_rate)
        return (self._last_seen[0] + LOOK_DIST_M * math.cos(direction),
                self._last_seen[1] + LOOK_DIST_M * math.sin(direction))

    def _look_step(self):
        """One step of the final rotation: rotates the BODY towards the look point
        (the arm camera may not be aligned with the body, but the body is what Nav2
        and the next navigation start from). Stops when aligned or at the timeout."""
        now = time.monotonic()
        robot = self._robot_pose()
        if robot is None:
            return
        rx, ry, ryaw = robot
        tx, ty = self._look_target
        desired = math.atan2(ty - ry, tx - rx)
        err = math.atan2(math.sin(desired - ryaw), math.cos(desired - ryaw))
        if abs(err) < self.deadband or now > self._look_until:
            reason = "allineato" if abs(err) < self.deadband else "tempo scaduto"
            self.get_logger().info(f"Rotazione finale terminata ({reason}, errore {math.degrees(err):+.1f} deg).")
            self._look_target = None
            self.cmd_pub.publish(Twist())
            self._was_rotating = False
            return
        cmd = Twist()
        cmd.angular.z = _clamp(self.rot_gain * err, -self.max_rot, self.max_rot)
        self.cmd_pub.publish(cmd)
        self._was_rotating = True

    def _advance_filter_to(self, now):
        """Propagates the filter to 'now' (constant velocity), but never beyond
        BLIND_PREDICT_SEC after the last measurement: after that the predicted point
        is FROZEN. Idempotent: it only covers the time not yet propagated."""
        if self._kf_last_predict_time is None:
            return
        if self._kf_last_update_time is not None:
            now = min(now, self._kf_last_update_time + BLIND_PREDICT_SEC)
        dt = now - self._kf_last_predict_time
        if dt > 0.0:
            self._kf_x.predict(dt)
            self._kf_y.predict(dt)
            self._kf_last_predict_time = now

    def _blind_follow_loop(self):
        """While no measurement arrives: follows the PREDICTED position for at most
        BLIND_FOLLOW_SEC, then stops every motion (single limit)."""
        if self._kf_last_update_time is None:
            return
        now = time.monotonic()
        elapsed = now - self._kf_last_update_time
        if elapsed < 1.5 / BLIND_GOAL_RATE_HZ:
            return  # measurements are arriving: nothing to predict

        if elapsed > BLIND_FOLLOW_SEC:
            if self._goal_active and self._cancel_requested_at is None:
                self.get_logger().warn(f"Target perso da {elapsed:.1f}s: fine inseguimento alla cieca, "
                                       f"cancello la navigazione.")
                self._cancel_navigation()
            self.mode = HOLD
            if not self._look_started:
                self._look_started = True
                self._start_look(now)
            return

        if not self._blind_announced:
            self._blind_announced = True
            self.get_logger().info(f"Target non misurato: inseguo la posizione predetta "
                                   f"(max {BLIND_FOLLOW_SEC:.1f} s).")
        self._advance_filter_to(now)   # stops by itself at BLIND_PREDICT_SEC: then the point is frozen
        frozen = elapsed > BLIND_PREDICT_SEC
        if frozen and not self._frozen_announced:
            self._frozen_announced = True
            self.get_logger().info(f"Predizione ferma dopo {BLIND_PREDICT_SEC:.1f} s: vado verso il punto "
                                   f"({self._kf_x.pos:.2f}, {self._kf_y.pos:.2f}) fino a "
                                   f"{BLIND_STANDOFF_M:.1f} m, per altri {BLIND_FOLLOW_SEC - elapsed:.1f} s.")
        robot = self._robot_pose()
        if robot is None:
            return
        rx, ry, _ = robot
        px, py = self._kf_x.pos, self._kf_y.pos
        d = math.hypot(px - rx, py - ry)
        heading = math.atan2(py - ry, px - rx)
        self.debug_target_pub.publish(self._pose(px, py, heading))
        standoff = self._blind_standoff(rx, ry, px, py, d, heading) if frozen else None
        self._decide(d, heading, px, py, predicted=True, standoff=standoff,
                     may_start=frozen and BLIND_START_NAV)

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