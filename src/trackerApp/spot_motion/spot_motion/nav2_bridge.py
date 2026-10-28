#!/usr/bin/env python3
"""
nav2_bridge.py (spot_motion) — person following with Nav2 + centring.

From WHERE the person is (and WHERE they are expected to go) to WHAT the robot
must do. Restores, on top of Nav2, the policy of the old SDK node
(spot_motion.py): "translate only if the person leaves the distance band,
always rotate to keep them centred".

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

ONE PREDICTION FOR THE WHOLE SYSTEM: THE TRAJECTORY FROM tracking_fsm
  This node has NO filter of its own. tracking_fsm publishes, at every call in
  TRACKING / RECOVERY:
    TARGET_STATE_TOPIC (nav_msgs/Odometry, odom): the state of its filter;
      child_frame_id "target/measured" (measured in this call),
      "target/predicted" (not measured) or "target/frozen" (prediction stopped);
    TARGET_PATH_TOPIC (nav_msgs/Path, odom): the FUTURE TRAJECTORY of the
      person from the last measurement, one pose every ~0.33 s, each with its
      ABSOLUTE time: straight if they were walking straight, along the arc if
      they were turning (tighter if the curve was tightening), until it is
      reliable. The same points are drawn in the debug image.
  The person position used here (centring, goals, blind following) is the
  point of the trajectory at the CURRENT ROS time, interpolated between two
  poses: it also compensates the perception latency (the image is 0.2-0.3 s
  old when it arrives). After the last pose the point stays there (frozen).

TARGET LOSS — BLIND FOLLOWING (single limit: BLIND_FOLLOW_SEC)
  When the person is not measured, the robot keeps following the trajectory
  for at most BLIND_FOLLOW_SEC after the last measurement:
  - while the point moves along the trajectory an ALREADY active navigation is
    updated; a NEW navigation is never started on a moving prediction;
  - at the end of the trajectory (or "target/frozen") the point is FROZEN: the
    robot goes towards it, stopping BLIND_STANDOFF_M from it (closer than the
    normal following distance, so that it reaches the corner and sees around
    it); a NEW navigation may start (BLIND_START_NAV), also from HOLD.
  Every point is checked on the Nav2 global costmap: if it is in an obstacle
  (or its inflation) it is moved back towards the last seen position until it
  is free (the last seen position at worst).
  Beyond BLIND_FOLLOW_SEC: the navigation is cancelled and the final rotation
  (LOOK) starts.

FINAL ROTATION AFTER BLIND FOLLOWING (LOOK)
  For at most LOOK_TIMEOUT_SEC the robot rotates IN PLACE (never translates)
  so that the camera looks where the person probably went, and tracking_fsm
  can re-acquire them. Any new measurement ends it at once.
    LOOK_AT = "trajectory" (default): LOOK_DIST_M beyond the END of the last
      trajectory, along its final direction (after a curve: round the corner).
      A trajectory without a direction (person standing): the last seen position.
    LOOK_AT = "last_seen": towards the last seen position.
    LOOK_AT = None: no final rotation.

WITHOUT THE TRAJECTORY (fallback, e.g. tracking_fsm without TF)
  The person position is the last target_3d (brought into odom with the TF at
  the time of the shot), with NO prediction: when it stops arriving the point
  is frozen at once at the last seen position.

CENTRING (HOLD)
  The error is computed NOW: the person position (in odom, fixed) is brought
  back into the camera frame with the current TF. So it accounts for the
  rotation the robot has already made and the arm misalignment.
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

CLOCKS
  The trajectory times are ROS times (of the images): this node reads them
  with its ROS clock (get_clock). Robot and containers must be synchronised
  (with use_sim_time in simulation).

DEBUG TOPICS (RViz)
  ~/filtered_target  person position used now (odom)
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
from nav_msgs.msg import Odometry, OccupancyGrid, Path
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from demo_interfaces.msg import TargetPose3D

# ============================================================
GLOBAL_FRAME = 'odom'       # must match global_frame in the Nav2 parameters
TARGET_STATE_TOPIC = '/person_follow/target_state'   # tracking_fsm: filter state (Odometry)
TARGET_PATH_TOPIC = '/person_follow/target_path'     # tracking_fsm: future trajectory (Path)
PATH_TIMEOUT_SEC = 1.0      # trajectory older than this: tracking_fsm is not publishing it -> fallback (target_3d)
BLIND_FOLLOW_SEC = 8.0      # blind: total time moving without measurements, then navigation stops + LOOK
                            # (tracking_fsm keeps looking for ~10 s before RECOVERY: the robot moves meanwhile)
BLIND_STANDOFF_M = 1.5      # blind, frozen point: stop this far from it (normal following uses target_distance)
BLIND_MIN_FROM_SEEN_M = 1.0 # blind: the goal is never closer than this to where the person was last SEEN (they
                            # may have stopped right there while not being detected: the frozen point would then
                            # be beyond them, and BLIND_STANDOFF_M alone would bring the robot too close)
BLIND_START_NAV = True      # blind, frozen point: a NEW navigation towards it may start (also from HOLD); on the
                            # moving prediction a new navigation is never started. False = only updates.
BLIND_GOAL_RATE_HZ = 5.0    # rate of the goal updates / loss checks while following blind
CANCEL_FORCE_SEC = 2.0      # if the cancellation result does not arrive, unblock anyway
LOOK_AT = "trajectory"      # final rotation: "trajectory", "last_seen" or None (off)
LOOK_TIMEOUT_SEC = 4.0      # at most this long rotating (0.5 rad/s -> ~115 deg)
LOOK_MIN_DIST = 0.5         # m: closer than this the direction is meaningless -> no final rotation
LOOK_DIST_M = 2.0           # "trajectory": distance of the look point beyond the end of the trajectory
COSTMAP_TOPIC = '/global_costmap/costmap'
COSTMAP_BLOCKED = 99        # OccupancyGrid value from which a cell is not reachable (99 inscribed, 100 lethal)
COSTMAP_STEP_M = 0.1        # step when moving a blocked point back towards the last seen position
# ============================================================

NAVIGATE = 'NAVIGATE'
HOLD = 'HOLD'


def _interpolate(path, t):
    """(x, y, at_end) of the trajectory [(t, x, y)] at time t: linear between two poses, the first
    pose before it, the LAST pose after it (at_end = True: the prediction is over, frozen point)."""
    if t <= path[0][0]:
        return path[0][1], path[0][2], len(path) == 1
    for (t0, x0, y0), (t1, x1, y1) in zip(path, path[1:]):
        if t <= t1:
            f = (t - t0) / (t1 - t0) if t1 > t0 else 1.0
            return x0 + f * (x1 - x0), y0 + f * (y1 - y0), False
    return path[-1][1], path[-1][2], True


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
        self._path = []                    # trajectory from tracking_fsm: [(ROS time s, x, y)] in odom
        self._path_time = None             # time.monotonic() it arrived
        self._state_kind = None            # "measured" / "predicted" / "frozen" (last tracking_fsm state)
        self._raw = None                   # fallback: last target_3d position (odom), no prediction
        self._last_meas_time = None        # time.monotonic() of the last REAL measurement
        self._last_seen = None             # person position at the last REAL measurement (odom)
        self._camera_frame = None
        self._warned_stamp_tf = False
        self._warned_frame = False
        self._blind_announced = False
        self._frozen_announced = False
        self._look_target = None           # point to rotate towards after blind following (odom)
        self._look_until = None            # time.monotonic() at which the final rotation gives up
        self._look_started = False         # final rotation already started for this loss
        self._costmap = None               # last Nav2 global costmap (OccupancyGrid)

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
        self.create_subscription(Odometry, TARGET_STATE_TOPIC, self._on_target_state, 10)
        self.create_subscription(Path, TARGET_PATH_TOPIC, self._on_target_path, 10)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL, history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(OccupancyGrid, COSTMAP_TOPIC, self._on_costmap, latched)
        self.create_timer(1.0 / BLIND_GOAL_RATE_HZ, self._blind_follow_loop)
        self.create_timer(0.05, self._rotation_loop)  # 20 Hz

        self.get_logger().info(
            f"nav2_bridge pronto: distanza={self.D:.2f} m (cammina oltre {self.D + self.enter_margin:.2f}, "
            f"si ferma sotto {self.D + self.exit_margin:.2f}), centraggio su "
            f"{'camera (ROI)' if self.center_on_camera else self.base_frame}, "
            f"rotazione su '{gp('cmd_vel_topic')}', "
            f"behavior_tree={self._behavior_tree or '(default del bt_navigator)'}, "
            f"segue la traiettoria di tracking_fsm, inseguimento alla cieca max {BLIND_FOLLOW_SEC:.1f} s.")

    # ------------------------------------------------------------------
    # Where the person is: the tracking_fsm trajectory (or the last target_3d)
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

    def _ros_now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _path_fresh(self, now=None):
        return self._path_time is not None and bool(self._path) and \
            (time.monotonic() if now is None else now) - self._path_time < PATH_TIMEOUT_SEC

    def _target_xy(self):
        """(x, y, frozen) of the person NOW: the trajectory point at the current ROS time, or the last
        target_3d without the trajectory (frozen). None if nothing is known."""
        if self._path_fresh():
            x, y, at_end = _interpolate(self._path, self._ros_now())
            return x, y, at_end or self._state_kind == 'frozen'
        if self._raw is not None:
            return self._raw[0], self._raw[1], True
        return None

    def _frame_ok(self, msg, what):
        if msg.header.frame_id == GLOBAL_FRAME:
            return True
        if not self._warned_frame:
            self._warned_frame = True
            self.get_logger().warn(f"{what} in '{msg.header.frame_id}', atteso '{GLOBAL_FRAME}': ignorato.")
        return False

    def _on_target_path(self, msg: Path):
        if not self._frame_ok(msg, "Traiettoria del target") or not msg.poses:
            return
        self._path = [(ps.header.stamp.sec + ps.header.stamp.nanosec * 1e-9,
                       ps.pose.position.x, ps.pose.position.y) for ps in msg.poses]
        self._path_time = time.monotonic()

    def _on_target_state(self, msg: Odometry):
        """State of the tracking_fsm filter: a "measured" message is a real measurement."""
        if not self._frame_ok(msg, "Stato del target"):
            return
        self._state_kind = msg.child_frame_id.split('/')[-1]
        if self._state_kind == 'measured':
            p = msg.pose.pose.position
            self._after_measurement(time.monotonic(), (p.x, p.y))

    def _on_target_pose(self, msg: TargetPose3D):
        """target_3d: only the camera frame (centring) while the trajectory arrives; without it, the
        person position (fallback, no prediction)."""
        self._camera_frame = msg.header.frame_id
        if self._path_fresh():
            return
        try:
            transform = self._lookup_at_stamp(GLOBAL_FRAME, msg.header.frame_id, msg.header.stamp)
        except tf2_ros.TransformException as ex:
            self.get_logger().warn(f"TF non disponibile ({msg.header.frame_id}->{GLOBAL_FRAME}): {ex}",
                                   throttle_duration_sec=2.0)
            return
        raw_pose = PoseStamped()
        raw_pose.header = msg.header
        raw_pose.pose.position = msg.position
        raw_pose.pose.orientation.w = 1.0
        world = do_transform_pose_stamped(raw_pose, transform)
        self._raw = (world.pose.position.x, world.pose.position.y)
        self._after_measurement(time.monotonic(), self._raw)

    def _after_measurement(self, now, seen):
        """A real measurement of the person: following resumes."""
        self._last_meas_time = now
        self._last_seen = seen
        self._frozen_announced = False
        if self._look_target is not None:
            self.get_logger().info("Target di nuovo misurato: fine rotazione finale.")
        self._look_target = None
        self._look_started = False
        if self._blind_announced:
            self.get_logger().info("Target di nuovo misurato: fine inseguimento alla cieca.")
            self._blind_announced = False

        target = self._target_xy()
        robot = self._robot_pose()
        if target is None or robot is None:
            return
        px, py, _ = target
        rx, ry, _ = robot
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

        fresh = (self._last_meas_time is not None
                 and time.monotonic() - self._last_meas_time < self.rot_timeout)
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

        err = self._centering_error()        # on the CURRENT point of the trajectory
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
        target = self._target_xy()
        if target is None:
            return None
        px, py, _ = target
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

    # ------------------------------------------------------------------
    # Blind following, costmap, final rotation
    # ------------------------------------------------------------------
    def _blind_follow_loop(self):
        """While no measurement arrives: follows the trajectory (moving, then its frozen end) for at
        most BLIND_FOLLOW_SEC, then stops every motion and starts the final rotation."""
        if self._last_meas_time is None:
            return
        now = time.monotonic()
        elapsed = now - self._last_meas_time
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

        target = self._target_xy()
        if target is None:
            return
        px, py, frozen = target
        if not self._blind_announced:
            self._blind_announced = True
            self.get_logger().info(f"Target non misurato: seguo la traiettoria prevista "
                                   f"(max {BLIND_FOLLOW_SEC:.1f} s).")
        if frozen and not self._frozen_announced:
            self._frozen_announced = True
            self.get_logger().info(f"Fine della traiettoria dopo {elapsed:.1f} s: vado verso il punto "
                                   f"({px:.2f}, {py:.2f}) fino a {BLIND_STANDOFF_M:.1f} m, "
                                   f"per altri {BLIND_FOLLOW_SEC - elapsed:.1f} s.")
        px, py = self._free_point(px, py)
        robot = self._robot_pose()
        if robot is None:
            return
        rx, ry, _ = robot
        d = math.hypot(px - rx, py - ry)
        heading = math.atan2(py - ry, px - rx)
        self.debug_target_pub.publish(self._pose(px, py, heading))
        standoff = self._blind_standoff(rx, ry, px, py, d, heading) if frozen else None
        self._decide(d, heading, px, py, predicted=True, standoff=standoff,
                     may_start=frozen and BLIND_START_NAV)

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

    def _on_costmap(self, msg: OccupancyGrid):
        self._costmap = msg

    def _blocked(self, x, y):
        """True if (x, y) is an obstacle (or its inflation) on the global costmap; unknown = free."""
        m = self._costmap
        if m is None or m.header.frame_id not in ('', GLOBAL_FRAME):
            return False
        info = m.info
        i = int((x - info.origin.position.x) / info.resolution)
        j = int((y - info.origin.position.y) / info.resolution)
        if not (0 <= i < info.width and 0 <= j < info.height):
            return False
        return m.data[j * info.width + i] >= COSTMAP_BLOCKED

    def _free_point(self, px, py):
        """The point itself if free, otherwise moved back towards the last seen position until free."""
        if self._last_seen is None or not self._blocked(px, py):
            return px, py
        lx, ly = self._last_seen
        n = int(math.hypot(px - lx, py - ly) / COSTMAP_STEP_M)
        for k in range(1, n + 1):
            f = 1.0 - k / max(n, 1)
            qx, qy = lx + f * (px - lx), ly + f * (py - ly)
            if not self._blocked(qx, qy):
                return qx, qy
        return lx, ly

    def _start_look(self, now):
        """Chooses the point to look at when blind following ends (see LOOK_AT)."""
        if LOOK_AT is None:
            return
        target = self._trajectory_look_point() if LOOK_AT == "trajectory" else self._last_seen
        robot = self._robot_pose()
        if target is None or robot is None:
            return
        if math.hypot(target[0] - robot[0], target[1] - robot[1]) < LOOK_MIN_DIST:
            return
        self._look_target = target
        self._look_until = now + LOOK_TIMEOUT_SEC
        self.get_logger().info(f"Rotazione finale verso ({target[0]:.2f}, {target[1]:.2f}), "
                               f"max {LOOK_TIMEOUT_SEC:.1f} s.")

    def _trajectory_look_point(self):
        """LOOK_DIST_M beyond the end of the last trajectory, along its final direction (after a
        curve: round the corner). Without a direction (person standing): the last seen position."""
        pts = self._path
        if not pts:
            return self._last_seen
        ex, ey = pts[-1][1], pts[-1][2]
        for _, x, y in reversed(pts[:-1]):
            if math.hypot(ex - x, ey - y) > 0.05:
                a = math.atan2(ey - y, ex - x)
                return ex + LOOK_DIST_M * math.cos(a), ey + LOOK_DIST_M * math.sin(a)
        return self._last_seen

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
