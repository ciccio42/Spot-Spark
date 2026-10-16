import os
import signal
import sys
import threading
import time
import traceback

import numpy as np
import yaml
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.signals import SignalHandlerOptions
from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from action_msgs.msg import GoalStatus
from ament_index_python.packages import get_package_share_directory, PackageNotFoundError

import tf2_ros
from tf2_geometry_msgs import do_transform_point
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, PoseStamped, Twist
from sensor_msgs.msg import CameraInfo, Image
from std_srvs.srv import Trigger
from nav2_msgs.action import NavigateToPose
from detection_service_msg.srv import Detection

# dependencies: yolo-detection service, navigation service, manipulation
from autonomous_manipulation_fsm.fsm import FSM, State
from autonomous_manipulation_fsm.utils import bbox_depth, deproject, standoff_goal, quat_from_yaw
from manipulation_interfaces.action import Manipulate, MoveArmToPose

class FSMNode(Node):

    def __init__(self):
        super().__init__("fsm_node")

        # Initialize the FSM with the initial state
        self.fsm = FSM(State.IDLE, on_transition=self.on_transition)

        # future of the async call (service/action) running in the current state, None if no call is pending
        self.pending = None
        # time the current state was entered, useful for timeouts
        self.state_start = self.get_clock().now()
        # False until the current state's handler has run once (used to log/act only on entry)
        self.state_entered = False

        # ---------- parameters ----------
        self.declare_parameter("rate", 10.0)
        self.declare_parameter("detection_service", "detection")
        self.declare_parameter("navigate_action", "navigate_to_pose")
        self.declare_parameter("manipulate_action", "manipulation_custom")
        self.declare_parameter("move_arm_action", "move_arm_to_pose")
        self.declare_parameter("open_gripper_service", "open_gripper")
        self.declare_parameter("open_gripper_timeout", 10.0)
        # arm poses recorded with manipulation_interfaces/save_arm_pose.py (hand in body)
        self.declare_parameter("arm_poses_file",
                               "/home/spot_ws/src/autonomous_manipulation_app/arm_poses.yaml")
        # poses of arm_poses_file commanded in NAVIGATE_TO_PLACE before opening the gripper
        self.declare_parameter("place_arm_sequence", ["safe", "post_pick_1", "post_pick_2",
                                                      "pre_place_1", "pre_place_2", "pre_open_gripper"])
        # time (s) the gripper stays open before closing, so the object can drop
        self.declare_parameter("place_release_delay", 1.0)
        # spot_driver services used to sit the robot at the end of a cycle and stand it up for the next
        self.declare_parameter("sit_service", "sit")
        self.declare_parameter("stand_service", "stand")
        self.declare_parameter("posture_timeout", 15.0)
        # back at the initial position Nav2 does not enforce the orientation (yaw_goal_tolerance 3.14):
        # the FSM turns the robot in place to the start yaw with cmd_vel (no Nav2 collision check)
        self.declare_parameter("align_home_yaw", True)
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("align_yaw_tolerance", 0.087)    # rad (~5 deg)
        self.declare_parameter("align_gain", 1.0)               # rad/s per rad of error
        self.declare_parameter("align_max_angular_vel", 0.5)    # rad/s
        self.declare_parameter("align_min_angular_vel", 0.15)   # rad/s, below this Spot barely turns
        self.declare_parameter("align_timeout", 20.0)           # s
        # before the return navigation, turn in place to face the initial position, so Nav2 starts
        # straight instead of making a large curve; skipped when closer than face_goal_min_distance (m)
        self.declare_parameter("face_goal_before_navigation", True)
        self.declare_parameter("face_goal_min_distance", 0.5)
        # optional poses of arm_poses_file commanded after the release, before arm_stow
        # (e.g. to bring the hand out of the place area first); empty = stow directly
        self.declare_parameter("post_place_arm_sequence", [""])
        self.declare_parameter("close_gripper_service", "close_gripper")
        # Nav2 global frame (odom in spot_motion/config/nav2_params_spot_real.yaml) and robot base frame
        self.declare_parameter("global_frame", "odom")
        self.declare_parameter("base_frame", "body")
        # hand camera intrinsics and depth registered to the hand color camera (same pixels as the detection)
        self.declare_parameter("camera_info_topic", "/camera/hand/camera_info")
        self.declare_parameter("depth_topic", "/depth_registered/hand/image")
        # distance (m) between the robot body and the ball at the end of the navigation
        self.declare_parameter("standoff_distance", 0.8)
        # below this extra distance (m) from the standoff the robot does not navigate (Nav2 xy_goal_tolerance is 0.30)
        self.declare_parameter("min_nav_distance", 0.3)
        # seconds without a valid ball detection before going to ERROR
        self.declare_parameter("detection_timeout", 15.0)
        self.declare_parameter("navigation_timeout", 60.0)
        # BT used for the goal; empty = Nav2 default BT that ends with SUCCESS when the goal is reached
        # (the spot_motion default follow_point_spot.xml never ends, it is meant for person following)
        self.declare_parameter("behavior_tree", "")
        # read Enter from the terminal to start (the /start service always works too)
        self.declare_parameter("use_stdin", True)
        # Trigger service called on shutdown (Ctrl+C / SIGTERM) from any state to reset the arm
        self.declare_parameter("arm_reset_service", "arm_stow")
        # spot_driver lease claim at startup (needed with auto_claim: False in spot.yaml)
        self.declare_parameter("claim_service", "claim")
        self.declare_parameter("claim_timeout", 10.0)
        # claim the lease again before every arm move (MoveArmToPose) and every navigation
        self.declare_parameter("claim_before_motion", True)
        # Trigger service called before every navigation to stow the arm (walking with the arm out is unsafe:
        # the Nav2 footprint does not include it)
        self.declare_parameter("arm_stow_service", "arm_stow")
        self.declare_parameter("arm_stow_timeout", 10.0)
        # max time (s) to wait for the arm reset on shutdown
        self.declare_parameter("shutdown_timeout", 7.0)
        

        gp = lambda name: self.get_parameter(name).value
        self.global_frame = gp("global_frame")
        self.base_frame = gp("base_frame")
        self.standoff_distance = float(gp("standoff_distance"))
        self.min_nav_distance = float(gp("min_nav_distance"))
        self.detection_timeout = float(gp("detection_timeout"))
        self.navigation_timeout = float(gp("navigation_timeout"))
        self.behavior_tree = gp("behavior_tree") or self.default_behavior_tree()
        self.use_stdin = bool(gp("use_stdin"))
        self.shutdown_timeout = float(gp("shutdown_timeout"))
        # arm poses for the detection, nested dicts {"position": {x,y,z}, "orientation": {x,y,z,w}, "duration"}
        self.far_detection_pose = self.declare_pose_parameter("far_detection_pose")
        self.open_gripper_service = gp("open_gripper_service")
        self.open_gripper_timeout = float(gp("open_gripper_timeout"))
        self.arm_poses_file = gp("arm_poses_file")
        self.place_arm_sequence = list(gp("place_arm_sequence"))
        self.place_release_delay = float(gp("place_release_delay"))
        self.sit_service = gp("sit_service")
        self.stand_service = gp("stand_service")
        self.posture_timeout = float(gp("posture_timeout"))
        self.align_home_yaw = bool(gp("align_home_yaw"))
        self.align_yaw_tolerance = float(gp("align_yaw_tolerance"))
        self.align_gain = float(gp("align_gain"))
        self.align_max_angular_vel = float(gp("align_max_angular_vel"))
        self.align_min_angular_vel = float(gp("align_min_angular_vel"))
        self.align_timeout = float(gp("align_timeout"))
        self.face_goal_before_navigation = bool(gp("face_goal_before_navigation"))
        self.face_goal_min_distance = float(gp("face_goal_min_distance"))
        self.cmd_vel_pub = self.create_publisher(Twist, gp("cmd_vel_topic"), 10)
        # [""] is the "empty list" default (ROS 2 cannot infer the type of an empty list)
        self.post_place_arm_sequence = [name for name in gp("post_place_arm_sequence") if name]
        self.near_detection_pose = self.declare_pose_parameter("near_detection_pose")
        self.close_gripper_service = gp("close_gripper_service")
        self.close_gripper_timeout = float(gp("detection_timeout"))
        # ---------- dependencies ----------
        # take the robot lease before anything else: every arm / gripper / posture command needs it
        self.claim_lease(gp("claim_service"), float(gp("claim_timeout")))
        # same service, called from the FSM tick before each motion: own callback group (blocking call)
        self.claim_service = gp("claim_service")
        self.claim_timeout = float(gp("claim_timeout"))
        self.claim_before_motion = bool(gp("claim_before_motion"))
        self.claim_client = self.create_client(Trigger, self.claim_service,
                                               callback_group=ReentrantCallbackGroup())

        # arm reset on shutdown (spot_driver); created first so it is available even if the node is
        # killed while waiting for the other servers below
        self.arm_reset_service = gp("arm_reset_service")
        # own callback group: on shutdown the FSM thread may still hold the default group
        self.arm_reset_client = self.create_client(Trigger, self.arm_reset_service,
                                                   callback_group=ReentrantCallbackGroup())
        # arm stow before the navigation, called from the FSM tick: own callback group, see wait_future()
        self.arm_stow_service = gp("arm_stow_service")
        self.arm_stow_timeout = float(gp("arm_stow_timeout"))
        self.arm_stow_client = self.create_client(Trigger, self.arm_stow_service,
                                                  callback_group=ReentrantCallbackGroup())
        # sit / stand (spot_driver), called from the FSM tick: own callback group
        self.sit_client = self.create_client(Trigger, self.sit_service,
                                             callback_group=ReentrantCallbackGroup())
        self.stand_client = self.create_client(Trigger, self.stand_service,
                                               callback_group=ReentrantCallbackGroup())

        # YOLO detection service (yolo_detector/detector_node)
        self.get_logger().info(f"[{self.get_name()}] Waiting for detection service '{gp('detection_service')}'...")
        self.detection_client = self.create_client(Detection, gp("detection_service"))
        while not self.detection_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service {}, waiting again...'.format(gp("detection_service")))
        
        # Nav2 (spot_motion/navigation_launch.py)
        self.nav_goal_handle = None
        # waiting for the action server is up
        self.nav_action = gp("navigate_action")
        # own callback group: navigate_to() blocks the FSM tick while waiting, the responses are
        # processed by another thread of the MultiThreadedExecutor
        self.nav_client = ActionClient(self, NavigateToPose, self.nav_action,
                                       callback_group=ReentrantCallbackGroup())
        while not self.nav_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().info('action server {}, waiting again...'.format(self.nav_action))

        # TF, camera intrinsics and depth, used to turn the detected pixel into a 3D point
        self.tf_buffer = tf2_ros.Buffer()
        # the listener spins its own thread: it gets a dedicated node, so that thread never touches the
        # FSM node (two executors spinning the same node break its action clients' wait sets)
        self.tf_node = rclpy.create_node(f"{self.get_name()}_tf_listener")
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self.tf_node, spin_thread=True)
        # wait for a valid tf_listener to be available (e.g. odom->body) before starting the FSM
        self.get_logger().info(f"[{self.get_name()}] Waiting for TF {self.global_frame} -> {self.base_frame}...")
        while True:
            try:
                self.tf_buffer.lookup_transform(self.global_frame, self.base_frame, Time(),
                                                timeout=Duration(seconds=1.0))
                break
            except tf2_ros.TransformException:
                self.get_logger().info(f"[{self.get_name()}] Waiting for TF {self.global_frame} -> {self.base_frame}...")
       
        # Wait for Manipulate action server
        self.get_logger().info(f"[{self.get_name()}] Waiting for Manipulate action server...")
        # own callback group: send_manipulate_goal() waits on it with the blocking send_goal()
        self.manipulate_client = ActionClient(self, Manipulate, gp("manipulate_action"),
                                              callback_group=ReentrantCallbackGroup())
        while not self.manipulate_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().info(f"[{self.get_name()}] Waiting for Manipulate action server...")  
            
        # Wait for MoveArmToPose action server
        self.get_logger().info(f"[{self.get_name()}] Waiting for MoveArmToPose action server...")
        # own callback group: on_detect waits on it with the blocking send_goal(), its responses are
        # processed by another thread of the MultiThreadedExecutor (see main)
        self.move_arm_client = ActionClient(self, MoveArmToPose, gp("move_arm_action"),
                                            callback_group=ReentrantCallbackGroup())
        while not self.move_arm_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().info(f"[{self.get_name()}] Waiting for MoveArmToPose action server...")
            
        # blocking clients (.call() inside the FSM tick) need their own callback group, otherwise the
        # response waits behind the timer in the default group and .call() never returns
        self.get_logger().info(f"[{self.get_name()}] Waiting for OpenGripper service server...")
        self.open_gripper_client = self.create_client(Trigger, gp("open_gripper_service"),
                                                      callback_group=ReentrantCallbackGroup())
        while not self.open_gripper_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f"[{self.get_name()}] Waiting for OpenGripper action server...")
        
        self.get_logger().info(f"[{self.get_name()}] Waiting for CloseGripper service server...")
        self.close_gripper_client = self.create_client(Trigger, gp("close_gripper_service"),
                                                       callback_group=ReentrantCallbackGroup())
        while not self.close_gripper_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f"[{self.get_name()}] Waiting for CloseGripper action server...")
            
                
        self.bridge = CvBridge()
        self.camera_info = None
        self.depth_msg = None
        self.create_subscription(CameraInfo, gp("camera_info_topic"), self.camera_info_callback, 1)
        self.create_subscription(Image, gp("depth_topic"), self.depth_callback, 1)

        # ---------- FSM data ----------
        self.ball_point = None      # PointStamped of the ball in global_frame
        # robot pose (x, y, yaw) in global_frame when the cycle was started from IDLE
        self.home_pose = None
        # set when the object has been placed: IDLE brings the robot back to home_pose and sits it
        self.return_home_pending = False
        # True after the robot was sat down by the FSM: it is stood up before the next cycle
        self.robot_sitting = False
        # set on shutdown: makes the blocking waits (wait_future) return
        self.stopping = False
        # DETECT runs in two modes:
        #   "far"  (from IDLE): arm to far_detection_pose, 3D ball position for the navigation
        #   "near" (from PICK): arm to near_detection_pose, bbox center (pixels) for the pick, back to PICK
        self.detection_mode = "far"
        # result of the near detection: {"frame_id", "x_center", "y_center", "stamp"}, None until available
        self.pick_target = None
        self.start_requested = threading.Event()
        # confirmation (Enter / start service) awaited by wait_for_confirmation() inside a state
        self.confirm_requested = threading.Event()
        self.awaiting_confirmation = False
        self.stdin_thread = None
        # start trigger alternative to Enter (needed under ros2 launch, where stdin is not available):
        #   ros2 service call /fsm_node/start std_srvs/srv/Trigger
        self.create_service(Trigger, "~/start", self.start_callback)

        # one handler per state; handlers must not block, they are called at every timer tick
        self.handlers = {
            State.IDLE: self.on_idle,
            State.DETECT: self.on_detect,
            State.NAVIGATE_TO_PICK: self.on_navigate_to_pick,
            State.PICK: self.on_pick,
            State.NAVIGATE_TO_PLACE: self.on_navigate_to_place,
            State.PLACE: self.on_place,
            State.RETURN_TO_IDLE: self.on_return_to_idle,
            State.ERROR: self.on_error,
        }

        # the timer also wakes the executor regularly, so Ctrl+C is handled within one tick
        rate = float(gp("rate"))
        self.timer = self.create_timer(1.0 / rate, self.step)

        self.get_logger().info(f"[{self.get_name()}] FSM started in state {self.fsm.state.name}")
        self.get_logger().info(f"[{self.get_name()}] Nav2 behavior tree: {self.behavior_tree or '<bt_navigator default>'}")

    def claim_lease(self, service, timeout):
        """Blocking, in the constructor: calls the spot_driver claim service (std_srvs/Trigger).
        Raises RuntimeError if the lease cannot be taken (e.g. held by the tablet).
        The executor is not spinning yet: the call goes through a temporary node with its own
        executor, so the FSM node is never attached to a second executor."""
        node = rclpy.create_node(f"{self.get_name()}_claim")
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        try:
            client = node.create_client(Trigger, service)
            self.get_logger().info(f"[{self.get_name()}] Waiting for the '{service}' service...")
            while not client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info(f"[{self.get_name()}] Waiting for the '{service}' service...")
            future = client.call_async(Trigger.Request())
            executor.spin_until_future_complete(future, timeout_sec=timeout)
            if not future.done():
                raise RuntimeError(f"'{service}' did not answer in {timeout:.0f} s")
            res = future.result()
            if not res.success:
                raise RuntimeError(f"Cannot claim the lease: {res.message}")
            self.get_logger().info(f"[{self.get_name()}] Lease claimed: {res.message}")
        finally:
            executor.remove_node(node)
            executor.shutdown()
            node.destroy_node()

    def declare_pose_parameter(self, name, duration=3.0):
        # ROS 2 parameters cannot be dicts: the nested YAML block is loaded as one double parameter
        # per field (e.g. far_detection_pose.position.x), here they are declared and put back in a dict
        fields = {
            "position": {"x": 0.0, "y": 0.0, "z": 0.0},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        }
        pose = {}
        for group, defaults in fields.items():
            pose[group] = {
                key: float(self.declare_parameter(f"{name}.{group}.{key}", default).value)
                for key, default in defaults.items()
            }
        pose["duration"] = float(self.declare_parameter(f"{name}.duration", duration).value)
        return pose

    def default_behavior_tree(self):
        try:
            return (get_package_share_directory("nav2_bt_navigator")
                    + "/behavior_trees/navigate_to_pose_w_replanning_and_recovery.xml")
        except PackageNotFoundError:
            self.get_logger().warn(f"[{self.get_name()}] nav2_bt_navigator not found, using the bt_navigator default BT")
            return ""

    # ---------- callbacks ----------

    def camera_info_callback(self, msg):
        self.camera_info = msg

    def depth_callback(self, msg):
        self.depth_msg = msg

    def wait_for_enter(self):
        # the keyboard is read in a separate thread so the executor is never blocked.
        # /dev/tty is the terminal the node was started from: it works with ros2 run and with
        # ros2 launch, which does not forward its stdin to the nodes
        try:
            terminal = open("/dev/tty")
        except OSError:
            terminal = sys.stdin
        while True:
            line = terminal.readline()
            if line == "":
                self.get_logger().warn(f"[{self.get_name()}] No terminal available, start with: ros2 service call "
                                       f"{self.get_fully_qualified_name()}/start std_srvs/srv/Trigger")
                return
            if line.strip() != "":
                continue
            if self.awaiting_confirmation:
                self.confirm_requested.set()
            elif self.fsm.state == State.IDLE:
                self.start_requested.set()

    def start_callback(self, request, response):
        # same meaning as Enter: confirms the step waiting in wait_for_confirmation(), or starts from IDLE
        if self.awaiting_confirmation:
            self.confirm_requested.set()
            response.success = True
            response.message = f"Confirmed in state {self.fsm.state.name}"
            return response
        if self.fsm.state != State.IDLE:
            response.success = False
            response.message = f"FSM is in state {self.fsm.state.name}, not IDLE"
            return response
        self.start_requested.set()
        response.success = True
        response.message = "Starting detection"
        return response

    def on_transition(self, old_state, new_state):
        self.get_logger().info(f"[{self.get_name()}] {old_state.name} -> {new_state.name}")
        # every state starts without pending calls
        self.pending = None
        self.state_start = self.get_clock().now()
        self.state_entered = False

    def step(self):
        try:
            entering = not self.state_entered
            self.state_entered = True
            self.handlers[self.fsm.state](entering)
        except Exception as e:
            self.get_logger().error(f"[{self.get_name()}] Error in state {self.fsm.state.name}: {e}")
            self.fsm.force(State.ERROR)

    def time_in_state(self):
        return (self.get_clock().now() - self.state_start).nanoseconds * 1e-9

    # ---------- state handlers ----------

    def on_idle(self, entering):
        # end of a cycle (object placed): back to the initial position, then sit
        if entering and self.return_home_pending:
            self.return_home_pending = False
            self.get_logger().info(f"[{self.get_name()}] Returning to the initial position and sitting the robot")  
            if not self.return_home_and_sit():
                return

        if entering:
            self.get_logger().info(f"[{self.get_name()}] In IDLE state, waiting for a command to start the process.")
            self.start_requested.clear()
            start_service = f"ros2 service call {self.get_fully_qualified_name()}/start std_srvs/srv/Trigger"
            if self.use_stdin:
                self.get_logger().info(f"[{self.get_name()}] Press Enter to start detection (or: {start_service})")
                # a single keyboard reader for the whole node, it keeps running across IDLE states
                if self.stdin_thread is None:
                    self.stdin_thread = threading.Thread(target=self.wait_for_enter, daemon=True)
                    self.stdin_thread.start()
            else:
                self.get_logger().info(f"[{self.get_name()}] Start with: {start_service}")

        # press enter to move to detection state
        if self.start_requested.is_set():
            self.start_requested.clear()
            if self.robot_sitting:
                self.get_logger().info(f"[{self.get_name()}] Standing up")
                self.call_trigger(self.stand_client, self.stand_service, self.posture_timeout)
                self.robot_sitting = False
            # the robot comes back here at the end of the cycle
            self.home_pose = self.robot_pose()
            self.get_logger().info(f"[{self.get_name()}] Initial position saved: ({self.home_pose[0]:.2f}, "
                                   f"{self.home_pose[1]:.2f}) yaw={np.degrees(self.home_pose[2]):.0f} deg "
                                   f"in {self.global_frame}")
            self.detection_mode = "far"
            self.pick_target = None
            self.fsm.transition(State.DETECT)

    def on_detect(self, entering):
        # Move the robot arm to the detection position of the current mode, once when entering the state
        if entering:
            self.get_logger().info(f"[{self.get_name()}] In DETECT state ({self.detection_mode}), "
                                   f"detecting objects to pick.")
            pose = self.near_detection_pose if self.detection_mode == "near" else self.far_detection_pose
            self.move_arm_to_detection_pose(pose, open_gripper=True, close_gripper=False)
            return

        # Start Inference
        if self.time_in_state() > self.detection_timeout:
            self.get_logger().error(f"[{self.get_name()}] No ball located in {self.detection_timeout:.0f} s")
            self.fsm.transition(State.ERROR)
            return

        # Call detection service - ask a new detection
        if self.pending is None:
            if not self.detection_client.service_is_ready():
                self.get_logger().info(f"[{self.get_name()}] Waiting for the detection service...",
                                       throttle_duration_sec=2.0)
                return
            self.pending = self.detection_client.call_async(Detection.Request())
            return
        if not self.pending.done():
            return
        res = self.pending.result()
        self.pending = None

        if not res.success:
            self.get_logger().info(f"[{self.get_name()}] Detection failed: {res.message}", throttle_duration_sec=1.0)
            return

        # the detector answers with its last processed frame: discard frames captured before the arm
        # reached the detection pose (state_start is reset when the arm is in position)
        if Time.from_msg(res.stamp) < self.state_start:
            self.get_logger().info(f"[{self.get_name()}] Detection from an image taken before the arm was in " f"position, waiting for a new frame", throttle_duration_sec=1.0)
            return

        self.get_logger().info(f"[{self.get_name()}] Detected {res.class_name} conf={res.confidence:.2f} " f"center=({res.center_x:.1f}, {res.center_y:.1f}) frame={res.frame_id}")

        # near detection: the pick only needs the bbox center in pixels, go back to PICK
        if self.detection_mode == "near":
            target = self.pixel_target(res)
            if target is None:
                return
            self.pick_target = target
            self.get_logger().info(f"[{self.get_name()}] Pick target: center=({target['x_center']:.1f}, "
                                   f"{target['y_center']:.1f}) px in {target['frame_id']}")
            self.fsm.transition(State.PICK)
            return

        # far detection: 3D position of the ball for the navigation
        ball = self.locate_ball(res)
        if ball is None:
            # depth/intrinsics/TF not available yet: retry with a new detection at the next tick
            return
        self.ball_point = ball
        self.get_logger().info(f"[{self.get_name()}] Ball in {self.global_frame}: "
                               f"({ball.point.x:.2f}, {ball.point.y:.2f}, {ball.point.z:.2f})")

        # If detection is successful, transition to NAVIGATE_TO_PICK
        self.fsm.transition(State.NAVIGATE_TO_PICK)

    def place_object(self, poses_file):
        """Blocking: moves the arm through place_arm_sequence (poses read from poses_file) with
        MoveArmToPose, opens the gripper, waits place_release_delay, closes the gripper, goes through
        post_place_arm_sequence and stows the arm. Returns True on success; on failure goes to ERROR."""
        try:
            poses = self.load_arm_poses(poses_file)
            missing = [name for name in self.place_arm_sequence + self.post_place_arm_sequence
                       if name not in poses]
            if missing:
                raise ValueError(f"poses {missing} not found in {poses_file} (available: {list(poses)})")
        except (OSError, yaml.YAMLError, ValueError) as e:
            self.get_logger().error(f"[{self.get_name()}] Cannot load the arm poses: {e}")
            self.fsm.transition(State.ERROR)
            return False

        # the gripper is not touched while moving: it is holding the object
        state = self.fsm.state
        for i, name in enumerate(self.place_arm_sequence, 1):
            self.get_logger().info(f"[{self.get_name()}] Arm pose {i}/{len(self.place_arm_sequence)}: {name}")
            self.move_arm_to_detection_pose(poses[name], open_gripper=False, close_gripper=False)
            if self.fsm.state != state:
                # the arm move failed and the FSM went to ERROR
                return False

        try:
            # release the object
            self.call_trigger(self.open_gripper_client, self.open_gripper_service, self.open_gripper_timeout)
            time.sleep(self.place_release_delay)
            # close the gripper again
            self.call_trigger(self.close_gripper_client, self.close_gripper_service, self.close_gripper_timeout)
        except RuntimeError as e:
            self.get_logger().error(f"[{self.get_name()}] {e}")
            self.fsm.transition(State.ERROR)
            return False

        # optional poses out of the place area, then back to stow
        for i, name in enumerate(self.post_place_arm_sequence, 1):
            self.get_logger().info(f"[{self.get_name()}] Post-place arm pose {i}/"
                                   f"{len(self.post_place_arm_sequence)}: {name}")
            self.move_arm_to_detection_pose(poses[name], open_gripper=False, close_gripper=False)
            if self.fsm.state != state:
                return False
        try:
            self.stow_arm()
        except RuntimeError as e:
            self.get_logger().error(f"[{self.get_name()}] {e}")
            self.fsm.transition(State.ERROR)
            return False
        return True

    @staticmethod
    def load_arm_poses(poses_file):
        """arm_poses.yaml -> {name: {"position": {x,y,z}, "orientation": {x,y,z,w}, "duration"}}."""
        with open(os.path.expanduser(poses_file)) as f:
            data = yaml.safe_load(f) or {}
        poses = {}
        for name, pose in data.items():
            try:
                poses[name] = {
                    "position": {k: float(pose["position"][k]) for k in ("x", "y", "z")},
                    "orientation": {k: float(pose["orientation"][k]) for k in ("x", "y", "z", "w")},
                    # optional per-pose duration (s)
                    "duration": float(pose.get("duration", 3.0)),
                }
            except (KeyError, TypeError, ValueError) as e:
                raise ValueError(f"pose '{name}' in {poses_file} is malformed: {e}") from e
        return poses

    def move_arm_to_detection_pose(self, detection_pose, open_gripper=False, close_gripper=False):
        """Blocking: moves the arm to far_detection_pose, goes to ERROR on failure."""
        arm_pose_goal = MoveArmToPose.Goal()
        # 
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = self.base_frame
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        pose_stamped.pose.position.x = detection_pose.get("position").get("x", 0.0)
        pose_stamped.pose.position.y = detection_pose.get("position").get("y", 0.0)
        pose_stamped.pose.position.z = detection_pose.get("position").get("z", 0.0)
        pose_stamped.pose.orientation.x = detection_pose.get("orientation").get("x", 0.0)
        pose_stamped.pose.orientation.y = detection_pose.get("orientation").get("y", 0.0)
        pose_stamped.pose.orientation.z = detection_pose.get("orientation").get("z", 0.0)
        pose_stamped.pose.orientation.w = detection_pose.get("orientation").get("w", 1.0)
        arm_pose_goal.pose_stamped = pose_stamped                         
        arm_pose_goal.duration = detection_pose.get("duration", 3.0)
        if not self.move_arm_client.server_is_ready():
            self.get_logger().error(f"[{self.get_name()}] MoveArmToPose action server not available")
            self.fsm.transition(State.ERROR)
            return
        try:
            self.reclaim_lease()
        except RuntimeError as e:
            self.get_logger().error(f"[{self.get_name()}] {e}")
            self.fsm.transition(State.ERROR)
            return
        # send the goal to the MoveArmToPose action server and wait for the result.
        # send_goal() blocks until the result arrives and returns the GetResult response (.status, .result);
        # it works only because the response is processed by another executor thread
        self.get_logger().info(f"[{self.get_name()}] Sending MoveArmToPose goal to move arm to detection position")
        response = self.move_arm_client.send_goal(arm_pose_goal)
        if response is None or response.status != GoalStatus.STATUS_SUCCEEDED or not response.result.success:
            message = response.result.message if response is not None else "no response"
            status = response.status if response is not None else "-"
            self.get_logger().error(f"[{self.get_name()}] MoveArmToPose action failed (status {status}): {message}")
            self.fsm.transition(State.ERROR)
            return
        self.get_logger().info(f"[{self.get_name()}] MoveArmToPose action succeeded: {response.result.message}")
        
        if open_gripper:
            # call the open_gripper service to open the gripper
            if not self.open_gripper_client.service_is_ready():
                self.get_logger().error(f"[{self.get_name()}] OpenGripper service not available")
                self.fsm.transition(State.ERROR)
                return
            self.get_logger().info(f"[{self.get_name()}] Calling OpenGripper service to open the gripper")
            open_response = self.open_gripper_client.call(Trigger.Request())
            if not open_response.success:
                self.get_logger().error(f"[{self.get_name()}] OpenGripper service failed: {open_response.message}")
                self.fsm.transition(State.ERROR)
                return
            self.get_logger().info(f"[{self.get_name()}] OpenGripper service succeeded: {open_response.message}")
        if close_gripper:
            # call the close_gripper service to close the gripper
            if not self.close_gripper_client.service_is_ready():
                self.get_logger().error(f"[{self.get_name()}] CloseGripper service not available")
                self.fsm.transition(State.ERROR)
                return
            self.get_logger().info(f"[{self.get_name()}] Calling CloseGripper service to close the gripper")
            close_response = self.close_gripper_client.call(Trigger.Request())
            if not close_response.success:
                self.get_logger().error(f"[{self.get_name()}] CloseGripper service failed: {close_response.message}")
                self.fsm.transition(State.ERROR)
                return
            self.get_logger().info(f"[{self.get_name()}] CloseGripper service succeeded: {close_response.message}")
        
        # the detection timeout starts after the arm is in position
        self.state_start = self.get_clock().now()

    def on_navigate_to_pick(self, entering):
        # Move the robot to the pick location; blocking: returns only when the motion is over
        self.get_logger().info(f"[{self.get_name()}] In NAVIGATE_TO_PICK state, navigating to the pick location.")
        robot = self.robot_xy()
        gx, gy, yaw, d = standoff_goal(robot, (self.ball_point.point.x, self.ball_point.point.y),
                                       self.standoff_distance)
        if d - self.standoff_distance < self.min_nav_distance:
            self.get_logger().info(f"[{self.get_name()}] Ball at {d:.2f} m, already close enough: no navigation")
            self.fsm.transition(State.PICK)
            return
        if not self.nav_client.server_is_ready():
            raise RuntimeError(f"Nav2 action server '{self.nav_action}' not available")

        self.get_logger().info(f"[{self.get_name()}] Ball at {d:.2f} m, goal ({gx:.2f}, {gy:.2f}) "
                               f"yaw={np.degrees(yaw):.0f} deg in {self.global_frame}")
        status = self.navigate_to(gx, gy, yaw)

        # Ask for a new detection to update the pick location

        # If navigation is successful, transition to PICK
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info(f"[{self.get_name()}] Navigation succeeded")
            self.fsm.transition(State.PICK)
        else:
            raise RuntimeError(f"Navigation ended with status {status}")

    def on_pick(self, entering):
        # first time in PICK (coming from the navigation): new detection from near_detection_pose,
        # DETECT comes back here with self.pick_target
        if self.detection_mode != "near" or self.pick_target is None:
            self.get_logger().info(f"[{self.get_name()}] In PICK state, new detection from the near pose.")
            self.detection_mode = "near"
            self.pick_target = None
            self.fsm.transition(State.DETECT)
            return

        if entering:
            t = self.pick_target
            self.get_logger().info(f"[{self.get_name()}] In PICK state, picking the object at "
                                   f"({t['x_center']:.1f}, {t['y_center']:.1f}) px in {t['frame_id']}")

        
        # Perform the pick operation with self.build_pick_goal() (Manipulate action)
        if not self.send_manipulate_goal(self.build_pick_goal()):
            return

         

        # after the pick move the robot arm to save-position


        # If pick is successful, transition to NAVIGATE_TO_PLACE
        self.fsm.transition(State.NAVIGATE_TO_PLACE)

    def on_navigate_to_place(self, entering):
        # Move the arm holding the object to the safe position (far_detection_pose), once when entering
        if entering:
            self.get_logger().info(f"[{self.get_name()}] In NAVIGATE_TO_PLACE state, moving the arm to the "
                                   f"safe position.")
            if not self.wait_for_confirmation("move the arm to the safe position and place the object"):
                return
            
            # open the gripper to release the object, then move the arm to the safe position
            # self.call_trigger(self.open_gripper_client, self.open_gripper_service, self.open_gripper_timeout)
            # self.stow_arm()
            
            # safe -> ... -> pre_open_gripper, then open the gripper
            if not self.place_object(self.arm_poses_file):
                return
            self.get_logger().info(f"[{self.get_name()}] Object placed")
            # cycle completed: IDLE brings the robot back to the initial position and sits it
            self.return_home_pending = True
            self.fsm.transition(State.IDLE)
            return

    def on_place(self, entering):
        pass

    def on_return_to_idle(self, entering):
        pass

    def on_error(self, entering):
        if entering:
            self.get_logger().error(f"[{self.get_name()}] In ERROR state, stopping and going back to IDLE.")
            self.cancel_navigation()
        self.fsm.transition(State.IDLE)

    # ---------- helpers ----------

    def locate_ball(self, res):
        """Detection response -> PointStamped of the ball in global_frame, None if data is missing."""
        if self.camera_info is None or self.depth_msg is None:
            self.get_logger().warn(f"[{self.get_name()}] No camera_info/depth received yet", throttle_duration_sec=2.0)
            return None

        info = self.camera_info
        fx, fy, cx, cy = info.k[0], info.k[4], info.k[2], info.k[5]
        x1, y1, x2, y2 = res.x1, res.y1, res.x2, res.y2
        if res.normalized:
            x1, x2 = x1 * info.width, x2 * info.width
            y1, y2 = y1 * info.height, y2 * info.height

        depth = self.bridge.imgmsg_to_cv2(self.depth_msg, desired_encoding="passthrough")
        depth_m = depth.astype(np.float32) / 1000.0 if depth.dtype == np.uint16 else depth.astype(np.float32)
        # depth and color image may have different resolutions: scale the bbox to the depth image
        sx, sy = depth_m.shape[1] / info.width, depth_m.shape[0] / info.height
        z = bbox_depth(depth_m, x1 * sx, y1 * sy, x2 * sx, y2 * sy)
        if z is None:
            self.get_logger().warn(f"[{self.get_name()}] No valid depth inside the bbox")
            return None

        u, v = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        px, py, pz = deproject(u, v, z, fx, fy, cx, cy)

        point = PointStamped()
        point.header.frame_id = res.frame_id or info.header.frame_id
        point.point.x, point.point.y, point.point.z = px, py, pz
        try:
            tf = self.tf_buffer.lookup_transform(self.global_frame, point.header.frame_id, Time(),
                                                 timeout=Duration(seconds=0.0))
        except tf2_ros.TransformException as e:
            self.get_logger().warn(f"[{self.get_name()}] TF {point.header.frame_id} -> {self.global_frame}: {e}")
            return None
        out = do_transform_point(point, tf)
        out.header.frame_id = self.global_frame
        return out

    def pixel_target(self, res):
        """Detection response -> bbox center in pixels of the detection image, None if not computable."""
        x, y = res.center_x, res.center_y
        # if res.normalized:
        #     if self.camera_info is None:
        #         self.get_logger().warn(f"[{self.get_name()}] Normalized detection but no camera_info yet",
        #                                throttle_duration_sec=2.0)
        #         return None
        #     x, y = x * self.camera_info.width, y * self.camera_info.height
        return {"frame_id": res.frame_id, "x_center": float(x), "y_center": float(y), "stamp": res.stamp}

    def send_manipulate_goal(self, goal):
        """Blocking: asks for confirmation, then sends the Manipulate goal and waits for the result.
        Returns True on success; on failure goes to ERROR and returns False."""
        # the arm must NOT move between the near detection and the pick: the pixel in the goal refers to
        # the camera pose at the image time (manipulation_node builds the frame tree at goal.timestamp),
        # and the grasp opens/closes the gripper by itself
        if not self.manipulate_client.server_is_ready():
            self.get_logger().error(f"[{self.get_name()}] Manipulate action server not available")
            self.fsm.transition(State.ERROR)
            return False

        # wait for Enter (or the start service) before starting the manipulation
        if not self.wait_for_confirmation("start picking"):
            return False

        self.get_logger().info(f"[{self.get_name()}] Sending Manipulate goal to pick the object at "
                               f"({goal.x_center:.1f}, {goal.y_center:.1f}) px in {goal.frame_id}")
        response = self.manipulate_client.send_goal(goal)
        if response is None or response.status != GoalStatus.STATUS_SUCCEEDED or not response.result.success:
            message = response.result.message if response is not None else "no response"
            status = response.status if response is not None else "-"
            self.get_logger().error(f"[{self.get_name()}] Manipulate action failed (status {status}): {message}")
            self.fsm.transition(State.ERROR)
            return False
        self.get_logger().info(f"[{self.get_name()}] Manipulate action succeeded: {response.result.message}")
        return True

    def wait_for_confirmation(self, action):
        """Blocks the FSM tick until Enter is pressed or the start service is called.
        Returns False if the node is shutting down."""
        start_service = f"ros2 service call {self.get_fully_qualified_name()}/start std_srvs/srv/Trigger"
        self.confirm_requested.clear()
        self.awaiting_confirmation = True
        if self.use_stdin:
            self.get_logger().info(f"[{self.get_name()}] Press Enter to {action} (or: {start_service})")
            if self.stdin_thread is None:
                self.stdin_thread = threading.Thread(target=self.wait_for_enter, daemon=True)
                self.stdin_thread.start()
        else:
            self.get_logger().info(f"[{self.get_name()}] To {action}: {start_service}")
        try:
            while not self.confirm_requested.wait(timeout=0.1):
                if self.stopping:
                    return False
            return True
        finally:
            self.awaiting_confirmation = False

    def build_pick_goal(self):
        """Manipulate goal from the near detection (self.pick_target)."""
        goal = Manipulate.Goal()
        goal.frame_id = self.pick_target["frame_id"]
        goal.x_center = self.pick_target["x_center"]
        goal.y_center = self.pick_target["y_center"]
        goal.timestamp = self.pick_target["stamp"]
        return goal

    def return_home_and_sit(self):
        """Blocking, in IDLE: asks for confirmation, walks back to home_pose (arm stowed, gripper
        closed by navigate_to) and sits the robot. Returns False on shutdown; raises on failure."""
        if self.home_pose is None:
            self.get_logger().warn(f"[{self.get_name()}] No initial position saved, not returning")
            return True
        x, y, yaw = self.home_pose
        if not self.wait_for_confirmation(f"go back to the initial position ({x:.2f}, {y:.2f}) and sit"):
            return False
        
        self.get_logger().info(f"[{self.get_name()}] Going back to the initial position")
        if self.face_goal_before_navigation:
            rx, ry, _ = self.robot_pose()
            distance = float(np.hypot(x - rx, y - ry))
            if distance > self.face_goal_min_distance:
                heading = float(np.arctan2(y - ry, x - rx))
                self.get_logger().info(f"[{self.get_name()}] Turning towards the initial position "
                                       f"({distance:.2f} m away, heading {np.degrees(heading):.0f} deg)")
                # the arm must be stowed before turning (navigate_to stows it again, already folded)
                self.reclaim_lease()
                self.stow_arm()
                if not self.rotate_to_yaw(heading):
                    return False
        status = self.navigate_to(x, y, yaw)
        if status != GoalStatus.STATUS_SUCCEEDED:
            raise RuntimeError(f"Navigation to the initial position ended with status {status}")
        if self.align_home_yaw:
            self.get_logger().info(f"[{self.get_name()}] Turning to the start orientation "
                                   f"({np.degrees(yaw):.0f} deg)")
            if not self.rotate_to_yaw(yaw):
                return False
        self.get_logger().info(f"[{self.get_name()}] Back at the initial position, sitting down")
        self.call_trigger(self.sit_client, self.sit_service, self.posture_timeout)
        self.robot_sitting = True
        return True

    def rotate_to_yaw(self, target_yaw):
        """Blocking: turns the robot in place to target_yaw (rad, in global_frame) publishing cmd_vel
        with a proportional control. Returns False on shutdown; raises RuntimeError on timeout."""
        self.reclaim_lease()
        deadline = time.monotonic() + self.align_timeout
        try:
            while True:
                _, _, yaw = self.robot_pose()
                error = float(np.arctan2(np.sin(target_yaw - yaw), np.cos(target_yaw - yaw)))
                if abs(error) <= self.align_yaw_tolerance:
                    self.get_logger().info(f"[{self.get_name()}] Orientation reached "
                                           f"(error {np.degrees(error):.1f} deg)")
                    return True
                if self.stopping:
                    return False
                if time.monotonic() > deadline:
                    raise RuntimeError(f"Orientation not reached in {self.align_timeout:.0f} s "
                                       f"(error {np.degrees(error):.1f} deg)")
                speed = min(self.align_max_angular_vel, max(self.align_min_angular_vel,
                                                            self.align_gain * abs(error)))
                cmd = Twist()
                cmd.angular.z = speed if error > 0 else -speed
                self.cmd_vel_pub.publish(cmd)
                time.sleep(0.1)
        finally:
            # stop now: Spot keeps executing the last cmd_vel for cmd_duration (0.7 s)
            self.cmd_vel_pub.publish(Twist())

    def robot_pose(self):
        """(x, y, yaw) of base_frame in global_frame."""
        tf = self.tf_buffer.lookup_transform(self.global_frame, self.base_frame, Time())
        t, q = tf.transform.translation, tf.transform.rotation
        yaw = float(np.arctan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z)))
        return t.x, t.y, yaw

    def robot_xy(self):
        tf = self.tf_buffer.lookup_transform(self.global_frame, self.base_frame, Time())
        return tf.transform.translation.x, tf.transform.translation.y

    def navigate_to(self, x, y, yaw):
        """Blocking: sends a NavigateToPose goal and waits until the motion is over.
        Returns the final GoalStatus; raises on rejection or after navigation_timeout (goal cancelled)."""
        
        # take the lease, then stow the arm and close the gripper before walking
        self.reclaim_lease()
        self.stow_arm()

        goal = NavigateToPose.Goal()
        goal.pose = PoseStamped()
        goal.pose.header.frame_id = self.global_frame
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x, goal.pose.pose.position.y = x, y
        q = quat_from_yaw(yaw)
        o = goal.pose.pose.orientation
        o.x, o.y, o.z, o.w = q
        goal.behavior_tree = self.behavior_tree

        deadline = time.monotonic() + self.navigation_timeout
        self.nav_feedback = None
        goal_handle = self.wait_future(
            self.nav_client.send_goal_async(goal, feedback_callback=self.nav_feedback_callback), deadline)
        if goal_handle is None:
            raise RuntimeError("No answer from Nav2 to the navigation goal")
        if not goal_handle.accepted:
            raise RuntimeError("Navigation goal rejected by Nav2")
        # kept so the goal can be cancelled on timeout or shutdown
        self.nav_goal_handle = goal_handle

        response = self.wait_future(goal_handle.get_result_async(), deadline)
        if response is None:
            self.cancel_navigation()
            raise RuntimeError(f"Navigation timeout ({self.navigation_timeout:.0f} s), goal cancelled; "
                               f"last Nav2 feedback: {self.describe_nav_feedback()}")
        self.nav_goal_handle = None
        return response.status

    def nav_feedback_callback(self, msg):
        # NavigateToPose feedback: kept for the timeout message and logged every few seconds
        self.nav_feedback = msg.feedback
        self.get_logger().info(f"[{self.get_name()}] Nav2: {self.describe_nav_feedback()}",
                               throttle_duration_sec=3.0)

    def describe_nav_feedback(self):
        fb = getattr(self, "nav_feedback", None)
        if fb is None:
            return "none received (Nav2 never started executing the goal)"
        return (f"distance remaining {fb.distance_remaining:.2f} m, "
                f"recoveries {fb.number_of_recoveries}, "
                f"navigating for {fb.navigation_time.sec} s")

    def reclaim_lease(self):
        """Blocking, from the FSM tick: calls the claim service before a motion (arm move, navigation)
        when claim_before_motion is true. Raises RuntimeError on failure."""
        if self.claim_before_motion:
            self.call_trigger(self.claim_client, self.claim_service, self.claim_timeout)

    def stow_arm(self):
        """Blocking: stows the arm, then closes the gripper. Every stow goes through here, so the
        robot never walks with the gripper open. Raises RuntimeError on failure."""
        self.call_trigger(self.arm_stow_client, self.arm_stow_service, self.arm_stow_timeout)
        self.call_trigger(self.close_gripper_client, self.close_gripper_service, self.close_gripper_timeout)

    def call_trigger(self, client, name, timeout):
        """Blocking std_srvs/Trigger call from the FSM tick; raises if unavailable, late or failed.
        The client must have its own callback group (its response is processed by another thread)."""
        if not client.service_is_ready():
            raise RuntimeError(f"Service '{name}' not available")
        self.get_logger().info(f"[{self.get_name()}] Calling '{name}'")
        res = self.wait_future(client.call_async(Trigger.Request()), time.monotonic() + timeout)
        if res is None:
            raise RuntimeError(f"Service '{name}' did not answer in {timeout:.0f} s")
        if not res.success:
            raise RuntimeError(f"Service '{name}' failed: {res.message}")
        self.get_logger().info(f"[{self.get_name()}] '{name}' succeeded: {res.message}")

    def wait_future(self, future, deadline):
        """Blocks the FSM thread until the future is done (its response is processed by another executor
        thread). Returns the result, or None on deadline or shutdown."""
        while not future.done():
            if self.stopping or time.monotonic() > deadline:
                return None
            time.sleep(0.05)
        return future.result()

    def cancel_navigation(self):
        if self.nav_goal_handle is not None:
            self.get_logger().info(f"[{self.get_name()}] Cancelling navigation goal")
            future = self.nav_goal_handle.cancel_goal_async()
            self.nav_goal_handle = None
            return future
        return None

    def shutdown_robot(self):
        """Called once when the node is killed, independently of the current state:
        stops the FSM, cancels the navigation and resets the arm through the arm_reset_service."""
        self.get_logger().warn(f"[{self.get_name()}] Shutdown in state {self.fsm.state.name}: "
                               f"resetting the arm with '{self.arm_reset_service}'")
        if getattr(self, "timer", None) is not None:
            self.timer.cancel()
        self.stopping = True

        cancel_future = self.cancel_navigation() if hasattr(self, "nav_goal_handle") else None
        if cancel_future is not None:
            self.executor.spin_until_future_complete(cancel_future, timeout_sec=1.0)

        if not self.arm_reset_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error(f"[{self.get_name()}] Service '{self.arm_reset_service}' not available, " f"arm NOT reset")
            return
        future = self.arm_reset_client.call_async(Trigger.Request())
        self.executor.spin_until_future_complete(future, timeout_sec=self.shutdown_timeout)
        if not future.done():
            self.get_logger().error(f"[{self.get_name()}] '{self.arm_reset_service}' did not answer in " f"{self.shutdown_timeout:.0f} s")
            return
        res = future.result()
        log = self.get_logger().info if res.success else self.get_logger().error
        log(f"[{self.get_name()}] '{self.arm_reset_service}': success={res.success} {res.message}")


_stopping = False


def _on_stop_signal(signum, frame):
    # first SIGINT/SIGTERM interrupts spin(); the following ones are ignored so they cannot interrupt
    # the shutdown (ros2 launch forwards a second SIGINT after Ctrl+C, then SIGTERM after sigterm_timeout)
    global _stopping
    if _stopping:
        return
    _stopping = True
    raise KeyboardInterrupt


def main(args=None):
    # rclpy must NOT shut its context down on Ctrl+C: the node still has to call the arm reset service
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    signal.signal(signal.SIGINT, _on_stop_signal)
    signal.signal(signal.SIGTERM, _on_stop_signal)

    # multi-threaded: blocking calls inside the FSM (send_goal) get their responses from another thread
    executor = MultiThreadedExecutor()
    node = None
    exit_code = 0
    try:
        node = FSMNode()
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    except Exception:
        # e.g. lease claim failed in the constructor: print it, os._exit below would hide it
        traceback.print_exc()
        exit_code = 1
    finally:
        if node is not None:
            node.shutdown_robot()
            executor.remove_node(node)
            node.destroy_node()
        # shutting down the context also stops the TF listener thread, which spins node.tf_node
        rclpy.try_shutdown()
        # an FSM thread may still be blocked in a call that does not watch for shutdown (send_goal of
        # the arm/pick, gripper .call()): Python would wait for it before exiting, and ros2 launch would
        # kill the node only after sigterm_timeout + sigkill_timeout. The cleanup is done: exit now.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)


if __name__ == "__main__":
    main()
