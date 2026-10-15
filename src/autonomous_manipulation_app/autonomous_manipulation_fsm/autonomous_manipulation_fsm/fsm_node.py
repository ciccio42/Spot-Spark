import signal
import sys
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.time import Time
from rclpy.duration import Duration
from rclpy.signals import SignalHandlerOptions
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from action_msgs.msg import GoalStatus
from ament_index_python.packages import get_package_share_directory, PackageNotFoundError

import tf2_ros
from tf2_geometry_msgs import do_transform_point
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, PoseStamped
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
        self.near_detection_pose = self.declare_pose_parameter("near_detection_pose")

        # ---------- dependencies ----------
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
        self.manipulate_client = ActionClient(self, Manipulate, gp("manipulate_action"))
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
        # set on shutdown: makes the blocking waits (wait_future) return
        self.stopping = False
        # DETECT runs in two modes:
        #   "far"  (from IDLE): arm to far_detection_pose, 3D ball position for the navigation
        #   "near" (from PICK): arm to near_detection_pose, bbox center (pixels) for the pick, back to PICK
        self.detection_mode = "far"
        # result of the near detection: {"frame_id", "x_center", "y_center", "stamp"}, None until available
        self.pick_target = None
        self.start_requested = threading.Event()
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
            if line.strip() == "" and self.fsm.state == State.IDLE:
                self.start_requested.set()

    def start_callback(self, request, response):
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
        # TODO: send the goal and wait for the result
        return

        # 

        # after the pick move the robot arm to save-position


        # If pick is successful, transition to NAVIGATE_TO_PLACE
        self.fsm.transition(State.NAVIGATE_TO_PLACE)

    def on_navigate_to_place(self, entering):
        pass

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
        if res.normalized:
            if self.camera_info is None:
                self.get_logger().warn(f"[{self.get_name()}] Normalized detection but no camera_info yet",
                                       throttle_duration_sec=2.0)
                return None
            x, y = x * self.camera_info.width, y * self.camera_info.height
        return {"frame_id": res.frame_id, "x_center": float(x), "y_center": float(y), "stamp": res.stamp}

    def build_pick_goal(self):
        """Manipulate goal from the near detection (self.pick_target)."""
        goal = Manipulate.Goal()
        goal.frame_id = self.pick_target["frame_id"]
        goal.x_center = self.pick_target["x_center"]
        goal.y_center = self.pick_target["y_center"]
        goal.timestamp = self.pick_target["stamp"]
        return goal

    def robot_xy(self):
        tf = self.tf_buffer.lookup_transform(self.global_frame, self.base_frame, Time())
        return tf.transform.translation.x, tf.transform.translation.y

    def navigate_to(self, x, y, yaw):
        """Blocking: sends a NavigateToPose goal and waits until the motion is over.
        Returns the final GoalStatus; raises on rejection or after navigation_timeout (goal cancelled)."""
        
        # stow the arm before walking
        self.call_trigger(self.arm_stow_client, self.arm_stow_service, self.arm_stow_timeout)

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
        goal_handle = self.wait_future(self.nav_client.send_goal_async(goal), deadline)
        if goal_handle is None:
            raise RuntimeError("No answer from Nav2 to the navigation goal")
        if not goal_handle.accepted:
            raise RuntimeError("Navigation goal rejected by Nav2")
        # kept so the goal can be cancelled on timeout or shutdown
        self.nav_goal_handle = goal_handle

        response = self.wait_future(goal_handle.get_result_async(), deadline)
        if response is None:
            self.cancel_navigation()
            raise RuntimeError(f"Navigation timeout ({self.navigation_timeout:.0f} s), goal cancelled")
        self.nav_goal_handle = None
        return response.status

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
    try:
        node = FSMNode()
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.shutdown_robot()
            executor.remove_node(node)
            node.destroy_node()
        # shutting down the context also stops the TF listener thread, which spins node.tf_node
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
