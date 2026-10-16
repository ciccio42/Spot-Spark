import argparse
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.action import ActionServer, ActionClient
from rclpy.action.server import ServerGoalHandle, CancelResponse
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from typing import Optional
import time
import threading
import math
import signal
from rclpy.time import Time


from bosdyn_msgs.conversions import convert
from bosdyn.client.robot_command import RobotCommandBuilder
from bosdyn.api import manipulation_api_pb2, geometry_pb2
from bosdyn.api.image_pb2 import ImageSource
from spot_msgs.action import Manipulation, RobotCommand
from bosdyn_api_msgs.msg import ManipulatorState
from geometry_msgs.msg import TransformStamped, PoseStamped
from bosdyn.client.frame_helpers import add_edge_to_tree
from bosdyn.client.math_helpers import Quat, SE3Pose
from tf2_ros import Buffer, TransformListener, TransformException
from action_msgs.msg import GoalStatus
from .utils import *
from manipulation_interfaces.action import Manipulate, MoveArmToPose


class ManipulationNode(Node):

    def __init__(self) -> None:

        super().__init__("manipulation_node")
        
        self.pin_model = self._create_pin_model()

        # MoveArmToPose completion: with an object in the gripper Spot may keep the arm command
        # IN_PROGRESS forever (it keeps correcting a few mm). After the requested duration the hand
        # pose is checked on TF: within these tolerances the move is considered done.
        self.declare_parameter("arm_goal_position_tolerance", 0.03)     # m
        self.declare_parameter("arm_goal_orientation_tolerance", 0.10)  # rad (~6 deg)
        # extra time (s) after the requested duration before the move is declared failed
        self.declare_parameter("arm_goal_timeout", 5.0)
        # cancel the robot_command goal once the arm is in tolerance. On every cancel the driver sends a
        # stop command a moment later, which overrides the next command (e.g. the gripper opening:
        # STATUS_COMMAND_OVERRIDDEN). Default false: the goal is left running, the next robot command
        # replaces it (a new robot_command goal makes the driver cancel + stop it BEFORE executing).
        self.declare_parameter("cancel_arm_goal_on_reached", False)
        self.arm_pos_tol = float(self.get_parameter("arm_goal_position_tolerance").value)
        self.arm_rot_tol = float(self.get_parameter("arm_goal_orientation_tolerance").value)
        self.arm_goal_timeout = float(self.get_parameter("arm_goal_timeout").value)
        self.cancel_arm_goal_on_reached = bool(self.get_parameter("cancel_arm_goal_on_reached").value)
        
        self.grasp_params = manipulation_api_pb2.GraspParams(
            grasp_palm_to_fingertip=0.5,
            grasp_params_frame_name=HAND_FRAME,
        )
        
        # TF history long enough to look up the transforms at the image time of a pick request,
        # even if the request arrives a while after the detection (e.g. after waiting for Enter)
        self.tfBuffer = Buffer(cache_time=rclpy.duration.Duration(seconds=120.0))
        self.listener = TransformListener(self.tfBuffer, self)
        
        self.cb_server = ReentrantCallbackGroup()
        self.cb_client = ReentrantCallbackGroup()

        self.manipulation_server = ActionServer(
            self,
            Manipulate,
            "/manipulation_custom",
            execute_callback = self.handle_manipulation,
            callback_group=self.cb_server,
            cancel_callback=self.manipulation_cancel_callback,
        )
        
        self.arm_pose_server = ActionServer(
            self, 
            MoveArmToPose,
            "/move_arm_to_pose",
            execute_callback = self.handle_move_arm_to_pose,
            callback_group=self.cb_server,
            cancel_callback=self.move_arm_cancel_callback,
        )
        
        self.manipulation_client = ActionClient(
            self,
            Manipulation, 
            "/manipulation",
            callback_group = self.cb_client
        )
        
        self.robot_command_client = ActionClient(
            self,
            RobotCommand,
            "/robot_command",
            callback_group=self.cb_client,
        )
        
        self._active_manipulation_goal_handle = None
        
        self._active_arm_goal_handle = None
        
        self.gripper_lock = threading.Lock()
        self.gripper_percentage = 0.0
        
        self.gripper_sub = self.create_subscription(
            ManipulatorState,
            '/manipulation_state',
            self.gripper_callback,
            10)
    
    def gripper_callback(self, msg):
        with self.gripper_lock:
            self.gripper_percentage = msg.gripper_open_percentage

    def _create_pin_model(self) -> dict:

        intrinsics = CAMERA_INTRINSICS[HAND_CAMERA_FRAME]
        focal_length = geometry_pb2.Vec2(x=intrinsics['fx'], y=intrinsics['fy'])
        center       = geometry_pb2.Vec2(x=intrinsics['cx'], y=intrinsics['cy'])

        camera_intrinsics = ImageSource.PinholeModel.CameraIntrinsics(
            focal_length=focal_length, principal_point=center
        )
        pin_model = ImageSource.PinholeModel(intrinsics=camera_intrinsics)

        return pin_model
        
    def _tformstamp_to_tform(self, tformstamp: TransformStamped, parent=None, child=None):
        """
        Converts a ROS 2 TransformStamped message into a Spot API SE3Pose object.
        Automatically handles orientation corrections for specific Spot cameras.

        Args:
            tformstamp (TransformStamped): The ROS transform to convert.
            parent (str, optional): The parent frame name.
            child (str, optional): The child frame name.

        Returns:
            SE3Pose: The equivalent Boston Dynamics transform.
        """
        traslation = tformstamp.transform.translation
        rotation1  = tformstamp.transform.rotation

        position  = geometry_pb2.Vec3(x=traslation.x, y=traslation.y, z=traslation.z)
        rotation2 = geometry_pb2.Quaternion(
            w=rotation1.w, x=rotation1.x, y=rotation1.y, z=rotation1.z
        )

        if parent is not None and child is not None:
            if parent == HEAD_FRAME and (
                child == RIGHT_FRAME
                or child == BACK_FRAME
            ):
                rotation2 = geometry_pb2.Quaternion(
                    w=-rotation1.w, x=-rotation1.x, y=-rotation1.y, z=-rotation1.z
                )

        transform = geometry_pb2.SE3Pose(position=position, rotation=rotation2)
        return transform
    
    def _create_frame_tree_snapshot(self, stamp=None):
        """Frame tree body -> odom/vision/hand camera for the Manipulation API.

        stamp: time of the image the pixel comes from (builtin_interfaces/Time). The transforms are
        looked up AT THAT TIME, so the pixel ray is computed from where the camera was when the image
        was taken, even if the arm or the body moved afterwards. None or zero = latest transforms.
        """
        frame_tree_edges = {}
                    
        dynamic_transforms = {}
        
        if stamp is not None and (stamp.sec or stamp.nanosec):
            lookup_time = rclpy.time.Time.from_msg(stamp)
        else:
            self.get_logger().warning("No image timestamp in the request: using the latest transforms")
            lookup_time = rclpy.time.Time()
                    
        timeout = 10.0
        
        for parent, child in BODY_TO_HAND_CAMERA_EDGES:
            
            self.get_logger().info(f"Waiting for transform {parent} → {child}...")
            
            start = time.time()
            while (time.time() - start) < timeout:
                if self.tfBuffer.can_transform(
                    parent,
                    child,
                    lookup_time,
                    timeout=rclpy.duration.Duration(seconds=0.1),
                ):
                    try:
                        transform = self.tfBuffer.lookup_transform(parent, child, lookup_time)
                        key = f"{parent}_to_{child}"
                        dynamic_transforms[key] = {
                            'parent':    parent,
                            'child':     child,
                            'transform': self._tformstamp_to_tform(transform, parent, child),
                        }
                        self.get_logger().info(f"✓ Cached: {parent} → {child}")
                        break

                    except TransformException as ex:
                        self.get_logger().warning(f"Lookup failed: {ex}, retrying...")

                time.sleep(0.1)
            else:
                self.get_logger().error(f"✗ Timeout waiting for {parent} → {child}")
                return False
        
        self.get_logger().info(f"✓ Successfully cached {len(dynamic_transforms)} dynamic transforms")
        
        body_tf = geometry_pb2.SE3Pose(
            position=geometry_pb2.Vec3(x=0.0, y=0.0, z=0.0),
            rotation=geometry_pb2.Quaternion(w=1.0, x=0.0, y=0.0, z=0.0),
        )
        
        for _, edge_data in dynamic_transforms.items():
            frame_tree_edges = add_edge_to_tree(
                frame_tree_edges,
                edge_data['transform'],
                edge_data['parent'],
                edge_data['child'],
            )
            self.get_logger().info(f"Added: {edge_data['parent']} → {edge_data['child']}")
        
        frame_tree_edges = add_edge_to_tree(frame_tree_edges, body_tf, "", BODY_FRAME)

        return geometry_pb2.FrameTreeSnapshot(child_to_parent_edge_map=frame_tree_edges)

    def manipulation_feedback(self, feedback_msg):
        self.get_logger().debug(f"Manipulation feedback: {feedback_msg.feedback}")
    
    async def handle_manipulation(self, goal_handle: ServerGoalHandle) -> Manipulate.Result:
        
        if not self.manipulation_client.wait_for_server(timeout_sec=5.0):
            goal_handle.abort()
            return Manipulate.Result()
        
        request = goal_handle.request
        
        timestamp = request.timestamp
        
        snapshot_tree = self._create_frame_tree_snapshot(timestamp)
        
        if snapshot_tree is False:
            goal_handle.abort()
            result = Manipulate.Result()
            result.success = False
            result.message = "Could not build frame tree snapshot."
            return result
        
        center = geometry_pb2.Vec2(x=request.x_center, y=request.y_center)
        
        frame_id = request.frame_id
        
        grasp = manipulation_api_pb2.PickObjectInImage(
            pixel_xy=center,
            transforms_snapshot_for_camera=snapshot_tree,
            frame_name_image_sensor=frame_id,
            camera_model=self.pin_model,
            grasp_params=self.grasp_params,
            # walk_gaze_mode = manipulation_api_pb2.WalkGazeMode.PICK_AUTO_GAZE 
        )
        
        grasp_request = manipulation_api_pb2.ManipulationApiRequest(pick_object_in_image=grasp)
        grasp_goal = Manipulation.Goal()
        convert(grasp_request, grasp_goal.command)
        
        self.get_logger().info("Sending grasp goal to action server...")
        
        inner_goal_handle = await self.manipulation_client.send_goal_async(
            grasp_goal,
            feedback_callback=self.manipulation_feedback,
        )
        
        if not inner_goal_handle.accepted:
            goal_handle.abort()
            result = Manipulate.Result()
            result.success = False
            result.message = "Manipulation failed."
            return result
        
        self._active_manipulation_goal_handle = inner_goal_handle
        
        try:
            inner_result = await inner_goal_handle.get_result_async()
        finally:
            self._active_manipulation_goal_handle = None
        
        result = self.map_manipulation_result(inner_result, goal_handle)

        self.get_logger().info(
            f"Returning manipulation result: success={result.success}, message='{result.message}'"
        )

        return result
    
    def map_manipulation_result(self, inner_result, goal_handle):
        
        manipulation_result = Manipulate.Result()
        success = False
        message = "Manipulation failed."
        
        self.get_logger().info("Mapping Manipulation Result")

        if inner_result.status == GoalStatus.STATUS_CANCELED:
            self.get_logger().info("Manipulation Canceled")
            message = "Manipulation goal was canceled."
            goal_handle.canceled()

        elif inner_result.status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info("Manipulation Apparently Succeeded")
            final_result = inner_result.result
            if final_result.success:
                final_state = final_result.result.current_state.value
                
                self.get_logger().info(f"Final Manipulation State: {final_state}")
                # State 6 corresponds to MANIP_STATE_GRASP_SUCCEEDED in BD API
                if final_state == 6:
                    
                    with self.gripper_lock:
                        gripper_percentage = self.gripper_percentage
                    self.get_logger().info(f"Gripper Percentage: {gripper_percentage}")
                    
                    if gripper_percentage > 5.0:
                        self.get_logger().info("Calling goal_handle.succeed()")

                        success = True
                        message = "Manipulation Successfully Completed!"
                        goal_handle.succeed()

                        self.get_logger().info("goal_handle.succeed() returned")
                    else:
                        goal_handle.abort()
                else:
                    goal_handle.abort()
            else:
                goal_handle.abort()
        else:
            goal_handle.abort()
        
        manipulation_result.success = success
        manipulation_result.message = message
        
        return manipulation_result
                
    def manipulation_cancel_callback(self, cancel_request):
        self.get_logger().info("Cancellation requested for manipulation goal.")

        inner_goal_handle = self._active_manipulation_goal_handle

        if inner_goal_handle is not None:
            self.get_logger().info("Forwarding cancellation to inner manipulation action.")
            inner_goal_handle.cancel_goal_async()

        return CancelResponse.ACCEPT
    
    async def handle_move_arm_to_pose(self, goal_handle: ServerGoalHandle,) -> MoveArmToPose.Result:
        
        success = False
        message = ""
        
        request = goal_handle.request

        pose_stamped = request.pose_stamped
        pose = pose_stamped.pose
        point = pose.position
        quat = pose.orientation
        
        hand_rotation = Quat(
            w=quat.w,
            x=quat.x,
            y=quat.y,
            z=quat.z,
        )

        hand_pose = SE3Pose(
            x=point.x,
            y=point.y,
            z=point.z,
            rot=hand_rotation,
        ).to_proto()

        command = RobotCommandBuilder.arm_pose_command_from_pose(
            hand_pose=hand_pose,
            frame_name=BODY_FRAME,
            seconds=request.duration,
        )

        robot_command_goal = RobotCommand.Goal()
        convert(command, robot_command_goal.command)

        inner_goal_handle = await self.robot_command_client.send_goal_async(
            robot_command_goal
        )

        if not inner_goal_handle.accepted:
            goal_handle.abort()
            result = MoveArmToPose.Result()
            result.success = success
            result.message = "Robot command goal was rejected."
            self.get_logger().info("Robot Command Rejected")
            return result

        self._active_arm_goal_handle = inner_goal_handle

        try:
            outcome, data = self._wait_arm_goal(inner_goal_handle, pose, request.duration)
        finally:
            self._active_arm_goal_handle = None

        if outcome == "reached":
            pos_err, rot_err = data
            goal_handle.succeed()
            result = MoveArmToPose.Result()
            result.success = True
            result.message = (f"Arm reached the requested pose (error {pos_err * 1000:.0f} mm, "
                              f"{math.degrees(rot_err):.1f} deg).")
            self.get_logger().info(result.message)
            return result

        if outcome == "timeout":
            result = MoveArmToPose.Result()
            result.success = False
            result.message = (f"Arm did not reach the requested pose within "
                              f"{request.duration + self.arm_goal_timeout:.1f} s ({data}).")
            self.get_logger().error(result.message)
            self.safe_abort(goal_handle)
            return result

        inner_result = data
        if inner_result.status == GoalStatus.STATUS_CANCELED:
            result = MoveArmToPose.Result()
            result.success = success
            result.message = "Arm movement was canceled."
            self.get_logger().info("Arm Movement Canceled")
            
            if goal_handle.is_cancel_requested:
                self.safe_cancel(goal_handle)
            else:
                self.safe_abort(goal_handle)
            return result

        if inner_result.status != GoalStatus.STATUS_SUCCEEDED:
            result = MoveArmToPose.Result()
            result.success = success
            result.message = f"Arm movement failed with status {inner_result.status}."
            self.get_logger().info(f"Arm movement failed with status {inner_result.status}.")
            self.safe_abort(goal_handle)
            return result

        goal_handle.succeed()
        result = MoveArmToPose.Result()
        result.success = True
        result.message = "Arm reached the requested pose."
        self.get_logger().info("Arm Movement Succeded")
        return result
    
    def _wait_arm_goal(self, inner_goal_handle, target_pose, duration):
        """Waits for an arm robot_command goal. Returns:
          ("result", GetResult response)  the driver ended the goal (completed, canceled, failed)
          ("reached", (pos_err, rot_err)) after `duration` the hand is within tolerance on TF; the goal
                                          is left running (canceled only with cancel_arm_goal_on_reached),
                                          Spot keeps holding the pose
          ("timeout", description)        not reached within duration + arm_goal_timeout: canceled
        Blocking: the result future and TF are processed by the other executor threads."""
        result_future = inner_goal_handle.get_result_async()
        start = time.monotonic()
        deadline = start + duration + self.arm_goal_timeout
        error = None
        while not result_future.done():
            now = time.monotonic()
            if now - start >= duration:
                error = self._hand_pose_error(target_pose)
                if error is not None and error[0] <= self.arm_pos_tol and error[1] <= self.arm_rot_tol:
                    if self.cancel_arm_goal_on_reached:
                        inner_goal_handle.cancel_goal_async()
                    return "reached", error
            if now > deadline:
                inner_goal_handle.cancel_goal_async()
                if error is None:
                    return "timeout", f"no TF {BODY_FRAME} -> {HAND_FRAME}"
                return "timeout", (f"error {error[0] * 1000:.0f} mm, {math.degrees(error[1]):.1f} deg")
            time.sleep(0.05)
        return "result", result_future.result()

    def _hand_pose_error(self, target_pose):
        """(position error m, rotation error rad) between the hand on TF and target_pose (in body)."""
        try:
            tf = self.tfBuffer.lookup_transform(BODY_FRAME, HAND_FRAME, Time())
        except TransformException:
            return None
        t, q = tf.transform.translation, tf.transform.rotation
        p, r = target_pose.position, target_pose.orientation
        pos_err = math.sqrt((t.x - p.x) ** 2 + (t.y - p.y) ** 2 + (t.z - p.z) ** 2)
        q_norm = math.sqrt(r.w ** 2 + r.x ** 2 + r.y ** 2 + r.z ** 2) or 1.0
        dot = abs(q.w * r.w + q.x * r.x + q.y * r.y + q.z * r.z) / q_norm
        rot_err = 2.0 * math.acos(min(1.0, dot))
        return pos_err, rot_err

    def move_arm_cancel_callback(self, cancel_request) -> CancelResponse:
        self.get_logger().info("Cancellation requested for arm movement.")

        inner_goal_handle = self._active_arm_goal_handle

        if inner_goal_handle is not None:
            self.get_logger().info("Forwarding cancellation to robot_command action.")
            inner_goal_handle.cancel_goal_async()

        return CancelResponse.ACCEPT

    def safe_cancel(self, goal_handle: ServerGoalHandle):
        try:
            goal_handle.canceled()
        except Exception as exc:
            self.get_logger().warning(f"Could not mark goal as canceled: {exc}")


    def safe_abort(self, goal_handle: ServerGoalHandle):
        try:
            goal_handle.abort()
        except Exception as exc:
            self.get_logger().warning(f"Could not abort goal: {exc}")
            
shutdown_requested = False

def signal_handler(sig, frame):
    """
    Intercepts system signals (e.g., SIGINT) to trigger a graceful shutdown loop.

    Args:
        sig (int): The signal number received.
        frame: The current stack frame at the time of the signal.
    """
    global shutdown_requested
    shutdown_requested = True
    
def main(args=None):
    
    rclpy.init()
    node = ManipulationNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        while rclpy.ok() and not shutdown_requested:
            executor.spin_once(timeout_sec=0.1)
    except KeyboardInterrupt:
        executor.shutdown()
        executor.remove_node(node)
        node.destroy_node()
        rclpy.shutdown()
    finally:
        executor.shutdown()
        executor.remove_node(node)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
