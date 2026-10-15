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

from bosdyn_msgs.conversions import convert
from bosdyn.client.robot_command import RobotCommandBuilder
from bosdyn.api import manipulation_api_pb2, geometry_pb2
from bosdyn.api.manipulation_api_pb2 import WalkGazeMode
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
        
        self.grasp_params = manipulation_api_pb2.GraspParams(
            grasp_palm_to_fingertip=0.5,
            grasp_params_frame_name=HAND_FRAME,
        )
        
        self.tfBuffer = Buffer()
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
    
    def _create_frame_tree_snapshot(self):
        
        frame_tree_edges = {}
                    
        dynamic_transforms = {}
        
        lookup_time = rclpy.time.Time()
                    
        timeout = 10.0
        
        for parent, child in ODOM_TO_HAND_CAMERA_EDGES:
            
            self.logger.info(f"Waiting for transform {parent} → {child}...")
            
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
                        self.logger.info(f"✓ Cached: {parent} → {child}")
                        break

                    except TransformException as ex:
                        self.logger.warning(f"Lookup failed: {ex}, retrying...")

                time.sleep(0.1)
            else:
                self.logger.error(f"✗ Timeout waiting for {parent} → {child}")
                return False
        
        self.logger.info(f"✓ Successfully cached {len(dynamic_transforms)} dynamic transforms")
        
        body_tf = geometry_pb2.SE3Pose(
            position=geometry_pb2.Vec3(x=0.0, y=0.0, z=0.0),
            rotation=geometry_pb2.Quaternion(w=1.0, x=0.0, y=0.0, z=0.0),
        )
        
        frame_tree_edges = add_edge_to_tree(frame_tree_edges, body_tf, "", BODY_FRAME)
        
        for _, edge_data in dynamic_transforms.items():
            frame_tree_edges = add_edge_to_tree(
                frame_tree_edges,
                edge_data['transform'],
                edge_data['parent'],
                edge_data['child'],
            )
            self.logger.debug(f"Added: {edge_data['parent']} → {edge_data['child']}")

        return geometry_pb2.FrameTreeSnapshot(child_to_parent_edge_map=frame_tree_edges)

    def manipulation_feedback(self, feedback_msg):
        self.logger.debug(f"Manipulation feedback: {feedback_msg.feedback}")
    
    async def handle_manipulation(self, goal_handle: ServerGoalHandle) -> Manipulate.Result:
        
        if not self.manipulation_client.wait_for_server(timeout_sec=5.0):
            goal_handle.abort()
            return Manipulate.Result()
        
        snapshot_tree = self._create_frame_tree_snapshot()
        
        if snapshot_tree is False:
            goal_handle.abort()
            result.success = False
            result.message = "Could not build frame tree snapshot."
            return result
            
        
        request = goal_handle.request
        
        center = geometry_pb2.Vec2(x=request.x_center, y=request.y_center)
        
        frame_id = request.frame_id
        
        grasp = manipulation_api_pb2.PickObjectInImage(
            pixel_xy=center,
            transforms_snapshot_for_camera=snapshot_tree,
            frame_name_image_sensor=frame_id,
            camera_model=self.pin_model,
            grasp_params=self.grasp_params,
            walk_gaze_mode = WalkGazeMode.PICK_AUTO_GAZE 
        )
        
        grasp_request = manipulation_api_pb2.ManipulationApiRequest(pick_object_in_image=grasp)
        grasp_goal = Manipulation.Goal()
        convert(grasp_request, grasp_goal.command)
        
        self.logger.info("Sending grasp goal to action server...")
        
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
        
        return self.map_manipulation_result(inner_result, goal_handle)
    
    def map_manipulation_result(self, inner_result, goal_handle):
        
        manipulation_result = Manipulate.Result()
        success = False
        message = "Manipulation failed."

        if inner_result.status == GoalStatus.STATUS_CANCELED:
            message = "Manipulation goal was canceled."
            goal_handle.canceled()

        elif inner_result.status == GoalStatus.STATUS_SUCCEEDED:
            
            final_result = inner_result.result
            if final_result.success:
                final_state = final_result.result.current_state.value
                
                # State 6 corresponds to MANIP_STATE_GRASP_SUCCEEDED in BD API
                if final_state == 6:
                    
                    with self.gripper_lock:
                        gripper_percentage = self._gripper_percentage
                        
                    if gripper_percentage > 5.0:
                        goal_handle.succeed()
                        success = True
                        message = "Manipulation Successfully Completed!"   
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
            return result

        self._active_arm_goal_handle = inner_goal_handle

        try:
            inner_result = await inner_goal_handle.get_result_async()
        finally:
            self._active_arm_goal_handle = None

        if inner_result.status == GoalStatus.STATUS_CANCELED:
            goal_handle.canceled()
            result = MoveArmToPose.Result()
            result.success = success
            result.message = "Arm movement was canceled."
            return result

        if inner_result.status != GoalStatus.STATUS_SUCCEEDED:
            goal_handle.abort()
            result = MoveArmToPose.Result()
            result.success = success
            result.message = f"Arm movement failed with status {inner_result.status}."
            return result

        goal_handle.succeed()
        result = MoveArmToPose.Result()
        result.success = True
        result.message = "Arm reached the requested pose."
        return result
    
    def move_arm_cancel_callback(self, cancel_request) -> CancelResponse:
        self.get_logger().info("Cancellation requested for arm movement.")

        inner_goal_handle = self._active_arm_goal_handle

        if inner_goal_handle is not None:
            self.get_logger().info("Forwarding cancellation to robot_command action.")
            inner_goal_handle.cancel_goal_async()

        return CancelResponse.ACCEPT

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
