# Nodo che riceve TargetInfoMessage e calcola la posa 3D del target in
# coordinate camera (posizione a distanza di sicurezza + yaw), pubblicando
# il risultato su TargetPose3D.
import time

from demo_interfaces.msg import TargetInfoMessage, TargetPose3D
from sensor_msgs.msg import CameraInfo
from demo_package.common import deproject_pixel_to_point, CameraIntrinsics
import math
import rclpy
from rclpy.node import Node


class Pose3DEstimationNode(Node):
    def __init__(self):
        super().__init__('pose_3d_estimation_node')

        self.camera_intrinsics = None
        self._camera_frame = None

        self.create_subscription(CameraInfo, '/camera/hand/camera_info', self._on_camera_info, 1)
        self.create_subscription(TargetInfoMessage, 'target_info', self._on_target_info, 1)
        self.target_pose_pub = self.create_publisher(TargetPose3D, 'target_3d', 1)

    def _on_camera_info(self, msg: CameraInfo):
        self.camera_intrinsics = CameraIntrinsics(msg)
        self._camera_frame = msg.header.frame_id

    def _on_target_info(self, msg: TargetInfoMessage):
        t0 = time.monotonic()
        if self.camera_intrinsics is None:
            self.get_logger().warn("Camera intrinsics non ancora ricevuti.", throttle_duration_sec=2.0)
            return

        x1, y1, x2, y2 = msg.bounding_box
        u, v = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        x, y, z = deproject_pixel_to_point(
            u, v, msg.depth_m, self.camera_intrinsics.fx, self.camera_intrinsics.fy,
            self.camera_intrinsics.cx, self.camera_intrinsics.cy)
        yaw = math.atan2(u - self.camera_intrinsics.cx, self.camera_intrinsics.fx)
        
        
        self.get_logger().info(
            f"Pubblicando targer_pose_3d: "
            f"({x:.2f}, {y:.2f}, {z:.2f}), yaw={math.degrees(yaw):.1f}°")

        out = TargetPose3D()
        out.header = msg.header
        out.header.frame_id = self._camera_frame
        out.position.x, out.position.y, out.position.z = x, y, z
        out.yaw = yaw
        self.target_pose_pub.publish(out)
        self.get_logger().warn(f"[Pose3DEstimationNode] round-trip: {(time.monotonic() - t0) * 1000:.0f} ms")


def main():
    rclpy.init()
    node = Pose3DEstimationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()