import sys
import os
import threading
from time import time
import rclpy
from rclpy.node import Node    
from yolo_detector.detector import Detector
from ament_index_python.packages import get_package_share_directory
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.wait_for_message import wait_for_message
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np
from cv_bridge import CvBridge
from detection_service_msg.srv import Detection

class DetectorNode(Node):
    
    def __init__(self):
        super().__init__("detection_node")
        
        # checkpoint file name, looked up in <share>/yolo_detector/yolo_checkpoint
        self.declare_parameter("checkpoint", "tennis_ball.pt")
        # optional full path to the YOLO model, overrides "checkpoint" when set
        self.declare_parameter("model_path", "")
        self.declare_parameter("camera_name", "/camera/hand/compressed")
        # inference device: "cuda:0", "cuda:1", ... or "cpu"
        self.declare_parameter("device", "cuda:0")
        # directory where received images are saved; default is the package source dir,
        # which is bind-mounted from the host so the files are visible outside the container
        self.declare_parameter("output_dir", "/home/yolo_ws/src/autonomous_manipulation_app/yolo_detector/tmp")
        # minimum confidence for a detection to be reported
        self.declare_parameter("conf_threshold", 0.5)
        # class returned by the "detection" service
        self.declare_parameter("target_class", "tennis_ball")
        # if true the service returns bbox/center normalized in [0, 1], otherwise in pixels
        self.declare_parameter("normalized", False)

        
        self.checkpoint = self.get_parameter("checkpoint").get_parameter_value().string_value
        self.model_path = self.get_parameter("model_path").get_parameter_value().string_value
        if not self.model_path:
            self.model_path = os.path.join(
                get_package_share_directory("yolo_detector"), "yolo_checkpoint", self.checkpoint)
        
        self.device = self.get_parameter("device").get_parameter_value().string_value
        self.conf_threshold = self.get_parameter("conf_threshold").get_parameter_value().double_value
        self.target_class = self.get_parameter("target_class").get_parameter_value().string_value
        self.normalized = self.get_parameter("normalized").get_parameter_value().bool_value
        
        self.detector = Detector(self.model_path, device=self.device)
        self.get_logger().info(f"[{self.get_name()}] DetectorNode initialized with model: {self.model_path} on device: {self.device}")
        self.get_logger().info(f"[{self.get_name()}] Model classes: {self.detector.model.names}, target class: {self.target_class}")
        
        # verify whether camera_name topic exists  and it is publishing compressed images
        self.camera_name = self.get_parameter("camera_name").get_parameter_value().string_value
        self.get_logger().info(f"[{self.get_name()}] Verifying camera topic: {self.camera_name}")
        
        camera_received = False
        while camera_received is False:
            self.get_logger().info(f"[{self.get_name()}] Waiting for camera topic: {self.camera_name} to be available...")
            camera_received, msg = wait_for_message(CompressedImage, self, self.camera_name, time_to_wait=5.0)
            
        # Subscribe to the camera topic
        self.subscription = self.create_subscription(
            CompressedImage,
            self.camera_name,
            self.image_callback,
            1)
        self.br = CvBridge()
        
        self.tmp_dir = self.get_parameter("output_dir").get_parameter_value().string_value
        os.makedirs(self.tmp_dir, exist_ok=True)
        self.get_logger().info(f"[{self.get_name()}] Saving images to: {self.tmp_dir}")
        
        # create a Publisher for the detection results
        self.publisher = self.create_publisher(CompressedImage, f"{self.camera_name}/detections", 1) 
        
        # best target detection of the last processed frame, None until the first frame is processed;
        # shared between image_callback and the service callback, guarded by a lock
        self.last_detection = None
        self.frame_processed = False
        self.last_frame_id = ""
        self.last_stamp = None
        self.detection_lock = threading.Lock()
        
        # Detection service server; it has its own callback group so it can answer
        # while image_callback is busy running YOLO
        self.service = self.create_service(
            Detection,
            "detection",
            self.detection_callback,
            callback_group=MutuallyExclusiveCallbackGroup())
        self.get_logger().info(f"[{self.get_name()}] Detection service ready on: {self.service.srv_name}")
        

    def image_callback(self, msg):
        # Convert the ROS CompressedImage message to a format suitable for YOLO detection
        # This is a placeholder for actual image conversion code
        self.get_logger().info(f"[{self.get_name()}] Received image from {self.camera_name}")
        
        # save the image to a temporary file for YOLO detection
        frame = self.br.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        cv2.imwrite(os.path.join(self.tmp_dir, f"image.jpg"), frame)
        
        # run detection on the image
        self.get_logger().info(f"[{self.get_name()}] Running detection on the image")
        results = self.detector.detect(frame)
        if not results:
            self.get_logger().warn(f"[{self.get_name()}] Detector returned no results")
            with self.detection_lock:
                self.last_detection = None
                self.frame_processed = True
                self.last_frame_id = msg.header.frame_id
                self.last_stamp = msg.header.stamp
            return
        r = results[0]

        # save the annotated image so it can be inspected from the host
        cv2.imwrite(os.path.join(self.tmp_dir, "detection.jpg"), r.plot())

        # keep only detections above the confidence threshold
        detections = []
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < self.conf_threshold:
                continue
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cx, cy, w, h = box.xywh[0].tolist()
            nx1, ny1, nx2, ny2 = box.xyxyn[0].tolist()
            ncx, ncy, _, _ = box.xywhn[0].tolist()
            detections.append({
                "class": r.names[cls_id],
                "conf": conf,
                "bbox": (x1, y1, x2, y2),
                "center": (cx, cy),
                "size": (w, h),
                "bbox_n": (nx1, ny1, nx2, ny2),
                "center_n": (ncx, ncy),
            })

        # keep the best target detection of this frame for the service (None if the ball is not there)
        targets = [d for d in detections if self.normalize_class(d["class"]) == self.normalize_class(self.target_class)]
        best = max(targets, key=lambda d: d["conf"]) if targets else None
        with self.detection_lock:
            self.last_detection = best
            self.frame_processed = True
            self.last_frame_id = msg.header.frame_id
            self.last_stamp = msg.header.stamp

        if not detections:
            self.get_logger().info(f"[{self.get_name()}] No detections above conf {self.conf_threshold:.2f}")
            return

        self.get_logger().info(f"[{self.get_name()}] {len(detections)} detection(s):")
        for i, d in enumerate(sorted(detections, key=lambda d: d["conf"], reverse=True)):
            x1, y1, x2, y2 = d["bbox"]
            cx, cy = d["center"]
            w, h = d["size"]
            self.get_logger().info(
                f"[{self.get_name()}]   #{i} {d['class']} conf={d['conf']:.2f} "
                f"center=({cx:.0f}, {cy:.0f}) size=({w:.0f}x{h:.0f}) "
                f"bbox=({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")


    @staticmethod
    def normalize_class(name):
        # "Tennis-Ball", "tennis ball" and "tennis_ball" are treated as the same class
        return name.lower().replace("-", "_").replace(" ", "_")

    def detection_callback(self, request, response):
        with self.detection_lock:
            best = self.last_detection
            frame_processed = self.frame_processed
            frame_id = self.last_frame_id
            stamp = self.last_stamp

        if best is None:
            response.success = False
            response.normalized = self.normalized
            response.message = (f"No {self.target_class} in the last frame" if frame_processed
                                else "No frame processed yet")
            response.class_name = ""
            response.frame_id = frame_id
            if stamp is not None:
                response.stamp = stamp
            response.confidence = -1.0
            response.x1 = response.y1 = response.x2 = response.y2 = -1.0
            response.center_x = response.center_y = -1.0
        else:
            x1, y1, x2, y2 = best["bbox_n"] if self.normalized else best["bbox"]
            cx, cy = best["center_n"] if self.normalized else best["center"]
            response.success = True
            response.normalized = self.normalized
            response.message = f"{best['class']} detected"
            response.class_name = best["class"]
            response.frame_id = frame_id
            if stamp is not None:
                response.stamp = stamp
            response.confidence = best["conf"]
            response.x1, response.y1, response.x2, response.y2 = x1, y1, x2, y2
            response.center_x, response.center_y = cx, cy

        self.get_logger().info(f"[{self.get_name()}] Detection service called: success={response.success}, {response.message}")
        return response


def main(args=None):
    rclpy.init(args=args)
    node = DetectorNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)


    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
