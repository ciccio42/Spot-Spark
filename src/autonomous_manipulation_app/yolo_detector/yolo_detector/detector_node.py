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
from geometry_msgs.msg import PointStamped
<<<<<<< HEAD
=======
from std_msgs.msg import String
from rclpy.qos import QoSProfile, DurabilityPolicy
>>>>>>> 1bf5e2586f1b13ad7d75c44d7ada20f636e0069f
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
        # live cv2 window with the camera image, the detections and the target sent by the service
        self.declare_parameter("show_image", True)
        self.declare_parameter("window_name", "hand camera - detections")
<<<<<<< HEAD
        # a click is accepted as a confirmation only if a client asked for one in the last
        # confirmation_request_window seconds (the FSM asks continuously while waiting)
        self.declare_parameter("confirmation_request_window", 2.0)
        # pixel the FSM sends to Manipulation, drawn in the "pick target" window
        self.declare_parameter("pick_target_topic", "/fsm_node/pick_target")
=======
        # left click on the window publishes the clicked pixel (PointStamped, x/y in pixels, header of the
        # shown image) on this topic; the FSM uses it as target when the YOLO detection fails (click fallback)
        self.declare_parameter("clicked_pixel_topic", "clicked_pixel")
        # seconds the last clicked point stays drawn on the window
        self.declare_parameter("click_marker_duration", 3.0)
        # what the FSM is waiting a click for (std_msgs/String, "" = nothing), drawn as a banner on the window
        self.declare_parameter("click_request_topic", "click_request")
>>>>>>> 1bf5e2586f1b13ad7d75c44d7ada20f636e0069f

        
        self.checkpoint = self.get_parameter("checkpoint").get_parameter_value().string_value
        self.model_path = self.get_parameter("model_path").get_parameter_value().string_value
        if not self.model_path:
            self.model_path = os.path.join(
                get_package_share_directory("yolo_detector"), "yolo_checkpoint", self.checkpoint)
        
        self.device = self.get_parameter("device").get_parameter_value().string_value
        self.conf_threshold = self.get_parameter("conf_threshold").get_parameter_value().double_value
        self.target_class = self.get_parameter("target_class").get_parameter_value().string_value
        self.normalized = self.get_parameter("normalized").get_parameter_value().bool_value
        self.show_image = self.get_parameter("show_image").get_parameter_value().bool_value
        self.window_name = self.get_parameter("window_name").get_parameter_value().string_value
<<<<<<< HEAD
        self.confirmation_request_window = (
            self.get_parameter("confirmation_request_window").get_parameter_value().double_value)
=======
        self.clicked_pixel_topic = self.get_parameter("clicked_pixel_topic").get_parameter_value().string_value
        self.click_marker_duration = self.get_parameter("click_marker_duration").get_parameter_value().double_value
>>>>>>> 1bf5e2586f1b13ad7d75c44d7ada20f636e0069f
        
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
        # pixel clicked on the cv2 window (published from the display thread)
        self.click_publisher = self.create_publisher(PointStamped, self.clicked_pixel_topic, 10)
        # request of the FSM shown on the window; transient local, like the FSM publisher
        self.click_request = ""
        self.create_subscription(
            String, self.get_parameter("click_request_topic").get_parameter_value().string_value,
            self.click_request_callback, QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
            callback_group=MutuallyExclusiveCallbackGroup())
        
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

        # cv2 window: image_callback only stores the frame to show, a dedicated thread owns the window
        # (imshow/waitKey always from the same thread, never blocking the ROS callbacks)
        self.display_frame = None
<<<<<<< HEAD
        # targets / frame_id / stamp of the frame currently shown in the window (for the mouse clicks)
        self.shown_frame = None
=======
        # (frame_id, stamp) of display_frame, and of the frame currently shown: a click refers to the latter
        self.display_header = None
        self.shown_header = None
        # last clicked pixel (u, v) and the time.time() of the click, drawn on the window for a while
        self.last_click = None
>>>>>>> 1bf5e2586f1b13ad7d75c44d7ada20f636e0069f
        self.display_lock = threading.Lock()
        # operator confirmation: detection clicked in the window, returned once by the service
        # (guarded by detection_lock); time of the last service call asking for a confirmation
        self.confirmed = None
        self.confirmation_requested_at = 0.0
        # "pick target" window: frozen image of the last confirmation (bbox center, click) and the pixel
        # the FSM sent to Manipulation; redrawn only when pick_view_dirty is set (guarded by display_lock)
        self.pick_snapshot = None
        self.sent_target = None
        self.pick_view_dirty = False
        self.create_subscription(PointStamped,
                                 self.get_parameter("pick_target_topic").get_parameter_value().string_value,
                                 self.pick_target_callback, 10)
        self.display_stop = threading.Event()
        self.display_thread = None
        if self.show_image:
            if not os.environ.get("DISPLAY"):
                self.get_logger().warn(f"[{self.get_name()}] show_image is true but DISPLAY is not set: "
                                       f"no window (export DISPLAY=:2 and xhost +local:docker on the host)")
            else:
                self.display_thread = threading.Thread(target=self.display_loop, daemon=True)
                self.display_thread.start()
        

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
<<<<<<< HEAD
            self.set_display_frame(frame, [], None, msg.header.frame_id, msg.header.stamp)
=======
            self.set_display_frame(frame, None, msg.header.frame_id, msg.header.stamp)
>>>>>>> 1bf5e2586f1b13ad7d75c44d7ada20f636e0069f
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

<<<<<<< HEAD
        self.set_display_frame(r.plot(), targets, best, msg.header.frame_id, msg.header.stamp)
=======
        self.set_display_frame(r.plot(), best, msg.header.frame_id, msg.header.stamp)
>>>>>>> 1bf5e2586f1b13ad7d75c44d7ada20f636e0069f

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


<<<<<<< HEAD
    def set_display_frame(self, image, targets, best, frame_id, stamp):
        """Annotates the image with the target returned by the service and hands it to the window,
        together with the frame data the mouse clicks refer to."""
=======
    def set_display_frame(self, image, best, frame_id, stamp):
        """Annotates the image with the target returned by the service and hands it to the window."""
>>>>>>> 1bf5e2586f1b13ad7d75c44d7ada20f636e0069f
        if self.display_thread is None:
            return
        view = image.copy()
        if best is not None:
            x1, y1, x2, y2 = (int(v) for v in best["bbox"])
            cx, cy = (int(v) for v in best["center"])
            cv2.rectangle(view, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.drawMarker(view, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
            label = f"TARGET {best['conf']:.2f} ({cx}, {cy})"
            status = (0, 255, 0)
        else:
            label = f"no {self.target_class}"
            status = (0, 0, 255)
        cv2.putText(view, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status, 2)
        cv2.putText(view, frame_id, (10, view.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        base = view.copy()

        # operator confirmation banner
        with self.detection_lock:
            confirmed = self.confirmed is not None
        if confirmed:
            banner, color = "CONFIRMED - robot moving on (right click = cancel)", (255, 255, 0)
        elif self.confirmation_pending():
            banner, color = "CLICK ON THE BALL TO CONFIRM", (0, 255, 255)
        else:
            banner = None
        if banner:
            cv2.putText(view, banner, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        with self.display_lock:
<<<<<<< HEAD
            self.display_frame = (view, {"targets": targets, "frame_id": frame_id, "stamp": stamp, "image": base})
=======
            self.display_frame = view
            self.display_header = (frame_id, stamp)
>>>>>>> 1bf5e2586f1b13ad7d75c44d7ada20f636e0069f

    def display_loop(self):
        """Owns the cv2 window: shows the latest annotated frame (~30 Hz) and handles the mouse clicks."""
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
<<<<<<< HEAD
            # clicks are delivered by waitKey below, i.e. in this thread
            cv2.setMouseCallback(self.window_name, self.on_mouse)
            while not self.display_stop.is_set():
                with self.display_lock:
                    item, self.display_frame = self.display_frame, None
                if item is not None:
                    view, meta = item
                    cv2.imshow(self.window_name, view)
                    with self.display_lock:
                        self.shown_frame = meta
                with self.display_lock:
                    dirty, self.pick_view_dirty = self.pick_view_dirty, False
                    snapshot, sent = self.pick_snapshot, self.sent_target
                if dirty and snapshot is not None:
                    cv2.imshow(f"{self.window_name} - pick target", self.draw_pick_view(snapshot, sent))
=======
            cv2.setMouseCallback(self.window_name, self.mouse_callback)
            view = None
            while not self.display_stop.is_set():
                with self.display_lock:
                    new_view, self.display_frame = self.display_frame, None
                    if new_view is not None:
                        view = new_view
                        self.shown_header = self.display_header
                    click = self.last_click
                request = self.click_request
                if view is not None:
                    shown = view
                    # banner with what the FSM is waiting for: any left click on the window is the OK
                    if request:
                        shown = view.copy()
                        h, w = shown.shape[:2]
                        cv2.rectangle(shown, (0, h - 70), (w, h - 25), (0, 255, 255), -1)
                        cv2.putText(shown, f"CLICK = OK: {request}", (10, h - 38),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                    # the clicked point is drawn on a copy, so it disappears after click_marker_duration
                    if click is not None and time() - click[2] < self.click_marker_duration:
                        shown = shown.copy() if shown is view else shown
                        cv2.circle(shown, (click[0], click[1]), 12, (255, 0, 255), 2)
                        cv2.drawMarker(shown, (click[0], click[1]), (255, 0, 255), cv2.MARKER_CROSS, 24, 2)
                        cv2.putText(shown, f"CLICK ({click[0]}, {click[1]})", (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
                    cv2.imshow(self.window_name, shown)
>>>>>>> 1bf5e2586f1b13ad7d75c44d7ada20f636e0069f
                cv2.waitKey(30)
        except cv2.error as e:
            # e.g. opencv-python-headless (no GUI support) or no X server reachable
            self.get_logger().warn(f"[{self.get_name()}] cv2 window not available, image display disabled: {e}")
        finally:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

<<<<<<< HEAD
    def pick_target_callback(self, msg):
        """Pixel the FSM is sending to Manipulation: drawn in the "pick target" window."""
        with self.display_lock:
            self.sent_target = msg
            self.pick_view_dirty = True

    def draw_pick_view(self, snapshot, sent):
        """Frozen confirmed image with bbox center, operator click, their difference and the pixel sent
        to Manipulation."""
        view = snapshot["image"].copy()
        det = snapshot["det"]
        x1, y1, x2, y2 = (int(v) for v in det["bbox"])
        cx, cy = (int(round(v)) for v in det["center"])
        kx, ky = snapshot["click"]
        dx, dy = kx - cx, ky - cy

        cv2.rectangle(view, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.drawMarker(view, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 18, 2)        # bbox center: green
        cv2.circle(view, (kx, ky), 7, (255, 0, 255), 2)                             # click: magenta
        if (dx, dy) != (0, 0):
            cv2.arrowedLine(view, (cx, cy), (kx, ky), (0, 255, 255), 1, tipLength=0.3)

        lines = [
            (f"bbox center : ({cx}, {cy})", (0, 255, 0)),
            (f"click       : ({kx}, {ky})", (255, 0, 255)),
            (f"click-center: ({dx:+d}, {dy:+d}) = {np.hypot(dx, dy):.1f} px", (0, 255, 255)),
        ]
        if sent is None:
            lines.append(("to Manipulation: not sent yet", (200, 200, 200)))
        else:
            sx, sy = int(round(sent.point.x)), int(round(sent.point.y))
            same_image = (sent.header.stamp.sec == snapshot["stamp"].sec
                          and sent.header.stamp.nanosec == snapshot["stamp"].nanosec)
            cv2.drawMarker(view, (sx, sy), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 22, 2)  # sent: red
            source = ("= click" if (sx, sy) == (kx, ky) else
                      "= bbox center" if (sx, sy) == (cx, cy) else "")
            lines.append((f"to Manipulation: ({sx}, {sy}) {source}"
                          f"{'' if same_image else '  (OTHER IMAGE!)'}", (0, 0, 255)))

        # text block with a dark background, bottom left
        line_h, pad = 22, 6
        top = view.shape[0] - pad - line_h * len(lines) - pad
        cv2.rectangle(view, (0, top), (min(view.shape[1], 470), view.shape[0]), (0, 0, 0), -1)
        for i, (text, color) in enumerate(lines):
            cv2.putText(view, text, (pad, top + pad + line_h * (i + 1) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
        return view

    def confirmation_pending(self):
        """True if a client asked for an operator confirmation recently."""
        return time() - self.confirmation_requested_at <= self.confirmation_request_window

    def on_mouse(self, event, x, y, flags, param):
        """Left click on a detected ball = confirmation; right click = cancel the confirmation.
        x, y are in pixels of the shown image."""
        if event == cv2.EVENT_RBUTTONDOWN:
            with self.detection_lock:
                had = self.confirmed is not None
                self.confirmed = None
            if had:
                self.get_logger().info(f"[{self.get_name()}] Operator cancelled the confirmation")
            return
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if not self.confirmation_pending():
            self.get_logger().info(f"[{self.get_name()}] Click ignored: no confirmation requested right now")
            return
        with self.display_lock:
            shown = self.shown_frame
        if shown is None:
            return
        # the clicked ball: the most confident target whose bbox contains the click
        hits = [d for d in shown["targets"]
                if d["bbox"][0] <= x <= d["bbox"][2] and d["bbox"][1] <= y <= d["bbox"][3]]
        if not hits:
            self.get_logger().warn(f"[{self.get_name()}] Click at ({x}, {y}) is not on a detected "
                                   f"{self.target_class}: click inside its box")
            return
        det = max(hits, key=lambda d: d["conf"])
        with self.detection_lock:
            self.confirmed = {"det": det, "frame_id": shown["frame_id"], "stamp": shown["stamp"],
                              "click": (float(x), float(y))}
        with self.display_lock:
            self.pick_snapshot = {"image": shown["image"], "det": det, "click": (x, y),
                                  "frame_id": shown["frame_id"], "stamp": shown["stamp"]}
            self.sent_target = None
            self.pick_view_dirty = True
        cx, cy = det["center"]
        self.get_logger().info(f"[{self.get_name()}] Operator confirmed {det['class']} conf={det['conf']:.2f} "
                               f"center=({cx:.0f}, {cy:.0f}) click=({x}, {y})")

    def fill_response(self, response, det, frame_id, stamp, message, click=None):
        """Fills the Detection response with det (None = no ball: success false, values -1) and the
        operator click (pixels, -1 if none)."""
        response.click_x, response.click_y = click if click is not None else (-1.0, -1.0)
        response.normalized = self.normalized
        response.message = message
        response.frame_id = frame_id
        if stamp is not None:
            response.stamp = stamp
        if det is None:
            response.success = False
            response.class_name = ""
            response.confidence = -1.0
            response.x1 = response.y1 = response.x2 = response.y2 = -1.0
            response.center_x = response.center_y = -1.0
            return
        x1, y1, x2, y2 = det["bbox_n"] if self.normalized else det["bbox"]
        cx, cy = det["center_n"] if self.normalized else det["center"]
        response.success = True
        response.class_name = det["class"]
        response.confidence = det["conf"]
        response.x1, response.y1, response.x2, response.y2 = x1, y1, x2, y2
        response.center_x, response.center_y = cx, cy
=======
    def click_request_callback(self, msg):
        self.click_request = msg.data
        if msg.data:
            self.get_logger().info(f"[{self.get_name()}] FSM waiting for a click: {msg.data}")

    def mouse_callback(self, event, x, y, flags, param):
        """Left click on the window: publishes the pixel with the header of the image shown at that time
        (called from the display thread, inside cv2.waitKey)."""
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        with self.display_lock:
            header = self.shown_header
            self.last_click = (x, y, time())
        if header is None:
            self.get_logger().warn(f"[{self.get_name()}] Click at ({x}, {y}) ignored: no image shown yet")
            return
        msg = PointStamped()
        msg.header.frame_id, msg.header.stamp = header
        # pixel coordinates of the image (cv2 reports them in image space even if the window is resized)
        msg.point.x, msg.point.y, msg.point.z = float(x), float(y), 0.0
        self.click_publisher.publish(msg)
        self.get_logger().info(f"[{self.get_name()}] Clicked pixel ({x}, {y}) in {msg.header.frame_id}, "
                               f"published on {self.click_publisher.topic_name}")
>>>>>>> 1bf5e2586f1b13ad7d75c44d7ada20f636e0069f

    @staticmethod
    def normalize_class(name):
        # "Tennis-Ball", "tennis ball" and "tennis_ball" are treated as the same class
        return name.lower().replace("-", "_").replace(" ", "_")

    def detection_callback(self, request, response):
        if request.require_confirmation:
            return self.confirmed_detection_callback(response)
        # no operator click without confirmation
        response.click_x = response.click_y = -1.0

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

    def confirmed_detection_callback(self, response):
        """Service answer when the operator confirmation is required: the clicked detection, once."""
        self.confirmation_requested_at = time()
        with self.detection_lock:
            confirmed, self.confirmed = self.confirmed, None
            frame_id, stamp = self.last_frame_id, self.last_stamp
        if confirmed is None:
            if self.display_thread is None:
                message = ("Operator confirmation required but the detector window is disabled "
                           "(show_image false or DISPLAY not set)")
            else:
                message = f"Waiting for operator confirmation: click on the ball in the '{self.window_name}' window"
            self.fill_response(response, None, frame_id, stamp, message)
            self.get_logger().info(f"[{self.get_name()}] {message}", throttle_duration_sec=5.0)
            return response
        det = confirmed["det"]
        self.fill_response(response, det, confirmed["frame_id"], confirmed["stamp"],
                           f"{det['class']} confirmed by the operator", click=confirmed["click"])
        self.get_logger().info(f"[{self.get_name()}] Detection service: returning the confirmed {det['class']}")
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
        node.display_stop.set()
        if node.display_thread is not None:
            node.display_thread.join(timeout=1.0)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
