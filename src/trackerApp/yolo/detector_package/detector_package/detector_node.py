#!/usr/bin/env python3
"""
detector_node.py

ROS 2 node (lives in the "yolo" container): runs YoloEInference and exposes
the synchronous detector_interfaces/srv/Detect service. It knows nothing about
TF, depth, Nav2, cones or ROI — its ONLY responsibility is: "given a full
frame (+ classes), answer with the detections". Everything else (cone
projection, any filtering of the results, visualisation) is the job of
demo_package, which calls this service.
"""

import time

import cv_bridge
import rclpy
from rclpy.node import Node

from detector_interfaces.msg import BoxDetection
from detector_interfaces.srv import Detect

from detector_package.yoloe_inference import YoloEInference


class DetectorNode(Node):
    def __init__(self):
        super().__init__('detector_node')

        self.declare_parameter('model_path', '/models/yoloe-11s-seg.pt')
        self.declare_parameter('default_classes', ['person','quadruped','quadruped animal','quadruped robot','robotic dog','four-legged robot', 'dog', 'robot','umanoid robot'])
        self.declare_parameter('conf_threshold', 0.35)
        self.declare_parameter('imgsz', 640)
        self.declare_parameter('service_name', 'detect')
        self.declare_parameter('use_tracker', False)  # True: model.track() with BoT-SORT (persistent
                                                        # track_id); False: plain detect(), for
                                                        # comparison/debug — no track_id, always -1.
        self.declare_parameter('tracker_config', '/home/yolo_ws/src/yolo/detector_package/oc_sort.yaml')  # path of the tracker config
                                                        # file — 'botsort.yaml' uses the Ultralytics default
                                                        # one (with_reid: False); to enable ReID, point
                                                        # to the absolute path of your own copy with
                                                        # with_reid: True (e.g. /models/botsort_reid.yaml)
        self.declare_parameter('reid_model_name', 'osnet_x1_0')  # model name (torchreid library) —
                                                        # see torchreid.models.show_avai_models() for the
                                                        # available variants (osnet_x0_25 = lighter)
        self.declare_parameter('reid_model_path', '/home/yolo_ws/osnet/osnet_x1_0_imagenet.pth')  # LOCAL path of the pre-trained weights
                                                        # (e.g. /models/osnet_x1_0_market1501.pth) for
                                                        # OUR ReID (cosine comparison in demo_package,
                                                        # independent of BoT-SORT). Empty string = disabled,
                                                        # BoxDetection.embedding is always empty.

        gp = self.get_parameter
        self.default_classes = list(gp('default_classes').value)
        self.use_tracker = gp('use_tracker').value
        self.tracker_config = gp('tracker_config').value

        self.bridge = cv_bridge.CvBridge()

        model_path = gp('model_path').value
        reid_model_name = gp('reid_model_name').value
        reid_model_path = gp('reid_model_path').value or None
        self.get_logger().info(f"Carico YOLOE da {model_path}...")
        self.inference = YoloEInference(
            model_path=model_path,
            imgsz=gp('imgsz').value,
            conf_threshold=gp('conf_threshold').value,
            reid_model_name=reid_model_name if reid_model_path else None,
            reid_model_path=reid_model_path,
        )
        self.inference.set_classes(self.default_classes)
        self.get_logger().info(
            f"Modello caricato. Classi di default: {self.default_classes}. "
            f"Tracciamento: {'ATTIVO (BoT-SORT, model.track(), config=' + self.tracker_config + ')' if self.use_tracker else 'DISATTIVATO (model.predict())'}. "
            f"ReID: {'ATTIVO (' + reid_model_name + ', ' + reid_model_path + ')' if reid_model_path else 'DISATTIVATO'}. "
            f"Pronto a ricevere richieste.")

        self.srv = self.create_service(Detect, gp('service_name').value, self._handle_detect)

    def _handle_detect(self, request, response):
        self.get_logger().info(
            f"Richiesta detect ricevuta: immagine {request.image.width}x{request.image.height}, "
            f"classi={list(request.target_classes) or self.default_classes}, "
            f"reset_tracker={request.reset_tracker}")
        try:
            if request.reset_tracker and self.use_tracker:
                self.inference.reset_tracker()
                self.get_logger().info("Tracker azzerato su richiesta.")

            t0 = time.monotonic()
            frame_bgr = self.bridge.imgmsg_to_cv2(request.image, desired_encoding='bgr8')
            t1 = time.monotonic()
            classes = list(request.target_classes) if request.target_classes else self.default_classes
            if self.use_tracker:
                detections = self.inference.track(frame_bgr, classes=classes, tracker=self.tracker_config)
            else:
                detections = self.inference.detect(frame_bgr, classes=classes)
            t2 = time.monotonic()
            for d in detections:
                d.embedding = self.inference.extract_embedding(frame_bgr, d.box)
            t3 = time.monotonic()

            response.detections = [
                BoxDetection(
                    x1=float(d.box[0]), y1=float(d.box[1]),
                    x2=float(d.box[2]), y2=float(d.box[3]),
                    score=d.score, class_name=d.class_name, track_id=d.track_id,
                    embedding=(d.embedding.tolist() if d.embedding is not None else []),
                )
                for d in detections
            ]
            self.get_logger().info(
                f"[timing] decodifica={(t1 - t0) * 1000:.0f}ms  inferenza={(t2 - t1) * 1000:.0f}ms  "
                f"embedding={(t3 - t2) * 1000:.0f}ms ({len(detections)} box)  "
                f"(immagine {request.image.width}x{request.image.height})")

        except Exception as ex:
            # Do not let a corrupted frame or a model error bring the
            # service down: answer empty and log it, the caller
            # (DetectClient) already treats this case as "no detection
            # available for this frame".
            self.get_logger().error(f"Detection fallita: {ex}")
            response.detections = []

        self.get_logger().info(f"Richiesta detect completata: {len(response.detections)} detection.")
        return response


def main():
    rclpy.init()
    node = DetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()