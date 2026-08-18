#!/usr/bin/env python3
"""
detector_node.py

Nodo ROS2 (vive nel container "yolo"): fa girare YoloEInference ed espone il
servizio sincrono detector_interfaces/srv/Detect. Non sa nulla di TF, depth,
Nav2, coni o ROI — la sua UNICA responsabilita' e': "dato un frame intero
(+ classi), rispondi con le detection". Tutto il resto (proiezione del
cono, eventuale filtro sui risultati, visualizzazione) e' compito di
demo_package, che chiama questo servizio.
"""

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
        self.declare_parameter('default_classes', ['person'])
        self.declare_parameter('conf_threshold', 0.35)
        self.declare_parameter('imgsz', 640)
        self.declare_parameter('service_name', 'detect')

        gp = self.get_parameter
        self.default_classes = list(gp('default_classes').value)

        self.bridge = cv_bridge.CvBridge()

        model_path = gp('model_path').value
        self.get_logger().info(f"Carico YOLOE da {model_path}...")
        self.inference = YoloEInference(
            model_path=model_path,
            imgsz=gp('imgsz').value,
            conf_threshold=gp('conf_threshold').value,
        )
        self.inference.set_classes(self.default_classes)
        self.get_logger().info(
            f"Modello caricato. Classi di default: {self.default_classes}. Pronto a ricevere richieste.")

        self.srv = self.create_service(Detect, gp('service_name').value, self._handle_detect)

    def _handle_detect(self, request, response):
        self.get_logger().info(
            f"Richiesta detect ricevuta: immagine {request.image.width}x{request.image.height}, "
            f"classi={list(request.target_classes) or self.default_classes}")
        try:
            frame_bgr = self.bridge.imgmsg_to_cv2(request.image, desired_encoding='bgr8')
            classes = list(request.target_classes) if request.target_classes else self.default_classes
            detections = self.inference.detect(frame_bgr, classes=classes)

            response.detections = [
                BoxDetection(
                    x1=float(d.box[0]), y1=float(d.box[1]),
                    x2=float(d.box[2]), y2=float(d.box[3]),
                    score=d.score, class_name=d.class_name,
                )
                for d in detections
            ]

        except Exception as ex:
            # Non lasciare che un frame corrotto o un errore del modello
            # facciano cadere il servizio: si risponde vuoto e si logga, il
            # chiamante (DetectClient) tratta gia' questo caso come "nessuna
            # detection disponibile per questo frame".
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