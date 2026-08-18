#!/usr/bin/env python3
"""
tracking_fsm.py (demo_package)

RISCRITTURA: design a CAMERA SINGOLA (solo braccio). front_localizer.py e
le camere frontali non servono piu' — tutto (ROI, detection, distanza)
avviene qui, su un'unica camera.

Stati: INIT -> WAITING_TRIGGER -> SEARCH -> TRACKING (per ora placeholder,
non fa nulla — verra' riempito in un passo successivo).

Flusso in SEARCH, ad ogni frame RGB della camera del braccio:
  1. Calcola il rettangolo di crop corrispondente alla ROI (apertura FOV +
     range + altezza, gli stessi parametri di common.py) proiettando gli
     angoli del volume 3D sulla camera — SOLO geometria/TF, nessuna depth
     ancora coinvolta qui.
  2. Ritaglia l'immagine RGB su quel rettangolo (non manda piu' l'immagine
     intera al servizio Detect: gli manda il crop).
  3. Il servizio Detect gira SOLO sul crop — le detection tornano in
     coordinate LOCALI al crop, le ritraduciamo in coordinate dell'immagine
     intera prima di usarle altrove.
  4. Per ogni box: legge la depth ToF della camera del braccio (riscalata
     alla risoluzione della depth se diversa da quella RGB — tipico dei
     sensori ToF) e verifica se la distanza e' nel raggio d'azione
     definito (CONE_MIN_RANGE/CONE_MAX_RANGE).
  5. Se c'e' ESATTAMENTE un box valido, e resta stabile (~stessa posizione)
     per `stability_frames_required` frame consecutivi, il target viene
     agganciato -> TRACKING.

Il topic di debug e' visibile in TUTTI gli stati (anche INIT/WAITING_TRIGGER),
non solo dopo l'avvio di SEARCH — cosi' puoi verificare a occhio la ROI e la
detection prima ancora di premere invio.
"""

import math
import threading
import time

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from geometry_msgs.msg import PoseStamped

from demo_package.common import (
    CameraIntrinsics, box_center, distance,
    extract_appearance_embedding, embedding_similarity,
    compute_roi_crop_rect, scale_box_to_depth, box_center_depth,
    CONE_FOV_DEG, CONE_MIN_RANGE, CONE_MAX_RANGE,
)
from demo_package.detect_client import DetectClient

INIT = "init"
WAITING_TRIGGER = "waiting_trigger"
SEARCH = "search"
TRACKING = "tracking"

# ============================================================
# Valori scritti qui, nessun argomento da terminale.
# ============================================================
HAND_RGB_TOPIC = '/out/compressed'
HAND_RGB_COMPRESSED = True   # True su bag, False sul robot vero se manca il nodo di ricompressione
HAND_CAMERA_INFO_TOPIC = '/camera/hand/camera_info'
HAND_DEPTH_TOPIC = '/depth/hand/image'   # depth ToF della camera del braccio — verifica il nome reale
GOAL_FRAME = 'map'
GOAL_UPDATE_TOPIC = 'goal_update'
DETECT_SERVICE = 'detect'
TARGET_CLASSES = ['person']
DEBUG_IMAGE_TOPIC = '/person_follow/hand_debug/compressed'  # suffisso /compressed: convenzione
                                                               # image_transport, stessa di /camera/hand/compressed

REID_SIMILARITY_THRESHOLD = 0.5
REID_EMA_ALPHA = 0.3
STABILITY_FRAMES_REQUIRED = 10    # frame consecutivi stabili prima di agganciare
STABILITY_PX_TOLERANCE = 40.0
LOST_FRAMES_THRESHOLD = 10        # frame consecutivi senza un box valido prima di tornare in SEARCH
STOPPING_DISTANCE = 1.0


class TrackingFSM(Node):
    def __init__(self):
        super().__init__('tracking_fsm')

        self.goal_frame = GOAL_FRAME
        self.fov_rad = math.radians(CONE_FOV_DEG)
        self.min_range = CONE_MIN_RANGE
        self.max_range = CONE_MAX_RANGE

        self.bridge = CvBridge()
        self.intrinsics = None
        self._hand_optical_frame = None
        self._latest_depth_image = None
        self._last_image_cb_time = None

        # "Blocchiamo tutto": camera_info, depth e image sono nello stesso
        # gruppo — quando _image_cb resta bloccata ad aspettare la risposta
        # del servizio Detect (call_sync, sincrona), anche gli aggiornamenti
        # di depth/camera_info si accodano e aspettano il loro turno.
        #
        # Il client del servizio DEVE stare in un gruppo diverso — altrimenti
        # la risposta non potrebbe mai essere elaborata mentre _image_cb e'
        # bloccata ad aspettarla (deadlock, non solo un rallentamento).
        self._main_group = MutuallyExclusiveCallbackGroup()
        self._service_group = ReentrantCallbackGroup()

        self.detect_client = DetectClient(self, DETECT_SERVICE, callback_group=self._service_group)

        self.state = INIT
        self.reference_embedding = None
        self.lost_frames = 0
        self.last_target_base = None
        self.last_published_pos = None
        self.last_published_time = 0.0
        self._stability_count = 0
        self._stability_box_center = None
        self._locked_box = None  # ultimo box agganciato, ridisegnato (statico) durante TRACKING

        self._manual_trigger_received = False
        self._stdin_thread = threading.Thread(target=self._wait_for_manual_trigger, daemon=True)
        self._stdin_thread.start()

        self.create_subscription(CameraInfo, HAND_CAMERA_INFO_TOPIC, self._camera_info_cb,
                                  qos_profile_sensor_data, callback_group=self._main_group)
        self.create_subscription(Image, HAND_DEPTH_TOPIC, self._depth_cb,
                                  qos_profile_sensor_data, callback_group=self._main_group)

        hand_rgb_msg_type = CompressedImage if HAND_RGB_COMPRESSED else Image
        self.create_subscription(hand_rgb_msg_type, HAND_RGB_TOPIC, self._image_cb,
                                  qos_profile_sensor_data, callback_group=self._main_group)

        self.goal_pub = self.create_publisher(PoseStamped, GOAL_UPDATE_TOPIC, 5)
        self.debug_pub = self.create_publisher(CompressedImage, DEBUG_IMAGE_TOPIC, 5)

        self.get_logger().info(
            f"TrackingFSM avviato (design camera singola) — stato iniziale: INIT. "
            f"ROI: FOV={math.degrees(self.fov_rad):.0f}° (crop larghezza), "
            f"range=[{self.min_range:.2f},{self.max_range:.2f}]m (post-detection, via depth)")

    # ------------------------------------------------------------------
    def _wait_for_manual_trigger(self):
        try:
            input("\n>>> Premi INVIO in questo terminale per avviare SEARCH...\n")
        except EOFError:
            pass
        self._manual_trigger_received = True

    def _camera_info_cb(self, msg):
        if self.intrinsics is None:
            self.intrinsics = CameraIntrinsics(msg)
            self.get_logger().info(
                f"Intrinseci camera braccio: fx={self.intrinsics.fx:.1f} fy={self.intrinsics.fy:.1f} "
                f"cx={self.intrinsics.cx:.1f} cy={self.intrinsics.cy:.1f}")

    def _depth_cb(self, depth_msg):
        self._latest_depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')

    # ------------------------------------------------------------------
    # Dispatch per stato — il debug e' costruito e pubblicato SEMPRE
    # (tutti gli stati), la detection gira SOLO in SEARCH.
    # ------------------------------------------------------------------
    def _image_cb(self, rgb_msg):
        t_start = time.monotonic()
        since_last = (t_start - self._last_image_cb_time) * 1000.0 if self._last_image_cb_time else -1.0
        self._last_image_cb_time = t_start

        if HAND_RGB_COMPRESSED:
            frame_bgr = self.bridge.compressed_imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        else:
            frame_bgr = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
        t_decode = time.monotonic()

        self._hand_optical_frame = rgb_msg.header.frame_id
        h_img, w_img = frame_bgr.shape[:2]

        publish_debug = self.debug_pub.get_subscription_count() > 0
        debug_frame = frame_bgr.copy() if publish_debug else None

        crop_rect = None
        if self.intrinsics is not None:
            crop_rect = compute_roi_crop_rect(w_img, h_img, self.intrinsics, self.fov_rad)

        if debug_frame is not None:
            if crop_rect is not None:
                cv2.rectangle(debug_frame, (crop_rect[0], crop_rect[1]), (crop_rect[2], crop_rect[3]),
                              (0, 200, 255), 2)
            cv2.putText(debug_frame, f"Stato: {self.state.upper()}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        t_crop_draw = time.monotonic()

        self.get_logger().info(
            f"[timing image_cb] dall'ultima chiamata={since_last:.0f}ms  decodifica={(t_decode - t_start) * 1000:.0f}ms  "
            f"crop+disegno={(t_crop_draw - t_decode) * 1000:.0f}ms  (stato={self.state}, crop_rect={crop_rect})",
            throttle_duration_sec=1.0)

        # ---- INIT: aspetta intrinseci + servizio Detect pronto ----
        if self.state == INIT:
            if self.intrinsics is not None and self.detect_client.client.service_is_ready():
                self.get_logger().info(
                    "INIT completato (intrinseci ricevuti, servizio Detect pronto) — "
                    "in attesa del trigger manuale.")
                self.state = WAITING_TRIGGER
            self._publish_debug(debug_frame, rgb_msg.header)
            self.get_logger().info(
                f"[timing image_cb] pubblicazione={(time.monotonic() - t_crop_draw) * 1000:.0f}ms",
                throttle_duration_sec=1.0)
            return

        # ---- WAITING_TRIGGER: aspetta INVIO ----
        if self.state == WAITING_TRIGGER:
            if self._manual_trigger_received:
                self.get_logger().info("Trigger manuale ricevuto — passo a SEARCH.")
                self.state = SEARCH
            self._publish_debug(debug_frame, rgb_msg.header)
            self.get_logger().info(
                f"[timing image_cb] pubblicazione={(time.monotonic() - t_crop_draw) * 1000:.0f}ms",
                throttle_duration_sec=1.0)
            return

        # ---- TRACKING: stessa chiamata di SEARCH, ma la risposta viene gestita diversamente ----
        # ---- SEARCH: qui gira davvero la detection, sul crop — CHIAMATA BLOCCANTE ----
        if crop_rect is None:
            self.get_logger().info(
                f"{self.state.upper()}: ROI non calcolabile questo frame (TF non disponibile) — "
                f"nessuna detection.", throttle_duration_sec=2.0)
            self._publish_debug(debug_frame, rgb_msg.header)
            return

        x1, y1, x2, y2 = crop_rect
        crop_bgr = frame_bgr[y1:y2, x1:x2]
        crop_msg = self.bridge.cv2_to_imgmsg(crop_bgr, encoding='bgr8')
        crop_msg.header = rgb_msg.header

        # BLOCCANTE: _image_cb non ritorna finche' non arriva la risposta.
        # Durante l'attesa, camera_info/depth restano in coda (stesso gruppo
        # di callback) — e' la scelta "blocchiamo tutto" che avevi chiesto.
        response = self.detect_client.call_sync(crop_msg, target_classes=TARGET_CLASSES)

        try:
            if self.state == SEARCH:
                self._handle_search_response(response, frame_bgr, crop_rect, debug_frame, rgb_msg.header)
            else:  # TRACKING
                self._handle_track_response(response, frame_bgr, crop_rect, debug_frame, rgb_msg.header)
        except Exception as ex:
            self.get_logger().error(f"Eccezione nell'elaborazione della risposta: {ex}", throttle_duration_sec=2.0)
            self._publish_debug(debug_frame, rgb_msg.header)

    # ------------------------------------------------------------------
    # SEARCH
    # ------------------------------------------------------------------
    def _handle_search_response(self, response, frame_bgr, crop_rect, debug_frame, header):
        if response is None:
            self._publish_debug(debug_frame, header)
            return

        crop_x1, crop_y1, _, _ = crop_rect
        depth_image = self._latest_depth_image

        valid_boxes = []  # (box_in_full_image_coords, dist)
        for det in response.detections:
            # le coordinate tornano relative al CROP: le ritraduciamo nell'immagine intera
            box_full = (det.x1 + crop_x1, det.y1 + crop_y1, det.x2 + crop_x1, det.y2 + crop_y1)

            if depth_image is None:
                if debug_frame is not None:
                    self._draw_box(debug_frame, box_full, (0, 255, 255), f"{det.class_name} (depth n/d)")
                continue

            box_depth = scale_box_to_depth(box_full, frame_bgr.shape, depth_image.shape)
            dist = box_center_depth(depth_image, box_depth)
            if dist is None:
                if debug_frame is not None:
                    self._draw_box(debug_frame, box_full, (128, 128, 128), f"{det.class_name} (depth invalida)")
                continue

            in_range = self.min_range <= dist <= self.max_range
            if not in_range:
                if debug_frame is not None:
                    self._draw_box(debug_frame, box_full, (0, 0, 220), f"{det.class_name} {dist:.2f}m (fuori range)")
                continue

            # NIENTE disegno qui: lo facciamo una volta sola piu' avanti, con
            # l'etichetta finale gia' completa (distanza + stato di stabilita'
            # o TARGET) — disegnarlo anche qui e poi di nuovo dopo avrebbe
            # sovrapposto due scritte nello stesso punto.
            valid_boxes.append((box_full, dist))

        if len(valid_boxes) != 1:
            if debug_frame is not None:
                for box_full, dist in valid_boxes:
                    self._draw_box(debug_frame, box_full, (0, 200, 0), f"person {dist:.2f}m")
            self.get_logger().info(
                f"SEARCH in attesa: {len(valid_boxes)} box nel raggio d'azione su {len(response.detections)} "
                f"rilevati nel crop (serve esattamente 1).", throttle_duration_sec=2.0)
            self._stability_count = 0
            self._stability_box_center = None
            self._publish_debug(debug_frame, header)
            return

        box_full, dist = valid_boxes[0]
        box_center_now = box_center(box_full)

        if (self._stability_box_center is not None
                and distance(box_center_now, self._stability_box_center) <= STABILITY_PX_TOLERANCE):
            self._stability_count += 1
        else:
            self._stability_count = 1
        self._stability_box_center = box_center_now

        self.get_logger().info(
            f"SEARCH: 1 box nel raggio d'azione a {dist:.2f}m — "
            f"stabilita' {self._stability_count}/{STABILITY_FRAMES_REQUIRED}",
            throttle_duration_sec=1.0)

        if self._stability_count < STABILITY_FRAMES_REQUIRED:
            if debug_frame is not None:
                self._draw_box(debug_frame, box_full, (0, 200, 0),
                                f"person {dist:.2f}m ({self._stability_count}/{STABILITY_FRAMES_REQUIRED})")
            self._publish_debug(debug_frame, header)
            return

        # Stabile per abbastanza frame: aggancio.
        self.reference_embedding = extract_appearance_embedding(frame_bgr, box_full)
        self._locked_box = box_full
        self.state = TRACKING
        self.lost_frames = 0
        self._stability_count = 0
        self._stability_box_center = None
        self.get_logger().info(f"Target agganciato a {dist:.2f}m (box={box_full}) — passo a TRACKING.")
        if debug_frame is not None:
            self._draw_box(debug_frame, box_full, (0, 255, 0), f"TARGET {dist:.2f}m")
        self._publish_debug(debug_frame, header)

    # ------------------------------------------------------------------
    # TRACKING
    # ------------------------------------------------------------------
    def _handle_track_response(self, response, frame_bgr, crop_rect, debug_frame, header):
        if response is None:
            self._register_lost_frame(debug_frame, header)
            return

        crop_x1, crop_y1, _, _ = crop_rect
        depth_image = self._latest_depth_image

        best_box, best_score, best_dist = None, -float("inf"), None
        candidates = []  # (box_full, dist, similarity) — solo quelli che passano entrambi i filtri

        for det in response.detections:
            box_full = (det.x1 + crop_x1, det.y1 + crop_y1, det.x2 + crop_x1, det.y2 + crop_y1)

            if depth_image is None:
                if debug_frame is not None:
                    self._draw_box(debug_frame, box_full, (0, 255, 255), f"{det.class_name} (depth n/d)")
                continue
            box_depth = scale_box_to_depth(box_full, frame_bgr.shape, depth_image.shape)
            dist = box_center_depth(depth_image, box_depth)
            if dist is None or not (self.min_range <= dist <= self.max_range):
                if debug_frame is not None:
                    self._draw_box(debug_frame, box_full, (0, 165, 255), f"{det.class_name} (fuori range)")
                continue

            b_embedding = extract_appearance_embedding(frame_bgr, box_full)
            similarity = (1.0 if self.reference_embedding is None
                          else embedding_similarity(b_embedding, self.reference_embedding))
            if similarity < REID_SIMILARITY_THRESHOLD:
                if debug_frame is not None:
                    self._draw_box(debug_frame, box_full, (0, 0, 220),
                                    f"{det.class_name} {dist:.2f}m (aspetto {similarity:.2f})")
                continue

            # NIENTE disegno qui: lo facciamo una volta sola dopo, cosi' il
            # migliore prende l'etichetta TARGET senza sovrapporsi a un'altra
            # scritta gia' disegnata nello stesso punto.
            candidates.append((box_full, dist, similarity))
            if similarity > best_score:
                best_score, best_box, best_dist = similarity, box_full, dist

        if debug_frame is not None:
            for box_full, dist, similarity in candidates:
                if box_full == best_box:
                    label = f"TARGET {dist:.2f}m"
                else:
                    label = f"person {dist:.2f}m score={similarity:.2f}"
                self._draw_box(debug_frame, box_full, (0, 255, 0), label)

        if best_box is not None:
            self.lost_frames = 0
            self._locked_box = best_box

            b_embedding = extract_appearance_embedding(frame_bgr, best_box)
            if b_embedding is not None and self.reference_embedding is not None:
                self.reference_embedding = (
                    (1 - REID_EMA_ALPHA) * self.reference_embedding + REID_EMA_ALPHA * b_embedding)
                self.reference_embedding = cv2.normalize(
                    self.reference_embedding, None, alpha=1.0, norm_type=cv2.NORM_L1).flatten()

            self.get_logger().info(
                f"TRACKING ok: box confermato a {best_dist:.2f}m, similarity={best_score:.2f}",
                throttle_duration_sec=1.0)
            self._publish_debug(debug_frame, header)
        else:
            self._register_lost_frame(debug_frame, header)

    def _register_lost_frame(self, debug_frame, header):
        self.lost_frames += 1
        self.get_logger().info(
            f"TRACKING: nessun box valido questo frame (lost_frames={self.lost_frames}/{LOST_FRAMES_THRESHOLD})",
            throttle_duration_sec=1.0)
        if self.lost_frames >= LOST_FRAMES_THRESHOLD:
            self.get_logger().info("Target perso — torno in SEARCH (nessun trigger manuale richiesto di nuovo).")
            self.state = SEARCH
            self.reference_embedding = None
            self._locked_box = None
            self._stability_count = 0
            self._stability_box_center = None
        self._publish_debug(debug_frame, header)

    # ------------------------------------------------------------------
    @staticmethod
    def _draw_box(frame, box, color, label):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    def _publish_debug(self, debug_frame, header):
        if debug_frame is None:
            return
        msg = self.bridge.cv2_to_compressed_imgmsg(debug_frame, dst_format='jpg')
        msg.header = header
        self.debug_pub.publish(msg)


def main():
    rclpy.init()
    node = TrackingFSM()
    # MultiThreadedExecutor OBBLIGATORIO qui, non facoltativo: e' quello che
    # permette al gruppo del client (self._service_group) di elaborare la
    # risposta del servizio Detect su un thread diverso da quello bloccato
    # in call_sync() dentro _image_cb (gruppo self._main_group). Con lo
    # SingleThreadedExecutor di rclpy.spin() la risposta non arriverebbe mai.
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()