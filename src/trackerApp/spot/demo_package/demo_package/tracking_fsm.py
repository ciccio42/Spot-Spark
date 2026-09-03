#!/usr/bin/env python3
"""
tracking_fsm.py (demo_package)

DUE METODI COMPLETAMENTE SEPARATI, non intrecciati — scelti con UN SOLO
switch in cima al file (TRACKING_METHOD). Non e' "quale similarita' uso
nel recovery": sono due architetture diverse, ciascuna con la propria
implementazione di SEARCH/TRACKING/RECOVERY, per poterle confrontare senza
che una contamini l'altra.

  "botsort_hsv": l'identita' del target si basa SOLO su BoT-SORT
      (track_id persistente, model.track()) + istogramma HSV come rete di
      sicurezza quando il track_id sparisce. Nessun embedding neurale
      coinvolto, in nessuno stato.

  "embedding_only": l'identita' si basa SOLO sull'embedding neurale
      (modello di classificazione separato, vedi embedding_model_path lato
      DetectorNode) — in TUTTI gli stati, SEARCH compreso. BoT-SORT/track_id
      non vengono mai usati per decidere chi e' il target (anche se il
      servizio li calcola comunque, semplicemente li ignoriamo).

Design comune a entrambi i metodi (invariato):
  - ROI come ritaglio dell'immagine (FOV via intrinseci, altezza da
    frazioni fisse — nessuna TF).
  - Distanza letta dalla depth ToF (patch centrata sul box).
  - Chiamata al servizio Detect SINCRONA/bloccante, executor multi-thread
    con gruppi di callback separati (client vs resto) per evitare deadlock.
  - TRACKING -> periodo di tolleranza a FRAME -> RECOVERY (timeout a
    TEMPO REALE) -> WAITING_TRIGGER se il target non si ritrova.
"""

import math
import threading
import time

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from geometry_msgs.msg import PoseStamped
from demo_interfaces.msg import TargetInfoMessage, TargetPose3D

from demo_package.common import (
    CameraIntrinsics, box_center, distance,
    extract_appearance_embedding, embedding_similarity,
    embedding_from_msg, neural_embedding_similarity,
    compute_roi_crop_rect, scale_box_to_depth, box_center_depth, rich_neural_embedding_similarity, deproject_pixel_to_point,
    CONE_FOV_DEG, CONE_MIN_RANGE, CONE_MAX_RANGE,
)
from demo_package.detect_client import DetectClient


INIT = "init"
WAITING_TRIGGER = "waiting_trigger"
SEARCH = "search"
TRACKING = "tracking"
RECOVERY = "recovery"

# ============================================================
# LO SWITCH — decide quale delle due architetture usare per l'intera
# sessione. Cambia questo, ricompila, testa; per confrontare i due
# metodi servono due sessioni separate, non uno switch a runtime.
# ============================================================
TRACKING_METHOD = "embedding_only"  # "botsort_hsv" oppure "embedding_only"

# ============================================================
# Valori scritti qui, nessun argomento da terminale.
# ============================================================
HAND_RGB_TOPIC = 'camera/hand/compressed'
HAND_RGB_COMPRESSED = True   # True su bag, False sul robot vero se manca il nodo di ricompressione
HAND_CAMERA_INFO_TOPIC = '/camera/hand/camera_info'
HAND_DEPTH_TOPIC = '/depth/hand/image'   # depth ToF della camera del braccio — verifica il nome reale
GOAL_FRAME = 'odom'

TARGET_POSE_TOPIC = 'target_info'  # suffisso _3d: convenzione demo_interfaces, stessa di /person_follow/target_info

DETECT_SERVICE = 'detect'
TARGET_CLASSES = ['person']
DEBUG_IMAGE_TOPIC = '/person_follow/hand_debug/compressed'  # suffisso /compressed: convenzione
                                                               # image_transport, stessa di /camera/hand/compressed

REID_SIMILARITY_THRESHOLD = 0.60  # alzata da 0.5 — mitigazione parziale, non risolve da sola i casi
                                    # di similarita' altissima per coincidenza (es. oggetti fuori
                                    # distribuzione per il modello di ReID, vedi MIN_DETECTION_CONFIDENCE)
MIN_DETECTION_CONFIDENCE = 0.10    # confidenza MINIMA della detection di YOLOE stessa (non l'aspetto)
                                    # per essere considerato un candidato — filtro indipendente
                                    # dall'aspetto, scarta classificazioni "person" incerte/al limite
REID_EMA_ALPHA = 0.3
TARGET_DISTANCE_THRESHOLD_FRAC = 0.30  # quanto puo' spostarsi (in pixel, come frazione della
                                         # diagonale immagine) il target da un frame all'altro —
                                         # oltre questa soglia, anche un aspetto simile viene scartato.
STABILITY_FRAMES_REQUIRED = 10    # frame consecutivi "stabili" (significato diverso nei due
                                    # metodi — vedi i rispettivi _handle_search_response_*)
                                    # prima di agganciare in SEARCH
TRACKING_GRACE_FRAMES = 30        # in TRACKING: frame di tentativo prima di passare a RECOVERY
RECOVERY_TIMEOUT_SEC = 15.0       # in RECOVERY: secondi di tempo REALE prima di arrendersi
REACQUISITION_STABILITY_FRAMES = 3   # quante volte di fila lo STESSO candidato deve risultare il
                                       # migliore match prima di essere CONFERMATO come il target
                                       # ritrovato (in TRACKING o RECOVERY) — un solo frame fortunato
                                       # (un'altra persona che per un istante somiglia e sta nel posto
                                       # giusto) non basta piu' a rubare l'identita' del target.
REACQUISITION_PX_TOLERANCE = 60.0    # quanto puo' spostarsi tra un tentativo e l'altro per essere
                                       # considerato "lo stesso candidato in corso di conferma"
STOPPING_DISTANCE = 1.0


REID_COMBINE = 0.40  # margine di vantaggio minimo (sopra la soglia REID_SIMILARITY_THRESHOLD)

W_SIMILARITY = 0.7
W_POSITION = 0.3

W_COSINE = 0.5
W_EUCLIDEAN = 1.0
W_MAGNITUDE = 0.8

EUCLIDEAN_SCALE= 10.0
MAGNITUDE_SCALE = 10.0


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
        
        self._last_response_time = None  # time.monotonic() dell'ultima risposta ricevuta dal servizio Detect

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
        self.reference_embedding = None  # HSV in botsort_hsv, vettore neurale in embedding_only
                                           # — mai entrambi insieme, il TIPO dipende da TRACKING_METHOD
        self.last_target_base = None
        self.last_published_pos = None
        self.last_published_time = 0.0

        self._locked_box = None  # ultimo box confermato — usato da entrambi i metodi
        self._locked_track_id = -1  # SOLO botsort_hsv: track_id del target agganciato

        self._stability_count = 0
        self._stability_track_id = -1              # SOLO botsort_hsv
        self._stability_reference_embedding = None  # SOLO embedding_only (confronto col frame precedente)

        self._tracking_miss_count = 0   # frame consecutivi di tentativo IN TRACKING, entrambi i metodi
        self._recovery_deadline = None  # time.monotonic() oltre il quale RECOVERY si arrende
        self._reacquisition_pending_box = None  # candidato in corso di conferma (vedi _confirm_reacquisition)
        self._reacquisition_count = 0

        self._pending_tracker_reset = False  # SOLO botsort_hsv: True = la PROSSIMA chiamata
                                               # chiede anche l'azzeramento del tracker BoT-SORT

        self._start_trigger_thread()
        
        sensor_qos_depth1 = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1
)

        self.create_subscription(CameraInfo, HAND_CAMERA_INFO_TOPIC, self._camera_info_cb,
                                  qos_profile_sensor_data, callback_group=self._main_group)
        self.create_subscription(Image, HAND_DEPTH_TOPIC, self._depth_cb,
                                  sensor_qos_depth1, callback_group=self._main_group)

        hand_rgb_msg_type = CompressedImage if HAND_RGB_COMPRESSED else Image
        self.create_subscription(hand_rgb_msg_type, HAND_RGB_TOPIC, self._image_cb,
                                  sensor_qos_depth1, callback_group=self._main_group)
        
        self.target_pose_pub = self.create_publisher(TargetInfoMessage, TARGET_POSE_TOPIC, 1)

        self.debug_pub = self.create_publisher(CompressedImage, DEBUG_IMAGE_TOPIC, 1)
        

        self.get_logger().info(
            f"TrackingFSM avviato — METODO ATTIVO: {TRACKING_METHOD}. Stato iniziale: INIT. "
            f"ROI: FOV={math.degrees(self.fov_rad):.0f}° (crop larghezza), "
            f"range=[{self.min_range:.2f},{self.max_range:.2f}]m (post-detection, via depth)")

    # ------------------------------------------------------------------
    def _start_trigger_thread(self):
        """Avvia (o riavvia) il thread che aspetta INVIO. Un thread Python
        non e' riavviabile una volta terminato — per questo, ogni volta che
        serve un nuovo trigger (il primo avvio, o il ritorno in
        WAITING_TRIGGER da RECOVERY), se ne crea uno nuovo, non si riusa il
        vecchio."""
        self._manual_trigger_received = False
        self._stdin_thread = threading.Thread(target=self._wait_for_manual_trigger, daemon=True)
        self._stdin_thread.start()

    def _enter_waiting_trigger(self):
        """Torna in WAITING_TRIGGER — chiamato quando RECOVERY scade senza
        ritrovare il target. Richiede di premere di nuovo INVIO: dopo un
        fallimento cosi' prolungato, decidere se/quando riprovare torna a
        essere una scelta della persona, non un ciclo automatico.

        NON azzera self.reference_embedding — e' voluto: e' la "memoria" di
        chi stavamo seguendo, e la vogliamo usare per filtrare la PROSSIMA
        SEARCH (vedi il controllo d'aspetto in _handle_search_response_*),
        cosi' un oggetto qualsiasi (es. un altro robot classificato per
        errore come "person") non venga agganciato solo perche' e' l'unico
        presente — deve anche somigliare a chi avevamo gia' imparato a
        riconoscere. Alla primissima ricerca in assoluto (mai stato
        agganciato nulla prima), reference_embedding e' ancora None: in
        quel caso specifico SEARCH non ha nulla con cui filtrare, e accetta
        il primo candidato stabile — limite di partenza inevitabile senza
        un passo di registrazione esplicito."""
        self.state = WAITING_TRIGGER
        self._locked_box = None
        self._locked_track_id = -1
        self._stability_count = 0
        self._stability_track_id = -1
        self._stability_reference_embedding = None
        self._tracking_miss_count = 0
        self._recovery_deadline = None
        self._reacquisition_pending_box = None
        self._reacquisition_count = 0
        self._pending_tracker_reset = True  # innocuo se TRACKING_METHOD="embedding_only"
                                               # (il campo semplicemente non viene guardato lato yolo
                                               # se use_tracker=False)
        self._start_trigger_thread()

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
        
        
    def _publish_target_info(self, box_full, dist, header):
        """"Pubblica le informazioni del target (bounding box, distanza) in un messaggio."""
        
        if self._last_response_time is not None:
            elapsed_ms = (time.monotonic() - self._last_response_time) * 1000.0
            self.get_logger().warn(f"[tracking_fsm] round-trip (sincrono): {elapsed_ms:.0f} ms")
        
        target_info_msg = TargetInfoMessage()
        target_info_msg.header = header
        target_info_msg.bounding_box = [float(v) for v in box_full]
        target_info_msg.depth_m = float(dist)
        target_info_msg.camera_rgb_topic = HAND_RGB_TOPIC

        target_info_msg.camera_depth_topic = HAND_DEPTH_TOPIC

        self.target_pose_pub.publish(target_info_msg)


    # ------------------------------------------------------------------
    # Dispatch per stato — il debug e' costruito e pubblicato SEMPRE
    # (tutti gli stati), la detection gira SOLO in SEARCH/TRACKING/RECOVERY.
    # ------------------------------------------------------------------
    def _image_cb(self, rgb_msg):
        t_start = time.time()
        
        image_stamp = rgb_msg.header.stamp.sec + rgb_msg.header.stamp.nanosec * 1e-9
        queue_delay_ms = (t_start - image_stamp) * 1000.0
        self.get_logger().warn(f"[tracking_fsm] _image_cb: latenza tra cattura ed elaborazione: queue delay={queue_delay_ms:.0f}ms", throttle_duration_sec=1.0)
        
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
            cv2.putText(debug_frame, f"[{TRACKING_METHOD}] {self.state.upper()}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
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
            return

        # ---- WAITING_TRIGGER: aspetta INVIO ----
        if self.state == WAITING_TRIGGER:
            if self._manual_trigger_received:
                self.get_logger().info("Trigger manuale ricevuto — passo a SEARCH.")
                self.state = SEARCH
            self._publish_debug(debug_frame, rgb_msg.header)
            return

        # ---- SEARCH / TRACKING / RECOVERY: qui gira la detection, sul crop — CHIAMATA BLOCCANTE ----
        if crop_rect is None:
            self.get_logger().info(
                f"{self.state.upper()}: ROI non calcolabile questo frame (intrinseci non ancora pronti) — "
                f"nessuna detection.", throttle_duration_sec=2.0)
            self._publish_debug(debug_frame, rgb_msg.header)
            return

        x1, y1, x2, y2 = crop_rect
        crop_bgr = frame_bgr[y1:y2, x1:x2]
        crop_msg = self.bridge.cv2_to_imgmsg(crop_bgr, encoding='bgr8')
        crop_msg.header = rgb_msg.header

        reset_now = self._pending_tracker_reset
        self._pending_tracker_reset = False
        response = self.detect_client.call_sync(crop_msg, target_classes=TARGET_CLASSES, reset_tracker=reset_now)
        
        self._last_response_time = time.monotonic()

        try:
            if TRACKING_METHOD == "botsort_hsv":
                if self.state == SEARCH:
                    self._handle_search_response_botsort(response, frame_bgr, crop_rect, debug_frame, rgb_msg.header)
                elif self.state == TRACKING:
                    self._handle_track_response_botsort(response, frame_bgr, crop_rect, debug_frame, rgb_msg.header)
                else:  # RECOVERY
                    self._handle_recovery_response_botsort(response, frame_bgr, crop_rect, debug_frame, rgb_msg.header)
            else:  # embedding_only
                if self.state == SEARCH:
                    self._handle_search_response_embedding(response, frame_bgr, crop_rect, debug_frame, rgb_msg.header)
                elif self.state == TRACKING:
                    self._handle_track_response_embedding(response, frame_bgr, crop_rect, debug_frame, rgb_msg.header)
                else:  # RECOVERY
                    self._handle_recovery_response_embedding(response, frame_bgr, crop_rect, debug_frame, rgb_msg.header)
        except Exception as ex:
            self.get_logger().error(f"Eccezione nell'elaborazione della risposta: {ex}", throttle_duration_sec=2.0)
            self._publish_debug(debug_frame, rgb_msg.header)

    # ====================================================================
    # METODO "botsort_hsv" — identita' via track_id (BoT-SORT), HSV come
    # rete di sicurezza SOLO quando il track_id sparisce.
    # ====================================================================

    def _handle_search_response_botsort(self, response, frame_bgr, crop_rect, debug_frame, header):
        if response is None:
            self._publish_debug(debug_frame, header)
            return

        crop_x1, crop_y1, _, _ = crop_rect
        depth_image = self._latest_depth_image

        valid_boxes = []  # (box_full, dist, track_id)
        for det in response.detections:
            box_full = (det.x1 + crop_x1, det.y1 + crop_y1, det.x2 + crop_x1, det.y2 + crop_y1)
            self._draw_box(debug_frame, box_full, (100, 100, 100), f"{det.class_name} det pre CONFIDENCE check YOLOE SEARCH id {det.track_id}", label_offset=3)
            
            if det.score < MIN_DETECTION_CONFIDENCE:
                if debug_frame is not None:
                    self._draw_box(debug_frame, box_full, (128, 0, 128),
                                    f"{det.class_name} (confidenza {det.score:.2f} troppo bassa)", label_offset=2)
                continue

            if depth_image is None:
                if debug_frame is not None:
                    self._draw_box(debug_frame, box_full, (0, 255, 255), f"{det.class_name} (depth n/d)", label_offset=2)
                continue
            box_depth = scale_box_to_depth(box_full, frame_bgr.shape, depth_image.shape)
            dist = box_center_depth(depth_image, box_depth)
            if dist is None:
                if debug_frame is not None:
                    self._draw_box(debug_frame, box_full, (128, 128, 128), f"{det.class_name} (depth invalida)", label_offset=2)
                continue
            if not (self.min_range <= dist <= self.max_range):
                if debug_frame is not None:
                    self._draw_box(debug_frame, box_full, (0, 0, 220), f"{det.class_name} {dist:.2f}m (fuori range)", label_offset=2)
                continue

            # Se conosciamo gia' l'aspetto del target (da una sessione
            # precedente in questo stesso avvio del nodo — vedi la nota in
            # _enter_waiting_trigger), un candidato che non gli somiglia
            # NON diventa un box valido, anche se e' l'unico presente e
            # classificato "person". Alla primissima ricerca in assoluto
            # (reference_embedding ancora None) questo filtro non si applica.
            if self.reference_embedding is not None:
                candidate_hsv = extract_appearance_embedding(frame_bgr, box_full)
                sim = embedding_similarity(candidate_hsv, self.reference_embedding)
                if sim < REID_SIMILARITY_THRESHOLD:
                    if debug_frame is not None:
                        self._draw_box(debug_frame, box_full, (255, 0, 255),
                                        f"{det.class_name} {dist:.2f}m (non e' il target noto, sim={sim:.2f})", label_offset=2)
                    continue

            valid_boxes.append((box_full, dist, det.track_id, det.score))

        if len(valid_boxes) != 1:
            if debug_frame is not None:
                for box_full, dist, _track_id, score in valid_boxes:
                    self._draw_box(debug_frame, box_full, (0, 200, 0), f"person {dist:.2f}m score={score:.2f}", label_offset=2)
            self.get_logger().info(
                f"SEARCH [botsort_hsv] in attesa: {len(valid_boxes)} box nel raggio d'azione su "
                f"{len(response.detections)} rilevati (serve esattamente 1).", throttle_duration_sec=2.0)
            self._stability_count = 0
            self._stability_track_id = -1
            self._publish_debug(debug_frame, header)
            return

        box_full, dist, track_id, score = valid_boxes[0]

        # Stabilita' basata SOLO sul track_id di BoT-SORT — se resta lo
        # stesso numero per N frame di fila, ci fidiamo. track_id=-1 (non
        # ancora agganciato da BoT-SORT) azzera il contatore: nessun segnale
        # affidabile su cui confermare continuita'.
        if track_id != -1 and track_id == self._stability_track_id:
            self._stability_count += 1
        elif track_id != -1:
            self._stability_count = 1
            self._stability_track_id = track_id
        else:
            self._stability_count = 0
            self._stability_track_id = -1

        self.get_logger().info(
            f"SEARCH [botsort_hsv]: 1 box nel raggio a {dist:.2f}m (track_id={track_id}, score={score:.2f}) — "
            f"stabilita' {self._stability_count}/{STABILITY_FRAMES_REQUIRED}", throttle_duration_sec=1.0)

        if self._stability_count < STABILITY_FRAMES_REQUIRED:
            if debug_frame is not None:
                label = (f"person {dist:.2f}m score={score:.2f} ({self._stability_count}/{STABILITY_FRAMES_REQUIRED})"
                         if track_id != -1 else f"person {dist:.2f}m score={score:.2f} (non ancora tracciato da BoT-SORT) id={track_id}")
                self._draw_box(debug_frame, box_full, (0, 200, 0), label, label_offset=0)
            self._publish_debug(debug_frame, header)
            return

        # Stabile per abbastanza frame (secondo BoT-SORT): aggancio.
        # Il riferimento HSV si calcola in locale, ORA, una volta sola.
        self.reference_embedding = extract_appearance_embedding(frame_bgr, box_full)
        self._locked_box = box_full
        self._locked_track_id = track_id
        self.state = TRACKING
        self._stability_count = 0
        self._stability_track_id = -1
        self.get_logger().info(
            f"[botsort_hsv] Target agganciato a {dist:.2f}m (track_id={track_id}) — passo a TRACKING.")
        if debug_frame is not None:
            self._draw_box(debug_frame, box_full, (0, 255, 0), f"TARGET {dist:.2f}m id={track_id}" , label_offset=0)
        self._publish_debug(debug_frame, header)

    def _attempt_reacquisition_botsort(self, response, frame_bgr, crop_rect, depth_image, debug_frame):
        """SOLO istogramma HSV (mai neurale) — usata dal periodo di
        tolleranza in TRACKING e da RECOVERY, quando il track_id agganciato
        e' sparito dalle detection."""
        if response is None:
            return None, -1, None, None

        crop_x1, crop_y1, _, _ = crop_rect
        h_img, w_img = frame_bgr.shape[:2]
        target_threshold_px = TARGET_DISTANCE_THRESHOLD_FRAC * math.hypot(w_img, h_img)
        last_center = box_center(self._locked_box)

        best_box, best_combined = None, -float("inf")
        best_dist_m, best_score, best_track_id = None, None, -1

        for det in response.detections:
            box_full = (det.x1 + crop_x1, det.y1 + crop_y1, det.x2 + crop_x1, det.y2 + crop_y1)
            self._draw_box(debug_frame, box_full, (200, 100, 100), f"{det.class_name} det pre_CONFIDENCE BOTSORT_TRACKING/RECOVERY id {det.track_id}" , label_offset=3)

            if det.score < MIN_DETECTION_CONFIDENCE:
                continue
            if depth_image is None:
                continue
            box_depth = scale_box_to_depth(box_full, frame_bgr.shape, depth_image.shape)
            dist_m = box_center_depth(depth_image, box_depth)
            if dist_m is None or not (self.min_range <= dist_m <= self.max_range):
                continue

            hsv_embedding = extract_appearance_embedding(frame_bgr, box_full)
            similarity = (1.0 if self.reference_embedding is None
                          else embedding_similarity(hsv_embedding, self.reference_embedding))
            if similarity < REID_SIMILARITY_THRESHOLD:
                continue

            dist_px = distance(box_center(box_full), last_center)
            if dist_px > target_threshold_px:
                continue

            combined = (W_SIMILARITY * similarity) - (W_POSITION * ( dist_px / target_threshold_px))
            if combined > best_combined:
                best_combined = combined
                best_box, best_dist_m, best_score, best_track_id = box_full, dist_m, similarity, det.track_id

        return best_box, best_track_id, best_dist_m, best_score

    def _handle_track_response_botsort(self, response, frame_bgr, crop_rect, debug_frame, header):
        depth_image = self._latest_depth_image
        
        for det in response.detections:
            box_full = (det.x1 + crop_rect[0], det.y1 + crop_rect[1],
                        det.x2 + crop_rect[0], det.y2 + crop_rect[1])
            self._draw_box(debug_frame, box_full, (100, 100, 100), f"{det.class_name} det pre CONFIDENCE check BOTSORT_TRACKING id {det.track_id}", label_offset=3)

        # ---- PERCORSO VELOCE: il track_id agganciato e' ancora tra le detection? ----
        fast_det = None
        if response is not None and self._locked_track_id != -1:
            crop_x1, crop_y1, _, _ = crop_rect
            for det in response.detections:
                if det.track_id == self._locked_track_id:
                    fast_det = det
                    break

        if fast_det is not None:
            box_full = (fast_det.x1 + crop_x1, fast_det.y1 + crop_y1,
                        fast_det.x2 + crop_x1, fast_det.y2 + crop_y1)
            dist_m = None
            if depth_image is not None:
                box_depth = scale_box_to_depth(box_full, frame_bgr.shape, depth_image.shape)
                dist_m = box_center_depth(depth_image, box_depth)
            in_range = dist_m is not None and (self.min_range <= dist_m <= self.max_range)

            self._locked_box = box_full
            self._tracking_miss_count = 0
            self._reset_reacquisition_confirmation()  # sul percorso veloce, nessuna conferma pendente ha senso

            if in_range:
                new_hsv = extract_appearance_embedding(frame_bgr, box_full)
                if new_hsv is not None and self.reference_embedding is not None:
                    self.reference_embedding = (
                        (1 - REID_EMA_ALPHA) * self.reference_embedding + REID_EMA_ALPHA * new_hsv)
                if debug_frame is not None:
                    # self._draw_box(debug_frame, box_full, (0, 255, 0),
                    #                 f"TARGET {dist_m:.2f}m id={self._locked_track_id}", label_offset=0)
                    pass
                self.get_logger().info(
                    f"TRACKING [botsort_hsv] ok (track_id={self._locked_track_id}): confermato a {dist_m:.2f}m",
                    throttle_duration_sec=1.0)
            else:
                # Fuori range ma il track_id e' ancora quello giusto: NON lo
                # contiamo come perso — restiamo agganciati, aspettiamo che rientri.
                if debug_frame is not None:
                    suffix = "(fuori range)" if dist_m is not None else "(depth n/d)"
                    # self._draw_box(debug_frame, box_full, (0, 200, 255),
                    #                 f"TARGET id={self._locked_track_id} {suffix}", label_offset=1)
                self.get_logger().info(
                    f"TRACKING [botsort_hsv]: track_id={self._locked_track_id} presente ma fuori range "
                    f"— resto agganciato, non conto come perso.", throttle_duration_sec=1.0)

            self._publish_debug(debug_frame, header)
            return

        # ---- Il track_id agganciato NON c'e': tentativo con HSV, ancora in TRACKING ----
        # Un candidato trovato NON viene adottato subito — deve essere il
        # migliore match per REACQUISITION_STABILITY_FRAMES tentativi di
        # fila (vedi _confirm_reacquisition): un solo frame fortunato (es.
        # un'altra persona che per un istante somiglia e sta nel posto
        # giusto) non deve poter rubare l'identita' del target.
        best_box, best_track_id, best_dist_m, best_score = self._attempt_reacquisition_botsort(
            response, frame_bgr, crop_rect, depth_image, debug_frame)
        
        if best_box is not None and best_score < REID_COMBINE:
            best_box = None  # scarta il candidato se la somiglianza e' troppo vicina alla soglia minima
            

        committed = False
        if best_box is not None:
            if self._confirm_reacquisition(best_box):
                committed = True
            else:
                if debug_frame is not None:
                    self._draw_box(debug_frame, best_box, (255, 165, 0),
                                    f"possibile target {best_dist_m:.2f}m "
                                    f"(conferma {self._reacquisition_count}/{REACQUISITION_STABILITY_FRAMES})" , label_offset=2)
                self.get_logger().info(
                    f"TRACKING [botsort_hsv]: candidato trovato, in attesa di conferma "
                    f"({self._reacquisition_count}/{REACQUISITION_STABILITY_FRAMES})...", throttle_duration_sec=1.0)
        else:
            self._reset_reacquisition_confirmation()

        if committed:
            if debug_frame is not None:
                self._draw_box(debug_frame, best_box, (0, 255, 0), f"TARGET {best_dist_m:.2f}m id={best_track_id}", label_offset=0)
            old_track_id = self._locked_track_id
            self._locked_box = best_box
            self._locked_track_id = best_track_id  # ADOTTIAMO il nuovo track_id
            self._tracking_miss_count = 0
            new_hsv = extract_appearance_embedding(frame_bgr, best_box)
            if new_hsv is not None and self.reference_embedding is not None:
                self.reference_embedding = (
                    (1 - REID_EMA_ALPHA) * self.reference_embedding + REID_EMA_ALPHA * new_hsv)
            self.get_logger().info(
                f"TRACKING [botsort_hsv]: target CONFERMATO a {best_dist_m:.2f}m, similarity={best_score:.2f} — "
                f"track_id {old_track_id} -> {best_track_id}", throttle_duration_sec=1.0)
            self._publish_debug(debug_frame, header)
            return

        # Nessuna corrispondenza confermata questo frame: avanza il
        # contatore di tolleranza — sia che non ci fosse nessun candidato,
        # sia che ce ne fosse uno ancora in attesa di conferma.
        self._tracking_miss_count += 1
        if self._tracking_miss_count < TRACKING_GRACE_FRAMES:
            self.get_logger().info(
                f"TRACKING [botsort_hsv]: target non trovato — tentativo "
                f"{self._tracking_miss_count}/{TRACKING_GRACE_FRAMES} prima di passare a RECOVERY.",
                throttle_duration_sec=1.0)
            self._publish_debug(debug_frame, header)
            return

        self.state = RECOVERY
        self._recovery_deadline = time.monotonic() + RECOVERY_TIMEOUT_SEC
        self._tracking_miss_count = 0
        self.get_logger().info(
            f"TRACKING [botsort_hsv]: target non ritrovato entro {TRACKING_GRACE_FRAMES} frame — "
            f"passo a RECOVERY (timeout {RECOVERY_TIMEOUT_SEC:.0f}s).")
        self._publish_debug(debug_frame, header)

    def _handle_recovery_response_botsort(self, response, frame_bgr, crop_rect, debug_frame, header):
        depth_image = self._latest_depth_image
        best_box, best_track_id, best_dist_m, best_score = self._attempt_reacquisition_botsort(
            response, frame_bgr, crop_rect, depth_image, debug_frame)
        
        if best_box is not None and best_score < REID_COMBINE:
            best_box = None  # scarta il candidato se la somiglianza e' troppo vicina alla soglia minima

        committed = False
        if best_box is not None:
            if self._confirm_reacquisition(best_box):
                committed = True
            else:
                if debug_frame is not None:
                    self._draw_box(debug_frame, best_box, (255, 165, 0),
                                    f"possibile target {best_dist_m:.2f}m "
                                    f"(conferma {self._reacquisition_count}/{REACQUISITION_STABILITY_FRAMES})", label_offset=2)
        else:
            self._reset_reacquisition_confirmation()

        if committed:
            if debug_frame is not None:
                self._draw_box(debug_frame, best_box, (0, 255, 0), f"TARGET {best_dist_m:.2f}m id={best_track_id}", label_offset=0)
            old_track_id = self._locked_track_id
            self._locked_box = best_box
            self._locked_track_id = best_track_id
            self.state = TRACKING
            self._tracking_miss_count = 0
            self._recovery_deadline = None
            new_hsv = extract_appearance_embedding(frame_bgr, best_box)
            if new_hsv is not None and self.reference_embedding is not None:
                self.reference_embedding = (
                    (1 - REID_EMA_ALPHA) * self.reference_embedding + REID_EMA_ALPHA * new_hsv)
            self.get_logger().info(
                f"RECOVERY [botsort_hsv]: target CONFERMATO a {best_dist_m:.2f}m, similarity={best_score:.2f} — "
                f"track_id {old_track_id} -> {best_track_id} — torno a TRACKING.")
            self._publish_debug(debug_frame, header)
            return

        remaining = self._recovery_deadline - time.monotonic()
        if remaining <= 0:
            self.get_logger().info(
                f"RECOVERY [botsort_hsv]: timeout di {RECOVERY_TIMEOUT_SEC:.0f}s scaduto — "
                f"torno in WAITING_TRIGGER.")
            self._enter_waiting_trigger()
        else:
            self.get_logger().info(
                f"RECOVERY [botsort_hsv]: nessuna corrispondenza confermata — {remaining:.0f}s rimanenti.",
                throttle_duration_sec=1.0)
        self._publish_debug(debug_frame, header)

    # ====================================================================
    # METODO "embedding_only" — identita' SOLO via embedding neurale, in
    # OGNI stato (SEARCH compreso). Nessun track_id/BoT-SORT coinvolto.
    # ====================================================================

    def _handle_search_response_embedding(self, response, frame_bgr, crop_rect, debug_frame, header):
        if response is None:
            self._publish_debug(debug_frame, header)
            return

        crop_x1, crop_y1, _, _ = crop_rect
        depth_image = self._latest_depth_image

        valid_boxes = []  # (box_full, dist, embedding)
        for det in response.detections:
            box_full = (det.x1 + crop_x1, det.y1 + crop_y1, det.x2 + crop_x1, det.y2 + crop_y1)
            self._draw_box(debug_frame, box_full, (100, 100, 100), f"{det.class_name} det pre MIN_DETECTION_CONFIDENCE check EMBEDDING_ONLY SEARCH", label_offset=3)

            if det.score < MIN_DETECTION_CONFIDENCE:
                continue
            if depth_image is None:
                if debug_frame is not None:
                    self._draw_box(debug_frame, box_full, (0, 255, 255), f"{det.class_name} (depth n/d)", label_offset=0)
                continue
            box_depth = scale_box_to_depth(box_full, frame_bgr.shape, depth_image.shape)
            dist = box_center_depth(depth_image, box_depth)
            if dist is None:
                if debug_frame is not None:
                    self._draw_box(debug_frame, box_full, (128, 128, 128), f"{det.class_name} (depth invalida)", label_offset=0)
                continue
            if not (self.min_range <= dist <= self.max_range):
                if debug_frame is not None:
                    self._draw_box(debug_frame, box_full, (0, 0, 220), f"{det.class_name} {dist:.2f}m (fuori range)", label_offset=0   )
                continue

            candidate_embedding = embedding_from_msg(det.embedding)
            self._draw_box(debug_frame, box_full, (0, 255, 0), f"{det.class_name} {dist:.2f}m (embedding: {candidate_embedding[:5]})", label_offset=1)
            

            # Stesso filtro del metodo botsort_hsv (vedi la nota gemella
            # li'), qui con l'embedding neurale invece dell'HSV — se
            # conosciamo gia' l'aspetto del target da una sessione
            # precedente, un candidato che non gli somiglia non diventa un
            # box valido.
            if self.reference_embedding is not None:
                
                sim = rich_neural_embedding_similarity(candidate_embedding, self.reference_embedding , w_cosine=W_COSINE, w_euclidean=W_EUCLIDEAN, w_magnitude=W_MAGNITUDE, euclidean_scale=EUCLIDEAN_SCALE, magnitude_scale=MAGNITUDE_SCALE)
                
                if sim < REID_SIMILARITY_THRESHOLD:
                    if debug_frame is not None:
                        self._draw_box(debug_frame, box_full, (255, 0, 255),
                                        f"{det.class_name} {dist:.2f}m (non e' il target noto, sim={sim:.2f})", label_offset=1)
                    continue

            valid_boxes.append((box_full, dist, candidate_embedding, det.score))
        
        if len(valid_boxes) != 1:
            if debug_frame is not None:
                for box_full, dist, _emb, score in valid_boxes:
                    self._draw_box(debug_frame, box_full, (0, 200, 0), f"person {dist:.2f}m score={score:.2f}", label_offset=1)
            self.get_logger().info(
                f"SEARCH [embedding_only] in attesa: {len(valid_boxes)} box nel raggio d'azione su "
                f"{len(response.detections)} rilevati (serve esattamente 1).", throttle_duration_sec=2.0)
            self._stability_count = 0
            self._stability_reference_embedding = None
            self._publish_debug(debug_frame, header)
            return

        box_full, dist, box_embedding, score = valid_boxes[0]

        if box_embedding is None:
            self.get_logger().warn(
                "SEARCH [embedding_only]: nessun embedding ricevuto — il DetectorNode ha "
                "embedding_model_path configurato? Senza embedding non posso confermare stabilita' "
                "in questo metodo.", throttle_duration_sec=2.0)
            self._stability_count = 0
            self._stability_reference_embedding = None
            self._publish_debug(debug_frame, header)
            return

        # Stabilita' basata sulla continuita' D'ASPETTO frame-su-frame — non
        # su un track_id (che qui non usiamo mai): l'embedding di questo box
        # deve restare simile a quello del frame PRECEDENTE per N frame di fila.
        if self._stability_reference_embedding is not None:
            sim = neural_embedding_similarity(box_embedding, self._stability_reference_embedding)
            if sim >= REID_SIMILARITY_THRESHOLD:
                self._stability_count += 1
            else:
                self._stability_count = 1
        else:
            self._stability_count = 1
        self._stability_reference_embedding = box_embedding  # confronto SEMPRE col frame appena visto

        self.get_logger().info(
            f"SEARCH [embedding_only]: 1 box nel raggio a {dist:.2f}m (score={score:.2f}) — "
            f"stabilita' d'aspetto {self._stability_count}/{STABILITY_FRAMES_REQUIRED}",
            throttle_duration_sec=1.0)

        if self._stability_count < STABILITY_FRAMES_REQUIRED:
            if debug_frame is not None:
                self._draw_box(debug_frame, box_full, (0, 200, 0),
                                f"person {dist:.2f}m score={score:.2f} ({self._stability_count}/{STABILITY_FRAMES_REQUIRED})", label_offset=1)
            self._publish_debug(debug_frame, header)
            return

        # Stabile per abbastanza frame: aggancio.
        self.reference_embedding = box_embedding
        self._locked_box = box_full
        self._publish_target_info(box_full, dist, header)
        self.state = TRACKING
        self._stability_count = 0
        self._stability_reference_embedding = None
        self.get_logger().info(f"[embedding_only] Target agganciato a {dist:.2f}m — passo a TRACKING.")
        if debug_frame is not None:
            self._draw_box(debug_frame, box_full, (0, 255, 0), f"TARGET {dist:.2f}m", label_offset=0)
        self._publish_debug(debug_frame, header)

    def _attempt_reacquisition_embedding(self, response, frame_bgr, crop_rect, depth_image, debug_frame):
        """SOLO embedding neurale (mai HSV, mai track_id) — chiamata ad
        OGNI frame in TRACKING (qui non esiste un 'percorso veloce' via
        track_id: ogni frame e' gia' un tentativo di riconoscimento pieno)
        e in RECOVERY."""
        if response is None:
            return None, None, None, None

        crop_x1, crop_y1, _, _ = crop_rect
        h_img, w_img = frame_bgr.shape[:2]
        target_threshold_px = TARGET_DISTANCE_THRESHOLD_FRAC * math.hypot(w_img, h_img)
        last_center = box_center(self._locked_box)

        best_box, best_combined = None, -float("inf")
        best_dist_m, best_score, best_embedding = None, None, None

        for det in response.detections:
            box_full = (det.x1 + crop_x1, det.y1 + crop_y1, det.x2 + crop_x1, det.y2 + crop_y1)

            if det.score < MIN_DETECTION_CONFIDENCE:
                continue
            if depth_image is None:
                continue
            box_depth = scale_box_to_depth(box_full, frame_bgr.shape, depth_image.shape)
            dist_m = box_center_depth(depth_image, box_depth)
            if dist_m is None or not (self.min_range <= dist_m <= self.max_range):
                continue

            b_embedding = embedding_from_msg(det.embedding)
            
            # self._draw_box(debug_frame, box_full, (100, 100, 100), f"")
            
            similarity = rich_neural_embedding_similarity(b_embedding, self.reference_embedding, w_cosine=W_COSINE, w_euclidean=W_EUCLIDEAN, w_magnitude=W_MAGNITUDE, euclidean_scale=EUCLIDEAN_SCALE, magnitude_scale=MAGNITUDE_SCALE)
                          
            if similarity < REID_SIMILARITY_THRESHOLD:
                continue

            dist_px = distance(box_center(box_full), last_center)
            if dist_px > target_threshold_px:
                continue

            combined = (W_SIMILARITY * similarity) - (W_POSITION * (dist_px / target_threshold_px))
            
            self.get_logger().info(f"Similarity: {similarity:.2f}, Distance: {dist_px:.2f}px, Combined: {combined:.2f}", throttle_duration_sec=1.0)
            
            self._draw_box(debug_frame, box_full, (0, 200, 0), f"person {dist_m:.2f}m score={det.score:.2f} sim={similarity:.2f} comb={combined:.2f}", label_offset=4)
            
            if combined > best_combined:
                best_combined = combined
                best_box, best_dist_m, best_score, best_embedding = box_full, dist_m, best_combined, b_embedding

        return best_box, best_dist_m, best_score, best_embedding

    def _handle_track_response_embedding(self, response, frame_bgr, crop_rect, debug_frame, header):
        # Nessun percorso veloce qui: senza track_id, OGNI frame e' un
        # tentativo di riconoscimento pieno (aspetto+posizione+distanza).
        # La conferma su piu' tentativi (vedi _confirm_reacquisition) scatta
        # SOLO se il frame precedente era gia' un "miss" — durante il
        # tracciamento continuo e senza interruzioni, la continuita' di
        # posizione+distanza frame-su-frame e' gia' una protezione
        # sufficiente, non serve rallentare anche quel caso.
        was_continuous = (self._tracking_miss_count == 0)
        depth_image = self._latest_depth_image
        
        
        for det in response.detections:
            box_full = (det.x1 + crop_rect[0], det.y1 + crop_rect[1], det.x2 + crop_rect[0], det.y2 + crop_rect[1])
            self._draw_box(debug_frame, box_full, (100, 100, 100), f"{det.class_name} - {det.score:.2f}", label_offset=3)
        
        best_box, best_dist_m, best_score, best_embedding = self._attempt_reacquisition_embedding(
            response, frame_bgr, crop_rect, depth_image, debug_frame)
        
        if best_box is not None and best_score <  REID_COMBINE :
            best_box = None  # non e' abbastanza simile da essere considerato un match valido

        committed = False
        if best_box is not None:
            if was_continuous:
                committed = True
                self._reset_reacquisition_confirmation()
            elif self._confirm_reacquisition(best_box):
                committed = True
            else:
                if debug_frame is not None:
                    self._draw_box(debug_frame, best_box, (255, 165, 0),
                                    f"possibile target {best_dist_m:.2f}m "
                                    f"(conferma {self._reacquisition_count}/{REACQUISITION_STABILITY_FRAMES})" , label_offset=2)
                self.get_logger().info(
                    f"TRACKING [embedding_only]: candidato trovato, in attesa di conferma "
                    f"({self._reacquisition_count}/{REACQUISITION_STABILITY_FRAMES})...", throttle_duration_sec=1.0)
        else:
            self._reset_reacquisition_confirmation()
            
        

        if committed:
            if debug_frame is not None:
                self._draw_box(debug_frame, best_box, (0, 255, 0), f"TARGET {best_dist_m:.2f}m", label_offset=0)
            self._locked_box = best_box
            self._tracking_miss_count = 0
            self._publish_target_info(best_box, best_dist_m, header)
            if best_embedding is not None and self.reference_embedding is not None:
                self.reference_embedding = (
                    (1 - REID_EMA_ALPHA) * self.reference_embedding + REID_EMA_ALPHA * best_embedding)
            self.get_logger().info(
                f"TRACKING [embedding_only] ok: confermato a {best_dist_m:.2f}m, similarity={best_score:.2f}",
                throttle_duration_sec=1.0)
            self._publish_debug(debug_frame, header)
            return

        self._tracking_miss_count += 1
        if self._tracking_miss_count < TRACKING_GRACE_FRAMES:
            self.get_logger().info(
                f"TRACKING [embedding_only]: target non trovato — tentativo "
                f"{self._tracking_miss_count}/{TRACKING_GRACE_FRAMES} prima di passare a RECOVERY.",
                throttle_duration_sec=1.0)
            self._publish_debug(debug_frame, header)
            return
        
        
        self.state = RECOVERY
        self._recovery_deadline = time.monotonic() + RECOVERY_TIMEOUT_SEC
        self._tracking_miss_count = 0
        self.get_logger().info(
            f"TRACKING [embedding_only]: target non ritrovato entro {TRACKING_GRACE_FRAMES} frame — "
            f"passo a RECOVERY (timeout {RECOVERY_TIMEOUT_SEC:.0f}s).")
        self._publish_debug(debug_frame, header)

    def _handle_recovery_response_embedding(self, response, frame_bgr, crop_rect, debug_frame, header):
        
        
        for det in response.detections:
            box_full = (det.x1 + crop_rect[0], det.y1 + crop_rect[1], det.x2 + crop_rect[0], det.y2 + crop_rect[1])
            self._draw_box(debug_frame, box_full, (100, 100, 100), f"{det.class_name} det pre MIN_DETECTION_CONFIDENCE check EMBEDDING_ONLY RECOVERY", label_offset=3)
        
        depth_image = self._latest_depth_image
        best_box, best_dist_m, best_score, best_embedding = self._attempt_reacquisition_embedding(
            response, frame_bgr, crop_rect, depth_image, debug_frame)
        
        
        if best_box is not None and best_score < REID_COMBINE :
            best_box = None  # non e' abbastanza simile da essere considerato un match valido

        committed = False
        if best_box is not None:
            if self._confirm_reacquisition(best_box):
                committed = True
            else:
                if debug_frame is not None:
                    self._draw_box(debug_frame, best_box, (255, 165, 0),
                                    f"possibile target {best_dist_m:.2f}m "
                                    f"(conferma {self._reacquisition_count}/{REACQUISITION_STABILITY_FRAMES})", label_offset=2)
        else:
            self._reset_reacquisition_confirmation()
            
        

        if committed:
            if debug_frame is not None:
                self._draw_box(debug_frame, best_box, (0, 255, 0), f"TARGET {best_dist_m:.2f}m", label_offset=0)
            self._locked_box = best_box
            self.state = TRACKING
            self._tracking_miss_count = 0
            self._publish_target_info(best_box, best_dist_m, header)
            self._recovery_deadline = None
            if best_embedding is not None and self.reference_embedding is not None:
                self.reference_embedding = (
                    (1 - REID_EMA_ALPHA) * self.reference_embedding + REID_EMA_ALPHA * best_embedding)
            self.get_logger().info(
                f"RECOVERY [embedding_only]: target CONFERMATO a {best_dist_m:.2f}m, "
                f"similarity={best_score:.2f} — torno a TRACKING.")
            self._publish_debug(debug_frame, header)
            return
        

        remaining = self._recovery_deadline - time.monotonic()
        if remaining <= 0:
            self.get_logger().info(
                f"RECOVERY [embedding_only]: timeout di {RECOVERY_TIMEOUT_SEC:.0f}s scaduto — "
                f"torno in WAITING_TRIGGER.")
            self._enter_waiting_trigger()
        else:
            self.get_logger().info(
                f"RECOVERY [embedding_only]: nessuna corrispondenza confermata — {remaining:.0f}s rimanenti.",
                throttle_duration_sec=1.0)
        self._publish_debug(debug_frame, header)

    # ------------------------------------------------------------------
    # Comune a entrambi i metodi
    # ------------------------------------------------------------------
    def _confirm_reacquisition(self, candidate_box):
        """Richiede che lo STESSO candidato (per posizione) sia il
        migliore match per REACQUISITION_STABILITY_FRAMES tentativi di
        fila prima di essere confermato come il target ritrovato — un solo
        frame fortunato (es. un passante che per un istante somiglia al
        target ed e' nel punto giusto) non basta piu' a rubargli
        l'identita'. Ritorna True quando la conferma e' raggiunta (e
        azzera il contatore per la prossima volta), False se serve ancora
        attesa — in quel caso il chiamante NON deve ancora agganciare
        nulla, solo continuare a provare al frame successivo."""
        candidate_center = box_center(candidate_box)
        if (self._reacquisition_pending_box is not None
                and distance(candidate_center, box_center(self._reacquisition_pending_box))
                <= REACQUISITION_PX_TOLERANCE):
            self._reacquisition_count += 1
        else:
            self._reacquisition_count = 1
        self._reacquisition_pending_box = candidate_box

        if self._reacquisition_count >= REACQUISITION_STABILITY_FRAMES:
            self._reacquisition_pending_box = None
            self._reacquisition_count = 0
            return True
        return False

    def _reset_reacquisition_confirmation(self):
        self._reacquisition_pending_box = None
        self._reacquisition_count = 0

    @staticmethod
    def _draw_box(frame, box, color, label, label_offset=0):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 2
        
        (text_w,text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
        
        h_img, w_img = frame.shape[:2]
        
        text_x = min(x1, max(0, w_img - text_w -4))  # 4 pixel di margine a destra
        line_height = text_h + 10  # 2 pixel di margine sopra e sotto
        
        text_y = y1 - 8 - label_offset * line_height
        text_y = max(text_h + 2, text_y)  # Evita che il testo vada sopra l'immagine
        
        cv2.rectangle(frame, (text_x, text_y - text_h - 4), (text_x + text_w + 4, text_y + 4), (0,0,0), -1)
        
        cv2.putText(frame, label, (text_x + 2, text_y), font, font_scale, color, thickness, cv2.LINE_AA)

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