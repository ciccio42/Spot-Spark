#!/usr/bin/env python3
"""
tracking_fsm.py (demo_package)

TWO COMPLETELY SEPARATE METHODS, not intertwined — chosen with ONE SINGLE
switch at the top of the file (TRACKING_METHOD). It is not "which similarity
do I use in recovery": they are two different architectures, each with its
own implementation of SEARCH/TRACKING/RECOVERY, so they can be compared
without one contaminating the other.

  "botsort_hsv": the target identity relies ONLY on BoT-SORT
      (persistent track_id, model.track()) + an HSV histogram as a safety
      net when the track_id disappears. No neural embedding involved, in
      any state.

  "embedding_only": the identity relies ONLY on the neural embedding
      (separate ReID model on the DetectorNode side) — in ALL states,
      SEARCH included. BoT-SORT/track_id are never used to decide who the
      target is (even if the service computes them anyway, we simply
      ignore them).

Design shared by both methods (unchanged):
  - ROI as an image crop (FOV via intrinsics, height from fixed
    fractions — no TF).
  - Distance read from the ToF depth (patch centred on the box).
  - SYNCHRONOUS/blocking call to the Detect service, multi-threaded
    executor with separate callback groups (client vs the rest) to avoid
    deadlocks.
  - TRACKING -> FRAME-based grace period -> RECOVERY (REAL-TIME
    timeout) -> WAITING_TRIGGER if the target is not found again.

Spot LEDs (AudioVisualClient, Boston Dynamics gRPC SDK):
  - Handled by a DEDICATED THREAD (_led_worker), never by _image_cb: gRPC
    calls to the robot must not add latency to perception.
  - Robot software 5.0.1 does NOT allow creating custom behaviors
    (AddOrModifyBehavior arrives in later versions): each state is mapped
    to the NAME of a behavior already present on the robot (STATI_LED).
    To list them: list_spot_led_behaviors.py.
  - On a state change the previous behavior is stopped and the new one
    started; every LED_REFRESH_SEC its expiry (LED_DURATION_SEC) is
    extended: if the node dies, the LEDs go back to normal by themselves.
  - If the A/V system is not available or no state has a behavior, the
    LEDs stay disabled but the FSM works normally.
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

import bosdyn.client
from bosdyn.api import audio_visual_pb2
from bosdyn.client.audio_visual import AudioVisualClient

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
# LED — for each FSM state, the NAME of an A/V behavior ALREADY present
# on the robot (robot software 5.0.1 does not allow creating new ones:
# AddOrModifyBehavior only exists from later versions).
# To see the available names, with colours and whether they have audio:
#     python3 list_spot_led_behaviors.py
# None = no behavior for that state (the LEDs go back to the robot's normal
# behaviour). Avoid behaviors with AUDIO (the script flags them): they would
# sound the buzzer at every state change.
# ============================================================
STATI_LED = {
    INIT:            None,
    WAITING_TRIGGER: None,                               # robot's normal LEDs
    SEARCH:          "internal_autonomous_operation",    # pulsing white   (priority 3)
    TRACKING:        "internal_wait_for_entity",         # pulsing green   (priority 6)
    RECOVERY:        "internal_autonomous_navigation",   # blinking green  (priority 4)
}
LED_DURATION_SEC = 5.0     # expiry of each run_behavior (if the node dies, the LEDs turn off by themselves)
LED_REFRESH_SEC = 2.0      # how often the worker renews the behavior (must be < LED_DURATION_SEC)

# ============================================================
# THE SWITCH — decides which of the two architectures to use for the whole
# session. Change this, rebuild, test; comparing the two methods takes
# two separate sessions, not a run-time switch.
# ============================================================
TRACKING_METHOD = "embedding_only"  # "botsort_hsv" or "embedding_only"

# ============================================================
# Values written here, no command-line arguments.
# ============================================================
HAND_RGB_TOPIC = 'camera/hand/compressed'
HAND_RGB_COMPRESSED = True   # True on bags, False on the real robot if the recompression node is missing
HAND_CAMERA_INFO_TOPIC = '/camera/hand/camera_info'
HAND_DEPTH_TOPIC = '/depth/hand/image'   # ToF depth of the arm camera — check the real name
GOAL_FRAME = 'odom'

TARGET_POSE_TOPIC = 'target_info'

DETECT_SERVICE = 'detect'
TARGET_CLASSES = ['person','quadruped','quadruped animal','quadruped robot','robotic dog','four-legged robot', 'dog', 'robot','umanoid robot']
DEBUG_IMAGE_TOPIC = '/person_follow/hand_debug/compressed'  # /compressed suffix: image_transport
                                                               # convention, same as /camera/hand/compressed

REID_SIMILARITY_THRESHOLD = 0.60  # minimum similarity threshold to be a candidate
MIN_DETECTION_CONFIDENCE = 0.10   # MINIMUM confidence of the YOLOE detection itself (not the appearance)
REID_EMA_ALPHA = 0.3
TARGET_DISTANCE_THRESHOLD_FRAC = 0.30  # how far the target can move (in pixels, as a fraction of
                                         # the image diagonal) from one frame to the next
STABILITY_FRAMES_REQUIRED = 10    # consecutive "stable" frames before locking in SEARCH
TRACKING_GRACE_FRAMES = 30        # in TRACKING: attempt frames before switching to RECOVERY
RECOVERY_TIMEOUT_SEC = 15.0       # in RECOVERY: seconds of REAL time before giving up
REACQUISITION_STABILITY_FRAMES = 3   # consecutive attempts to confirm a re-lock
REACQUISITION_PX_TOLERANCE = 60.0    # maximum movement between attempts to count as "same candidate"
STOPPING_DISTANCE = 1.0

REID_COMBINE = 0.40  # minimum threshold on the WINNER's score (best_score)

W_SIMILARITY = 0.7
W_POSITION = 0.3

W_COSINE = 0.5
W_EUCLIDEAN = 1.0
W_MAGNITUDE = 0.8

EUCLIDEAN_SCALE = 10.0
MAGNITUDE_SCALE = 10.0


class TrackingFSM(Node):
    def __init__(self, robot=None):
        super().__init__('tracking_fsm')

        # ---------------- LED ----------------
        self.robot = robot
        self.audio_visual_client = None
        if self.robot is not None:
            try:
                self.audio_visual_client = self.robot.ensure_client(AudioVisualClient.default_service_name)
                self.get_logger().info("Client AudioVisual gRPC agganciato con successo.")
            except Exception as ex:
                self.get_logger().error(f"Impossibile creare AudioVisualClient: {ex} — LED disattivati.")
                self.audio_visual_client = None

        if self.audio_visual_client is not None and all(v is None for v in STATI_LED.values()):
            self.get_logger().warn(
                "[LED] nessun behavior associato agli stati in STATI_LED — lancia "
                "list_spot_led_behaviors.py e compila il dizionario. LED disattivati.")
            self.audio_visual_client = None

        self._led_lock = threading.Lock()
        self._led_desired_state = None
        self._led_applied_name = None   # behavior currently running on the robot
        self._led_wake = threading.Event()
        self._led_stop = threading.Event()
        self._led_thread = None
        if self.audio_visual_client is not None:
            self._led_thread = threading.Thread(target=self._led_worker, daemon=True)
            self._led_thread.start()

        # ---------------- Perception ----------------
        self.goal_frame = GOAL_FRAME
        self.fov_rad = math.radians(CONE_FOV_DEG)
        self.min_range = CONE_MIN_RANGE
        self.max_range = CONE_MAX_RANGE

        self.bridge = CvBridge()
        self.intrinsics = None
        self._hand_optical_frame = None
        self._latest_depth_image = None
        self._last_image_cb_time = None

        self._last_response_time = None  # time.monotonic() of the last response received from the Detect service

        # "We block everything": camera_info, depth and image are in the same
        # group — while _image_cb is blocked waiting for the response of the
        # Detect service (call_sync, synchronous), depth/camera_info updates
        # are queued too and wait for their turn.
        #
        # The service client MUST be in a different group — otherwise the
        # response could never be processed while _image_cb is blocked
        # waiting for it (deadlock, not just a slowdown).
        self._main_group = MutuallyExclusiveCallbackGroup()
        self._service_group = ReentrantCallbackGroup()

        self.detect_client = DetectClient(self, DETECT_SERVICE, callback_group=self._service_group)

        self.state = INIT
        self.reference_embedding = None  # HSV in botsort_hsv, neural vector in embedding_only
        self.last_target_base = None
        self.last_published_pos = None
        self.last_published_time = 0.0

        self._locked_box = None  # last confirmed box — used by both methods
        self._locked_track_id = -1  # botsort_hsv ONLY: track_id of the locked target

        self._stability_count = 0
        self._stability_track_id = -1              # botsort_hsv ONLY
        self._stability_reference_embedding = None  # embedding_only ONLY (comparison with the previous frame)

        self._tracking_miss_count = 0   # consecutive attempt frames IN TRACKING, both methods
        self._recovery_deadline = None  # time.monotonic() after which RECOVERY gives up
        self._reacquisition_pending_box = None  # candidate being confirmed (see _confirm_reacquisition)
        self._reacquisition_count = 0

        self._pending_tracker_reset = False  # botsort_hsv ONLY

        self._start_trigger_thread()

        sensor_qos_depth1 = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
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
            f"range=[{self.min_range:.2f},{self.max_range:.2f}]m (post-detection, via depth). "
            f"LED: {'ATTIVI' if self.audio_visual_client is not None else 'DISATTIVATI'}")

        self._request_led_state(self.state)

    # ==================================================================
    # Spot LEDs — dedicated thread, never gRPC calls inside _image_cb
    # ==================================================================
    def _request_led_state(self, state):
        """Called by the FSM (zero cost): records the desired state and
        wakes up the worker. No network call here."""
        if self.audio_visual_client is None:
            return
        with self._led_lock:
            if state == self._led_desired_state:
                return
            self._led_desired_state = state
        self._led_wake.set()

    def _led_worker(self):
        """Runs on the robot the behavior mapped to the current state and
        renews it periodically. On a state change it stops the previous
        behavior and starts the new one."""
        # run_behavior converts times to robot time: time sync is required.
        try:
            self.robot.time_sync.wait_for_sync(timeout_sec=10.0)
        except Exception as ex:
            self.get_logger().error(f"[LED] time sync con il robot non stabilita: {ex} — LED disattivati.")
            return

        # Check once that the configured names exist on the robot.
        try:
            available = {live.name: live for live in self.audio_visual_client.list_behaviors()}
        except Exception as ex:
            self.get_logger().error(
                f"[LED] list_behaviors fallita: {ex} — il robot ha il sistema A/V? LED disattivati.")
            return
        for state, name in STATI_LED.items():
            if name is None:
                continue
            if name not in available:
                self.get_logger().error(
                    f"[LED] behavior {name!r} (stato {state.upper()}) NON presente sul robot. "
                    f"Disponibili: {sorted(available)}")
            elif available[name].behavior.audio_sequence_group.ListFields():
                self.get_logger().warn(
                    f"[LED] behavior {name!r} (stato {state.upper()}) contiene AUDIO: suonera' il buzzer.")

        last_run = 0.0
        while not self._led_stop.is_set():
            self._led_wake.wait(timeout=LED_REFRESH_SEC)
            self._led_wake.clear()
            if self._led_stop.is_set():
                break

            with self._led_lock:
                desired_state = self._led_desired_state
            name = STATI_LED.get(desired_state)
            if name is not None and name not in available:
                name = None  # wrong name: already reported at start-up, do not retry every time

            now = time.time()
            try:
                if name != self._led_applied_name:
                    if self._led_applied_name is not None:
                        self.audio_visual_client.stop_behavior(self._led_applied_name)
                    if name is not None:
                        self.audio_visual_client.run_behavior(name, now + LED_DURATION_SEC, restart=True)
                        last_run = now
                    self._led_applied_name = name
                    self.get_logger().info(f"[LED] stato {str(desired_state).upper()} -> behavior {name!r}")
                elif name is not None and now - last_run >= LED_REFRESH_SEC:
                    self.audio_visual_client.run_behavior(name, now + LED_DURATION_SEC, restart=False)
                    last_run = now
            except Exception as ex:
                # _led_applied_name is not updated on error: retry on the next round.
                self.get_logger().error(f"[LED] aggiornamento fallito: {ex}", throttle_duration_sec=5.0)

    def shutdown_leds(self):
        """Stops the worker and the behavior running on the robot."""
        self._led_stop.set()
        self._led_wake.set()
        if self._led_thread is not None:
            self._led_thread.join(timeout=2.0)
        if self.audio_visual_client is not None and self._led_applied_name is not None:
            try:
                self.audio_visual_client.stop_behavior(self._led_applied_name)
            except Exception:
                pass

    # ==================================================================
    def _start_trigger_thread(self):
        """Starts (or restarts) the thread waiting for ENTER. A Python
        thread cannot be restarted once finished — that is why a new one
        is created every time a new trigger is needed."""
        self._manual_trigger_received = False
        self._stdin_thread = threading.Thread(target=self._wait_for_manual_trigger, daemon=True)
        self._stdin_thread.start()

    def _enter_waiting_trigger(self):
        """Goes back to WAITING_TRIGGER — called when RECOVERY expires
        without finding the target again. Requires pressing ENTER again."""
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
        self._pending_tracker_reset = True
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
        """Publishes the target information (bounding box, distance)."""
        if box_full is None or dist is None:
            return  # no valid target: publish nothing

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
    # Per-state dispatch — the debug image is built and published ALWAYS
    # (every state), detection runs ONLY in SEARCH/TRACKING/RECOVERY.
    # ------------------------------------------------------------------
    def _image_cb(self, rgb_msg):
        # Two SEPARATE clocks: time.time() only to compare against the
        # ROS stamp (queue delay); time.monotonic() for all internal timings.
        t_wall = time.time()
        t_start = time.monotonic()

        image_stamp = rgb_msg.header.stamp.sec + rgb_msg.header.stamp.nanosec * 1e-9
        queue_delay_ms = (t_wall - image_stamp) * 1000.0
        self.get_logger().warn(
            f"[tracking_fsm] _image_cb: latenza tra cattura ed elaborazione: queue delay={queue_delay_ms:.0f}ms",
            throttle_duration_sec=1.0)

        since_last = (t_start - self._last_image_cb_time) * 1000.0 if self._last_image_cb_time else -1.0
        self._last_image_cb_time = t_start

        # LED: only record the desired state (no network call here).
        self._request_led_state(self.state)

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

        # ---- INIT: wait for intrinsics + Detect service ready ----
        if self.state == INIT:
            if self.intrinsics is not None and self.detect_client.client.service_is_ready():
                self.get_logger().info(
                    "INIT completato (intrinseci ricevuti, servizio Detect pronto) — "
                    "in attesa del trigger manuale.")
                self.state = WAITING_TRIGGER
            self._publish_debug(debug_frame, rgb_msg.header)
            return

        # ---- WAITING_TRIGGER: wait for ENTER ----
        if self.state == WAITING_TRIGGER:
            if self._manual_trigger_received:
                self.get_logger().info("Trigger manuale ricevuto — passo a SEARCH.")
                self.state = SEARCH
            self._publish_debug(debug_frame, rgb_msg.header)
            return

        # ---- SEARCH / TRACKING / RECOVERY: detection runs here, on the crop — BLOCKING CALL ----
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

        # State possibly changed during processing: update the LEDs right away.
        self._request_led_state(self.state)

    # ====================================================================
    # METHOD "botsort_hsv" — identity via track_id (BoT-SORT), HSV as a
    # safety net ONLY when the track_id disappears.
    # ====================================================================

    def _handle_search_response_botsort(self, response, frame_bgr, crop_rect, debug_frame, header):
        if response is None:
            self._publish_debug(debug_frame, header)
            return

        crop_x1, crop_y1, _, _ = crop_rect
        depth_image = self._latest_depth_image

        valid_boxes = []  # (box_full, dist, track_id, score)
        for det in response.detections:
            box_full = (det.x1 + crop_x1, det.y1 + crop_y1, det.x2 + crop_x1, det.y2 + crop_y1)
            self._draw_box(debug_frame, box_full, (100, 100, 100), f"{det.class_name} det pre CONFIDENCE check YOLOE SEARCH id {det.track_id}", label_offset=3)

            if det.score < MIN_DETECTION_CONFIDENCE:
                self._draw_box(debug_frame, box_full, (128, 0, 128),
                                f"{det.class_name} (confidenza {det.score:.2f} troppo bassa)", label_offset=2)
                continue

            if depth_image is None:
                self._draw_box(debug_frame, box_full, (0, 255, 255), f"{det.class_name} (depth n/d)", label_offset=2)
                continue
            box_depth = scale_box_to_depth(box_full, frame_bgr.shape, depth_image.shape)
            dist = box_center_depth(depth_image, box_depth)
            if dist is None:
                self._draw_box(debug_frame, box_full, (128, 128, 128), f"{det.class_name} (depth invalida)", label_offset=2)
                continue
            if not (self.min_range <= dist <= self.max_range):
                self._draw_box(debug_frame, box_full, (0, 0, 220), f"{det.class_name} {dist:.2f}m (fuori range)", label_offset=2)
                continue

            if self.reference_embedding is not None:
                candidate_hsv = extract_appearance_embedding(frame_bgr, box_full)
                sim = embedding_similarity(candidate_hsv, self.reference_embedding)
                if sim < REID_SIMILARITY_THRESHOLD:
                    self._draw_box(debug_frame, box_full, (255, 0, 255),
                                    f"{det.class_name} {dist:.2f}m (non e' il target noto, sim={sim:.2f})", label_offset=2)
                    continue

            valid_boxes.append((box_full, dist, det.track_id, det.score))

        if len(valid_boxes) != 1:
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
            label = (f"person {dist:.2f}m score={score:.2f} ({self._stability_count}/{STABILITY_FRAMES_REQUIRED})"
                     if track_id != -1 else f"person {dist:.2f}m score={score:.2f} (non ancora tracciato da BoT-SORT) id={track_id}")
            self._draw_box(debug_frame, box_full, (0, 200, 0), label, label_offset=0)
            self._publish_debug(debug_frame, header)
            return

        self.reference_embedding = extract_appearance_embedding(frame_bgr, box_full)
        self._locked_box = box_full
        self._locked_track_id = track_id
        self.state = TRACKING
        self._stability_count = 0
        self._stability_track_id = -1
        self.get_logger().info(
            f"[botsort_hsv] Target agganciato a {dist:.2f}m (track_id={track_id}) — passo a TRACKING.")
        self._draw_box(debug_frame, box_full, (0, 255, 0), f"TARGET {dist:.2f}m id={track_id}", label_offset=0)
        self._publish_debug(debug_frame, header)

    def _attempt_reacquisition_botsort(self, response, frame_bgr, crop_rect, depth_image, debug_frame):
        """HSV histogram ONLY (never neural)."""
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
            self._draw_box(debug_frame, box_full, (200, 100, 100), f"{det.class_name} det pre_CONFIDENCE BOTSORT_TRACKING/RECOVERY id {det.track_id}", label_offset=3)

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

            combined = (W_SIMILARITY * similarity) - (W_POSITION * (dist_px / target_threshold_px))
            if combined > best_combined:
                best_combined = combined
                best_box, best_dist_m, best_score, best_track_id = box_full, dist_m, similarity, det.track_id

        return best_box, best_track_id, best_dist_m, best_score

    def _handle_track_response_botsort(self, response, frame_bgr, crop_rect, debug_frame, header):
        depth_image = self._latest_depth_image

        if response is not None:
            for det in response.detections:
                box_full = (det.x1 + crop_rect[0], det.y1 + crop_rect[1],
                            det.x2 + crop_rect[0], det.y2 + crop_rect[1])
                self._draw_box(debug_frame, box_full, (100, 100, 100), f"{det.class_name} det pre CONFIDENCE check BOTSORT_TRACKING id {det.track_id}", label_offset=3)

        # ---- FAST PATH ----
        fast_det = None
        crop_x1, crop_y1, _, _ = crop_rect
        if response is not None and self._locked_track_id != -1:
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
            self._reset_reacquisition_confirmation()

            if in_range:
                new_hsv = extract_appearance_embedding(frame_bgr, box_full)
                if new_hsv is not None and self.reference_embedding is not None:
                    self.reference_embedding = (
                        (1 - REID_EMA_ALPHA) * self.reference_embedding + REID_EMA_ALPHA * new_hsv)
                self.get_logger().info(
                    f"TRACKING [botsort_hsv] ok (track_id={self._locked_track_id}): confermato a {dist_m:.2f}m",
                    throttle_duration_sec=1.0)
            else:
                self.get_logger().info(
                    f"TRACKING [botsort_hsv]: track_id={self._locked_track_id} presente ma fuori range "
                    f"— resto agganciato, non conto come perso.", throttle_duration_sec=1.0)

            self._publish_debug(debug_frame, header)
            return

        # ---- The locked track_id is NOT there: attempt with HSV ----
        best_box, best_track_id, best_dist_m, best_score = self._attempt_reacquisition_botsort(
            response, frame_bgr, crop_rect, depth_image, debug_frame)

        if best_box is not None and best_score < REID_COMBINE:
            best_box = None

        committed = False
        if best_box is not None:
            if self._confirm_reacquisition(best_box):
                committed = True
            else:
                self._draw_box(debug_frame, best_box, (255, 165, 0),
                                f"possibile target {best_dist_m:.2f}m "
                                f"(conferma {self._reacquisition_count}/{REACQUISITION_STABILITY_FRAMES})", label_offset=2)
                self.get_logger().info(
                    f"TRACKING [botsort_hsv]: candidato trovato, in attesa di conferma "
                    f"({self._reacquisition_count}/{REACQUISITION_STABILITY_FRAMES})...", throttle_duration_sec=1.0)
        else:
            self._reset_reacquisition_confirmation()

        if committed:
            self._draw_box(debug_frame, best_box, (0, 255, 0), f"TARGET {best_dist_m:.2f}m id={best_track_id}", label_offset=0)
            old_track_id = self._locked_track_id
            self._locked_box = best_box
            self._locked_track_id = best_track_id
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
            best_box = None

        committed = False
        if best_box is not None:
            if self._confirm_reacquisition(best_box):
                committed = True
            else:
                self._draw_box(debug_frame, best_box, (255, 165, 0),
                                f"possibile target {best_dist_m:.2f}m "
                                f"(conferma {self._reacquisition_count}/{REACQUISITION_STABILITY_FRAMES})", label_offset=2)
        else:
            self._reset_reacquisition_confirmation()

        if committed:
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
    # METHOD "embedding_only" — identity ONLY via neural embedding, in
    # EVERY state (SEARCH included). No track_id/BoT-SORT involved.
    # ====================================================================

    def _handle_search_response_embedding(self, response, frame_bgr, crop_rect, debug_frame, header):

        self.reference_embedding = None  # in SEARCH we do not trust any previous embedding — a new stable candidate is needed
        if response is None:
            self._publish_debug(debug_frame, header)
            return

        crop_x1, crop_y1, _, _ = crop_rect
        depth_image = self._latest_depth_image

        valid_boxes = []  # (box_full, dist, embedding, score)
        for det in response.detections:
            box_full = (det.x1 + crop_x1, det.y1 + crop_y1, det.x2 + crop_x1, det.y2 + crop_y1)
            self._draw_box(debug_frame, box_full, (100, 100, 100), f"{det.class_name} det pre MIN_DETECTION_CONFIDENCE check EMBEDDING_ONLY SEARCH", label_offset=3)

            if det.score < MIN_DETECTION_CONFIDENCE:
                continue
            if depth_image is None:
                self._draw_box(debug_frame, box_full, (0, 255, 255), f"{det.class_name} (depth n/d)", label_offset=0)
                continue
            box_depth = scale_box_to_depth(box_full, frame_bgr.shape, depth_image.shape)
            dist = box_center_depth(depth_image, box_depth)
            if dist is None:
                self._draw_box(debug_frame, box_full, (128, 128, 128), f"{det.class_name} (depth invalida)", label_offset=0)
                continue
            if not (self.min_range <= dist <= self.max_range):
                self._draw_box(debug_frame, box_full, (0, 0, 220), f"{det.class_name} {dist:.2f}m (fuori range)", label_offset=0)
                continue

            candidate_embedding = embedding_from_msg(det.embedding)
            emb_preview = candidate_embedding[:5] if candidate_embedding is not None else "n/d"
            self._draw_box(debug_frame, box_full, (0, 255, 0), f"{det.class_name} {dist:.2f}m (embedding: {emb_preview})", label_offset=1)

            if self.reference_embedding is not None:
                sim = rich_neural_embedding_similarity(candidate_embedding, self.reference_embedding, w_cosine=W_COSINE, w_euclidean=W_EUCLIDEAN, w_magnitude=W_MAGNITUDE, euclidean_scale=EUCLIDEAN_SCALE, magnitude_scale=MAGNITUDE_SCALE)
                if sim < REID_SIMILARITY_THRESHOLD:
                    self._draw_box(debug_frame, box_full, (255, 0, 255),
                                    f"{det.class_name} {dist:.2f}m (non e' il target noto, sim={sim:.2f})", label_offset=1)
                    continue

            valid_boxes.append((box_full, dist, candidate_embedding, det.score))

        if len(valid_boxes) != 1:
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
                "SEARCH [embedding_only]: nessun embedding ricevuto — il DetectorNode ha il modello "
                "di ReID configurato? Senza embedding non posso confermare stabilita'.",
                throttle_duration_sec=2.0)
            self._stability_count = 0
            self._stability_reference_embedding = None
            self._publish_debug(debug_frame, header)
            return

        if self._stability_reference_embedding is not None:
            sim = neural_embedding_similarity(box_embedding, self._stability_reference_embedding)
            if sim >= REID_SIMILARITY_THRESHOLD:
                self._stability_count += 1
            else:
                self._stability_count = 1
        else:
            self._stability_count = 1
        self._stability_reference_embedding = box_embedding

        self.get_logger().info(
            f"SEARCH [embedding_only]: 1 box nel raggio a {dist:.2f}m (score={score:.2f}) — "
            f"stabilita' d'aspetto {self._stability_count}/{STABILITY_FRAMES_REQUIRED}",
            throttle_duration_sec=1.0)

        if self._stability_count < STABILITY_FRAMES_REQUIRED:
            self._draw_box(debug_frame, box_full, (0, 200, 0),
                            f"person {dist:.2f}m score={score:.2f} ({self._stability_count}/{STABILITY_FRAMES_REQUIRED})", label_offset=1)
            self._publish_debug(debug_frame, header)
            return

        # Stable for enough frames: lock.
        self.reference_embedding = box_embedding
        self._locked_box = box_full
        self._publish_target_info(box_full, dist, header)
        self.state = TRACKING
        self._stability_count = 0
        self._stability_reference_embedding = None
        self.get_logger().info(f"[embedding_only] Target agganciato a {dist:.2f}m — passo a TRACKING.")
        self._draw_box(debug_frame, box_full, (0, 255, 0), f"TARGET {dist:.2f}m", label_offset=0)
        self._publish_debug(debug_frame, header)

    def _attempt_reacquisition_embedding(self, response, frame_bgr, crop_rect, depth_image, debug_frame):
        """Neural embedding ONLY (never HSV, never track_id). The returned
        best_score = COMBINED score (then compared with REID_COMBINE)."""
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
        was_continuous = (self._tracking_miss_count == 0)
        depth_image = self._latest_depth_image

        if response is not None:
            for det in response.detections:
                box_full = (det.x1 + crop_rect[0], det.y1 + crop_rect[1], det.x2 + crop_rect[0], det.y2 + crop_rect[1])
                self._draw_box(debug_frame, box_full, (100, 100, 100), f"{det.class_name} - {det.score:.2f}", label_offset=3)

        best_box, best_dist_m, best_score, best_embedding = self._attempt_reacquisition_embedding(
            response, frame_bgr, crop_rect, depth_image, debug_frame)

        if best_box is not None and best_score < REID_COMBINE:
            best_box = None

        committed = False
        if best_box is not None:
            if was_continuous:
                committed = True
                self._reset_reacquisition_confirmation()
            elif self._confirm_reacquisition(best_box):
                committed = True
            else:
                self._draw_box(debug_frame, best_box, (255, 165, 0),
                                f"possibile target {best_dist_m:.2f}m "
                                f"(conferma {self._reacquisition_count}/{REACQUISITION_STABILITY_FRAMES})", label_offset=2)
                self.get_logger().info(
                    f"TRACKING [embedding_only]: candidato trovato, in attesa di conferma "
                    f"({self._reacquisition_count}/{REACQUISITION_STABILITY_FRAMES})...", throttle_duration_sec=1.0)
        else:
            self._reset_reacquisition_confirmation()

        if committed:
            self._draw_box(debug_frame, best_box, (0, 255, 0), f"TARGET {best_dist_m:.2f}m", label_offset=0)
            self._locked_box = best_box
            self._tracking_miss_count = 0
            self._publish_target_info(best_box, best_dist_m, header)
            if best_embedding is not None and self.reference_embedding is not None:
                self.reference_embedding = (
                    (1 - REID_EMA_ALPHA) * self.reference_embedding + REID_EMA_ALPHA * best_embedding)
            self.get_logger().info(
                f"TRACKING [embedding_only] ok: confermato a {best_dist_m:.2f}m, combinato={best_score:.2f}",
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

        if response is not None:
            for det in response.detections:
                box_full = (det.x1 + crop_rect[0], det.y1 + crop_rect[1], det.x2 + crop_rect[0], det.y2 + crop_rect[1])
                self._draw_box(debug_frame, box_full, (100, 100, 100), f"{det.class_name} det pre MIN_DETECTION_CONFIDENCE check EMBEDDING_ONLY RECOVERY", label_offset=3)

        depth_image = self._latest_depth_image
        best_box, best_dist_m, best_score, best_embedding = self._attempt_reacquisition_embedding(
            response, frame_bgr, crop_rect, depth_image, debug_frame)

        if best_box is not None and best_score < REID_COMBINE:
            best_box = None

        committed = False
        if best_box is not None:
            if self._confirm_reacquisition(best_box):
                committed = True
            else:
                self._draw_box(debug_frame, best_box, (255, 165, 0),
                                f"possibile target {best_dist_m:.2f}m "
                                f"(conferma {self._reacquisition_count}/{REACQUISITION_STABILITY_FRAMES})", label_offset=2)
        else:
            self._reset_reacquisition_confirmation()

        if committed:
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
                f"combinato={best_score:.2f} — torno a TRACKING.")
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
    # Shared by both methods
    # ------------------------------------------------------------------
    def _confirm_reacquisition(self, candidate_box):
        """The SAME candidate (by position) must be the best match for
        REACQUISITION_STABILITY_FRAMES attempts in a row before being
        confirmed as the recovered target."""
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
        # With nobody subscribed to the debug topic the frame is None:
        # nothing to draw. Without this check, closing rqt raised an
        # exception on every frame and blocked tracking.
        if frame is None or box is None:
            return
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 2

        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)
        h_img, w_img = frame.shape[:2]

        text_x = min(x1, max(0, w_img - text_w - 4))
        line_height = text_h + 10
        text_y = y1 - 8 - label_offset * line_height
        text_y = max(text_h + 2, text_y)

        cv2.rectangle(frame, (text_x, text_y - text_h - 4), (text_x + text_w + 4, text_y + 4), (0, 0, 0), -1)
        cv2.putText(frame, label, (text_x + 2, text_y), font, font_scale, color, thickness, cv2.LINE_AA)

    def _publish_debug(self, debug_frame, header):
        if debug_frame is None:
            return
        msg = self.bridge.cv2_to_compressed_imgmsg(debug_frame, dst_format='jpg')
        msg.header = header
        self.debug_pub.publish(msg)


def main():
    rclpy.init()

    # 1. Spot SDK (recent versions already register AudioVisualClient;
    #    the explicit registration stays harmless for older ones)
    sdk = bosdyn.client.create_standard_sdk('TrackingFSM_LED_Client')
    sdk.register_service_client(AudioVisualClient)

    # 2. Robot
    robot_ip = "192.168.80.3"
    robot = sdk.create_robot(robot_ip)

    # 3. Credentials
    SPOT_USERNAME = "admin"
    SPOT_PASSWORD = "prb4e3wparqx"

    try:
        robot.authenticate(SPOT_USERNAME, SPOT_PASSWORD)
        robot.start_time_sync()   # required: run_behavior converts times to robot time
        print("[SDK SPOT] Connessione gRPC per i LED inizializzata con successo.")
    except Exception as e:
        print(f"[ERRORE SDK SPOT] Impossibile autenticarsi sul robot per i LED: {e}")
        print("[AVVISO] La FSM funzionera' ma i LED rimarranno invariati.")
        robot = None

    node = TrackingFSM(robot=robot)

    # MultiThreadedExecutor is MANDATORY: it lets the client group
    # (self._service_group) process the Detect service response
    # while _image_cb (self._main_group) is blocked in call_sync().
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown_leds()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()