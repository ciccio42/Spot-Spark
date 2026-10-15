#!/usr/bin/env python3
"""
tracking_fsm.py (demo_package) — ROS node + state machine of the person following.

INIT -> WAITING_TRIGGER (ENTER) -> SEARCH -> TRACKING -> RECOVERY -> WAITING_TRIGGER

Two separate methods, chosen with TRACKING_METHOD:
  "embedding_only": identity from the ReID embeddings — logic in tracking_embedding.py
  "botsort_hsv":    identity from BoT-SORT track_id + HSV histogram — in this file, unchanged
Configuration, debug image, metrics and LEDs: tracking_utils.py
"""

import math
import threading
import time
import zlib
from types import SimpleNamespace

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from geometry_msgs.msg import PointStamped
from demo_interfaces.msg import TargetInfoMessage

import bosdyn.client
from bosdyn.client.audio_visual import AudioVisualClient

from demo_package.common import (
    CameraIntrinsics, box_center, distance, extract_appearance_embedding, embedding_similarity,
    compute_roi_crop_rect, scale_box_to_depth, box_center_depth, CONE_MIN_RANGE, CONE_MAX_RANGE,
)
from demo_package.detect_client import DetectClient
from demo_package import tracking_embedding as emb
from demo_package import tracking_utils as U
from demo_package.tracking_utils import INIT, WAITING_TRIGGER, SEARCH, TRACKING, RECOVERY

TRACKING_METHOD = "embedding_only"   # "embedding_only" or "botsort_hsv" — one method per session
B = SimpleNamespace(**U.BOTSORT)     # botsort_hsv values


class TrackingFSM(Node):
    def __init__(self, robot=None):
        super().__init__('tracking_fsm')
        self.leds = U.SpotLeds(robot, self.get_logger())
        self.metrics = U.Metrics(TRACKING_METHOD)
        self.tracker = emb.TargetTracker()
        if TRACKING_METHOD == "embedding_only":
            self.metrics.extra = self._embedding_metrics

        self.bridge = CvBridge()
        self.intrinsics = None
        self._latest_depth_image = None
        self.min_range, self.max_range = CONE_MIN_RANGE, CONE_MAX_RANGE
        self.state = INIT
        self._reset_session()
        self._pending_tracker_reset = False
        self._last_image_crc = None
        self._start_trigger_thread()
        self._setup_ros()
        self.get_logger().info(f"TrackingFSM avviato — METODO: {TRACKING_METHOD}. "
                               f"LED: {'ATTIVI' if self.leds.active else 'DISATTIVATI'}")

    def _setup_ros(self):
        # image, depth, camera_info in one group; the service client in another (else: deadlock
        # while _image_cb waits for the synchronous Detect response)
        main_group, service_group = MutuallyExclusiveCallbackGroup(), ReentrantCallbackGroup()
        self.detect_client = DetectClient(self, U.DETECT_SERVICE, callback_group=service_group)
        qos1 = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.create_subscription(CameraInfo, U.HAND_CAMERA_INFO_TOPIC, self._camera_info_cb,
                                 qos_profile_sensor_data, callback_group=main_group)
        self.create_subscription(Image, U.HAND_DEPTH_TOPIC, self._depth_cb, qos1, callback_group=main_group)
        self.create_subscription(CompressedImage if U.HAND_RGB_COMPRESSED else Image, U.HAND_RGB_TOPIC,
                                 self._image_cb, qos1, callback_group=main_group)
        self.target_pose_pub = self.create_publisher(TargetInfoMessage, U.TARGET_POSE_TOPIC, 1)
        self.debug_pub = self.create_publisher(CompressedImage, U.DEBUG_IMAGE_TOPIC, 1)
        self.kf_pub = self.create_publisher(PointStamped, U.KF_TOPIC, 10)   # for rqt_plot / PlotJuggler

    def _reset_session(self):
        """State of one following session (cleared when going back to WAITING_TRIGGER)."""
        self._tracking_miss_count = 0
        self._recovery_deadline = None
        self._stability_count = 0
        self._stability_reference_embedding = None
        self.tracker.reset()
        # botsort_hsv
        self.reference_embedding = None
        self._locked_box = None
        self._locked_track_id = -1
        self._stability_track_id = -1
        self._reset_reacquisition_confirmation()
        self._pending_tracker_reset = True

    # ==================================================================
    # ROS callbacks
    # ==================================================================
    def _camera_info_cb(self, msg):
        if self.intrinsics is None:
            self.intrinsics = CameraIntrinsics(msg)

    def _depth_cb(self, msg):
        self._latest_depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def _image_cb(self, rgb_msg):
        if self._is_duplicate(rgb_msg):          # same image re-published by the driver: nothing new
            self.metrics.duplicate_frames += 1
            return
        self.metrics.count_frame(self.state)
        self.leds.request(self.state)
        frame = (self.bridge.compressed_imgmsg_to_cv2 if U.HAND_RGB_COMPRESSED else self.bridge.imgmsg_to_cv2)(
            rgb_msg, desired_encoding='bgr8')
        debug = frame.copy() if (self.debug_pub.get_subscription_count() > 0
                                 and self.metrics.total_frames % U.DEBUG_PUBLISH_EVERY_N == 0) else None
        crop_rect = self._crop_rect(frame)
        U.DebugCanvas(debug).crop(crop_rect)
        header = rgb_msg.header

        if self.state in (INIT, WAITING_TRIGGER) or crop_rect is None:
            self._step_idle(debug)
            self._publish_debug(debug, header)
        else:
            response = self._run_detector(frame, crop_rect, header)
            try:
                self._dispatch(response, frame, crop_rect, debug, header)
                if TRACKING_METHOD != "botsort_hsv":     # the botsort handlers publish it themselves
                    self._publish_debug(debug, header)
            except Exception as ex:
                self.get_logger().error(f"Eccezione nell'elaborazione della risposta: {ex}", throttle_duration_sec=2.0)
                self._publish_debug(debug, header)
            self.metrics.proc_ms_sum += (time.monotonic() - self._t_response) * 1000.0
        self.leds.request(self.state)

    def _is_duplicate(self, msg):
        """True if the image bytes are identical to the previous message (CRC32, ~0.1 ms). A
        duplicate would cost a Detect call and give sim_ff = 1.00 (seen in the 13/10 video)."""
        data = getattr(msg, 'data', None)
        if data is None:
            return False
        try:
            crc = zlib.crc32(data if isinstance(data, (bytes, bytearray)) else memoryview(data).cast('B'))
        except (TypeError, ValueError):
            crc = zlib.crc32(bytes(data))
        duplicate = crc == self._last_image_crc
        self._last_image_crc = crc
        return duplicate

    def _dispatch(self, response, frame, crop_rect, debug, header):
        if TRACKING_METHOD == "botsort_hsv":
            if debug is not None:
                cv2.putText(debug, self._header(), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            handler = {SEARCH: self._handle_search_response_botsort, TRACKING: self._handle_track_response_botsort,
                       RECOVERY: self._handle_recovery_response_botsort}[self.state]
            handler(response, frame, crop_rect, debug, header)
        else:
            dets = U.build_detections(response, frame.shape, crop_rect, self._latest_depth_image, frame,
                                      fy=self.intrinsics.fy if self.intrinsics is not None else None)
            if self.state == SEARCH:
                self._search_embedding(dets, debug, header)
            else:
                self._track_embedding(dets, debug, header)

    # ==================================================================
    # FSM helpers
    # ==================================================================
    def _header(self):
        fov = U.FOV_TRACKING_DEG if self.state in (TRACKING, RECOVERY) else U.FOV_SEARCH_DEG
        return f"[{TRACKING_METHOD}] {self.state.upper()} FOV {fov:.0f}"

    def _crop_rect(self, frame):
        if self.intrinsics is None:
            return None
        fov = U.FOV_TRACKING_DEG if self.state in (TRACKING, RECOVERY) else U.FOV_SEARCH_DEG
        return compute_roi_crop_rect(frame.shape[1], frame.shape[0], self.intrinsics, math.radians(fov))

    def _step_idle(self, debug):
        """INIT: wait for intrinsics + Detect service. WAITING_TRIGGER: wait for ENTER."""
        if self.state == INIT and self.intrinsics is not None and self.detect_client.client.service_is_ready():
            self.state = WAITING_TRIGGER
            self.get_logger().info("INIT completato — in attesa del trigger manuale.")
        elif self.state == WAITING_TRIGGER and self._manual_trigger_received:
            self.state = SEARCH
            self.get_logger().info("Trigger manuale ricevuto — passo a SEARCH.")
        U.draw_header_only(debug, [self._header()] + (["premi INVIO per avviare SEARCH"]
                                                      if self.state == WAITING_TRIGGER else []))

    def _run_detector(self, frame, crop_rect, header):
        x1, y1, x2, y2 = crop_rect
        crop_msg = self.bridge.cv2_to_imgmsg(frame[y1:y2, x1:x2], encoding='bgr8')
        crop_msg.header = header
        reset, self._pending_tracker_reset = self._pending_tracker_reset, False
        t_call = time.monotonic()
        response = self.detect_client.call_sync(crop_msg, target_classes=U.TARGET_CLASSES, reset_tracker=reset)
        self._t_response = time.monotonic()
        self.metrics.count_detect((self._t_response - t_call) * 1000.0)
        return response

    def _publish_target_info(self, box, dist, header):
        if box is None or dist is None:
            return
        msg = TargetInfoMessage()
        msg.header = header
        msg.bounding_box = [float(v) for v in box]
        msg.depth_m = float(dist)
        msg.camera_rgb_topic = U.HAND_RGB_TOPIC
        msg.camera_depth_topic = U.HAND_DEPTH_TOPIC
        self.target_pose_pub.publish(msg)

    def _publish_kf(self, header):
        """Target Kalman estimate: x = u (px), y = v (px), z = depth (m). Plot it live with
        rqt_plot /person_follow/target_kf/point/x ... (or PlotJuggler)."""
        row = self.tracker.last_kf
        if row is None:
            return
        msg = PointStamped()
        msg.header = header
        msg.point.x, msg.point.y, msg.point.z = float(row["post_u"]), float(row["post_v"]), float(row["post_d"])
        self.kf_pub.publish(msg)

    def _publish_debug(self, debug, header):
        if debug is None:
            return
        msg = self.bridge.cv2_to_compressed_imgmsg(debug, dst_format='jpg')
        msg.header = header
        self.debug_pub.publish(msg)

    def _go_recovery(self, timeout_sec):
        self.state = RECOVERY
        self._recovery_deadline = time.monotonic() + timeout_sec
        self._tracking_miss_count = 0
        self.get_logger().info(f"Target perso — passo a RECOVERY (timeout {timeout_sec:.0f}s).")

    def _enter_waiting_trigger(self):
        self.state = WAITING_TRIGGER
        self._reset_session()
        self._start_trigger_thread()
        self.get_logger().info("RECOVERY scaduto — torno in WAITING_TRIGGER (premi INVIO).")

    def _start_trigger_thread(self):
        self._manual_trigger_received = False
        threading.Thread(target=self._wait_for_manual_trigger, daemon=True).start()

    def _wait_for_manual_trigger(self):
        try:
            input("\n>>> Premi INVIO in questo terminale per avviare SEARCH...\n")
        except EOFError:
            pass
        self._manual_trigger_received = True

    # ==================================================================
    # METHOD "embedding_only" (identity logic in tracking_embedding.py)
    # ==================================================================
    def _search_embedding(self, dets, debug, header):
        """Exactly one person in range with a stable appearance for STABILITY_FRAMES_REQUIRED frames;
        at the lock: that person = TARGET, the other detections = the people of this call."""
        valid = [d for d in dets if d.dist is not None and self.min_range <= d.dist <= self.max_range
                 and d.embedding is not None]
        cand = valid[0] if len(valid) == 1 else None
        if cand is None:
            self._stability_count, self._stability_reference_embedding = 0, None
        else:
            stable = self._stability_reference_embedding is not None and \
                emb.cosine(cand.embedding, self._stability_reference_embedding) >= emb.SEARCH_STABILITY_MIN_COS
            self._stability_count = self._stability_count + 1 if stable else 1
            self._stability_reference_embedding = cand.embedding
        U.draw_search(debug, self._header(), dets, cand, self._stability_count)
        if cand is None or self._stability_count < emb.STABILITY_FRAMES_REQUIRED:
            return

        self.tracker.lock(cand, [d for d in dets if d is not cand], time.monotonic())
        self.metrics.save_gallery(self.tracker.gallery, "anchor")
        self._stability_count, self._stability_reference_embedding = 0, None
        self._tracking_miss_count = 0
        self.state = TRACKING
        self._publish_target_info(cand.box, cand.dist, header)
        self.get_logger().info(f"[embedding_only] Target agganciato a {cand.dist:.2f}m — passo a TRACKING.")

    def _track_embedding(self, dets, debug, header):
        """TRACKING / RECOVERY: one tracker step; here only counters and state changes."""
        now = time.monotonic()
        res = self.tracker.step(dets, now, recovery=self.state == RECOVERY)
        self._publish_kf(header)
        if res.gallery_added:
            self.metrics.save_gallery(self.tracker.gallery, res.gallery_added)
        if res.seen:
            if not res.occluded:              # occluded: its depth is the occluder's -> not published
                self._publish_target_info(res.det.box, res.det.dist, header)
        else:
            self.metrics.count_miss(self.state, res.reason)

        if self.state == TRACKING:
            self._tracking_miss_count = 0 if res.seen else self._tracking_miss_count + 1
            extra = f"mancati {self._tracking_miss_count}/{emb.TRACKING_GRACE_FRAMES}"
            if self._tracking_miss_count >= emb.TRACKING_GRACE_FRAMES:
                self._go_recovery(emb.RECOVERY_TIMEOUT_SEC)
                self.tracker.enter_recovery()
        else:
            extra = f"{max(0.0, self._recovery_deadline - now):.0f}s rimanenti"
            if res.status == emb.CONFIRMED:
                self.state, self._recovery_deadline = TRACKING, None
                self.get_logger().info("RECOVERY [embedding_only]: target RITROVATO — torno a TRACKING.")
        U.draw_tracking(debug, self._header(), dets, res, self.tracker, extra, now)
        if self.state == RECOVERY and now >= self._recovery_deadline:
            self._enter_waiting_trigger()

    def _embedding_metrics(self):
        tr = self.tracker
        tracking = self.metrics.frames_by_state.get(TRACKING, 0)
        return {
            "target_confirmed_frames": tr.n_confirmed, "target_kept_frames": tr.n_kept,
            "target_kept_pct_of_tracking": round(100.0 * tr.n_kept / tracking, 2) if tracking else 0.0,
            "gallery_views_now": len(tr.gallery), "gallery_added": dict(tr.gallery_added),
            "intruder_blocked_detections": tr.n_intruder_blocked,      # (B) sticky intruders
            "calls_without_near_bonus": tr.n_no_near_bonus_calls,      # (A) somebody at the predicted point
            "sim_stats": tr.sim_stats(),                               # threshold calibration
            "tracker_step_avg_ms": round(tr.step_ms_sum / tr.n_steps, 3) if tr.n_steps else 0.0,
            "tracker_step_max_ms": round(tr.step_ms_max, 3),
            "thresholds": {"TARGET_HIGH": emb.TARGET_HIGH, "TARGET_LOW": emb.TARGET_LOW,
                           "REACQ_THRESHOLD": emb.REACQ_THRESHOLD, "GALLERY_VETO_MIN": emb.GALLERY_VETO_MIN,
                           "INTRUDER_RELEASE_SIM": emb.INTRUDER_RELEASE_SIM,
                           "weights": [emb.W_COSINE, emb.W_EUCLIDEAN, emb.W_MAGNITUDE]},
        }

    # ==================================================================
    # METHOD "botsort_hsv" helpers
    # ==================================================================
    @staticmethod
    def _draw_box(frame, box, color, label, label_offset=0):
        if frame is None or box is None:
            return
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        tx = min(x1, max(0, frame.shape[1] - tw - 4))
        ty = max(th + 2, y1 - 8 - label_offset * (th + 10))
        cv2.rectangle(frame, (tx, ty - th - 4), (tx + tw + 4, ty + 4), (0, 0, 0), -1)
        cv2.putText(frame, label, (tx + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

    def _confirm_reacquisition(self, candidate_box):
        same = self._reacquisition_pending_box is not None and distance(
            box_center(candidate_box), box_center(self._reacquisition_pending_box)) <= B.REACQUISITION_PX_TOLERANCE
        self._reacquisition_count = self._reacquisition_count + 1 if same else 1
        self._reacquisition_pending_box = candidate_box
        if self._reacquisition_count >= B.REACQUISITION_STABILITY_FRAMES:
            self._reset_reacquisition_confirmation()
            return True
        return False

    def _reset_reacquisition_confirmation(self):
        self._reacquisition_pending_box = None
        self._reacquisition_count = 0

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

            if det.score < B.MIN_DETECTION_CONFIDENCE:
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
                if sim < B.REID_SIMILARITY_THRESHOLD:
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
            f"stabilita' {self._stability_count}/{B.STABILITY_FRAMES_REQUIRED}", throttle_duration_sec=1.0)

        if self._stability_count < B.STABILITY_FRAMES_REQUIRED:
            label = (f"person {dist:.2f}m score={score:.2f} ({self._stability_count}/{B.STABILITY_FRAMES_REQUIRED})"
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
        target_threshold_px = B.TARGET_DISTANCE_THRESHOLD_FRAC * math.hypot(w_img, h_img)
        last_center = box_center(self._locked_box)

        best_box, best_combined = None, -float("inf")
        best_dist_m, best_score, best_track_id = None, None, -1

        for det in response.detections:
            box_full = (det.x1 + crop_x1, det.y1 + crop_y1, det.x2 + crop_x1, det.y2 + crop_y1)
            self._draw_box(debug_frame, box_full, (200, 100, 100), f"{det.class_name} det pre_CONFIDENCE BOTSORT_TRACKING/RECOVERY id {det.track_id}", label_offset=3)

            if det.score < B.MIN_DETECTION_CONFIDENCE:
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
            if similarity < B.REID_SIMILARITY_THRESHOLD:
                continue

            dist_px = distance(box_center(box_full), last_center)
            if dist_px > target_threshold_px:
                continue

            combined = (B.W_SIMILARITY * similarity) - (B.W_POSITION * (dist_px / target_threshold_px))
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
                        (1 - B.REID_EMA_ALPHA) * self.reference_embedding + B.REID_EMA_ALPHA * new_hsv)
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

        if best_box is not None and best_score < B.REID_COMBINE:
            best_box = None

        committed = False
        if best_box is not None:
            if self._confirm_reacquisition(best_box):
                committed = True
            else:
                self._draw_box(debug_frame, best_box, (255, 165, 0),
                                f"possibile target {best_dist_m:.2f}m "
                                f"(conferma {self._reacquisition_count}/{B.REACQUISITION_STABILITY_FRAMES})", label_offset=2)
                self.get_logger().info(
                    f"TRACKING [botsort_hsv]: candidato trovato, in attesa di conferma "
                    f"({self._reacquisition_count}/{B.REACQUISITION_STABILITY_FRAMES})...", throttle_duration_sec=1.0)
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
                    (1 - B.REID_EMA_ALPHA) * self.reference_embedding + B.REID_EMA_ALPHA * new_hsv)
            self.get_logger().info(
                f"TRACKING [botsort_hsv]: target CONFERMATO a {best_dist_m:.2f}m, similarity={best_score:.2f} — "
                f"track_id {old_track_id} -> {best_track_id}", throttle_duration_sec=1.0)
            self._publish_debug(debug_frame, header)
            return

        self._tracking_miss_count += 1
        if self._tracking_miss_count < B.TRACKING_GRACE_FRAMES:
            self.get_logger().info(
                f"TRACKING [botsort_hsv]: target non trovato — tentativo "
                f"{self._tracking_miss_count}/{B.TRACKING_GRACE_FRAMES} prima di passare a RECOVERY.",
                throttle_duration_sec=1.0)
            self._publish_debug(debug_frame, header)
            return

        self.state = RECOVERY
        self._recovery_deadline = time.monotonic() + B.RECOVERY_TIMEOUT_SEC
        self._tracking_miss_count = 0
        self.get_logger().info(
            f"TRACKING [botsort_hsv]: target non ritrovato entro {B.TRACKING_GRACE_FRAMES} frame — "
            f"passo a RECOVERY (timeout {B.RECOVERY_TIMEOUT_SEC:.0f}s).")
        self._publish_debug(debug_frame, header)

    def _handle_recovery_response_botsort(self, response, frame_bgr, crop_rect, debug_frame, header):
        depth_image = self._latest_depth_image
        best_box, best_track_id, best_dist_m, best_score = self._attempt_reacquisition_botsort(
            response, frame_bgr, crop_rect, depth_image, debug_frame)

        if best_box is not None and best_score < B.REID_COMBINE:
            best_box = None

        committed = False
        if best_box is not None:
            if self._confirm_reacquisition(best_box):
                committed = True
            else:
                self._draw_box(debug_frame, best_box, (255, 165, 0),
                                f"possibile target {best_dist_m:.2f}m "
                                f"(conferma {self._reacquisition_count}/{B.REACQUISITION_STABILITY_FRAMES})", label_offset=2)
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
                    (1 - B.REID_EMA_ALPHA) * self.reference_embedding + B.REID_EMA_ALPHA * new_hsv)
            self.get_logger().info(
                f"RECOVERY [botsort_hsv]: target CONFERMATO a {best_dist_m:.2f}m, similarity={best_score:.2f} — "
                f"track_id {old_track_id} -> {best_track_id} — torno a TRACKING.")
            self._publish_debug(debug_frame, header)
            return

        remaining = self._recovery_deadline - time.monotonic()
        if remaining <= 0:
            self.get_logger().info(
                f"RECOVERY [botsort_hsv]: timeout di {B.RECOVERY_TIMEOUT_SEC:.0f}s scaduto — "
                f"torno in WAITING_TRIGGER.")
            self._enter_waiting_trigger()
        else:
            self.get_logger().info(
                f"RECOVERY [botsort_hsv]: nessuna corrispondenza confermata — {remaining:.0f}s rimanenti.",
                throttle_duration_sec=1.0)
        self._publish_debug(debug_frame, header)


def main():
    rclpy.init()
    sdk = bosdyn.client.create_standard_sdk('TrackingFSM_LED_Client')
    sdk.register_service_client(AudioVisualClient)
    robot = sdk.create_robot("192.168.80.3")
    try:
        robot.authenticate("admin", "prb4e3wparqx")
        robot.start_time_sync()   # run_behavior converts times to robot time
    except Exception as e:
        print(f"[ERRORE SDK SPOT] autenticazione per i LED fallita: {e} — la FSM funziona, LED invariati.")
        robot = None
    node = TrackingFSM(robot=robot)
    executor = MultiThreadedExecutor(num_threads=4)   # mandatory: see _setup_ros
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.metrics.save(final=True)
        node.leds.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
