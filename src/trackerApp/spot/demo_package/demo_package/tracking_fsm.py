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

embedding_only, identity of the target (see the constant blocks for details):
  - MULTI-VIEW GALLERY: the target is a set of views (front, back...),
    similarity = maximum over the views (REID_GALLERY_*, GALLERY_*).
  - HYSTERESIS: strong threshold to lock / re-lock, lower HOLD threshold
    only to KEEP a target seen a few frames ago (REID_HOLD_THRESHOLD, HOLD_*).
  - INTRUDER memory during occlusions, Kalman prediction in the image
    (decides WHERE to look, never WHO the target is).
  - CSV log of the three ReID components (REID_LOG_COMPONENTS).

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

import csv
import datetime
import json
import os
from collections import Counter

import cv2
import numpy as np
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
    rich_neural_embedding_components, combine_reid_components,
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
TARGET_CLASSES = ['person','quadruped','quadruped animal','quadruped robot','robotic dog','four-legged robot', 'dog', 'robot','humanoid robot']
DEBUG_IMAGE_TOPIC = '/person_follow/hand_debug/compressed'  # /compressed suffix: image_transport
                                                               # convention, same as /camera/hand/compressed

REID_SIMILARITY_THRESHOLD = 0.60  # minimum similarity threshold to be a candidate
MIN_DETECTION_CONFIDENCE = 0.10   # MINIMUM confidence of the YOLOE detection itself (not the appearance)
REID_EMA_ALPHA = 0.3
REID_EMA_MIN_SIM = 0.70           # the reference is updated (EMA) only with matches at least this similar:
                                   # a borderline match (just above REID_SIMILARITY_THRESHOLD) can still be
                                   # followed, but it never pulls the reference towards itself. Without it a
                                   # single wrong borderline lock (e.g. a stranger standing where the target
                                   # is predicted, right after the target left) would make the stranger more
                                   # and more "similar" frame after frame. None = always update (old behaviour).
TARGET_DISTANCE_THRESHOLD_FRAC = 0.30  # how far the target can move (in pixels, as a fraction of
                                         # the image diagonal) from one frame to the next
STABILITY_FRAMES_REQUIRED = 10    # consecutive "stable" frames before locking in SEARCH
TRACKING_GRACE_FRAMES = 30        # in TRACKING: attempt frames before switching to RECOVERY
RECOVERY_TIMEOUT_SEC = 15.0       # in RECOVERY: seconds of REAL time before giving up
REACQUISITION_STABILITY_FRAMES = 2   # consecutive attempts to confirm a re-lock
REACQUISITION_PX_TOLERANCE = 60.0    # maximum movement between attempts to count as "same candidate"
STOPPING_DISTANCE = 1.0

REID_COMBINE = 0.30  # minimum threshold on the WINNER's score (best_score)

W_SIMILARITY = 0.7
W_POSITION = 0.3

W_COSINE = 0.5
W_EUCLIDEAN = 1.0
W_MAGNITUDE = 0.8

EUCLIDEAN_SCALE = 10.0
MAGNITUDE_SCALE = 10.0

FOV_SEARCH_DEG= 35.0
FOV_TRACKING_DEG = 60.0

METRICS_DIR ='/home/spot_ws/src/codice_carmine/metriche'
METRICS_SAVE_PERIOD_SEC = 5.0

# ============================================================
# TRACKING / RECOVERY (embedding_only) — IDENTITY FIRST
# The target is ALWAYS decided by the ReID: in every state and every frame a
# candidate must pass the same strict test (combined similarity >=
# REID_SIMILARITY_THRESHOLD, winner's combined score >= REID_COMBINE).
# Nothing below can admit someone who fails it.
# ============================================================
SHORT_GAP_FRAMES = 2         # up to this many missed frames the target is re-locked AT ONCE (same strict
                               # identity test, near the last confirmed position or the prediction,
                               # unambiguous); after that the WHOLE frame is searched and a multi-frame
                               # confirmation is required. See also PREDICTION_GAP_FRAMES.
REID_AMBIGUITY_MARGIN = 0.08 # whole-frame search: the winner must be more similar than EVERY other person
                               # in the frame by at least this, otherwise nobody is taken this frame
REACQ_IMMEDIATE_MIN_SIM = 0.60  # embedding_only, re-lock AFTER a long loss (whole-frame search, RECOVERY):
                               # IMMEDIATE when the winner's similarity is >= this. The winner has already
                               # passed EVERY identity check (similarity >= REID_SIMILARITY_THRESHOLD, clearly
                               # more similar than every other person in view, not more similar to a known
                               # intruder, REID_COMBINE): the multi-frame confirmation only adds delay.
                               # Set it higher (e.g. 0.75) to require the REACQUISITION_STABILITY_FRAMES
                               # confirmation for borderline matches only; None = always confirm (old
                               # behaviour). botsort_hsv is not affected.
INTRUDER_KEPT_WHILE_LOST = True  # a known intruder is NOT forgotten while the target is lost (TRACKING misses,
                               # RECOVERY): it is forgotten only OCCLUSION_HOLD_FRAMES frames after the target
                               # has been confirmed again. Otherwise an occluder that walks away while the
                               # target is still hidden would lose its "intruder" label and could be re-locked.
REACQUISITION_MAX_GAP = 2    # empty frames (no candidate, e.g. a ToF hole) tolerated during a pending
                               # confirmation before it starts over
OCCLUSION_COVERAGE = 0.20    # another person covering more than this fraction of the target box = occlusion
OCCLUSION_NEAR_MARGIN = 0.5  # while the target is CONFIRMED, a person whose box touches the target box widened by
                               # this fraction of its width on each side is already treated as a (future)
                               # occluder: its embedding is learnt BEFORE it covers the target. Between
                               # "20% overlap" and "target hidden" there may be no frame at all at a low frame
                               # rate, and then the intruder would never be learnt. None = only the 20% rule.
OCCLUSION_HOLD_FRAMES = 6    # occlusion rules stay active this many frames after the overlap ends.
                               # During an occlusion the embedding of the OTHER person (learnt while both
                               # were visible) is used too: a candidate more similar to it than to the
                               # target is not the target, whatever its similarity to the target.
TRACK_MIN_RANGE = 0.3        # while TRACKING/RECOVERY the accepted distance is wider than in SEARCH,
TRACK_MAX_RANGE = 6.0        # and a missing depth is tolerated: they never change WHO the target is

# Image-space Kalman prediction of the target box centre (blue dot in the debug image).
# It decides WHERE to look, never WHO the target is: a box near the prediction must still
# pass the same strict identity test (REID_SIMILARITY_THRESHOLD, REID_COMBINE, ambiguity,
# intruder memory). The prediction can only WIDEN the search area, never narrow it: the
# area around the last confirmed position stays valid too.
# Motion while the target is missing is done by nav2_bridge, with its own prediction in odom.
KF_PROCESS_VAR_PX = 5000.0
KF_MEASUREMENT_VAR_PX = 100.0
PREDICTION_MAX_SEC = 2.0     # after this long without seeing the target the prediction is no longer used
GATE_SIGMA_K = 3.0           # prediction gate radius = base radius + GATE_SIGMA_K * std of the prediction
GATE_MAX_FRAC = 0.45         # ...capped at this fraction of the image diagonal
PREDICTION_FREE_SIGMA = 1.0  # position term: a distance from the prediction within this many std of the
                               # prediction costs nothing (being 40 px off a prediction that is uncertain by
                               # 40 px is "exactly where expected"). Only the POSITION term: the identity
                               # thresholds are untouched.
PREDICTION_GAP_FRAMES = 5    # with a usable prediction (fresh, inside the image) and NO known intruder (a person
                               # seen close to / overlapping the CONFIRMED target, see _update_occlusion_state),
                               # the target is re-locked at
                               # once (same strict identity test, inside the gates) up to this many missed
                               # frames, instead of SHORT_GAP_FRAMES. With a known intruder the window stays
                               # SHORT_GAP_FRAMES and the whole-frame search with confirmation takes over.

# ============================================================
# HOLD THRESHOLD (embedding_only) — hysteresis on the ReID similarity
# Two thresholds: REID_SIMILARITY_THRESHOLD (strong) is needed to LOCK and to RE-LOCK after a
# long loss (whole-frame search, RECOVERY) and is unchanged. REID_HOLD_THRESHOLD (weak) only
# KEEPS a target seen a few frames ago (continuous mode) when its similarity drops, e.g. seen
# from behind or against the light. A weak candidate (REID_HOLD_THRESHOLD <= sim <
# REID_SIMILARITY_THRESHOLD) is kept only if ALL of these hold:
#   1. position: it is the box nearest to the prediction / last confirmed position, with its
#      centre within HOLD_MAX_OFFSET_W box widths (whoever walks NEXT TO the target is excluded);
#   2. relative identity: more similar than every person NEAR the expected position by
#      HOLD_RIVAL_MARGIN (there position cannot help: crossing); at least as similar as every
#      person FAR from it (there position separates them, and breaks a tie);
#   3. depth: valid, and within HOLD_MAX_DEPTH_JUMP_M of the last confirmed distance (whoever
#      walks IN FRONT of the target is closer to the camera and is excluded);
#   4. intruder: with a known intruder, sim_target - sim_intruder >= HOLD_INTRUDER_MARGIN.
# A weak match is followed (published to nav2, Kalman updated) but it NEVER updates the target
# reference (EMA), NEVER learns a new intruder and NEVER makes the known intruder be forgotten:
# only a strong match does. No extra model: only the embeddings the DetectorNode already sends.
# ============================================================
REID_HOLD_THRESHOLD = 0.45     # None = hold threshold disabled (strict single threshold, old behaviour)
HOLD_MAX_OFFSET_W = 0.5        # max distance of the weak box centre from the expected position, in box widths
HOLD_RIVAL_MARGIN = 0.10       # the weak candidate must beat every NEAR rival (see below) by this
HOLD_RIVAL_RADIUS_W = 1.0      # a rival is NEAR if its centre is within this many box widths of the expected
                               # position (last confirmed box / Kalman prediction). Near rival = position
                               # cannot tell them apart (crossing, overlapping): APPEARANCE must decide, with
                               # HOLD_RIVAL_MARGIN. Far rival = position already separates them (e.g. walking
                               # side by side): position breaks a TIE, but a far rival MORE similar than the
                               # candidate still blocks it (position never overrides appearance).
                               # None = every person in the gate is a near rival (stricter, previous behaviour).
HOLD_MAX_DEPTH_JUMP_M = 0.8    # max jump from the last confirmed distance (m)
HOLD_INTRUDER_MARGIN = 0.10    # with a known intruder: sim_target - sim_intruder >= this

# ============================================================
# MULTI-VIEW GALLERY (embedding_only) — the target reference is a SET of views, not one vector
# Similarity to the target = MAXIMUM over the views of the gallery (front, back, side, against
# the light...). The views are stored as they are, never averaged: no view is "diluted" into
# the others and the norm of the stored vectors does not shrink (the magnitude and Euclidean
# terms of the similarity are sensitive to the norm, an EMA average is not).
# View 0 is the one locked in SEARCH (anchor): it is NEVER removed.
# A new view is added ONLY if it is NEW (similarity to every stored view < GALLERY_NOVELTY_SIM)
# and only in two cases:
#   a) STRONG match with sim >= GALLERY_STRONG_ADD_MIN_SIM, in a clean frame (no other person
#      in the gate or close to the target, no occlusion / known intruder);
#   b) CHAIN of weak matches: GALLERY_CHAIN_FRAMES consecutive frames kept by the hold
#      threshold, starting right after a clean strong match (anchor), with NO missed frame in
#      between, every frame clean and continuous with the previous one (box IoU >=
#      GALLERY_CHAIN_MIN_IOU, depth step <= GALLERY_CHAIN_MAX_DEPTH_STEP_M). This is how the
#      back view of the target enters the gallery: the identity is guaranteed by the
#      continuity from a strong match, not by the similarity.
# When the gallery is full, the oldest view (except the anchor) is replaced.
# With the gallery active the EMA of the reference is NOT used (the gallery replaces it).
# ============================================================
REID_GALLERY_SIZE = 8                 # 0 = gallery disabled: single reference + EMA (previous behaviour)
GALLERY_NOVELTY_SIM = 0.80            # a view is "new" if its similarity to EVERY stored view is below this
GALLERY_STRONG_ADD_MIN_SIM = 0.70     # strong matches added only from this similarity (same as REID_EMA_MIN_SIM)
GALLERY_CHAIN_FRAMES = 4              # weak frames in a row (after a strong anchor) before adding a view
GALLERY_CHAIN_MIN_IOU = 0.5           # chain continuity: IoU with the box of the previous frame
GALLERY_CHAIN_MAX_DEPTH_STEP_M = 0.3  # chain continuity: depth change from the previous frame (m)
GALLERY_ALONE_MARGIN = 1.0            # "clean" frame: no other box touches the target box widened by this
                                      # fraction of its width on each side

# ============================================================
# ReID COMPONENT LOG (embedding_only) — for choosing / tuning the similarity on real data
# One CSV row per person compared with the target in TRACKING/RECOVERY: the three components
# (cosine, euclidean_sim, magnitude_sim), the combined similarity and, once the frame is
# decided, the label: "target_strong", "target_weak" (the person that became the target this
# frame), "other" (another person in a frame where the target was confirmed), "unconfirmed"
# (frame without a confirmed target: identity unknown). Analysis: analizza_reid_components.py
# ============================================================
REID_LOG_COMPONENTS = True

# ---- Speed ----
DEDUPE_IOU = 0.6             # embedding_only: overlapping boxes (same subject under two prompts) are merged
DEBUG_PUBLISH_EVERY_N = 2    # debug image only every N frames (copy + drawing + JPEG); 1 = always


class _ConstantVelocityKalman1D:
    """1D constant-velocity Kalman filter (same filter as the motion node).
    dt is the REAL elapsed time between frames: the frame rate is irregular."""

    def __init__(self, process_var, measurement_var):
        self.pos, self.vel = 0.0, 0.0
        self.P = [[1e3, 0.0], [0.0, 1e3]]
        self.q, self.r = process_var, measurement_var

    def reset(self, pos):
        self.pos, self.vel = pos, 0.0
        self.P = [[self.r, 0.0], [0.0, 1e4]]

    def predict(self, dt):
        self.pos += self.vel * dt
        p00, p01, p10, p11 = self.P[0][0], self.P[0][1], self.P[1][0], self.P[1][1]
        self.P = [[p00 + dt * (p10 + p01) + dt * dt * p11 + self.q * dt, p01 + dt * p11],
                  [p10 + dt * p11, p11 + self.q * dt]]

    def update(self, z):
        innovation = z - self.pos
        s_ = self.P[0][0] + self.r
        k_pos, k_vel = self.P[0][0] / s_, self.P[1][0] / s_
        self.pos += k_pos * innovation
        self.vel += k_vel * innovation
        p00, p01 = self.P[0][0], self.P[0][1]
        self.P = [[self.P[0][0] - k_pos * p00, self.P[0][1] - k_pos * p01],
                  [self.P[1][0] - k_vel * p00, self.P[1][1] - k_vel * p01]]

    def pos_std(self):
        """Standard deviation of the predicted position (px): grows with every frame
        predicted without a measurement, shrinks with every update."""
        return math.sqrt(max(self.P[0][0], 0.0))



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
        self.fov_search_rad = math.radians(FOV_SEARCH_DEG)
        self.fov_tracking_rad = math.radians(FOV_TRACKING_DEG)
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
        self._reacquisition_pending_time = None # time.monotonic() when it was last seen
        self._reacquisition_count = 0
        self._reacquisition_gap = 0      # consecutive empty frames during a pending confirmation

        self._pending_tracker_reset = False  # botsort_hsv ONLY

        # Prediction of the target in the image (embedding_only): WHERE to look, never WHO
        self._kf_u = _ConstantVelocityKalman1D(KF_PROCESS_VAR_PX, KF_MEASUREMENT_VAR_PX)
        self._kf_v = _ConstantVelocityKalman1D(KF_PROCESS_VAR_PX, KF_MEASUREMENT_VAR_PX)
        self._kf_active = False
        self._kf_last_predict_time = None
        self._kf_last_update_time = None

        # Occlusion (embedding_only)
        self._occlusion_hold = 0           # > 0: someone overlaps / overlapped the target
        self._intruder_embedding = None    # embedding of whoever overlaps the target

        # Hold threshold (embedding_only)
        self._locked_dist = None           # last confirmed distance (m), for the depth-consistency check

        # Multi-view gallery (embedding_only)
        self._gallery = []                 # stored views of the target; [0] = anchor from SEARCH
        self._gallery_chain_anchored = False  # True right after a clean strong match
        self._gallery_chain_len = 0        # consecutive clean, continuous weak frames since the anchor
        self._gallery_prev_box = None      # box / depth of the previous committed frame (chain continuity)
        self._gallery_prev_dist = None

        # ReID component log (embedding_only)
        self._reid_rows = []               # rows of the current frame, labelled once the frame is decided
        self._reid_csv_file = None
        self._reid_csv_writer = None

        self._metrics_start_wall = time.time()
        self._metrics_start = time.monotonic()
        self._metrics_total_frames = 0
        self._metrics_frames_by_state = {s : 0 for s in (INIT, WAITING_TRIGGER, SEARCH, TRACKING, RECOVERY)}
        self._metrics_miss_reasons = Counter()  # why the target was not confirmed, per state
        self._metrics_detect_ms_sum = 0.0       # round-trip of the Detect service call
        self._metrics_detect_ms_max = 0.0
        self._metrics_proc_ms_sum = 0.0         # our processing of the response
        self._metrics_detect_calls = 0
        self._metrics_weak_hold_frames = 0      # frames kept thanks to the hold threshold (embedding_only)
        self._metrics_gallery_added = Counter() # views added to the gallery, by source ("strong", "weak_chain")

        self._metrics_last_save = self._metrics_start
        self._metrics_final_saved = False
        
        os.makedirs(METRICS_DIR, exist_ok=True)
        stamp = datetime.datetime.fromtimestamp(self._metrics_start_wall).strftime("%Y%m%d_%H%M%S")
        
        self._metrics_path = os.path.join(METRICS_DIR, f"tracking_metrics_{TRACKING_METHOD}_{stamp}.json")

        if REID_LOG_COMPONENTS and TRACKING_METHOD == "embedding_only":
            self._reid_csv_path = os.path.join(METRICS_DIR, f"reid_components_{stamp}.csv")
            try:
                self._reid_csv_file = open(self._reid_csv_path, "w", newline="")
                self._reid_csv_writer = csv.DictWriter(self._reid_csv_file, fieldnames=self._REID_CSV_FIELDS)
                self._reid_csv_writer.writeheader()
            except OSError as ex:
                self.get_logger().error(f"[ReID log] impossibile aprire {self._reid_csv_path}: {ex} — log disattivato.")
                self._reid_csv_file = self._reid_csv_writer = None

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
    # Metrics — frame counts, time, state distribution, FPS, etc. Saved in JSON
    # ==================================================================
    def _count_frame(self):
        """Counts the frame for metrics purposes, and saves the JSON every METRICS_SAVE_PERIOD_SEC seconds."""
        self._metrics_total_frames += 1
        self._metrics_frames_by_state[self.state] = self._metrics_frames_by_state.get(self.state, 0) + 1
        now = time.monotonic()
        if now - self._metrics_last_save >= METRICS_SAVE_PERIOD_SEC:
            self.save_metrics(final=False)
            self._metrics_last_save = now

    def _metrics_snapshot(self, final):
        duration = time.monotonic() - self._metrics_start
        by_state = dict(self._metrics_frames_by_state)
        total = self._metrics_total_frames
        active = by_state[SEARCH] + by_state[TRACKING] + by_state[RECOVERY]
        pct = lambda n, d: round(100.0 * n / d, 2) if d else 0.0
        return {
            "tracking_method": TRACKING_METHOD,
            "start_time": datetime.datetime.fromtimestamp(self._metrics_start_wall).isoformat(timespec="seconds"),
            "end_time": datetime.datetime.now().isoformat(timespec="seconds"),
            "final": final,
            "duration_s": round(duration, 2),
            "total_frames": total,
            "frames_by_state": by_state,
            "tracking_frames": by_state[TRACKING],
            "recovery_frames": by_state[RECOVERY],
            "active_frames": active,
            "tracking_pct_of_total": pct(by_state[TRACKING], total),
            "recovery_pct_of_total": pct(by_state[RECOVERY], total),
            "tracking_pct_of_active": pct(by_state[TRACKING], active),
            "recovery_pct_of_active": pct(by_state[RECOVERY], active),
            "effective_fps": round(total / duration, 2) if duration > 0 else 0.0,
            "miss_reasons": dict(self._metrics_miss_reasons.most_common()),
            "reid_hold_threshold": REID_HOLD_THRESHOLD,
            "weak_hold_frames": self._metrics_weak_hold_frames,
            "weak_hold_pct_of_tracking": pct(self._metrics_weak_hold_frames, by_state[TRACKING]),
            "reid_gallery_size_max": REID_GALLERY_SIZE,
            "gallery_views_now": len(self._gallery),
            "gallery_added": dict(self._metrics_gallery_added),
            "avg_detect_roundtrip_ms": round(self._metrics_detect_ms_sum / self._metrics_detect_calls, 1)
                                       if self._metrics_detect_calls else 0.0,
            "max_detect_roundtrip_ms": round(self._metrics_detect_ms_max, 1),
            "avg_processing_ms": round(self._metrics_proc_ms_sum / self._metrics_detect_calls, 1)
                                 if self._metrics_detect_calls else 0.0,
        }

    def save_metrics(self, final=False):
        """Saves the metrics snapshot in JSON format. If final=True, also appends a summary row to metrics_summary.csv."""
        if self._metrics_final_saved:
            return
        try:
            data = self._metrics_snapshot(final)
            tmp = self._metrics_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, self._metrics_path)
            if final:
                self._metrics_final_saved = True
                row = {k: v for k, v in data.items()
                       if k not in ("frames_by_state", "final", "miss_reasons", "gallery_added")}
                row.update({f"frames_{s}": n for s, n in data["frames_by_state"].items()})
                row["miss_reasons"] = json.dumps(data["miss_reasons"])
                row["gallery_added"] = json.dumps(data["gallery_added"])
                fieldnames = list(row.keys())
                # A CSV written by an older version has different columns: never append
                # misaligned rows to it, start a new file instead.
                summary = os.path.join(METRICS_DIR, "metrics_summary_v2.csv")
                new_file = not os.path.exists(summary)
                if not new_file:
                    with open(summary, newline="") as f:
                        header = next(csv.reader(f), [])
                    if header != fieldnames:
                        summary = os.path.join(
                            METRICS_DIR, f"metrics_summary_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv")
                        new_file = True
                with open(summary, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=fieldnames)
                    if new_file:
                        w.writeheader()
                    w.writerow(row)
                print(f"[METRICHE] salvate: {self._metrics_path} (+ riga in {summary})")
        except Exception as ex:
            print(f"[METRICHE] salvataggio fallito: {ex}")
    
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
        self._reacquisition_gap = 0
        self._pending_tracker_reset = True
        self._kf_active = False
        self._occlusion_hold = 0
        self._intruder_embedding = None
        self._locked_dist = None
        self._gallery = []
        self._gallery_chain_reset()
        self._gallery_prev_box = self._gallery_prev_dist = None
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
        
        self._count_frame()

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

        publish_debug = (self.debug_pub.get_subscription_count() > 0
                         and self._metrics_total_frames % DEBUG_PUBLISH_EVERY_N == 0)
        debug_frame = frame_bgr.copy() if publish_debug else None

        crop_rect = None
        fov_raf = self.fov_tracking_rad if self.state in (TRACKING, RECOVERY) else self.fov_search_rad
        if self.intrinsics is not None:
            crop_rect = compute_roi_crop_rect(w_img, h_img, self.intrinsics, fov_raf)

        if debug_frame is not None:
            if crop_rect is not None:
                cv2.rectangle(debug_frame, (crop_rect[0], crop_rect[1]), (crop_rect[2], crop_rect[3]),
                              (0, 200, 255), 2)
            views = (f" viste {len(self._gallery)}" if TRACKING_METHOD == "embedding_only" and REID_GALLERY_SIZE
                     and self._gallery else "")
            cv2.putText(debug_frame, f"[{TRACKING_METHOD}] {self.state.upper()} FOV {math.degrees(fov_raf):.0f}{views}", (10, 30),
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
        t_call = time.monotonic()
        response = self.detect_client.call_sync(crop_msg, target_classes=TARGET_CLASSES, reset_tracker=reset_now)

        self._last_response_time = time.monotonic()
        detect_ms = (self._last_response_time - t_call) * 1000.0
        self._metrics_detect_calls += 1
        self._metrics_detect_ms_sum += detect_ms
        self._metrics_detect_ms_max = max(self._metrics_detect_ms_max, detect_ms)

        if TRACKING_METHOD == "embedding_only" and response is not None:
            response.detections = self._dedupe_detections(response.detections)

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
            self._reid_rows = []  # rows of a frame that was never decided: not logged
            self._publish_debug(debug_frame, rgb_msg.header)

        self._metrics_proc_ms_sum += (time.monotonic() - self._last_response_time) * 1000.0

        # State possibly changed during processing: update the LEDs right away.
        self._request_led_state(self.state)

    @staticmethod
    def _iou(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / union if union > 0 else 0.0

    def _dedupe_detections(self, detections):
        """Class-agnostic suppression of duplicates: the same subject detected under
        two prompts ('dog' and 'quadruped') arrives as two overlapping boxes, which
        would block SEARCH ("exactly 1 box"). Keeps the highest-score box among those
        overlapping more than DEDUPE_IOU."""
        kept = []
        for det in sorted(detections, key=lambda d: d.score, reverse=True):
            box = (det.x1, det.y1, det.x2, det.y2)
            if all(self._iou(box, (k.x1, k.y1, k.x2, k.y2)) <= DEDUPE_IOU for k in kept):
                kept.append(det)
        return kept

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
        self._gallery = [box_embedding]          # view 0 = anchor, never removed
        self._gallery_chain_anchored = True      # the lock itself is a strong, clean match
        self._gallery_chain_len = 0
        self._gallery_prev_box, self._gallery_prev_dist = box_full, dist
        self._locked_box = box_full
        self._kf_reset(box_full)
        self._tracking_miss_count = 0
        self._reset_reacquisition_confirmation()
        self._occlusion_hold = 0
        self._intruder_embedding = None
        self._locked_dist = dist
        self._publish_target_info(box_full, dist, header)
        self.state = TRACKING
        self._stability_count = 0
        self._stability_reference_embedding = None
        self.get_logger().info(f"[embedding_only] Target agganciato a {dist:.2f}m — passo a TRACKING.")
        self._draw_box(debug_frame, box_full, (0, 255, 0), f"TARGET {dist:.2f}m", label_offset=0)
        self._publish_debug(debug_frame, header)

    # ------------------------------------------------------------------
    # Prediction of the target position in the image (blue dot): decides
    # WHERE to look, never WHO the target is.
    # ------------------------------------------------------------------
    def _kf_reset(self, box):
        u, v = box_center(box)
        self._kf_u.reset(u)
        self._kf_v.reset(v)
        now = time.monotonic()
        self._kf_last_predict_time = now
        self._kf_last_update_time = now
        self._kf_active = True

    def _kf_predict_now(self):
        if not self._kf_active:
            return
        now = time.monotonic()
        dt = now - self._kf_last_predict_time
        self._kf_last_predict_time = now
        if dt > 0.0 and now - self._kf_last_update_time <= PREDICTION_MAX_SEC:
            self._kf_u.predict(dt)
            self._kf_v.predict(dt)

    def _kf_update(self, box):
        if not self._kf_active or time.monotonic() - self._kf_last_update_time > PREDICTION_MAX_SEC:
            # Re-lock after a long loss (e.g. from RECOVERY): the old velocity means
            # nothing any more, and an update with a far measurement would give the
            # filter an absurd velocity -> start again from the new position.
            self._kf_reset(box)
            return
        u, v = box_center(box)
        self._kf_u.update(u)
        self._kf_v.update(v)
        self._kf_last_update_time = time.monotonic()

    def _kf_fresh(self):
        """True if the prediction can be used: filter active and target confirmed
        less than PREDICTION_MAX_SEC ago."""
        return self._kf_active and time.monotonic() - self._kf_last_update_time <= PREDICTION_MAX_SEC

    def _kf_prediction(self, frame_shape):
        """(u, v, radius, std) of the prediction gate, or None if the prediction is not fresh.
        radius = base radius (TARGET_DISTANCE_THRESHOLD_FRAC of the diagonal) + GATE_SIGMA_K
        standard deviations of the prediction, capped at GATE_MAX_FRAC of the diagonal:
        the longer the target is not seen, the wider the area where it can be."""
        if not self._kf_fresh():
            return None
        h_img, w_img = frame_shape[:2]
        diag = math.hypot(w_img, h_img)
        u, v = self._kf_u.pos, self._kf_v.pos
        if not (0.0 <= u < w_img and 0.0 <= v < h_img):
            # The prediction says the target has walked OUT of the image: whoever is at
            # the border is not it -> no prediction gate (only the last-position area).
            return None
        std = math.hypot(self._kf_u.pos_std(), self._kf_v.pos_std())
        radius = min(TARGET_DISTANCE_THRESHOLD_FRAC * diag + GATE_SIGMA_K * std, GATE_MAX_FRAC * diag)
        return u, v, radius, std

    def _predicted_box(self):
        """The last confirmed box moved onto the prediction (same size), or the last
        confirmed box itself if the prediction is not fresh."""
        if self._locked_box is None or not self._kf_fresh():
            return self._locked_box
        x1, y1, x2, y2 = self._locked_box
        hw, hh = (x2 - x1) / 2.0, (y2 - y1) / 2.0
        u, v = self._kf_u.pos, self._kf_v.pos
        return (u - hw, v - hh, u + hw, v + hh)

    def _draw_prediction(self, frame):
        if frame is None or not self._kf_active:
            return
        h_img, w_img = frame.shape[:2]
        inside = 0.0 <= self._kf_u.pos < w_img and 0.0 <= self._kf_v.pos < h_img
        u = int(min(max(self._kf_u.pos, 0.0), w_img - 1))
        v = int(min(max(self._kf_v.pos, 0.0), h_img - 1))
        if not self._kf_fresh():
            label = "predizione scaduta"
        elif not inside:
            label = "predizione fuori inquadratura"
        else:
            label = "predizione"
        cv2.circle(frame, (u, v), 6, (255, 255, 0), -1)
        cv2.putText(frame, label, (u + 8, v), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 0), 1, cv2.LINE_AA)

    # ------------------------------------------------------------------
    # Re-identification in TRACKING / RECOVERY — identity first
    # ------------------------------------------------------------------
    # Rejection stages in evaluation order; when nobody survives, the reported reason
    # is the furthest stage reached by any detection.
    _MISS_STAGES = ("no_detections", "low_confidence", "no_depth", "out_of_range",
                    "looks_like_intruder", "low_similarity", "outside_gate")

    def _attempt_reacquisition_embedding(self, response, frame_bgr, crop_rect, depth_image, debug_frame, mode,
                                         require_unambiguous=False, protected=False):
        """Neural embedding ONLY (never HSV, never track_id).

        IDENTITY FIRST, in every mode: a candidate is considered only if its
        combined ReID similarity with the target reference is >=
        REID_SIMILARITY_THRESHOLD — the same strict test that made occlusions
        safe (an occluder, or a single box that is not the target, is not taken).
        The Kalman prediction never admits anyone: it only decides where to look.

        mode "continuous" (target seen a few frames ago, see _handle_track_response_embedding):
            only near where the target is expected. Two areas, a box is inside if it is in
            EITHER of them (the prediction only widens the search, never narrows it):
              - around the LAST CONFIRMED position, radius TARGET_DISTANCE_THRESHOLD_FRAC
                of the diagonal (the area used before the prediction existed);
              - around the KALMAN PREDICTION, radius growing with its uncertainty
                (_kf_prediction), if the prediction is fresh.
            dist = distance from the NEARER of the two centres (from the prediction, minus
            PREDICTION_FREE_SIGMA std of the prediction itself); the best combined score
            W_SIMILARITY*sim - W_POSITION*dist/radius wins. While the target walks, the
            prediction is where it actually is: the position penalty stays small and a
            walking target is no longer rejected as "below_combine"/"outside_gate".
        mode "reacquire" (longer loss): the WHOLE frame is searched; the MOST
            SIMILAR person wins, only if more similar than EVERY other person in the
            frame by at least REID_AMBIGUITY_MARGIN.
        In both modes the winner must also reach REID_COMBINE. A missing depth is
        tolerated in continuous mode (it does not change who the target is).
        require_unambiguous: also in continuous mode, the winner must be more similar
            than every other person evaluated by REID_AMBIGUITY_MARGIN.
        protected (occlusion in progress): a candidate more similar to the memorised
            INTRUDER than to the target is rejected, and it does not count as a rival.
        Hold threshold (continuous mode only): if NOBODY passes REID_SIMILARITY_THRESHOLD,
            a box with REID_HOLD_THRESHOLD <= sim < REID_SIMILARITY_THRESHOLD can still keep
            the lock, under the conditions of _weak_hold_candidate. It is returned with
            weak=True: the caller follows it but never learns from it.

        Returns a dict: box, dist (None if the depth is not valid), sim, combined,
        embedding, second_sim, weak (True = kept by the hold threshold), reason (None on success).
        """
        result = {"box": None, "dist": None, "sim": None, "combined": None,
                  "embedding": None, "second_sim": None, "weak": False, "reason": "no_detections"}
        if response is None:
            result["reason"] = "no_response"
            return result

        crop_x1, crop_y1, _, _ = crop_rect
        h_img, w_img = frame_bgr.shape[:2]
        target_threshold_px = TARGET_DISTANCE_THRESHOLD_FRAC * math.hypot(w_img, h_img)
        last_center = box_center(self._locked_box)
        continuous = (mode == "continuous")

        # Prediction gate (continuous mode only, and only if the prediction is fresh).
        prediction = self._kf_prediction(frame_bgr.shape) if continuous else None
        if prediction is not None:
            pred_center, pred_radius, pred_std = (prediction[0], prediction[1]), prediction[2], prediction[3]
        else:
            pred_center, pred_radius, pred_std = None, target_threshold_px, 0.0
        # Normalisation of the position term: the wider of the two radii (the penalty is
        # never larger than before the prediction existed).
        norm_radius = max(target_threshold_px, pred_radius)

        if debug_frame is not None:
            if continuous:
                cv2.circle(debug_frame, (int(last_center[0]), int(last_center[1])), int(target_threshold_px),
                           (180, 180, 180), 1)
                if pred_center is not None:
                    cv2.circle(debug_frame, (int(pred_center[0]), int(pred_center[1])), int(pred_radius),
                               (255, 255, 0), 1)
            else:
                cv2.putText(debug_frame, "RICERCA ReID su tutta l'inquadratura", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)

        furthest_stage = 0
        evaluated = []   # (sim, box) of every person compared — rivals for the ambiguity test
        candidates = []  # (rank, sim, combined, box, dist_m, embedding)
        gate_people = [] # (sim, box) of every person inside the gate — rivals for the hold threshold
        weak = []        # (combined, sim, box, dist_m, embedding, sim_intruder): in the gate, hold <= sim < strong

        for det in response.detections:
            box_full = (det.x1 + crop_x1, det.y1 + crop_y1, det.x2 + crop_x1, det.y2 + crop_y1)

            if det.score < MIN_DETECTION_CONFIDENCE:
                furthest_stage = max(furthest_stage, 1)
                continue

            dist_m = None
            if depth_image is not None:
                box_depth = scale_box_to_depth(box_full, frame_bgr.shape, depth_image.shape)
                dist_m = box_center_depth(depth_image, box_depth)
            if dist_m is None:
                if not continuous:
                    furthest_stage = max(furthest_stage, 2)
                    continue
            elif not (TRACK_MIN_RANGE <= dist_m <= TRACK_MAX_RANGE):
                furthest_stage = max(furthest_stage, 3)
                continue

            b_embedding = embedding_from_msg(det.embedding)
            # Similarity to the TARGET: maximum over the views of the gallery (or the single
            # reference if the gallery is disabled). components = (cosine, euclidean_sim,
            # magnitude_sim) of the best view, for the log.
            similarity, components, best_view = self._target_similarity(b_embedding)

            # Similarity to the known intruder (if any): used by the occlusion test right below
            # and by the hold threshold (condition 4, see _weak_hold_candidate).
            sim_intruder = None
            if self._intruder_embedding is not None:
                sim_intruder = rich_neural_embedding_similarity(b_embedding, self._intruder_embedding, w_cosine=W_COSINE, w_euclidean=W_EUCLIDEAN, w_magnitude=W_MAGNITUDE, euclidean_scale=EUCLIDEAN_SCALE, magnitude_scale=MAGNITUDE_SCALE)

            log_row = self._reid_log_row(mode, box_full, dist_m, det.score, components, similarity,
                                         best_view, sim_intruder, b_embedding)

            # Occlusion: compare with the INTRUDER too. Whoever looks more like the person
            # who is covering the target than like the target, is not the target.
            if protected and sim_intruder is not None:
                if sim_intruder >= similarity:
                    furthest_stage = max(furthest_stage, 4)
                    self._draw_box(debug_frame, box_full, (0, 0, 255),
                                   f"intruso (sim intruso {sim_intruder:.2f} >= target {similarity:.2f})", label_offset=5)
                    continue

            evaluated.append((similarity, box_full))

            center = box_center(box_full)
            dist_last = distance(center, last_center)
            dist_pred = distance(center, pred_center) if pred_center is not None else float("inf")
            in_gate = dist_last <= target_threshold_px or dist_pred <= pred_radius
            if log_row is not None:
                log_row["in_gate"] = int(in_gate) if continuous else ""
            # Position term: distance from the nearer centre; from the prediction, only the part
            # beyond its own uncertainty counts (PREDICTION_FREE_SIGMA).
            free_px = PREDICTION_FREE_SIGMA * pred_std
            dist_px = min(dist_last, max(0.0, dist_pred - free_px))
            combined = (W_SIMILARITY * similarity) - (W_POSITION * (dist_px / norm_radius)) \
                if continuous else W_SIMILARITY * similarity
            dist_label = f"{dist_m:.2f}m" if dist_m is not None else "depth n/d"
            self._draw_box(debug_frame, box_full, (0, 200, 0),
                           f"{det.class_name} {dist_label} sim={similarity:.2f} comb={combined:.2f}", label_offset=4)

            if continuous and in_gate:
                gate_people.append((similarity, box_full))

            if similarity < REID_SIMILARITY_THRESHOLD:
                # Below the strong threshold: not a candidate, but it may still KEEP the lock
                # through the hold threshold (continuous mode, inside the gate only).
                if continuous and in_gate and REID_HOLD_THRESHOLD is not None \
                        and similarity >= REID_HOLD_THRESHOLD:
                    weak.append((combined, similarity, box_full, dist_m, b_embedding, sim_intruder))
                furthest_stage = max(furthest_stage, 5)
                continue
            if continuous and not in_gate:
                furthest_stage = max(furthest_stage, 6)
                continue

            rank = combined if continuous else similarity
            candidates.append((rank, similarity, combined, box_full, dist_m, b_embedding))

        if not candidates:
            # Nobody passes the strong threshold: try the hold threshold (continuous mode only,
            # weak is always empty in reacquire mode).
            hold, weak_reason = self._weak_hold_candidate(weak, gate_people)
            if hold is not None:
                combined, similarity, box_full, dist_m, b_embedding, second_sim = hold
                result.update({"box": box_full, "dist": dist_m, "sim": similarity, "combined": combined,
                               "embedding": b_embedding, "second_sim": second_sim, "weak": True,
                               "reason": None})
                self.get_logger().info(
                    f"[{mode}] MANTENIMENTO: sim={similarity:.2f} (< {REID_SIMILARITY_THRESHOLD:.2f}, "
                    f">= {REID_HOLD_THRESHOLD:.2f}), dist={dist_m:.2f}m"
                    f"{'' if second_sim is None else f', rivale: {second_sim:.2f}'}", throttle_duration_sec=1.0)
                return result
            result["reason"] = weak_reason or self._MISS_STAGES[furthest_stage]
            return result

        candidates.sort(key=lambda c: c[0], reverse=True)
        _, similarity, combined, box_full, dist_m, b_embedding = candidates[0]
        second_sim = max((sim for sim, box in evaluated if box is not box_full), default=None)
        result.update({"sim": similarity, "combined": combined, "second_sim": second_sim})

        # The winner must be CLEARLY the most similar person in view (always in the
        # whole-frame search, and in continuous mode right after a missed frame).
        if (not continuous or require_unambiguous) and second_sim is not None \
                and similarity - second_sim < REID_AMBIGUITY_MARGIN:
            result["reason"] = "ambiguous"
            return result

        if REID_COMBINE is not None and combined < REID_COMBINE:
            result["reason"] = "below_combine"
            return result

        result.update({"box": box_full, "dist": dist_m, "embedding": b_embedding, "reason": None})
        self.get_logger().info(
            f"[{mode}] Similarity: {similarity:.2f}, Combined: {combined:.2f}"
            f"{'' if second_sim is None else f', rivale: {second_sim:.2f}'}", throttle_duration_sec=1.0)
        return result

    def _weak_hold_candidate(self, weak, gate_people):
        """Hold threshold: may a box BELOW the strong threshold keep the lock?

        weak:        boxes in the gate with REID_HOLD_THRESHOLD <= sim < REID_SIMILARITY_THRESHOLD,
                     as (combined, sim, box, dist_m, embedding, sim_intruder).
        gate_people: (sim, box) of EVERY person inside the gate (rivals).

        Conditions, in order (the first that fails gives the miss reason):
          1. position: the weak box NEAREST to the expected position (last confirmed box or the
             same box moved onto the prediction) is chosen; its centre must be within
             HOLD_MAX_OFFSET_W widths of the last confirmed box. Whoever walks NEXT TO the
             target is about one box width away and is excluded.            -> weak_off_prediction
          2. relative identity, with the help of the position:
             - a rival NEAR the expected position (HOLD_RIVAL_RADIUS_W box widths: crossing,
               overlapping) cannot be told apart by position: the candidate must be more similar
               by HOLD_RIVAL_MARGIN, otherwise nobody is taken                -> weak_ambiguous
             - a rival FAR from it (e.g. walking side by side) is already separated by position:
               a tie goes to the candidate on the prediction, but a far rival MORE similar than
               the candidate still blocks it (position never overrides appearance)
                                                                             -> weak_rival_more_similar
          3. depth: valid and within HOLD_MAX_DEPTH_JUMP_M of the last confirmed distance.
             Whoever walks IN FRONT of the target is closer to the camera.  -> weak_no_depth /
                                                                                weak_depth_jump
          4. intruder: with a known intruder, sim_target - sim_intruder >= HOLD_INTRUDER_MARGIN.
                                                                             -> weak_intruder

        Returns ((combined, sim, box, dist_m, embedding, second_sim), None) on success,
        (None, reason) on failure, (None, None) if there was no weak box at all."""
        if not weak or self._locked_box is None:
            return None, None

        # 1. Position: nearest weak box to the expected position.
        ref_centers = [box_center(self._locked_box)]
        predicted = self._predicted_box()
        if predicted is not None and predicted is not self._locked_box:
            ref_centers.append(box_center(predicted))
        ref_w = max(self._locked_box[2] - self._locked_box[0], 1.0)

        def offset(box):
            c = box_center(box)
            return min(distance(c, r) for r in ref_centers)

        best = min(weak, key=lambda w: offset(w[2]))
        combined, sim, box, dist_m, emb, sim_intruder = best
        if offset(box) > HOLD_MAX_OFFSET_W * ref_w:
            return None, "weak_off_prediction"

        # 2. Relative identity. NEAR rivals (position cannot separate them from the candidate):
        #    the candidate must beat them by HOLD_RIVAL_MARGIN. FAR rivals (position separates
        #    them): the candidate only has to be at least as similar — position breaks the tie.
        near, far = [], []
        for s, b in gate_people:
            if b is box:
                continue
            is_near = HOLD_RIVAL_RADIUS_W is None or offset(b) <= HOLD_RIVAL_RADIUS_W * ref_w
            (near if is_near else far).append(s)
        second_sim = max(near + far) if (near or far) else None
        if near and sim - max(near) < HOLD_RIVAL_MARGIN:
            return None, "weak_ambiguous"
        if far and sim < max(far):
            return None, "weak_rival_more_similar"

        # 3. Depth consistency.
        if dist_m is None or self._locked_dist is None:
            return None, "weak_no_depth"
        if abs(dist_m - self._locked_dist) > HOLD_MAX_DEPTH_JUMP_M:
            return None, "weak_depth_jump"

        # 4. Known intruder.
        if sim_intruder is not None and sim - sim_intruder < HOLD_INTRUDER_MARGIN:
            return None, "weak_intruder"

        return (combined, sim, box, dist_m, emb, second_sim), None

    # ------------------------------------------------------------------
    # Multi-view gallery of the target (embedding_only)
    # ------------------------------------------------------------------
    def _target_similarity(self, embedding):
        """Similarity of a candidate to the TARGET.
        Gallery active: the MAXIMUM over the stored views — the candidate only has to look like
        ONE way the target has been seen (from the front, from behind...).
        Gallery disabled: the single reference (EMA), as before.
        Returns (similarity, components of the best view or None, index of the best view or -1)."""
        views = self._gallery if (REID_GALLERY_SIZE and self._gallery) else \
            ([self.reference_embedding] if self.reference_embedding is not None else [])
        best_sim, best_comp, best_idx = 0.0, None, -1
        for idx, view in enumerate(views):
            comp = rich_neural_embedding_components(embedding, view, EUCLIDEAN_SCALE, MAGNITUDE_SCALE)
            sim = combine_reid_components(comp, W_COSINE, W_EUCLIDEAN, W_MAGNITUDE)
            if best_idx == -1 or sim > best_sim:
                best_sim, best_comp, best_idx = sim, comp, idx
        return best_sim, best_comp, best_idx

    def _gallery_chain_reset(self):
        self._gallery_chain_anchored = False
        self._gallery_chain_len = 0

    def _alone_near(self, response, crop_rect, box):
        """True if NO other detection touches the target box widened by GALLERY_ALONE_MARGIN of
        its width on each side (low-confidence and out-of-range boxes included: a person half
        visible next to the target still makes the crop of the target unreliable)."""
        if response is None:
            return True
        x1, y1, x2, y2 = box
        margin = GALLERY_ALONE_MARGIN * (x2 - x1)
        zone = (x1 - margin, y1, x2 + margin, y2)
        cx, cy = crop_rect[0], crop_rect[1]
        for det in response.detections:
            other = (det.x1 + cx, det.y1 + cy, det.x2 + cx, det.y2 + cy)
            if other == box:
                continue  # the target itself
            if self._coverage(other, zone) > 0.0:
                return False
        return True

    def _gallery_try_add(self, embedding, source):
        """Adds the view if it is NEW (similarity to every stored view < GALLERY_NOVELTY_SIM).
        Gallery full: the oldest view is replaced, never the anchor (index 0)."""
        if embedding is None or not self._gallery:
            return False
        nearest, _, nearest_idx = self._target_similarity(embedding)
        if nearest >= GALLERY_NOVELTY_SIM:
            return False  # already known: nothing new to learn
        if len(self._gallery) >= REID_GALLERY_SIZE:
            self._gallery.pop(1)  # oldest non-anchor view
        self._gallery.append(embedding)
        self._metrics_gallery_added[source] += 1
        self.get_logger().info(
            f"[galleria] nuova vista ({source}): somiglianza massima alle viste salvate {nearest:.2f} "
            f"(vista {nearest_idx}) — viste ora {len(self._gallery)}/{REID_GALLERY_SIZE}")
        return True

    def _update_gallery(self, res, response, crop_rect):
        """Called after EVERY committed frame (TRACKING and RECOVERY). Decides whether the
        committed view enters the gallery (see the GALLERY_* constants)."""
        if not REID_GALLERY_SIZE or not self._gallery:
            return
        box, dist, emb = res["box"], res["dist"], res["embedding"]
        prev_box, prev_dist = self._gallery_prev_box, self._gallery_prev_dist
        self._gallery_prev_box, self._gallery_prev_dist = box, dist

        # Clean frame: valid embedding and depth, no occlusion / known intruder, nobody close.
        clean = (emb is not None and dist is not None and self._occlusion_hold == 0
                 and self._intruder_embedding is None and self._alone_near(response, crop_rect, box))
        if not clean:
            self._gallery_chain_reset()
            return

        if not res["weak"]:
            # Strong match: may be added directly, and it (re)anchors the chain.
            if res["sim"] is not None and res["sim"] >= GALLERY_STRONG_ADD_MIN_SIM:
                self._gallery_try_add(emb, "strong")
            self._gallery_chain_anchored = True
            self._gallery_chain_len = 0
            return

        # Weak match: counts only if the chain is anchored and this frame is continuous with the
        # previous one (same box, same depth): identity guaranteed by continuity, not by similarity.
        continuous = (prev_box is not None and prev_dist is not None
                      and self._iou(box, prev_box) >= GALLERY_CHAIN_MIN_IOU
                      and abs(dist - prev_dist) <= GALLERY_CHAIN_MAX_DEPTH_STEP_M)
        if not self._gallery_chain_anchored or not continuous:
            self._gallery_chain_reset()
            return
        self._gallery_chain_len += 1
        if self._gallery_chain_len >= GALLERY_CHAIN_FRAMES:
            self._gallery_try_add(emb, "weak_chain")
            self._gallery_chain_len = 0  # the next view needs another full chain

    # ------------------------------------------------------------------
    # ReID component log (embedding_only)
    # ------------------------------------------------------------------
    _REID_CSV_FIELDS = ("t", "frame", "state", "mode", "cx", "cy", "w", "h", "depth_m", "det_score",
                        "cosine", "euclidean_sim", "magnitude_sim", "sim", "norm", "view_norm",
                        "best_view", "gallery_views", "sim_intruder", "in_gate", "label")

    def _view(self, idx):
        """Stored view number idx (gallery) or the single reference."""
        if REID_GALLERY_SIZE and self._gallery:
            return self._gallery[idx] if 0 <= idx < len(self._gallery) else None
        return self.reference_embedding

    def _reid_log_row(self, mode, box, dist_m, det_score, components, similarity, best_view, sim_intruder,
                      embedding=None):
        """Prepares the CSV row of one compared person; the label is set at the end of the
        frame (_flush_reid_log). None if the log is disabled."""
        if self._reid_csv_writer is None:
            return None
        cosine, euclidean_sim, magnitude_sim = components if components is not None else ("", "", "")
        cx, cy = box_center(box)
        row = {"t": round(time.monotonic() - self._metrics_start, 3), "frame": self._metrics_total_frames,
               "state": self.state, "mode": mode, "cx": round(cx, 1), "cy": round(cy, 1),
               "w": round(box[2] - box[0], 1), "h": round(box[3] - box[1], 1),
               "depth_m": "" if dist_m is None else round(dist_m, 3), "det_score": round(det_score, 3),
               "cosine": cosine if cosine == "" else round(cosine, 4),
               "euclidean_sim": euclidean_sim if euclidean_sim == "" else round(euclidean_sim, 4),
               "magnitude_sim": magnitude_sim if magnitude_sim == "" else round(magnitude_sim, 4),
               "sim": round(similarity, 4),
               "norm": "" if embedding is None else round(float(np.linalg.norm(embedding)), 3),
               "view_norm": "" if self._view(best_view) is None else round(float(np.linalg.norm(self._view(best_view))), 3),
               "best_view": best_view,
               "gallery_views": len(self._gallery) if REID_GALLERY_SIZE else 1,
               "sim_intruder": "" if sim_intruder is None else round(sim_intruder, 4),
               "in_gate": "", "label": "", "_box": box}
        self._reid_rows.append(row)
        return row

    def _flush_reid_log(self, committed_box, weak):
        """Labels and writes the rows of the frame: the committed person is the target, the others
        are "other"; with no confirmed target, every row is "unconfirmed"."""
        rows, self._reid_rows = self._reid_rows, []
        if self._reid_csv_writer is None or not rows:
            return
        for row in rows:
            box = row.pop("_box")
            if committed_box is None:
                row["label"] = "unconfirmed"
            elif box == committed_box:
                row["label"] = "target_weak" if weak else "target_strong"
            else:
                row["label"] = "other"
            self._reid_csv_writer.writerow(row)
        try:
            self._reid_csv_file.flush()
        except OSError:
            pass

    def close_reid_log(self):
        if self._reid_csv_file is not None:
            try:
                self._reid_csv_file.close()
                print(f"[ReID log] salvato: {self._reid_csv_path}")
            except OSError:
                pass
            self._reid_csv_file = self._reid_csv_writer = None

    def _commit_embedding_target(self, res, debug_frame, header):
        """Common part of a confirmed (re)lock in TRACKING/RECOVERY."""
        best_box = res["box"]
        weak = res.get("weak", False)
        dist_label = f"{res['dist']:.2f}m" if res["dist"] is not None else "depth n/d"
        if weak:
            # Kept by the hold threshold: different colour in the debug image.
            self._draw_box(debug_frame, best_box, (0, 255, 255),
                           f"TARGET {dist_label} (mantenimento sim={res['sim']:.2f})", label_offset=0)
            self._metrics_weak_hold_frames += 1
        else:
            self._draw_box(debug_frame, best_box, (0, 255, 0), f"TARGET {dist_label}", label_offset=0)
        self._locked_box = best_box
        if res["dist"] is not None:
            self._locked_dist = res["dist"]
        self._tracking_miss_count = 0
        self._kf_update(best_box)
        self._publish_target_info(best_box, res["dist"], header)  # skipped by itself if the depth is missing
        # Reference frozen during an occlusion: a crop mixing target and occluder must
        # never pull the reference towards the other person.
        # Only CONFIDENT matches update it (REID_EMA_MIN_SIM): a borderline match is followed but
        # never allowed to reshape the reference. A weak match (hold threshold) NEVER updates it.
        # With the gallery active there is no EMA: new views are added by _update_gallery instead.
        confident = REID_EMA_MIN_SIM is None or (res["sim"] is not None and res["sim"] >= REID_EMA_MIN_SIM)
        if not REID_GALLERY_SIZE and not weak and self._occlusion_hold == 0 and confident \
                and res["embedding"] is not None \
                and self.reference_embedding is not None:
            self.reference_embedding = (
                (1 - REID_EMA_ALPHA) * self.reference_embedding + REID_EMA_ALPHA * res["embedding"])

    @staticmethod
    def _coverage(other, target):
        """Fraction of the target box covered by another box."""
        ix1, iy1 = max(other[0], target[0]), max(other[1], target[1])
        ix2, iy2 = min(other[2], target[2]), min(other[3], target[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area = (target[2] - target[0]) * (target[3] - target[1])
        return inter / area if area > 0 else 0.0

    def _update_occlusion_state(self, response, crop_rect, committed_box, strong=True):
        """After each TRACKING frame: is someone overlapping the target?
        Target area = the box just confirmed; if the target was NOT confirmed, both the
        last confirmed box AND the same box moved onto the prediction (the target may
        have walked on while hidden: whoever covers where it is now is an occluder too).
        The other person's embedding is learnt ONLY when the target was confirmed
        in this frame: then the overlapping box is certainly someone else. In that case a
        person merely CLOSE to the target (OCCLUSION_NEAR_MARGIN) counts too: it is learnt
        while it approaches, before it can hide the target.
        strong=False (target kept by the hold threshold): the committed box is used as the
        target area, but NOTHING is learnt (if the weak match were wrong, the real target
        could be learnt as "intruder") and the known intruder is NOT forgotten: only a
        strong confirmation counts as "target confirmed again"."""
        if committed_box is not None:
            targets = [committed_box]
        else:
            targets = [b for b in (self._locked_box, self._predicted_box()) if b is not None]
        if not targets:
            return
        near_zone = None
        if committed_box is not None and strong and OCCLUSION_NEAR_MARGIN is not None:
            x1, y1, x2, y2 = committed_box
            margin = OCCLUSION_NEAR_MARGIN * (x2 - x1)
            near_zone = (x1 - margin, y1, x2 + margin, y2)
        intruder, intruder_dist = None, None
        if response is not None:
            cx, cy = crop_rect[0], crop_rect[1]
            for det in response.detections:
                box = (det.x1 + cx, det.y1 + cy, det.x2 + cx, det.y2 + cy)
                if committed_box is not None and box == committed_box:
                    continue  # the target itself
                overlapping = any(self._coverage(box, t) > OCCLUSION_COVERAGE for t in targets)
                near = near_zone is not None and self._coverage(box, near_zone) > 0.0
                if overlapping or near:
                    # the closest one to the target is the one that matters
                    d = distance(box_center(box), box_center(targets[0]))
                    if intruder is None or d < intruder_dist:
                        intruder, intruder_dist = det, d
        if intruder is not None:
            if self._occlusion_hold == 0:
                self.get_logger().info("Occlusione: un'altra persona e' vicina / si sovrappone al target — protezione attiva.")
            self._occlusion_hold = OCCLUSION_HOLD_FRAMES
            if committed_box is not None and strong:
                emb = embedding_from_msg(intruder.embedding)
                if emb is not None:
                    self._intruder_embedding = emb if self._intruder_embedding is None else \
                        0.5 * self._intruder_embedding + 0.5 * emb
        elif self._occlusion_hold > 0:
            if INTRUDER_KEPT_WHILE_LOST and (committed_box is None or not strong) \
                    and self._intruder_embedding is not None:
                return  # target lost / only weakly kept: keep the known intruder (and the protection) as they are
            self._occlusion_hold -= 1
            if self._occlusion_hold == 0:
                self._intruder_embedding = None
                self.get_logger().info("Occlusione terminata.")

    def _count_miss(self, reason):
        self._metrics_miss_reasons[f"{self.state}:{reason}"] += 1

    def _draw_miss(self, debug_frame, reason):
        if debug_frame is None:
            return
        cv2.putText(debug_frame, f"target non confermato: {reason}", (10, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2, cv2.LINE_AA)

    def _handle_track_response_embedding(self, response, frame_bgr, crop_rect, debug_frame, header):
        self._kf_predict_now()
        self._draw_prediction(debug_frame)
        depth_image = self._latest_depth_image

        # Short gap: re-locked at once, near the last confirmed position OR near the
        # prediction, with the SAME strict identity test (+ ambiguity test after a miss).
        # The window is longer (PREDICTION_GAP_FRAMES) while the prediction is fresh and
        # no intruder is known; with a known intruder it stays SHORT_GAP_FRAMES.
        # After that: whole-frame identity search + multi-frame confirmation.
        protected = self._occlusion_hold > 0
        # Longer window only if the prediction is usable (fresh AND inside the image) and no
        # intruder is known.
        if self._kf_prediction(frame_bgr.shape) is not None and self._intruder_embedding is None:
            gap_window = PREDICTION_GAP_FRAMES
        else:
            gap_window = SHORT_GAP_FRAMES
        short_gap = self._tracking_miss_count <= gap_window
        mode = "continuous" if short_gap else "reacquire"
        if protected and debug_frame is not None:
            cv2.putText(debug_frame, f"OCCLUSIONE: confronto anche con l'intruso ({self._occlusion_hold})",
                        (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        if response is not None:
            for det in response.detections:
                box_full = (det.x1 + crop_rect[0], det.y1 + crop_rect[1], det.x2 + crop_rect[0], det.y2 + crop_rect[1])
                self._draw_box(debug_frame, box_full, (100, 100, 100), f"{det.class_name} - {det.score:.2f}", label_offset=3)

        res = self._attempt_reacquisition_embedding(response, frame_bgr, crop_rect, depth_image, debug_frame, mode,
                                                    require_unambiguous=self._tracking_miss_count > 0,
                                                    protected=protected)
        best_box = res["box"]

        committed = False
        miss_reason = res["reason"]
        if best_box is not None:
            if short_gap or self._immediate_relock(res):
                committed = True
                self._reset_reacquisition_confirmation()
            elif self._confirm_reacquisition(best_box):
                committed = True
            else:
                miss_reason = "pending_confirmation"
                self._draw_box(debug_frame, best_box, (255, 165, 0),
                                f"possibile target sim={res['sim']:.2f} "
                                f"(conferma {self._reacquisition_count}/{REACQUISITION_STABILITY_FRAMES})", label_offset=2)
                self.get_logger().info(
                    f"TRACKING [embedding_only]: candidato trovato, in attesa di conferma "
                    f"({self._reacquisition_count}/{REACQUISITION_STABILITY_FRAMES})...", throttle_duration_sec=1.0)
        else:
            self._tolerate_empty_confirmation_frame()

        if committed:
            self._commit_embedding_target(res, debug_frame, header)
            self._update_occlusion_state(response, crop_rect, best_box, strong=not res["weak"])
            self._update_gallery(res, response, crop_rect)   # after the occlusion state: uses it
            self._flush_reid_log(best_box, res["weak"])
            self.get_logger().info(
                f"TRACKING [embedding_only] ok ({mode}{', mantenimento' if res['weak'] else ''}"
                f"{', occlusione' if protected else ''}): "
                f"sim={res['sim']:.2f}, combinato={res['combined']:.2f}", throttle_duration_sec=1.0)
            self._publish_debug(debug_frame, header)
            return

        self._update_occlusion_state(response, crop_rect, None)
        self._gallery_chain_reset()          # a missed frame breaks the continuity of the chain
        self._flush_reid_log(None, False)
        self._count_miss(miss_reason)
        self._draw_miss(debug_frame, miss_reason)
        self._tracking_miss_count += 1
        if self._tracking_miss_count < TRACKING_GRACE_FRAMES:
            self.get_logger().info(
                f"TRACKING [embedding_only]: target non confermato ({miss_reason}) — tentativo "
                f"{self._tracking_miss_count}/{TRACKING_GRACE_FRAMES} prima di passare a RECOVERY.",
                throttle_duration_sec=1.0)
            self._publish_debug(debug_frame, header)
            return

        self.state = RECOVERY
        self._recovery_deadline = time.monotonic() + RECOVERY_TIMEOUT_SEC
        self._tracking_miss_count = 0
        self._reset_reacquisition_confirmation()
        self.get_logger().info(
            f"TRACKING [embedding_only]: target non ritrovato entro {TRACKING_GRACE_FRAMES} frame — "
            f"passo a RECOVERY (timeout {RECOVERY_TIMEOUT_SEC:.0f}s).")
        self._publish_debug(debug_frame, header)

    def _handle_recovery_response_embedding(self, response, frame_bgr, crop_rect, debug_frame, header):
        self._kf_predict_now()
        self._draw_prediction(debug_frame)

        if response is not None:
            for det in response.detections:
                box_full = (det.x1 + crop_rect[0], det.y1 + crop_rect[1], det.x2 + crop_rect[0], det.y2 + crop_rect[1])
                self._draw_box(debug_frame, box_full, (100, 100, 100), f"{det.class_name} det pre MIN_DETECTION_CONFIDENCE check EMBEDDING_ONLY RECOVERY", label_offset=3)

        depth_image = self._latest_depth_image
        res = self._attempt_reacquisition_embedding(response, frame_bgr, crop_rect, depth_image, debug_frame,
                                                    "reacquire", protected=self._occlusion_hold > 0)
        best_box = res["box"]

        committed = False
        miss_reason = res["reason"]
        if best_box is not None:
            if self._immediate_relock(res):
                committed = True
                self._reset_reacquisition_confirmation()
            elif self._confirm_reacquisition(best_box):
                committed = True
            else:
                miss_reason = "pending_confirmation"
                self._draw_box(debug_frame, best_box, (255, 165, 0),
                                f"possibile target sim={res['sim']:.2f} "
                                f"(conferma {self._reacquisition_count}/{REACQUISITION_STABILITY_FRAMES})", label_offset=2)
        else:
            self._tolerate_empty_confirmation_frame()

        if committed:
            self._commit_embedding_target(res, debug_frame, header)
            self._update_gallery(res, response, crop_rect)
            self._flush_reid_log(best_box, res["weak"])
            self.state = TRACKING
            self._recovery_deadline = None
            dist_label = f"{res['dist']:.2f}m" if res["dist"] is not None else "depth n/d"
            self.get_logger().info(
                f"RECOVERY [embedding_only]: target CONFERMATO a {dist_label}, "
                f"sim={res['sim']:.2f}, combinato={res['combined']:.2f} — torno a TRACKING.")
            self._publish_debug(debug_frame, header)
            return

        self._gallery_chain_reset()
        self._flush_reid_log(None, False)
        self._count_miss(miss_reason)
        self._draw_miss(debug_frame, miss_reason)
        remaining = self._recovery_deadline - time.monotonic()
        if remaining <= 0:
            self.get_logger().info(
                f"RECOVERY [embedding_only]: timeout di {RECOVERY_TIMEOUT_SEC:.0f}s scaduto — "
                f"torno in WAITING_TRIGGER.")
            self._enter_waiting_trigger()
        else:
            self.get_logger().info(
                f"RECOVERY [embedding_only]: nessuna corrispondenza confermata ({miss_reason}) — "
                f"{remaining:.0f}s rimanenti.", throttle_duration_sec=1.0)
        self._publish_debug(debug_frame, header)

    # ------------------------------------------------------------------
    # Shared by both methods
    # ------------------------------------------------------------------
    @staticmethod
    def _immediate_relock(res):
        """embedding_only: re-lock after a long loss without the multi-frame confirmation?
        res comes from _attempt_reacquisition_embedding, so the candidate has ALREADY passed
        every identity check (threshold, ambiguity, intruder, REID_COMBINE); here only its
        similarity is compared with REACQ_IMMEDIATE_MIN_SIM."""
        return REACQ_IMMEDIATE_MIN_SIM is not None and res["sim"] is not None \
            and res["sim"] >= REACQ_IMMEDIATE_MIN_SIM

    def _confirm_reacquisition(self, candidate_box):
        """The SAME candidate (by position) must be the best match for
        REACQUISITION_STABILITY_FRAMES attempts in a row before being
        confirmed as the recovered target.

        "Same position" = within REACQUISITION_PX_TOLERANCE of where the pending
        candidate WAS, or (embedding_only, prediction fresh) of where it should be NOW
        if it kept moving with the target's estimated velocity — a person walking
        across the image moves more than 60 px between two slow frames. Only a
        tolerance: identity was already checked by the caller on every frame."""
        self._reacquisition_gap = 0
        now = time.monotonic()
        candidate_center = box_center(candidate_box)
        same = False
        if self._reacquisition_pending_box is not None:
            pending_center = box_center(self._reacquisition_pending_box)
            same = distance(candidate_center, pending_center) <= REACQUISITION_PX_TOLERANCE
            if not same and self._kf_fresh() and self._reacquisition_pending_time is not None:
                dt = now - self._reacquisition_pending_time
                expected = (pending_center[0] + self._kf_u.vel * dt, pending_center[1] + self._kf_v.vel * dt)
                same = distance(candidate_center, expected) <= REACQUISITION_PX_TOLERANCE
        if same:
            self._reacquisition_count += 1
        else:
            self._reacquisition_count = 1
        self._reacquisition_pending_box = candidate_box
        self._reacquisition_pending_time = now

        if self._reacquisition_count >= REACQUISITION_STABILITY_FRAMES:
            self._reacquisition_pending_box = None
            self._reacquisition_count = 0
            return True
        return False

    def _reset_reacquisition_confirmation(self):
        self._reacquisition_pending_box = None
        self._reacquisition_pending_time = None
        self._reacquisition_count = 0
        self._reacquisition_gap = 0

    def _tolerate_empty_confirmation_frame(self):
        """embedding_only: a frame WITHOUT any candidate (ToF hole, missed detection)
        does not throw a pending confirmation away at once; it starts over after more
        than REACQUISITION_MAX_GAP empty frames in a row. A different candidate
        (other position) still restarts it immediately."""
        if self._reacquisition_pending_box is None:
            return
        self._reacquisition_gap += 1
        if self._reacquisition_gap > REACQUISITION_MAX_GAP:
            self._reset_reacquisition_confirmation()

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
        node.save_metrics(final=True)
        node.close_reid_log()
        node.shutdown_leds()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()