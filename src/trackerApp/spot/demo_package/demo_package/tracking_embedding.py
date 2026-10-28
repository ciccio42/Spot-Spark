#!/usr/bin/env python3
"""
tracking_embedding.py (demo_package) — everything of the "embedding_only" method that concerns
the ReID embeddings: parameters, similarity, multi-view gallery, Kalman filter, tracker.
No ROS and no OpenCV here: this file is tested on its own.

HOW IT WORKS (TargetTracker.step, every call to the detector)
  The TARGET is kept in memory for the whole session (its last clean embedding, its gallery and
  ONE Kalman filter), also while it is not seen. The OTHER people are only those of the PREVIOUS
  call (box + embedding, no Kalman): the list is rewritten every call.
    1. Kalman prediction of the target (where it should be now).
    2. Similarities of EVERY detection: sim_target = similarity to the gallery (long-term memory),
       lifted by the similarity to the last clean target embedding (short-term) by FF_MAX_LEAD at
       most; similarity to each person of the previous call.
    3. Candidates — every detection, wherever it is (the Kalman never excludes anybody):
         near the prediction (target seen in the last HOLD_MAX_MISSES calls, inside the gate):
             sim_target >= TARGET_LOW            (the position is the second proof)
         anywhere else, or after a loss:  sim_target >= REACQ_THRESHOLD   (appearance only)
         always: gallery veto; never more similar to a person of the previous call; never the
                 continuation (same place) of a person of the previous call; same class group
                 as at the lock; far / after a loss also a plausible distance
    4. APPEARANCE decides: the candidate most similar to the target, if clearly more similar than
       the second (AMBIGUITY_MARGIN); on a tie the one near the prediction wins (the Kalman only
       breaks ties). Near: CONFIRMED (>= TARGET_HIGH) / KEPT. Far (a jump
       or a re-acquisition): at once from REACQ_IMMEDIATE_SIM, otherwise REACQ_CONFIRM_FRAMES calls
       in the same place; the Kalman is restarted where the target really is.
    5. The target embedding and the gallery are updated only in CLEAN frames (nobody touching the
       target box). The new list of people = every detection CLEARLY not the target.
"""

import math
import time
from collections import deque
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Optional

import numpy as np

from demo_package.common import (
    box_center, distance,
    rich_neural_embedding_components, combine_reid_components, neural_embedding_similarity,
)

# ======================================================================
# PARAMETERS — the similarities are on the scale of the weighted ReID similarity: when the weights
# change, recalibrate them from "sim_stats" in the metrics JSON (target / other people distributions).
# ======================================================================
W_COSINE = 0.9                   # ReID similarity = weighted mean of cosine, Euclidean and magnitude terms
W_EUCLIDEAN = 0.1
W_MAGNITUDE = 0.0
EUCLIDEAN_SCALE = 10.0
MAGNITUDE_SCALE = 10.0

MIN_DETECTION_CONFIDENCE = 0.10  # YOLOE confidence
TRACK_MIN_RANGE = 0.3            # usable depth (m); outside it the depth is treated as missing
TRACK_MAX_RANGE = 6.0
PERSON_MAX_HEIGHT_M = 2.1        # depth implying a taller person (box height px x depth / fy) is not the
                                 # person's (background through a gap) -> treated as missing
DEDUPE_IOU = 0.6                 # same subject under two prompts

STABILITY_FRAMES_REQUIRED = 10   # SEARCH: frames of one stable person before the lock
SEARCH_STABILITY_MIN_COS = 0.60  # SEARCH: plain cosine between consecutive frames (independent of the weights)

# PROVISIONAL values from the 13/10 video (your weights): you, frame to frame 0.71-0.79; you, same
# pose seconds later vs the gallery 0.52-0.60; side 0.47-0.69; back / behind the jacket 0.39-0.50;
# Pepper (not a person) 0.31-0.36; ANOTHER PERSON: not measured yet -> recalibrate with
# "sim_stats" of the metrics JSON of a session with a second person.
TARGET_HIGH = 0.76               # target CONFIRMED
TARGET_LOW = 0.64                # target KEPT (uncertain) between LOW and HIGH
GALLERY_VETO_MIN = 0.30          # never the target below this similarity to the gallery (None = off)
FF_MAX_LEAD = 0.20               # sim_target = max(gallery, min(last embedding, gallery + this)): the short-term
                                 # memory can lift a detection only this much above the long-term one. If an
                                 # impostor was followed for a while, the last embedding IS the impostor: without
                                 # this cap it would always beat the real target (seen on the robot, 13/10)
HOLD_MAX_MISSES = 5              # calls without the target before the re-acquisition rules apply
SHORT_GAP_FROZEN_GATE = True     # short gap: the gate keeps the size it had at the last sighting (the
                                 # centre still follows the prediction). Off: it widens at every miss
                                 # (~160 px / 1.2 m after 1 miss, ~360 px / 2 m after 3) and a person
                                 # never seen before, standing where the target vanished, gets in.
OTHER_CONTINUITY_MAX = 1.0       # a detection CONTINUES a person of the previous call if it is closer to that person's
                                 # box than to the target's (cost = |du| / box width + 2 |ln(height ratio)|, at most
                                 # this) and not clearly more similar to the target: while it walks over the target
                                 # its crop is mixed (part of the target inside) and the appearance alone is fooled
GALLERY_TIEBREAK = 0.06         # tie on sim_target: the one clearly more similar to the GALLERY wins, then the position
AMBIGUITY_MARGIN = 0.1          # appearance decides; two candidates closer than this in sim_target are a tie:
                                 # the one near the Kalman prediction wins, otherwise nobody is taken
CLEAN_MARGIN = 0.5               # clean frame: no other box touches the target box widened by this (x width)

# (A) NO POSITION BONUS WHEN SOMEBODY ELSE IS THERE. Near the Kalman prediction the threshold is
# TARGET_LOW instead of REACQ_THRESHOLD and the position breaks ties: the position is a second proof
# only if nobody else could be at the predicted point. If in the previous call another person touched
# the target's box at the prediction (widened by CLEAN_MARGIN: beside it, covering it), whoever is
# found there now may be that person (photos IMG_0347/0348, 14/10: the occluder taken as KEPT at
# 0.46) -> no bonus (REACQ_THRESHOLD, a tie is "ambiguous") UNLESS the DEPTH tells them apart: the
# detection is at the target's predicted depth (within OCCLUSION_DEPTH_JUMP_M) and closer to it than
# to the depth of every person who was there (video 14/10 11:07: the target at 3.4 m, sim 0.49, lost
# because a person 1.9 m away walked in front of it). Nobody around: the bonus stays.
NEAR_BONUS_ONLY_WHEN_ALONE = True
# (B) STICKY INTRUDERS. A person of the previous call who continues there (box closer to theirs than
# to the target's by INTRUDER_STICKY_MARGIN, see _box_cost) STAYS an intruder: never a candidate, and
# kept in the people list even when its similarity gets close to the target's (it happens while it
# covers the target: mixed crop, and the target's own similarity drops). Released only if the GALLERY
# recognises it clearly (sim_gal >= INTRUDER_RELEASE_SIM): the way out if the target itself was
# labelled an intruder by mistake.
INTRUDER_STICKY = True
INTRUDER_STICKY_MARGIN = 0.2
INTRUDER_RELEASE_SIM = 0.62

# DEPTH NOT BELONGING TO THE TARGET. The depth is read in the centre of the box: on a partial
# occlusion it falls on the OCCLUDER (closer), through a gap between arms and body on the BACKGROUND
# (farther) -> 3D gate fails although the target is right there. Rule: depth far from the prediction
# while the box SIZE agrees with the predicted depth (a person really closer would have a bigger box,
# really farther a smaller one) -> the depth is discarded: gate in 2D, Kalman updated without depth,
# nothing learnt, not published to Nav2 ("TARGET OCCLUSO" / "DEPTH SCARTATA" in the debug image).
OCCLUSION_DEPTH_FALLBACK = True
OCCLUSION_DEPTH_JUMP_M = 0.5     # |measured - predicted depth| above this
OCCLUSION_MAX_SIZE_RATIO = 1.15  # box height vs expected height at the predicted depth (tolerance)
# ... but if the Kalman depth itself is the wrong one (it happened in the 13/10 video: 4.5 m read on
# the background was accepted, then the true 2.0 m was always discarded as "occluded"), discarding
# forever never heals it. RE-SYNC: DEPTH_RESYNC_FRAMES consecutive discarded depths that agree with
# each other (spread <= DEPTH_RESYNC_MAX_SPREAD_M) with NO detection overlapping the target (nothing
# visible in front) -> the Kalman depth is reset to the measured one.
DEPTH_RESYNC_FRAMES = 3
DEPTH_RESYNC_MAX_SPREAD_M = 0.3

REACQ_THRESHOLD = 0.71           # re-acquisition (was 0.60: you, frontal in RECOVERY, stayed at 0.52-0.58)
REACQ_IMMEDIATE_SIM = 0.75       # between REACQ_THRESHOLD and this: REACQ_CONFIRM_FRAMES calls in the same place
REACQ_CONFIRM_FRAMES = 2
REACQ_PX_TOLERANCE = 60.0
REACQ_MAX_GAP = 2
REACQ_REQUIRE_DEPTH = True
# Far from the prediction / after a loss the appearance alone is not enough (seen on the robot,
# 13/10: a robot much farther away taken as the target). Also required:
REACQ_MAX_RANGE_M = 4.0          # not farther than this (None = off): wait until it comes closer
REACQ_DEPTH_TOL_M = 1.0          # |depth - last target depth| <= this + REACQ_DEPTH_SPEED_MPS x seconds
REACQ_DEPTH_SPEED_MPS = 0.8      #   since it was last seen (a walking person moves ~1 m/s)
# CLASS: the target keeps the class group it had at the lock. A detection can be the target only if
# YOLOE scores it at least as much for a class of that group as for any other class (a box merged
# from 'person' 0.4 + 'humanoid robot' 0.6 is a robot, not a person).
CLASS_GROUPS = (("person",),
                ("quadruped", "quadruped animal", "quadruped robot", "robotic dog", "four-legged robot", "dog"),
                ("humanoid robot",))
PERSON_GROUP, QUADRUPED_GROUP, HUMANOID_GROUP = 0, 1, 2
# GENERIC labels: YOLOE calls both a quadruped (Spot) and a humanoid simply "robot" in some frames. Such a
# label is COMPATIBLE with both robot groups (never with a person), so that a target is not lost when
# its label switches between "quadruped robot" and "robot" (or "humanoid robot" and "robot").
GENERIC_ROBOT_CLASSES = ("robot",)
CLASS_VOTE = True                # the target group is decided by MAJORITY over the SEARCH stability
                                 # frames (False: from the last frame only, as before)

# TARGET PROFILES. OSNet is trained on people: on a QUADRUPED it gives less stable embeddings (lower
# similarities, frame to frame and with the gallery). The quadruped profile has its own, lower
# thresholds and more tolerance; it is used ONLY when the target is a quadruped. Persons and
# humanoids use the standard values above, unchanged.
QUADRUPED_PROFILE_ENABLED = True
QUADRUPED = dict(                # PROVISIONAL values: to be calibrated on a session with the quadruped
    SEARCH_STABILITY_MIN_COS=0.45,   # (sim_stats of the metrics JSON, cosine between frames in SEARCH)
    SEARCH_MAX_MISSED_FRAMES=1,      # SEARCH: frames with no detection that do not restart the count
    TARGET_HIGH=0.50,
    TARGET_LOW=0.38,
    REACQ_THRESHOLD=0.44,
    REACQ_IMMEDIATE_SIM=0.54,
    GALLERY_VETO_MIN=0.22,
    GALLERY_STRONG_ADD_MIN_SIM=0.60,
    INTRUDER_RELEASE_SIM=0.54,
    TRACKING_GRACE_FRAMES=45,        # ~15 s at 3 Hz: YOLOE misses a quadruped more often
)

TRACKING_GRACE_FRAMES = 30       # TRACKING -> RECOVERY after this many calls without the target
RECOVERY_TIMEOUT_SEC = 15.0      # RECOVERY -> WAITING_TRIGGER

# PREDICTIVE FILTER of the target. "odom": position and velocity in the odom frame (3D), the
# measurement (u, v, d) is the box centre and depth seen through the camera pose at the time of the
# image (TF): a person standing still stays still while Spot rotates or walks, so the prediction in
# the image is compensated for the robot motion. "image": the previous filter, in pixels (KalmanUVD).
PREDICTIVE_FILTER = "odom"
KF3_ACCEL_XY = 0.5               # odom filter: white-noise acceleration, horizontal (m^2/s^3) and height
KF3_ACCEL_Z = 0.1
KF3_MEAS_STD_V_PX = 20.0         # vertical centre of the box: noisier than u (legs/head cut by the crop)
KF3_INIT_VEL_STD_XY = 1.5        # m/s, velocity unknown when the filter (re)starts
KF3_INIT_VEL_STD_Z = 0.2
KF3_DEFAULT_DEPTH = 3.0          # started without any depth: placed here on the ray, very uncertain
KF3_UNKNOWN_DEPTH_STD = 2.0
KF3_PREDICT_MAX_SEC = 3.0        # not measured: the prediction keeps moving at most this long, ...
KF3_PREDICT_MAX_STD_M = 1.5      # ... while its horizontal uncertainty stays below this, ...
KF3_PREDICT_MAX_DIST_M = 3.0     # ... and at most this far from the last measured position; then FROZEN
# FUTURE TRAJECTORY (the ONE prediction shared with nav2_bridge and drawn in the debug image): from
# the last measurement, every KF_FUTURE_STEP_SEC, until the KF3_PREDICT_MAX_* limits. Straight if the
# person was walking straight; along their ARC if they were turning (turn rate from the walking
# direction at the last measurements), tighter and tighter if the turn rate was growing.
TURN_MIN_SPEED = 0.3             # m/s: below this the walking direction is not reliable
TURN_WINDOW_SEC = 0.8            # turn rate = change of walking direction over this window
TURN_MIN_DEG_S = 10.0            # below this turn rate the person is walking straight
CURVE_TREND_WINDOW_SEC = 1.5     # curve that TIGHTENS: trend of the turn rate over this window ...
CURVE_MAX_TIGHTEN_DEG_S2 = 90.0  # ... at most this fast (deg/s per s) ...
CURVE_MAX_RATE_DEG_S = 120.0     # ... up to this turn rate (a person turning a corner on the spot)
CURVE_ARC_MAX_M = 1.5            # along the arc at most this far from the last seen position ...
CURVE_MAX_TURN_DEG = 90.0        # ... and at most this much turn: then the trajectory ends there
DEFAULT_INTRINSICS = (550.0, 550.0, 320.0, 240.0)   # fx, fy, cx, cy: only if the node gives no camera
                                                     # (offline tests); the node uses the CameraInfo
KF_ACCEL_PX = 40000.0            # "image" filter: white-noise acceleration, image (px^2/s^3) and depth (m^2/s^3)
KF_ACCEL_DEPTH = 1.0
KF_MEAS_STD_PX = 10.0            # measurement noise: box centre (px), ToF depth (m)
KF_MEAS_STD_DEPTH = 0.15
KF_INIT_VEL_STD_PX = 300.0
KF_INIT_VEL_STD_DEPTH = 1.5
GATE_CHI2_3D = 11.34             # gate: chi-square 99%, (u, v, d) / (u, v) without depth
GATE_CHI2_2D = 9.21
PREDICTION_MAX_SEC = 2.0         # the target prediction is drawn only this long after the last measurement

REID_GALLERY_SIZE = 8            # GALLERY: view 0 = SEARCH lock, never removed
GALLERY_NOVELTY_SIM = 0.80       # a view is added only if less similar than this to every stored view
GALLERY_STRONG_ADD_MIN_SIM = 0.70  # clean frame this similar to the gallery: add (if new) and anchor the chain
GALLERY_CHAIN_FRAMES = 4         # ... or N clean continuous frames after an anchor
GALLERY_CHAIN_MIN_IOU = 0.5
GALLERY_CHAIN_MAX_DEPTH_STEP_M = 0.3

KF_TRAIL_LEN = 15                # target filter positions drawn as a trail in the debug image
KF_FUTURE_STEP_SEC = 1.0 / 3.0   # debug image: predicted points every this many seconds ...
KF_FUTURE_STEPS = 9              # ... this many of them (9 x 0.33 s = 3 s ahead)

CONFIRMED, KEPT, PENDING, MISSED = "confirmed", "kept", "pending", "missed"
_REASONS = ("no_detections", "no_embedding", "no_depth", "low_similarity", "far_low_similarity",
            "gallery_veto", "implausible_distance", "wrong_class", "looks_like_other", "continues_other",
            "intruder", "ambiguous")


# ======================================================================
# Similarity and geometry helpers
# ======================================================================
def components(e1, e2):
    """(cosine, euclidean_sim, magnitude_sim) or None if an embedding is missing."""
    return rich_neural_embedding_components(e1, e2, EUCLIDEAN_SCALE, MAGNITUDE_SCALE)


def similarity(e1, e2):
    """Weighted ReID similarity, 0.0 if an embedding is missing."""
    return combine_reid_components(components(e1, e2), W_COSINE, W_EUCLIDEAN, W_MAGNITUDE)


def cosine(e1, e2):
    return neural_embedding_similarity(e1, e2)


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _touches(a, b):
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def class_group(name):
    """Index of the CLASS_GROUPS group of a YOLOE class (a generic or unknown class: its own name)."""
    for i, group in enumerate(CLASS_GROUPS):
        if name in group:
            return i
    return name


def class_compatible(name, group):
    """A YOLOE class is compatible with the target group: same group, or a GENERIC robot label for a
    robot target (quadruped or humanoid)."""
    if class_group(name) == group:
        return True
    return name in GENERIC_ROBOT_CLASSES and group in (QUADRUPED_GROUP, HUMANOID_GROUP)


def vote_class_group(score_dicts):
    """Target group by MAJORITY over the SEARCH stability frames ([{class: score}] of the candidate).
    Each frame votes with its BEST class (the label YOLOE gave it): person, quadruped, humanoid, or
    generic "robot". 1) person against robot (a generic vote is a robot vote); 2) among robots,
    quadruped against humanoid (generic votes do not count here; tie: higher total score).
    Only generic votes: humanoid group, as before (when "robot" was in that group)."""
    votes, total = {}, {}
    for scores in score_dicts:
        best = max(scores, key=scores.get)
        g = "generic" if best in GENERIC_ROBOT_CLASSES else class_group(best)
        votes[g] = votes.get(g, 0) + 1
        total[g] = total.get(g, 0.0) + scores[best]
    if not votes:
        return None
    robot = sum(n for g, n in votes.items() if g in (QUADRUPED_GROUP, HUMANOID_GROUP, "generic"))
    others = {g: n for g, n in votes.items() if g not in (QUADRUPED_GROUP, HUMANOID_GROUP, "generic")}
    if others:                                  # person (or an unknown class) against the robots
        g_best = max(others, key=lambda g: (others[g], total[g]))
        if others[g_best] > robot:
            return g_best
    q, h = votes.get(QUADRUPED_GROUP, 0), votes.get(HUMANOID_GROUP, 0)
    if (q, total.get(QUADRUPED_GROUP, 0.0)) > (h, total.get(HUMANOID_GROUP, 0.0)):
        return QUADRUPED_GROUP
    return HUMANOID_GROUP


def profile_for_group(group):
    """Thresholds for this target group: the quadruped profile, or the standard values (read now,
    so that a change of the module constants is seen)."""
    std = dict(SEARCH_STABILITY_MIN_COS=SEARCH_STABILITY_MIN_COS, SEARCH_MAX_MISSED_FRAMES=0,
               TARGET_HIGH=TARGET_HIGH, TARGET_LOW=TARGET_LOW, REACQ_THRESHOLD=REACQ_THRESHOLD,
               REACQ_IMMEDIATE_SIM=REACQ_IMMEDIATE_SIM, GALLERY_VETO_MIN=GALLERY_VETO_MIN,
               GALLERY_STRONG_ADD_MIN_SIM=GALLERY_STRONG_ADD_MIN_SIM, INTRUDER_RELEASE_SIM=INTRUDER_RELEASE_SIM,
               TRACKING_GRACE_FRAMES=TRACKING_GRACE_FRAMES)
    if QUADRUPED_PROFILE_ENABLED and group == QUADRUPED_GROUP:
        return SimpleNamespace(name="quadrupede", **{**std, **QUADRUPED})
    return SimpleNamespace(name="standard", **std)


def _widen(box, frac):
    m = frac * (box[2] - box[0])
    return (box[0] - m, box[1], box[2] + m, box[3])


# ======================================================================
# Kalman filter of one person: state (u, v, d, du, dv, dd), constant velocity, real dt
# ======================================================================
_UNKNOWN_DEPTH_VAR = 1e6


class KalmanUVD:
    """Box centre (px) and depth (m) with their velocities. Process noise from the white-noise
    acceleration model; gate with the Mahalanobis distance (it widens by itself while the person
    is not seen). A measurement without depth updates only (u, v)."""

    def __init__(self, u, v, d, t):
        self.x = np.array([u, v, 0.0 if d is None else d, 0.0, 0.0, 0.0], dtype=float)
        rp, rd = KF_MEAS_STD_PX ** 2, KF_MEAS_STD_DEPTH ** 2
        self.P = np.diag([rp, rp, _UNKNOWN_DEPTH_VAR if d is None else rd,
                          KF_INIT_VEL_STD_PX ** 2, KF_INIT_VEL_STD_PX ** 2, KF_INIT_VEL_STD_DEPTH ** 2])
        self.t = self.t_update = t

    def predict_to(self, t):
        dt = t - self.t
        if dt <= 0.0:
            return
        F = np.eye(6)
        F[0, 3] = F[1, 4] = F[2, 5] = dt
        Q = np.zeros((6, 6))
        for i, q in enumerate((KF_ACCEL_PX, KF_ACCEL_PX, KF_ACCEL_DEPTH)):
            Q[i, i] = q * dt ** 3 / 3.0
            Q[i, i + 3] = Q[i + 3, i] = q * dt ** 2 / 2.0
            Q[i + 3, i + 3] = q * dt
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.t = t

    @staticmethod
    def _H_R(with_depth):
        n = 3 if with_depth else 2
        H = np.zeros((n, 6))
        for i in range(n):
            H[i, i] = 1.0
        R = np.diag([KF_MEAS_STD_PX ** 2, KF_MEAS_STD_PX ** 2, KF_MEAS_STD_DEPTH ** 2][:n])
        return H, R

    def mahalanobis2(self, u, v, d=None, P=None):
        """(squared Mahalanobis distance from the prediction, degrees of freedom); P: covariance
        to gate with (default: the current one)."""
        H, R = self._H_R(d is not None)
        z = np.array([u, v] if d is None else [u, v, d], dtype=float)
        y = z - H @ self.x
        S = H @ (self.P if P is None else P) @ H.T + R
        return float(y @ np.linalg.solve(S, y)), (2 if d is None else 3)

    def update(self, u, v, d, t):
        if d is not None and self.P[2, 2] >= _UNKNOWN_DEPTH_VAR * 0.5:
            self.x[2], self.P[2, 2] = d, KF_MEAS_STD_DEPTH ** 2      # first depth ever: initialise
            self.P[2, :2] = self.P[:2, 2] = 0.0
        H, R = self._H_R(d is not None)
        z = np.array([u, v] if d is None else [u, v, d], dtype=float)
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I_KH = np.eye(6) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T                # Joseph form
        self.t = self.t_update = t

    def reset_depth(self, d):
        """Forget the depth estimate (and its velocity) and restart it from the measurement d."""
        self.x[2], self.x[5] = d, 0.0
        self.P[2, :] = self.P[:, 2] = 0.0
        self.P[5, :] = self.P[:, 5] = 0.0
        self.P[2, 2], self.P[5, 5] = KF_MEAS_STD_DEPTH ** 2, KF_INIT_VEL_STD_DEPTH ** 2

    @property
    def center(self):
        return float(self.x[0]), float(self.x[1])

    @property
    def depth(self):
        return None if self.P[2, 2] >= _UNKNOWN_DEPTH_VAR * 0.5 else float(self.x[2])

    @property
    def pos_std(self):
        return math.sqrt(max(0.5 * (self.P[0, 0] + self.P[1, 1]), 0.0))

    # --- same interface as KalmanOdom (the tracker and the overlay use only these) ---
    def set_camera(self, cam):
        pass                                   # image coordinates: the camera pose is not used

    def stop(self):
        self.x[3:6] = 0.0

    def should_freeze(self, t):
        return False                           # the tracker applies the HOLD_MAX_MISSES rule

    def point(self):
        return (float(self.x[0]), float(self.x[1]), float(self.x[2]))

    @staticmethod
    def project(p):
        return (p[0], p[1])

    def predicted_points(self, horizons):
        """[(u, v, radius px)] where the target is expected after each horizon (s)."""
        out = []
        for h in horizons:
            var = 0.5 * (self.P[0, 0] + self.P[1, 1]) + h * h * 0.5 * (self.P[3, 3] + self.P[4, 4]) \
                + KF_ACCEL_PX * h ** 3 / 3.0
            out.append((float(self.x[0] + self.x[3] * h), float(self.x[1] + self.x[4] * h),
                        2.0 * math.sqrt(var)))
        return out

    def state(self):
        return None                            # no position in odom

    def trajectory(self):
        return []                              # no shared trajectory with the image filter

    curve = False


# ======================================================================
# Filter in odom: state (X, Y, Z, VX, VY, VZ), measurement (u, v, d) through the camera
# ======================================================================
@dataclass
class Camera:
    """Pinhole camera, optical frame (x right, y down, z forward), and its pose in odom at the time
    of the image: p_odom = R @ p_cam + t. Without a pose (R = I, t = 0) odom = the camera frame."""
    fx: float
    fy: float
    cx: float
    cy: float
    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    t: np.ndarray = field(default_factory=lambda: np.zeros(3))
    posed: bool = False                        # True: R, t come from the TF (robot motion compensated)

    MIN_Z = 0.2                                # closer / behind the camera: not projectable

    def to_cam(self, p):
        return self.R.T @ (np.asarray(p, dtype=float) - self.t)

    def project(self, p):
        """(u, v, depth) of an odom point, None if it is behind the camera."""
        c = self.to_cam(p)
        if c[2] < self.MIN_Z:
            return None
        return self.fx * c[0] / c[2] + self.cx, self.fy * c[1] / c[2] + self.cy, float(c[2])

    def backproject(self, u, v, d):
        c = np.array([(u - self.cx) * d / self.fx, (v - self.cy) * d / self.fy, d], dtype=float)
        return self.R @ c + self.t

    def jac_back(self, u, v, d):
        """d(odom point) / d(u, v, d)."""
        J = np.array([[d / self.fx, 0.0, (u - self.cx) / self.fx],
                      [0.0, d / self.fy, (v - self.cy) / self.fy],
                      [0.0, 0.0, 1.0]])
        return self.R @ J

    def jac_proj(self, p):
        """d(u, v, d) / d(odom point) (3x3)."""
        c = self.to_cam(p)
        z = max(c[2], self.MIN_Z)
        J = np.array([[self.fx / z, 0.0, -self.fx * c[0] / z ** 2],
                      [0.0, self.fy / z, -self.fy * c[1] / z ** 2],
                      [0.0, 0.0, 1.0]])
        return J @ self.R.T


def default_camera():
    return Camera(*DEFAULT_INTRINSICS)


class TurnEstimator:
    """Walking direction, turn rate and its trend (tightening curve), from the filter velocity at
    every measurement."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._hist, self._rates = [], []
        self.heading, self.rate, self.accel = None, 0.0, 0.0

    def add(self, t, vx, vy):
        if math.hypot(vx, vy) < TURN_MIN_SPEED:
            return
        h = math.atan2(vy, vx)
        self.heading = h
        self._hist = [(ti, a) for ti, a in self._hist if t - ti <= TURN_WINDOW_SEC] + [(t, h)]
        t0, h0 = self._hist[0]
        if t - t0 > 1e-3:
            self.rate = math.atan2(math.sin(h - h0), math.cos(h - h0)) / (t - t0)
            self._rates.append((t, self.rate))
        self._rates = [(ti, w) for ti, w in self._rates if t - ti <= CURVE_TREND_WINDOW_SEC]
        self.accel = 0.0
        if len(self._rates) >= 3:
            ts, ws = [ti for ti, _ in self._rates], [w for _, w in self._rates]
            tm, wm = sum(ts) / len(ts), sum(ws) / len(ws)
            den = sum((ti - tm) ** 2 for ti in ts)
            a = sum((ti - tm) * (w - wm) for ti, w in zip(ts, ws)) / den if den > 1e-9 else 0.0
            if a * self.rate > 0.0:          # only a TIGHTENING curve is extrapolated
                self.accel = math.copysign(min(abs(a), math.radians(CURVE_MAX_TIGHTEN_DEG_S2)), a)

    @property
    def turning(self):
        return self.heading is not None and abs(self.rate) >= math.radians(TURN_MIN_DEG_S)


class KalmanOdom:
    """Target position and velocity in odom, constant velocity (white-noise acceleration).
    Measurement = box centre (u, v) and depth d, i.e. the odom point seen through the camera at the
    time of the image (extended Kalman filter): the robot motion is compensated by the camera pose.
    Without depth only (u, v) update the filter. Same interface as KalmanUVD for the tracker."""

    def __init__(self, u, v, d, t, cam, depth_hint=None):
        self.cam = cam or default_camera()
        self.depth_known = d is not None
        self.turn = TurnEstimator()
        self._place(u, v, d if d is not None else (depth_hint or KF3_DEFAULT_DEPTH), d is not None)
        self.t = self.t_update = self.t_meas = t
        self.frozen = False
        self._snapshot()

    def _place(self, u, v, d, depth_known):
        """Position on the ray of (u, v) at depth d, velocity unknown."""
        J = self.cam.jac_back(u, v, d)
        sd = KF_MEAS_STD_DEPTH if depth_known else KF3_UNKNOWN_DEPTH_STD
        Rm = np.diag([KF_MEAS_STD_PX ** 2, KF3_MEAS_STD_V_PX ** 2, sd ** 2])
        self.x = np.concatenate([self.cam.backproject(u, v, d), np.zeros(3)])
        self.P = np.zeros((6, 6))
        self.P[:3, :3] = J @ Rm @ J.T
        self.P[3:, 3:] = np.diag([KF3_INIT_VEL_STD_XY ** 2, KF3_INIT_VEL_STD_XY ** 2, KF3_INIT_VEL_STD_Z ** 2])
        self.p_meas = self.x[:3].copy()        # position at the last measurement (freeze distance)

    def set_camera(self, cam):
        if cam is not None:
            self.cam = cam

    @staticmethod
    def _FQ(dt):
        F = np.eye(6)
        F[0, 3] = F[1, 4] = F[2, 5] = dt
        Q = np.zeros((6, 6))
        for i, q in enumerate((KF3_ACCEL_XY, KF3_ACCEL_XY, KF3_ACCEL_Z)):
            Q[i, i] = q * dt ** 3 / 3.0
            Q[i, i + 3] = Q[i + 3, i] = q * dt ** 2 / 2.0
            Q[i + 3, i + 3] = q * dt
        return F, Q

    def predict_to(self, t):
        dt = t - self.t
        if dt <= 0.0:
            return
        F, Q = self._FQ(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.t = t

    def _H_R(self, with_depth):
        n = 3 if with_depth else 2
        H = np.zeros((n, 6))
        H[:, :3] = self.cam.jac_proj(self.x[:3])[:n]
        R = np.diag([KF_MEAS_STD_PX ** 2, KF3_MEAS_STD_V_PX ** 2, KF_MEAS_STD_DEPTH ** 2][:n])
        return H, R

    def mahalanobis2(self, u, v, d=None, P=None):
        dof = 2 if d is None else 3
        pr = self.cam.project(self.x[:3])
        if pr is None:
            return float("inf"), dof
        H, R = self._H_R(d is not None)
        y = np.array([u, v] if d is None else [u, v, d], dtype=float) - np.array(pr[:dof])
        S = H @ (self.P if P is None else P) @ H.T + R
        return float(y @ np.linalg.solve(S, y)), dof

    def update(self, u, v, d, t):
        pr = self.cam.project(self.x[:3])
        if pr is None or (d is not None and not self.depth_known):
            vel, Pv = self.x[3:].copy(), self.P[3:, 3:].copy()   # first depth ever (or the prediction is
            self._place(u, v, d if d is not None else (pr[2] if pr else KF3_DEFAULT_DEPTH), d is not None)
            if pr is not None:                                 # behind the camera): restart the position
                self.x[3:], self.P[3:, 3:] = vel, Pv
            self.depth_known = self.depth_known or d is not None
        else:
            H, R = self._H_R(d is not None)
            n = H.shape[0]
            y = np.array([u, v, d][:n], dtype=float) - np.array(pr[:n])
            S = H @ self.P @ H.T + R
            K = self.P @ H.T @ np.linalg.inv(S)
            self.x = self.x + K @ y
            I_KH = np.eye(6) - K @ H
            self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T            # Joseph form
        self.p_meas = self.x[:3].copy()
        self.t = self.t_update = self.t_meas = t
        self.frozen = False
        self.turn.add(t, self.x[3], self.x[4])
        self._snapshot()

    def reset_depth(self, d):
        """The filter depth is wrong (see DEPTH_RESYNC_*): move the point along the current ray to d."""
        u, v = self.center
        self._place(u, v, d, True)
        self.depth_known = True
        self.turn.reset()
        self.t_meas = self.t
        self._snapshot()

    def stop(self):
        """Prediction frozen at the END of the future trajectory (the same point for the tracker,
        the overlay and nav2_bridge): no more extrapolation."""
        self.x[:3] = self._traj[-1][1:]
        self.x[3:] = 0.0
        self.frozen = True

    def should_freeze(self, t):
        """Not measured beyond the end of the future trajectory (KF3_PREDICT_MAX_*, curve limits)."""
        return not self.frozen and t - self.t_meas > self._traj[-1][0] + 1e-6

    # --- future trajectory ---------------------------------------------------------------------
    def _snapshot(self):
        """At every measurement: the trajectory from here (kept until the next measurement)."""
        self.x_meas, self.P_meas = self.x.copy(), self.P.copy()
        self._traj = self._trajectory()

    def _trajectory(self):
        """[(h, X, Y, Z)]: where the target is expected h seconds after the last measurement, every
        KF_FUTURE_STEP_SEC. Ends when the horizontal uncertainty exceeds KF3_PREDICT_MAX_STD_M, after
        KF3_PREDICT_MAX_SEC, beyond KF3_PREDICT_MAX_DIST_M, or at the end of the curve (CURVE_*)."""
        x0, P0 = self.x_meas, self.P_meas
        p0 = x0[:3]
        pts = [(0.0, float(p0[0]), float(p0[1]), float(p0[2]))]
        speed = math.hypot(x0[3], x0[4])
        curve = self.turn.turning and speed >= TURN_MIN_SPEED
        heading, w_max = math.atan2(x0[4], x0[3]), math.radians(CURVE_MAX_RATE_DEG_S)
        pos, s_done, turned, t_arc = p0.copy(), 0.0, 0.0, 0.0
        step = KF_FUTURE_STEP_SEC
        for k in range(1, int(round(KF3_PREDICT_MAX_SEC / step)) + 1):
            h = k * step
            F, Q = self._FQ(h)
            P = F @ P0 @ F.T + Q
            if math.sqrt(max(0.5 * (P[0, 0] + P[1, 1]), 0.0)) > KF3_PREDICT_MAX_STD_M:
                break
            if curve:
                ds, end = step / 10.0, False
                for _ in range(10):
                    w = max(-w_max, min(w_max, self.turn.rate + self.turn.accel * t_arc))
                    heading += w * ds
                    turned += w * ds
                    pos = pos + speed * ds * np.array([math.cos(heading), math.sin(heading), 0.0])
                    s_done += speed * ds
                    t_arc += ds
                    if s_done >= CURVE_ARC_MAX_M or abs(turned) >= math.radians(CURVE_MAX_TURN_DEG):
                        end = True
                        break
                pts.append((h, float(pos[0]), float(pos[1]), float(pos[2])))
                if end:
                    break
            else:
                pos = p0 + x0[3:] * h
                if math.hypot(pos[0] - p0[0], pos[1] - p0[1]) > KF3_PREDICT_MAX_DIST_M:
                    break
                pts.append((h, float(pos[0]), float(pos[1]), float(pos[2])))
        return pts

    def trajectory(self):
        """[(t, X, Y, Z)] with absolute times (same clock as the tracker)."""
        return [(self.t_meas + h, x, y, z) for h, x, y, z in self._traj]

    @property
    def curve(self):
        return self.turn.turning

    @property
    def horizontal_std(self):
        return math.sqrt(max(0.5 * (self.P[0, 0] + self.P[1, 1]), 0.0))

    @property
    def center(self):
        pr = self.cam.project(self.x[:3])
        return (-1e6, -1e6) if pr is None else (float(pr[0]), float(pr[1]))

    @property
    def depth(self):
        pr = self.cam.project(self.x[:3])
        return None if pr is None or not self.depth_known else float(pr[2])

    @property
    def pos_std(self):
        """Uncertainty of the prediction in the image (px)."""
        if self.cam.project(self.x[:3]) is None:
            return 1e6
        J = self.cam.jac_proj(self.x[:3])[:2]
        C = J @ self.P[:3, :3] @ J.T
        return math.sqrt(max(0.5 * (C[0, 0] + C[1, 1]), 0.0))

    def point(self):
        return self.x[:3].copy()

    def project(self, p):
        pr = self.cam.project(p)
        return None if pr is None else (pr[0], pr[1])

    def predicted_points(self, horizons):
        """[(u, v, radius px)]: where the target is expected after each horizon (s), seen from the
        CURRENT camera pose, with a radius of 2 std of the predicted position."""
        out = []
        for h in horizons:
            F, Q = self._FQ(h)
            x, P = F @ self.x, F @ self.P @ F.T + Q
            pr = self.cam.project(x[:3])
            if pr is None:
                continue
            J = self.cam.jac_proj(x[:3])[:2]
            C = J @ P[:3, :3] @ J.T
            out.append((float(pr[0]), float(pr[1]), 2.0 * math.sqrt(max(0.5 * (C[0, 0] + C[1, 1]), 0.0))))
        return out

    def state(self):
        """(position, velocity, 6x6 covariance) in odom."""
        return self.x[:3].copy(), self.x[3:].copy(), self.P.copy()


def make_filter(u, v, d, t, cam=None, depth_hint=None):
    """The predictive filter chosen with PREDICTIVE_FILTER."""
    if PREDICTIVE_FILTER == "odom":
        return KalmanOdom(u, v, d, t, cam, depth_hint)
    return KalmanUVD(u, v, d, t)


# ======================================================================
# Multi-view gallery of the target (long-term memory)
# ======================================================================
class Gallery:
    def __init__(self):
        self.views = []
        self.thumbs = []                 # image crop of each view (same order), for debugging / the thesis
        self._anchored = False
        self._chain = 0
        self._prev_box = self._prev_dist = None

    def reset(self, anchor=None, box=None, dist=None, thumb=None):
        self.views = [anchor] if anchor is not None else []
        self.thumbs = [thumb] if anchor is not None else []
        self._anchored, self._chain = anchor is not None, 0
        self._prev_box, self._prev_dist = box, dist

    def __len__(self):
        return len(self.views)

    def best(self, embedding):
        """(max similarity over the views, its components, view index); (0, None, -1) if none."""
        best = (0.0, None, -1)
        if embedding is None:
            return best
        for idx, view in enumerate(self.views):
            comp = components(embedding, view)
            sim = combine_reid_components(comp, W_COSINE, W_EUCLIDEAN, W_MAGNITUDE)
            if best[2] == -1 or sim > best[0]:
                best = (sim, comp, idx)
        return best

    def try_add(self, embedding, thumb=None):
        if embedding is None or not self.views or not REID_GALLERY_SIZE:
            return False
        if self.best(embedding)[0] >= GALLERY_NOVELTY_SIM:
            return False                                  # already known
        if len(self.views) >= REID_GALLERY_SIZE:
            self.views.pop(1)                             # oldest after the anchor
            self.thumbs.pop(1)
        self.views.append(embedding)
        self.thumbs.append(None if thumb is None else thumb.copy())
        return True

    def chain_reset(self):
        self._anchored, self._chain = False, 0

    def update(self, embedding, box, dist, sim_gallery, clean, thumb=None):
        """After a CONFIRMED / KEPT frame. Returns "strong", "chain" or None (view added?)."""
        prev_box, prev_dist = self._prev_box, self._prev_dist
        self._prev_box, self._prev_dist = box, dist
        if not self.views or not clean or embedding is None or dist is None:
            self.chain_reset()
            return None
        strong = GALLERY_STRONG_ADD_MIN_SIM if getattr(self, "strong_add_min", None) is None else self.strong_add_min
        if sim_gallery >= strong:                          # sure from the long-term memory itself
            self._anchored, self._chain = True, 0
            return "strong" if self.try_add(embedding, thumb) else None
        continuous = (prev_box is not None and prev_dist is not None
                      and iou(box, prev_box) >= GALLERY_CHAIN_MIN_IOU
                      and abs(dist - prev_dist) <= GALLERY_CHAIN_MAX_DEPTH_STEP_M)
        if not self._anchored or not continuous:
            self.chain_reset()
            return None
        self._chain += 1
        if self._chain < GALLERY_CHAIN_FRAMES:
            return None
        self._chain = 0
        return "chain" if self.try_add(embedding, thumb) else None


# ======================================================================
# Data
# ======================================================================
@dataclass(eq=False)       # identity comparison: the fields contain numpy arrays
class Detection:
    box: tuple
    score: float
    class_name: str
    dist: Optional[float]
    embedding: Optional[np.ndarray]
    sim_ff: float = 0.0              # similarity to the last clean target embedding (frame-to-frame)
    sim_gal: float = 0.0             # similarity to the gallery
    gal_comp: Optional[tuple] = None
    gal_view: int = -1
    sim_target: float = 0.0          # max(sim_gal, min(sim_ff, sim_gal + FF_MAX_LEAD))
    sims_other: list = field(default_factory=list)   # similarity to each person of the previous call
    note: str = ""                   # why it is not a target candidate
    near_bonus: bool = False         # (A) near the prediction AND allowed the TARGET_LOW bonus
    depth_occluded: bool = False     # its depth does not belong to the target (occluder / background)
    thumb: Optional[np.ndarray] = None   # image crop (a view into the frame), only for the gallery images
    class_scores: dict = field(default_factory=dict)   # every class YOLOE gave to this box (dedupe)

    @property
    def center(self):
        return box_center(self.box)

    @property
    def best_other(self):
        return max(self.sims_other) if self.sims_other else None


@dataclass(eq=False)
class Person:
    """The target, kept all session (box, depth, last clean embedding, the ONE Kalman filter)."""
    box: tuple
    dist: Optional[float]
    embedding: Optional[np.ndarray]
    kf: object                            # KalmanOdom or KalmanUVD (PREDICTIVE_FILTER)
    misses: int = 0
    gate_P: Optional[np.ndarray] = None   # covariance of the gate at the last MEASUREMENT (frozen gate)
    unmeasured: int = 0                   # calls since the Kalman was last updated with a measurement
                                          # (a miss OR an occluded frame: seen, but nothing measured)

    @classmethod
    def from_detection(cls, d, t, cam=None):
        return cls(d.box, d.dist, d.embedding, make_filter(d.center[0], d.center[1], d.dist, t, cam))

    def follow(self, d, t, learn=True, use_depth=True):
        dist = d.dist if use_depth else None
        self.kf.update(d.center[0], d.center[1], dist, t)
        self.box = d.box
        if dist is not None:
            self.dist = dist
        if learn and d.embedding is not None:
            self.embedding = d.embedding
        self.misses = 0
        self.unmeasured = 0


@dataclass(eq=False)
class Other:
    """A person of the PREVIOUS call that is not the target: last box and embedding (no Kalman)."""
    box: tuple
    embedding: Optional[np.ndarray]


@dataclass
class FrameResult:
    status: str                       # CONFIRMED, KEPT, PENDING, MISSED
    mode: str                         # "breve" or "riacquisizione"
    det: Optional[Detection] = None
    reason: Optional[str] = None
    clean: bool = False
    gallery_added: Optional[str] = None
    pending_count: int = 0
    occluded: bool = False            # target seen, but its depth was discarded (not published)
    occluder_visible: bool = False    # ... and a detection overlaps it (a real occlusion)
    depth_resync: int = 0             # consecutive coherent discarded depths (re-sync at DEPTH_RESYNC_FRAMES)
    jump: bool = False                # taken FAR from the prediction (by appearance): Kalman restarted there

    @property
    def seen(self):
        return self.status in (CONFIRMED, KEPT)


# ======================================================================
# Tracker
# ======================================================================
class TargetTracker:
    """The node calls lock() once in SEARCH, then step() at every call in TRACKING / RECOVERY."""

    def __init__(self):
        self.n_confirmed = self.n_kept = self.n_steps = 0     # metrics: whole node life
        self.gallery_added = {}
        self.step_ms_sum = self.step_ms_max = 0.0
        # sim_target samples for the metrics JSON ("sim_stats": threshold calibration), whole node life
        self.sim_samples = {"target_confirmed": [], "target_kept": [], "others_target_seen": []}
        self.n_intruder_blocked = 0      # detections excluded because they continue an intruder (B)
        self.n_no_near_bonus_calls = 0   # calls run without the near-prediction bonus (A)
        self.reset()

    def reset(self):
        """Forgets target, people and gallery (back to WAITING_TRIGGER)."""
        self.target = None
        self.others = []                 # people of the PREVIOUS call (rewritten every call)
        self.gallery = Gallery()
        self.last_kf = None              # Kalman row of the last call (published on KF_TOPIC)
        self.target_group = None
        self.p = profile_for_group(None) # thresholds of the target's profile (set at the lock)
        if not hasattr(self, "cam"):
            self.cam = default_camera()  # replaced by the node's camera (intrinsics + pose) at every call
        self._at_prediction = []         # (A) depths of the people at the predicted point (prev. call)
        self._prev_other_boxes = []      # (A) (box, depth) of the previous call that were not the target
        self.trail = deque(maxlen=KF_TRAIL_LEN)   # (filter point, seen, ...) of the target, last calls
        self._discarded_depths = []
        self._reset_pending()

    def lock(self, target_det, other_dets, t, cam=None, group=None):
        """SEARCH: the chosen detection = TARGET, the other detections = people of this call.
        cam: Camera with the pose of the camera in odom at the time of the image (None: unknown).
        group: class group voted over the SEARCH frames (None: from this detection)."""
        self.reset()
        if cam is not None:
            self.cam = cam
        self.target = Person.from_detection(target_det, t, self.cam)
        if group is None:
            group = vote_class_group([target_det.class_scores or {target_det.class_name: target_det.score}])
        self.target_group = group
        self.p = profile_for_group(group)
        self.gallery.strong_add_min = self.p.GALLERY_STRONG_ADD_MIN_SIM
        self.gallery.reset(target_det.embedding, target_det.box, target_det.dist, target_det.thumb)
        self.others = [Other(d.box, d.embedding) for d in other_dets if d.embedding is not None]

    def switch_to_posed_camera(self, cam, t):
        """The TF became available only after the lock: the filter restarts in odom from its current
        estimate (seen from the same image coordinates), so the robot motion is compensated from now on."""
        self.cam = cam
        tgt = self.target
        if tgt is None or PREDICTIVE_FILTER != "odom":
            return
        (u, v), d = tgt.kf.center, tgt.kf.depth
        if abs(u) > 1e5:
            return
        tgt.kf = make_filter(u, v, d, t, cam, depth_hint=tgt.dist)
        tgt.gate_P = tgt.kf.P.copy()

    def enter_recovery(self):
        self._reset_pending()

    def target_prediction(self, t):
        """(centre, std px) of the target prediction, None if older than PREDICTION_MAX_SEC."""
        if self.target is None or t - self.target.kf.t_update > PREDICTION_MAX_SEC:
            return None
        return self.target.kf.center, self.target.kf.pos_std

    # ------------------------------------------------------------------
    def step(self, dets, t, recovery=False, cam=None):
        t0 = time.perf_counter()
        tgt = self.target
        if cam is not None:
            self.cam = cam                     # camera pose of THIS image (None: keep the last one)
        tgt.kf.set_camera(self.cam)
        if PREDICTIVE_FILTER == "odom":
            tgt.kf.predict_to(t)               # not measured: the prediction keeps moving while it is
            if tgt.kf.should_freeze(t):        # reliable (KF3_PREDICT_MAX_*), then it stops there
                tgt.kf.stop()
        else:
            if tgt.unmeasured > HOLD_MAX_MISSES:
                tgt.kf.stop()                  # not measured for a while (lost, or occluded): the last
            tgt.kf.predict_to(t)               # known position, no more extrapolation
        # (A) who else was at the predicted point in the previous call (their depths)
        self._at_prediction = self._someone_at_prediction() if NEAR_BONUS_ONLY_WHEN_ALONE else []
        if self._at_prediction and not recovery:
            self.n_no_near_bonus_calls += 1
        if tgt.unmeasured == 0 or tgt.gate_P is None:
            tgt.gate_P = tgt.kf.P.copy()      # one call after the last MEASUREMENT: the tight gate.
        # An OCCLUDED frame (seen, depth discarded, Kalman not updated) must NOT refresh it: P grows at
        # every unmeasured call (~140 px / 0.84 m of std after 2 calls) and an intruder 2 m closer and
        # 200 px aside got inside and was taken as KEPT (photos IMG_0347/0348, 14/10).
        tgt.unmeasured += 1                   # reset by follow() if this call measures the target
        continuity = not recovery and tgt.misses <= HOLD_MAX_MISSES
        res = FrameResult(MISSED, "breve" if continuity else "riacquisizione")

        self._compute_similarities(dets)
        cands = self._candidates(dets, continuity)
        j_t = self._decide(res, dets, cands, continuity)
        self._update_target(res, dets, t)
        self._rewrite_others(res, dets, j_t)
        self._collect_sims(res, dets)
        self._log_kf(res, t)
        self._prev_other_boxes = [(d.box, d.dist) for d in dets if not (res.seen and d is res.det)]

        ms = (time.perf_counter() - t0) * 1000.0
        self.step_ms_sum += ms
        self.step_ms_max = max(self.step_ms_max, ms)
        self.n_steps += 1
        return res

    # 2. similarities ----------------------------------------------------
    def _compute_similarities(self, dets):
        tgt = self.target
        for d in dets:
            d.sim_gal, d.gal_comp, d.gal_view = self.gallery.best(d.embedding)
            d.sim_ff = similarity(tgt.embedding, d.embedding) if tgt.embedding is not None else 0.0
            d.sim_target = max(d.sim_gal, min(d.sim_ff, d.sim_gal + FF_MAX_LEAD))
            d.sims_other = [similarity(o.embedding, d.embedding) for o in self.others]

    @staticmethod
    def _gate(person, d, use_depth=True, P=None):
        """(inside the Kalman gate, normalised Mahalanobis^2 in [0, 1])."""
        use_depth = use_depth and d.dist is not None and person.kf.depth is not None
        m2, dof = person.kf.mahalanobis2(d.center[0], d.center[1], d.dist if use_depth else None, P)
        chi2 = GATE_CHI2_3D if dof == 3 else GATE_CHI2_2D
        return m2 <= chi2, min(m2 / chi2, 1.0)

    def _candidates(self, dets, continuity):
        """EVERY detection is examined, wherever it is: the Kalman never excludes anybody.
             near the prediction (inside the gate, target seen in the last HOLD_MAX_MISSES calls)
                 -> sim_target >= TARGET_LOW is enough (the position is the second proof)
             anywhere else, or after a loss
                 -> sim_target >= REACQ_THRESHOLD and depth required (only the appearance counts)
           always: gallery veto, never more similar to a person of the previous call, never an
           intruder that continues (B).
           (A) somebody else was at the predicted point in the previous call: no near bonus
           (REACQ_THRESHOLD near the prediction too) unless the depth tells them apart (_near_bonus_ok).
           Returns {j: near}."""
        out = {}
        for j, d in enumerate(dets):
            d.depth_occluded = False
            if d.embedding is None:
                d.note = "no_embedding"
                continue
            if self.p.GALLERY_VETO_MIN is not None and d.sim_gal < self.p.GALLERY_VETO_MIN:
                d.note = "gallery_veto"
                continue
            if d.best_other is not None and d.best_other >= d.sim_target:
                d.note = "looks_like_other"
                continue
            if self._continues_other(d):
                d.note = "continues_other"
                continue
            if self._sticky_intruder(d):
                d.note = "intruder"
                self.n_intruder_blocked += 1
                continue
            if not self._class_ok(d):
                d.note = "wrong_class"
                continue
            near = self._target_gate(d)[0] if continuity else False
            if not near:                      # its depth may still be an occluder's / the background
                d.depth_occluded = continuity and self._depth_from_occluder(d)
                if REACQ_REQUIRE_DEPTH and d.dist is None:
                    d.note = "no_depth"
                    continue
                if d.sim_target < self.p.REACQ_THRESHOLD:
                    d.note = "far_low_similarity" if continuity else "low_similarity"
                    continue
                if not self._reacq_distance_ok(d):
                    d.note = "implausible_distance"
                    continue
            else:
                d.near_bonus = self._near_bonus_ok(d)
                if d.sim_target < (self.p.TARGET_LOW if d.near_bonus else self.p.REACQ_THRESHOLD):
                    d.note = "low_similarity"
                    continue
            out[j] = near
        return out

    def _someone_at_prediction(self):
        """(A) Depths (None = unknown) of the detections of the previous call (not the target) that
        touched the target's box at the predicted point, widened by CLEAN_MARGIN. [] = nobody."""
        zone = _widen(self._target_ref_box(), CLEAN_MARGIN)
        return [dist for box, dist in self._prev_other_boxes if _touches(box, zone)]

    def _near_bonus_ok(self, d):
        """(A) The near-prediction bonus for d: nobody else was there, or the depth separates d from
        them (d at the target's predicted depth and nearer to it than to each of their depths)."""
        if not self._at_prediction:
            return True
        kf_d = self.target.kf.depth
        if d.dist is None or kf_d is None or d.depth_occluded:
            return False
        off = abs(d.dist - kf_d)
        return off <= OCCLUSION_DEPTH_JUMP_M and \
            all(od is not None and abs(d.dist - od) > off for od in self._at_prediction)

    def _sticky_intruder(self, d):
        """(B) d continues a person of the previous call (an intruder): its box is closer, by
        INTRUDER_STICKY_MARGIN, to that person's box than to the target's (last full box at the
        prediction). Only geometry: the appearance of a mixed crop is not trusted. Released if the
        gallery recognises it clearly (>= INTRUDER_RELEASE_SIM)."""
        if not INTRUDER_STICKY or not self.others or self.target is None:
            return False
        if d.sim_gal >= self.p.INTRUDER_RELEASE_SIM:
            return False
        c_other = min(self._box_cost(d.box, o.box) for o in self.others)
        c_target = self._box_cost(d.box, self._target_ref_box())
        return c_other <= OTHER_CONTINUITY_MAX and c_other < c_target - INTRUDER_STICKY_MARGIN

    def _class_ok(self, d):
        """YOLOE scores this box at least as much for the target's class group as for any other."""
        if getattr(self, "target_group", None) is None:
            return True
        scores = d.class_scores or {d.class_name: d.score}
        mine = max((s for c, s in scores.items() if class_compatible(c, self.target_group)), default=0.0)
        others = max((s for c, s in scores.items() if not class_compatible(c, self.target_group)), default=0.0)
        return mine > 0.0 and mine >= others

    def _reacq_distance_ok(self, d):
        """Far / re-acquisition: the depth must be plausible for the target (see REACQ_MAX_RANGE_M,
        REACQ_DEPTH_TOL_M, REACQ_DEPTH_SPEED_MPS). Without depth it is decided by REACQ_REQUIRE_DEPTH."""
        if d.dist is None:
            return True
        if REACQ_MAX_RANGE_M is not None and d.dist > REACQ_MAX_RANGE_M:
            return False
        last = self.target.dist
        if last is None:
            return True
        elapsed = max(0.0, self.target.kf.t - self.target.kf.t_update)
        return abs(d.dist - last) <= REACQ_DEPTH_TOL_M + REACQ_DEPTH_SPEED_MPS * elapsed

    @staticmethod
    def _box_cost(box, ref):
        """How much `box` looks like the continuation of `ref`: horizontal shift in box widths +
        size change (a person 1 m closer has a box ~1.5x taller: 2 |ln 1.5| = 0.8)."""
        w, h = max(ref[2] - ref[0], 1.0), max(ref[3] - ref[1], 1.0)
        du = abs((box[0] + box[2]) - (ref[0] + ref[2])) / 2.0
        return du / w + 2.0 * abs(math.log(max(box[3] - box[1], 1.0) / h))

    def _continues_other(self, d):
        """It is a person of the previous call walking on: closer (position AND size) to that person's
        box than to the target's (its last full box, where the Kalman predicts it). Clear geometry
        decides alone; otherwise only if the appearance is not clearly the target's either."""
        if not self.others or d.best_other is None:
            return False
        c_other = min(self._box_cost(d.box, o.box) for o in self.others)
        c_target = self._box_cost(d.box, self._target_ref_box())
        if c_other > OTHER_CONTINUITY_MAX or c_other >= c_target - 0.2:   # not clearly theirs
            return False
        if c_target - c_other >= 0.5:          # clear geometry (e.g. a closer, bigger person): it
            return True                        # beats the appearance, which a mixed crop fools
        return d.sim_target < d.best_other + AMBIGUITY_MARGIN

    def _target_ref_box(self):
        """The target's last FULL box, moved to where the Kalman predicts it."""
        b, (u, v) = self.target.box, self.target.kf.center
        w, h = b[2] - b[0], b[3] - b[1]
        return (u - w / 2.0, v - h / 2.0, u + w / 2.0, v + h / 2.0)

    def _target_gate(self, d):
        """Kalman gate of the target; on a partial occlusion (see OCCLUSION_*) the depth is ignored."""
        d.depth_occluded = self._depth_from_occluder(d)   # checked first: while the target is
        P = self.target.gate_P if SHORT_GAP_FROZEN_GATE else None
        return self._gate(self.target, d, use_depth=not d.depth_occluded, P=P)   # hidden the depth gate widens

    def _depth_from_occluder(self, d):
        tgt = self.target
        pred = tgt.kf.depth
        if not OCCLUSION_DEPTH_FALLBACK or d.dist is None or pred is None or tgt.dist is None:
            return False
        if abs(d.dist - pred) <= OCCLUSION_DEPTH_JUMP_M:
            return False
        ratio = (d.box[3] - d.box[1]) / ((tgt.box[3] - tgt.box[1]) * tgt.dist / pred)
        if d.dist < pred:                      # occluder in front: the box must not look closer
            return ratio <= OCCLUSION_MAX_SIZE_RATIO
        return ratio >= 1.0 / OCCLUSION_MAX_SIZE_RATIO   # background: the box must not look farther

    # 4. decision ---------------------------------------------------------
    def _decide(self, res, dets, cands, continuity):
        """APPEARANCE decides: the candidate most similar to the target, if clearly more similar
        (AMBIGUITY_MARGIN) than the second. On a tie the gallery decides, then the one near the
        Kalman prediction wins (the position only breaks ties, and only if nobody else was at the
        predicted point, see (A)); otherwise nobody.
        Near the prediction: CONFIRMED / KEPT at once. Elsewhere (a jump, or after a loss): at once
        from REACQ_IMMEDIATE_SIM, otherwise REACQ_CONFIRM_FRAMES calls in the same place."""
        order = sorted(cands, key=lambda j: dets[j].sim_target, reverse=True)
        j_t = order[0] if order else None
        if j_t is not None and len(order) > 1:
            j2 = order[1]
            if dets[j_t].sim_target - dets[j2].sim_target < AMBIGUITY_MARGIN:
                dg = dets[j_t].sim_gal - dets[j2].sim_gal      # tie: first the LONG-TERM memory decides
                if abs(dg) >= GALLERY_TIEBREAK:                # (the last embedding may be the impostor's)
                    j_t = j_t if dg > 0 else j2
                elif cands[j_t] == cands[j2] or \
                        not dets[j_t if cands[j_t] else j2].near_bonus:   # (A) the near one is not
                    dets[j_t].note, j_t = "ambiguous", None                 # told apart: no position
                                                                            # tie-break
                elif cands[j2]:
                    j_t = j2                                   # then the position: the near one
        if j_t is None:
            if self._pending_box is not None:
                self._pending_gap += 1
                if self._pending_gap > REACQ_MAX_GAP:
                    self._reset_pending()
            notes = [d.note for d in dets if d.note]
            res.reason = max(notes, key=_REASONS.index) if notes else "no_detections"
            return None

        d, near = dets[j_t], cands[j_t]
        res.det = d
        if near:
            res.status = CONFIRMED if d.sim_target >= self.p.TARGET_HIGH else KEPT
            self._reset_pending()
        elif d.sim_target >= self.p.REACQ_IMMEDIATE_SIM or self._confirm_pending(d.box):
            res.status, res.jump = CONFIRMED, True
            self._reset_pending()
        else:
            res.status, res.reason, res.pending_count = PENDING, "pending_confirmation", self._pending_count
        return j_t

    # 5. updates ------------------------------------------------------------
    def _update_target(self, res, dets, t):
        tgt = self.target
        if not res.seen:
            tgt.misses += 1
            self.gallery.chain_reset()
            return
        d = res.det
        if res.jump:                          # somewhere else than predicted: the old motion is meaningless
            tgt.kf = make_filter(d.center[0], d.center[1], tgt.kf.depth if d.depth_occluded else d.dist, t,
                                 self.cam, depth_hint=tgt.kf.depth)
            tgt.gate_P = tgt.kf.P.copy()
            self._discarded_depths = []
        if d.depth_occluded and not self._depth_resync(res, d, dets):
            res.occluded = True              # depth discarded: seen, but nothing is measured from
            tgt.misses = 0                   # this crop (a strip: its centre is not the body's centre,
            tgt.kf.t_update = t              # it would give the Kalman a false velocity) and nothing learnt
            self.gallery.chain_reset()
            self.n_kept += 1
            return
        if not d.depth_occluded:
            self._discarded_depths = []
        was_seen = tgt.misses == 0 and not res.jump
        # a look-alike in view (gallery similarity close to the chosen one): learn nothing, the choice
        # may be the wrong one and the memory would be contaminated with the impostor
        lookalike = any(o is not d and o.embedding is not None and o.sim_gal >= d.sim_gal - AMBIGUITY_MARGIN
                        for o in dets)
        res.clean = self._clean(d, dets) and not lookalike
        tgt.follow(d, t, learn=res.clean)   # the target embedding is learnt only in clean frames
        # the gallery (long-term memory) only from continuous tracking: never from the first call
        # after a miss or a jump (e.g. an intruder's mixed crop taken while the target was hidden)
        res.gallery_added = self.gallery.update(d.embedding, d.box, d.dist, d.sim_gal,
                                                res.clean and was_seen, d.thumb)
        if res.gallery_added:
            self.gallery_added[res.gallery_added] = self.gallery_added.get(res.gallery_added, 0) + 1
        if res.status == CONFIRMED:
            self.n_confirmed += 1
        else:
            self.n_kept += 1

    def _depth_resync(self, res, d, dets):
        """Discarded depth: True if the Kalman depth must be re-synced to it (see DEPTH_RESYNC_*)."""
        zone = _widen(d.box, 0.15)           # an occluder is on / right next to the visible part
        kf_d = self.target.kf.depth           # ... and IN FRONT of the target (or too close to measure)
        res.occluder_visible = any(
            o is not d and _touches(o.box, zone)
            and (o.dist is None or kf_d is None or o.dist < kf_d - OCCLUSION_DEPTH_JUMP_M / 2)
            for o in dets)
        if res.occluder_visible or d.dist is None:
            self._discarded_depths = []
            return False
        self._discarded_depths.append(d.dist)
        recent = self._discarded_depths[-DEPTH_RESYNC_FRAMES:]
        res.depth_resync = len(recent)
        if len(recent) < DEPTH_RESYNC_FRAMES or max(recent) - min(recent) > DEPTH_RESYNC_MAX_SPREAD_M:
            return False
        self.target.kf.reset_depth(float(np.median(recent)))
        self.target.gate_P = self.target.kf.P.copy()
        self._discarded_depths = []
        res.depth_resync = 0
        return True

    def _clean(self, d, dets):
        """Nobody touches the target box widened by CLEAN_MARGIN (neither a detection nor where a
        person of the previous call was: it may be hidden behind the target)."""
        if d.embedding is None or d.dist is None:
            return False
        zone = _widen(d.box, CLEAN_MARGIN)
        return not any(o is not d and _touches(o.box, zone) for o in dets) and \
            not any(_touches(o.box, zone) for o in self.others)

    def _rewrite_others(self, res, dets, j_t):
        """New list of people (box + embedding, no Kalman) = the detections CLEARLY not the target:
          - target seen: clearly less similar to the target than the chosen one (by AMBIGUITY_MARGIN),
            wherever they are: a person that walks up to the target and covers it stays a person;
          - target not seen: a NEW entry only if clearly different from the target (sim_target <
            TARGET_LOW - AMBIGUITY_MARGIN, below the target's worst views): once in the list the
            target would be most similar to itself a call earlier and blocked for good (video 14/10
            11:07: entered at 0.44, then lost at 0.46-0.49 for 10 s). A person already in the list
            stays while still more similar to itself than to the target.
        The real target must never enter this list (it would then be blocked as 'looks_like_other').
          - (B) whoever CONTINUES a person of the previous call stays in the list in any case (sticky
            intruder, released only by the gallery: see _sticky_intruder)."""
        ref = res.det if res.seen else None
        new = []
        for j, d in enumerate(dets):
            if j == j_t or d.embedding is None:
                continue
            if self._sticky_intruder(d):
                other = True
            elif res.seen:
                other = d.sim_target < ref.sim_target - AMBIGUITY_MARGIN or self._continues_other(d)
                # never a detection the GALLERY rates as the target at least as much as the chosen one:
                # if the chosen one is wrong, that is the real target, and as an "other" it would be
                # blocked for good ('similar to another person' = to itself)
                other = other and d.sim_gal < ref.sim_gal - AMBIGUITY_MARGIN
            else:
                # target not seen. A NEW entry only if clearly not the target (below its worst
                # views: TARGET_LOW - AMBIGUITY_MARGIN); 0.44 in a bad pose let the target itself in
                # (video 14/10 11:07). Whoever already was a person and is still more similar to
                # that person than to the target STAYS (an intruder covering the target: mixed
                # crop, its similarity to the target can be high)
                stays = d.best_other is not None and d.best_other >= d.sim_target
                other = d.sim_target < self.p.TARGET_LOW - AMBIGUITY_MARGIN or self._continues_other(d) or stays
            if other:
                new.append(Other(d.box, d.embedding))
        self.others = new

    def _collect_sims(self, res, dets):
        """sim_target samples for "sim_stats" in the metrics JSON: the target (CONFIRMED / KEPT,
        not occluded) and the other people of the calls where the target was identified."""
        if not res.seen:
            return
        for d in dets:
            if d.embedding is None:
                continue
            if d is res.det:
                if not res.occluded:
                    key = "target_confirmed" if res.status == CONFIRMED else "target_kept"
                    self.sim_samples[key].append(float(d.sim_target))
            else:
                self.sim_samples["others_target_seen"].append(float(d.sim_target))

    def sim_stats(self):
        """{group: {n, min, p10, median, p90, max}} of the sim_target samples (metrics JSON)."""
        out = {}
        for key, vals in self.sim_samples.items():
            if not vals:
                out[key] = {"n": 0}
                continue
            a = np.asarray(vals)
            out[key] = {"n": int(a.size), "min": round(float(a.min()), 3),
                        "p10": round(float(np.percentile(a, 10)), 3), "median": round(float(np.median(a)), 3),
                        "p90": round(float(np.percentile(a, 90)), 3), "max": round(float(a.max()), 3)}
        return out

    def _log_kf(self, res, t):
        """Target Kalman estimate of this call: published on KF_TOPIC and drawn as the trail."""
        kf = self.target.kf
        d = res.det if res.seen else None
        (u, v), kd = kf.center, kf.depth
        self.last_kf = {"post_u": float(u), "post_v": float(v), "post_d": float(kd) if kd is not None else 0.0}
        self.trail.append((kf.point(), res.seen, kd,
                           None if d is None or d.dist is None else float(d.dist), res.occluded))

    # re-acquisition confirmation ------------------------------------------
    def _reset_pending(self):
        self._pending_box, self._pending_count, self._pending_gap = None, 0, 0

    def _confirm_pending(self, box):
        self._pending_gap = 0
        same = self._pending_box is not None and \
            distance(box_center(box), box_center(self._pending_box)) <= REACQ_PX_TOLERANCE
        self._pending_count = self._pending_count + 1 if same else 1
        self._pending_box = box
        return self._pending_count >= REACQ_CONFIRM_FRAMES
