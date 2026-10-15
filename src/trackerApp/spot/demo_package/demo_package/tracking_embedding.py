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
W_COSINE = 0.3                   # ReID similarity = weighted mean of cosine, Euclidean and magnitude terms
W_EUCLIDEAN = 0.7
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
TARGET_HIGH = 0.60               # target CONFIRMED
TARGET_LOW = 0.45                # target KEPT (uncertain) between LOW and HIGH
GALLERY_VETO_MIN = 0.43# never the target below this similarity to the gallery (None = off)
FF_MAX_LEAD = 0.15               # sim_target = max(gallery, min(last embedding, gallery + this)): the short-term
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
AMBIGUITY_MARGIN = 0.08          # appearance decides; two candidates closer than this in sim_target are a tie:
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

REACQ_THRESHOLD = 0.50           # re-acquisition (was 0.60: you, frontal in RECOVERY, stayed at 0.52-0.58)
REACQ_IMMEDIATE_SIM = 0.55       # between REACQ_THRESHOLD and this: REACQ_CONFIRM_FRAMES calls in the same place
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
                ("robot", "humanoid robot"))

TRACKING_GRACE_FRAMES = 30       # TRACKING -> RECOVERY after this many calls without the target
RECOVERY_TIMEOUT_SEC = 15.0      # RECOVERY -> WAITING_TRIGGER

KF_ACCEL_PX = 40000.0            # Kalman: white-noise acceleration, image (px^2/s^3) and depth (m^2/s^3)
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

KF_TRAIL_LEN = 15                # target Kalman positions drawn as a trail in the debug image
KF_FUTURE_SEC = 1.0              # debug image: where the Kalman expects the target in this many seconds

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
    """Index of the CLASS_GROUPS group of a YOLOE class (an unknown class is a group of its own)."""
    for i, group in enumerate(CLASS_GROUPS):
        if name in group:
            return i
    return name


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
        if sim_gallery >= GALLERY_STRONG_ADD_MIN_SIM:      # sure from the long-term memory itself
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
    kf: KalmanUVD
    misses: int = 0
    gate_P: Optional[np.ndarray] = None   # covariance of the gate at the last MEASUREMENT (frozen gate)
    unmeasured: int = 0                   # calls since the Kalman was last updated with a measurement
                                          # (a miss OR an occluded frame: seen, but nothing measured)

    @classmethod
    def from_detection(cls, d, t):
        return cls(d.box, d.dist, d.embedding, KalmanUVD(d.center[0], d.center[1], d.dist, t))

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
        self._at_prediction = []         # (A) depths of the people at the predicted point (prev. call)
        self._prev_other_boxes = []      # (A) (box, depth) of the previous call that were not the target
        self.trail = deque(maxlen=KF_TRAIL_LEN)   # (u, v, seen) of the target Kalman, last calls
        self._discarded_depths = []
        self._reset_pending()

    def lock(self, target_det, other_dets, t):
        """SEARCH: the chosen detection = TARGET, the other detections = people of this call."""
        self.reset()
        self.target = Person.from_detection(target_det, t)
        self.target_group = class_group(target_det.class_name)
        self.gallery.reset(target_det.embedding, target_det.box, target_det.dist, target_det.thumb)
        self.others = [Other(d.box, d.embedding) for d in other_dets if d.embedding is not None]

    def enter_recovery(self):
        self._reset_pending()

    def target_prediction(self, t):
        """(centre, std px) of the target prediction, None if older than PREDICTION_MAX_SEC."""
        if self.target is None or t - self.target.kf.t_update > PREDICTION_MAX_SEC:
            return None
        return self.target.kf.center, self.target.kf.pos_std

    # ------------------------------------------------------------------
    def step(self, dets, t, recovery=False):
        t0 = time.perf_counter()
        tgt = self.target
        if tgt.unmeasured > HOLD_MAX_MISSES:
            tgt.kf.x[3:6] = 0.0      # not measured for a while (lost, or occluded): the last known
                                     # position, no more extrapolation
        tgt.kf.predict_to(t)
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
            if GALLERY_VETO_MIN is not None and d.sim_gal < GALLERY_VETO_MIN:
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
                if d.sim_target < REACQ_THRESHOLD:
                    d.note = "far_low_similarity" if continuity else "low_similarity"
                    continue
                if not self._reacq_distance_ok(d):
                    d.note = "implausible_distance"
                    continue
            else:
                d.near_bonus = self._near_bonus_ok(d)
                if d.sim_target < (TARGET_LOW if d.near_bonus else REACQ_THRESHOLD):
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
        if d.sim_gal >= INTRUDER_RELEASE_SIM:
            return False
        c_other = min(self._box_cost(d.box, o.box) for o in self.others)
        c_target = self._box_cost(d.box, self._target_ref_box())
        return c_other <= OTHER_CONTINUITY_MAX and c_other < c_target - INTRUDER_STICKY_MARGIN

    def _class_ok(self, d):
        """YOLOE scores this box at least as much for the target's class group as for any other."""
        if getattr(self, "target_group", None) is None:
            return True
        scores = d.class_scores or {d.class_name: d.score}
        mine = max((s for c, s in scores.items() if class_group(c) == self.target_group), default=0.0)
        others = max((s for c, s in scores.items() if class_group(c) != self.target_group), default=0.0)
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
            res.status = CONFIRMED if d.sim_target >= TARGET_HIGH else KEPT
            self._reset_pending()
        elif d.sim_target >= REACQ_IMMEDIATE_SIM or self._confirm_pending(d.box):
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
            tgt.kf = KalmanUVD(d.center[0], d.center[1], tgt.kf.depth if d.depth_occluded else d.dist, t)
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
                other = d.sim_target < TARGET_LOW - AMBIGUITY_MARGIN or self._continues_other(d) or stays
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
        self.last_kf = {"post_u": float(kf.x[0]), "post_v": float(kf.x[1]), "post_d": float(kf.x[2])}
        self.trail.append((float(kf.x[0]), float(kf.x[1]), res.seen, float(kf.x[2]),
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
