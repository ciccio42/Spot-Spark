#!/usr/bin/env python3
"""
tracking_utils.py (demo_package) — everything of tracking_fsm.py that is NOT identity logic:
node configuration (topics, FOV, LEDs, botsort_hsv values), conversion of the detector response,
debug image, metrics, Spot LEDs.
"""

import datetime
import json
import os
import threading
import time
from collections import Counter

import cv2
from bosdyn.client.audio_visual import AudioVisualClient

from demo_package.common import embedding_from_msg, scale_box_to_depth, box_center_depth
from demo_package import tracking_embedding as emb

# ======================================================================
# NODE CONFIGURATION
# ======================================================================
INIT, WAITING_TRIGGER, SEARCH, TRACKING, RECOVERY = "init", "waiting_trigger", "search", "tracking", "recovery"
STATES = (INIT, WAITING_TRIGGER, SEARCH, TRACKING, RECOVERY)

HAND_RGB_TOPIC = 'camera/hand/compressed'
HAND_RGB_COMPRESSED = True        # True on bags, False on the real robot if the recompression node is missing
HAND_CAMERA_INFO_TOPIC = '/camera/hand/camera_info'
HAND_DEPTH_TOPIC = '/depth/hand/image'
TARGET_POSE_TOPIC = 'target_info'
DETECT_SERVICE = 'detect'
TARGET_CLASSES = ['person', 'quadruped', 'quadruped animal', 'quadruped robot', 'robotic dog',
                  'four-legged robot', 'dog', 'robot', 'humanoid robot']
DEBUG_IMAGE_TOPIC = '/person_follow/hand_debug/compressed'
DEBUG_PUBLISH_EVERY_N = 1         # debug image every N processed frames; 1 = always (camera ~3 Hz)
DEBUG_SHOW_KF = True              # Kalman of the target drawn in the debug image (see DebugCanvas.kalman)
FOV_SEARCH_DEG = 35.0
FOV_TRACKING_DEG = 60.0
METRICS_DIR = '/home/spot_ws/src/codice_carmine/metriche'
METRICS_SAVE_PERIOD_SEC = 5.0
SAVE_GALLERY_IMAGES = True        # metriche/gallery_<session>/: one image per view added + current strip
KF_TOPIC = '/person_follow/target_kf'   # PointStamped (x=u px, y=v px, z=depth m) of the target Kalman

# LEDs: for each state the NAME of an A/V behavior already on the robot (list_spot_led_behaviors.py)
STATI_LED = {
    INIT: None,
    WAITING_TRIGGER: None,
    SEARCH: "internal_autonomous_operation",     # pulsing white
    TRACKING: "internal_wait_for_entity",        # pulsing green
    RECOVERY: "internal_autonomous_navigation",  # blinking green
}
LED_DURATION_SEC = 5.0
LED_REFRESH_SEC = 2.0

# botsort_hsv ONLY (HSV-histogram correlations: a different scale from the ReID one)
BOTSORT = dict(
    REID_SIMILARITY_THRESHOLD=0.60, MIN_DETECTION_CONFIDENCE=0.10, REID_EMA_ALPHA=0.3,
    TARGET_DISTANCE_THRESHOLD_FRAC=0.30, STABILITY_FRAMES_REQUIRED=10, TRACKING_GRACE_FRAMES=30,
    RECOVERY_TIMEOUT_SEC=15.0, REACQUISITION_STABILITY_FRAMES=2, REACQUISITION_PX_TOLERANCE=60.0,
    REID_COMBINE=0.30, W_SIMILARITY=0.7, W_POSITION=0.3,
)


# ======================================================================
# Detector response -> detections
# ======================================================================
def dedupe(detections):
    """Same subject under two prompts ('person' and 'humanoid robot'): keep the best-scoring box
    among those overlapping more than DEDUPE_IOU (class-agnostic). Returns [(det, {class: score})]:
    every class given to that subject, so the tracker can tell a robot also called 'person'."""
    kept = []
    for det in sorted(detections, key=lambda d: d.score, reverse=True):
        box = (det.x1, det.y1, det.x2, det.y2)
        for k, scores in kept:
            if emb.iou(box, (k.x1, k.y1, k.x2, k.y2)) > emb.DEDUPE_IOU:
                scores[det.class_name] = max(scores.get(det.class_name, 0.0), det.score)
                break
        else:
            kept.append((det, {det.class_name: det.score}))
    return kept


def build_detections(response, frame_shape, crop_rect, depth_image, frame=None, fy=None):
    """Response of the Detect service -> list of tracking_embedding.Detection in full-image
    coordinates, with depth (None if missing or outside [TRACK_MIN_RANGE, TRACK_MAX_RANGE]).
    frame: if given, each detection carries its image crop (a view, no copy) for the gallery images.
    fy: focal length (px) of the RGB camera; if given, a depth implying a person taller than
        PERSON_MAX_HEIGHT_M (box height x depth / fy) is impossible -> treated as missing."""
    if response is None:
        return []
    cx, cy = crop_rect[0], crop_rect[1]
    out = []
    for det, class_scores in dedupe(response.detections):
        if det.score < emb.MIN_DETECTION_CONFIDENCE:
            continue
        box = (det.x1 + cx, det.y1 + cy, det.x2 + cx, det.y2 + cy)
        dist = None
        if depth_image is not None:
            dist = box_center_depth(depth_image, scale_box_to_depth(box, frame_shape, depth_image.shape))
            if dist is not None and not (emb.TRACK_MIN_RANGE <= dist <= emb.TRACK_MAX_RANGE):
                dist = None
            if dist is not None and fy and (box[3] - box[1]) * dist / fy > emb.PERSON_MAX_HEIGHT_M:
                dist = None
        d = emb.Detection(box, det.score, det.class_name, dist, embedding_from_msg(det.embedding),
                          class_scores=class_scores)
        if frame is not None and SAVE_GALLERY_IMAGES:
            x1, y1, x2, y2 = (max(0, int(v)) for v in box)
            d.thumb = frame[y1:y2, x1:x2]
        out.append(d)
    return out


# ======================================================================
# Debug image: ONE box per detection, labels that never overlap, a two-line panel
# ======================================================================
GREEN, YELLOW, ORANGE, RED, CYAN, WHITE = (0, 220, 0), (0, 220, 255), (0, 140, 255), \
    (0, 0, 255), (255, 255, 0), (255, 255, 255)   # BGR
# Why a detection was NOT taken as the target (shown on its box only when the target is not seen)
REASON_IT = {"no_detections": "nessuna detection", "no_embedding": "no embedding",
             "no_depth": "no profondita'", "outside_gate": "fuori gate KF",
             "low_similarity": "sim bassa", "far_low_similarity": "lontano e sim < riacq",
             "gallery_veto": "veto galleria",
             "looks_like_other": "simile a un'altra persona", "continues_other": "e' l'altra persona che passa", "assigned_to_other": "altra persona",
             "ambiguous": "ambiguo", "wrong_class": "classe diversa",
             "implausible_distance": "distanza non plausibile", "intruder": "intruso (era rosso)", "pending_confirmation": "in conferma"}
_FONT, _SCALE = cv2.FONT_HERSHEY_SIMPLEX, 0.5


class DebugCanvas:
    """Draws on one frame; remembers where the labels are so that they never overlap."""

    def __init__(self, frame):
        self.frame = frame
        self._labels = []

    def _free_spot(self, x, y, w, h):
        h_img, w_img = self.frame.shape[:2]
        x = min(max(0, x), max(0, w_img - w))
        y = max(h, y)
        for _ in range(20):
            rect = (x, y - h, x + w, y)
            if not any(rect[0] < r[2] and r[0] < rect[2] and rect[1] < r[3] and r[1] < rect[3] for r in self._labels):
                self._labels.append(rect)
                return x, y
            y += h + 2
        return x, y

    def box(self, box, color, label, thick=2):
        if self.frame is None or box is None:
            return
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(self.frame, (x1, y1), (x2, y2), color, thick)
        (tw, th), _ = cv2.getTextSize(label, _FONT, _SCALE, 1)
        x, y = self._free_spot(x1, y1 - 4, tw + 6, th + 8)
        cv2.rectangle(self.frame, (x, y - th - 8), (x + tw + 6, y), (0, 0, 0), -1)
        cv2.putText(self.frame, label, (x + 3, y - 4), _FONT, _SCALE, color, 1, cv2.LINE_AA)

    def panel(self, lines):
        """Top-left lines: "text" (white) or ("text", colour)."""
        if self.frame is None:
            return
        y = 8
        for line in lines:
            text, color = (line, WHITE) if isinstance(line, str) else line
            (tw, th), _ = cv2.getTextSize(text, _FONT, 0.55, 1)
            cv2.rectangle(self.frame, (4, y), (14 + tw, y + th + 10), (0, 0, 0), -1)
            cv2.putText(self.frame, text, (9, y + th + 4), _FONT, 0.55, color, 1, cv2.LINE_AA)
            self._labels.append((4, y, 14 + tw, y + th + 10))
            y += th + 12

    def prediction(self, center, std):
        if self.frame is None or center is None:
            return
        c = (int(center[0]), int(center[1]))
        cv2.circle(self.frame, c, 4, CYAN, -1)
        cv2.circle(self.frame, c, max(6, int(2.0 * std)), CYAN, 1)

    def kalman(self, tracker, res):
        """Target Kalman, nothing else:
             scia ciano      where the target has been (estimate of the last KF_TRAIL_LEN calls;
                             filled = seen, ring = only predicted)
             pallino ciano   where it is now
             freccia magenta where the Kalman expects it in KF_FUTURE_SEC (current velocity)"""
        tgt = tracker.target
        if self.frame is None or tgt is None or not tracker.trail:
            return
        MAG = (255, 0, 255)
        pts = list(tracker.trail)
        xy = [(int(p[0]), int(p[1])) for p in pts]
        for a, b in zip(xy, xy[1:]):
            cv2.line(self.frame, a, b, CYAN, 2, cv2.LINE_AA)
        for (u, v), p in zip(xy[:-1], pts[:-1]):
            cv2.circle(self.frame, (u, v), 3, CYAN, -1 if p[2] else 1, cv2.LINE_AA)
        now = xy[-1]
        cv2.circle(self.frame, now, 6, CYAN, -1, cv2.LINE_AA)
        fu = tgt.kf.x[0] + tgt.kf.x[3] * emb.KF_FUTURE_SEC
        fv = tgt.kf.x[1] + tgt.kf.x[4] * emb.KF_FUTURE_SEC
        fut = (int(fu), int(fv))
        if abs(fut[0] - now[0]) + abs(fut[1] - now[1]) >= 4:
            cv2.arrowedLine(self.frame, now, fut, MAG, 2, cv2.LINE_AA, tipLength=0.25)
        cv2.circle(self.frame, fut, 5, MAG, 2, cv2.LINE_AA)
        cv2.putText(self.frame, f"+{emb.KF_FUTURE_SEC:g}s", (fut[0] + 7, fut[1] - 7), _FONT, 0.45, MAG, 1, cv2.LINE_AA)

    def crop(self, crop_rect):
        if self.frame is not None and crop_rect is not None:
            cv2.rectangle(self.frame, tuple(crop_rect[:2]), tuple(crop_rect[2:]), (0, 200, 255), 1)


def _dist_text(d):
    return f"{d.dist:.1f}m" if d.dist is not None else "--m"


def draw_tracking(frame, header, dets, res, tracker, extra_line, now):
    """TRACKING / RECOVERY: one box per detection.
      green  TARGET OK         sim >= TARGET_HIGH (confirmed)
      yellow TARGET MANTENUTO  TARGET_LOW <= sim < TARGET_HIGH (kept: followed, but uncertain)
      yellow TARGET OCCLUSO / DEPTH SCARTATA  depth read on an occluder / the background (position only)
      orange candidate under re-acquisition confirmation
      red    every other person (when the target is not seen, + why it was not taken)"""
    if frame is None:
        return
    cv = DebugCanvas(frame)
    reason = REASON_IT.get(res.reason, res.reason)
    word = {emb.CONFIRMED: ("CONFERMATO", GREEN), emb.KEPT: ("MANTENUTO (incerto)", YELLOW),
            emb.PENDING: (f"CONFERMA {res.pending_count}/{emb.REACQ_CONFIRM_FRAMES}", ORANGE),
            emb.MISSED: (f"NON VISTO: {reason}", ORANGE)}[res.status]
    if res.occluded:
        why = "occluso" if res.occluder_visible else \
            f"depth incoerente {res.depth_resync}/{emb.DEPTH_RESYNC_FRAMES}"
        word = (f"SOLO POSIZIONE, {why}", YELLOW)
    cv.panel([f"{header} | viste {len(tracker.gallery)}", (f"target {word[0]}  {extra_line}", word[1])])
    if DEBUG_SHOW_KF:
        cv.kalman(tracker, res)
    for d in dets:
        if d is res.det and res.seen:
            kf_d = tracker.target.kf.depth
            tag = ("TARGET OCCLUSO" if res.occluder_visible else "TARGET DEPTH SCARTATA") if res.occluded else \
                "TARGET OK" if res.status == emb.CONFIRMED else "TARGET MANTENUTO"
            dist = f"~{kf_d:.1f}m (stima KF)" if res.occluded and kf_d is not None else _dist_text(d)
            cv.box(d.box, word[1], f"{tag} {dist} sim {d.sim_target:.2f} "
                                   f"(f {d.sim_ff:.2f} g {d.sim_gal:.2f})", 3)
        elif d is res.det:
            cv.box(d.box, ORANGE, f"candidato {_dist_text(d)} sim {d.sim_target:.2f}")
        else:
            why = f" - {REASON_IT.get(d.note, d.note)}" if (not res.seen and d.note) else ""
            cv.box(d.box, RED, f"{_dist_text(d)} sim {d.sim_target:.2f}{why}", 2)


def draw_search(frame, header, dets, candidate, stability):
    """SEARCH: one box per detection; the candidate (if exactly one in range) shows its stability."""
    if frame is None:
        return
    cv = DebugCanvas(frame)
    n_valid = 1 if candidate is not None else sum(1 for d in dets if d.dist is not None)
    second = (f"aggancio {stability}/{emb.STABILITY_FRAMES_REQUIRED}", GREEN) if candidate is not None \
        else (f"persone nel range: {n_valid} (serve esattamente 1)", ORANGE)
    cv.panel([header, second])
    for d in dets:
        if d is candidate:
            cv.box(d.box, GREEN, f"candidato {_dist_text(d)} {stability}/{emb.STABILITY_FRAMES_REQUIRED}", 3)
        else:
            cv.box(d.box, RED, f"persona {_dist_text(d)}", 2)


def draw_header_only(frame, lines):
    if frame is not None:
        DebugCanvas(frame).panel(lines)


# ======================================================================
# Metrics: ONE JSON per node run (rewritten every METRICS_SAVE_PERIOD_SEC, final at shutdown)
# ======================================================================
class Metrics:
    def __init__(self, method):
        self.method = method
        self.start_wall, self.start = time.time(), time.monotonic()
        self.total_frames = 0
        self.frames_by_state = {s: 0 for s in STATES}
        self.miss_reasons = Counter()
        self.detect_ms_sum = self.detect_ms_max = self.proc_ms_sum = 0.0
        self.detect_calls = 0
        self.duplicate_frames = 0          # camera frames identical to the previous one (skipped)
        self.extra = lambda: {}            # method-specific fields (set by the node)
        self._last_save, self._final_saved = self.start, False
        os.makedirs(METRICS_DIR, exist_ok=True)
        stamp = datetime.datetime.fromtimestamp(self.start_wall).strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(METRICS_DIR, f"tracking_metrics_{method}_{stamp}.json")
        self.gallery_dir = os.path.join(METRICS_DIR, f"gallery_{stamp}")

    def count_frame(self, state):
        self.total_frames += 1
        self.frames_by_state[state] = self.frames_by_state.get(state, 0) + 1
        if time.monotonic() - self._last_save >= METRICS_SAVE_PERIOD_SEC:
            self.save(final=False)
            self._last_save = time.monotonic()

    def count_miss(self, state, reason):
        self.miss_reasons[f"{state}:{reason}"] += 1

    def count_detect(self, ms):
        self.detect_calls += 1
        self.detect_ms_sum += ms
        self.detect_ms_max = max(self.detect_ms_max, ms)

    def save_gallery(self, gallery, added_kind):
        """Current gallery as one strip (gallery_now.jpg) + the new view as its own file."""
        if not SAVE_GALLERY_IMAGES or not gallery.thumbs or gallery.thumbs[-1] is None:
            return
        try:
            os.makedirs(self.gallery_dir, exist_ok=True)
            n = sum(1 for f in os.listdir(self.gallery_dir) if f != "gallery_now.jpg")
            t = round(time.monotonic() - self.start, 1)
            cv2.imwrite(os.path.join(self.gallery_dir, f"{n:03d}_{added_kind}_t{t}s.jpg"), gallery.thumbs[-1])
            tiles = []
            for i, th in enumerate(gallery.thumbs):
                if th is None or th.size == 0:
                    continue
                tile = cv2.resize(th, (int(160 * th.shape[1] / max(th.shape[0], 1)), 160))
                cv2.putText(tile, "anchor" if i == 0 else f"view {i}", (3, 14), _FONT, 0.45, GREEN, 1, cv2.LINE_AA)
                tiles.append(tile)
            if tiles:
                cv2.imwrite(os.path.join(self.gallery_dir, "gallery_now.jpg"), cv2.hconcat(tiles))
        except (OSError, cv2.error) as ex:
            print(f"[gallery] salvataggio fallito: {ex}")

    def snapshot(self, final):
        duration = time.monotonic() - self.start
        st = dict(self.frames_by_state)
        total, active = self.total_frames, st[SEARCH] + st[TRACKING] + st[RECOVERY]
        pct = lambda n, d: round(100.0 * n / d, 2) if d else 0.0
        data = {
            "tracking_method": self.method,
            "start_time": datetime.datetime.fromtimestamp(self.start_wall).isoformat(timespec="seconds"),
            "end_time": datetime.datetime.now().isoformat(timespec="seconds"),
            "final": final, "duration_s": round(duration, 2), "total_frames": total, "frames_by_state": st,
            "tracking_frames": st[TRACKING], "recovery_frames": st[RECOVERY], "active_frames": active,
            "tracking_pct_of_total": pct(st[TRACKING], total), "recovery_pct_of_total": pct(st[RECOVERY], total),
            "tracking_pct_of_active": pct(st[TRACKING], active), "recovery_pct_of_active": pct(st[RECOVERY], active),
            "effective_fps": round(total / duration, 2) if duration > 0 else 0.0,
            "duplicate_frames_skipped": self.duplicate_frames,
            "miss_reasons": dict(self.miss_reasons.most_common()),
            "avg_detect_roundtrip_ms": round(self.detect_ms_sum / self.detect_calls, 1) if self.detect_calls else 0.0,
            "max_detect_roundtrip_ms": round(self.detect_ms_max, 1),
            "avg_processing_ms": round(self.proc_ms_sum / self.detect_calls, 1) if self.detect_calls else 0.0,
        }
        data.update(self.extra())
        return data

    def save(self, final=False):
        if self._final_saved:
            return
        try:
            data = self.snapshot(final)
            with open(self.path + ".tmp", "w") as f:
                json.dump(data, f, indent=2)
            os.replace(self.path + ".tmp", self.path)
            if final:
                self._final_saved = True
                print(f"[METRICHE] salvate: {self.path}")
        except Exception as ex:
            print(f"[METRICHE] salvataggio fallito: {ex}")


# ======================================================================
# Spot LEDs: a dedicated thread does every gRPC call; the FSM only records the desired state
# ======================================================================
class SpotLeds:
    def __init__(self, robot, logger):
        self._robot, self._log = robot, logger
        self.client = None
        if robot is not None:
            try:
                self.client = robot.ensure_client(AudioVisualClient.default_service_name)
            except Exception as ex:
                logger.error(f"Impossibile creare AudioVisualClient: {ex} — LED disattivati.")
        if self.client is not None and all(v is None for v in STATI_LED.values()):
            logger.warn("[LED] nessun behavior associato agli stati — LED disattivati.")
            self.client = None
        self._lock, self._wake, self._stop = threading.Lock(), threading.Event(), threading.Event()
        self._desired = self._applied = None
        self._thread = None
        if self.client is not None:
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()

    @property
    def active(self):
        return self.client is not None

    def request(self, state):
        if self.client is None:
            return
        with self._lock:
            if state == self._desired:
                return
            self._desired = state
        self._wake.set()

    def _worker(self):
        try:
            self._robot.time_sync.wait_for_sync(timeout_sec=10.0)
            available = {live.name: live for live in self.client.list_behaviors()}
        except Exception as ex:
            self._log.error(f"[LED] time sync / list_behaviors fallita: {ex} — LED disattivati.")
            return
        for state, name in STATI_LED.items():
            if name is not None and name not in available:
                self._log.error(f"[LED] behavior {name!r} ({state}) NON presente sul robot: {sorted(available)}")
            elif name is not None and available[name].behavior.audio_sequence_group.ListFields():
                self._log.warn(f"[LED] behavior {name!r} ({state}) contiene AUDIO: suonera' il buzzer.")
        last_run = 0.0
        while not self._stop.is_set():
            self._wake.wait(timeout=LED_REFRESH_SEC)
            self._wake.clear()
            if self._stop.is_set():
                break
            with self._lock:
                desired = self._desired
            name = STATI_LED.get(desired)
            name = name if name in available else None
            now = time.time()
            try:
                if name != self._applied:
                    if self._applied is not None:
                        self.client.stop_behavior(self._applied)
                    if name is not None:
                        self.client.run_behavior(name, now + LED_DURATION_SEC, restart=True)
                        last_run = now
                    self._applied = name
                elif name is not None and now - last_run >= LED_REFRESH_SEC:
                    self.client.run_behavior(name, now + LED_DURATION_SEC, restart=False)
                    last_run = now
            except Exception as ex:
                self._log.error(f"[LED] aggiornamento fallito: {ex}", throttle_duration_sec=5.0)

    def shutdown(self):
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self.client is not None and self._applied is not None:
            try:
                self.client.stop_behavior(self._applied)
            except Exception:
                pass
