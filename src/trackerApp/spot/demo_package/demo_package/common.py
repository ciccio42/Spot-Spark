#!/usr/bin/env python3
"""
common.py (demo_package)

Geometry, depth-reading and lightweight ReID functions used by
tracking_fsm.py (single-camera design: arm only). No dependency on rclpy
NOR on ultralytics/YOLOE: the actual detection goes through the Detect
service (see detect_client.py) exposed by the DetectorNode in the yolo
container — only the "pure" functions that do not depend on how the
detection is done remain here.
"""

import math

import cv2
import numpy as np


# ============================================================
# ROI — SINGLE source of truth. Change the values here; they automatically
# apply both to the crop (compute_roi_crop_rect) and to the post-detection
# distance check in tracking_fsm.py.
# ============================================================
CONE_FOV_DEG = 35.0        # TOTAL ROI aperture, in degrees — crop width (via intrinsics)
CONE_MIN_RANGE = 1.5       # minimum distance, in metres (POST-detection check, via depth)
CONE_MAX_RANGE = 3.5       # maximum distance, in metres (POST-detection check, via depth)
CROP_TOP_MARGIN_FRAC = 0   # fraction of the image height cut from the TOP in the crop
CROP_BOTTOM_MARGIN_FRAC = 0  # fraction of the image height cut from the BOTTOM in the crop
                                  # No physical meaning ("height above ground") — purely
                                  # pixels: the target only has to fall INSIDE the crop, the real
                                  # distance is checked AFTERWARDS on the box depth (box_center_depth),
                                  # not on the crop itself. No TF involved.


# ============================================================
# 2D/3D geometry
# ============================================================

def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def distance(p1, p2):
    """Euclidean distance; works for both 2D and 3D points (tuples of the
    same length)."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


def deproject_pixel_to_point(u, v, depth, fx, fy, cx, cy):
    """Inverse pinhole model: pixel (u, v) + depth (metres) -> 3D point in
    the camera OPTICAL frame (x right, y down, z forward — standard
    OpenCV/ROS convention)."""
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth
    return x, y, z


def project_point_to_pixel(x, y, z, fx, fy, cx, cy):
    """Forward pinhole projection: 3D point in the camera optical frame ->
    pixel (u, v). None if the point is behind the camera (z <= 0)."""
    if z <= 1e-6:
        return None
    u = fx * x / z + cx
    v = fy * y / z + cy
    return u, v


class CameraIntrinsics:
    """Extracts fx, fy, cx, cy from the K matrix (row-major 3x3) of a
    sensor_msgs/CameraInfo message. In ROS 2 the field is called 'k' (lowercase)."""

    __slots__ = ("fx", "fy", "cx", "cy")

    def __init__(self, camera_info_msg):
        k = camera_info_msg.k
        self.fx = k[0]
        self.fy = k[4]
        self.cx = k[2]
        self.cy = k[5]


# ============================================================
# Depth
# ============================================================

def box_center_depth(depth_image, box, patch_frac=0.2, min_patch_px=3, min_valid_pixels=3):
    """Depth (median, in metres) over a SMALL patch centred on the box —
    not the whole box, not a single pixel.

    Why not the whole box: if the box is not tight around the subject
    (common with YOLOE), the background pixels at the edges contaminate the
    median, pulling the computed distance towards the background.

    Why not a single pixel: depth sensors often have holes at isolated
    points (noise, reflections) — one unlucky pixel would give None even
    with the subject perfectly visible there.

    `patch_frac`: fraction of the box width/height to sample, centred.
    With patch_frac=0.2 on a 100x200px box, the patch is 20x40px."""
    h, w = depth_image.shape[:2]
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    box_w, box_h = (x2 - x1), (y2 - y1)
    patch_w = max(box_w * patch_frac, min_patch_px)
    patch_h = max(box_h * patch_frac, min_patch_px)

    px1 = max(0, int(cx - patch_w / 2))
    px2 = min(w, int(cx + patch_w / 2))
    py1 = max(0, int(cy - patch_h / 2))
    py2 = min(h, int(cy + patch_h / 2))
    if px2 <= px1 or py2 <= py1:
        return None

    crop = depth_image[py1:py2, px1:px2].astype(np.float32)
    if depth_image.dtype == np.uint16:
        crop = crop / 1000.0

    valid = crop[(crop > 0.05) & np.isfinite(crop)]
    print(f"Min valid value: {valid.min() if valid.size > 0 else 'N/A'}, Max valid value: {valid.max() if valid.size > 0 else 'N/A'}, Valid pixel count: {valid.size}")
    if valid.size < min_valid_pixels:
        return None
    return float(np.median(valid))


def scale_box_to_depth(box, rgb_shape, depth_shape):
    """Rescales the coordinates of a box (in pixels of the RGB image the
    detection ran on) to the resolution of the depth image, if the two have
    different sizes — common with ToF sensors, often at a much lower
    resolution than the RGB camera. If the two resolutions match, returns
    the box unchanged."""
    rgb_h, rgb_w = rgb_shape[:2]
    depth_h, depth_w = depth_shape[:2]
    if (rgb_w, rgb_h) == (depth_w, depth_h):
        return box
    scale_x = depth_w / rgb_w
    scale_y = depth_h / rgb_h
    x1, y1, x2, y2 = box
    return (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)


# ============================================================
# Appearance (lightweight ReID) — same logic as yoloe_offline_reid.py
# ============================================================

def extract_appearance_embedding(frame_bgr, box, hist_bins=(8, 8, 8)):
    frame_h, frame_w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame_w, x2), min(frame_h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame_bgr[y1:y2, x1:x2]
    hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_crop], [0, 1, 2], None, list(hist_bins), [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, None, alpha=1.0, norm_type=cv2.NORM_L1).flatten()
    return hist.astype(np.float32)


def embedding_similarity(e1, e2):
    """Similarity in [-1, 1] (1 = identical). 0.0 if an embedding is missing.
    For the HSV histogram of extract_appearance_embedding — see
    neural_embedding_similarity for the new neural embedding (from
    detector_interfaces/BoxDetection.embedding, field filled on the yolo side)."""
    if e1 is None or e2 is None:
        return 0.0
    return float(cv2.compareHist(e1, e2, cv2.HISTCMP_CORREL))


def embedding_from_msg(embedding_field):
    """Converts the ROS field BoxDetection.embedding (float32[], empty if the
    DetectorNode had no embedding model configured) into a numpy array, or
    None if it is empty — so the caller can treat 'empty array from the
    message' and 'no embedding' with the same check (`is None`), without
    checking the length every time."""
    if not embedding_field:
        return None
    return np.array(embedding_field, dtype=np.float32)


def neural_embedding_similarity(e1, e2):
    """Cosine similarity in [-1, 1] (1 = identical) between two NEURAL
    embeddings (from a separate classification model, penultimate layer —
    not the HSV histogram, which uses embedding_similarity/HISTCMP_CORREL).
    0.0 if an embedding is missing."""
    if e1 is None or e2 is None:
        return 0.0
    n1 = np.linalg.norm(e1)
    n2 = np.linalg.norm(e2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    return float(np.dot(e1, e2)/(n1 * n2))

def rich_neural_embedding_similarity(e1, e2, w_cosine=1.0, w_euclidean=1.0 , w_magnitude=1.0, euclidean_scale=10.0 , magnitude_scale=10.0):
    """Combines cosine similarity and Euclidean distance on the raw vectors (sensitive to magnitude too, not only direction) into a single similarity score. Returns 0.0 if an embedding is missing."""
    
    if e1 is None or e2 is None:
        return 0.0
    
    n1 = np.linalg.norm(e1)
    n2 = np.linalg.norm(e2)
    
    cosine = float(np.dot(e1, e2) / (n1 * n2)) if n1 > 1e-8 and n2 > 1e-8 else 0.0
    euclidean_dist = np.linalg.norm(e1 - e2)
    euclidean_sim = 1.0 / (1.0 + euclidean_dist / euclidean_scale)  # Normalise the Euclidean distance into a similarity score between 0 and 1
    magnitude_diff = abs(n1 - n2)
    magnitude_sim = 1.0 / (1.0 + magnitude_diff / magnitude_scale)  # Normalise the magnitude difference into a similarity score
    total_weight = w_cosine + w_euclidean + w_magnitude
    
    print(f"Cosine: {cosine:.4f}, Euclidean Sim: {euclidean_sim:.4f}, Magnitude Sim: {magnitude_sim:.4f}, Total Weight: {total_weight:.4f}")
    
    
    return (w_cosine * cosine + w_euclidean * euclidean_sim + w_magnitude * magnitude_sim) / total_weight 


# ============================================================
# ROI as an image crop (single-camera design: arm)
# ============================================================

def compute_roi_crop_rect(image_width, image_height, intrinsics, fov_rad,
                           top_margin_frac=CROP_TOP_MARGIN_FRAC, bottom_margin_frac=CROP_BOTTOM_MARGIN_FRAC):
    """Computes the crop rectangle (x1, y1, x2, y2) in pixels. NO TF
    involved, on either axis:

    - WIDTH: from the camera intrinsics (fx, cx) and the FOV, as before —
      half_width_px = fx * tan(fov_rad / 2).

    - HEIGHT: FIXED fraction of the image, cut from the top and the
      bottom. No physical meaning ("height above ground") — the target
      only has to fall INSIDE the crop; the real distance/range is checked
      AFTERWARDS, on the depth of the detected box (box_center_depth in
      tracking_fsm.py), not on the crop itself.

    Always computable once the intrinsics are known (unlike the previous
    version, it does not depend on the arm pose at that instant) —
    returns None only if the configured margins are degenerate (e.g.
    overlapping margins)."""
    half_width_px = intrinsics.fx * math.tan(fov_rad / 2.0)
    x1 = max(0, int(intrinsics.cx - half_width_px))
    x2 = min(image_width, int(intrinsics.cx + half_width_px))

    y1 = max(0, int(image_height * top_margin_frac))
    y2 = min(image_height, int(image_height * (1.0 - bottom_margin_frac)))

    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)
