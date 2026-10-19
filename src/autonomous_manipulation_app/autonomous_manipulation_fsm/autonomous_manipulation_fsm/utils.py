STATE_ENUMERATION = {
    "IDLE": 0,
    "NAVIGATE_TO_PICK": 1,
    "PICK": 2,
    "NAVIGATE_TO_PLACE": 3,
    "PLACE": 4,
    "RETURN_TO_IDLE": 5,
    "ERROR": 6,
}

import math

import numpy as np


def bbox_depth(depth_m, x1, y1, x2, y2, shrink=0.5):
    """Median of the valid depth values (meters) inside the central part of the bbox, None if there are none.

    shrink=0.5 keeps the central 50% of the box, so the background around the ball is ignored.
    """
    h, w = depth_m.shape[:2]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    hw, hh = (x2 - x1) * shrink / 2.0, (y2 - y1) * shrink / 2.0
    u0, u1 = int(max(0, cx - hw)), int(min(w, cx + hw + 1))
    v0, v1 = int(max(0, cy - hh)), int(min(h, cy + hh + 1))
    roi = depth_m[v0:v1, u0:u1]
    valid = roi[np.isfinite(roi) & (roi > 0.0)]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def depth_around(depth_m, u, v, radius):
    """Median of the valid depth values (meters) in the square of half size radius (depth pixels)
    centered on (u, v), and the number of valid pixels; (None, 0) if there are none."""
    h, w = depth_m.shape[:2]
    u0, u1 = int(max(0, u - radius)), int(min(w, u + radius + 1))
    v0, v1 = int(max(0, v - radius)), int(min(h, v + radius + 1))
    roi = depth_m[v0:v1, u0:u1]
    valid = roi[np.isfinite(roi) & (roi > 0.0)]
    if valid.size == 0:
        return None, 0
    return float(np.median(valid)), int(valid.size)


def deproject(u, v, depth, fx, fy, cx, cy):
    """Pixel (u, v) + depth (m) -> 3D point in the camera optical frame (x right, y down, z forward)."""
    return ((u - cx) * depth / fx, (v - cy) * depth / fy, depth)


def standoff_goal(robot_xy, target_xy, standoff):
    """Point at `standoff` meters from the target on the robot->target line, facing the target.

    Returns (x, y, yaw, distance robot->target).
    """
    dx, dy = target_xy[0] - robot_xy[0], target_xy[1] - robot_xy[1]
    d = math.hypot(dx, dy)
    yaw = math.atan2(dy, dx)
    if d <= standoff:
        return robot_xy[0], robot_xy[1], yaw, d
    return target_xy[0] - standoff * dx / d, target_xy[1] - standoff * dy / d, yaw, d


def quat_from_yaw(yaw):
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))
