import math

import numpy as np


def estimate_hand_target_from_nlf(
    pose2d,
    primary_idx,
    secondary_idx,
    img_w,
    img_h,
):
    """Estimate hand target directly from NLF hand keypoints in pseudo image."""
    if pose2d is None:
        return None, None, None, None

    pts = np.asarray(pose2d, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 2:
        return None, None, None, None

    p0 = None
    p1 = None
    if 0 <= primary_idx < len(pts):
        cand = pts[primary_idx, :2]
        if np.all(np.isfinite(cand)):
            p0 = cand
    if 0 <= secondary_idx < len(pts):
        cand = pts[secondary_idx, :2]
        if np.all(np.isfinite(cand)):
            p1 = cand

    if p0 is None and p1 is None:
        return None, None, None, None
    if p0 is None:
        target = p1.copy()
    elif p1 is None:
        target = p0.copy()
    else:
        target = 0.5 * (p0 + p1)

    target[0] = float(np.clip(target[0], 0.0, float(img_w - 1)))
    target[1] = float(np.clip(target[1], 0.0, float(img_h - 1)))
    span = float(np.linalg.norm(p1 - p0)) if (p0 is not None and p1 is not None) else None
    return target, p0, p1, span


def image_point_to_yaw_pitch_delta(target_xy, img_w, img_h, fov_x_deg):
    """Convert image offset in pseudo view into yaw/pitch deltas in degrees."""
    if target_xy is None:
        return 0.0, 0.0

    x = float(target_xy[0])
    y = float(target_xy[1])
    cx = 0.5 * (float(img_w) - 1.0)
    cy = 0.5 * (float(img_h) - 1.0)
    hx = max(cx, 1.0)
    hy = max(cy, 1.0)
    nx = (x - cx) / hx
    ny = (y - cy) / hy

    half_fov = math.radians(max(float(fov_x_deg), 1e-3) * 0.5)
    tan_half = math.tan(half_fov)
    delta_yaw = math.degrees(math.atan(nx * tan_half))
    delta_pitch = -math.degrees(math.atan(ny * tan_half))
    return float(delta_yaw), float(delta_pitch)


def pick_mediapipe_hand(mp_result):
    """Prefer the highest-score MediaPipe hand, tie-breaking by image spread."""
    if mp_result is None or not getattr(mp_result, "hand_landmarks", None):
        return None
    n = len(mp_result.hand_landmarks)
    if n == 0:
        return None

    best_i = None
    best_score = -1.0
    best_area = -1.0
    for i in range(n):
        score = 0.0
        if getattr(mp_result, "handedness", None) and i < len(mp_result.handedness):
            hs = mp_result.handedness[i]
            if hs and len(hs) > 0 and hasattr(hs[0], "score"):
                score = float(hs[0].score)

        lms = mp_result.hand_landmarks[i]
        xs = [lm.x for lm in lms if np.isfinite(lm.x)]
        ys = [lm.y for lm in lms if np.isfinite(lm.y)]
        area = 0.0
        if xs and ys:
            area = float((max(xs) - min(xs)) * (max(ys) - min(ys)))

        if score > best_score or (abs(score - best_score) < 1e-6 and area > best_area):
            best_score = score
            best_area = area
            best_i = i
    return best_i


def landmark_to_pixel(lm, width, height):
    x = float(np.clip(lm.x * width, 0.0, float(width - 1)))
    y = float(np.clip(lm.y * height, 0.0, float(height - 1)))
    return np.array([x, y], dtype=np.float32)


def backproject_pixel_with_depth(uv, z, K):
    """Backproject image pixel uv to camera coordinates using fixed depth z."""
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    x = (float(uv[0]) - cx) * float(z) / max(fx, 1e-8)
    y = (float(uv[1]) - cy) * float(z) / max(fy, 1e-8)
    return np.array([x, y, float(z)], dtype=np.float32)
