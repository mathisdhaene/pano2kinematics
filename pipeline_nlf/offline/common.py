import math

import numpy as np


def compute_torso_depth(vertices3d, torso_indices, min_valid=2):
    if vertices3d is None:
        return None

    zs = []
    for idx in torso_indices:
        if idx < 0 or idx >= len(vertices3d):
            continue
        pt = vertices3d[idx]
        if not np.all(np.isfinite(pt)):
            continue
        z = float(pt[2]) * 1e-3
        if 0.1 < z < 10.0:
            zs.append(z)

    if len(zs) < min_valid:
        return None

    return float(np.median(zs))


def update_fov_from_depth_nlf(
    depth,
    ref_depth,
    ref_fov,
    current_fov,
    frame_idx,
    fov_min=12.0,
    fov_max=75.0,
    gain_far=6.0,
    gain_close=0.8,
    exponent_far=2.2,
    exponent_close=0.7,
    deadband=0.02,
    lerp=0.20,
    init_boost_frames=50,
    init_boost_gain=3.0,
):
    if depth is None or not np.isfinite(depth):
        return current_fov

    rel = depth / ref_depth

    if frame_idx < init_boost_frames:
        deadband = 0.0
        boost = init_boost_gain
    else:
        boost = 1.0

    if abs(rel - 1.0) < deadband:
        return current_fov

    if rel > 1.0:
        effect = (rel - 1.0) ** exponent_far
        target = ref_fov / (1.0 + gain_far * effect * boost)
    else:
        effect = (1.0 - rel) ** exponent_close
        target = ref_fov * (1.0 + gain_close * effect * boost)

    target = float(np.clip(target, fov_min, fov_max))
    return float(current_fov + lerp * (target - current_fov))


def update_fov_from_pose2d_framing(
    pose2d,
    current_fov,
    frame_w,
    frame_h,
    fov_max=75.0,
    trigger_extent=0.82,
    target_extent=0.72,
    lerp=0.35,
):
    if pose2d is None:
        return current_fov

    pts = np.asarray(pose2d, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 2:
        return current_fov

    valid = np.all(np.isfinite(pts[:, :2]), axis=1)
    pts = pts[valid, :2]
    if len(pts) < 3:
        return current_fov

    cx = 0.5 * (float(frame_w) - 1.0)
    cy = 0.5 * (float(frame_h) - 1.0)
    hx = max(cx, 1.0)
    hy = max(cy, 1.0)

    u = (pts[:, 0] - cx) / hx
    v = (pts[:, 1] - cy) / hy
    extent = float(max(np.max(np.abs(u)), np.max(np.abs(v))))

    if extent <= trigger_extent:
        return current_fov

    scale = extent / max(target_extent, 1e-6)
    half = math.radians(max(current_fov, 1e-3) * 0.5)
    target = math.degrees(2.0 * math.atan(math.tan(half) * scale))
    target = float(np.clip(target, current_fov, fov_max))
    return float(current_fov + lerp * (target - current_fov))


def compute_fov_from_markers(p_left, p_right, safety_margin=1.05, fov_min=12.0, fov_max=75.0):
    dx = abs(p_left[0] - p_right[0])
    z = 0.5 * (p_left[2] + p_right[2])

    if z <= 0 or dx <= 0:
        return None

    fov_rad = 2.0 * math.atan(dx / (2.0 * z))
    fov_deg = math.degrees(fov_rad) * safety_margin
    return float(np.clip(fov_deg, fov_min, fov_max))


def pick_best_nlf_candidate(poses2d_arr, poses3d_arr, unc_arr, torso_indices=(5, 6, 11, 12)):
    n = min(len(poses2d_arr), len(poses3d_arr), len(unc_arr))
    if n == 0:
        return None

    best_idx = None
    best_unc = float("inf")
    for i in range(n):
        kp_unc = np.asarray(unc_arr[i], dtype=np.float32)
        valid_idx = [j for j in torso_indices if j < len(kp_unc) and np.isfinite(kp_unc[j])]
        score = (
            float(np.mean(kp_unc[valid_idx]))
            if valid_idx
            else float(np.mean(kp_unc[np.isfinite(kp_unc)]))
            if np.any(np.isfinite(kp_unc))
            else float("inf")
        )
        if score < best_unc:
            best_unc = score
            best_idx = i

    if best_idx is None:
        return None

    return (
        np.asarray(poses2d_arr[best_idx], dtype=np.float32),
        np.asarray(poses3d_arr[best_idx], dtype=np.float32),
        np.asarray(unc_arr[best_idx], dtype=np.float32),
    )


def evaluate_nlf_track_quality(
    pose2d,
    kp_unc,
    prev_center=None,
    img_w=384,
    img_h=384,
    unc_threshold=0.5,
    use_jump_check=False,
    max_jump_ratio=0.75,
    jump_only_if_unc_above=0.35,
    torso_indices=(5, 6, 11, 12),
):
    if pose2d is None or kp_unc is None:
        return False, float("inf"), None, "missing"

    pts = np.asarray(pose2d, dtype=np.float32)
    unc = np.asarray(kp_unc, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 2 or unc.ndim != 1:
        return False, float("inf"), None, "shape"

    valid_torso = [i for i in torso_indices if i < len(pts) and i < len(unc)]
    if len(valid_torso) < 2:
        return False, float("inf"), None, "torso_idx"

    torso_pts = pts[valid_torso, :2]
    torso_unc = unc[valid_torso]
    good_mask = np.all(np.isfinite(torso_pts), axis=1) & np.isfinite(torso_unc)
    torso_pts = torso_pts[good_mask]
    torso_unc = torso_unc[good_mask]
    if len(torso_pts) < 2:
        return False, float("inf"), None, "torso_nan"

    center = torso_pts.mean(axis=0)
    torso_unc_mean = float(np.mean(torso_unc))

    span_x = float(np.max(torso_pts[:, 0]) - np.min(torso_pts[:, 0]))
    span_y = float(np.max(torso_pts[:, 1]) - np.min(torso_pts[:, 1]))
    min_span = max(6.0, 0.015 * min(img_w, img_h))
    if span_x < min_span and span_y < min_span:
        return False, torso_unc_mean, center, "torso_small"

    if torso_unc_mean >= unc_threshold:
        return False, torso_unc_mean, center, "unc"

    if use_jump_check and prev_center is not None and torso_unc_mean >= float(jump_only_if_unc_above):
        jump = float(np.linalg.norm(center - np.asarray(prev_center, dtype=np.float32)))
        max_jump = float(max_jump_ratio) * min(img_w, img_h)
        if jump > max_jump:
            return False, torso_unc_mean, center, "jump"

    return True, torso_unc_mean, center, "ok"
