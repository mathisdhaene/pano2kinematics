import numpy as np


def estimate_camera_offset_from_vertices3d(vertices3d):
    """Estimate yaw/pitch offsets in degrees from camera-frame 3D vertices."""
    vertices = np.asarray(vertices3d, dtype=np.float32)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        return 0.0, 0.0

    center = np.nanmean(vertices, axis=0)
    if not np.all(np.isfinite(center)):
        return 0.0, 0.0

    x, y, z = center
    norm = float(np.linalg.norm(center))
    if norm < 1e-6 or not np.isfinite(norm):
        return 0.0, 0.0

    yaw = np.arctan2(x, z)
    pitch = np.arcsin(np.clip(y / norm, -1.0, 1.0))
    return float(np.degrees(yaw)), float(np.degrees(pitch))


def estimate_camera_offset_from_pose2d(pose2d, img_w=256, img_h=256):
    """Estimate yaw/pitch offsets from torso-centered 2D keypoints."""
    if pose2d is None or len(pose2d.shape) != 2 or pose2d.shape[1] != 2:
        print("[TRACK-2D] pose2d invalid shape:", None if pose2d is None else pose2d.shape)
        return 0.0, 0.0

    torso_indices = [5, 6, 11, 12]

    try:
        valid_indices = [idx for idx in torso_indices if idx < pose2d.shape[0]]
        torso_pts = pose2d[valid_indices]
        torso_pts = torso_pts[np.all(np.isfinite(torso_pts), axis=1)]
        if len(torso_pts) == 0:
            raise ValueError("No valid torso keypoints")
        center = torso_pts.mean(axis=0)
    except Exception:
        pts = pose2d[np.all(np.isfinite(pose2d), axis=1)]
        if len(pts) == 0:
            print("[TRACK-2D] No valid 2D pts, returning 0.")
            return 0.0, 0.0
        center = pts.mean(axis=0)

    center = np.asarray(center).reshape(2,)
    cx, cy = float(center[0]), float(center[1])
    dx = (cx - img_w / 2) / (img_w / 2)
    dy = (cy - img_h / 2) / (img_h / 2)
    return dx * 10.0, -dy * 10.0


def estimate_camera_offset_from_shoulder_2d(
    pose2d,
    idx_shoulder,
    img_w=256,
    img_h=256,
    gain_yaw=10.0,
    gain_pitch=10.0,
):
    """Estimate yaw/pitch offsets from a chosen shoulder keypoint."""
    if pose2d is None or pose2d.ndim != 2 or pose2d.shape[1] != 2:
        return 0.0, 0.0

    try:
        x, y = pose2d[idx_shoulder]
        if not np.isfinite(x) or not np.isfinite(y):
            return 0.0, 0.0
    except Exception:
        return 0.0, 0.0

    dx = (x - img_w / 2) / (img_w / 2)
    dy = (y - img_h / 2) / (img_h / 2)
    return dx * gain_yaw, -dy * gain_pitch


# Backward-compatible aliases
compute_yaw_pitch_from_vertices = estimate_camera_offset_from_vertices3d
compute_yaw_pitch_from_2d = estimate_camera_offset_from_pose2d
compute_yaw_pitch_from_shoulder_2d = estimate_camera_offset_from_shoulder_2d
