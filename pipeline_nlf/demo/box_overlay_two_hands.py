import cv2
import numpy as np


BOX_EDGE_INDICES = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def _normalize(v, eps=1e-8):
    arr = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(arr))
    if not np.isfinite(n) or n < eps:
        return None
    return arr / n


def _project_on_plane(v, n):
    v = np.asarray(v, dtype=np.float32)
    n_unit = _normalize(n)
    if n_unit is None:
        return None
    return v - float(np.dot(v, n_unit)) * n_unit


def _pseudo_camera_up_prior(rotation_np):
    world_up_pre_remap = np.array([0.0, -1.0, 0.0], dtype=np.float32)
    return _normalize(world_up_pre_remap @ np.asarray(rotation_np, dtype=np.float32))


def _camera_points_to_pixels(points_cam_m, K, image_shape):
    pts = np.asarray(points_cam_m, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=bool)

    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    h, w = image_shape[:2]

    uv = np.full((len(pts), 2), np.nan, dtype=np.float32)
    valid = np.zeros((len(pts),), dtype=bool)
    for i, (x, y, z) in enumerate(pts):
        if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
            continue
        if z <= 1e-4:
            continue
        u = fx * (x / z) + cx
        v = fy * (y / z) + cy
        if np.isfinite(u) and np.isfinite(v):
            uv[i] = [u, v]
            valid[i] = (-0.5 <= u < w + 0.5) and (-0.5 <= v < h + 0.5)
    return uv, valid


def _project_vertices_to_equirectangular(vertices, image_width, image_height):
    pts = np.asarray(vertices, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        return np.empty((0, 2), dtype=np.float32)

    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    valid = np.isfinite(norms[:, 0]) & (norms[:, 0] > 1e-8)
    directions = np.full_like(pts, np.nan, dtype=np.float32)
    directions[valid] = pts[valid] / norms[valid]
    x = directions[:, 0]
    y = directions[:, 1]
    z = directions[:, 2]
    yaw = np.arctan2(x, z)
    pitch = np.arcsin(np.clip(y, -1.0, 1.0))
    u = (yaw + np.pi) / (2 * np.pi) * float(image_width)
    v = ((pitch + np.pi / 2.0) / np.pi) * float(image_height)
    return np.stack([u, v], axis=1)


def _make_box_corners(length_m, width_m, height_m):
    l2 = 0.5 * float(length_m)
    w2 = 0.5 * float(width_m)
    h = float(height_m)
    return np.asarray(
        [
            [-l2, -w2, 0.0],
            [+l2, -w2, 0.0],
            [+l2, +w2, 0.0],
            [-l2, +w2, 0.0],
            [-l2, -w2, h],
            [+l2, -w2, h],
            [+l2, +w2, h],
            [-l2, +w2, h],
        ],
        dtype=np.float32,
    )


def _wrap_equi_segment(p0, p1, width):
    p0 = np.asarray(p0, dtype=np.float32)
    p1 = np.asarray(p1, dtype=np.float32)
    width = float(width)
    if not (np.all(np.isfinite(p0)) and np.all(np.isfinite(p1))):
        return []

    du = float(p1[0] - p0[0])
    if abs(du) <= width * 0.5:
        return [(p0, p1)]

    if du > 0:
        p1_shift = p1.copy()
        p1_shift[0] -= width
        denom = p1_shift[0] - p0[0]
        if abs(float(denom)) < 1e-6:
            return []
        y_edge = p0[1] + (width - p0[0]) * (p1_shift[1] - p0[1]) / denom
        return [
            (p0, np.array([width - 1.0, y_edge], dtype=np.float32)),
            (np.array([0.0, y_edge], dtype=np.float32), p1),
        ]

    p1_shift = p1.copy()
    p1_shift[0] += width
    denom = p1_shift[0] - p0[0]
    if abs(float(denom)) < 1e-6:
        return []
    y_edge = p0[1] + (0.0 - p0[0]) * (p1_shift[1] - p0[1]) / denom
    return [
        (p0, np.array([0.0, y_edge], dtype=np.float32)),
        (np.array([width - 1.0, y_edge], dtype=np.float32), p1),
    ]


def _mean_valid_points(points):
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        return None
    valid = np.all(np.isfinite(pts), axis=1)
    if not np.any(valid):
        return None
    return np.mean(pts[valid], axis=0)


def _get_two_hand_centers(vertices3d_m, idx_left_mc5, idx_left_mc2, idx_right_mc5, idx_right_mc2):
    if max(idx_left_mc5, idx_left_mc2, idx_right_mc5, idx_right_mc2) >= len(vertices3d_m):
        return None

    left_center = _mean_valid_points([vertices3d_m[idx_left_mc5], vertices3d_m[idx_left_mc2]])
    right_center = _mean_valid_points([vertices3d_m[idx_right_mc5], vertices3d_m[idx_right_mc2]])
    if left_center is None or right_center is None:
        return None
    return left_center, right_center


def _estimate_box_pose_from_priors(
    vertices3d_mm,
    rotation_np,
    x_sign,
    *,
    idx_left_shoulder,
    idx_right_shoulder,
    idx_left_mc5,
    idx_left_mc2,
    idx_right_mc5,
    idx_right_mc2,
    box_center_local_offset,
):
    if vertices3d_mm is None:
        return None

    vertices3d_m = np.asarray(vertices3d_mm, dtype=np.float32) * 1e-3
    if max(idx_left_shoulder, idx_right_shoulder) >= len(vertices3d_m):
        return None

    p_ls = vertices3d_m[idx_left_shoulder]
    p_rs = vertices3d_m[idx_right_shoulder]
    if not (np.all(np.isfinite(p_ls)) and np.all(np.isfinite(p_rs))):
        return None

    hand_centers = _get_two_hand_centers(vertices3d_m, idx_left_mc5, idx_left_mc2, idx_right_mc5, idx_right_mc2)
    if hand_centers is None:
        return None
    p_left_hand, p_right_hand = hand_centers

    z_box = _pseudo_camera_up_prior(rotation_np)
    if z_box is None:
        return None

    shoulder_axis = p_rs - p_ls
    shoulder_floor = _project_on_plane(shoulder_axis, z_box)
    shoulder_floor = _normalize(shoulder_floor)
    if shoulder_floor is None:
        return None

    x_box = _normalize(float(x_sign) * np.cross(z_box, shoulder_floor))
    if x_box is None:
        return None
    y_box = _normalize(np.cross(z_box, x_box))
    if y_box is None:
        return None
    z_box = _normalize(np.cross(x_box, y_box))
    if z_box is None:
        return None

    rotation_box = np.column_stack((x_box, y_box, z_box)).astype(np.float32)
    hands_midpoint = 0.5 * (p_left_hand + p_right_hand)
    translation_box = hands_midpoint - (rotation_box @ box_center_local_offset)

    return {
        "R_box": rotation_box,
        "t_box": translation_box,
        "x_sign": int(x_sign),
    }


def _estimate_box_pose_hypotheses(vertices3d_mm, rotation_np, *, box_draw_both_x_hypotheses, box_long_axis_sign, **kwargs):
    x_signs = [box_long_axis_sign]
    if box_draw_both_x_hypotheses:
        x_signs = [+1, -1]

    poses = []
    for x_sign in x_signs:
        pose = _estimate_box_pose_from_priors(vertices3d_mm, rotation_np, x_sign, **kwargs)
        if pose is not None:
            poses.append(pose)
    return poses


def _draw_box_axes(image, origin_cam_m, R_box, K, axis_length_m, axis_colors):
    axes_local = np.asarray(
        [[0.0, 0.0, 0.0], [axis_length_m, 0.0, 0.0], [0.0, axis_length_m, 0.0], [0.0, 0.0, axis_length_m]],
        dtype=np.float32,
    )
    pts_cam = (R_box @ axes_local.T).T + origin_cam_m[None, :]
    uv, _ = _camera_points_to_pixels(pts_cam, K, image.shape)
    if len(uv) != 4 or not np.all(np.isfinite(uv[0])):
        return

    origin_px = tuple(np.round(uv[0]).astype(np.int32))
    cv2.circle(image, origin_px, 3, (255, 255, 255), -1)
    for axis_idx, axis_name in enumerate(("x", "y", "z"), start=1):
        if not np.all(np.isfinite(uv[axis_idx])):
            continue
        end_px = tuple(np.round(uv[axis_idx]).astype(np.int32))
        cv2.line(image, origin_px, end_px, axis_colors[axis_name], 2, cv2.LINE_AA)


def _draw_wireframe_box(image, R_box, t_box, K, color, box_l, box_w, box_h, thickness=2):
    corners_local = _make_box_corners(box_l, box_w, box_h)
    corners_cam = (R_box @ corners_local.T).T + t_box[None, :]
    uv, _ = _camera_points_to_pixels(corners_cam, K, image.shape)
    for i0, i1 in BOX_EDGE_INDICES:
        if i0 >= len(uv) or i1 >= len(uv):
            continue
        if not (np.all(np.isfinite(uv[i0])) and np.all(np.isfinite(uv[i1]))):
            continue
        p0 = tuple(np.round(uv[i0]).astype(np.int32))
        p1 = tuple(np.round(uv[i1]).astype(np.int32))
        cv2.line(image, p0, p1, color, thickness, cv2.LINE_AA)


def _draw_wireframe_box_equi(image, R_box, t_box, rotation_np, color, box_l, box_w, box_h, thickness=2):
    corners_local = _make_box_corners(box_l, box_w, box_h)
    corners_pseudo = (R_box @ corners_local.T).T + t_box[None, :]
    corners_equi = corners_pseudo @ np.asarray(rotation_np, dtype=np.float32).T
    uv = _project_vertices_to_equirectangular(corners_equi, image.shape[1], image.shape[0])
    for i0, i1 in BOX_EDGE_INDICES:
        if i0 >= len(uv) or i1 >= len(uv):
            continue
        for q0, q1 in _wrap_equi_segment(uv[i0], uv[i1], image.shape[1]):
            p0 = tuple(np.round(q0).astype(np.int32))
            p1 = tuple(np.round(q1).astype(np.int32))
            cv2.line(image, p0, p1, color, thickness, cv2.LINE_AA)


def _draw_box_axes_equi(image, origin_cam_m, R_box, rotation_np, axis_length_m, axis_colors):
    axes_local = np.asarray(
        [[0.0, 0.0, 0.0], [axis_length_m, 0.0, 0.0], [0.0, axis_length_m, 0.0], [0.0, 0.0, axis_length_m]],
        dtype=np.float32,
    )
    pts_pseudo = (R_box @ axes_local.T).T + origin_cam_m[None, :]
    pts_equi = pts_pseudo @ np.asarray(rotation_np, dtype=np.float32).T
    uv = _project_vertices_to_equirectangular(pts_equi, image.shape[1], image.shape[0])
    if len(uv) != 4 or not np.all(np.isfinite(uv[0])):
        return

    origin_px = tuple(np.round(uv[0]).astype(np.int32))
    cv2.circle(image, origin_px, 4, (255, 255, 255), -1)
    for axis_idx, axis_name in enumerate(("x", "y", "z"), start=1):
        if not np.all(np.isfinite(uv[axis_idx])):
            continue
        for q0, q1 in _wrap_equi_segment(uv[0], uv[axis_idx], image.shape[1]):
            p0 = tuple(np.round(q0).astype(np.int32))
            p1 = tuple(np.round(q1).astype(np.int32))
            cv2.line(image, p0, p1, axis_colors[axis_name], 2, cv2.LINE_AA)


def draw_box_debug_overlay(
    image,
    vertices3d_mm,
    kp_unc,
    K,
    rotation_np,
    *,
    box_debug_enabled,
    box_draw_both_x_hypotheses,
    box_long_axis_sign,
    box_l,
    box_w,
    box_h,
    box_axis_len,
    box_center_z_from_hands,
    box_center_local_offset,
    idx_left_shoulder,
    idx_right_shoulder,
    idx_left_mc5,
    idx_left_mc2,
    idx_right_mc5,
    idx_right_mc2,
    box_hypothesis_colors,
    box_axis_colors,
):
    if not box_debug_enabled:
        return

    poses = _estimate_box_pose_hypotheses(
        vertices3d_mm,
        rotation_np,
        box_draw_both_x_hypotheses=box_draw_both_x_hypotheses,
        box_long_axis_sign=box_long_axis_sign,
        idx_left_shoulder=idx_left_shoulder,
        idx_right_shoulder=idx_right_shoulder,
        idx_left_mc5=idx_left_mc5,
        idx_left_mc2=idx_left_mc2,
        idx_right_mc5=idx_right_mc5,
        idx_right_mc2=idx_right_mc2,
        box_center_local_offset=box_center_local_offset,
    )

    text_y = 20
    for pose in poses:
        colors = box_hypothesis_colors[int(pose["x_sign"])]
        _draw_wireframe_box(image, pose["R_box"], pose["t_box"], K, colors["wire"], box_l, box_w, box_h, thickness=2)
        _draw_box_axes(image, pose["t_box"], pose["R_box"], K, box_axis_len, box_axis_colors)

        origin_uv, _ = _camera_points_to_pixels(pose["t_box"][None, :], K, image.shape)
        if len(origin_uv) == 1 and np.all(np.isfinite(origin_uv[0])):
            origin_px = tuple(np.round(origin_uv[0]).astype(np.int32))
            cv2.circle(image, origin_px, 4, colors["origin"], -1)

        label = f"box2 x={pose['x_sign']:+d} hands=2 zoff={box_center_z_from_hands:.3f}m"
        cv2.putText(image, label, (8, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colors["text"], 1, cv2.LINE_AA)
        text_y += 18


def draw_box_debug_overlay_equi(
    image,
    vertices3d_mm,
    kp_unc,
    rotation_np,
    *,
    box_debug_enabled,
    box_draw_both_x_hypotheses,
    box_long_axis_sign,
    box_l,
    box_w,
    box_h,
    box_axis_len,
    box_center_local_offset,
    idx_left_shoulder,
    idx_right_shoulder,
    idx_left_mc5,
    idx_left_mc2,
    idx_right_mc5,
    idx_right_mc2,
    box_hypothesis_colors,
    box_axis_colors,
):
    if not box_debug_enabled:
        return

    poses = _estimate_box_pose_hypotheses(
        vertices3d_mm,
        rotation_np,
        box_draw_both_x_hypotheses=box_draw_both_x_hypotheses,
        box_long_axis_sign=box_long_axis_sign,
        idx_left_shoulder=idx_left_shoulder,
        idx_right_shoulder=idx_right_shoulder,
        idx_left_mc5=idx_left_mc5,
        idx_left_mc2=idx_left_mc2,
        idx_right_mc5=idx_right_mc5,
        idx_right_mc2=idx_right_mc2,
        box_center_local_offset=box_center_local_offset,
    )
    if not poses:
        return

    text_y = 20
    for pose in poses:
        colors = box_hypothesis_colors[int(pose["x_sign"])]
        _draw_wireframe_box_equi(image, pose["R_box"], pose["t_box"], rotation_np, colors["wire"], box_l, box_w, box_h, thickness=2)
        _draw_box_axes_equi(image, pose["t_box"], pose["R_box"], rotation_np, box_axis_len, box_axis_colors)
        label = f"box2-360 x={pose['x_sign']:+d}"
        cv2.putText(image, label, (8, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors["text"], 2, cv2.LINE_AA)
        text_y += 24
