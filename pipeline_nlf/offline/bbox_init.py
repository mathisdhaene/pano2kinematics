import numpy as np


def pick_largest_bbox_candidate(result):
    if not result or result.boxes is None:
        return None

    boxes_xywh = result.boxes.xywh.cpu().numpy()
    if len(boxes_xywh) == 0:
        return None

    track_ids = result.boxes.id.int().cpu().tolist() if result.boxes.id is not None else [-1] * len(boxes_xywh)

    best = None
    best_area = -1.0
    for idx, (cx, cy, w, h) in enumerate(boxes_xywh):
        area = float(max(w, 0.0) * max(h, 0.0))
        if area <= best_area:
            continue
        best_area = area
        best = {
            "index": idx,
            "track_id": int(track_ids[idx]),
            "bbox_xywh": np.asarray([cx, cy, w, h], dtype=np.float32),
            "area": area,
        }

    return best


def find_person_bbox(result, track_id):
    if not result or result.boxes is None:
        return None

    boxes_xywh = result.boxes.xywh.cpu().numpy()
    if len(boxes_xywh) == 0:
        return None

    track_ids = result.boxes.id.int().cpu().tolist() if result.boxes.id is not None else [-1] * len(boxes_xywh)

    for idx, bbox in enumerate(boxes_xywh):
        if int(track_ids[idx]) == int(track_id):
            return np.asarray(bbox, dtype=np.float32)

    return None


def align_equi_to_bbox(bbox_xywh, frame_width=3840, frame_height=1920):
    if bbox_xywh is None or len(bbox_xywh) < 4:
        return False, 0.0, 0.0

    cx, cy, _, _ = [float(v) for v in bbox_xywh[:4]]
    yaw_deg = (cx / float(frame_width)) * 360.0 - 180.0
    pitch_deg = -(cy / float(frame_height)) * 180.0 + 90.0
    print(f"[ALIGN] bbox cx={cx:.1f}, cy={cy:.1f}, yaw={yaw_deg:.1f}, pitch={pitch_deg:.1f}")
    return True, yaw_deg, pitch_deg


def compute_base_fov_from_bbox(
    bbox_xywh,
    frame_width,
    frame_height,
    safety_margin=1.15,
    fov_min=12.0,
    fov_max=75.0,
):
    if bbox_xywh is None or len(bbox_xywh) < 4:
        return None

    _, _, bbox_w, bbox_h = [float(v) for v in bbox_xywh[:4]]
    if bbox_w <= 0.0 or bbox_h <= 0.0:
        return None

    horiz_fov = 360.0 * (bbox_w / max(float(frame_width), 1.0))
    vert_fov = 180.0 * (bbox_h / max(float(frame_height), 1.0))
    base_fov = max(horiz_fov, vert_fov) * float(safety_margin)
    return float(np.clip(base_fov, fov_min, fov_max))
