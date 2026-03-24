import numpy as np


MARGIN_PERCENTAGE = 0.01
CENTER_BAND_RATIO = 0.10


def calculate_angle(v1, v2):
    dot_product = np.dot(v1, v2)
    magnitude = np.linalg.norm(v1) * np.linalg.norm(v2)
    cosine_angle = dot_product / magnitude if magnitude != 0 else 0
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))


def is_t_pose(keypoints):
    left_shoulder, right_shoulder = keypoints[5], keypoints[6]
    left_elbow, right_elbow = keypoints[7], keypoints[8]
    left_hip, right_hip = keypoints[11], keypoints[12]

    left_vec = np.array(left_elbow) - np.array(left_shoulder)
    right_vec = np.array(right_elbow) - np.array(right_shoulder)
    left_vertical = np.array(left_shoulder) - np.array(left_hip)
    right_vertical = np.array(right_shoulder) - np.array(right_hip)

    left_angle = calculate_angle(left_vec, left_vertical)
    right_angle = calculate_angle(right_vec, right_vertical)
    return (left_angle <= 110 and right_angle <= 110), left_angle, right_angle


def is_bbox_too_close_to_edge(x, y, w, h, frame_width, frame_height, margin_percentage=MARGIN_PERCENTAGE):
    margin_x = frame_width * margin_percentage
    margin_y = frame_height * margin_percentage
    return (
        x - w / 2 < margin_x
        or x + w / 2 > frame_width - margin_x
        or y - h / 2 < margin_y
        or y + h / 2 > frame_height - margin_y
    )


def is_bbox_middle(x, y, w, h, frame_width, frame_height, margin_percentage=CENTER_BAND_RATIO):
    center_x = frame_width * 0.5
    band_half = frame_width * float(margin_percentage)
    return (center_x - band_half) <= float(x) <= (center_x + band_half)


def start_tracking(result, id_a_suivre, frame_width=3840, frame_height=1920, center_band_ratio=CENTER_BAND_RATIO):
    if not (result and result.boxes is not None) or id_a_suivre is not None:
        return id_a_suivre

    boxes_xywh = result.boxes.xywh.cpu().numpy()
    track_ids = result.boxes.id.int().cpu().tolist() if result.boxes.id is not None else [-1] * len(boxes_xywh)

    candidates = []
    center_x = frame_width * 0.5
    for (cx, cy, bw, bh), track_id in zip(boxes_xywh, track_ids):
        if track_id == -1:
            continue
        dist = abs(float(cx) - center_x)
        in_band = is_bbox_middle(cx, cy, bw, bh, frame_width, frame_height, center_band_ratio)
        candidates.append((in_band, dist, track_id))

    if not candidates:
        return None

    in_band_candidates = [candidate for candidate in candidates if candidate[0]]
    chosen = min(in_band_candidates, key=lambda item: item[1]) if in_band_candidates else min(candidates, key=lambda item: item[1])
    return chosen[2]


def detect_t_pose(result, t_pose_person, t_pose_threshold, t_pose_duration, width=3840, height=1920):
    if result and result.boxes is not None and t_pose_person is None:
        boxes = result.boxes.xywh.cpu().numpy()
        track_ids = result.boxes.id.int().cpu().tolist() if result.boxes.id is not None else [-1] * len(boxes)
        keypoints = result.keypoints.xy.cpu().numpy()

        for idx, keypoint in enumerate(keypoints):
            t_pose_detected, _, _ = is_t_pose(keypoint)
            x, y, w, h = boxes[idx]

            if is_bbox_too_close_to_edge(x, y, w, h, width, height):
                continue

            if t_pose_detected:
                track_id = track_ids[idx]
                if track_id not in t_pose_duration:
                    t_pose_duration[track_id] = 0
                t_pose_duration[track_id] += 1

                if t_pose_duration[track_id] >= t_pose_threshold:
                    return True, track_id, t_pose_threshold, t_pose_duration

    return False, t_pose_person, t_pose_threshold, t_pose_duration


def align_equi_to_tposed(result, t_pose_person, frame_width=3840, frame_height=1920):
    if result and result.boxes is not None:
        boxes = result.boxes.xywh.cpu().numpy()
        track_ids = result.boxes.id.int().cpu().tolist() if result.boxes.id is not None else [-1] * len(boxes)
        keypoints = result.keypoints.xy.cpu().numpy()

        print(f"Frame 1: Found {len(boxes)} persons, Track IDs: {track_ids}")
        for idx, keypoint in enumerate(keypoints):
            print(track_ids[idx], " track_ids[idx]")
            if track_ids[idx] == t_pose_person:
                if len(keypoint) <= 6:
                    continue

                right_shoulder = np.asarray(keypoint[6], dtype=np.float32)
                if not np.all(np.isfinite(right_shoulder)):
                    continue

                cx = float(right_shoulder[0])
                cy = float(right_shoulder[1])
                new_yaw = (cx / frame_width) * 360 - 180
                pitch = -(cy / frame_height) * 180 + 90
                print(f"[ALIGN] shoulders cx={cx:.1f}, cy={cy:.1f}, yaw={new_yaw:.1f}, pitch={pitch:.1f}")
                return True, new_yaw, pitch

    return False, 0.0, 0.0


def main_tracking(result, id_a_suivre, rotate_frame, pitch, frame_width=3840, frame_height=1920):
    if result and result.boxes is not None and id_a_suivre is not None:
        boxes = result.boxes.xywh.cpu().numpy()
        track_ids = result.boxes.id.int().cpu().tolist() if result.boxes.id is not None else [-1] * len(boxes)
        keypoints = result.keypoints.xy.cpu().numpy()

        for idx, _ in enumerate(keypoints):
            if track_ids[idx] == id_a_suivre:
                x, y, w, h = boxes[idx]
                y = y - h / 6
                new_yaw = (x / frame_width) * 360 - 180
                pitch = -(y / frame_height) * 180 + 90
                print(f"[TRACK] cy={y:.1f}  pitch(deg)={pitch:.1f}")
                rotate_frame += new_yaw
                return rotate_frame, pitch

    return rotate_frame, pitch


track_with_bbox_center = main_tracking
