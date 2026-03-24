def track_with_shoulder_keypoint(result, id_a_suivre, rotate_frame, pitch, frame_width=3840, frame_height=1920):
    """Track the followed subject using the shoulder keypoint instead of bbox center."""
    if result and result.boxes is not None and id_a_suivre is not None:
        boxes = result.boxes.xywh.cpu().numpy()
        track_ids = result.boxes.id.int().cpu().tolist() if result.boxes.id is not None else [-1] * len(boxes)
        keypoints = result.keypoints.xy.cpu().numpy()

        for idx, _ in enumerate(keypoints):
            if track_ids[idx] == id_a_suivre:
                shoulder = keypoints[idx][6]
                cx = shoulder[0]
                cy = shoulder[1]
                yaw = (cx / frame_width) * 360 - 180
                pitch = -(cy / frame_height) * 180 + 90
                return rotate_frame, pitch

    return rotate_frame, pitch


# Backward-compatible alias
main_tracking = track_with_shoulder_keypoint
