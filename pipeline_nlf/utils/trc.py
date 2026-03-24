import numpy as np


def generate_trc_file(pred_verts, cfg_bio, output_path, frame_rate, R):
    num_frames = len(pred_verts)
    num_markers = len(cfg_bio)
    marker_headers = "\t".join([f"X{i+1}\tY{i+1}\tZ{i+1}" for i in range(num_markers)])
    header = [
        f"PathFileType\t4\t(X/Y/Z)\t{output_path}",
        "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames",
        f"{frame_rate:.2f}\t{frame_rate:.2f}\t{num_frames}\t{num_markers}\tmm\t{frame_rate:.2f}\t1\t{num_frames}",
        "Frame#\tTime\t" + "\t".join([f"{name}\t\t" for name in cfg_bio.keys()]),
        "\t\t" + marker_headers,
    ]

    trc_data = []
    for frame_idx, frame_verts in enumerate(pred_verts):
        time = frame_idx / frame_rate
        frame_data = [str(frame_idx + 1), f"{time:.5f}"]
        for marker in cfg_bio.keys():
            idx = cfg_bio[marker]
            p_world = np.array(frame_verts[idx]) @ np.array(R[frame_idx])
            frame_data.extend([f"{p_world[0]:.5f}", f"{p_world[1]:.5f}", f"{p_world[2]:.5f}"])
        trc_data.append("\t".join(frame_data))

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(header) + "\n")
        f.write("\n".join(trc_data) + "\n")

    print(f"TRC file successfully saved to {output_path}")


def generate_trc_file_subset(pred_verts, cfg_bio, output_path, frame_rate, R):
    num_frames = len(pred_verts)
    first_valid_frame = next((frame for frame in pred_verts if len(frame) > 0), None)
    if first_valid_frame is None:
        raise ValueError("No valid frames with markers found.")

    num_markers = len(first_valid_frame)
    marker_names = list(cfg_bio.keys())
    if len(marker_names) != num_markers:
        raise ValueError("cfg_bio and pred_verts length mismatch.")

    marker_headers = "\t".join([f"X{i+1}\tY{i+1}\tZ{i+1}" for i in range(num_markers)])
    marker_names_line = "Frame#\tTime\t" + "\t".join([f"{name}\t\t" for name in marker_names])
    header = [
        f"PathFileType\t4\t(X/Y/Z)\t{output_path}",
        "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames",
        f"{frame_rate:.2f}\t{frame_rate:.2f}\t{num_frames}\t{num_markers}\tmm\t{frame_rate:.2f}\t1\t{num_frames}",
        marker_names_line,
        "\t\t" + marker_headers,
    ]

    trc_data = []
    for frame_idx, frame_verts in enumerate(pred_verts):
        time = frame_idx / frame_rate
        frame_data = [str(frame_idx + 1), f"{time:.5f}"]
        transformed = np.array(frame_verts) @ np.array(R[frame_idx])
        for x, y, z in transformed:
            frame_data.extend([f"{x:.5f}", f"{y:.5f}", f"{z:.5f}"])
        trc_data.append("\t".join(frame_data))

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(header) + "\n")
        f.write("\n".join(trc_data) + "\n")

    print(f"TRC file successfully saved to {output_path}")


def generate_trc_file_world_frame(
    pred_verts,
    cfg_bio,
    output_path,
    timestamps_s,
    R_list,
    units="mm",
    world_offset_y=0.75,
):
    marker_names = list(cfg_bio.keys())
    marker_count = len(marker_names)
    frame_count = len(pred_verts)

    if not (len(R_list) == len(timestamps_s) == frame_count):
        raise ValueError("pred_verts, timestamps_s, and R_list must match in length.")

    timestamps = np.asarray(timestamps_s, dtype=np.float64)
    timestamps -= timestamps[0]

    fixed_verts = []
    for verts_mm, rotation in zip(pred_verts, R_list):
        verts_mm = np.asarray(verts_mm, dtype=np.float64)
        rotation = np.asarray(rotation, dtype=np.float64)

        if verts_mm.shape != (marker_count, 3):
            verts_mm = np.full((marker_count, 3), np.nan)

        verts_m = verts_mm / 1000.0
        verts_cam = verts_m @ rotation.T

        verts_world = np.zeros_like(verts_cam)
        for idx, (x, y, z) in enumerate(verts_cam):
            verts_world[idx] = np.array([-z, -y + world_offset_y, -x])

        fixed_verts.append(verts_world * 1000.0)

    diffs = np.diff(timestamps)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    data_rate = (1.0 / np.median(diffs)) if diffs.size else 0.0

    marker_headers = "\t".join([f"X{i+1}\tY{i+1}\tZ{i+1}" for i in range(marker_count)])
    marker_names_line = "Frame#\tTime\t" + "\t".join([f"{name}\t\t" for name in marker_names])
    header = [
        f"PathFileType\t4\t(X/Y/Z)\t{output_path}",
        "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames",
        f"{data_rate:.2f}\t{data_rate:.2f}\t{frame_count}\t{marker_count}\t{units}\t{data_rate:.2f}\t1\t{frame_count}",
        marker_names_line,
        "\t\t" + marker_headers,
    ]

    lines = []
    for idx in range(frame_count):
        verts = fixed_verts[idx]
        row = [str(idx + 1), f"{float(timestamps[idx]):.5f}"]
        for x, y, z in verts:
            if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                row += [f"{x:.5f}", f"{y:.5f}", f"{z:.5f}"]
            else:
                row += ["", "", ""]
        lines.append("\t".join(row))

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(header) + "\n")
        f.write("\n".join(lines) + "\n")

    print(f"[TRC] Saved world-frame TRC -> {output_path}")
    print(f"[TRC] Frames={frame_count}, Markers={marker_count}, DataRate~{data_rate:.2f} Hz")


def generate_trc_file_subset_native_time(
    pred_verts,
    cfg_bio,
    output_path,
    timestamps_s,
    R_list,
    units="mm",
    scale=None,
):
    marker_names = list(cfg_bio.keys())
    marker_count = len(marker_names)
    frame_count = len(pred_verts)
    if not (len(R_list) == len(timestamps_s) == frame_count):
        raise ValueError("Lengths of pred_verts, R_list, and timestamps_s must match.")

    timestamps = np.asarray(timestamps_s, dtype=np.float64)
    if timestamps.size == 0:
        raise ValueError("Empty timestamps_s.")
    timestamps = timestamps - timestamps[0]
    if not np.all(np.diff(timestamps) >= 0):
        order = np.argsort(timestamps)
        timestamps = timestamps[order]
        pred_verts = [pred_verts[i] for i in order]
        R_list = [R_list[i] for i in order]

    fixed_verts = []
    fixed_rotations = []
    identity = np.eye(3, dtype=np.float64)
    for verts, rotation in zip(pred_verts, R_list):
        verts_arr = np.asarray(verts, dtype=np.float64)
        if verts_arr.shape != (marker_count, 3):
            verts_arr = np.full((marker_count, 3), np.nan, dtype=np.float64)
        if scale is not None:
            verts_arr = verts_arr * float(scale)

        rotation_arr = np.asarray(rotation, dtype=np.float64)
        if rotation_arr.shape != (3, 3) or not np.isfinite(rotation_arr).all():
            rotation_arr = identity.copy()

        fixed_verts.append(verts_arr)
        fixed_rotations.append(rotation_arr)

    diffs = np.diff(timestamps)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    data_rate = (1.0 / np.median(diffs)) if diffs.size else 0.0

    marker_headers = "\t".join([f"X{i+1}\tY{i+1}\tZ{i+1}" for i in range(marker_count)])
    marker_names_line = "Frame#\tTime\t" + "\t".join([f"{name}\t\t" for name in marker_names])
    header = [
        f"PathFileType\t4\t(X/Y/Z)\t{output_path}",
        "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames",
        f"{data_rate:.2f}\t{data_rate:.2f}\t{frame_count}\t{marker_count}\t{units}\t{data_rate:.2f}\t1\t{frame_count}",
        marker_names_line,
        "\t\t" + marker_headers,
    ]

    lines = []
    for idx in range(frame_count):
        verts = fixed_verts[idx] @ fixed_rotations[idx]
        row = [str(idx + 1), f"{float(timestamps[idx]):.5f}"]
        for x, y, z in verts:
            if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                row += [f"{x:.5f}", f"{y:.5f}", f"{z:.5f}"]
            else:
                row += ["", "", ""]
        lines.append("\t".join(row))

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(header) + "\n")
        f.write("\n".join(lines) + "\n")

    print(
        f"TRC (native time) saved -> {output_path}  |  ~DataRate {data_rate:.2f} Hz  |  Frames {frame_count}  Markers {marker_count}"
    )
