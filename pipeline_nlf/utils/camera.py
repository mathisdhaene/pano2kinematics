import numpy as np


def compute_camera_parameters(out_width, out_height, fov, yaw, pitch):
    fov_rad = np.radians(fov)
    yaw_rad = np.radians(yaw)
    pitch_rad = np.radians(pitch)

    f_x = f_y = out_width / (2 * np.tan(fov_rad / 2))
    c_x, c_y = out_width / 2, out_height / 2

    intrinsic = np.array([[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]])
    rot_yaw = np.array(
        [
            [np.cos(yaw_rad), 0, np.sin(yaw_rad)],
            [0, 1, 0],
            [-np.sin(yaw_rad), 0, np.cos(yaw_rad)],
        ]
    )
    rot_pitch = np.array(
        [
            [1, 0, 0],
            [0, np.cos(pitch_rad), -np.sin(pitch_rad)],
            [0, np.sin(pitch_rad), np.cos(pitch_rad)],
        ],
        dtype=np.float32,
    )

    rotation = rot_yaw @ rot_pitch
    translation = np.zeros((3, 1))
    return intrinsic, rotation, translation
