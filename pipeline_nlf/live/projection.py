import numpy as np


def project_vertices_to_equirectangular(vertices, image_width, image_height):
    vertices = np.asarray(vertices)
    norms = np.linalg.norm(vertices, axis=1, keepdims=True)
    directions = vertices / norms
    x, y, z = directions[:, 0], directions[:, 1], directions[:, 2]
    yaw = np.arctan2(x, z)
    pitch = np.arcsin(y)
    u = (yaw + np.pi) / (2 * np.pi) * image_width
    v = ((pitch + np.pi / 2) / np.pi) * image_height
    return np.stack([u, v], axis=1)


def project_mesh_from_pseudo_to_equi(vertices_local, yaw_deg, pitch_deg, width, height):
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    rot_yaw = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    rot_pitch = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)], [0, np.sin(pitch), np.cos(pitch)]])
    vertices_rotated = vertices_local @ (rot_yaw @ rot_pitch).T
    return project_vertices_to_equirectangular(vertices_rotated, width, height)
