#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
august_parsed_live_cpu.py

This is your working `august_parsed.py` pipeline with ONE change: the video source can now be
- an MP4 file (original behavior), or
- a Ricoh Theta live stream via GStreamer shmsink using GstShmReader (`--live --shm-socket ...`).

Everything else — T‑pose search, alignment, FOLLOW PID, EquiLib pseudo‑camera, NLF inference,
reprojection, tracked/pseudo writers, TRC export — is preserved.

Run (live):
  python3 august_parsed_live_cpu.py \
    --live --shm-socket /tmp/theta_bgr.sock \
    --fps 30 --device cpu \
    --yolo weights/yolo_models/yolo11m-pose.pt \
    --tracker bytetrack.yaml \
    --bio-cfg configs/biomeca.yaml \
    --nlf-weights weights/nlf/nlf_s_multi.torchscript \
    -o output_nlf/markerless_live.mp4

Run (offline, unchanged):
  python3 august_parsed_live_cpu.py \
    -i /path/to/video.mp4 -o output_nlf/markerless_1.mp4 \
    --fps 30 --device cpu \
    --yolo weightsyolo_models/yolo11m-pose.pt \
    --tracker bytetrack.yaml \
    --bio-cfg configs/biomeca.yaml \
    --nlf-weights weights/nlf/nlf_s_multi.torchscript
"""

import argparse
from pathlib import Path
import os
import time
from collections import defaultdict

import socket

# Socket Setup
HOST = '127.0.0.1'
PORT = 5555
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((HOST, PORT))

import torch
import numpy as np
import cv2
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
import yaml

import torchvision.transforms as T
import torch._dynamo
try:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
except Exception:
    Axes3D = None
import importlib.util
from pipeline_nlf.utils import *
# ======================================

torch._dynamo.config.suppress_errors = True
torch._dynamo.config.verbose = True

# -------------------- CLI --------------------

def build_parser():
    p = argparse.ArgumentParser(
        description="Run NLF pipeline with YOLO tracking on an input .mp4 or a live shmsrc"
    )
    # Compat: -input/-output
    p.add_argument("-input", "-i", "--input", dest="input_path",
                   default="data/videos/sample_1.mp4",
                   help="Chemin vers la vidéo d'entrée (.mp4)")
    p.add_argument("-output", "-o", "--output", dest="output_path",
                   default="output_nlf/markerless_1.mp4",
                   help="Chemin vers la vidéo de sortie (.mp4)")

    # Options
    p.add_argument("--fps", type=float, default=30.0, help="FPS des sorties")
    p.add_argument("--device", default="cpu", help="cuda:0 / cpu")
    p.add_argument("--max-frames", type=int, default=3000, help="Limiter le nb de frames")
    p.add_argument("--yolo", default="yolo_models/yolo11m-pose.pt",
                   help="Chemin du modèle YOLO pose")
    p.add_argument("--tracker", default="bytetrack.yaml", help="Tracker pour YOLO")
    p.add_argument("--bio-cfg", default="configs/biomeca.yaml", help="YAML biomécanique")
    p.add_argument("--nlf-weights", default="weights/nlf/nlf_s_multi.torchscript",
                   help="Chemin du modèle NLF TorchScript")

    # NEW: live mode
    p.add_argument("--live", action="store_true", help="Read from shmsrc via GstShmReader")
    p.add_argument("--shm-socket", default="/tmp/theta_bgr.sock", help="Base path to shmsink socket")
    p.add_argument("--display-only", action="store_true",
               help="Do not write videos/TRC; show ori_img live")

    return p


def compute_dynamic_fov_from_bbox(bbox, img_w=3840):
    """
    Compute dynamic FOV based on bbox size.
    bbox = (cx, cy, w, h)
    Returns FOV in degrees, clamped to [20°, 70°].
    """
    _, _, w, h = bbox
    size = max(w, h)
    s = size / img_w  # normalized scale

    min_fov = 20.0
    max_fov = 70.0

    # Linear mapping: s = 0.30 → 70°, s = 0.0 → 20°
    FOV = min_fov + (max_fov - min_fov) * (s / 0.30)
    FOV = max(min_fov, min(FOV, max_fov))

    return FOV


def _load_equi2pers_numpy_run():
    equi2pers_path = Path(__file__).resolve().parent / "equilib" / "equi2pers" / "numpy.py"
    spec = importlib.util.spec_from_file_location("equi2pers_numpy", equi2pers_path)
    module = importlib.util.module_from_spec(spec)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load equi2pers numpy module from {equi2pers_path}")
    spec.loader.exec_module(module)
    return module.run


equi2pers_run = _load_equi2pers_numpy_run()


def preprocess(img: np.ndarray, device: str = "cpu") -> torch.Tensor:
    img_tensor = (
        torch.from_numpy(img)
        .to(device, non_blocking=True)
        .permute(2, 0, 1)
        .float()
        .div(255)
        .unsqueeze(0)
    )
    return img_tensor


def compute_camera_parameters(out_width, out_height, fov, yaw, pitch):
    fov_rad = np.radians(fov)
    yaw_rad = np.radians(yaw)
    pitch_rad = np.radians(pitch)

    f_x = f_y = out_width / (2 * np.tan(fov_rad / 2))
    c_x, c_y = out_width / 2, out_height / 2

    K = np.array([[f_x, 0, c_x], [0, f_y, c_y], [0, 0, 1]])

    R_yaw = np.array(
        [[np.cos(yaw_rad), 0, np.sin(yaw_rad)], [0, 1, 0], [-np.sin(yaw_rad), 0, np.cos(yaw_rad)]]
    )
    R_pitch = np.array(
        [[1, 0, 0], [0, np.cos(pitch_rad), -np.sin(pitch_rad)], [0, np.sin(pitch_rad), np.cos(pitch_rad)]],
        dtype=np.float32,
    )

    R = R_pitch @ R_yaw
    t = np.zeros((3, 1))
    return K, R, t


def rotate_image(image, yaw):
    width = int(image.shape[1])
    pixel = int((yaw + 180) * width / 360)
    return np.roll(image, int((width / 2) - pixel), axis=1)


def calculate_angle(v1, v2):
    dot_product = np.dot(v1, v2)
    magnitude = np.linalg.norm(v1) * np.linalg.norm(v2)
    cosine_angle = dot_product / magnitude if magnitude != 0 else 0
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return angle


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
    return (left_angle <= 130 and right_angle <= 130), left_angle, right_angle


MARGIN_PERCENTAGE = 0.01
MARGIN_PERCENTAGE_middle = 0.05
CENTER_BAND_RATIO = 0.10


def is_bbox_too_close_to_edge(x, y, w, h, frame_width, frame_height, margin_percentage=MARGIN_PERCENTAGE):
    margin_x = frame_width * margin_percentage
    margin_y = frame_height * margin_percentage
    if x - w / 2 < margin_x or x + w / 2 > frame_width - margin_x or y - h / 2 < margin_y or y + h / 2 > frame_height - margin_y:
        return True
    return False


def is_bbox_middle(x, y, w, h, frame_width, frame_height, margin_percentage=CENTER_BAND_RATIO):
    cx = float(x)
    band_half = frame_width * float(margin_percentage)
    center_x = frame_width * 0.5
    return (center_x - band_half) <= cx <= (center_x + band_half)


def start_tracking(result, id_a_suivre, frame_width=3840, frame_height=1920, center_band_ratio=CENTER_BAND_RATIO):
    if not (result and result.boxes is not None) or id_a_suivre is not None:
        return id_a_suivre

    boxes_xywh = result.boxes.xywh.cpu().numpy()
    track_ids = (result.boxes.id.int().cpu().tolist() if result.boxes.id is not None else [-1] * len(boxes_xywh))

    candidates = []
    center_x = frame_width * 0.5
    for (cx, cy, bw, bh), tid in zip(boxes_xywh, track_ids):
        if tid == -1:
            continue
        dist = abs(float(cx) - center_x)
        in_band = is_bbox_middle(cx, cy, bw, bh, frame_width, frame_height, center_band_ratio)
        candidates.append((in_band, dist, tid))

    if not candidates:
        return None

    in_band_candidates = [c for c in candidates if c[0]]
    chosen = min(in_band_candidates, key=lambda x: x[1]) if in_band_candidates else min(candidates, key=lambda x: x[1])
    _, _, chosen_id = chosen
    return chosen_id


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
                if track_ids[idx] not in t_pose_duration:
                    t_pose_duration[track_ids[idx]] = 0
                t_pose_duration[track_ids[idx]] = 1 + t_pose_duration[track_ids[idx]]

                if t_pose_duration[track_ids[idx]] >= t_pose_threshold:
                    return True, track_ids[idx], t_pose_threshold, t_pose_duration

    return False, t_pose_person, t_pose_threshold, t_pose_duration


def align_equi_to_tposed(result, t_pose_person):
    if not result or result.boxes is None:
        return False, 0.0, 0.0

    w_img, h_img = 3840, 1920
    boxes = result.boxes.xywh.cpu().numpy()

    track_ids = (
        result.boxes.id.int().cpu().tolist()
        if result.boxes.id is not None
        else [-1] * len(boxes)
    )

    keypoints = result.keypoints.xy.cpu().numpy()

    for idx in range(len(keypoints)):
        if track_ids[idx] != t_pose_person:
            continue

        kp = keypoints[idx]
        L = kp[5]
        R = kp[6]

        cx = R[0]
        cy = R[1]

        yaw_deg = (cx / w_img) * 360.0 - 180.0
        pitch_deg = -(cy / h_img) * 180.0 + 90.0

        print(f"[ALIGN] shoulders cx={cx:.1f}, cy={cy:.1f}, yaw={yaw_deg:.1f}, pitch={pitch_deg:.1f}")

        return True, yaw_deg, pitch_deg

    return False, 0.0, 0.0


def main_tracking(result, id_a_suivre, rotate_frame, pitch):
    if result and result.boxes is not None and id_a_suivre is not None:
        boxes = result.boxes.xywh.cpu().numpy()

        if result.boxes.id is not None:
            track_ids = result.boxes.id.int().cpu().tolist()
        else:
            track_ids = [-1] * len(boxes)

        keypoints = result.keypoints.xy.cpu().numpy()

        for idx, keypoint in enumerate(keypoints):
            if track_ids[idx] == id_a_suivre:
                L = keypoints[idx][5]
                R = keypoints[idx][6]

                cx = R[0]
                cy = R[1]

                yaw = (cx / 3840) * 360 - 180
                pitch = -(cy / 1920) * 180 + 90

                return rotate_frame, pitch

    return rotate_frame, pitch


def preprocess_for_nlf_cpu(frame: torch.Tensor, device: str) -> torch.Tensor:
    # CPU-safe version of preprocess_for_nlf (no CUDA, no float16)
    return (frame * 255.0).to(dtype=torch.float32, device=device)



# -------------------- Recorder (unchanged) --------------------

class YOLOTrackedRecorder:
    def __init__(self, model, out_path, fps=30.0, device="cuda:0", tracker="bytetrack.yaml", kp_conf_thresh=None):
        self.model = model
        self.out_path = str(out_path)
        self.fps = float(fps)
        self.device = device
        self.tracker = tracker
        self.kp_conf_thresh = kp_conf_thresh
        self.writer = None
        self.size = None  # (w, h)

    def _ensure_writer(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        if self.writer is None:
            self.size = (w, h)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(self.out_path, fourcc, self.fps, (w, h))

    def process(self, frame_bgr):
        self._ensure_writer(frame_bgr)
        results = self.model.track(
            frame_bgr,
            device=self.device,
            tracker=self.tracker,
            persist=True,
            verbose=False
        )[0]
        im_annot = results.plot()
        if im_annot.shape[1] != self.size[0] or im_annot.shape[0] != self.size[1]:
            im_annot = cv2.resize(im_annot, self.size, interpolation=cv2.INTER_LINEAR)
        self.writer.write(im_annot)
        return results

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None

# -------------------- Projection helpers (unchanged) --------------------
# --- small helpers copied from your original august_parsed.py ---


def ensure_parent_dir(path: Path):
    """Create parent directory for a file path if it doesn’t exist."""
    path.parent.mkdir(parents=True, exist_ok=True)

def derive_side_outputs(main_out: Path):
    """
    Given /path/out.mp4, return:
      /path/out_tracked.mp4, /path/out_pseudo.mp4, /path/out.trc
    """
    stem = main_out.with_suffix("")  # /path/out
    out_tracked = stem.with_name(stem.name + "_tracked").with_suffix(".mp4")
    out_pseudo  = stem.with_name(stem.name + "_pseudo").with_suffix(".mp4")
    out_trc     = stem.with_suffix(".trc")
    return out_tracked, out_pseudo, out_trc


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
    R_yaw = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                      [0, 1, 0],
                      [-np.sin(yaw), 0, np.cos(yaw)]])
    R_pitch = np.array([[1, 0, 0],
                        [0, np.cos(pitch), -np.sin(pitch)],
                        [0, np.sin(pitch),  np.cos(pitch)]])
    R_combined = R_yaw @ R_pitch
    vertices_rotated = vertices_local @ R_combined.T
    return project_vertices_to_equirectangular(vertices_rotated, width, height)

# -------------------- Robust shmsrc reader (embedded) --------------------

import math

def _pick_shm_paths(base: str):
    yield base
    for i in range(10):
        yield f"{base}.{i}"

class GstShmReader:
    """Minimal, robust shmsrc → appsink BGR reader."""
    def __init__(self, sock_base, w, h, fps, timeout_ms=3000):
        try:
            import gi
            gi.require_version("Gst", "1.0")
            from gi.repository import Gst
        except Exception as exc:
            raise RuntimeError(
                "GStreamer bindings (gi) are required for --live. "
                "Install python3-gi / pygobject or run without --live."
            ) from exc

        self.Gst = Gst
        Gst.init(None)
        self.w, self.h = int(w), int(h)
        self.pipeline = None
        self.sink = None
        self.bus = None
        self._closed = False
        self.try_pull_timeout_ns = 500_000_000  # 500ms

        caps = "video/x-raw,format=BGR"
        last_err = None

        for path in _pick_shm_paths(sock_base):
            deadline = time.time() + 3.0
            while not os.path.exists(path) and time.time() < deadline:
                time.sleep(0.05)
            if not os.path.exists(path):
                continue

            pipe = (
                f"shmsrc socket-path={path} is-live=true do-timestamp=true ! "
                f"{caps} ! "
                "appsink name=sink drop=true sync=false max-buffers=1 emit-signals=true"
            )
            try:
                pl = Gst.parse_launch(pipe)
                sink = pl.get_by_name("sink")
                bus = pl.get_bus()
                ret = pl.set_state(Gst.State.PLAYING)
                if ret == Gst.StateChangeReturn.FAILURE:
                    msg = bus.timed_pop_filtered(timeout_ms * Gst.MSECOND, Gst.MessageType.ERROR)
                    if msg:
                        err, dbg = msg.parse_error()
                        last_err = (err, dbg)
                        pl.set_state(Gst.State.NULL)
                        continue
                    pl.set_state(Gst.State.NULL)
                    continue
                self.pipeline, self.sink, self.bus = pl, sink, bus
                print(f"[GstShmReader] Connected to {path}")
                break
            except Exception as e:
                last_err = e
                try: pl.set_state(Gst.State.NULL)
                except Exception: pass
                continue

        if self.pipeline is None:
            if isinstance(last_err, tuple):
                err, dbg = last_err
                raise RuntimeError(f"Failed to open shmsrc: {err.message} [{dbg}]")
            raise RuntimeError(f"Failed to open any shmsrc under {sock_base}")

        self._logged_size = False

    def read(self):
        Gst = self.Gst
        if self._closed or self.pipeline is None:
            return False, None
        if self.bus:
            msg = self.bus.pop_filtered(Gst.MessageType.ERROR | Gst.MessageType.EOS)
            if msg:
                return False, None
        sample = self.sink.emit("try-pull-sample", self.try_pull_timeout_ns)
        if not sample:
            return False, None
        w = h = None
        caps = sample.get_caps()
        if caps:
            st = caps.get_structure(0)
            if st:
                w = st.get_value('width')
                h = st.get_value('height')
        buf = sample.get_buffer()
        ok, mapinfo = buf.map(Gst.MapFlags.READ)
        if not ok:
            return False, None
        arr = np.frombuffer(mapinfo.data, dtype=np.uint8)
        pixels = arr.size // 3
        if w is None or h is None:
            h_guess = int(round(math.sqrt(max(pixels, 0) / 2)))
            w_guess = 2 * h_guess
            if w_guess * h_guess == pixels and h_guess > 0:
                w, h = w_guess, h_guess
            else:
                w, h = self.w, self.h
        expected = int(w) * int(h) * 3
        if arr.size != expected:
            buf.unmap(mapinfo)
            return False, None
        frame = arr.reshape((int(h), int(w), 3)).copy()
        buf.unmap(mapinfo)
        if not self._logged_size:
            print(f"[GstShmReader] negotiated size: {w}x{h}")
            self._logged_size = True
        return True, frame

    def close(self):
        Gst = self.Gst
        if self._closed:
            return
        self._closed = True
        try:
            if self.pipeline:
                self.pipeline.set_state(Gst.State.PAUSED)
                self.pipeline.set_state(Gst.State.READY)
                self.pipeline.set_state(Gst.State.NULL)
        finally:
            self.pipeline = None
            self.sink = None
            try:
                Gst.deinit()
            except Exception:
                pass

# -------------------- MAIN (identical logic; only source differs) --------------------

def main():
    args = build_parser().parse_args()
    if args.display_only:
        cv2.namedWindow("markerless_live", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("markerless_live", 960, 480)

    display_only = args.display_only

    input_path = Path(args.input_path)
    output_main = Path(args.output_path)

    # Side outputs
    out_tracked, out_pseudo, out_trc = derive_side_outputs(output_main)

    # Dirs
    ensure_parent_dir(output_main)
    ensure_parent_dir(out_tracked)
    ensure_parent_dir(out_pseudo)
    ensure_parent_dir(out_trc)



    # Register torchvision custom ops BEFORE loading TS (fixes torchvision::nms)
    import torchvision
    _ = torchvision.ops.nms

    model = torch.jit.load(args.nlf_weights, map_location=args.device).to(args.device).eval()



    device = args.device
    model_yolo = YOLO(args.yolo)
    model_yolo2 = YOLO(args.yolo)

    rec = None
    if not display_only:
        rec = YOLOTrackedRecorder(model_yolo, out_tracked, fps=args.fps, device=device, tracker=args.tracker)

    # ---- Video source ----
    reader = None
    cap = None
    if args.live:
        reader = GstShmReader(args.shm_socket, w=3840, h=1920, fps=args.fps)
    else:
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Impossible d’ouvrir la vidéo d’entrée: {input_path}")

    # Writers (disabled in display-only mode)
    if not display_only:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_main), fourcc, args.fps, (3840, 1920))
        pseudo_out = cv2.VideoWriter(str(out_pseudo), fourcc, args.fps, (256, 256))
    else:
        out = None
        pseudo_out = None

    rec = None
    if not display_only:
        rec = YOLOTrackedRecorder(model_yolo, out_tracked, fps=args.fps, device=device, tracker=args.tracker)


    # ---- Tracking & states ----
    count = 0
    county = 0
    beta_premiere_frame = None
    track_history = defaultdict(lambda: [])
    t_pose_threshold = 5  # @30 FPS
    t_pose_person1 = None
    t_pose_duration1 = {}
    t_pose_person2 = None
    t_pose_duration2 = {}
    tpose_found = False
    initialisation = False
    id_a_suivre = None
    t_pose_person = None
    rotate_frame = 0
    pitch = 0.0

    # Export TRC / rotations
    vertices_list = []
    R_list = []

    # NLF precomputes (unchanged)
    cano_verts = np.load("pipeline_nlf/canonical_verts/smpl.npy")
    indices = [4271, 4779, 1297, 3171, 3077, 3014, 5273, 4223, 5287, 5336,
               4873, 4978, 4794, 5208, 5153, 5567, 5691, 5524, 5456, 3470]
    selected_points = torch.tensor(cano_verts[indices]).float().to(args.device)
    weights_subset = model.get_weights_for_canonical_points(selected_points)
    precomputed_box = [torch.tensor([[0, 0, 256, 256]], device=args.device)]

    # -------------------- Loop --------------------
    try:
        while (args.live or (cap is not None and cap.isOpened())) and count < args.max_frames:
            total_start_time = time.time()

            if args.live:
                ok, framee = reader.read()
                if not ok or framee is None:
                    continue
            else:
                ok, framee = cap.read()
                if not ok:
                    break

            count += 1

            start_rotate = time.time()
            frame = rotate_image(framee, rotate_frame)
            # --- Immediate live preview so you always see something ---

# ----------------------------------------------------------

            end_rotate = time.time()

            if not tpose_found:
                shifted_image_180 = rotate_image(frame, 180)
                result1 = model_yolo.track(frame, persist=True, device=device, tracker=args.tracker, save=False)[0]
                if rec is not None:
                    rec.process(frame)

                result2 = model_yolo2.track(shifted_image_180, persist=True, device=device, tracker=args.tracker, save=False)[0]
                tpose_found1, t_pose_person1, t_pose_threshold, t_pose_duration1 = detect_t_pose(result1, t_pose_person1, t_pose_threshold, t_pose_duration1)
                tpose_found2, t_pose_person2, t_pose_threshold, t_pose_duration2 = detect_t_pose(result2, t_pose_person2, t_pose_threshold, t_pose_duration2)
                if tpose_found1:
                    tpose_found = tpose_found1
                    t_pose_person = t_pose_person1
                    image_id = 0
                elif tpose_found2:
                    tpose_found = tpose_found2
                    t_pose_person = t_pose_person2
                    image_id = 1

                ori_img = np.clip(frame, 0, 255).astype(np.uint8)
                if display_only:
                    preview = cv2.resize(ori_img, (960, 480))
                    cv2.imshow("markerless_live", preview)
                    if (cv2.waitKey(1) & 0xFF) == 27:  # ESC
                        break

            elif tpose_found and not initialisation:
                shifted_image_180 = rotate_image(frame, 180)
                if image_id == 0:
                    if rec is not None:
                        rec.process(frame)

                    result1 = model_yolo.track(frame, persist=True, device=device, tracker=args.tracker, save=False)[0]
                    initialisation, rotate_frame, pitch = align_equi_to_tposed(result1, t_pose_person)
                elif image_id == 1:
                    result2 = model_yolo2.track(shifted_image_180, persist=True, device=device, tracker=args.tracker, save=False)[0]
                    if rec is not None:
                        rec.process(frame)

                    initialisation, rotate_frame, pitch = align_equi_to_tposed(result2, t_pose_person)
                    rotate_frame += 180

            elif tpose_found and initialisation and id_a_suivre is None:
                result = model_yolo.track(frame, persist=True, device=device, tracker=args.tracker, save=False)[0]
                id_a_suivre = start_tracking(result, id_a_suivre)

            elif tpose_found and initialisation and id_a_suivre is not None:
                track_start = time.time()
                county += 1

                result = model_yolo.track(frame, persist=True, device=device, tracker=args.tracker, save=False)[0]
                if rec is not None:
                    rec.process(frame)

                track_end = time.time()
                rotate_frame, pitch = main_tracking(result, id_a_suivre, rotate_frame, pitch)

                temps_avant_preprocess = time.time()
                preprocessed_equi_rendering = preprocess(framee, device)
                temps_apres_preprocess = time.time()

                if "cuda" in device and torch.cuda.is_available():
                    torch.cuda.synchronize()
                start_camera = time.time()
                # ---- Dynamic FOV based on bbox size ----
                bbox = None
                for idx, tid in enumerate(result.boxes.id.int().cpu().tolist()):
                    if tid == id_a_suivre:
                        bbox = result.boxes.xywh.cpu().numpy()[idx]
                        break

                if bbox is not None:
                    FOV = compute_dynamic_fov_from_bbox(bbox)
                else:
                    FOV = 55.0   # fallback
                with torch.inference_mode():
                    preprocessed_equi_np = preprocessed_equi_rendering.detach().cpu().numpy()
                    pseudo_camera_np = equi2pers_run(
                        equi=preprocessed_equi_np,
                        rots=[{
                            'yaw': float(np.deg2rad(-rotate_frame)),
                            'roll': 0.0,
                            'pitch': float(np.deg2rad(-pitch))
                        }],
                        height=256,
                        width=256,
                        fov_x=FOV,        # <--- DYNAMIC FOV HERE
                        skew=0.0,
                        z_down=False,
                        mode="bilinear"
                    )
                    pseudo_camera = torch.from_numpy(pseudo_camera_np).to(device)

                if "cuda" in device and torch.cuda.is_available():
                    torch.cuda.synchronize()

                temps_apres_transfo = time.time()
                input_tensor = preprocess_for_nlf_cpu(pseudo_camera, device)

                K, R, t = compute_camera_parameters(
                        input_tensor.size(2),
                        input_tensor.size(3),
                        FOV,            # <---- USE dynamic FOV
                        rotate_frame,
                        -pitch
                    )

                R_list.append(R.tolist())

                if "cuda" in device and torch.cuda.is_available():
                    torch.cuda.synchronize()
                inference_start = time.time()
                with torch.inference_mode():
                    pred = model.estimate_poses_batched(
                        input_tensor,
                        precomputed_box,
                        weights_subset,
                        default_fov_degrees=35,
                        internal_batch_size=1,
                        num_aug=1,
                        antialias_factor=1
                    )
                if "cuda" in device and torch.cuda.is_available():
                    torch.cuda.synchronize()

                poses3d = pred["poses3d"][0]
                vertices3d = poses3d.cpu().numpy()[0]
                inference_end = time.time()

                pose2d = pred["poses2d"][0].squeeze()
                pseudo_frame = pseudo_camera.squeeze(0).permute(1, 2, 0).cpu().numpy()
                pseudo_frame = (pseudo_frame * 255).astype(np.uint8).copy()
                for point in pose2d:
                    if not torch.isfinite(point).all():
                        continue
                    x, y = int(round(point[0].item())), int(round(point[1].item()))
                    if 0 <= x < pseudo_frame.shape[1] and 0 <= y < pseudo_frame.shape[0]:
                        cv2.circle(pseudo_frame, (x, y), 2, (0, 0, 255), -1)

                # ---------------- LIVE VIEW OF PSEUDO PERSPECTIVE + KEYPOINTS ----------------
                if display_only:
                    preview_pseudo = cv2.resize(pseudo_frame, (512, 512))
                    cv2.imshow("pseudo_camera", preview_pseudo)
                    if (cv2.waitKey(1) & 0xFF) == 27:
                        break
                        
                # ---------------- SAVE PSEUDO FRAME ----------------
                if pseudo_out is not None:
                    if pseudo_frame.shape[1] != 256 or pseudo_frame.shape[0] != 256:
                        pseudo_frame = cv2.resize(pseudo_frame, (256, 256))
                    pseudo_out.write(pseudo_frame)
                    print('PITCHHHHHHHHHHHHH')
                    print('')
                    print(pitch)


                if poses3d.shape[0] > 0:
                    projected_2d = project_mesh_from_pseudo_to_equi(vertices3d, rotate_frame, pitch, 3840, 1920)
                    projected_2d = np.array(projected_2d)
                    unnormalized_tensor = unNormalize(input_tensor).cpu()
                    ori_img = preprocessed_equi_rendering.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    ori_img = np.clip(ori_img * 255.0, 0, 255).astype(np.uint8)
                    vertices_list.append(vertices3d.tolist())
                    for point in projected_2d:
                        center = tuple(int(x) for x in np.round(point).tolist())
                        cv2.circle(ori_img, center, 5, (0, 0, 255), -1)

                if display_only:
                    preview = cv2.resize(ori_img, (3840, 1920))
                    cv2.imshow("markerless_live", preview)
                    if (cv2.waitKey(1) & 0xFF) == 27:
                        break
                
# assumes vertices3d already in meters; if it's in millimeters, see below
                vertices3d = vertices3d  / 1000.0
                print(vertices3d)
                V_trans = [(x,y,z) for (x, y, z) in vertices3d]
                V_cam = np.asarray(V_trans, dtype=np.float32) 
                R = np.asarray(R, dtype=np.float32)        # shape (3,3)

                V_world = V_cam @ R.T 
                print(V_world)
                print(V_world)
                print('end')
                V_world = [(-z, -y + 0.75, -x) for (x, y, z) in V_world]
                V_world = np.asarray(V_world, dtype=np.float32)
                sock.sendall(V_world.astype(np.float32).ravel(order="C").tobytes())
                print(f"Sent frame {county} marker data to RTOSIM")

                total_end_time = time.time()
                total_duration = total_end_time - total_start_time
                fps = 1 / total_duration if total_duration > 0 else 0
                print(count)

        rec.close()
    finally:
        if cap is not None:
            cap.release()
        if reader is not None:
            reader.close()
        if out is not None:
            out.release()
        if pseudo_out is not None:
            pseudo_out.release()
        cv2.destroyAllWindows()


    print('county', county)
    # TRC à côté de la sortie principale
    #generate_trc_file_subset(vertices_list, cfg_bio, str(out_trc), int(args.fps), R_list)

if __name__ == "__main__":
    main()
