#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
december_online_with_outputs.py

Goal:
- Behave EXACTLY like december_online_minimal.py in terms of:
  * Phase 1: YOLO T-pose search (0° + 180°)
  * Phase 2: One-shot alignment
  * Phase 3: NLF-only tracking with:
      - Orientation update from 2D (idx_shoulder=13)
      - Depth filtering (EMA)
      - Depth outlier clamp vs ref_depth
      - Calibration AFTER several NLF frames (county > 3)
      - Anatomical FOV from MC5–CLAG
      - Depth-based FOV update with gain_far=6.0 and the same law
- Add what november_offline.py had:
  * Optional videos (equi main, pseudo, YOLO overlay)
  * TRC output in world frame (generate_trc_file_world_frame)
  * Offline MP4 support and optional LIVE shmsrc support
"""

import argparse
from pathlib import Path
import os
import time
import math
import socket
import subprocess
import shlex
from typing import Tuple

import numpy as np
import torch
import cv2
from ultralytics import YOLO
import yaml
import torchvision
import torch._dynamo

from pipeline_nlf.adaptive_fov import estimate_camera_offset_from_shoulder_2d
from pipeline_nlf.utils import (
    rotate_image,
    preprocess,
    compute_camera_parameters,
    detect_t_pose,
    align_equi_to_tposed,
    generate_trc_file_world_frame,
)
from equilib.equi2pers.torch_impl import run as equi2pers_run

# ---------------- TorchDynamo config ----------------
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.verbose = False

# Register torchvision custom ops before TorchScript (torchvision::nms)
_ = torchvision.ops.nms

# ====================================================
#              GStreamer (needed for GstShmReader)
# ====================================================
#import gi
#gi.require_version("Gst", "1.0")
#from gi.repository import Gst

# ====================================================
#              CLI
# ====================================================

def build_parser():
    p = argparse.ArgumentParser(
        description="December minimal logic + November outputs (videos + TRC), offline or live."
    )

    # Source
    p.add_argument("-i", "--input", dest="input_path",
                   help="Input equirect MP4 (Theta). Ignored in --live mode.")
    p.add_argument("--live", action="store_true",
                   help="Read from shmsrc via GstShmReader instead of MP4.")
    p.add_argument("--shm-socket", default="/tmp/theta_bgr.sock",
                   help="Base path to shmsink socket (e.g. /tmp/theta_bgr.sock).")
    p.add_argument("--frame-width", type=int, default=3840,
                   help="Equirectangular frame width.")
    p.add_argument("--frame-height", type=int, default=1920,
                   help="Equirectangular frame height.")

    # Compute
    p.add_argument("--device", default="cuda:0",
                   help="NLF/YOLO/EquiLib device: cuda:0 or cpu.")
    p.add_argument("--fps", type=float, default=30.0,
                   help="Nominal FPS (used for live, and for writers if needed).")
    p.add_argument("--max-frames", type=int, default=100000,
                   help="Limit number of frames.")

    # Models
    p.add_argument("--yolo", default="yolo_models/yolo11x-pose.pt",
                   help="YOLO pose weights.")
    p.add_argument("--tracker", default="bytetrack.yaml",
                   help="ByteTrack tracker config.")
    p.add_argument("--nlf-weights", default="weights/nlf/nlf_l_multi_0.3.2.torchscript",
                   help="NLF TorchScript model path.")

    # Behavior toggles
    p.add_argument("--skip-tpose", action="store_true",
                   help="Skip T-pose detection + alignment (debug).")

    # Outputs
    p.add_argument("-o", "--output", dest="output_path", default="output_nlf/markerless_1.mp4",
                   help="Main equi output mp4 path (or base path if --no-video).")
    p.add_argument("--bio-cfg", default="configs/biomeca.yaml",
                   help="YAML biomechanics config for TRC naming/mapping.")
    p.add_argument("--no-video", action="store_true",
                   help="Disable ALL video writing (still writes TRC if possible).")
    p.add_argument("--display", action="store_true",
                   help="Show OpenCV windows (equi + pseudo).")

    # TRC timing strategy (optional)
    p.add_argument("--trc-use-ffprobe", action="store_true",
                   help="For offline MP4: rebuild uniform timestamps using ffprobe (like november script).")

    # RTOSIM socket
    p.add_argument("--rtosim-host", default="127.0.0.1")
    p.add_argument("--rtosim-port", type=int, default=5555)

    return p


# ====================================================
#              EXACT FOV LOGIC FROM december_online_minimal.py
# ====================================================

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
        z = float(pt[2]) * 1e-3  # mm → m
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

    gain_far=6.0,          # IMPORTANT: matches december_online_minimal.py
    gain_close=0.8,
    exponent_far=2.2,
    exponent_close=0.7,

    deadband=0.02,
    lerp=0.20,

    init_boost_frames=50,
    init_boost_gain=3.0
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
    new_fov = current_fov + lerp * (target - current_fov)
    return float(new_fov)


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


def compute_fov_from_markers(p_left, p_right, safety_margin=1.05,
                             fov_min=12.0, fov_max=75.0):
    dx = abs(p_left[0] - p_right[0])
    z = 0.5 * (p_left[2] + p_right[2])

    if z <= 0 or dx <= 0:
        return None

    fov_rad = 2.0 * math.atan(dx / (2.0 * z))
    fov_deg = math.degrees(fov_rad)
    fov_deg *= safety_margin
    fov_deg = float(np.clip(fov_deg, fov_min, fov_max))
    return fov_deg


# ====================================================
#              Video + TRC helpers (November-like)
# ====================================================

def ensure_parent_dir(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

def derive_side_outputs(main_out: Path):
    stem = main_out.with_suffix("")  # /path/out
    out_tracked = stem.with_name(stem.name + "_tracked").with_suffix(".mp4")
    out_pseudo  = stem.with_name(stem.name + "_pseudo").with_suffix(".mp4")
    out_trc     = stem.with_suffix(".trc")
    return out_tracked, out_pseudo, out_trc

def get_video_fps_and_duration(video_path: str) -> Tuple[int, float, float]:
    cmd_frames = (
        f'ffprobe -v error -select_streams v:0 '
        f'-count_frames -show_entries stream=nb_read_frames '
        f'-of csv=p=0 "{video_path}"'
    )
    out_frames = subprocess.check_output(shlex.split(cmd_frames), text=True).strip()
    num_frames = int(out_frames)

    cmd_dur = (
        f'ffprobe -v error -select_streams v:0 '
        f'-show_entries stream=duration '
        f'-of csv=p=0 "{video_path}"'
    )
    out_dur = subprocess.check_output(shlex.split(cmd_dur), text=True).strip()
    duration = float(out_dur)

    fps_real = num_frames / duration if duration > 0 else 0.0
    print(f"[INFO] ffprobe: duration={duration:.3f}s, frames={num_frames}, fps≈{fps_real:.3f}")
    return num_frames, duration, fps_real


# ====================================================
#              GstShmReader
# ====================================================

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
                print(f"[LIVE] Connected to {path}")
                break
            except Exception as e:
                last_err = e
                try:
                    pl.set_state(Gst.State.NULL)
                except Exception:
                    pass
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


# ====================================================
#              MAIN
# ====================================================

def main():
    args = build_parser().parse_args()

    # ---- outputs paths ----
    output_main = Path(args.output_path)
    out_tracked, out_pseudo, out_trc = derive_side_outputs(output_main)
    ensure_parent_dir(output_main)
    ensure_parent_dir(out_tracked)
    ensure_parent_dir(out_pseudo)
    ensure_parent_dir(out_trc)

    # ---- biomech config for TRC ----
    with open(args.bio_cfg, "r") as f:
        cfg_bio = yaml.safe_load(f)

    # ---- RTOSIM socket ----
    sock = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((args.rtosim_host, args.rtosim_port))
        print(f"[Socket] Connected → RTOSIM {args.rtosim_host}:{args.rtosim_port}")
    except Exception as e:
        print(f"[Socket] WARNING: could not connect to RTOSIM: {e}")
        sock = None

    # ---- display ----
    if args.display:
        cv2.namedWindow("equi", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("equi", 960, 480)
        cv2.namedWindow("pseudo", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("pseudo", 512, 512)

    # ---- device ----
    device = args.device

    # ---- load NLF ----
    model = torch.jit.load(args.nlf_weights, map_location=device).to(device).eval()

    # Precompute NLF subset weights (20 vertices)
    cano_verts = np.load("pipeline_nlf/canonical_verts/smpl.npy")
    indices = [
        4271, 4779, 1297, 3171, 3077, 3014,
        5273, 4223, 5287, 5336, 4873, 4978,
        4794, 5208, 5153, 5567, 5691, 5524,
        5456, 3470
    ]
    selected_points = torch.tensor(cano_verts[indices]).float().to(device)
    weights_subset = model.get_weights_for_canonical_points(selected_points)

    # Fixed 384×384 box for NLF
    box = torch.tensor([[0, 0, 384, 384, 1.0]], device=device, dtype=torch.float32)
    boxes = [box]

    # YOLO for T-pose only (exactly like december_online_minimal)
    model_yolo = YOLO(args.yolo)
    model_yolo2 = YOLO(args.yolo)

    # ---- source: LIVE or MP4 ----
    reader = None
    cap = None
    fps_meta = args.fps

    if args.live:
        reader = GstShmReader(args.shm_socket, w=args.frame_width, h=args.frame_height, fps=args.fps)
    else:
        if not args.input_path:
            raise ValueError("You must provide -i/--input when not using --live.")
        cap = cv2.VideoCapture(str(Path(args.input_path)))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {args.input_path}")
        fps_meta = cap.get(cv2.CAP_PROP_FPS)
        if fps_meta <= 0 or np.isnan(fps_meta):
            fps_meta = args.fps
        print(f"[MAIN] Offline mode: OpenCV decode. Input FPS ~ {fps_meta:.3f}")

    # ---- writers (november-like) ----
    if args.no_video:
        equi_writer = None
        pseudo_writer = None
        yolo_writer = None
    else:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        #equi_writer = cv2.VideoWriter(
            #str(output_main), fourcc, args.fps, (args.frame_width, args.frame_height)
        #)
        pseudo_writer = cv2.VideoWriter(str(out_pseudo), fourcc, args.fps, (384, 384))
        yolo_writer = cv2.VideoWriter(
            str(out_tracked), fourcc, args.fps, (args.frame_width, args.frame_height)
        )

    # ---- state (matches december_online_minimal) ----
    count = 0
    county = 0

    t_pose_threshold = 15
    t_pose_person1 = None
    t_pose_duration1 = {}
    t_pose_person2 = None
    t_pose_duration2 = {}
    tpose_found = False
    initialised = False
    t_pose_person = None
    image_id = 0  # 0 original, 1 rotated 180

    rotate_frame = 0.0
    pitch = 0.0

    depth_filtered = None
    fov_x = 40.0
    ref_depth = None
    ref_fov = None
    torso_indices = [3, 4, 5, 19]

    # TRC buffers
    vertices_list = []
    R_list = []
    timestamps_s = []
    t0 = None

    if args.skip_tpose:
        print("[INFO] --skip-tpose enabled: directly entering NLF tracking.")
        tpose_found = True
        initialised = True

    try:
        while count < args.max_frames:
            # ---- read frame ----
            if args.live:
                ok, frame_equi = reader.read()
                if not ok or frame_equi is None:
                    continue
            else:
                ok, frame_equi = cap.read()
                if not ok or frame_equi is None:
                    break

            count += 1
            t_start = time.time()

            # ---- timestamp ----
            if args.live:
                t = (count - 1) / fps_meta
            else:
                t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                if t_ms is None or t_ms <= 0 or np.isnan(t_ms):
                    t = (count - 1) / fps_meta
                else:
                    t = t_ms / 1000.0
            if t0 is None:
                t0 = t
            t_norm = t - t0

            # ---- preview ----
            if args.display:
                cv2.imshow("equi", cv2.resize(frame_equi, (960, 480)))
                if (cv2.waitKey(1) & 0xFF) == 27:
                    break

            # Apply yaw rotation
            frame = rotate_image(frame_equi, rotate_frame)

            # ---------- PHASE 1: T-pose search (YOLO only) ----------
            if not tpose_found:
                shifted_180 = rotate_image(frame, 180)

                result1 = model_yolo.track(
                    frame, persist=True, device=device, tracker=args.tracker, verbose=False
                )[0]
                result2 = model_yolo2.track(
                    shifted_180, persist=True, device=device, tracker=args.tracker, verbose=False
                )[0]

                if yolo_writer is not None:
                    yolo_vis = result1.plot()
                    if yolo_vis.shape[:2] != (args.frame_height, args.frame_width):
                        yolo_vis = cv2.resize(yolo_vis, (args.frame_width, args.frame_height))
                    yolo_writer.write(yolo_vis)

                tpose_found1, t_pose_person1, t_pose_threshold, t_pose_duration1 = \
                    detect_t_pose(result1, t_pose_person1, t_pose_threshold, t_pose_duration1)
                tpose_found2, t_pose_person2, t_pose_threshold, t_pose_duration2 = \
                    detect_t_pose(result2, t_pose_person2, t_pose_threshold, t_pose_duration2)

                if tpose_found1:
                    tpose_found = True
                    t_pose_person = t_pose_person1
                    image_id = 0
                    print(f"[TPOSE] Found on original frame at count={count}")
                elif tpose_found2:
                    tpose_found = True
                    t_pose_person = t_pose_person2
                    image_id = 1
                    print(f"[TPOSE] Found on 180° rotated frame at count={count}")

                #if equi_writer is not None:
                    #equi_writer.write(np.clip(frame_equi, 0, 255).astype(np.uint8))
                continue

            # ---------- PHASE 2: One-shot alignment ----------
            if tpose_found and not initialised:
                shifted_180 = rotate_image(frame, 180)

                if image_id == 0:
                    result1 = model_yolo.track(
                        frame, persist=True, device=device, tracker=args.tracker, verbose=False
                    )[0]
                    initialised, rotate_frame, pitch = align_equi_to_tposed(result1, t_pose_person, frame_width=args.frame_width, frame_height=args.frame_height)
                    print(f"[ALIGN] image_id=0 rotate={rotate_frame:.2f} pitch={pitch:.2f}")

                    if yolo_writer is not None:
                        yolo_vis = result1.plot()
                        if yolo_vis.shape[:2] != (args.frame_height, args.frame_width):
                            yolo_vis = cv2.resize(yolo_vis, (args.frame_width, args.frame_height))
                        yolo_writer.write(yolo_vis)
                else:
                    result2 = model_yolo2.track(
                        shifted_180, persist=True, device=device, tracker=args.tracker, verbose=False
                    )[0]
                    initialised, rotate_frame, pitch = align_equi_to_tposed(result2, t_pose_person, frame_width=args.frame_width, frame_height=args.frame_height)
                    rotate_frame += 180.0
                    print(f"[ALIGN] image_id=1 rotate={rotate_frame:.2f} pitch={pitch:.2f}")

                    if yolo_writer is not None:
                        yolo_vis = result2.plot()
                        if yolo_vis.shape[:2] != (args.frame_height, args.frame_width):
                            yolo_vis = cv2.resize(yolo_vis, (args.frame_width, args.frame_height))
                        yolo_writer.write(yolo_vis)

                rotate_frame = (rotate_frame + 180.0) % 360.0 - 180.0
                if initialised:
                    print("[ALIGN] Initialisation complete, entering NLF-only tracking.")
                else:
                    print("[ALIGN] Still not initialised, continuing...")

                #if equi_writer is not None:
                    #equi_writer.write(np.clip(frame_equi, 0, 255).astype(np.uint8))
                continue

            # ---------- PHASE 3: NLF-only tracking (NO YOLO) ----------
            county += 1

            # 1) pseudo camera (IMPORTANT: run EquiLib on SAME device -> same behavior as minimal)
            equi_tensor = preprocess(frame_equi, device)
            with torch.inference_mode():
                pseudo_camera = equi2pers_run(
                    equi=equi_tensor,
                    rots=[{
                        "yaw": float(np.deg2rad(-rotate_frame)),
                        "roll": 0.0,
                        "pitch": float(np.deg2rad(-pitch)),
                    }],
                    height=384,
                    width=384,
                    fov_x=float(fov_x),
                    skew=0.0,
                    z_down=False,
                    mode="bilinear",
                )

            # pseudo_camera already on device, just ensure float32
            input_tensor = pseudo_camera.to(dtype=torch.float32)

            K, R, t_cam = compute_camera_parameters(
                input_tensor.size(2),
                input_tensor.size(3),
                float(fov_x),
                rotate_frame,
                pitch,
            )

            R_np = np.asarray(R, dtype=np.float32)

            with torch.inference_mode():
                pred = model.estimate_poses_batched(
                    input_tensor,
                    boxes,
                    weights_subset,
                    intrinsic_matrix=None,
                    distortion_coeffs=None,
                    extrinsic_matrix=None,
                    world_up_vector=None,
                    default_fov_degrees=float(fov_x),
                    internal_batch_size=1,
                    antialias_factor=1,
                    num_aug=1,
                    rot_aug_max_degrees=0.0,
                )

            poses3d = pred["poses3d"][0]
            vertices3d = poses3d.cpu().numpy()[0]  # (20,3) mm

            pose2d = pred["poses2d"][0].cpu().numpy()
            if pose2d.ndim == 3:
                pose2d = pose2d[0]

            # 3) Orientation update (same)
            delta_yaw, delta_pitch = estimate_camera_offset_from_shoulder_2d(pose2d, idx_shoulder=13)
            alpha_yaw_pitch = 0.30
            rotate_frame += alpha_yaw_pitch * delta_yaw
            pitch += alpha_yaw_pitch * delta_pitch
            pitch = float(np.clip(pitch, -80.0, 80.0))
            rotate_frame = float((rotate_frame + 180.0) % 360.0 - 180.0)

            # 4) Depth-based FOV update (EXACT LOGIC)
            raw_depth = compute_torso_depth(vertices3d, torso_indices)

            if raw_depth is not None:
                if depth_filtered is None:
                    depth_filtered = raw_depth
                else:
                    alpha = 0.15
                    depth_filtered = depth_filtered * (1 - alpha) + raw_depth * alpha
            else:
                depth_filtered = None

            torso_depth = depth_filtered

            if depth_filtered is not None and ref_depth is not None:
                # Reject implausible jumps without disabling real close/far motion:
                # use a bounded ratio window instead of snapping to ref_depth.
                rel_depth = depth_filtered / max(ref_depth, 1e-6)
                rel_depth = float(np.clip(rel_depth, 0.35, 2.0))
                torso_depth = float(ref_depth * rel_depth)

            IDX_MC5 = 17
            IDX_CLAG = 2

            if county > 3 and torso_depth is not None and ref_depth is None:
                ref_depth = float(torso_depth)
                try:
                    p_mc5 = vertices3d[IDX_MC5]
                    p_clag = vertices3d[IDX_CLAG]
                    p_mc5_m = p_mc5 * 1e-3
                    p_clag_m = p_clag * 1e-3
                    anatomical_fov = compute_fov_from_markers(
                        p_left=p_mc5_m,
                        p_right=p_clag_m,
                        safety_margin=1.05,
                        fov_min=12.0,
                        fov_max=75.0
                    )
                    if anatomical_fov is not None:
                        ref_fov = float(anatomical_fov)
                        fov_x = float(anatomical_fov)
                        print(f"[INIT-FOV] MC5–CLAG anatomical calibration → FOV={anatomical_fov:.2f}°")
                    else:
                        ref_fov = float(fov_x)
                        print("[INIT-FOV] WARNING: MC5/CLAG invalid → fallback initial FOV.")
                except Exception as e:
                    ref_fov = float(fov_x)
                    print(f"[INIT-FOV] ERROR in anatomical FOV computation: {e}")

                print(f"[DEPTH-FOV] Calibrated ref_depth = {ref_depth:.3f} m using anatomical FOV")

            if ref_depth is not None and ref_fov is not None and torso_depth is not None:
                old_fov = fov_x
                fov_x = update_fov_from_depth_nlf(
                    depth=torso_depth,
                    ref_depth=ref_depth,
                    ref_fov=ref_fov,
                    current_fov=fov_x,
                    frame_idx=county,
                )
                print(
                    f"[DEPTH-FOV] depth={torso_depth:.3f} m  ref={ref_depth:.3f} m  "
                    f"FOV_calib={ref_fov:.1f}°  fov={old_fov:.1f}→{fov_x:.1f}"
                )

            old_fov_frame = fov_x
            fov_x = update_fov_from_pose2d_framing(
                pose2d=pose2d,
                current_fov=fov_x,
                frame_w=input_tensor.size(3),
                frame_h=input_tensor.size(2),
            )
            if fov_x > old_fov_frame + 1e-3:
                print(f"[FRAME-FOV] landmarks near border -> fov={old_fov_frame:.1f}→{fov_x:.1f}")

            # ---------- Videos ----------
            pseudo_frame = None
            if pseudo_writer is not None or args.display:
                pseudo_frame = pseudo_camera.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                pseudo_frame = (pseudo_frame * 255).astype(np.uint8).copy()
                for pt in pose2d:
                    if not np.all(np.isfinite(pt)):
                        continue
                    x, y = int(pt[0]), int(pt[1])
                    if 0 <= x < 384 and 0 <= y < 384:
                        cv2.circle(pseudo_frame, (x, y), 2, (0, 0, 255), -1

                        )

            if pseudo_writer is not None and pseudo_frame is not None:
                pseudo_writer.write(pseudo_frame)

            #if equi_writer is not None:
                #equi_writer.write(np.clip(frame_equi, 0, 255).astype(np.uint8))

            # ---------- TRC buffers ----------
            timestamps_s.append(float(t_norm))
            vertices_list.append(vertices3d.tolist())
            R_list.append(R_np.tolist())

            # ---------- RTOSIM stream ----------
            if sock is not None:
                try:
                    vertices_m = vertices3d / 1000.0
                    V_cam = np.asarray(vertices_m, dtype=np.float32)
                    V_world = V_cam @ R_np.T
                    V_world = np.asarray(
                        [(-z, -y + 0.75, -x) for (x, y, z) in V_world],
                        dtype=np.float32,
                    )
                    sock.sendall(V_world.astype(np.float32).ravel(order="C").tobytes())
                except Exception as e:
                    print(f"[Socket] WARNING: failed to send frame {county}: {e}")

            if args.display and pseudo_frame is not None:
                cv2.imshow("pseudo", cv2.resize(pseudo_frame, (512, 512)))
                if (cv2.waitKey(1) & 0xFF) == 27:
                    break

            # dt = time.time() - t_start
            # fps_inst = (1.0 / dt) if dt > 0 else 0.0

    finally:
        if cap is not None:
            cap.release()
        if reader is not None:
            reader.close()
        #if equi_writer is not None:
            #equi_writer.release()
        if pseudo_writer is not None:
            pseudo_writer.release()
        if yolo_writer is not None:
            yolo_writer.release()
        if args.display:
            cv2.destroyAllWindows()
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            try:
                sock.close()
            except Exception:
                pass

    print(f"[MAIN] Processed frames: {count}")
    print(f"[MAIN] Valid NLF frames for TRC: {len(vertices_list)}")

    # ---- TRC writing (november-like) ----
    if len(vertices_list) == 0:
        print("[TRC] Skipped: no vertices collected (likely no T-pose / no NLF frames).")
        return

    # For offline MP4: optionally rebuild timestamps using ffprobe uniform spacing
    if (not args.live) and args.trc_use_ffprobe:
        try:
            _, _, fps_real = get_video_fps_and_duration(str(Path(args.input_path)))
            num_frames = len(vertices_list)
            timestamps_s = [i / fps_real for i in range(num_frames)]
            print("[TRC] Using ffprobe-derived fps for uniform timestamps.")
        except Exception as e:
            print(f"[TRC] ffprobe failed ({e}); keeping OpenCV timestamps.")

    print("[TRC] Writing TRC world frame ...")
    generate_trc_file_world_frame(
        vertices_list,
        cfg_bio,
        out_trc,
        timestamps_s[:len(vertices_list)],
        R_list,
        units="mm",
        world_offset_y=0.75
    )
    print(f"[TRC] Done → {out_trc}")


if __name__ == "__main__":
    main()
