#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
august_parsed_live_enabled.py

This is your working `august_parsed.py` pipeline with ONE change: the video source can now be
- an MP4 file (original behavior), or
- a Ricoh Theta live stream via GStreamer shmsink using GstShmReader (`--live --shm-socket ...`).

Everything else — T‑pose search, alignment, FOLLOW PID, EquiLib pseudo‑camera, NLF inference,
reprojection, tracked/pseudo writers, TRC export — is preserved.

Run (live):
  python3 august_parsed_live_enabled.py \
    --live --shm-socket /tmp/theta_bgr.sock \
    --fps 30 --device cuda:0 \
    --yolo weights/yolo_models/yolo11m-pose.pt \
    --tracker bytetrack.yaml \
    --bio-cfg configs/biomeca.yaml \
    --nlf-weights weights/nlf/nlf_s_multi.torchscript \
    -o output_nlf/markerless_live.mp4

Run (offline, unchanged):
  python3 august_parsed_live_enabled.py \
    -i /path/to/video.mp4 -o output_nlf/markerless_1.mp4 \
    --fps 30 --device cuda:0 \
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
from pipeline_nlf.utils import *
from mpl_toolkits.mplot3d import Axes3D  # noqa
from equilib.equi2pers.torch_impl import run as equi2pers_run
from equilib.numpy_utils.rotation import create_rotation_matrices  # noqa
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
    p.add_argument("--device", default="cuda:0", help="cuda:0 / cpu")
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

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

import math

def _pick_shm_paths(base: str):
    yield base
    for i in range(10):
        yield f"{base}.{i}"

class GstShmReader:
    """Minimal, robust shmsrc → appsink BGR reader."""
    def __init__(self, sock_base, w, h, fps, timeout_ms=3000):
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

                torch.cuda.synchronize() if "cuda" in device else None
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
                    pseudo_camera = equi2pers_run(
                        equi=preprocessed_equi_rendering,
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

                torch.cuda.synchronize() if "cuda" in device else None

                temps_apres_transfo = time.time()
                input_tensor = preprocess_for_nlf(pseudo_camera)

                K, R, t = compute_camera_parameters(
                        input_tensor.size(2),
                        input_tensor.size(3),
                        FOV,            # <---- USE dynamic FOV
                        rotate_frame,
                        -pitch
                    )

                R_list.append(R.tolist())

                torch.cuda.synchronize() if "cuda" in device else None
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
                torch.cuda.synchronize() if "cuda" in device else None

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
