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
import time
from collections import defaultdict

import socket

# Socket Setup
HOST = '127.0.0.1'
PORT = 5555
sock = None
try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((HOST, PORT))
    print(f"[Socket] Connected -> RTOSIM {HOST}:{PORT}")
except Exception as e:
    print(f"[Socket] WARNING: could not connect to RTOSIM: {e}")
    sock = None

import torch
import numpy as np
import cv2
from ultralytics import YOLO
import yaml

import torch._dynamo
from pipeline_nlf.live import (
    GstShmReader,
    YOLOTrackedRecorder,
    derive_side_outputs,
    ensure_parent_dir,
    project_mesh_from_pseudo_to_equi,
)
from pipeline_nlf.utils import (
    align_equi_to_tposed,
    compute_camera_parameters,
    detect_t_pose,
    generate_trc_file_world_frame,
    main_tracking,
    preprocess,
    preprocess_for_nlf,
    rotate_image,
    start_tracking,
    unNormalize,
)
try:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
except Exception:
    Axes3D = None
from equilib.equi2pers.torch_impl import run as equi2pers_run
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
    p.add_argument("--max-frames", type=int, default=10000, help="Limiter le nb de frames")
    p.add_argument("--yolo", default="yolo_models/yolo11x-pose.pt",
                   help="Chemin du modèle YOLO pose")
    p.add_argument("--tracker", default="bytetrack.yaml", help="Tracker pour YOLO")
    p.add_argument("--bio-cfg", default="configs/biomeca.yaml", help="YAML biomécanique")
    p.add_argument("--nlf-weights", default="weights/nlf/nlf_s_multi.torchscript",
                   help="Chemin du modèle NLF TorchScript")

    # NEW: live mode
    p.add_argument("--live", action="store_true", help="Read from shmsrc via GstShmReader")
    p.add_argument("--shm-socket", default="/tmp/theta_bgr.sock", help="Base path to shmsink socket")
    p.add_argument("--frame-width", type=int, default=3840, help="Equirectangular frame width")
    p.add_argument("--frame-height", type=int, default=1920, help="Equirectangular frame height")
    p.add_argument("--display-only", action="store_true",
               help="Do not write videos/TRC; show ori_img live")
    p.add_argument("--no-video", action="store_true",
               help="Disable all video writing outputs")

    return p


# -------------------- MAIN (identical logic; only source differs) --------------------

def main():
    args = build_parser().parse_args()
    if args.display_only:
        cv2.namedWindow("pseudo_camera", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("pseudo_camera", 512, 512)

    display_only = args.display_only
    write_video = not args.display_only and not args.no_video

    input_path = Path(args.input_path)
    output_main = Path(args.output_path)

    # Side outputs
    out_tracked, out_pseudo, out_trc = derive_side_outputs(output_main)

    # Dirs
    ensure_parent_dir(output_main)
    ensure_parent_dir(out_tracked)
    ensure_parent_dir(out_pseudo)
    ensure_parent_dir(out_trc)

    with open(args.bio_cfg, "r") as f:
        cfg_bio = yaml.safe_load(f)



    # Register torchvision custom ops BEFORE loading TS (fixes torchvision::nms)
    import torchvision
    _ = torchvision.ops.nms

    model = torch.jit.load(args.nlf_weights, map_location=args.device).to(args.device).eval()



    device = args.device
    model_yolo = YOLO(args.yolo)
    model_yolo2 = YOLO(args.yolo)

    # ---- Video source ----
    reader = None
    cap = None
    if args.live:
        reader = GstShmReader(args.shm_socket, w=args.frame_width, h=args.frame_height, fps=args.fps)
    else:
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Impossible d’ouvrir la vidéo d’entrée: {input_path}")

    # Writers (disabled in display-only or --no-video mode)
    if write_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_main), fourcc, args.fps, (args.frame_width, args.frame_height))
        pseudo_out = cv2.VideoWriter(str(out_pseudo), fourcc, args.fps, (256, 256))
    else:
        out = None
        pseudo_out = None

    rec = None
    if write_video:
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
    timestamps_s = []

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

                # In display-only mode we only show the pseudo-camera preview.

            elif tpose_found and not initialisation:
                shifted_image_180 = rotate_image(frame, 180)
                if image_id == 0:
                    if rec is not None:
                        rec.process(frame)

                    result1 = model_yolo.track(frame, persist=True, device=device, tracker=args.tracker, save=False)[0]
                    initialisation, rotate_frame, pitch = align_equi_to_tposed(
                        result1, t_pose_person, args.frame_width, args.frame_height
                    )
                elif image_id == 1:
                    result2 = model_yolo2.track(shifted_image_180, persist=True, device=device, tracker=args.tracker, save=False)[0]
                    if rec is not None:
                        rec.process(frame)

                    initialisation, rotate_frame, pitch = align_equi_to_tposed(
                        result2, t_pose_person, args.frame_width, args.frame_height
                    )
                    rotate_frame += 180

            elif tpose_found and initialisation and id_a_suivre is None:
                result = model_yolo.track(frame, persist=True, device=device, tracker=args.tracker, save=False)[0]
                id_a_suivre = start_tracking(result, id_a_suivre)

            elif tpose_found and initialisation and id_a_suivre is not None:
                track_start = time.time()
                county += 1

                result = model_yolo.track(frame, persist=True, device=device, tracker=args.tracker, show=False)[0]
                if rec is not None:
                    rec.process(frame)

                track_end = time.time()
                rotate_frame, pitch = main_tracking(
                    result, id_a_suivre, rotate_frame, pitch, args.frame_width, args.frame_height
                )

                temps_avant_preprocess = time.time()
                preprocessed_equi_rendering = preprocess(framee, device)
                temps_apres_preprocess = time.time()

                torch.cuda.synchronize() if "cuda" in device else None
                start_camera = time.time()
                FOV = 55.0
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
                        fov_x=FOV,
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
                        FOV,
                        rotate_frame,
                        pitch
                    )
                print('rotate_frame', rotate_frame)

                torch.cuda.synchronize() if "cuda" in device else None
                inference_start = time.time()
                with torch.inference_mode():
                    pred = model.estimate_poses_batched(
                        input_tensor,
                        precomputed_box,
                        weights_subset,
                        default_fov_degrees=55,
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



                if poses3d.shape[0] > 0 and not display_only:
                    projected_2d = project_mesh_from_pseudo_to_equi(
                        vertices3d, rotate_frame, pitch, args.frame_width, args.frame_height
                    )
                    projected_2d = np.array(projected_2d)
                    unnormalized_tensor = unNormalize(input_tensor).cpu()
                    ori_img = preprocessed_equi_rendering.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    ori_img = np.clip(ori_img * 255.0, 0, 255).astype(np.uint8)
                    vertices_list.append(vertices3d.tolist())
                    R_list.append(R.tolist())
                    if args.live or cap is None:
                        t_norm = float(len(timestamps_s)) / float(args.fps)
                    else:
                        t_norm = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    timestamps_s.append(float(t_norm))
                    for point in projected_2d:
                        center = tuple(int(x) for x in np.round(point).tolist())
                        cv2.circle(ori_img, center, 5, (0, 0, 255), -1)

                # Do not render/show equirectangular preview in display-only mode.
                
# assumes vertices3d already in meters; if it's in millimeters, see below
                vertices3d = vertices3d  / 1000.0
                V_trans = [(x,y,z) for (x, y, z) in vertices3d]
                V_cam = np.asarray(V_trans, dtype=np.float32) 
                R = np.asarray(R, dtype=np.float32)        # shape (3,3)

                V_world = V_cam @ R.T 
                print('end')
                V_world = [(-z, -y + 0.75, -x) for (x, y, z) in V_world]
                V_world = np.asarray(V_world, dtype=np.float32)
                if sock is not None:
                    try:
                        sock.sendall(V_world.astype(np.float32).ravel(order="C").tobytes())
                        print(f"Sent frame {county} marker data to RTOSIM")
                    except Exception as e:
                        print(f"[Socket] WARNING: failed to send frame {county}: {e}")

                total_end_time = time.time()
                total_duration = total_end_time - total_start_time
                fps = 1 / total_duration if total_duration > 0 else 0
                print(count)

        if rec is not None:
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
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            try:
                sock.close()
            except Exception:
                pass
        cv2.destroyAllWindows()


    print('county', county)
    # TRC à côté de la sortie principale
    if len(vertices_list) > 0:
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
        print(f"[TRC] Done -> {out_trc}")
    else:
        print("[TRC] Skipped: no vertices were produced.")

if __name__ == "__main__":
    main()
