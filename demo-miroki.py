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

import numpy as np
import torch
import cv2
from ultralytics import YOLO
import yaml
import torchvision
import torch._dynamo

from pipeline_nlf.live import GstShmReader
from pipeline_nlf.adaptive_fov import estimate_camera_offset_from_pose2d
from pipeline_nlf.offline import (
    compute_fov_from_markers,
    compute_torso_depth,
    derive_side_outputs,
    ensure_parent_dir,
    evaluate_nlf_track_quality,
    get_video_fps_and_duration,
    pick_best_nlf_candidate,
    update_fov_from_depth_nlf,
    update_fov_from_pose2d_framing,
)
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
    p.add_argument(
        "--rtosim-yaw-port",
        type=int,
        default=5556,
        help="Optional separate TCP port to stream yaw only (float32 per frame), preserving legacy vertices payload.",
    )

    return p


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
    yaw_sock = None
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((args.rtosim_host, args.rtosim_port))
        print(f"[Socket] Connected → RTOSIM {args.rtosim_host}:{args.rtosim_port}")
    except Exception as e:
        print(f"[Socket] WARNING: could not connect to RTOSIM: {e}")
        sock = None

    if args.rtosim_yaw_port is not None:
        try:
            yaw_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            yaw_sock.connect((args.rtosim_host, args.rtosim_yaw_port))
            print(f"[Socket] Connected → YAW {args.rtosim_host}:{args.rtosim_yaw_port}")
        except Exception as e:
            print(f"[Socket] WARNING: could not connect to YAW socket: {e}")
            yaw_sock = None

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

    t_pose_threshold = 5
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

    # NLF-only robust tracking state (single subject)
    track_state = "TRACK"  # TRACK -> HOLD -> RECOVER
    # Uncertainty-first hysteresis:
    # - keep threshold while already tracking
    # - stricter threshold to re-enter TRACK from HOLD/RECOVER
    unc_keep_threshold = 0.60
    unc_recover_threshold = 0.45
    hold_bad_frames = 3
    recover_bad_frames = 12
    recover_good_frames = 2
    recover_unc_hard = 0.90
    enable_active_recover_scan = True
    bad_counter = 0
    good_counter = 0
    last_valid_pose2d = None
    last_valid_vertices3d = None
    last_valid_center = None
    hold_scan_speed = 0.12
    recover_scan_speed = 0.25
    recover_scan_dir = 1.0
    recover_scan_count = 0
    recover_scan_flip_every = 30
    max_step_yaw = 2.5
    max_step_pitch = 1.0

    depth_filtered = None
    fov_x = 40.0
    ref_depth = None
    ref_fov = None
    torso_indices = [3, 4, 5, 19]
    printed_vertices_shape = False

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

            poses3d_all = pred["poses3d"][0].detach().cpu().numpy()
            poses2d_all = pred["poses2d"][0].detach().cpu().numpy()
            unc_all = pred["uncertainties"][0].detach().cpu().numpy()
            if poses3d_all.ndim == 2:
                poses3d_all = poses3d_all[None, ...]
            if poses2d_all.ndim == 2:
                poses2d_all = poses2d_all[None, ...]
            if unc_all.ndim == 1:
                unc_all = unc_all[None, ...]

            picked = pick_best_nlf_candidate(poses2d_all, poses3d_all, unc_all)
            pose2d = None
            vertices3d = None
            kp_unc = None
            if picked is not None:
                pose2d, vertices3d, kp_unc = picked
                if not printed_vertices_shape and vertices3d is not None:
                    print(f"[RTOSIM] vertices3d shape={vertices3d.shape}")
                    printed_vertices_shape = True

            prev_center_eval = last_valid_center if track_state == "TRACK" else None
            unc_threshold_eval = unc_keep_threshold if track_state == "TRACK" else unc_recover_threshold
            is_good, torso_unc, center_xy, bad_reason = evaluate_nlf_track_quality(
                pose2d,
                kp_unc,
                prev_center=prev_center_eval,
                img_w=int(input_tensor.size(3)),
                img_h=int(input_tensor.size(2)),
                unc_threshold=unc_threshold_eval,
                use_jump_check=False,
            )

            if is_good:
                bad_counter = 0
                good_counter += 1
                if track_state != "TRACK" and good_counter >= recover_good_frames:
                    track_state = "TRACK"
                    print(f"[NLF-TRACK] RECOVER -> TRACK (torso_unc={torso_unc:.3f})")
                last_valid_pose2d = pose2d.copy()
                last_valid_vertices3d = vertices3d.copy()
                last_valid_center = center_xy.copy() if center_xy is not None else None
            else:
                good_counter = 0
                bad_counter += 1
                if track_state == "TRACK" and bad_counter >= hold_bad_frames:
                    track_state = "HOLD"
                    print(f"[NLF-TRACK] TRACK -> HOLD (reason={bad_reason}, torso_unc={torso_unc:.3f})")
                if (
                    track_state == "HOLD"
                    and bad_counter >= recover_bad_frames
                    and torso_unc >= recover_unc_hard
                    and enable_active_recover_scan
                ):
                    track_state = "RECOVER"
                    recover_scan_count = 0
                    print(f"[NLF-TRACK] HOLD -> RECOVER after {bad_counter} bad frames")

            if track_state == "HOLD":
                recover_scan_count += 1
                rotate_frame += recover_scan_dir * hold_scan_speed
                if recover_scan_count % recover_scan_flip_every == 0:
                    recover_scan_dir *= -1.0

            if track_state == "RECOVER":
                recover_scan_count += 1
                rotate_frame += recover_scan_dir * recover_scan_speed
                if recover_scan_count % recover_scan_flip_every == 0:
                    recover_scan_dir *= -1.0

            # 3) Orientation update (only on reliable NLF frames)
            if track_state == "TRACK" and pose2d is not None:
                delta_yaw, delta_pitch = estimate_camera_offset_from_pose2d(
                    pose2d,
                    img_w=int(input_tensor.size(3)),
                    img_h=int(input_tensor.size(2)),
                )
                print(rotate_frame,"yaw")
                alpha_yaw_pitch = 0.95
                step_yaw = float(np.clip(alpha_yaw_pitch * delta_yaw, -max_step_yaw, max_step_yaw))
                step_pitch = float(np.clip(alpha_yaw_pitch * delta_pitch, -max_step_pitch, max_step_pitch))
                rotate_frame += step_yaw
                pitch += step_pitch

            pitch = float(np.clip(pitch, -80.0, 80.0))
            rotate_frame = float((rotate_frame + 180.0) % 360.0 - 180.0)

            # 4) Depth/FOV updates only when we trust the pose
            if track_state == "TRACK" and vertices3d is not None:
                raw_depth = compute_torso_depth(vertices3d, torso_indices)

                if raw_depth is not None:
                    if depth_filtered is None:
                        depth_filtered = raw_depth
                    else:
                        alpha = 0.15
                        depth_filtered = depth_filtered * (1 - alpha) + raw_depth * alpha

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
            elif bad_counter % 15 == 1:
                print(f"[NLF-TRACK] state={track_state} reason={bad_reason} torso_unc={torso_unc:.3f}")

            # ---------- Videos ----------
            pseudo_frame = None
            if pseudo_writer is not None or args.display:
                pseudo_frame = pseudo_camera.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
                pseudo_frame = (pseudo_frame * 255).astype(np.uint8).copy()
                pose2d_vis = pose2d if pose2d is not None else last_valid_pose2d
                if pose2d_vis is None:
                    pose2d_vis = np.empty((0, 2), dtype=np.float32)
                for pt in pose2d_vis:
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
            if track_state == "TRACK" and vertices3d is not None:
                timestamps_s.append(float(t_norm))
                vertices_list.append(vertices3d.tolist())
                R_list.append(R_np.tolist())

            # ---------- RTOSIM stream ----------
            if sock is not None and track_state == "TRACK" and vertices3d is not None:
                try:
                    vertices_m = vertices3d / 1000.0
                    V_cam = np.asarray(vertices_m, dtype=np.float32)
                    V_world = V_cam @ R_np.T
                    #V_world = np.asarray(
                        #[(-z, -y + 0.75, -x) for (x, y, z) in V_world],
                        #dtype=np.float32,
                    #)
                    payload = V_world.astype(np.float32).ravel(order="C")
                    sock.sendall(payload.tobytes())
                except Exception as e:
                    print(f"[Socket] WARNING: failed to send frame {county}: {e}")
            if yaw_sock is not None and track_state == "TRACK":
                try:
                    yaw_sock.sendall(np.array([float(rotate_frame)], dtype=np.float32).tobytes())
                except Exception as e:
                    print(f"[Socket] WARNING: failed to send yaw frame {county}: {e}")

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
        if yaw_sock is not None:
            try:
                yaw_sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            try:
                yaw_sock.close()
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
