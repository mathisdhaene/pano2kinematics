import shlex
import subprocess
from pathlib import Path

from pipeline_nlf.live.paths import derive_side_outputs, ensure_parent_dir


def derive_side_outputs_with_hand(main_out: Path):
    stem = main_out.with_suffix("")
    out_tracked = stem.with_name(stem.name + "_tracked").with_suffix(".mp4")
    out_pseudo = stem.with_name(stem.name + "_pseudo").with_suffix(".mp4")
    out_hand = stem.with_name(stem.name + "_hand").with_suffix(".mp4")
    out_trc = stem.with_suffix(".trc")
    return out_tracked, out_pseudo, out_hand, out_trc


def get_video_fps_and_duration(video_path: str):
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


def pick_shm_paths(base: str):
    yield base
    for i in range(10):
        yield f"{base}.{i}"
