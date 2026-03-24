#!/usr/bin/env bash
set -euo pipefail

ROOT="/media/mathis/PANO/gst/analyse/data/repaired_videos"
JOBS=4
MIN_FREE_MIB=5000

gpu_free_mib() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d ' '
}

wait_for_free_gpu() {
  while true; do
    free="$(gpu_free_mib)"
    if [ "$free" -ge "$MIN_FREE_MIB" ]; then
      return 0
    fi
    sleep 1
  done
}

export -f gpu_free_mib wait_for_free_gpu
export MIN_FREE_MIB

find "$ROOT" -type f -name "sample_*.mp4" -print0 | \
xargs -0 -n 1 -P "$JOBS" bash -lc '
  video="$1"
  dir="$(dirname "$video")"
  base="$(basename "$video" .mp4)"
  out="$dir/SII_${base}.trc"

  if [ -e "$out" ]; then
    echo "PID=$$ SKIP existing output: $out"
    exit 0
  fi

  wait_for_free_gpu
  echo "PID=$$ free=$(gpu_free_mib)MiB: $video -> $out"

  uv run live_gpu.py -i "$video" -o "$out" --no-video
' _ 
