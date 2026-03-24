#!/usr/bin/env bash
set -euo pipefail

ROOT="/media/mathis/PANO/gst/analyse/data/repaired_videos"

find "$ROOT" -type f -name "sample_*.mp4" | while read -r video; do
    dir="$(dirname "$video")"
    base="$(basename "$video" .mp4)"
    out="$dir/${base}.trc"

    echo "Processing: $video"
    echo " -> $out"

    uv run february_offline.py \
        -i "$video" \
        -o "$out" \
        --no-video
done
