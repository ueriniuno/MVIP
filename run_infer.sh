#!/usr/bin/env bash
# MVIP inference on one video.
# Usage: bash run_infer.sh [video.mp4] [output_dir] [extra scripts/infer.py args]
set -euo pipefail
cd "$(dirname "$0")"
VIDEO="${1:-test_videos/horsejump_high.mp4}"
OUT="${2:-outputs/infer_$(basename "${VIDEO%.*}")}"
shift $(( $# > 2 ? 2 : $# ))
python scripts/infer.py --input-video "$VIDEO" --output-dir "$OUT" "$@"
