#!/usr/bin/env bash
# Evaluation: generate a video for every clip in the manifest, then compute the metrics.
# Usage: bash run_eval.sh <data_root> [output_dir] [extra scripts/infer.py args, e.g. --baseline captioning]
set -euo pipefail
cd "$(dirname "$0")"
DATA_ROOT="${1:?usage: bash run_eval.sh <data_root> [output_dir] [extra args]}"
OUT="${2:-outputs/eval}"
shift $(( $# > 2 ? 2 : $# ))
python scripts/infer.py --data-root "$DATA_ROOT" --output-dir "$OUT" "$@"
python eval/evaluate_metrics.py --output-root "$OUT"
