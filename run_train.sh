#!/usr/bin/env bash
# Paper training: 100 outer steps of the released schedule.
# Usage: bash run_train.sh <data_root> [output_dir] [extra scripts/train.py args, e.g. --resume]
set -euo pipefail
cd "$(dirname "$0")"
DATA_ROOT="${1:?usage: bash run_train.sh <data_root> [output_dir] [extra args]}"
OUT="${2:-outputs/train}"
shift $(( $# > 2 ? 2 : $# ))
python scripts/train.py --data-root "$DATA_ROOT" --output-dir "$OUT" "$@"
