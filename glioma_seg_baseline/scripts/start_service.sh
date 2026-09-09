#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export PYTHONPATH="$ROOT_DIR/src:${PYTHONPATH:-}"
export ANSWER_BASE="${ANSWER_BASE:-/2026aicompetition/workspace/answer}"
export CORE_DATASET_ID="${CORE_DATASET_ID:-401}"
export TOTAL_DATASET_ID="${TOTAL_DATASET_ID:-402}"

python -m glioma_baseline.service --host 0.0.0.0 --port 8000
