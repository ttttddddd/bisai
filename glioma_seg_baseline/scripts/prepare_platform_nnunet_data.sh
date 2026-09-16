#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export ANNOTATION_ROOT="${ANNOTATION_ROOT:-/2026aicompetition/datasets/training/annotation}"
export LABEL_ROOT="${LABEL_ROOT:-/2026aicompetition/datasets/training/label}"
export nnUNet_raw="${nnUNet_raw:-/2026aicompetition/workspace/nnUNet_raw}"

MAX_CASES="${MAX_CASES:-0}"
EXTRA_ARGS=()
if [ "${INCLUDE_EMPTY:-0}" = "1" ]; then
  EXTRA_ARGS+=("--include-empty")
fi
if [ "${VALIDATE_NIFTI:-0}" = "1" ]; then
  EXTRA_ARGS+=("--validate-nifti")
fi

python "$ROOT_DIR/tools/platform_to_nnunet.py" \
  --annotation-root "$ANNOTATION_ROOT" \
  --label-root "$LABEL_ROOT" \
  --out-root "$nnUNet_raw" \
  --max-cases "$MAX_CASES" \
  "${EXTRA_ARGS[@]}"
