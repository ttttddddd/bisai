#!/usr/bin/env bash
set -euo pipefail

export nnUNet_raw="${nnUNet_raw:-/2026aicompetition/workspace/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-/2026aicompetition/workspace/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-/2026aicompetition/workspace/nnUNet_results}"

NNUNET_SRC="${NNUNET_SRC:-/2026aicompetition/public_models/MIC-DKFZ/nnUNet}"
LOG_DIR="${LOG_DIR:-/2026aicompetition/workspace/logs}"

mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results" "$LOG_DIR"

if ! command -v nnUNetv2_plan_and_preprocess >/dev/null 2>&1; then
  if [ -d "$NNUNET_SRC" ]; then
    python -m pip install -e "$NNUNET_SRC"
  else
    echo "nnU-Net command not found and NNUNET_SRC does not exist: $NNUNET_SRC" >&2
    exit 1
  fi
fi

echo '{"timestamp":"'$(date -u +"%Y-%m-%dT%H:%M:%S.000Z")'","epoch":1,"step":0,"phase":"train","mode":"training","loss":null,"lr":null,"data_source":"brats_mock","checkpoint":null,"pretrained_from":"/2026aicompetition/public_models/MIC-DKFZ/nnUNet"}' >> "$LOG_DIR/glioma_seg_train.jsonl"

nnUNetv2_plan_and_preprocess -d 401 --verify_dataset_integrity
nnUNetv2_plan_and_preprocess -d 402 --verify_dataset_integrity

nnUNetv2_train 401 3d_fullres 0
nnUNetv2_train 402 3d_fullres 0

echo '{"timestamp":"'$(date -u +"%Y-%m-%dT%H:%M:%S.000Z")'","epoch":null,"step":null,"phase":"val","mode":"training","loss":null,"lr":null,"data_source":"brats_mock","checkpoint":"nnUNet_results"}' >> "$LOG_DIR/glioma_seg_train.jsonl"
