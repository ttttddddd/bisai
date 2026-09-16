#!/usr/bin/env bash
set -euo pipefail

export nnUNet_raw="${nnUNet_raw:-/2026aicompetition/workspace/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-/2026aicompetition/workspace/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-/2026aicompetition/workspace/nnUNet_results}"
export ANNOTATION_ROOT="${ANNOTATION_ROOT:-/2026aicompetition/datasets/training/annotation}"
export LABEL_ROOT="${LABEL_ROOT:-/2026aicompetition/datasets/training/label}"

NNUNET_SRC="${NNUNET_SRC:-/2026aicompetition/public_models/MIC-DKFZ/nnUNet}"
LOG_DIR="${LOG_DIR:-/2026aicompetition/workspace/logs}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results" "$LOG_DIR"

if ! command -v nnUNetv2_plan_and_preprocess >/dev/null 2>&1; then
  if [ -d "$NNUNET_SRC" ]; then
    python -m pip install -e "$NNUNET_SRC"
  else
    echo "nnU-Net command not found and NNUNET_SRC does not exist: $NNUNET_SRC" >&2
    exit 1
  fi
fi

if [ ! -d "$nnUNet_raw/Dataset401_GliomaCoreT1CE" ] || [ ! -d "$nnUNet_raw/Dataset402_GliomaTotalFLAIR" ]; then
  bash "$ROOT_DIR/scripts/prepare_platform_nnunet_data.sh"
fi

echo '{"timestamp":"'$(date -u +"%Y-%m-%dT%H:%M:%S.000Z")'","epoch":1,"step":0,"phase":"train","mode":"training","loss":null,"lr":null,"data_source":"platform_annotation","checkpoint":null,"pretrained_from":"'$NNUNET_SRC'"}' >> "$LOG_DIR/glioma_seg_train.jsonl"

nnUNetv2_plan_and_preprocess -d 401 --verify_dataset_integrity
nnUNetv2_plan_and_preprocess -d 402 --verify_dataset_integrity

nnUNetv2_train 401 3d_fullres 0
nnUNetv2_train 402 3d_fullres 0

echo '{"timestamp":"'$(date -u +"%Y-%m-%dT%H:%M:%S.000Z")'","epoch":null,"step":null,"phase":"val","mode":"training","loss":null,"lr":null,"data_source":"platform_annotation","checkpoint":"nnUNet_results"}' >> "$LOG_DIR/glioma_seg_train.jsonl"
