from __future__ import annotations

import copy
import os
from pathlib import Path


DEFAULT_CONFIG = {
    "paths": {
        "annotation_root": "/2026aicompetition/datasets/training/annotation",
        "label_root": "/2026aicompetition/datasets/training/annotation",
        "nnunet_src": "/2026aicompetition/public_models/MIC-DKFZ/nnUNet",
        "nnunet_raw": "/2026aicompetition/workspace/nnUNet_raw",
        "nnunet_preprocessed": "/2026aicompetition/workspace/nnUNet_preprocessed",
        "nnunet_results": "/2026aicompetition/workspace/nnUNet_results",
        "answer_root": "/2026aicompetition/workspace/answer",
        "log_dir": "/2026aicompetition/workspace/logs",
    },
    "service": {
        "host": "0.0.0.0",
        "port": 8000,
        "callback_url": "",
    },
    "nnunet": {
        "core_dataset_id": 401,
        "total_dataset_id": 402,
        "trainer": "nnUNetTrainer",
        "configuration": "3d_fullres",
        "fold": "0",
        "checkpoint": "checkpoint_final.pth",
    },
    "inference": {
        "dataset_root": "/2026aicompetition/datasets/verification/original",
        "validate_nifti": False,
        "fallback_to_zero_mask": True,
        "duplicate_topk": 20,
    },
}


def _deep_update(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _coerce_scalar(value: str):
    value = value.strip()
    if value in ("", "''", '""'):
        return ""
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    return value.strip("'\"")


def _load_simple_yaml(path: Path) -> dict:
    """Parse the small two-level config used by this baseline without PyYAML."""
    data: dict = {}
    current: str | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not line.startswith(" ") and line.endswith(":"):
            current = line[:-1].strip()
            data[current] = {}
            continue
        if current and line.startswith("  ") and ":" in line:
            key, value = line.strip().split(":", 1)
            data[current][key.strip()] = _coerce_scalar(value)
    return data


def load_config(path: str | Path | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path:
        p = Path(path)
        if p.exists():
            try:
                import yaml
                data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            except Exception:
                data = _load_simple_yaml(p)
            _deep_update(cfg, data)

    cfg["paths"]["answer_root"] = os.environ.get("ANSWER_BASE", cfg["paths"]["answer_root"])
    cfg["paths"]["annotation_root"] = os.environ.get("ANNOTATION_ROOT", cfg["paths"]["annotation_root"])
    cfg["paths"]["label_root"] = os.environ.get("LABEL_ROOT", cfg["paths"]["label_root"])
    cfg["paths"]["nnunet_src"] = os.environ.get("NNUNET_SRC", cfg["paths"]["nnunet_src"])
    cfg["paths"]["nnunet_raw"] = os.environ.get("nnUNet_raw", cfg["paths"]["nnunet_raw"])
    cfg["paths"]["nnunet_preprocessed"] = os.environ.get("nnUNet_preprocessed", cfg["paths"]["nnunet_preprocessed"])
    cfg["paths"]["nnunet_results"] = os.environ.get("nnUNet_results", cfg["paths"]["nnunet_results"])
    cfg["service"]["callback_url"] = os.environ.get("CALLBACK_URL", cfg["service"]["callback_url"])
    cfg["inference"]["dataset_root"] = os.environ.get("DATASET_ROOT", cfg["inference"]["dataset_root"])
    cfg["nnunet"]["core_dataset_id"] = int(os.environ.get("CORE_DATASET_ID", cfg["nnunet"]["core_dataset_id"]))
    cfg["nnunet"]["total_dataset_id"] = int(os.environ.get("TOTAL_DATASET_ID", cfg["nnunet"]["total_dataset_id"]))
    return cfg
