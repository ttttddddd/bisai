#!/usr/bin/env python3
"""
Convert a small BraTS subset into two nnU-Net v2 datasets:

- Dataset401_GliomaCore: label = BraTS TC = label 1 or ET label
- Dataset402_GliomaTotal: label = BraTS WT = any non-zero label

Expected BraTS case layout, for example:
  BraTS-GLI-00001-000/
    BraTS-GLI-00001-000-t1n.nii.gz
    BraTS-GLI-00001-000-t1c.nii.gz
    BraTS-GLI-00001-000-t2w.nii.gz
    BraTS-GLI-00001-000-t2f.nii.gz
    BraTS-GLI-00001-000-seg.nii.gz

Usage:
  python tools/brats_to_nnunet_mini.py \
    --brats-root /path/to/BraTS \
    --out-root /path/to/nnUNet_raw \
    --num-cases 8
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np


MODALITIES = [
    ("t1n", "T1"),
    ("t1c", "T1CE"),
    ("t2w", "T2"),
    ("t2f", "FLAIR"),
]


def find_file(case_dir: Path, suffix: str) -> Path:
    matches = sorted(case_dir.glob(f"*-{suffix}.nii.gz"))
    if not matches:
        matches = sorted(case_dir.glob(f"*_{suffix}.nii.gz"))
    if not matches:
        raise FileNotFoundError(f"Missing {suffix} image in {case_dir}")
    return matches[0]


def copy_modalities(case_dir: Path, images_tr: Path, case_id: str) -> None:
    for channel, (suffix, _name) in enumerate(MODALITIES):
        src = find_file(case_dir, suffix)
        dst = images_tr / f"{case_id}_{channel:04d}.nii.gz"
        shutil.copy2(src, dst)


def save_binary_label(seg_path: Path, out_path: Path, task: str, et_label: int) -> None:
    img = nib.load(str(seg_path))
    seg = img.get_fdata().astype(np.int16)
    if task == "core":
        mask = np.logical_or(seg == 1, seg == et_label).astype(np.uint8)
    elif task == "total":
        mask = (seg > 0).astype(np.uint8)
    else:
        raise ValueError(task)
    nib.save(nib.Nifti1Image(mask, img.affine, img.header), str(out_path))


def write_dataset_json(dataset_dir: Path, name: str, num_cases: int) -> None:
    data = {
        "channel_names": {str(i): modality for i, (_suffix, modality) in enumerate(MODALITIES)},
        "labels": {"background": 0, "foreground": 1},
        "numTraining": num_cases,
        "file_ending": ".nii.gz",
        "name": name,
        "description": "Mini BraTS-derived dataset for running through nnU-Net v2.",
        "reference": "BraTS labels mapped to competition proxy masks.",
    }
    (dataset_dir / "dataset.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def make_dataset(out_root: Path, dataset_id: int, name: str, cases: list[Path], task: str, et_label: int) -> None:
    dataset_dir = out_root / f"Dataset{dataset_id}_{name}"
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    for idx, case_dir in enumerate(cases):
        case_id = f"BRATS_{idx:03d}"
        copy_modalities(case_dir, images_tr, case_id)
        seg_path = find_file(case_dir, "seg")
        save_binary_label(seg_path, labels_tr / f"{case_id}.nii.gz", task, et_label)

    write_dataset_json(dataset_dir, name, len(cases))
    print(f"Wrote {dataset_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--brats-root", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--num-cases", type=int, default=8)
    parser.add_argument(
        "--et-label",
        type=int,
        default=4,
        help="Enhancing tumor label. Classic BraTS uses 4. Some newer converted labels may differ.",
    )
    args = parser.parse_args()

    cases = [p for p in sorted(args.brats_root.iterdir()) if p.is_dir()]
    complete_cases = []
    for case_dir in cases:
        try:
            for suffix, _name in MODALITIES:
                find_file(case_dir, suffix)
            find_file(case_dir, "seg")
            complete_cases.append(case_dir)
        except FileNotFoundError:
            continue

    selected = complete_cases[: args.num_cases]
    if not selected:
        raise SystemExit("No complete BraTS cases found. Check --brats-root and file suffixes.")

    make_dataset(args.out_root, 401, "GliomaCore", selected, "core", args.et_label)
    make_dataset(args.out_root, 402, "GliomaTotal", selected, "total", args.et_label)


if __name__ == "__main__":
    main()
