#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np


def find_file(case_dir: Path, suffix: str) -> Path:
    patterns = [f"*-{suffix}.nii.gz", f"*_{suffix}.nii.gz", f"*{suffix}.nii.gz"]
    for pattern in patterns:
        matches = sorted(case_dir.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Missing {suffix} in {case_dir}")


def write_dataset_json(dataset_dir: Path, name: str, modality: str, num_cases: int) -> None:
    dataset = {
        "channel_names": {"0": modality},
        "labels": {"background": 0, "foreground": 1},
        "numTraining": num_cases,
        "file_ending": ".nii.gz",
        "name": name,
        "description": "Track 4 glioma segmentation baseline generated from BraTS.",
        "reference": "Core=BraTS TC, Total=BraTS WT.",
    }
    (dataset_dir / "dataset.json").write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")


def save_label(seg_path: Path, out_path: Path, task: str, et_label: int) -> None:
    img = nib.load(str(seg_path))
    seg = img.get_fdata().astype(np.int16)
    if task == "core":
        mask = np.logical_or(seg == 1, seg == et_label).astype(np.uint8)
    elif task == "total":
        mask = (seg > 0).astype(np.uint8)
    else:
        raise ValueError(task)
    nib.save(nib.Nifti1Image(mask, img.affine, img.header), str(out_path))


def make_dataset(
    out_root: Path,
    dataset_id: int,
    dataset_name: str,
    image_suffix: str,
    modality: str,
    cases: list[Path],
    task: str,
    et_label: int,
) -> None:
    dataset_dir = out_root / f"Dataset{dataset_id}_{dataset_name}"
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    for idx, case_dir in enumerate(cases):
        case_id = f"GLIOMA_{idx:03d}"
        shutil.copy2(find_file(case_dir, image_suffix), images_tr / f"{case_id}_0000.nii.gz")
        save_label(find_file(case_dir, "seg"), labels_tr / f"{case_id}.nii.gz", task, et_label)

    write_dataset_json(dataset_dir, dataset_name, modality, len(cases))
    print(f"Wrote {dataset_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--brats-root", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--num-cases", type=int, default=8)
    parser.add_argument("--et-label", type=int, default=4)
    args = parser.parse_args()

    complete_cases: list[Path] = []
    for case_dir in sorted(p for p in args.brats_root.iterdir() if p.is_dir()):
        try:
            find_file(case_dir, "t1c")
            find_file(case_dir, "t2f")
            find_file(case_dir, "seg")
            complete_cases.append(case_dir)
        except FileNotFoundError:
            continue

    selected = complete_cases[: args.num_cases]
    if not selected:
        raise SystemExit("No complete BraTS cases found. Expected t1c, t2f and seg files.")

    make_dataset(args.out_root, 401, "GliomaCoreT1CE", "t1c", "T1CE", selected, "core", args.et_label)
    make_dataset(args.out_root, 402, "GliomaTotalFLAIR", "t2f", "FLAIR", selected, "total", args.et_label)


if __name__ == "__main__":
    main()
