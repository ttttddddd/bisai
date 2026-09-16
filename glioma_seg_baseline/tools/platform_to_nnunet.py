#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from glioma_baseline.nifti_io import copy_zero_mask_like, is_same_geometry  # noqa: E402
from glioma_baseline.sequence_select import SeriesImage, discover_series, select_flair_or_t2, select_t1ce  # noqa: E402


CORE_TERMS = ("core", "t1ce", "t1c", "tumor", "enhance", "瘤体", "肿瘤核心", "核心")
TOTAL_TERMS = ("total", "abnormal", "flair", "t2", "edema", "whole", "异常", "水肿", "全肿瘤")
IMAGE_TERMS = ("image", "img", "series", "t1", "t2", "flair")


@dataclass(frozen=True)
class LabelRecord:
    accession: str
    series_uid: str
    task: str
    mask_name: str
    series_label: str = ""


def is_nifti(path: Path) -> bool:
    return path.name.lower().endswith((".nii", ".nii.gz"))


def strip_nii_suffix(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return path.stem


def nnunet_dataset_json(name: str, num_cases: int) -> dict:
    return {
        "channel_names": {"0": "MRI"},
        "labels": {"background": 0, "lesion": 1},
        "numTraining": num_cases,
        "file_ending": ".nii.gz",
        "dataset_name": name,
    }


def write_dataset_json(dataset_dir: Path, name: str, num_cases: int) -> None:
    (dataset_dir / "dataset.json").write_text(
        json.dumps(nnunet_dataset_json(name, num_cases), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_table(path: Path) -> list[dict[str, str]]:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            return [dict(row) for row in csv.DictReader(f)]
    if path.suffix.lower() in (".xlsx", ".xls"):
        try:
            import pandas as pd
        except Exception as exc:
            raise RuntimeError("Reading Excel labels requires pandas and openpyxl.") from exc
        return pd.read_excel(path).fillna("").astype(str).to_dict("records")
    return []


def candidate_label_files(label_root: Path | None, explicit: Path | None) -> list[Path]:
    if explicit:
        return [explicit] if explicit.exists() else []
    if not label_root or not label_root.exists():
        return []
    return sorted([p for p in label_root.rglob("*") if p.suffix.lower() in (".xlsx", ".xls", ".csv")])


def read_label_records(label_root: Path | None, explicit: Path | None) -> dict[str, list[LabelRecord]]:
    records: dict[str, list[LabelRecord]] = defaultdict(list)
    for table in candidate_label_files(label_root, explicit):
        try:
            rows = load_table(table)
        except Exception:
            continue
        for row in rows:
            accession = first_value(row, ("AccessionNumber", "accession", "StudyUID", "检查号"))
            series_uid = first_value(row, ("SeriesUid", "SeriesUID", "series_uid", "序列号"))
            mask_name = first_value(row, ("Maskname", "MaskName", "mask_name", "Mask", "标注文件", "标签文件"))
            if not accession or not series_uid or not mask_name:
                continue
            records[accession].append(
                LabelRecord(
                    accession=accession,
                    series_uid=series_uid,
                    task=first_value(row, ("Task", "task", "任务", "RoiName", "ROIName", "LabelName")),
                    mask_name=mask_name,
                    series_label=first_value(row, ("SeriesLabel", "DetailDescription", "Description", "序列标签", "序列描述")),
                )
            )
    return records


def first_value(row: dict, keys: tuple[str, ...]) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def task_matches(text: str, terms: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(term.lower() in lowered for term in terms)


def find_mask_path(annotation_root: Path, record: LabelRecord) -> Path | None:
    candidates = [
        annotation_root / record.accession / record.series_uid / record.mask_name,
        annotation_root / record.accession / record.mask_name,
        annotation_root / record.accession / record.series_uid / f"{record.mask_name}.nii.gz",
        annotation_root / record.accession / record.series_uid / f"{record.mask_name}.nii",
    ]
    for path in candidates:
        if path.is_file() and is_nifti(path):
            return path
    accession_dir = annotation_root / record.accession
    if accession_dir.exists():
        for path in accession_dir.rglob("*"):
            if path.is_file() and is_nifti(path) and path.name == record.mask_name:
                return path
    return None


def scan_masks_for_series(series: SeriesImage, terms: tuple[str, ...]) -> list[Path]:
    masks: list[Path] = []
    for path in sorted(series.image_path.parent.rglob("*")):
        if not path.is_file() or not is_nifti(path) or path == series.image_path:
            continue
        name = path.name.lower()
        if any(term in name for term in IMAGE_TERMS) and not task_matches(name, terms):
            continue
        if task_matches(name, terms):
            masks.append(path)
    return masks


def label_masks_from_records(
    annotation_root: Path,
    records: list[LabelRecord],
    selected: SeriesImage,
    terms: tuple[str, ...],
) -> list[Path]:
    masks: list[Path] = []
    for record in records:
        text = " ".join([record.task, record.mask_name, record.series_label])
        if record.series_uid != selected.series_uid:
            continue
        if not task_matches(text, terms):
            continue
        path = find_mask_path(annotation_root, record)
        if path:
            masks.append(path)
    return masks


def union_masks(mask_paths: list[Path], reference_image: Path, output_mask: Path) -> bool:
    if not mask_paths:
        return False
    try:
        import nibabel as nib
        import numpy as np

        ref = nib.load(str(reference_image))
        merged = np.zeros(ref.shape, dtype=np.uint8)
        used = 0
        for mask_path in mask_paths:
            if not is_same_geometry(reference_image, mask_path):
                continue
            mask = nib.load(str(mask_path)).get_fdata() > 0
            if mask.shape != merged.shape:
                continue
            merged[mask] = 1
            used += 1
        if used == 0:
            return False
        output_mask.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(merged, ref.affine, ref.header), str(output_mask))
        return True
    except Exception:
        pass

    try:
        import SimpleITK as sitk
    except Exception as exc:
        raise RuntimeError("Merging platform masks requires nibabel or SimpleITK.") from exc

    ref = sitk.ReadImage(str(reference_image))
    merged = sitk.Image(ref.GetSize(), sitk.sitkUInt8)
    merged.CopyInformation(ref)
    used = 0
    for mask_path in mask_paths:
        if not is_same_geometry(reference_image, mask_path):
            continue
        mask = sitk.ReadImage(str(mask_path))
        if mask.GetSize() != ref.GetSize():
            continue
        merged = sitk.Or(merged, sitk.Cast(mask > 0, sitk.sitkUInt8))
        used += 1
    if used == 0:
        return False
    output_mask.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(merged, str(output_mask))
    return True


def copy_image(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def build_dataset(
    dataset_dir: Path,
    dataset_name: str,
    studies: dict[str, list[SeriesImage]],
    labels: dict[str, list[LabelRecord]],
    annotation_root: Path,
    target: str,
    max_cases: int,
    include_empty: bool,
) -> int:
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    terms = CORE_TERMS if target == "core" else TOTAL_TERMS
    written = 0
    for accession, series_list in sorted(studies.items()):
        if max_cases and written >= max_cases:
            break
        selected = select_t1ce(series_list) if target == "core" else select_flair_or_t2(series_list)
        if selected is None:
            selected = series_list[0]

        mask_paths = label_masks_from_records(annotation_root, labels.get(accession, []), selected, terms)
        if not mask_paths:
            mask_paths = scan_masks_for_series(selected, terms)

        case_id = f"PLAT_{written:04d}"
        image_out = images_tr / f"{case_id}_0000.nii.gz"
        mask_out = labels_tr / f"{case_id}.nii.gz"

        if mask_paths:
            merged = union_masks(mask_paths, selected.image_path, mask_out)
            if not merged and not include_empty:
                continue
        elif include_empty:
            copy_zero_mask_like(selected.image_path, mask_out)
        else:
            continue

        copy_image(selected.image_path, image_out)
        written += 1

    write_dataset_json(dataset_dir, dataset_name, written)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert platform glioma annotation data to nnU-Net v2 datasets.")
    parser.add_argument("--annotation-root", default="/2026aicompetition/datasets/training/annotation", type=Path)
    parser.add_argument("--label-root", default="/2026aicompetition/datasets/training/label", type=Path)
    parser.add_argument("--label-file", default=None, type=Path)
    parser.add_argument("--out-root", default="/2026aicompetition/workspace/nnUNet_raw", type=Path)
    parser.add_argument("--max-cases", default=0, type=int)
    parser.add_argument("--include-empty", action="store_true", help="Include zero masks when no label mask is found.")
    parser.add_argument("--validate-nifti", action="store_true")
    args = parser.parse_args()

    studies = discover_series(args.annotation_root, validate_nifti=args.validate_nifti)
    labels = read_label_records(args.label_root, args.label_file)
    if not studies:
        raise SystemExit(f"No platform NIfTI studies found under {args.annotation_root}")

    core_count = build_dataset(
        args.out_root / "Dataset401_GliomaCoreT1CE",
        "GliomaCoreT1CE",
        studies,
        labels,
        args.annotation_root,
        "core",
        args.max_cases,
        args.include_empty,
    )
    total_count = build_dataset(
        args.out_root / "Dataset402_GliomaTotalFLAIR",
        "GliomaTotalFLAIR",
        studies,
        labels,
        args.annotation_root,
        "total",
        args.max_cases,
        args.include_empty,
    )
    print(f"Platform studies found: {len(studies)}")
    print(f"Dataset401_GliomaCoreT1CE cases: {core_count}")
    print(f"Dataset402_GliomaTotalFLAIR cases: {total_count}")
    if core_count == 0 or total_count == 0:
        print("Warning: one dataset has zero cases. Check label_root/label_file or use --include-empty only for dry runs.")


if __name__ == "__main__":
    main()
