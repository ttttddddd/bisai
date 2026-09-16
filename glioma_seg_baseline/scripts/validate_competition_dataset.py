#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from glioma_baseline.nifti_io import check_nifti
from glioma_baseline.sequence_select import discover_series, select_flair_or_t2, select_t1ce


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-csv", default="")
    parser.add_argument("--validate-nifti", action="store_true")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    studies = discover_series(dataset_path, validate_nifti=False)
    rows = []

    print(f"dataset_path={dataset_path}")
    print(f"cases_with_nifti={len(studies)}")

    no_t1ce = no_flair_or_t2 = broken = 0
    for accession, series in sorted(studies.items()):
        t1ce = select_t1ce(series)
        flair = select_flair_or_t2(series)
        if t1ce is None:
            no_t1ce += 1
        if flair is None:
            no_flair_or_t2 += 1

        broken_count = 0
        if args.validate_nifti:
            for item in series:
                if not check_nifti(item.image_path):
                    broken_count += 1
            broken += broken_count

        rows.append(
            {
                "AccessionNumber": accession,
                "num_series": len(series),
                "selected_t1ce": t1ce.series_uid if t1ce else "",
                "selected_flair_or_t2": flair.series_uid if flair else "",
                "broken_nifti": broken_count,
            }
        )

    print(f"missing_t1ce_cases={no_t1ce}")
    print(f"missing_flair_or_t2_cases={no_flair_or_t2}")
    if args.validate_nifti:
        print(f"broken_nifti_files={broken}")

    for row in rows[:20]:
        print(row)

    if args.output_csv:
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["AccessionNumber", "num_series", "selected_t1ce", "selected_flair_or_t2", "broken_nifti"],
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
