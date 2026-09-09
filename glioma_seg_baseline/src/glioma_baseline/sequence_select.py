from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


NIFTI_SUFFIXES = (".nii", ".nii.gz")


@dataclass(frozen=True)
class SeriesImage:
    accession: str
    series_uid: str
    image_path: Path
    description: str = ""


def is_nifti(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(NIFTI_SUFFIXES)


def _score_t1ce(text: str) -> int:
    t = text.lower()
    score = 0
    if any(k in t for k in ("t1ce", "t1c", "t1+c", "t1_c", "t1gd", "t1-gd", "t1wi+c", "contrast", "enh")):
        score += 10
    if "t1" in t:
        score += 3
    if any(k in t for k in ("flair", "t2", "dwi", "adc", "swi", "localizer", "scout")):
        score -= 4
    return score


def _score_flair(text: str) -> int:
    t = text.lower()
    score = 0
    if any(k in t for k in ("flair", "t2flair", "t2-flair", "t2_f", "t2f", "dark-fluid", "dark fluid")):
        score += 10
    if "t2" in t:
        score += 2
    if any(k in t for k in ("t1", "dwi", "adc", "swi", "localizer", "scout")):
        score -= 3
    return score


def _score_t2(text: str) -> int:
    t = text.lower()
    score = 0
    if any(k in t for k in ("t2w", "t2wi", "t2_tse", "t2 ")):
        score += 8
    if "t2" in t:
        score += 4
    if "flair" in t:
        score -= 2
    if any(k in t for k in ("t1", "dwi", "adc", "swi", "localizer", "scout")):
        score -= 4
    return score


def discover_series(dataset_path: Path) -> dict[str, list[SeriesImage]]:
    """Find NIfTI images under a competition-style dataset directory.

    The expected competition layout is usually:
      dataset_path/AccessionNumber/SeriesUid/*.nii

    This function is permissive and also accepts direct NIfTI files under an
    accession directory.
    """
    studies: dict[str, list[SeriesImage]] = {}
    for accession_dir in sorted(p for p in dataset_path.iterdir() if p.is_dir()):
        accession = accession_dir.name
        series: list[SeriesImage] = []
        for path in sorted(accession_dir.rglob("*")):
            if not path.is_file() or not is_nifti(path):
                continue
            rel = path.relative_to(accession_dir)
            series_uid = rel.parts[0] if len(rel.parts) > 1 else path.stem.replace(".nii", "")
            description = " ".join([series_uid, path.name])
            series.append(SeriesImage(accession, series_uid, path, description))
        if series:
            studies[accession] = series
    return studies


def select_t1ce(series: list[SeriesImage]) -> SeriesImage | None:
    ranked = sorted(series, key=lambda s: _score_t1ce(f"{s.description} {s.image_path}"), reverse=True)
    return ranked[0] if ranked and _score_t1ce(f"{ranked[0].description} {ranked[0].image_path}") > 0 else None


def select_flair_or_t2(series: list[SeriesImage]) -> SeriesImage | None:
    flair_ranked = sorted(series, key=lambda s: _score_flair(f"{s.description} {s.image_path}"), reverse=True)
    if flair_ranked and _score_flair(f"{flair_ranked[0].description} {flair_ranked[0].image_path}") > 0:
        return flair_ranked[0]
    t2_ranked = sorted(series, key=lambda s: _score_t2(f"{s.description} {s.image_path}"), reverse=True)
    return t2_ranked[0] if t2_ranked and _score_t2(f"{t2_ranked[0].description} {t2_ranked[0].image_path}") > 0 else None

