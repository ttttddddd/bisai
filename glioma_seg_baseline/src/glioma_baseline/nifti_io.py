from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def check_nifti(path: Path) -> bool:
    """Return True when a NIfTI image exists and can be opened."""
    path = Path(path)
    if not path.is_file():
        return False
    try:
        import nibabel as nib

        img = nib.load(str(path))
        _ = img.shape
        _ = img.affine
        return True
    except Exception:
        pass
    try:
        import SimpleITK as sitk

        img = sitk.ReadImage(str(path))
        _ = img.GetSize()
        return True
    except Exception:
        return False


def copy_zero_mask_like(reference_image: Path, output_mask: Path) -> None:
    """Create an all-zero NIfTI mask with the same geometry as reference_image."""
    output_mask.parent.mkdir(parents=True, exist_ok=True)
    try:
        import nibabel as nib
        import numpy as np

        img = nib.load(str(reference_image))
        mask = np.zeros(img.shape, dtype=np.uint8)
        nib.save(nib.Nifti1Image(mask, img.affine, img.header), str(output_mask))
        return
    except Exception:
        pass

    try:
        import SimpleITK as sitk

        img = sitk.ReadImage(str(reference_image))
        mask = sitk.Image(img.GetSize(), sitk.sitkUInt8)
        mask.CopyInformation(img)
        sitk.WriteImage(mask, str(output_mask))
        return
    except Exception as exc:
        raise RuntimeError(
            "Cannot create NIfTI mask. Install nibabel or SimpleITK in the runtime environment."
        ) from exc


def ensure_binary_mask(mask_path: Path) -> None:
    """Force a NIfTI mask to contain only 0/1 values when nibabel is available."""
    try:
        import nibabel as nib
        import numpy as np
    except Exception:
        return
    img = nib.load(str(mask_path))
    data = (img.get_fdata() > 0).astype(np.uint8)
    nib.save(nib.Nifti1Image(data, img.affine, img.header), str(mask_path))


def mask_voxel_count(mask_path: Path) -> int:
    try:
        import nibabel as nib
    except Exception:
        return 0
    img = nib.load(str(mask_path))
    return int((img.get_fdata() > 0).sum())


def is_same_geometry(reference_image: Path, candidate_image: Path) -> bool:
    try:
        import nibabel as nib
        import numpy as np

        ref = nib.load(str(reference_image))
        cand = nib.load(str(candidate_image))
        return ref.shape == cand.shape and np.allclose(ref.affine, cand.affine, atol=1e-3)
    except Exception:
        pass

    try:
        import SimpleITK as sitk

        ref = sitk.ReadImage(str(reference_image))
        cand = sitk.ReadImage(str(candidate_image))
        return (
            ref.GetSize() == cand.GetSize()
            and ref.GetSpacing() == cand.GetSpacing()
            and ref.GetOrigin() == cand.GetOrigin()
            and ref.GetDirection() == cand.GetDirection()
        )
    except Exception:
        return True


def copy_or_compress_nifti(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() == dst.resolve():
        return
    shutil.copy2(src, dst)


def command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def run_checked(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)
