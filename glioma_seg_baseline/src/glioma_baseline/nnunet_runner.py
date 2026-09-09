from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from .nifti_io import command_exists, copy_or_compress_nifti, copy_zero_mask_like, ensure_binary_mask, run_checked


class NnUNetPredictor:
    """Small wrapper around nnU-Net v2 command line prediction."""

    def __init__(
        self,
        dataset_id: int,
        trainer: str = "nnUNetTrainer",
        configuration: str = "3d_fullres",
        fold: str = "0",
        checkpoint: str = "checkpoint_final.pth",
    ) -> None:
        self.dataset_id = dataset_id
        self.trainer = trainer
        self.configuration = configuration
        self.fold = fold
        self.checkpoint = checkpoint

    def available(self) -> bool:
        if not command_exists("nnUNetv2_predict"):
            return False
        results = os.environ.get("nnUNet_results")
        return bool(results and Path(results).exists())

    def predict_one(self, image: Path, output_mask: Path) -> bool:
        """Predict one single-channel case.

        Returns True when nnU-Net was used. Returns False when nnU-Net is not
        available, so the caller can fall back to an all-zero mask.
        """
        if not self.available():
            copy_zero_mask_like(image, output_mask)
            return False

        with tempfile.TemporaryDirectory(prefix="glioma_nnunet_") as tmp:
            tmp_path = Path(tmp)
            images_ts = tmp_path / "imagesTs"
            pred_dir = tmp_path / "pred"
            images_ts.mkdir()
            pred_dir.mkdir()
            case_id = "CASE_000"
            copy_or_compress_nifti(image, images_ts / f"{case_id}_0000.nii.gz")
            cmd = [
                "nnUNetv2_predict",
                "-i",
                str(images_ts),
                "-o",
                str(pred_dir),
                "-d",
                str(self.dataset_id),
                "-c",
                self.configuration,
                "-tr",
                self.trainer,
                "-f",
                self.fold,
                "-chk",
                self.checkpoint,
            ]
            try:
                run_checked(cmd)
            except Exception:
                copy_zero_mask_like(image, output_mask)
                return False
            predicted = pred_dir / f"{case_id}.nii.gz"
            if not predicted.exists():
                copy_zero_mask_like(image, output_mask)
                return False
            copy_or_compress_nifti(predicted, output_mask)
            ensure_binary_mask(output_mask)
            return True

