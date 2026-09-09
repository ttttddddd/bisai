from __future__ import annotations

import json
from pathlib import Path


def default_prediction(accession: str, core_uri: str, flair_uri: str, processing_time_ms: int = 0) -> dict:
    """Return a competition-compatible prediction skeleton.

    Segmentation is the real target of this baseline. Classification fields are
    conservative placeholders so the JSON remains structurally complete.
    """
    return {
        "AccessionNumber": accession,
        "IsNotHumanBodyProb": 0.0,
        "IsStitchedProb": 0.0,
        "ProcessingTime_ms": processing_time_ms,
        "SegmentationMaskURI": {
            "core": core_uri,
            "flair": flair_uri,
        },
        "Prediction": {
            "TumorProbability": 0.5,
            "Location": "NA/UNK",
            "Morphology": {
                "predicted": "Regular",
                "probabilities": {"Regular": 0.5, "Irregular": 0.5},
            },
            "WHO_Grade": {
                "predicted": None,
                "probabilities": {"1": 0.0, "2": 0.0, "3": 0.0, "4": 0.0},
            },
            "Enhancement": {"present": False, "EnhancementProbability": 0.5},
            "EnhancementPattern": {
                "predicted": "None",
                "probabilities": {
                    "None": 0.125,
                    "Ring": 0.125,
                    "RimEnhancing": 0.125,
                    "Nodular": 0.125,
                    "GroundGlass": 0.125,
                    "Gyriform": 0.125,
                    "Multifocal": 0.125,
                    "Other": 0.125,
                },
            },
            "Necrosis": {"present": False, "NecrosisProbability": 0.5},
            "CysticChange": {"present": False, "CysticChangeProbability": 0.5},
            "Hemorrhage": {"present": False, "HemorrhageProbability": 0.5},
            "Calcification": {"present": False, "CalcificationProbability": 0.5},
            "Margin": {"clear": True, "MarginClearProbability": 0.5},
            "Lobulation": {"present": False, "LobulationProbability": 0.5},
            "Signal_T2WI": {
                "predicted": "High",
                "probabilities": {"Low": 1 / 3, "Iso": 1 / 3, "High": 1 / 3},
            },
            "Signal_FLAIR": {
                "predicted": "High",
                "probabilities": {"Low": 1 / 3, "Iso": 1 / 3, "High": 1 / 3},
            },
        },
        "Interpretation": {
            "Conclusion": "Baseline segmentation output; classification fields are placeholders.",
            "AttentionMapURI": "",
        },
    }


def write_prediction_json(path: Path, prediction: dict) -> None:
    path.write_text(json.dumps(prediction, ensure_ascii=False, indent=2), encoding="utf-8")

