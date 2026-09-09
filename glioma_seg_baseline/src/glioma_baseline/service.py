from __future__ import annotations

import argparse
import json
import os
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .nifti_io import copy_zero_mask_like
from .nnunet_runner import NnUNetPredictor
from .prediction_json import default_prediction, write_prediction_json
from .sequence_select import discover_series, select_flair_or_t2, select_t1ce


ANSWER_BASE = Path(os.environ.get("ANSWER_BASE", "/2026aicompetition/workspace/answer"))
CALLBACK_URL = os.environ.get("CALLBACK_URL", "")


def relative_uri(from_dir: Path, target: Path) -> str:
    try:
        return "./" + target.relative_to(from_dir).as_posix()
    except ValueError:
        return target.as_posix()


def callback(request_id: str, evaluation_id: str) -> None:
    if not CALLBACK_URL:
        return
    payload = {
        "request_id": request_id,
        "evaluationId": evaluation_id,
        "predPath": str(ANSWER_BASE / evaluation_id),
    }
    request = urllib.request.Request(
        CALLBACK_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30):
            pass
    except OSError:
        pass


def write_duplicate_pairs(answer_dir: Path) -> None:
    path = answer_dir / "duplicate_pairs.jsonl"
    if not path.exists():
        path.write_text('{"StudyUID":"DUMMY_A","StudyUID_dup":"DUMMY_B","PairProb":0.0}\n', encoding="utf-8")


def run_inference(request_id: str, evaluation_id: str, dataset_path: str) -> None:
    start = time.time()
    answer_dir = ANSWER_BASE / evaluation_id
    answer_dir.mkdir(parents=True, exist_ok=True)
    write_duplicate_pairs(answer_dir)

    core_predictor = NnUNetPredictor(dataset_id=int(os.environ.get("CORE_DATASET_ID", "401")))
    total_predictor = NnUNetPredictor(dataset_id=int(os.environ.get("TOTAL_DATASET_ID", "402")))

    studies = discover_series(Path(dataset_path))
    for accession, series in studies.items():
        accession_dir = answer_dir / accession
        accession_dir.mkdir(parents=True, exist_ok=True)

        t1ce = select_t1ce(series)
        flair = select_flair_or_t2(series)
        if t1ce is None:
            t1ce = series[0]
        if flair is None:
            flair = t1ce

        core_dir = accession_dir / t1ce.series_uid
        flair_dir = accession_dir / flair.series_uid
        core_mask = core_dir / f"{t1ce.series_uid}_core.nii.gz"
        flair_mask = flair_dir / f"{flair.series_uid}_flair.nii.gz"
        core_dir.mkdir(parents=True, exist_ok=True)
        flair_dir.mkdir(parents=True, exist_ok=True)

        core_predictor.predict_one(t1ce.image_path, core_mask)
        if flair.image_path == t1ce.image_path:
            copy_zero_mask_like(flair.image_path, flair_mask)
        else:
            total_predictor.predict_one(flair.image_path, flair_mask)

        processing_ms = int((time.time() - start) * 1000)
        prediction = default_prediction(
            accession=accession,
            core_uri=relative_uri(accession_dir, core_mask),
            flair_uri=relative_uri(accession_dir, flair_mask),
            processing_time_ms=processing_ms,
        )
        write_prediction_json(accession_dir / "prediction.json", prediction)

    callback(request_id, evaluation_id)


class InferenceHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path != "/health":
            self.send_error(404)
            return
        self._json_response({"status": "active"})

    def do_POST(self) -> None:
        if self.path != "/call":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            request_id = payload["request_id"]
            input_data = payload["input"]
            evaluation_id = str(input_data.get("evaluation_id") or input_data["evaluationId"])
            dataset_path = input_data["dataset_path"]
        except Exception:
            self.send_error(400, "invalid request body")
            return

        threading.Thread(
            target=run_inference,
            args=(request_id, evaluation_id, dataset_path),
            daemon=True,
        ).start()
        self._json_response({"status": "received"})

    def _json_response(self, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    ThreadingHTTPServer((args.host, args.port), InferenceHandler).serve_forever()


if __name__ == "__main__":
    main()
