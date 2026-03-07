from __future__ import annotations

import argparse
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from .predict_next import _prepare_inputs, _resolve_model_name


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m spotify.predict_service",
        description="Serve Spotify next-artist predictions over HTTP.",
    )
    parser.add_argument("--run-dir", type=str, required=True, help="Path to outputs/runs/<run_id>.")
    parser.add_argument("--model-name", type=str, default=None, help="Optional deep model checkpoint name override.")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Path to raw Streaming_History JSON files.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port.")
    parser.add_argument(
        "--include-video",
        action="store_true",
        help="Include video history files by default when rebuilding request context.",
    )
    return parser.parse_args()


class PredictionService:
    def __init__(self, run_dir: Path, data_dir: Path, model_name: str, include_video: bool, logger: logging.Logger):
        import tensorflow as tf

        self.run_dir = run_dir
        self.data_dir = data_dir
        self.model_name = model_name
        self.include_video = include_video
        self.logger = logger
        self._predict_lock = threading.Lock()

        self.model_path = run_dir / f"best_{model_name}.keras"
        logger.info("Loading model checkpoint: %s", self.model_path)
        self.model = tf.keras.models.load_model(self.model_path, compile=False)

        metadata_path = run_dir / "feature_metadata.json"
        with metadata_path.open("r", encoding="utf-8") as infile:
            metadata = json.load(infile)
        self.artist_labels = list(metadata.get("artist_labels", []))

    def predict(self, *, top_k: int, recent_artists: list[str] | None, include_video: bool) -> dict[str, object]:
        seq_batch, ctx_batch, sequence_names = _prepare_inputs(
            run_dir=self.run_dir,
            data_dir=self.data_dir,
            recent_artists=recent_artists,
            include_video=include_video,
            logger=self.logger,
        )

        with self._predict_lock:
            preds = self.model.predict((seq_batch, ctx_batch), verbose=0)

        if isinstance(preds, (tuple, list)):
            artist_probs = np.asarray(preds[0])[0]
        else:
            artist_probs = np.asarray(preds)[0]

        top_k = max(1, int(top_k))
        top_indices = np.argsort(artist_probs)[::-1][:top_k]
        predictions: list[dict[str, object]] = []
        for rank, idx in enumerate(top_indices, start=1):
            label_idx = int(idx)
            artist_name = (
                self.artist_labels[label_idx]
                if 0 <= label_idx < len(self.artist_labels)
                else str(label_idx)
            )
            predictions.append(
                {
                    "rank": rank,
                    "artist_label": label_idx,
                    "artist_name": artist_name,
                    "probability": float(artist_probs[label_idx]),
                }
            )

        return {
            "model_name": self.model_name,
            "sequence_tail": sequence_names,
            "predictions": predictions,
        }


def _build_handler(service: PredictionService):
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, status_code: int, payload: dict[str, object]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args) -> None:
            service.logger.info("HTTP %s - %s", self.address_string(), fmt % args)

        def do_GET(self) -> None:  # noqa: N802
            if self.path.rstrip("/") == "/health":
                self._send_json(
                    200,
                    {
                        "status": "ok",
                        "model_name": service.model_name,
                        "run_dir": str(service.run_dir),
                    },
                )
                return
            self._send_json(404, {"error": "not_found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path.rstrip("/") != "/predict":
                self._send_json(404, {"error": "not_found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self._send_json(400, {"error": "invalid_content_length"})
                return

            try:
                raw = self.rfile.read(length) if length > 0 else b"{}"
                payload = json.loads(raw.decode("utf-8"))
            except Exception:
                self._send_json(400, {"error": "invalid_json"})
                return

            top_k = payload.get("top_k", 5)
            include_video = payload.get("include_video", service.include_video)
            recent_artists = payload.get("recent_artists")
            if isinstance(recent_artists, str):
                recent_artists = [part.strip() for part in recent_artists.split("|") if part.strip()]
            if recent_artists is not None and not isinstance(recent_artists, list):
                self._send_json(400, {"error": "recent_artists must be a list or pipe-separated string"})
                return
            if isinstance(recent_artists, list):
                recent_artists = [str(item).strip() for item in recent_artists if str(item).strip()]

            try:
                result = service.predict(
                    top_k=int(top_k),
                    recent_artists=recent_artists,
                    include_video=bool(include_video),
                )
                self._send_json(200, result)
            except Exception as exc:
                self._send_json(500, {"error": str(exc)})

    return Handler


def main() -> int:
    args = _parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("spotify.predict_service")

    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    data_dir = Path(args.data_dir).expanduser().resolve()
    model_name = _resolve_model_name(run_dir, args.model_name)

    service = PredictionService(
        run_dir=run_dir,
        data_dir=data_dir,
        model_name=model_name,
        include_video=bool(args.include_video),
        logger=logger,
    )
    server = ThreadingHTTPServer((str(args.host), int(args.port)), _build_handler(service))
    logger.info("Prediction service listening on http://%s:%d", args.host, int(args.port))
    try:
        server.serve_forever()
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
