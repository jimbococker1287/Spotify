from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import joblib
import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m spotify.predict_next",
        description="Load a trained run and predict top-k next artist recommendations.",
    )
    parser.add_argument("--run-dir", type=str, required=True, help="Path to outputs/runs/<run_id>.")
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Deep model checkpoint to load (without best_ prefix). Defaults to best val_top1 deep model.",
    )
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Path to raw Streaming_History JSON files.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of next-artist predictions to display.")
    parser.add_argument(
        "--recent-artists",
        type=str,
        default=None,
        help="Optional pipe-separated artist names to use as the recent sequence override.",
    )
    parser.add_argument(
        "--include-video",
        action="store_true",
        help="Include video history files while rebuilding the latest context.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as infile:
        return json.load(infile)


def _resolve_model_name(run_dir: Path, explicit: str | None) -> str:
    if explicit:
        candidate = run_dir / f"best_{explicit}.keras"
        if not candidate.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {candidate}")
        return explicit

    results_path = run_dir / "run_results.json"
    if not results_path.exists():
        raise FileNotFoundError(
            "run_results.json is missing, so best deep model cannot be inferred. "
            "Pass --model-name explicitly."
        )

    rows = _load_json(results_path)
    deep_rows = [row for row in rows if str(row.get("model_type", "")) == "deep"]
    if not deep_rows:
        raise RuntimeError("No deep model rows found in run_results.json.")

    def _score(row: dict) -> float:
        try:
            return float(row.get("val_top1", float("nan")))
        except Exception:
            return float("nan")

    best = max(deep_rows, key=_score)
    model_name = str(best.get("model_name", "")).strip()
    if not model_name:
        raise RuntimeError("Could not infer best deep model name from run_results.json.")

    checkpoint = run_dir / f"best_{model_name}.keras"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Inferred checkpoint not found: {checkpoint}")
    return model_name


def _prepare_inputs(
    run_dir: Path,
    data_dir: Path,
    recent_artists: list[str] | None,
    include_video: bool,
    logger: logging.Logger,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    # Import data pipeline lazily so TensorFlow can initialize first on macOS.
    from .data import append_audio_features, engineer_features, load_streaming_history

    metadata_path = run_dir / "feature_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing feature metadata: {metadata_path}")
    metadata = _load_json(metadata_path)

    context_features = list(metadata["context_features"])
    skew_features = list(metadata.get("skew_context_features", []))
    artist_labels = list(metadata["artist_labels"])
    sequence_length = int(metadata["sequence_length"])
    artist_to_label = {artist: idx for idx, artist in enumerate(artist_labels)}

    df = load_streaming_history(data_dir.expanduser().resolve(), include_video=include_video, logger=logger)
    df = engineer_features(
        df=df,
        max_artists=len(artist_labels),
        logger=logger,
        artist_classes=artist_labels,
    )
    df = append_audio_features(df, enable_spotify_features=False, logger=logger)
    df = df.sort_values("ts").reset_index(drop=True)

    if recent_artists:
        unknown = [artist for artist in recent_artists if artist not in artist_to_label]
        if unknown:
            raise RuntimeError(f"Unknown artists in --recent-artists: {', '.join(unknown)}")
        if len(recent_artists) < sequence_length:
            raise RuntimeError(
                f"--recent-artists must include at least {sequence_length} artists "
                f"(received {len(recent_artists)})."
            )
        sequence_names = recent_artists[-sequence_length:]
        sequence_labels = np.array([artist_to_label[name] for name in sequence_names], dtype="int32")
    else:
        if len(df) < sequence_length:
            raise RuntimeError(
                f"Not enough rows in rebuilt history for sequence length {sequence_length}: {len(df)} rows."
            )
        sequence_labels = df["artist_label"].to_numpy(dtype="int32")[-sequence_length:]
        sequence_names = [
            artist_labels[idx]
            for idx in sequence_labels.tolist()
        ]

    missing_context = [name for name in context_features if name not in df.columns]
    if missing_context:
        raise RuntimeError(f"Context columns missing from engineered frame: {', '.join(missing_context)}")

    context_row = df[context_features].iloc[-1].to_numpy(dtype="float32")
    for key in skew_features:
        if key in context_features:
            idx = context_features.index(key)
            context_row[idx] = float(np.log1p(max(float(context_row[idx]), 0.0)))

    scaler_path = run_dir / "context_scaler.joblib"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Missing scaler file: {scaler_path}")
    scaler = joblib.load(scaler_path)
    context_scaled = scaler.transform(context_row.reshape(1, -1)).astype("float32")

    return sequence_labels.reshape(1, -1), context_scaled, sequence_names


def main() -> int:
    args = _parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    mpl_config_dir = run_dir / ".mplconfig"
    xdg_cache_dir = run_dir / ".cache"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    xdg_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache_dir))

    import tensorflow as tf

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("spotify.predict")

    data_dir = Path(args.data_dir)
    model_name = _resolve_model_name(run_dir, args.model_name)
    model_path = run_dir / f"best_{model_name}.keras"
    logger.info("Loading model checkpoint: %s", model_path)
    model = tf.keras.models.load_model(model_path, compile=False)

    recent_artists = None
    if args.recent_artists:
        recent_artists = [part.strip() for part in args.recent_artists.split("|") if part.strip()]

    seq_batch, ctx_batch, sequence_names = _prepare_inputs(
        run_dir=run_dir,
        data_dir=data_dir,
        recent_artists=recent_artists,
        include_video=bool(args.include_video),
        logger=logger,
    )

    preds = model.predict((seq_batch, ctx_batch), verbose=0)
    if isinstance(preds, (tuple, list)):
        artist_probs = np.asarray(preds[0])[0]
    else:
        artist_probs = np.asarray(preds)[0]

    metadata = _load_json(run_dir / "feature_metadata.json")
    artist_labels = list(metadata["artist_labels"])

    top_k = max(1, int(args.top_k))
    top_indices = np.argsort(artist_probs)[::-1][:top_k]

    print(f"model={model_name}")
    print("sequence_tail=" + " | ".join(sequence_names))
    print("top_predictions:")
    for rank, label_idx in enumerate(top_indices, start=1):
        artist_name = artist_labels[int(label_idx)]
        prob = float(artist_probs[int(label_idx)])
        print(f"{rank}. {artist_name} ({prob:.4f})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
