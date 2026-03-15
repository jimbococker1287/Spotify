from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path

import joblib
import numpy as np

from .champion_alias import resolve_prediction_run_dir
from .env import load_local_env
from .serving import load_predictor, resolve_model_row


@dataclass(frozen=True)
class PredictionInputContext:
    artist_labels: list[str]
    artist_to_label: dict[str, int]
    sequence_length: int
    latest_sequence_labels: np.ndarray
    latest_sequence_names: list[str]
    context_scaled: np.ndarray


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m spotify.predict_next",
        description="Load a trained run and predict top-k next artist recommendations.",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Path to outputs/runs/<run_id> or outputs/models/champion. Defaults to champion alias.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Serveable model name to load. Defaults to champion alias or best serveable model in the run.",
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


def _resolve_model_name(run_dir: Path, explicit: str | None, alias_model_name: str | None = None) -> str:
    row = resolve_model_row(
        run_dir,
        explicit_model_name=explicit,
        alias_model_name=alias_model_name,
    )
    model_name = str(row.get("model_name", "")).strip()
    if not model_name:
        raise RuntimeError("Could not resolve a serveable model name from run results.")
    return model_name


def load_prediction_input_context(
    run_dir: Path,
    data_dir: Path,
    include_video: bool,
    logger: logging.Logger,
) -> PredictionInputContext:
    # Import data pipeline lazily so TensorFlow can initialize first on macOS.
    from .data import append_audio_features, engineer_features, load_streaming_history

    metadata_path = run_dir / "feature_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing feature metadata: {metadata_path}")
    metadata = _load_json(metadata_path)

    artist_labels = list(metadata["artist_labels"])
    context_features = list(metadata["context_features"])
    skew_features = list(metadata.get("skew_context_features", []))
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

    if len(df) < sequence_length:
        raise RuntimeError(
            f"Not enough rows in rebuilt history for sequence length {sequence_length}: {len(df)} rows."
        )

    missing_context = [name for name in context_features if name not in df.columns]
    if missing_context:
        raise RuntimeError(f"Context columns missing from engineered frame: {', '.join(missing_context)}")

    latest_sequence_labels = df["artist_label"].to_numpy(dtype="int32")[-sequence_length:].copy()
    latest_sequence_names = [artist_labels[idx] for idx in latest_sequence_labels.tolist()]

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

    return PredictionInputContext(
        artist_labels=artist_labels,
        artist_to_label=artist_to_label,
        sequence_length=sequence_length,
        latest_sequence_labels=latest_sequence_labels,
        latest_sequence_names=latest_sequence_names,
        context_scaled=context_scaled,
    )


def _prepare_inputs(
    run_dir: Path,
    data_dir: Path,
    recent_artists: list[str] | None,
    include_video: bool,
    logger: logging.Logger,
    *,
    context: PredictionInputContext | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    if context is None:
        context = load_prediction_input_context(
            run_dir=run_dir,
            data_dir=data_dir,
            include_video=include_video,
            logger=logger,
        )

    if recent_artists:
        unknown = [artist for artist in recent_artists if artist not in context.artist_to_label]
        if unknown:
            raise RuntimeError(f"Unknown artists in --recent-artists: {', '.join(unknown)}")
        if len(recent_artists) < context.sequence_length:
            raise RuntimeError(
                f"--recent-artists must include at least {context.sequence_length} artists "
                f"(received {len(recent_artists)})."
            )
        sequence_names = recent_artists[-context.sequence_length :]
        sequence_labels = np.array([context.artist_to_label[name] for name in sequence_names], dtype="int32")
    else:
        sequence_labels = context.latest_sequence_labels.copy()
        sequence_names = list(context.latest_sequence_names)

    return sequence_labels.reshape(1, -1), context.context_scaled.copy(), sequence_names


def main() -> int:
    load_local_env()
    args = _parse_args()
    run_dir, champion_alias_model_name = resolve_prediction_run_dir(args.run_dir)

    mpl_config_dir = run_dir / ".mplconfig"
    xdg_cache_dir = run_dir / ".cache"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    xdg_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache_dir))

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("spotify.predict")

    data_dir = Path(args.data_dir)
    model_row = resolve_model_row(
        run_dir,
        explicit_model_name=args.model_name,
        alias_model_name=champion_alias_model_name,
    )
    metadata = _load_json(run_dir / "feature_metadata.json")
    artist_labels = list(metadata["artist_labels"])
    predictor = load_predictor(
        run_dir=run_dir,
        row=model_row,
        artist_labels=artist_labels,
    )

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

    artist_probs = predictor.predict_proba(seq_batch, ctx_batch)[0]

    top_k = max(1, int(args.top_k))
    top_indices = np.argsort(artist_probs)[::-1][:top_k]

    print(f"model={predictor.model_name}")
    print(f"model_type={predictor.model_type}")
    print("sequence_tail=" + " | ".join(sequence_names))
    print("top_predictions:")
    for rank, label_idx in enumerate(top_indices, start=1):
        artist_name = artist_labels[int(label_idx)]
        prob = float(artist_probs[int(label_idx)])
        print(f"{rank}. {artist_name} ({prob:.4f})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
