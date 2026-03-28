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
from .public_catalog import SpotifyArtistMetadata, SpotifyPublicCatalogClient, SpotifyPublicCatalogError
from .serving import load_predictor, resolve_model_row


@dataclass(frozen=True)
class PredictionInputContext:
    artist_labels: list[str]
    artist_to_label: dict[str, int]
    sequence_length: int
    latest_sequence_labels: np.ndarray
    latest_sequence_names: list[str]
    context_scaled: np.ndarray
    context_raw: np.ndarray | None = None
    context_features: list[str] | None = None
    friction_reference: dict[str, object] | None = None
    scaler_mean: np.ndarray | None = None
    scaler_scale: np.ndarray | None = None


_PREDICTION_CONTEXT_CACHE_VERSION = 3


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
    parser.add_argument(
        "--spotify-public-metadata",
        action="store_true",
        help="Enrich printed predictions with Spotify public artist metadata via the Web API.",
    )
    parser.add_argument(
        "--spotify-market",
        type=str,
        default="US",
        help="Two-letter market code to use when looking up Spotify public artist metadata.",
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


def _signature_for_paths(paths: list[Path]) -> tuple[tuple[str, int, int], ...]:
    signature: list[tuple[str, int, int]] = []
    for path in sorted({item.resolve() for item in paths if item.exists()}):
        stat = path.stat()
        signature.append(
            (
                str(path),
                int(stat.st_size),
                int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9))),
            )
        )
    return tuple(signature)


def prediction_source_signature(
    *,
    run_dir: Path,
    data_dir: Path,
    include_video: bool,
) -> tuple[tuple[str, int, int], ...]:
    root = data_dir.expanduser().resolve()
    json_paths = sorted(path for path in root.rglob("*.json") if path.is_file())
    if not include_video:
        json_paths = [path for path in json_paths if "Streaming_History_Video_" not in path.name]
    paths = list(json_paths)
    paths.extend(
        [
            run_dir / "feature_metadata.json",
            run_dir / "context_scaler.joblib",
        ]
    )
    return _signature_for_paths(paths)


def _prediction_context_cache_path(run_dir: Path, *, include_video: bool) -> Path:
    cache_dir = run_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    suffix = "audio_video" if include_video else "audio"
    return cache_dir / f"prediction_input_context_{suffix}.joblib"


def _load_cached_prediction_input_context(
    *,
    cache_path: Path,
    signature: tuple[tuple[str, int, int], ...],
    logger: logging.Logger,
) -> PredictionInputContext | None:
    if not cache_path.exists():
        return None
    try:
        payload = joblib.load(cache_path)
    except Exception as exc:
        logger.warning("Prediction context cache load failed for %s: %s", cache_path, exc)
        return None
    if not isinstance(payload, dict):
        return None
    if int(payload.get("version", -1)) != _PREDICTION_CONTEXT_CACHE_VERSION:
        return None
    if tuple(payload.get("signature", ())) != signature:
        return None
    context = payload.get("context")
    if not isinstance(context, PredictionInputContext):
        return None
    logger.info("Loaded cached prediction context: %s", cache_path)
    return context


def _store_prediction_input_context_cache(
    *,
    cache_path: Path,
    signature: tuple[tuple[str, int, int], ...],
    context: PredictionInputContext,
    logger: logging.Logger,
) -> None:
    payload = {
        "version": _PREDICTION_CONTEXT_CACHE_VERSION,
        "signature": signature,
        "context": context,
    }
    try:
        joblib.dump(payload, cache_path, compress=3)
    except Exception as exc:
        logger.warning("Prediction context cache write failed for %s: %s", cache_path, exc)


def _is_friction_feature_name(feature_name: str) -> bool:
    name = str(feature_name).lower()
    if name == "offline":
        return True
    if any(token in name for token in ("bitrate", "reachability", "allow_downgrade", "cloud_stats_events")):
        return False
    return any(
        token in name
        for token in (
            "error",
            "fatal",
            "stutter",
            "stall",
            "not_played",
            "fail",
            "connection_none",
            "offline",
        )
    )


def _friction_feature_indices(context_features: list[str]) -> list[int]:
    return [
        idx
        for idx, feature_name in enumerate(context_features)
        if _is_friction_feature_name(str(feature_name))
    ]


def load_prediction_input_context(
    run_dir: Path,
    data_dir: Path,
    include_video: bool,
    logger: logging.Logger,
) -> PredictionInputContext:
    signature = prediction_source_signature(
        run_dir=run_dir,
        data_dir=data_dir,
        include_video=include_video,
    )
    cache_path = _prediction_context_cache_path(run_dir, include_video=include_video)
    cached = _load_cached_prediction_input_context(
        cache_path=cache_path,
        signature=signature,
        logger=logger,
    )
    if cached is not None:
        return cached

    # Import data pipeline lazily so TensorFlow can initialize first on macOS.
    from .data import append_audio_features, append_technical_log_features, engineer_features, load_streaming_history

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
    df = append_technical_log_features(df, data_dir=data_dir.expanduser().resolve(), logger=logger)
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

    context_frame = df[context_features].to_numpy(dtype="float32", copy=True)
    for key in skew_features:
        if key in context_features:
            idx = context_features.index(key)
            context_frame[:, idx] = np.log1p(np.maximum(context_frame[:, idx], 0.0)).astype("float32", copy=False)

    context_row = context_frame[-1].copy()

    scaler_path = run_dir / "context_scaler.joblib"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Missing scaler file: {scaler_path}")
    scaler = joblib.load(scaler_path)
    context_scaled = scaler.transform(context_row.reshape(1, -1)).astype("float32")

    friction_indices = _friction_feature_indices(context_features)
    friction_reference: dict[str, object] | None = None
    if friction_indices:
        friction_values = np.asarray(context_frame[:, friction_indices], dtype="float32")
        friction_medians = np.nanmedian(friction_values, axis=0).astype("float32", copy=False)
        centered = np.maximum(friction_values - friction_medians.reshape(1, -1), 0.0)
        aggregate = np.sum(centered, axis=1)
        positive_aggregate = aggregate[aggregate > 0.0]
        if len(positive_aggregate):
            friction_threshold = max(
                float(np.quantile(aggregate, 0.75)) if len(aggregate) else 0.0,
                float(np.quantile(positive_aggregate, 0.50)),
            )
        else:
            friction_threshold = 0.0
        friction_reference = {
            "feature_names": [context_features[idx] for idx in friction_indices],
            "median_values": friction_medians.astype("float32", copy=False),
            "aggregate_threshold": friction_threshold,
        }

    scaler_mean = getattr(scaler, "mean_", None)
    scaler_scale = getattr(scaler, "scale_", None)
    context = PredictionInputContext(
        artist_labels=artist_labels,
        artist_to_label=artist_to_label,
        sequence_length=sequence_length,
        latest_sequence_labels=latest_sequence_labels,
        latest_sequence_names=latest_sequence_names,
        context_scaled=context_scaled,
        context_raw=context_row.reshape(1, -1).astype("float32"),
        context_features=context_features,
        friction_reference=friction_reference,
        scaler_mean=np.asarray(scaler_mean, dtype="float32") if scaler_mean is not None else None,
        scaler_scale=np.asarray(scaler_scale, dtype="float32") if scaler_scale is not None else None,
    )
    _store_prediction_input_context_cache(
        cache_path=cache_path,
        signature=signature,
        context=context,
        logger=logger,
    )
    return context


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


def _build_public_catalog_client(
    args: argparse.Namespace,
    logger: logging.Logger,
) -> SpotifyPublicCatalogClient | None:
    if not bool(args.spotify_public_metadata):
        return None

    client = SpotifyPublicCatalogClient.from_env(market=str(args.spotify_market or "US"))
    if client is None:
        logger.warning(
            "Spotify public metadata was requested but SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET are not set. "
            "Continuing without enrichment."
        )
        return None
    return client


def _metadata_suffix(metadata: SpotifyArtistMetadata | None) -> str:
    if metadata is None:
        return ""

    details: list[str] = []
    if metadata.popularity is not None:
        details.append(f"popularity={metadata.popularity}")
    if metadata.followers_total is not None:
        details.append(f"followers={metadata.followers_total}")
    if metadata.genres:
        details.append("genres=" + ", ".join(metadata.genres[:3]))
    if metadata.spotify_url:
        details.append(f"url={metadata.spotify_url}")
    if not details:
        return ""
    return " | spotify: " + " | ".join(details)


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
    feature_metadata = _load_json(run_dir / "feature_metadata.json")
    artist_labels = list(feature_metadata["artist_labels"])
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
    public_catalog_client = _build_public_catalog_client(args, logger)

    top_k = max(1, int(args.top_k))
    top_indices = np.argsort(artist_probs)[::-1][:top_k]

    print(f"model={predictor.model_name}")
    print(f"model_type={predictor.model_type}")
    print("sequence_tail=" + " | ".join(sequence_names))
    print("top_predictions:")
    for rank, label_idx in enumerate(top_indices, start=1):
        artist_name = artist_labels[int(label_idx)]
        prob = float(artist_probs[int(label_idx)])
        artist_metadata: SpotifyArtistMetadata | None = None
        if public_catalog_client is not None:
            try:
                artist_metadata = public_catalog_client.search_artist(artist_name)
            except SpotifyPublicCatalogError as exc:
                logger.warning("Spotify public metadata lookup failed for %s: %s", artist_name, exc)
                public_catalog_client = None
        print(f"{rank}. {artist_name} ({prob:.4f}){_metadata_suffix(artist_metadata)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
