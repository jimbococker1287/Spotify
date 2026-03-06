from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import logging
import sys

DEFAULT_MODEL_NAMES: tuple[str, ...] = (
    "gru_artist",
    "memory_net_artist",
    "lstm",
    "transformer",
    "dense",
    "tcn",
    "cnn_lstm",
    "gru",
    "cnn",
    "attention_rnn",
    "tft",
    "transformer_xl",
    "memory_net",
    "graph_seq",
)

CANONICAL_AUDIO_FILES: tuple[str, ...] = (
    "Streaming_History_Audio_2014-2022_0.json",
    "Streaming_History_Audio_2022_1.json",
    "Streaming_History_Audio_2022_2.json",
    "Streaming_History_Audio_2022-2023_3.json",
    "Streaming_History_Audio_2023_4.json",
    "Streaming_History_Audio_2023-2024_5.json",
    "Streaming_History_Audio_2024_6.json",
    "Streaming_History_Audio_2024_7.json",
)

CANONICAL_VIDEO_FILE = "Streaming_History_Video_2018-2024.json"

DEFAULT_BATCH_SIZE = 1024
DEFAULT_EPOCHS = 50
DEFAULT_SEQUENCE_LENGTH = 30
DEFAULT_MAX_ARTISTS = 200
DEFAULT_SEED = 42

PROFILE_PRESETS: dict[str, dict[str, object]] = {
    "dev": {
        "batch_size": 256,
        "epochs": 2,
        "sequence_length": DEFAULT_SEQUENCE_LENGTH,
        "max_artists": 40,
        "random_seed": DEFAULT_SEED,
        "include_video": False,
        "enable_spotify_features": False,
        "enable_shap": False,
        "model_names": ("dense", "lstm"),
    },
    "small": {
        "batch_size": 512,
        "epochs": 10,
        "sequence_length": DEFAULT_SEQUENCE_LENGTH,
        "max_artists": 120,
        "random_seed": DEFAULT_SEED,
        "include_video": True,
        "enable_spotify_features": False,
        "enable_shap": False,
        "model_names": ("dense", "gru", "lstm", "transformer"),
    },
    "full": {
        "batch_size": DEFAULT_BATCH_SIZE,
        "epochs": DEFAULT_EPOCHS,
        "sequence_length": DEFAULT_SEQUENCE_LENGTH,
        "max_artists": DEFAULT_MAX_ARTISTS,
        "random_seed": DEFAULT_SEED,
        "include_video": True,
        "enable_spotify_features": True,
        "enable_shap": True,
        "model_names": DEFAULT_MODEL_NAMES,
    },
}


@dataclass(frozen=True)
class PipelineConfig:
    project_root: Path
    data_dir: Path
    output_dir: Path
    db_path: Path
    scaler_path: Path
    log_path: Path
    profile: str = "full"
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH
    max_artists: int = DEFAULT_MAX_ARTISTS
    batch_size: int = DEFAULT_BATCH_SIZE
    epochs: int = DEFAULT_EPOCHS
    random_seed: int = DEFAULT_SEED
    include_video: bool = True
    enable_spotify_features: bool = True
    enable_shap: bool = True
    model_names: tuple[str, ...] = DEFAULT_MODEL_NAMES


def _default_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _split_models(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return DEFAULT_MODEL_NAMES
    model_names = tuple(part.strip() for part in raw.split(",") if part.strip())
    return model_names or DEFAULT_MODEL_NAMES


def _resolve_profile(name: str | None) -> tuple[str, dict[str, object]]:
    profile_name = (name or "full").strip().lower()
    if profile_name not in PROFILE_PRESETS:
        valid = ", ".join(sorted(PROFILE_PRESETS))
        raise ValueError(f"Unknown profile '{profile_name}'. Valid profiles: {valid}")
    return profile_name, dict(PROFILE_PRESETS[profile_name])


def build_config(args: argparse.Namespace) -> PipelineConfig:
    project_root = _default_project_root()
    profile_name, preset = _resolve_profile(args.profile)

    data_dir = Path(args.data_dir).expanduser().resolve() if args.data_dir else (project_root / "data" / "raw")
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else (project_root / "outputs")
    db_path = Path(args.db_path).expanduser().resolve() if args.db_path else (output_dir / "spotify_training.db")
    scaler_path = Path(args.scaler_path).expanduser().resolve() if args.scaler_path else (output_dir / "context_scaler.joblib")
    log_path = Path(args.log_path).expanduser().resolve() if args.log_path else (output_dir / "train.log")

    batch_size = int(args.batch) if args.batch is not None else int(preset["batch_size"])
    epochs = int(args.epochs) if args.epochs is not None else int(preset["epochs"])
    sequence_length = int(args.sequence_length) if args.sequence_length is not None else int(preset["sequence_length"])
    max_artists = int(args.max_artists) if args.max_artists is not None else int(preset["max_artists"])
    random_seed = int(args.seed) if args.seed is not None else int(preset["random_seed"])

    include_video = bool(preset["include_video"])
    if args.no_video:
        include_video = False

    enable_spotify_features = bool(preset["enable_spotify_features"])
    if args.no_spotify_features:
        enable_spotify_features = False

    enable_shap = bool(preset["enable_shap"])
    if args.no_shap:
        enable_shap = False

    model_names = _split_models(args.models) if args.models else tuple(preset["model_names"])

    return PipelineConfig(
        project_root=project_root,
        data_dir=data_dir,
        output_dir=output_dir,
        db_path=db_path,
        scaler_path=scaler_path,
        log_path=log_path,
        profile=profile_name,
        sequence_length=sequence_length,
        max_artists=max_artists,
        batch_size=batch_size,
        epochs=epochs,
        random_seed=random_seed,
        include_video=include_video,
        enable_spotify_features=enable_spotify_features,
        enable_shap=enable_shap,
        model_names=model_names,
    )


def configure_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("spotify")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


def add_cli_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", type=str, default=None, help="Directory containing Streaming_History_*.json files.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory where models/charts/histories are written.")
    parser.add_argument("--db-path", type=str, default=None, help="SQLite output path.")
    parser.add_argument("--scaler-path", type=str, default=None, help="Path for persisted context scaler.")
    parser.add_argument("--log-path", type=str, default=None, help="Path for pipeline log file.")
    parser.add_argument(
        "--profile",
        type=str,
        choices=("dev", "small", "full"),
        default="full",
        help="Preset run profile. CLI flags override profile values.",
    )

    parser.add_argument("--batch", type=int, default=None, help="Effective training batch size.")
    parser.add_argument("--epochs", type=int, default=None, help="Max epochs per model.")
    parser.add_argument("--sequence-length", type=int, default=None, help="Sequence window length.")
    parser.add_argument("--max-artists", type=int, default=None, help="Top-N artists to keep.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")

    parser.add_argument("--models", type=str, default=None, help="Comma-separated model names to train (overrides profile).")
    parser.add_argument("--no-video", action="store_true", help="Exclude Streaming_History_Video files.")
    parser.add_argument("--no-spotify-features", action="store_true", help="Skip Spotipy audio feature fetch.")
    parser.add_argument("--no-shap", action="store_true", help="Skip SHAP explainability step.")
