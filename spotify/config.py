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
DEFAULT_CLASSICAL_MODEL_NAMES: tuple[str, ...] = (
    "logreg",
    "random_forest",
    "extra_trees",
    "hist_gbm",
    "knn",
    "gaussian_nb",
    "mlp",
)
DEFAULT_OPTUNA_MODEL_NAMES: tuple[str, ...] = DEFAULT_CLASSICAL_MODEL_NAMES
DEFAULT_BACKTEST_MODEL_NAMES: tuple[str, ...] = DEFAULT_CLASSICAL_MODEL_NAMES

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
        "enable_classical_models": True,
        "model_names": ("dense", "lstm"),
        "classical_model_names": ("logreg", "random_forest", "knn"),
        "classical_max_train_samples": 10_000,
        "classical_max_eval_samples": 8_000,
        "enable_mlflow": False,
        "mlflow_tracking_uri": None,
        "mlflow_experiment": "spotify-experiment-lab",
        "enable_optuna": False,
        "optuna_trials": 0,
        "optuna_timeout_seconds": 0,
        "optuna_model_names": DEFAULT_OPTUNA_MODEL_NAMES,
        "enable_temporal_backtest": False,
        "temporal_backtest_folds": 0,
        "temporal_backtest_model_names": DEFAULT_BACKTEST_MODEL_NAMES,
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
        "enable_classical_models": True,
        "model_names": ("dense", "gru", "lstm", "transformer"),
        "classical_model_names": ("logreg", "random_forest", "extra_trees", "knn", "mlp"),
        "classical_max_train_samples": 25_000,
        "classical_max_eval_samples": 15_000,
        "enable_mlflow": True,
        "mlflow_tracking_uri": None,
        "mlflow_experiment": "spotify-experiment-lab",
        "enable_optuna": True,
        "optuna_trials": 12,
        "optuna_timeout_seconds": 600,
        "optuna_model_names": DEFAULT_OPTUNA_MODEL_NAMES,
        "enable_temporal_backtest": True,
        "temporal_backtest_folds": 3,
        "temporal_backtest_model_names": DEFAULT_BACKTEST_MODEL_NAMES,
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
        "enable_classical_models": True,
        "model_names": DEFAULT_MODEL_NAMES,
        "classical_model_names": DEFAULT_CLASSICAL_MODEL_NAMES,
        "classical_max_train_samples": 60_000,
        "classical_max_eval_samples": 30_000,
        "enable_mlflow": True,
        "mlflow_tracking_uri": None,
        "mlflow_experiment": "spotify-experiment-lab",
        "enable_optuna": True,
        "optuna_trials": 30,
        "optuna_timeout_seconds": 1_800,
        "optuna_model_names": DEFAULT_OPTUNA_MODEL_NAMES,
        "enable_temporal_backtest": True,
        "temporal_backtest_folds": 5,
        "temporal_backtest_model_names": DEFAULT_BACKTEST_MODEL_NAMES,
    },
}


@dataclass(frozen=True)
class PipelineConfig:
    project_root: Path
    data_dir: Path
    output_dir: Path
    db_path: Path  # base path, overridden per-run
    scaler_path: Path  # base path, overridden per-run
    log_path: Path  # base path, overridden per-run
    profile: str = "full"
    run_name: str | None = None
    sequence_length: int = DEFAULT_SEQUENCE_LENGTH
    max_artists: int = DEFAULT_MAX_ARTISTS
    batch_size: int = DEFAULT_BATCH_SIZE
    epochs: int = DEFAULT_EPOCHS
    random_seed: int = DEFAULT_SEED
    include_video: bool = True
    enable_spotify_features: bool = True
    enable_shap: bool = True
    enable_classical_models: bool = True
    classical_only: bool = False
    model_names: tuple[str, ...] = DEFAULT_MODEL_NAMES
    classical_model_names: tuple[str, ...] = DEFAULT_CLASSICAL_MODEL_NAMES
    classical_max_train_samples: int = 60_000
    classical_max_eval_samples: int = 30_000
    enable_mlflow: bool = False
    mlflow_tracking_uri: str | None = None
    mlflow_experiment: str = "spotify-experiment-lab"
    enable_optuna: bool = False
    optuna_trials: int = 0
    optuna_timeout_seconds: int = 0
    optuna_model_names: tuple[str, ...] = DEFAULT_OPTUNA_MODEL_NAMES
    enable_temporal_backtest: bool = False
    temporal_backtest_folds: int = 0
    temporal_backtest_model_names: tuple[str, ...] = DEFAULT_BACKTEST_MODEL_NAMES


def _default_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _split_models(raw: str | None) -> tuple[str, ...]:
    if not raw:
        return DEFAULT_MODEL_NAMES
    model_names = tuple(part.strip() for part in raw.split(",") if part.strip())
    return model_names or DEFAULT_MODEL_NAMES


def _split_csv_models(raw: str | None, default: tuple[str, ...]) -> tuple[str, ...]:
    if not raw:
        return default
    model_names = tuple(part.strip() for part in raw.split(",") if part.strip())
    return model_names or default


def _split_classical_models(raw: str | None) -> tuple[str, ...]:
    return _split_csv_models(raw, DEFAULT_CLASSICAL_MODEL_NAMES)


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

    enable_classical_models = bool(preset["enable_classical_models"])
    if args.no_classical_models:
        enable_classical_models = False

    classical_only = bool(args.classical_only)
    if classical_only:
        enable_classical_models = True

    model_names = _split_models(args.models) if args.models else tuple(preset["model_names"])
    if classical_only:
        model_names = ()

    classical_model_names = (
        _split_classical_models(args.classical_models)
        if args.classical_models
        else tuple(preset["classical_model_names"])
    )

    classical_max_train_samples = (
        int(args.classical_max_train_samples)
        if args.classical_max_train_samples is not None
        else int(preset["classical_max_train_samples"])
    )
    classical_max_eval_samples = (
        int(args.classical_max_eval_samples)
        if args.classical_max_eval_samples is not None
        else int(preset["classical_max_eval_samples"])
    )

    if args.mlflow is None:
        enable_mlflow = bool(preset["enable_mlflow"])
    else:
        enable_mlflow = bool(args.mlflow)
    mlflow_tracking_uri = (
        str(args.mlflow_tracking_uri).strip() if args.mlflow_tracking_uri else preset.get("mlflow_tracking_uri")
    )
    mlflow_experiment = str(args.mlflow_experiment).strip() if args.mlflow_experiment else str(preset["mlflow_experiment"])

    if args.optuna is None:
        enable_optuna = bool(preset["enable_optuna"])
    else:
        enable_optuna = bool(args.optuna)
    optuna_trials = int(args.optuna_trials) if args.optuna_trials is not None else int(preset["optuna_trials"])
    optuna_timeout_seconds = (
        int(args.optuna_timeout_seconds)
        if args.optuna_timeout_seconds is not None
        else int(preset["optuna_timeout_seconds"])
    )
    optuna_model_names = (
        _split_csv_models(args.optuna_models, DEFAULT_OPTUNA_MODEL_NAMES)
        if args.optuna_models
        else (classical_model_names if args.classical_models else tuple(preset["optuna_model_names"]))
    )

    if args.temporal_backtest is None:
        enable_temporal_backtest = bool(preset["enable_temporal_backtest"])
    else:
        enable_temporal_backtest = bool(args.temporal_backtest)
    temporal_backtest_folds = (
        int(args.backtest_folds)
        if args.backtest_folds is not None
        else int(preset["temporal_backtest_folds"])
    )
    temporal_backtest_model_names = (
        _split_csv_models(args.backtest_models, DEFAULT_BACKTEST_MODEL_NAMES)
        if args.backtest_models
        else (classical_model_names if args.classical_models else tuple(preset["temporal_backtest_model_names"]))
    )

    return PipelineConfig(
        project_root=project_root,
        data_dir=data_dir,
        output_dir=output_dir,
        db_path=db_path,
        scaler_path=scaler_path,
        log_path=log_path,
        profile=profile_name,
        run_name=args.run_name,
        sequence_length=sequence_length,
        max_artists=max_artists,
        batch_size=batch_size,
        epochs=epochs,
        random_seed=random_seed,
        include_video=include_video,
        enable_spotify_features=enable_spotify_features,
        enable_shap=enable_shap,
        enable_classical_models=enable_classical_models,
        classical_only=classical_only,
        model_names=model_names,
        classical_model_names=classical_model_names,
        classical_max_train_samples=classical_max_train_samples,
        classical_max_eval_samples=classical_max_eval_samples,
        enable_mlflow=enable_mlflow,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment=mlflow_experiment,
        enable_optuna=enable_optuna,
        optuna_trials=optuna_trials,
        optuna_timeout_seconds=optuna_timeout_seconds,
        optuna_model_names=optuna_model_names,
        enable_temporal_backtest=enable_temporal_backtest,
        temporal_backtest_folds=temporal_backtest_folds,
        temporal_backtest_model_names=temporal_backtest_model_names,
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
    parser.set_defaults(mlflow=None, optuna=None, temporal_backtest=None)

    parser.add_argument("--data-dir", type=str, default=None, help="Directory containing Streaming_History_*.json files.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory where models/charts/histories are written.")
    parser.add_argument("--db-path", type=str, default=None, help="SQLite output path.")
    parser.add_argument("--scaler-path", type=str, default=None, help="Path for persisted context scaler.")
    parser.add_argument("--log-path", type=str, default=None, help="Path for pipeline log file.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional label appended to run id.")
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
    parser.add_argument(
        "--classical-models",
        type=str,
        default=None,
        help="Comma-separated classical model names to train (overrides profile).",
    )
    parser.add_argument(
        "--classical-max-train-samples",
        type=int,
        default=None,
        help="Cap training rows used for classical model benchmarking.",
    )
    parser.add_argument(
        "--classical-max-eval-samples",
        type=int,
        default=None,
        help="Cap validation/test rows used for classical model benchmarking.",
    )
    parser.add_argument("--no-classical-models", action="store_true", help="Disable classical ML model benchmarking.")
    parser.add_argument("--classical-only", action="store_true", help="Run only classical ML benchmarks (skip deep models).")
    parser.add_argument("--mlflow", dest="mlflow", action="store_true", help="Enable MLflow run tracking.")
    parser.add_argument("--no-mlflow", dest="mlflow", action="store_false", help="Disable MLflow run tracking.")
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=None,
        help="MLflow tracking URI. Defaults to sqlite:///.../outputs/mlruns/mlflow.db.",
    )
    parser.add_argument(
        "--mlflow-experiment",
        type=str,
        default=None,
        help="MLflow experiment name.",
    )
    parser.add_argument("--optuna", dest="optuna", action="store_true", help="Enable Optuna hyperparameter tuning.")
    parser.add_argument("--no-optuna", dest="optuna", action="store_false", help="Disable Optuna hyperparameter tuning.")
    parser.add_argument("--optuna-trials", type=int, default=None, help="Number of Optuna trials per tuned model.")
    parser.add_argument(
        "--optuna-timeout-seconds",
        type=int,
        default=None,
        help="Optuna timeout (seconds) per model. 0 disables timeout.",
    )
    parser.add_argument(
        "--optuna-models",
        type=str,
        default=None,
        help="Comma-separated classical model names to tune with Optuna.",
    )
    parser.add_argument(
        "--temporal-backtest",
        dest="temporal_backtest",
        action="store_true",
        help="Enable temporal backtesting on classical models.",
    )
    parser.add_argument(
        "--no-temporal-backtest",
        dest="temporal_backtest",
        action="store_false",
        help="Disable temporal backtesting on classical models.",
    )
    parser.add_argument("--backtest-folds", type=int, default=None, help="Number of temporal backtest folds.")
    parser.add_argument(
        "--backtest-models",
        type=str,
        default=None,
        help="Comma-separated classical model names for temporal backtesting.",
    )
    parser.add_argument("--no-video", action="store_true", help="Exclude Streaming_History_Video files.")
    parser.add_argument("--no-spotify-features", action="store_true", help="Skip Spotipy audio feature fetch.")
    parser.add_argument("--no-shap", action="store_true", help="Skip SHAP explainability step.")
