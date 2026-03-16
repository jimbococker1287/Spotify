from __future__ import annotations

from spotify.cli import build_parser
from spotify.config import build_config


def test_default_config_paths() -> None:
    parser = build_parser()
    args = parser.parse_args([])
    config = build_config(args)

    assert config.data_dir == (config.project_root / "data" / "raw")
    assert config.output_dir == (config.project_root / "outputs")
    assert config.db_path.name == "spotify_training.db"
    assert config.profile == "full"
    assert config.sequence_length == 30
    assert config.epochs == 50
    assert config.enable_spotify_features is False
    assert config.enable_classical_models is True
    assert config.enable_mlflow is True
    assert config.enable_optuna is True
    assert config.optuna_trials == 18
    assert config.optuna_timeout_seconds == 1_200
    assert config.enable_temporal_backtest is True
    assert config.temporal_backtest_folds == 4
    assert config.classical_max_train_samples == 50_000
    assert config.classical_max_eval_samples == 25_000
    assert set(config.optuna_model_names) == set(config.classical_model_names)
    assert set(config.temporal_backtest_model_names) == set(config.classical_model_names)


def test_model_subset_parsing() -> None:
    parser = build_parser()
    args = parser.parse_args(["--models", "gru,lstm", "--epochs", "7"])
    config = build_config(args)

    assert config.model_names == ("gru", "lstm")
    assert config.epochs == 7


def test_profile_defaults_are_applied() -> None:
    parser = build_parser()
    args = parser.parse_args(["--profile", "dev"])
    config = build_config(args)

    assert config.profile == "dev"
    assert config.epochs == 2
    assert config.batch_size == 256
    assert config.max_artists == 40
    assert config.model_names == ("dense", "lstm")
    assert config.classical_model_names == ("logreg", "random_forest", "knn")
    assert config.enable_mlflow is False
    assert config.enable_optuna is False
    assert config.enable_temporal_backtest is False


def test_fast_profile_defaults_are_applied() -> None:
    parser = build_parser()
    args = parser.parse_args(["--profile", "fast"])
    config = build_config(args)

    assert config.profile == "fast"
    assert config.epochs == 4
    assert config.model_names == ("dense", "gru_artist", "lstm")
    assert config.classical_model_names == ("logreg", "extra_trees", "mlp", "session_knn")
    assert config.enable_mlflow is True
    assert config.enable_optuna is True
    assert config.enable_temporal_backtest is True


def test_cli_overrides_profile_defaults() -> None:
    parser = build_parser()
    args = parser.parse_args(["--profile", "small", "--epochs", "3", "--batch", "64", "--no-video"])
    config = build_config(args)

    assert config.profile == "small"
    assert config.epochs == 3
    assert config.batch_size == 64
    assert config.include_video is False


def test_classical_only_disables_deep_models() -> None:
    parser = build_parser()
    args = parser.parse_args(["--profile", "full", "--classical-only"])
    config = build_config(args)

    assert config.classical_only is True
    assert config.enable_classical_models is True
    assert config.model_names == ()


def test_new_elite_flags_override_profile_defaults() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--profile",
            "dev",
            "--mlflow",
            "--optuna",
            "--optuna-trials",
            "5",
            "--optuna-models",
            "logreg,hist_gbm",
            "--temporal-backtest",
            "--backtest-folds",
            "4",
            "--backtest-models",
            "logreg,random_forest",
        ]
    )
    config = build_config(args)

    assert config.enable_mlflow is True
    assert config.enable_optuna is True
    assert config.optuna_trials == 5
    assert config.optuna_model_names == ("logreg", "hist_gbm")
    assert config.enable_temporal_backtest is True
    assert config.temporal_backtest_folds == 4
    assert config.temporal_backtest_model_names == ("logreg", "random_forest")


def test_classical_subset_becomes_default_for_optuna_and_backtest() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--profile",
            "small",
            "--classical-models",
            "logreg,random_forest",
        ]
    )
    config = build_config(args)

    assert config.classical_model_names == ("logreg", "random_forest")
    assert config.optuna_model_names == ("logreg", "random_forest")
    assert config.temporal_backtest_model_names == ("logreg", "random_forest")
