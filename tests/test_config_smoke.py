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


def test_cli_overrides_profile_defaults() -> None:
    parser = build_parser()
    args = parser.parse_args(["--profile", "small", "--epochs", "3", "--batch", "64", "--no-video"])
    config = build_config(args)

    assert config.profile == "small"
    assert config.epochs == 3
    assert config.batch_size == 64
    assert config.include_video is False
