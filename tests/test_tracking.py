from __future__ import annotations

from spotify.tracking import resolve_mlflow_artifact_policy, should_log_mlflow_artifact


def test_resolve_mlflow_artifact_policy_normalizes_light_mode() -> None:
    mode, max_artifact_mb = resolve_mlflow_artifact_policy(mode_raw="light", max_artifact_mb_raw="12.5")

    assert mode == "metadata"
    assert max_artifact_mb == 12.5


def test_should_log_mlflow_artifact_keeps_metadata_and_skips_binaries(tmp_path) -> None:
    metadata_path = tmp_path / "run_manifest.json"
    metadata_path.write_text("{}", encoding="utf-8")
    plot_path = tmp_path / "leaderboard.png"
    plot_path.write_bytes(b"plot")
    model_path = tmp_path / "classical_mlp.joblib"
    model_path.write_bytes(b"binary")
    sqlite_path = tmp_path / "spotify_training.db"
    sqlite_path.write_bytes(b"db")

    assert should_log_mlflow_artifact(metadata_path, mode="metadata", max_artifact_mb=25.0)
    assert should_log_mlflow_artifact(plot_path, mode="metadata", max_artifact_mb=25.0)
    assert not should_log_mlflow_artifact(model_path, mode="metadata", max_artifact_mb=25.0)
    assert not should_log_mlflow_artifact(sqlite_path, mode="metadata", max_artifact_mb=25.0)


def test_should_log_mlflow_artifact_respects_size_cap(tmp_path) -> None:
    csv_path = tmp_path / "large_metrics.csv"
    csv_path.write_bytes(b"x" * (2 * 1024 * 1024))

    assert not should_log_mlflow_artifact(csv_path, mode="metadata", max_artifact_mb=1.0)
    assert should_log_mlflow_artifact(csv_path, mode="metadata", max_artifact_mb=3.0)
