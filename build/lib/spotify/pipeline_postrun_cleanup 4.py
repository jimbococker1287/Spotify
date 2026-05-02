from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
from typing import Any

from .artifact_cleanup import prune_mlflow_artifacts, prune_old_auxiliary_artifacts, prune_run_artifacts
from .pipeline_helpers import _write_json_artifact


@dataclass
class ArtifactCleanupOutcome:
    cleanup_summary: dict[str, object]
    retention_summary: dict[str, object]
    mlflow_artifact_cleanup_summary: dict[str, object]


def _cleanup_defaults() -> dict[str, object]:
    artifact_cleanup_mode = os.getenv("SPOTIFY_ARTIFACT_CLEANUP", "light")
    artifact_cleanup_min_mb_raw = os.getenv("SPOTIFY_ARTIFACT_CLEANUP_MIN_MB", "100").strip()
    try:
        artifact_cleanup_min_mb = max(0.0, float(artifact_cleanup_min_mb_raw))
    except Exception:
        artifact_cleanup_min_mb = 100.0
    prune_old_prediction_bundles_raw = os.getenv("SPOTIFY_PRUNE_OLD_PREDICTION_BUNDLES", "1").strip().lower()
    prune_old_prediction_bundles = prune_old_prediction_bundles_raw in ("1", "true", "yes", "on")
    prune_old_run_dbs_raw = os.getenv("SPOTIFY_PRUNE_OLD_RUN_DATABASES", "1").strip().lower()
    prune_old_run_dbs = prune_old_run_dbs_raw in ("1", "true", "yes", "on")
    keep_full_runs_raw = os.getenv("SPOTIFY_KEEP_FULL_RUNS", "2").strip()
    try:
        keep_full_runs = max(0, int(keep_full_runs_raw))
    except Exception:
        keep_full_runs = 2
    return {
        "cleanup_mode": artifact_cleanup_mode,
        "min_size_mb": artifact_cleanup_min_mb,
        "prune_prediction_bundles": prune_old_prediction_bundles,
        "prune_run_databases": prune_old_run_dbs,
        "keep_full_runs": keep_full_runs,
    }


def run_artifact_cleanup_and_retention(
    *,
    artifact_paths: list[Path],
    config: Any,
    logger,
    phase_recorder,
    result_rows: list[dict[str, object]],
    run_dir: Path,
    selected_model: tuple[str, str] | None,
) -> ArtifactCleanupOutcome:
    mlflow_artifact_cleanup_summary: dict[str, object] = {
        "enabled": False,
        "artifact_mode": "off",
        "max_artifact_mb": 0.0,
        "status": "not_run",
        "artifact_dir_count": 0,
        "artifact_dirs": [],
        "deleted_file_count": 0,
        "deleted_files": [],
        "freed_bytes": 0,
    }
    settings = _cleanup_defaults()

    with phase_recorder.phase("artifact_cleanup_and_retention") as phase:
        cleanup_summary = prune_run_artifacts(
            run_dir=run_dir,
            result_rows=result_rows,
            selected_model=selected_model,
            logger=logger,
            cleanup_mode=settings["cleanup_mode"],
            min_size_mb=settings["min_size_mb"],
        )
        _write_json_artifact(run_dir / "artifact_cleanup.json", cleanup_summary, artifact_paths)

        retention_summary = prune_old_auxiliary_artifacts(
            output_dir=config.output_dir,
            current_run_dir=run_dir,
            logger=logger,
            keep_last_full_runs=settings["keep_full_runs"],
            prune_prediction_bundles=settings["prune_prediction_bundles"],
            prune_run_databases=settings["prune_run_databases"],
        )
        _write_json_artifact(run_dir / "artifact_retention.json", retention_summary, artifact_paths)
        mlflow_artifact_cleanup_summary = prune_mlflow_artifacts(
            output_dir=config.output_dir,
            logger=logger,
        )
        _write_json_artifact(
            run_dir / "mlflow_artifact_cleanup.json",
            mlflow_artifact_cleanup_summary,
            artifact_paths,
        )
        phase["cleanup_mode"] = settings["cleanup_mode"]
        phase["cleanup_deleted_files"] = int(cleanup_summary.get("deleted_file_count", 0) or 0)
        phase["retention_deleted_prediction_bundles"] = int(
            retention_summary.get("deleted_prediction_bundle_count", 0) or 0
        )
        phase["mlflow_deleted_files"] = int(mlflow_artifact_cleanup_summary.get("deleted_file_count", 0) or 0)

    return ArtifactCleanupOutcome(
        cleanup_summary=cleanup_summary,
        retention_summary=retention_summary,
        mlflow_artifact_cleanup_summary=mlflow_artifact_cleanup_summary,
    )


__all__ = ["ArtifactCleanupOutcome", "run_artifact_cleanup_and_retention"]
