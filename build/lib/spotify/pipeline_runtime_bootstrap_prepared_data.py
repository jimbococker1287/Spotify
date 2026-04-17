from __future__ import annotations

from pathlib import Path

from .config import PipelineConfig
from .pipeline_runtime_dependency_types import PipelineRuntimeDeps


def prepare_pipeline_training_data(
    *,
    artifact_paths: list[Path],
    config: PipelineConfig,
    deps: PipelineRuntimeDeps,
    logger,
    phase_recorder,
    raw_df,
    run_dir: Path,
):
    with phase_recorder.phase(
        "prepare_training_data",
        max_artists=config.max_artists,
        sequence_length=config.sequence_length,
        enable_spotify_features=config.enable_spotify_features,
    ) as phase:
        prepared, cache_info = deps.load_or_prepare_training_data(
            data_dir=config.data_dir,
            include_video=config.include_video,
            enable_spotify_features=config.enable_spotify_features,
            max_artists=config.max_artists,
            sequence_length=config.sequence_length,
            scaler_path=run_dir / "context_scaler.joblib",
            cache_root=config.output_dir / "cache" / "prepared_data",
            raw_df=raw_df,
            logger=logger,
        )
        phase["cache_enabled"] = bool(cache_info.enabled)
        phase["cache_hit"] = bool(cache_info.hit)
        phase["cache_fingerprint"] = cache_info.fingerprint
        phase["prepared_rows"] = int(len(prepared.df))
        phase["num_artists"] = int(prepared.num_artists)
        phase["num_context_features"] = int(prepared.num_ctx)

    artifact_paths.append(run_dir / "context_scaler.joblib")
    return prepared, cache_info


__all__ = ["prepare_pipeline_training_data"]
