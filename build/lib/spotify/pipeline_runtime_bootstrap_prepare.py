from __future__ import annotations

from pathlib import Path

from .pipeline_runtime_bootstrap_loading import load_pipeline_runtime_raw_data
from .pipeline_runtime_bootstrap_prepared_data import prepare_pipeline_training_data
from .config import PipelineConfig
from .pipeline_runtime_bootstrap_types import PipelinePreparedTrainingState
from .pipeline_runtime_dependency_types import PipelineRuntimeDeps


def load_and_prepare_pipeline_training_data(
    *,
    artifact_paths: list[Path],
    config: PipelineConfig,
    deps: PipelineRuntimeDeps,
    logger,
    phase_recorder,
    run_dir: Path,
) -> PipelinePreparedTrainingState:
    raw_df = load_pipeline_runtime_raw_data(
        artifact_paths=artifact_paths,
        config=config,
        deps=deps,
        logger=logger,
        phase_recorder=phase_recorder,
        run_dir=run_dir,
    )
    prepared, cache_info = prepare_pipeline_training_data(
        artifact_paths=artifact_paths,
        config=config,
        deps=deps,
        logger=logger,
        phase_recorder=phase_recorder,
        raw_df=raw_df,
        run_dir=run_dir,
    )

    return PipelinePreparedTrainingState(
        raw_df=raw_df,
        prepared=prepared,
        cache_info=cache_info,
    )


__all__ = ["load_and_prepare_pipeline_training_data"]
