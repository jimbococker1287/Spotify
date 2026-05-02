from __future__ import annotations

from pathlib import Path

from .config import PipelineConfig
from .pipeline_runtime_bootstrap_finalize import finalize_pipeline_runtime_inputs
from .pipeline_runtime_bootstrap_prepare import load_and_prepare_pipeline_training_data
from .pipeline_runtime_bootstrap_types import PipelinePreparedDataOutputs
from .pipeline_runtime_dependency_types import PipelineRuntimeDeps


def prepare_pipeline_runtime_inputs(
    *,
    artifact_paths: list[Path],
    config: PipelineConfig,
    deps: PipelineRuntimeDeps,
    logger,
    phase_recorder,
    run_dir: Path,
    tracker,
) -> PipelinePreparedDataOutputs:
    prepared_state = load_and_prepare_pipeline_training_data(
        artifact_paths=artifact_paths,
        config=config,
        deps=deps,
        logger=logger,
        phase_recorder=phase_recorder,
        run_dir=run_dir,
    )
    return finalize_pipeline_runtime_inputs(
        artifact_paths=artifact_paths,
        config=config,
        deps=deps,
        logger=logger,
        prepared_state=prepared_state,
        run_dir=run_dir,
        tracker=tracker,
    )


__all__ = ["prepare_pipeline_runtime_inputs"]
