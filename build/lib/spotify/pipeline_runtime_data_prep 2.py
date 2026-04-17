from __future__ import annotations

from pathlib import Path

from .config import PipelineConfig
from .pipeline_runtime_bootstrap_data import prepare_pipeline_runtime_inputs
from .pipeline_runtime_bootstrap_types import PipelineBootstrapOutputs
from .pipeline_runtime_dependency_bundle import load_pipeline_runtime_dependencies
from .pipeline_runtime_environment import configure_pipeline_runtime_environment
from .pipeline_runtime_tracker import init_mlflow_tracker


def bootstrap_pipeline_runtime(
    *,
    artifact_paths: list[Path],
    config: PipelineConfig,
    logger,
    phase_recorder,
    run_dir: Path,
    run_id: str,
) -> PipelineBootstrapOutputs:
    tracker = init_mlflow_tracker(config=config, logger=logger, phase_recorder=phase_recorder, run_id=run_id)
    deps = load_pipeline_runtime_dependencies(phase_recorder=phase_recorder)
    prepared_inputs = prepare_pipeline_runtime_inputs(
        artifact_paths=artifact_paths,
        config=config,
        deps=deps,
        logger=logger,
        phase_recorder=phase_recorder,
        run_dir=run_dir,
        tracker=tracker,
    )
    return PipelineBootstrapOutputs(
        tracker=tracker,
        deps=deps,
        raw_df=prepared_inputs.raw_df,
        prepared=prepared_inputs.prepared,
        cache_info_payload=prepared_inputs.cache_info_payload,
        artist_labels=prepared_inputs.artist_labels,
    )


__all__ = [
    "PipelineBootstrapOutputs",
    "bootstrap_pipeline_runtime",
    "configure_pipeline_runtime_environment",
]
