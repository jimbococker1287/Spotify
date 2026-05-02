from __future__ import annotations

from .pipeline_runtime_bootstrap_metadata import (
    build_cache_info_payload,
    collect_artist_labels,
    write_feature_metadata,
)
from .pipeline_runtime_bootstrap_metrics import (
    log_prepared_baseline_metrics,
    log_prepared_cache_status,
)
from .pipeline_runtime_bootstrap_types import PipelinePreparedDataOutputs, PipelinePreparedTrainingState


def finalize_pipeline_runtime_inputs(
    *,
    artifact_paths: list[Path],
    config: PipelineConfig,
    deps: PipelineRuntimeDeps,
    logger,
    prepared_state: PipelinePreparedTrainingState,
    run_dir: Path,
    tracker,
) -> PipelinePreparedDataOutputs:
    assert isinstance(prepared_state.cache_info, deps.PreparedDataCacheInfo)
    cache_info_payload = build_cache_info_payload(cache_info=prepared_state.cache_info)
    log_prepared_cache_status(cache_info=prepared_state.cache_info, logger=logger)

    artist_labels = collect_artist_labels(prepared=prepared_state.prepared)
    write_feature_metadata(
        artifact_paths=artifact_paths,
        artist_labels=artist_labels,
        config=config,
        deps=deps,
        prepared=prepared_state.prepared,
        run_dir=run_dir,
    )

    log_prepared_baseline_metrics(
        compute_baselines=deps.compute_baselines,
        logger=logger,
        prepared=prepared_state.prepared,
        tracker=tracker,
    )

    return PipelinePreparedDataOutputs(
        raw_df=prepared_state.raw_df,
        prepared=prepared_state.prepared,
        cache_info_payload=cache_info_payload,
        artist_labels=artist_labels,
    )


__all__ = ["finalize_pipeline_runtime_inputs"]
