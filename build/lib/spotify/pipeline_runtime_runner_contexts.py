from __future__ import annotations

from .pipeline_runtime_analysis import PipelineAnalysisContext
from .pipeline_runtime_experiments import PipelineExperimentContext
from .pipeline_runtime_runner_types import PipelineRunPolicy, PipelineRunSetup


def build_experiment_context(
    *,
    backtest_rows: list[dict[str, object]],
    cache_info_payload: dict[str, object],
    config,
    optuna_rows: list[dict[str, object]],
    policy: PipelineRunPolicy,
    prepared,
    result_rows: list[dict[str, object]],
    setup: PipelineRunSetup,
) -> PipelineExperimentContext:
    return PipelineExperimentContext(
        artifact_paths=setup.artifact_paths,
        backtest_rows=backtest_rows,
        cache_fingerprint=str(cache_info_payload.get("fingerprint", "")),
        config=config,
        logger=setup.logger,
        optuna_rows=optuna_rows,
        phase_recorder=setup.phase_recorder,
        prepared=prepared,
        result_rows=result_rows,
        run_classical_models=policy.run_classical_models,
        run_deep_backtest=policy.run_deep_backtest,
        run_deep_models=policy.run_deep_models,
        run_dir=setup.run_dir,
    )


def build_analysis_context(
    *,
    artist_labels: list[str],
    backtest_rows: list[dict[str, object]],
    cache_info_payload: dict[str, object],
    classical_feature_bundle,
    config,
    optuna_rows: list[dict[str, object]],
    policy: PipelineRunPolicy,
    prepared,
    raw_df,
    result_rows: list[dict[str, object]],
    setup: PipelineRunSetup,
) -> PipelineAnalysisContext:
    return PipelineAnalysisContext(
        artifact_paths=setup.artifact_paths,
        artist_labels=artist_labels,
        backtest_rows=backtest_rows,
        cache_info_payload=cache_info_payload,
        classical_feature_bundle=classical_feature_bundle,
        config=config,
        history_dir=setup.history_dir,
        logger=setup.logger,
        manifest_path=setup.manifest_path,
        optuna_rows=optuna_rows,
        phase_recorder=setup.phase_recorder,
        prepared=prepared,
        raw_df=raw_df,
        result_rows=result_rows,
        run_classical_models=policy.run_classical_models,
        run_dir=setup.run_dir,
        run_id=setup.run_id,
    )


__all__ = [
    "build_analysis_context",
    "build_experiment_context",
]
