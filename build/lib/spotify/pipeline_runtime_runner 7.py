from __future__ import annotations

from .config import PipelineConfig
from .pipeline_runtime_bootstrap import (
    bootstrap_pipeline_runtime,
    build_analysis_deps,
    build_experiment_deps,
    configure_pipeline_runtime_environment,
)
from .pipeline_runtime_analysis import run_analysis_and_postrun
from .pipeline_runtime_experiments import run_experiment_stages
from .pipeline_runtime_runner_contexts import build_analysis_context, build_experiment_context
from .pipeline_runtime_runner_finalize import finalize_pipeline_run, track_pipeline_outputs
from .pipeline_runtime_runner_setup import initialize_pipeline_run, resolve_pipeline_run_policy


def run_pipeline(config: PipelineConfig) -> None:
    setup = initialize_pipeline_run(config=config)
    configure_pipeline_runtime_environment(config=config, run_dir=setup.run_dir)

    result_rows: list[dict[str, object]] = []
    optuna_rows: list[dict[str, object]] = []
    backtest_rows: list[dict[str, object]] = []
    cache_info_payload: dict[str, object] = {}
    tracker = None
    manifest_payload: dict[str, object] | None = None
    final_tracker_status = "FAILED"
    strict_gate_error: str | None = None

    try:
        policy = resolve_pipeline_run_policy(config=config, logger=setup.logger)

        bootstrap = bootstrap_pipeline_runtime(
            artifact_paths=setup.artifact_paths,
            config=config,
            logger=setup.logger,
            phase_recorder=setup.phase_recorder,
            run_dir=setup.run_dir,
            run_id=setup.run_id,
        )
        tracker = bootstrap.tracker
        runtime_deps = bootstrap.deps
        raw_df = bootstrap.raw_df
        prepared = bootstrap.prepared
        cache_info_payload = bootstrap.cache_info_payload
        artist_labels = bootstrap.artist_labels

        experiment_outputs = run_experiment_stages(
            context=build_experiment_context(
                backtest_rows=backtest_rows,
                cache_info_payload=cache_info_payload,
                config=config,
                optuna_rows=optuna_rows,
                policy=policy,
                prepared=prepared,
                result_rows=result_rows,
                setup=setup,
            ),
            deps=build_experiment_deps(runtime_deps=runtime_deps),
        )
        classical_feature_bundle = experiment_outputs.classical_feature_bundle

        if not result_rows:
            raise RuntimeError("No models were run. Enable deep and/or classical models.")

        analysis_outputs = run_analysis_and_postrun(
            context=build_analysis_context(
                artist_labels=artist_labels,
                backtest_rows=backtest_rows,
                cache_info_payload=cache_info_payload,
                classical_feature_bundle=classical_feature_bundle,
                config=config,
                optuna_rows=optuna_rows,
                policy=policy,
                prepared=prepared,
                raw_df=raw_df,
                result_rows=result_rows,
                setup=setup,
            ),
            deps=build_analysis_deps(runtime_deps=runtime_deps),
        )
        manifest_payload = analysis_outputs.manifest_payload
        strict_gate_error = analysis_outputs.strict_gate_error

        track_pipeline_outputs(
            artifact_paths=setup.artifact_paths,
            backtest_rows=backtest_rows,
            result_rows=result_rows,
            tracker=tracker,
        )

        if strict_gate_error is not None:
            raise RuntimeError(strict_gate_error)

        setup.logger.info("Pipeline completed successfully")
        final_tracker_status = "FINISHED"
    except KeyboardInterrupt:
        final_tracker_status = "KILLED"
        raise
    except Exception:
        final_tracker_status = "FAILED"
        raise
    finally:
        finalize_pipeline_run(
            final_tracker_status=final_tracker_status,
            logger=setup.logger,
            manifest_path=setup.manifest_path,
            manifest_payload=manifest_payload,
            phase_recorder=setup.phase_recorder,
            run_dir=setup.run_dir,
            tracker=tracker,
        )


__all__ = [
    "run_pipeline",
]
