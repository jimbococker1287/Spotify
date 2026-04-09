from __future__ import annotations

from pathlib import Path

from .config import DEFAULT_MODEL_NAMES, PipelineConfig, configure_logging
from .pipeline_runtime_bootstrap import (
    bootstrap_pipeline_runtime,
    build_analysis_deps,
    build_experiment_deps,
    configure_pipeline_runtime_environment,
)
from .pipeline_runtime_analysis import (
    PipelineAnalysisContext,
    run_analysis_and_postrun,
)
from .pipeline_helpers import (
    _build_run_id,
    _track_file,
)
from .pipeline_runtime_experiments import (
    PipelineExperimentContext,
    run_experiment_stages,
)
from .run_artifacts import write_json
from .run_timing import RunPhaseRecorder


def run_pipeline(config: PipelineConfig) -> None:
    run_id = _build_run_id(config)
    run_dir = config.output_dir / "runs" / run_id
    history_dir = config.output_dir / "history"
    manifest_path = run_dir / "run_manifest.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)

    logger = configure_logging(run_dir / "train.log")
    phase_recorder = RunPhaseRecorder(run_id=run_id)
    logger.info("Starting Spotify training pipeline")
    logger.info("Run ID: %s", run_id)
    if config.run_name:
        logger.info("Run Name: %s", config.run_name)
    logger.info("Data directory: %s", config.data_dir)
    logger.info("Output root: %s", config.output_dir)
    logger.info("Run output directory: %s", run_dir)

    configure_pipeline_runtime_environment(config=config, run_dir=run_dir)

    result_rows: list[dict[str, object]] = []
    optuna_rows: list[dict[str, object]] = []
    backtest_rows: list[dict[str, object]] = []
    cache_info_payload: dict[str, object] = {}
    artifact_paths: list[Path] = [run_dir / "train.log"]
    tracker = None
    manifest_payload: dict[str, object] | None = None
    final_tracker_status = "FAILED"
    strict_gate_error: str | None = None

    try:
        run_deep_models = (not config.classical_only) and bool(config.model_names)
        run_classical_models = bool(config.enable_classical_models)
        if config.classical_only:
            run_classical_models = True
        run_deep_backtest = bool(config.enable_temporal_backtest) and any(
            model_name in DEFAULT_MODEL_NAMES for model_name in config.temporal_backtest_model_names
        )

        bootstrap = bootstrap_pipeline_runtime(
            artifact_paths=artifact_paths,
            config=config,
            logger=logger,
            phase_recorder=phase_recorder,
            run_dir=run_dir,
            run_id=run_id,
        )
        tracker = bootstrap.tracker
        runtime_deps = bootstrap.deps
        raw_df = bootstrap.raw_df
        prepared = bootstrap.prepared
        cache_info_payload = bootstrap.cache_info_payload
        artist_labels = bootstrap.artist_labels

        experiment_outputs = run_experiment_stages(
            context=PipelineExperimentContext(
                artifact_paths=artifact_paths,
                backtest_rows=backtest_rows,
                cache_fingerprint=str(cache_info_payload.get("fingerprint", "")),
                config=config,
                logger=logger,
                optuna_rows=optuna_rows,
                phase_recorder=phase_recorder,
                prepared=prepared,
                result_rows=result_rows,
                run_classical_models=run_classical_models,
                run_deep_backtest=run_deep_backtest,
                run_deep_models=run_deep_models,
                run_dir=run_dir,
            ),
            deps=build_experiment_deps(runtime_deps=runtime_deps),
        )
        classical_feature_bundle = experiment_outputs.classical_feature_bundle

        if not result_rows:
            raise RuntimeError("No models were run. Enable deep and/or classical models.")

        analysis_outputs = run_analysis_and_postrun(
            context=PipelineAnalysisContext(
                artifact_paths=artifact_paths,
                artist_labels=artist_labels,
                backtest_rows=backtest_rows,
                cache_info_payload=cache_info_payload,
                classical_feature_bundle=classical_feature_bundle,
                config=config,
                history_dir=history_dir,
                logger=logger,
                manifest_path=manifest_path,
                optuna_rows=optuna_rows,
                phase_recorder=phase_recorder,
                prepared=prepared,
                raw_df=raw_df,
                result_rows=result_rows,
                run_classical_models=run_classical_models,
                run_dir=run_dir,
                run_id=run_id,
            ),
            deps=build_analysis_deps(runtime_deps=runtime_deps),
        )
        manifest_payload = analysis_outputs.manifest_payload
        strict_gate_error = analysis_outputs.strict_gate_error

        if tracker is not None:
            tracker.log_result_rows(result_rows)
            tracker.log_backtest_rows(backtest_rows)
        for path in artifact_paths:
            _track_file(tracker, path)

        if strict_gate_error is not None:
            raise RuntimeError(strict_gate_error)

        logger.info("Pipeline completed successfully")
        final_tracker_status = "FINISHED"
    except KeyboardInterrupt:
        final_tracker_status = "KILLED"
        raise
    except Exception:
        final_tracker_status = "FAILED"
        raise
    finally:
        try:
            phase_json_path, phase_csv_path, phase_payload = phase_recorder.write_artifacts(
                run_dir=run_dir,
                final_status=final_tracker_status,
            )
            slowest_phase = phase_payload.get("slowest_phase", {})
            if manifest_payload is not None:
                manifest_payload["pipeline_status"] = final_tracker_status
                manifest_payload["phase_timings"] = {
                    "json_path": str(phase_json_path),
                    "csv_path": str(phase_csv_path),
                    "total_seconds": float(phase_payload.get("total_seconds", 0.0)),
                    "measured_seconds": float(phase_payload.get("measured_seconds", 0.0)),
                    "unmeasured_overhead_seconds": float(phase_payload.get("unmeasured_overhead_seconds", 0.0)),
                    "phase_count": int(phase_payload.get("phase_count", 0)),
                    "completed_phase_count": int(phase_payload.get("completed_phase_count", 0)),
                    "non_skipped_phase_count": int(phase_payload.get("non_skipped_phase_count", 0)),
                    "slowest_phase": dict(slowest_phase) if isinstance(slowest_phase, dict) else {},
                    "slowest_phases": list(phase_payload.get("slowest_phases", [])),
                }
                write_json(manifest_path, manifest_payload)
            logger.info(
                "Recorded pipeline phase timings: total=%.2fs slowest=%s (%.2fs)",
                float(phase_payload.get("total_seconds", 0.0)),
                str((slowest_phase or {}).get("phase_name", "n/a")),
                float((slowest_phase or {}).get("duration_seconds", 0.0)),
            )
        except Exception as exc:
            logger.warning("Unable to persist pipeline phase timings: %s", exc)
        if tracker is not None:
            tracker.end(status=final_tracker_status)


__all__ = [
    "run_pipeline",
]
