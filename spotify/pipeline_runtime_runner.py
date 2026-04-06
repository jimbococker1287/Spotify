from __future__ import annotations

from pathlib import Path
import os
import random

import numpy as np

from .config import DEFAULT_MODEL_NAMES, PipelineConfig, configure_logging
from .pipeline_helpers import (
    _build_run_id,
    _track_file,
    _write_json_artifact,
)
from .pipeline_postrun import PipelinePostRunContext, PipelinePostRunDeps, run_pipeline_postrun
from .pipeline_runtime_experiments import (
    PipelineExperimentContext,
    PipelineExperimentDeps,
    run_experiment_stages,
)
from .run_artifacts import write_json
from .run_timing import RunPhaseRecorder
from .tracking import MlflowTracker


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

    isolate_mpl_cache = os.getenv("SPOTIFY_ISOLATE_MPL_CACHE", "0").strip().lower() in ("1", "true", "yes", "on")
    if isolate_mpl_cache:
        mpl_config_dir = run_dir / ".mplconfig"
        xdg_cache_dir = run_dir / ".cache"
    else:
        mpl_config_dir = config.output_dir / ".mplconfig"
        xdg_cache_dir = config.output_dir / ".cache"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    xdg_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache_dir))

    np.random.seed(config.random_seed)
    random.seed(config.random_seed)

    result_rows: list[dict[str, object]] = []
    optuna_rows: list[dict[str, object]] = []
    backtest_rows: list[dict[str, object]] = []
    cache_info_payload: dict[str, object] = {}
    artifact_paths: list[Path] = [run_dir / "train.log"]
    tracker: MlflowTracker | None = None
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

        with phase_recorder.phase("mlflow_tracking_init", enabled=config.enable_mlflow) as phase:
            tracker = MlflowTracker(
                enabled=config.enable_mlflow,
                run_id=run_id,
                run_name=config.run_name,
                tracking_uri=config.mlflow_tracking_uri,
                experiment_name=config.mlflow_experiment,
                default_tracking_dir=config.output_dir / "mlruns",
                logger=logger,
            )
            tracker.log_params(
                {
                    "profile": config.profile,
                    "run_name": config.run_name or "",
                    "sequence_length": config.sequence_length,
                    "max_artists": config.max_artists,
                    "batch_size": config.batch_size,
                    "epochs": config.epochs,
                    "random_seed": config.random_seed,
                    "include_video": config.include_video,
                    "enable_spotify_features": config.enable_spotify_features,
                    "enable_shap": config.enable_shap,
                    "classical_only": config.classical_only,
                    "deep_models": config.model_names,
                    "classical_models": config.classical_model_names,
                    "enable_retrieval_stack": config.enable_retrieval_stack,
                    "enable_self_supervised_pretraining": config.enable_self_supervised_pretraining,
                    "enable_friction_analysis": config.enable_friction_analysis,
                    "enable_moonshot_lab": config.enable_moonshot_lab,
                    "retrieval_candidate_k": config.retrieval_candidate_k,
                    "enable_optuna": config.enable_optuna,
                    "optuna_trials": config.optuna_trials,
                    "optuna_models": config.optuna_model_names,
                    "enable_temporal_backtest": config.enable_temporal_backtest,
                    "temporal_backtest_folds": config.temporal_backtest_folds,
                    "temporal_backtest_models": config.temporal_backtest_model_names,
                    "temporal_backtest_adaptation_mode": os.getenv("SPOTIFY_BACKTEST_ADAPTATION_MODE", "cold"),
                }
            )
            phase["tracking_enabled"] = bool(config.enable_mlflow)

        with phase_recorder.phase("dependency_imports") as phase:
            from .backtesting import run_temporal_backtest
            from .analytics_db import refresh_analytics_database
            from .benchmarks import build_classical_feature_bundle, run_classical_benchmarks
            from .control_room import write_control_room_report
            from .data import (
                PreparedDataCacheInfo,
                SKEW_CONTEXT_FEATURES,
                load_streaming_history,
                load_or_prepare_training_data,
            )
            from .data_quality import run_data_quality_gate
            from .ensemble import build_probability_ensemble
            from .governance import evaluate_champion_gate
            from .drift import run_drift_diagnostics
            from .explainability import run_shap_analysis
            from .evaluation import run_extended_evaluation
            from .friction import run_friction_proxy_analysis
            from .modeling import build_model_builders
            from .monitoring import ResourceMonitor
            from .moonshot_lab import run_moonshot_lab
            from .policy_eval import run_policy_simulation
            from .research_artifacts import (
                write_ablation_summary,
                write_benchmark_protocol,
                write_experiment_registry,
                write_significance_summary,
            )
            from .retrieval import train_retrieval_stack
            from .robustness import run_robustness_slice_evaluation
            from .reporting import (
                VAL_KEY,
                append_backtest_history,
                append_experiment_history,
                append_optuna_history,
                persist_to_sqlite,
                plot_backtest_history,
                plot_history_best_runs,
                plot_learning_curves,
                plot_model_comparison,
                plot_optuna_best_runs,
                plot_run_leaderboard,
                restore_deep_reporting_artifacts,
                save_deep_reporting_artifacts,
                save_histories_json,
                save_utilization_plot,
                write_run_report,
            )
            from .training import compute_baselines, resolve_cached_deep_training_artifacts, train_and_evaluate_models
            from .tuning import run_optuna_tuning
            phase["import_group"] = "pipeline_dependencies"

        with phase_recorder.phase("data_loading", include_video=config.include_video) as phase:
            raw_df = load_streaming_history(
                config.data_dir,
                include_video=config.include_video,
                logger=logger,
            )
            phase["raw_rows"] = int(len(raw_df))
            phase["raw_columns"] = int(len(getattr(raw_df, "columns", [])))
        data_quality_report_path = run_dir / "data_quality_report.json"
        with phase_recorder.phase("data_quality_gate") as phase:
            run_data_quality_gate(raw_df, report_path=data_quality_report_path, logger=logger)
            phase["report_path"] = data_quality_report_path
        artifact_paths.append(data_quality_report_path)

        with phase_recorder.phase(
            "prepare_training_data",
            max_artists=config.max_artists,
            sequence_length=config.sequence_length,
            enable_spotify_features=config.enable_spotify_features,
        ) as phase:
            prepared, cache_info = load_or_prepare_training_data(
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
        assert isinstance(cache_info, PreparedDataCacheInfo)
        cache_info_payload = {
            "enabled": cache_info.enabled,
            "hit": cache_info.hit,
            "fingerprint": cache_info.fingerprint,
            "cache_path": str(cache_info.cache_path) if cache_info.cache_path else "",
            "metadata_path": str(cache_info.metadata_path) if cache_info.metadata_path else "",
            "source_file_count": cache_info.source_file_count,
        }
        if cache_info.enabled:
            logger.info(
                "Prepared cache status: %s (fingerprint=%s)",
                ("HIT" if cache_info.hit else "MISS"),
                cache_info.fingerprint,
            )
        artifact_paths.append(run_dir / "context_scaler.joblib")

        artist_label_frame = (
            prepared.df[["artist_label", "master_metadata_album_artist_name"]]
            .drop_duplicates(subset=["artist_label"])
            .sort_values("artist_label")
        )
        artist_labels = artist_label_frame["master_metadata_album_artist_name"].astype(str).tolist()
        metadata_path = run_dir / "feature_metadata.json"
        _write_json_artifact(
            metadata_path,
            {
                "sequence_length": config.sequence_length,
                "context_features": list(prepared.context_features),
                "skew_context_features": list(SKEW_CONTEXT_FEATURES),
                "artist_labels": artist_labels,
            },
            artifact_paths,
        )

        baseline_metrics = compute_baselines(prepared, logger)
        tracker.log_params(
            {
                "data_records": len(prepared.df),
                "num_artists": prepared.num_artists,
                "num_context_features": prepared.num_ctx,
                **baseline_metrics,
            }
        )
        experiment_outputs = run_experiment_stages(
            context=PipelineExperimentContext(
                artifact_paths=artifact_paths,
                backtest_rows=backtest_rows,
                cache_fingerprint=cache_info.fingerprint,
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
            deps=PipelineExperimentDeps(
                ResourceMonitor=ResourceMonitor,
                VAL_KEY=VAL_KEY,
                build_classical_feature_bundle=build_classical_feature_bundle,
                build_model_builders=build_model_builders,
                persist_to_sqlite=persist_to_sqlite,
                plot_learning_curves=plot_learning_curves,
                plot_model_comparison=plot_model_comparison,
                resolve_cached_deep_training_artifacts=resolve_cached_deep_training_artifacts,
                restore_deep_reporting_artifacts=restore_deep_reporting_artifacts,
                run_classical_benchmarks=run_classical_benchmarks,
                run_optuna_tuning=run_optuna_tuning,
                run_shap_analysis=run_shap_analysis,
                run_temporal_backtest=run_temporal_backtest,
                save_deep_reporting_artifacts=save_deep_reporting_artifacts,
                save_histories_json=save_histories_json,
                save_utilization_plot=save_utilization_plot,
                train_and_evaluate_models=train_and_evaluate_models,
                train_retrieval_stack=train_retrieval_stack,
            ),
        )
        classical_feature_bundle = experiment_outputs.classical_feature_bundle

        if not result_rows:
            raise RuntimeError("No models were run. Enable deep and/or classical models.")

        ensemble_result = build_probability_ensemble(
            data=prepared,
            results=result_rows,
            sequence_length=config.sequence_length,
            run_dir=run_dir,
            logger=logger,
        )
        if ensemble_result is not None:
            result_rows.append(dict(ensemble_result.row))
            artifact_paths.extend(ensemble_result.artifact_paths)

        with phase_recorder.phase(
            "extended_evaluation",
            enable_conformal=config.enable_conformal,
            conformal_alpha=config.conformal_alpha,
        ) as phase:
            extended_eval_artifacts = run_extended_evaluation(
                data=prepared,
                results=result_rows,
                sequence_length=config.sequence_length,
                run_dir=run_dir,
                random_seed=config.random_seed,
                max_train_samples=config.classical_max_train_samples,
                enable_conformal=config.enable_conformal,
                conformal_alpha=config.conformal_alpha,
                logger=logger,
                feature_bundle=classical_feature_bundle,
            )
            artifact_paths.extend(extended_eval_artifacts)
            phase["artifact_count"] = int(len(extended_eval_artifacts))
        with phase_recorder.phase("drift_diagnostics") as phase:
            drift_artifacts = run_drift_diagnostics(
                data=prepared,
                sequence_length=config.sequence_length,
                output_dir=run_dir / "analysis",
                logger=logger,
            )
            artifact_paths.extend(drift_artifacts)
            phase["artifact_count"] = int(len(drift_artifacts))
        with phase_recorder.phase("robustness_slice_evaluation") as phase:
            robustness_artifacts = run_robustness_slice_evaluation(
                data=prepared,
                results=result_rows,
                sequence_length=config.sequence_length,
                run_dir=run_dir,
                logger=logger,
            )
            artifact_paths.extend(robustness_artifacts)
            phase["artifact_count"] = int(len(robustness_artifacts))
        with phase_recorder.phase("policy_simulation") as phase:
            policy_artifacts = run_policy_simulation(
                data=prepared,
                results=result_rows,
                run_dir=run_dir,
                logger=logger,
            )
            artifact_paths.extend(policy_artifacts)
            phase["artifact_count"] = int(len(policy_artifacts))
        if config.enable_friction_analysis:
            with phase_recorder.phase("friction_proxy_analysis") as phase:
                friction_artifacts = run_friction_proxy_analysis(
                    data=prepared,
                    output_dir=run_dir / "analysis",
                    logger=logger,
                )
                artifact_paths.extend(friction_artifacts)
                phase["artifact_count"] = int(len(friction_artifacts))
        else:
            phase_recorder.skip("friction_proxy_analysis", reason="friction_analysis_disabled")
            logger.info("Skipping friction proxy analysis for this run.")

        if config.enable_moonshot_lab:
            with phase_recorder.phase("moonshot_lab") as phase:
                moonshot_artifacts = run_moonshot_lab(
                    data=prepared,
                    results=result_rows,
                    run_dir=run_dir,
                    sequence_length=config.sequence_length,
                    artist_labels=artist_labels,
                    random_seed=config.random_seed,
                    logger=logger,
                )
                artifact_paths.extend(moonshot_artifacts)
                phase["artifact_count"] = int(len(moonshot_artifacts))
        else:
            phase_recorder.skip("moonshot_lab", reason="moonshot_lab_disabled")
            logger.info("Skipping moonshot lab for this run.")

        postrun_result = run_pipeline_postrun(
            context=PipelinePostRunContext(
                artifact_paths=artifact_paths,
                backtest_rows=backtest_rows,
                cache_info_payload=cache_info_payload,
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
            deps=PipelinePostRunDeps(
                append_backtest_history=append_backtest_history,
                append_experiment_history=append_experiment_history,
                append_optuna_history=append_optuna_history,
                evaluate_champion_gate=evaluate_champion_gate,
                plot_backtest_history=plot_backtest_history,
                plot_history_best_runs=plot_history_best_runs,
                plot_optuna_best_runs=plot_optuna_best_runs,
                plot_run_leaderboard=plot_run_leaderboard,
                refresh_analytics_database=refresh_analytics_database,
                write_ablation_summary=write_ablation_summary,
                write_benchmark_protocol=write_benchmark_protocol,
                write_control_room_report=write_control_room_report,
                write_experiment_registry=write_experiment_registry,
                write_run_report=write_run_report,
                write_significance_summary=write_significance_summary,
            ),
        )
        manifest_payload = postrun_result.manifest_payload
        strict_gate_error = postrun_result.strict_gate_error

        with phase_recorder.phase("leaderboard_logging") as phase:
            result_rows_sorted = sorted(
                result_rows,
                key=lambda row: float(row.get("val_top1", float("-inf"))),
                reverse=True,
            )
            logger.info("Run Leaderboard (sorted by val top-1):")
            for row in result_rows_sorted:
                fit_seconds = row.get("fit_seconds")
                fit_display = "n/a"
                if fit_seconds not in ("", None):
                    fit_display = f"{float(fit_seconds):.2f}"
                logger.info(
                    "%s [%s]: val_top1=%.4f test_top1=%.4f fit_s=%s",
                    row.get("model_name"),
                    row.get("model_type"),
                    float(row.get("val_top1", np.nan)),
                    float(row.get("test_top1", np.nan)),
                    fit_display,
                )
            phase["result_count"] = int(len(result_rows_sorted))

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
