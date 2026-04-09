from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import random
from typing import Any

import numpy as np

from .config import PipelineConfig
from .pipeline_helpers import _write_json_artifact
from .pipeline_runtime_analysis import PipelineAnalysisDeps
from .pipeline_runtime_experiments import PipelineExperimentDeps
from .tracking import MlflowTracker


@dataclass
class PipelineRuntimeDeps:
    PreparedDataCacheInfo: type
    SKEW_CONTEXT_FEATURES: Any
    ResourceMonitor: Any
    VAL_KEY: str
    append_backtest_history: Any
    append_experiment_history: Any
    append_optuna_history: Any
    build_classical_feature_bundle: Any
    build_model_builders: Any
    build_probability_ensemble: Any
    compute_baselines: Any
    evaluate_champion_gate: Any
    load_or_prepare_training_data: Any
    load_streaming_history: Any
    persist_to_sqlite: Any
    plot_backtest_history: Any
    plot_history_best_runs: Any
    plot_learning_curves: Any
    plot_model_comparison: Any
    plot_optuna_best_runs: Any
    plot_run_leaderboard: Any
    refresh_analytics_database: Any
    resolve_cached_deep_training_artifacts: Any
    restore_deep_reporting_artifacts: Any
    run_classical_benchmarks: Any
    run_data_quality_gate: Any
    run_drift_diagnostics: Any
    run_extended_evaluation: Any
    run_friction_proxy_analysis: Any
    run_moonshot_lab: Any
    run_optuna_tuning: Any
    run_policy_simulation: Any
    run_robustness_slice_evaluation: Any
    run_shap_analysis: Any
    run_temporal_backtest: Any
    save_deep_reporting_artifacts: Any
    save_histories_json: Any
    save_utilization_plot: Any
    train_and_evaluate_models: Any
    train_retrieval_stack: Any
    write_ablation_summary: Any
    write_benchmark_protocol: Any
    write_control_room_report: Any
    write_experiment_registry: Any
    write_run_report: Any
    write_significance_summary: Any


@dataclass
class PipelineBootstrapOutputs:
    tracker: MlflowTracker
    deps: PipelineRuntimeDeps
    raw_df: Any
    prepared: Any
    cache_info_payload: dict[str, object]
    artist_labels: list[str]


def configure_pipeline_runtime_environment(*, config: PipelineConfig, run_dir: Path) -> None:
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


def _init_mlflow_tracker(*, config: PipelineConfig, logger, phase_recorder, run_id: str) -> MlflowTracker:
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
        return tracker


def load_pipeline_runtime_dependencies(*, phase_recorder) -> PipelineRuntimeDeps:
    with phase_recorder.phase("dependency_imports") as phase:
        from .analytics_db import refresh_analytics_database
        from .backtesting import run_temporal_backtest
        from .benchmarks import build_classical_feature_bundle, run_classical_benchmarks
        from .control_room import write_control_room_report
        from .data import PreparedDataCacheInfo, SKEW_CONTEXT_FEATURES, load_or_prepare_training_data, load_streaming_history
        from .data_quality import run_data_quality_gate
        from .drift import run_drift_diagnostics
        from .ensemble import build_probability_ensemble
        from .evaluation import run_extended_evaluation
        from .explainability import run_shap_analysis
        from .friction import run_friction_proxy_analysis
        from .governance import evaluate_champion_gate
        from .modeling import build_model_builders
        from .monitoring import ResourceMonitor
        from .moonshot_lab import run_moonshot_lab
        from .policy_eval import run_policy_simulation
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
        from .research_artifacts import (
            write_ablation_summary,
            write_benchmark_protocol,
            write_experiment_registry,
            write_significance_summary,
        )
        from .retrieval import train_retrieval_stack
        from .robustness import run_robustness_slice_evaluation
        from .training import compute_baselines, resolve_cached_deep_training_artifacts, train_and_evaluate_models
        from .tuning import run_optuna_tuning

        phase["import_group"] = "pipeline_dependencies"

    return PipelineRuntimeDeps(
        PreparedDataCacheInfo=PreparedDataCacheInfo,
        SKEW_CONTEXT_FEATURES=SKEW_CONTEXT_FEATURES,
        ResourceMonitor=ResourceMonitor,
        VAL_KEY=VAL_KEY,
        append_backtest_history=append_backtest_history,
        append_experiment_history=append_experiment_history,
        append_optuna_history=append_optuna_history,
        build_classical_feature_bundle=build_classical_feature_bundle,
        build_model_builders=build_model_builders,
        build_probability_ensemble=build_probability_ensemble,
        compute_baselines=compute_baselines,
        evaluate_champion_gate=evaluate_champion_gate,
        load_or_prepare_training_data=load_or_prepare_training_data,
        load_streaming_history=load_streaming_history,
        persist_to_sqlite=persist_to_sqlite,
        plot_backtest_history=plot_backtest_history,
        plot_history_best_runs=plot_history_best_runs,
        plot_learning_curves=plot_learning_curves,
        plot_model_comparison=plot_model_comparison,
        plot_optuna_best_runs=plot_optuna_best_runs,
        plot_run_leaderboard=plot_run_leaderboard,
        refresh_analytics_database=refresh_analytics_database,
        resolve_cached_deep_training_artifacts=resolve_cached_deep_training_artifacts,
        restore_deep_reporting_artifacts=restore_deep_reporting_artifacts,
        run_classical_benchmarks=run_classical_benchmarks,
        run_data_quality_gate=run_data_quality_gate,
        run_drift_diagnostics=run_drift_diagnostics,
        run_extended_evaluation=run_extended_evaluation,
        run_friction_proxy_analysis=run_friction_proxy_analysis,
        run_moonshot_lab=run_moonshot_lab,
        run_optuna_tuning=run_optuna_tuning,
        run_policy_simulation=run_policy_simulation,
        run_robustness_slice_evaluation=run_robustness_slice_evaluation,
        run_shap_analysis=run_shap_analysis,
        run_temporal_backtest=run_temporal_backtest,
        save_deep_reporting_artifacts=save_deep_reporting_artifacts,
        save_histories_json=save_histories_json,
        save_utilization_plot=save_utilization_plot,
        train_and_evaluate_models=train_and_evaluate_models,
        train_retrieval_stack=train_retrieval_stack,
        write_ablation_summary=write_ablation_summary,
        write_benchmark_protocol=write_benchmark_protocol,
        write_control_room_report=write_control_room_report,
        write_experiment_registry=write_experiment_registry,
        write_run_report=write_run_report,
        write_significance_summary=write_significance_summary,
    )


def bootstrap_pipeline_runtime(
    *,
    artifact_paths: list[Path],
    config: PipelineConfig,
    logger,
    phase_recorder,
    run_dir: Path,
    run_id: str,
) -> PipelineBootstrapOutputs:
    tracker = _init_mlflow_tracker(config=config, logger=logger, phase_recorder=phase_recorder, run_id=run_id)
    deps = load_pipeline_runtime_dependencies(phase_recorder=phase_recorder)

    with phase_recorder.phase("data_loading", include_video=config.include_video) as phase:
        raw_df = deps.load_streaming_history(
            config.data_dir,
            include_video=config.include_video,
            logger=logger,
        )
        phase["raw_rows"] = int(len(raw_df))
        phase["raw_columns"] = int(len(getattr(raw_df, "columns", [])))

    data_quality_report_path = run_dir / "data_quality_report.json"
    with phase_recorder.phase("data_quality_gate") as phase:
        deps.run_data_quality_gate(raw_df, report_path=data_quality_report_path, logger=logger)
        phase["report_path"] = data_quality_report_path
    artifact_paths.append(data_quality_report_path)

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

    assert isinstance(cache_info, deps.PreparedDataCacheInfo)
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
            "skew_context_features": list(deps.SKEW_CONTEXT_FEATURES),
            "artist_labels": artist_labels,
        },
        artifact_paths,
    )

    baseline_metrics = deps.compute_baselines(prepared, logger)
    tracker.log_params(
        {
            "data_records": len(prepared.df),
            "num_artists": prepared.num_artists,
            "num_context_features": prepared.num_ctx,
            **baseline_metrics,
        }
    )

    return PipelineBootstrapOutputs(
        tracker=tracker,
        deps=deps,
        raw_df=raw_df,
        prepared=prepared,
        cache_info_payload=cache_info_payload,
        artist_labels=artist_labels,
    )


def build_experiment_deps(*, runtime_deps: PipelineRuntimeDeps) -> PipelineExperimentDeps:
    return PipelineExperimentDeps(
        ResourceMonitor=runtime_deps.ResourceMonitor,
        VAL_KEY=runtime_deps.VAL_KEY,
        build_classical_feature_bundle=runtime_deps.build_classical_feature_bundle,
        build_model_builders=runtime_deps.build_model_builders,
        persist_to_sqlite=runtime_deps.persist_to_sqlite,
        plot_learning_curves=runtime_deps.plot_learning_curves,
        plot_model_comparison=runtime_deps.plot_model_comparison,
        resolve_cached_deep_training_artifacts=runtime_deps.resolve_cached_deep_training_artifacts,
        restore_deep_reporting_artifacts=runtime_deps.restore_deep_reporting_artifacts,
        run_classical_benchmarks=runtime_deps.run_classical_benchmarks,
        run_optuna_tuning=runtime_deps.run_optuna_tuning,
        run_shap_analysis=runtime_deps.run_shap_analysis,
        run_temporal_backtest=runtime_deps.run_temporal_backtest,
        save_deep_reporting_artifacts=runtime_deps.save_deep_reporting_artifacts,
        save_histories_json=runtime_deps.save_histories_json,
        save_utilization_plot=runtime_deps.save_utilization_plot,
        train_and_evaluate_models=runtime_deps.train_and_evaluate_models,
        train_retrieval_stack=runtime_deps.train_retrieval_stack,
    )


def build_analysis_deps(*, runtime_deps: PipelineRuntimeDeps) -> PipelineAnalysisDeps:
    return PipelineAnalysisDeps(
        append_backtest_history=runtime_deps.append_backtest_history,
        append_experiment_history=runtime_deps.append_experiment_history,
        append_optuna_history=runtime_deps.append_optuna_history,
        build_probability_ensemble=runtime_deps.build_probability_ensemble,
        evaluate_champion_gate=runtime_deps.evaluate_champion_gate,
        plot_backtest_history=runtime_deps.plot_backtest_history,
        plot_history_best_runs=runtime_deps.plot_history_best_runs,
        plot_optuna_best_runs=runtime_deps.plot_optuna_best_runs,
        plot_run_leaderboard=runtime_deps.plot_run_leaderboard,
        refresh_analytics_database=runtime_deps.refresh_analytics_database,
        run_drift_diagnostics=runtime_deps.run_drift_diagnostics,
        run_extended_evaluation=runtime_deps.run_extended_evaluation,
        run_friction_proxy_analysis=runtime_deps.run_friction_proxy_analysis,
        run_moonshot_lab=runtime_deps.run_moonshot_lab,
        run_policy_simulation=runtime_deps.run_policy_simulation,
        run_robustness_slice_evaluation=runtime_deps.run_robustness_slice_evaluation,
        write_ablation_summary=runtime_deps.write_ablation_summary,
        write_benchmark_protocol=runtime_deps.write_benchmark_protocol,
        write_control_room_report=runtime_deps.write_control_room_report,
        write_experiment_registry=runtime_deps.write_experiment_registry,
        write_run_report=runtime_deps.write_run_report,
        write_significance_summary=runtime_deps.write_significance_summary,
    )


__all__ = [
    "PipelineBootstrapOutputs",
    "PipelineRuntimeDeps",
    "bootstrap_pipeline_runtime",
    "build_analysis_deps",
    "build_experiment_deps",
    "configure_pipeline_runtime_environment",
    "load_pipeline_runtime_dependencies",
]
