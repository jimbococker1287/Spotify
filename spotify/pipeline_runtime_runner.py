from __future__ import annotations

from pathlib import Path
import os
import random
import sys

import numpy as np

from .config import DEFAULT_MODEL_NAMES, PipelineConfig, configure_logging
from .pipeline_helpers import (
    _append_existing_artifact_path,
    _build_run_id,
    _release_deep_runtime_resources,
    _track_file,
    _write_json_artifact,
)
from .pipeline_postrun import PipelinePostRunContext, PipelinePostRunDeps, run_pipeline_postrun
from .pipeline_runtime_shortlists import (
    _resolve_shortlist_top_n,
    _shortlist_classical_model_names,
    _tuned_backtest_specs,
)
from .run_artifacts import write_json
from .run_timing import RunPhaseRecorder
from .runtime import configure_process_env, load_tensorflow_runtime, select_distribution_strategy
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
    deep_cache_stats: dict[str, object] = {}
    classical_cache_stats: dict[str, object] = {}
    optuna_cache_stats: dict[str, object] = {}
    artifact_paths: list[Path] = [run_dir / "train.log"]
    tracker: MlflowTracker | None = None
    manifest_payload: dict[str, object] | None = None
    final_tracker_status = "FAILED"
    strict_gate_error: str | None = None
    selected_optuna_model_names = config.optuna_model_names
    selected_backtest_model_names = config.temporal_backtest_model_names
    tuned_results = []
    tuned_backtest_specs: dict[str, dict[str, object]] = {}
    deep_cache_plan = None

    try:
        run_deep_models = (not config.classical_only) and bool(config.model_names)
        run_classical_models = bool(config.enable_classical_models)
        if config.classical_only:
            run_classical_models = True
        run_deep_backtest = bool(config.enable_temporal_backtest) and any(
            model_name in DEFAULT_MODEL_NAMES for model_name in config.temporal_backtest_model_names
        )

        tf = None
        strategy = None

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

        if run_deep_models:
            deep_cache_plan = resolve_cached_deep_training_artifacts(
                data=prepared,
                selected_model_names=config.model_names,
                batch_size=config.batch_size,
                epochs=config.epochs,
                output_dir=run_dir,
                logger=logger,
                random_seed=config.random_seed,
                cache_root=config.output_dir / "cache" / "deep_training",
                cache_fingerprint=cache_info.fingerprint,
            )
            deep_cache_stats = {
                "enabled": bool(deep_cache_plan.enabled),
                "fingerprint": str(deep_cache_plan.fingerprint),
                "hit_model_names": list(deep_cache_plan.hit_model_names),
                "miss_model_names": list(deep_cache_plan.miss_model_names),
            }

        needs_tf_for_deep_training = bool(run_deep_models and deep_cache_plan is not None and deep_cache_plan.miss_model_names)
        needs_tf_for_deep_backtest = bool(run_deep_backtest)
        if needs_tf_for_deep_training or needs_tf_for_deep_backtest:
            with phase_recorder.phase(
                "tensorflow_runtime_init",
                run_deep_models=run_deep_models,
                run_deep_backtest=run_deep_backtest,
                deep_cache_hit_models=list(deep_cache_stats.get("hit_model_names", [])),
                deep_cache_miss_models=list(deep_cache_stats.get("miss_model_names", [])),
            ) as phase:
                configure_process_env()
                tf = load_tensorflow_runtime(logger)
                tf.random.set_seed(config.random_seed)
                strategy = select_distribution_strategy(tf, logger=logger)
                device_count = int(getattr(strategy, "num_replicas_in_sync", 1))
                phase["device_count"] = device_count
                phase["initialized_for_deep_training"] = bool(needs_tf_for_deep_training)
                phase["initialized_for_deep_backtest"] = bool(needs_tf_for_deep_backtest)
                phase["cache_hit_count"] = int(len(deep_cache_stats.get("hit_model_names", [])))
                phase["cache_miss_count"] = int(len(deep_cache_stats.get("miss_model_names", [])))
                logger.info("Number of devices: %s", device_count)
        else:
            phase_recorder.skip(
                "tensorflow_runtime_init",
                reason=(
                    "deep_training_fully_cached_and_deep_backtest_disabled"
                    if run_deep_models
                    else "deep_models_and_deep_backtest_disabled"
                ),
            )

        model_builders = None
        if run_deep_models:
            with phase_recorder.phase(
                "deep_model_training",
                model_names=list(config.model_names),
                batch_size=config.batch_size,
                epochs=config.epochs,
            ) as phase:
                deep_model_names_to_build = tuple(deep_cache_plan.miss_model_names) if deep_cache_plan is not None else config.model_names
                if deep_model_names_to_build:
                    model_builders = build_model_builders(
                        sequence_length=config.sequence_length,
                        num_artists=prepared.num_artists,
                        num_ctx=prepared.num_ctx,
                        selected_names=deep_model_names_to_build,
                    )
                else:
                    model_builders = []

                disable_monitor = os.getenv("SPOTIFY_DISABLE_MONITOR", "auto").strip().lower()
                monitor_enabled = bool(deep_model_names_to_build)
                if disable_monitor in ("1", "true", "yes", "on"):
                    monitor_enabled = False
                elif disable_monitor == "auto" and sys.platform == "darwin":
                    monitor_enabled = False

                monitor = ResourceMonitor(logger) if monitor_enabled else None
                if monitor is not None:
                    monitor.start()
                try:
                    artifacts = train_and_evaluate_models(
                        data=prepared,
                        model_builders=model_builders,
                        batch_size=config.batch_size,
                        epochs=config.epochs,
                        output_dir=run_dir,
                        strategy=strategy,
                        logger=logger,
                        random_seed=config.random_seed,
                        cache_root=config.output_dir / "cache" / "deep_training",
                        cache_fingerprint=cache_info.fingerprint,
                        cache_stats_out=deep_cache_stats,
                        cache_plan=deep_cache_plan,
                    )
                finally:
                    if monitor is not None:
                        monitor.stop()

                cpu_usage = monitor.cpu_usage if monitor is not None else []
                gpu_usage = monitor.gpu_usage if monitor is not None else []
                sqlite_path = run_dir / "spotify_training.db"
                restored_reporting = restore_deep_reporting_artifacts(
                    histories=artifacts.histories,
                    cpu_usage=cpu_usage,
                    gpu_usage=gpu_usage,
                    output_dir=run_dir,
                    db_path=sqlite_path,
                    cache_root=config.output_dir / "cache" / "deep_reporting",
                    cache_fingerprint=cache_info.fingerprint,
                )
                if restored_reporting is not None:
                    (
                        model_comparison_path,
                        learning_paths,
                        histories_path,
                        utilization_path,
                        sqlite_path,
                    ) = restored_reporting
                else:
                    model_comparison_path = plot_model_comparison(artifacts.histories, run_dir)
                    learning_paths = plot_learning_curves(artifacts.histories, run_dir)
                    histories_path = save_histories_json(artifacts.histories, run_dir)
                    utilization_path = save_utilization_plot(cpu_usage, gpu_usage, run_dir)
                    sqlite_path = persist_to_sqlite(
                        df=prepared.df,
                        histories=artifacts.histories,
                        cpu_usage=cpu_usage,
                        gpu_usage=gpu_usage,
                        db_path=sqlite_path,
                    )
                    save_deep_reporting_artifacts(
                        histories=artifacts.histories,
                        cpu_usage=cpu_usage,
                        gpu_usage=gpu_usage,
                        output_dir=run_dir,
                        db_path=sqlite_path,
                        cache_root=config.output_dir / "cache" / "deep_reporting",
                        cache_fingerprint=cache_info.fingerprint,
                    )
                artifact_paths.extend([model_comparison_path, histories_path, utilization_path, *learning_paths])

                if config.enable_shap:
                    run_shap_analysis(
                        artifacts.histories,
                        run_dir,
                        prepared,
                        logger,
                        cache_root=config.output_dir / "cache" / "shap",
                        cache_fingerprint=cache_info.fingerprint,
                    )
                else:
                    logger.info("Skipping SHAP analysis because --no-shap was set.")

                artifact_paths.append(sqlite_path)

                logger.info("Final Validation Artist Accuracy (Top-1 / Top-5):")
                for name, history in artifacts.histories.items():
                    val_key = VAL_KEY if VAL_KEY in history.history else "val_sparse_categorical_accuracy"
                    top1 = history.history[val_key][-1]
                    top5_key = "val_artist_output_top_5" if "val_artist_output_top_5" in history.history else "val_top_5"
                    top5 = history.history.get(top5_key, [np.nan])[-1]
                    logger.info("%s: Top-1=%.4f | Top-5=%.4f", name, top1, top5)

                    result_rows.append(
                        {
                            "model_name": name,
                            "model_type": "deep",
                            "model_family": "neural",
                            "val_top1": float(artifacts.val_metrics.get(name, {}).get("top1", np.nan)),
                            "val_top5": float(artifacts.val_metrics.get(name, {}).get("top5", np.nan)),
                            "val_ndcg_at5": float(artifacts.val_metrics.get(name, {}).get("ndcg_at5", np.nan)),
                            "val_mrr_at5": float(artifacts.val_metrics.get(name, {}).get("mrr_at5", np.nan)),
                            "val_coverage_at5": float(artifacts.val_metrics.get(name, {}).get("coverage_at5", np.nan)),
                            "val_diversity_at5": float(artifacts.val_metrics.get(name, {}).get("diversity_at5", np.nan)),
                            "test_top1": float(artifacts.test_metrics.get(name, {}).get("top1", np.nan)),
                            "test_top5": float(artifacts.test_metrics.get(name, {}).get("top5", np.nan)),
                            "test_ndcg_at5": float(artifacts.test_metrics.get(name, {}).get("ndcg_at5", np.nan)),
                            "test_mrr_at5": float(artifacts.test_metrics.get(name, {}).get("mrr_at5", np.nan)),
                            "test_coverage_at5": float(artifacts.test_metrics.get(name, {}).get("coverage_at5", np.nan)),
                            "test_diversity_at5": float(artifacts.test_metrics.get(name, {}).get("diversity_at5", np.nan)),
                            "fit_seconds": float(artifacts.fit_seconds.get(name, np.nan)),
                            "epochs": len(history.history.get("loss", [])),
                            "prediction_bundle_path": str(artifacts.prediction_bundle_paths.get(name, "")),
                        }
                    )
                    _append_existing_artifact_path(artifact_paths, artifacts.prediction_bundle_paths.get(name, ""))
                phase["trained_model_count"] = int(len(artifacts.histories))
                phase["monitor_enabled"] = bool(monitor_enabled)
                phase["cpu_samples"] = int(len(cpu_usage))
                phase["gpu_samples"] = int(len(gpu_usage))
                phase["cache_enabled"] = bool(deep_cache_stats.get("enabled", False))
                phase["cache_fingerprint"] = str(deep_cache_stats.get("fingerprint", ""))
                phase["cache_hit_models"] = list(deep_cache_stats.get("hit_model_names", []))
                phase["cache_hit_count"] = int(len(deep_cache_stats.get("hit_model_names", [])))
                phase["cache_miss_models"] = list(deep_cache_stats.get("miss_model_names", []))
                phase["cache_miss_count"] = int(len(deep_cache_stats.get("miss_model_names", [])))
                phase["reporting_cache_reused"] = bool(restored_reporting is not None)
        else:
            phase_recorder.skip("deep_model_training", reason="deep_models_disabled")
            logger.info("Skipping deep models for this run.")

        classical_feature_bundle = build_classical_feature_bundle(prepared) if run_classical_models else None
        classical_results = []

        if run_classical_models:
            with phase_recorder.phase(
                "classical_benchmarks",
                model_names=list(config.classical_model_names),
                max_train_samples=config.classical_max_train_samples,
                max_eval_samples=config.classical_max_eval_samples,
            ) as phase:
                classical_results = run_classical_benchmarks(
                    data=prepared,
                    output_dir=run_dir,
                    selected_models=config.classical_model_names,
                    random_seed=config.random_seed,
                    max_train_samples=config.classical_max_train_samples,
                    max_eval_samples=config.classical_max_eval_samples,
                    logger=logger,
                    feature_bundle=classical_feature_bundle,
                    cache_root=config.output_dir / "cache" / "classical_benchmarks",
                    cache_fingerprint=cache_info.fingerprint,
                    cache_stats_out=classical_cache_stats,
                )
                artifact_paths.append(run_dir / "classical_results.json")
                for row in classical_results:
                    result_rows.append(
                        {
                            "model_name": row.model_name,
                            "model_type": "classical",
                            "model_family": row.model_family,
                            "val_top1": row.val_top1,
                            "val_top5": row.val_top5,
                            "val_ndcg_at5": row.val_ndcg_at5,
                            "val_mrr_at5": row.val_mrr_at5,
                            "val_coverage_at5": row.val_coverage_at5,
                            "val_diversity_at5": row.val_diversity_at5,
                            "test_top1": row.test_top1,
                            "test_top5": row.test_top5,
                            "test_ndcg_at5": row.test_ndcg_at5,
                            "test_mrr_at5": row.test_mrr_at5,
                            "test_coverage_at5": row.test_coverage_at5,
                            "test_diversity_at5": row.test_diversity_at5,
                            "fit_seconds": row.fit_seconds,
                            "epochs": "",
                            "prediction_bundle_path": row.prediction_bundle_path,
                            "estimator_artifact_path": row.estimator_artifact_path,
                        }
                    )
                    _append_existing_artifact_path(artifact_paths, row.prediction_bundle_path)
                    _append_existing_artifact_path(artifact_paths, row.estimator_artifact_path)
                phase["model_count"] = int(len(classical_results))
                phase["cache_enabled"] = bool(classical_cache_stats.get("enabled", False))
                phase["cache_fingerprint"] = str(classical_cache_stats.get("fingerprint", ""))
                phase["cache_hit_models"] = list(classical_cache_stats.get("hit_model_names", []))
                phase["cache_hit_count"] = int(len(classical_cache_stats.get("hit_model_names", [])))
                phase["cache_miss_models"] = list(classical_cache_stats.get("miss_model_names", []))
                phase["cache_miss_count"] = int(len(classical_cache_stats.get("miss_model_names", [])))
        else:
            phase_recorder.skip("classical_benchmarks", reason="classical_models_disabled")
            logger.info("Skipping classical model benchmarks for this run.")

        if classical_results:
            selected_optuna_model_names = _shortlist_classical_model_names(
                config.optuna_model_names,
                classical_results,
                top_n=_resolve_shortlist_top_n("SPOTIFY_OPTUNA_SHORTLIST_TOP_N"),
                logger=logger,
                stage_label="Optuna",
            )
            selected_backtest_model_names = _shortlist_classical_model_names(
                config.temporal_backtest_model_names,
                classical_results,
                top_n=_resolve_shortlist_top_n("SPOTIFY_BACKTEST_SHORTLIST_TOP_N"),
                logger=logger,
                stage_label="Temporal backtest",
            )

        if run_classical_models and config.enable_optuna:
            optuna_dir = run_dir / "optuna"
            with phase_recorder.phase(
                "optuna_tuning",
                model_names=list(selected_optuna_model_names),
                candidate_model_names=list(config.optuna_model_names),
                trials=config.optuna_trials,
                timeout_seconds=config.optuna_timeout_seconds,
            ) as phase:
                tuned_results = run_optuna_tuning(
                    data=prepared,
                    output_dir=optuna_dir,
                    selected_models=selected_optuna_model_names,
                    random_seed=config.random_seed,
                    trials=config.optuna_trials,
                    timeout_seconds=config.optuna_timeout_seconds,
                    max_train_samples=config.classical_max_train_samples,
                    max_eval_samples=config.classical_max_eval_samples,
                    logger=logger,
                    feature_bundle=classical_feature_bundle,
                    cache_root=config.output_dir / "cache" / "optuna",
                    cache_fingerprint=cache_info.fingerprint,
                    cache_stats_out=optuna_cache_stats,
                )
                for row in tuned_results:
                    payload = {
                        "model_name": row.model_name,
                        "base_model_name": row.base_model_name,
                        "model_type": "classical_tuned",
                        "model_family": row.model_family,
                        "val_top1": row.val_top1,
                        "val_top5": row.val_top5,
                        "val_ndcg_at5": row.val_ndcg_at5,
                        "val_mrr_at5": row.val_mrr_at5,
                        "val_coverage_at5": row.val_coverage_at5,
                        "val_diversity_at5": row.val_diversity_at5,
                        "test_top1": row.test_top1,
                        "test_top5": row.test_top5,
                        "test_ndcg_at5": row.test_ndcg_at5,
                        "test_mrr_at5": row.test_mrr_at5,
                        "test_coverage_at5": row.test_coverage_at5,
                        "test_diversity_at5": row.test_diversity_at5,
                        "fit_seconds": row.fit_seconds,
                        "epochs": "",
                        "n_trials": row.n_trials,
                        "best_params": row.best_params,
                        "prediction_bundle_path": row.prediction_bundle_path,
                        "estimator_artifact_path": row.estimator_artifact_path,
                    }
                    result_rows.append(payload)
                    optuna_rows.append(payload)
                    _append_existing_artifact_path(artifact_paths, row.prediction_bundle_path)
                    _append_existing_artifact_path(artifact_paths, row.estimator_artifact_path)
                if optuna_dir.exists():
                    artifact_paths.extend(sorted(p for p in optuna_dir.glob("*") if p.is_file()))
                phase["result_count"] = int(len(tuned_results))
                phase["cache_enabled"] = bool(optuna_cache_stats.get("enabled", False))
                phase["cache_fingerprint"] = str(optuna_cache_stats.get("fingerprint", ""))
                phase["cache_hit_models"] = list(optuna_cache_stats.get("hit_model_names", []))
                phase["cache_hit_count"] = int(len(optuna_cache_stats.get("hit_model_names", [])))
                phase["cache_miss_models"] = list(optuna_cache_stats.get("miss_model_names", []))
                phase["cache_miss_count"] = int(len(optuna_cache_stats.get("miss_model_names", [])))
                selected_backtest_model_names, tuned_backtest_specs = _tuned_backtest_specs(
                    selected_backtest_model_names,
                    tuned_results,
                    logger=logger,
                )
        elif config.enable_optuna:
            phase_recorder.skip("optuna_tuning", reason="classical_models_disabled")
            logger.info("Skipping Optuna tuning because classical models are disabled.")
        else:
            phase_recorder.skip("optuna_tuning", reason="optuna_disabled")

        if config.enable_retrieval_stack:
            with phase_recorder.phase(
                "retrieval_stack",
                candidate_k=config.retrieval_candidate_k,
                enable_self_supervised_pretraining=config.enable_self_supervised_pretraining,
            ) as phase:
                retrieval_result = train_retrieval_stack(
                    data=prepared,
                    output_dir=run_dir,
                    random_seed=config.random_seed,
                    candidate_k=config.retrieval_candidate_k,
                    enable_self_supervised_pretraining=config.enable_self_supervised_pretraining,
                    logger=logger,
                )
                for row in retrieval_result.rows:
                    result_rows.append(dict(row))
                artifact_paths.extend(retrieval_result.artifact_paths)
                phase["result_count"] = int(len(retrieval_result.rows))
        else:
            phase_recorder.skip("retrieval_stack", reason="retrieval_disabled")
            logger.info("Skipping retrieval stack for this run.")

        if config.enable_temporal_backtest:
            backtest_dir = run_dir / "backtest"
            with phase_recorder.phase(
                "temporal_backtest",
                folds=config.temporal_backtest_folds,
                model_names=list(selected_backtest_model_names),
                candidate_model_names=list(config.temporal_backtest_model_names),
                adaptation_mode=os.getenv("SPOTIFY_BACKTEST_ADAPTATION_MODE", "cold"),
            ) as phase:
                deep_backtest_builders = model_builders
                if run_deep_backtest and deep_backtest_builders is None:
                    deep_backtest_names = tuple(
                        model_name for model_name in selected_backtest_model_names if model_name in DEFAULT_MODEL_NAMES
                    )
                    if deep_backtest_names:
                        deep_backtest_builders = build_model_builders(
                            sequence_length=config.sequence_length,
                            num_artists=prepared.num_artists,
                            num_ctx=prepared.num_ctx,
                            selected_names=deep_backtest_names,
                        )
                backtest_results = run_temporal_backtest(
                    data=prepared,
                    output_dir=backtest_dir,
                    selected_models=selected_backtest_model_names,
                    random_seed=config.random_seed,
                    folds=config.temporal_backtest_folds,
                    max_train_samples=config.classical_max_train_samples,
                    max_eval_samples=config.classical_max_eval_samples,
                    logger=logger,
                    feature_bundle=classical_feature_bundle,
                    deep_model_builders=deep_backtest_builders,
                    strategy=strategy,
                    adaptation_mode=os.getenv("SPOTIFY_BACKTEST_ADAPTATION_MODE", "cold"),
                    tuned_model_specs=tuned_backtest_specs,
                )
                for row in backtest_results:
                    backtest_rows.append(
                        {
                            "model_name": row.model_name,
                            "model_type": row.model_type,
                            "model_family": row.model_family,
                            "adaptation_mode": row.adaptation_mode,
                            "fold": row.fold,
                            "train_rows": row.train_rows,
                            "test_rows": row.test_rows,
                            "fit_seconds": row.fit_seconds,
                            "top1": row.top1,
                            "top5": row.top5,
                        }
                    )
                if backtest_dir.exists():
                    artifact_paths.extend(sorted(p for p in backtest_dir.glob("*") if p.is_file()))
                phase["row_count"] = int(len(backtest_rows))
                phase["deep_backtest_builders"] = bool(deep_backtest_builders)
        else:
            phase_recorder.skip("temporal_backtest", reason="temporal_backtest_disabled")

        if tf is not None:
            with phase_recorder.phase("release_deep_runtime_resources") as phase:
                _release_deep_runtime_resources(tf, logger)
                phase["tensorflow_loaded"] = True
        else:
            phase_recorder.skip("release_deep_runtime_resources", reason="tensorflow_not_initialized")

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
