from __future__ import annotations

from datetime import datetime
from pathlib import Path
import os
import random
import sys

import numpy as np

from .artifact_cleanup import prune_mlflow_artifacts, prune_old_auxiliary_artifacts, prune_run_artifacts
from .champion_alias import best_serveable_model, write_champion_alias
from .config import DEFAULT_MODEL_NAMES, PipelineConfig, configure_logging
from .pipeline_helpers import (
    _analysis_prefix_for_model_type,
    _append_existing_artifact_path,
    _build_run_id,
    _load_current_risk_metrics,
    _release_deep_runtime_resources,
    _track_file,
    _write_json_artifact,
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
    champion_alias_payload: dict[str, object] = {
        "updated": False,
        "alias_file": "",
        "run_id": "",
        "run_dir": "",
        "model_name": "",
        "reason": "gate_not_evaluated",
    }
    mlflow_artifact_cleanup_summary: dict[str, object] = {
        "enabled": False,
        "artifact_mode": "off",
        "max_artifact_mb": 0.0,
        "status": "not_run",
        "artifact_dir_count": 0,
        "artifact_dirs": [],
        "deleted_file_count": 0,
        "deleted_files": [],
        "freed_bytes": 0,
    }
    artifact_paths: list[Path] = [run_dir / "train.log"]
    tracker: MlflowTracker | None = None
    manifest_payload: dict[str, object] | None = None
    final_tracker_status = "FAILED"

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
        if run_deep_models or run_deep_backtest:
            with phase_recorder.phase(
                "tensorflow_runtime_init",
                run_deep_models=run_deep_models,
                run_deep_backtest=run_deep_backtest,
            ) as phase:
                configure_process_env()
                tf = load_tensorflow_runtime(logger)
                tf.random.set_seed(config.random_seed)
                strategy = select_distribution_strategy(tf, logger=logger)
                device_count = int(getattr(strategy, "num_replicas_in_sync", 1))
                phase["device_count"] = device_count
                logger.info("Number of devices: %s", device_count)
        else:
            phase_recorder.skip(
                "tensorflow_runtime_init",
                reason="deep_models_and_deep_backtest_disabled",
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
                save_histories_json,
                save_utilization_plot,
                write_run_report,
            )
            from .training import compute_baselines, train_and_evaluate_models
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

        model_builders = None
        if run_deep_models:
            with phase_recorder.phase(
                "deep_model_training",
                model_names=list(config.model_names),
                batch_size=config.batch_size,
                epochs=config.epochs,
            ) as phase:
                model_builders = build_model_builders(
                    sequence_length=config.sequence_length,
                    num_artists=prepared.num_artists,
                    num_ctx=prepared.num_ctx,
                    selected_names=config.model_names,
                )

                disable_monitor = os.getenv("SPOTIFY_DISABLE_MONITOR", "auto").strip().lower()
                monitor_enabled = True
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
                    )
                finally:
                    if monitor is not None:
                        monitor.stop()

                model_comparison_path = plot_model_comparison(artifacts.histories, run_dir)
                learning_paths = plot_learning_curves(artifacts.histories, run_dir)
                histories_path = save_histories_json(artifacts.histories, run_dir)
                cpu_usage = monitor.cpu_usage if monitor is not None else []
                gpu_usage = monitor.gpu_usage if monitor is not None else []
                utilization_path = save_utilization_plot(cpu_usage, gpu_usage, run_dir)
                artifact_paths.extend([model_comparison_path, histories_path, utilization_path, *learning_paths])

                if config.enable_shap:
                    run_shap_analysis(artifacts.histories, run_dir, prepared, logger)
                else:
                    logger.info("Skipping SHAP analysis because --no-shap was set.")

                sqlite_path = persist_to_sqlite(
                    df=prepared.df,
                    histories=artifacts.histories,
                    cpu_usage=cpu_usage,
                    gpu_usage=gpu_usage,
                    db_path=run_dir / "spotify_training.db",
                )
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
        else:
            phase_recorder.skip("deep_model_training", reason="deep_models_disabled")
            logger.info("Skipping deep models for this run.")

        classical_feature_bundle = build_classical_feature_bundle(prepared) if run_classical_models else None

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
        else:
            phase_recorder.skip("classical_benchmarks", reason="classical_models_disabled")
            logger.info("Skipping classical model benchmarks for this run.")

        if run_classical_models and config.enable_optuna:
            optuna_dir = run_dir / "optuna"
            with phase_recorder.phase(
                "optuna_tuning",
                model_names=list(config.optuna_model_names),
                trials=config.optuna_trials,
                timeout_seconds=config.optuna_timeout_seconds,
            ) as phase:
                tuned_results = run_optuna_tuning(
                    data=prepared,
                    output_dir=optuna_dir,
                    selected_models=config.optuna_model_names,
                    random_seed=config.random_seed,
                    trials=config.optuna_trials,
                    timeout_seconds=config.optuna_timeout_seconds,
                    max_train_samples=config.classical_max_train_samples,
                    max_eval_samples=config.classical_max_eval_samples,
                    logger=logger,
                    feature_bundle=classical_feature_bundle,
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
                model_names=list(config.temporal_backtest_model_names),
                adaptation_mode=os.getenv("SPOTIFY_BACKTEST_ADAPTATION_MODE", "cold"),
            ) as phase:
                deep_backtest_builders = model_builders
                if run_deep_backtest and deep_backtest_builders is None:
                    deep_backtest_names = tuple(
                        model_name for model_name in config.temporal_backtest_model_names if model_name in DEFAULT_MODEL_NAMES
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
                    selected_models=config.temporal_backtest_model_names,
                    random_seed=config.random_seed,
                    folds=config.temporal_backtest_folds,
                    max_train_samples=config.classical_max_train_samples,
                    max_eval_samples=config.classical_max_eval_samples,
                    logger=logger,
                    feature_bundle=classical_feature_bundle,
                    deep_model_builders=deep_backtest_builders,
                    strategy=strategy,
                    adaptation_mode=os.getenv("SPOTIFY_BACKTEST_ADAPTATION_MODE", "cold"),
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

        with phase_recorder.phase("history_exports") as phase:
            leaderboard_path = plot_run_leaderboard(result_rows, run_dir)
            if leaderboard_path is not None:
                artifact_paths.append(leaderboard_path)

            history_csv = append_experiment_history(
                history_csv=history_dir / "experiment_history.csv",
                run_id=run_id,
                profile=config.profile,
                run_name=config.run_name,
                results=result_rows,
                data_records=len(prepared.df),
            )
            history_plot = plot_history_best_runs(history_csv, history_dir)
            artifact_paths.append(history_csv)
            if history_plot is not None:
                artifact_paths.append(history_plot)
            phase["leaderboard_plot_written"] = bool(leaderboard_path is not None)
            phase["experiment_history_rows"] = int(len(result_rows))

        champion_gate_threshold_raw = os.getenv("SPOTIFY_CHAMPION_GATE_MAX_REGRESSION", "0.005").strip()
        try:
            champion_gate_threshold = max(0.0, float(champion_gate_threshold_raw))
        except Exception:
            champion_gate_threshold = 0.005
        champion_gate_metric = os.getenv("SPOTIFY_CHAMPION_GATE_METRIC", "backtest_top1").strip().lower()
        champion_gate_match_profile_raw = os.getenv("SPOTIFY_CHAMPION_GATE_MATCH_PROFILE", "1").strip().lower()
        champion_gate_match_profile = champion_gate_match_profile_raw in ("1", "true", "yes", "on")
        champion_gate_significance_raw = os.getenv("SPOTIFY_CHAMPION_GATE_SIGNIFICANCE", "0").strip().lower()
        champion_gate_require_significance = champion_gate_significance_raw in ("1", "true", "yes", "on")
        champion_gate_significance_z_raw = os.getenv("SPOTIFY_CHAMPION_GATE_SIGNIFICANCE_Z", "1.96").strip()
        try:
            champion_gate_significance_z = max(0.0, float(champion_gate_significance_z_raw))
        except Exception:
            champion_gate_significance_z = 1.96
        current_risk_metrics = _load_current_risk_metrics(run_dir, result_rows)
        gate_max_selective_risk_raw = os.getenv("SPOTIFY_CHAMPION_GATE_MAX_SELECTIVE_RISK", "").strip()
        gate_max_abstention_raw = os.getenv("SPOTIFY_CHAMPION_GATE_MAX_ABSTENTION_RATE", "").strip()
        try:
            gate_max_selective_risk = float(gate_max_selective_risk_raw) if gate_max_selective_risk_raw else None
        except Exception:
            gate_max_selective_risk = None
        try:
            gate_max_abstention_rate = float(gate_max_abstention_raw) if gate_max_abstention_raw else None
        except Exception:
            gate_max_abstention_rate = None
        with phase_recorder.phase(
            "champion_gate_and_alias",
            metric_source=champion_gate_metric,
            require_profile_match=champion_gate_match_profile,
            require_significant_lift=champion_gate_require_significance,
        ) as phase:
            champion_gate = evaluate_champion_gate(
                history_csv=history_csv,
                current_run_id=run_id,
                current_results=result_rows,
                regression_threshold=champion_gate_threshold,
                backtest_history_csv=(history_dir / "backtest_history.csv"),
                current_backtest_rows=backtest_rows,
                metric_source=champion_gate_metric,
                current_profile=config.profile,
                require_profile_match=champion_gate_match_profile,
                require_significant_lift=champion_gate_require_significance,
                significance_z=champion_gate_significance_z,
                current_risk_metrics=current_risk_metrics,
                max_selective_risk=gate_max_selective_risk,
                max_abstention_rate=gate_max_abstention_rate,
            )
            _write_json_artifact(run_dir / "champion_gate.json", champion_gate, artifact_paths)
            logger.info(
                "Champion gate: source=%s promoted=%s regression=%.6f threshold=%.6f",
                str(champion_gate.get("metric_source", champion_gate_metric)),
                bool(champion_gate.get("promoted", False)),
                float(champion_gate.get("regression", 0.0)),
                float(champion_gate.get("threshold", champion_gate_threshold)),
            )
            strict_gate_raw = os.getenv("SPOTIFY_CHAMPION_GATE_STRICT", "0").strip().lower()
            strict_gate = strict_gate_raw in ("1", "true", "yes", "on")
            strict_gate_error: str | None = None
            if strict_gate and not bool(champion_gate.get("promoted", False)):
                strict_gate_error = (
                    "Champion gate failed in strict mode: "
                    f"regression={champion_gate.get('regression')} threshold={champion_gate.get('threshold')}"
                )

            champion_model: tuple[str, str] | None = None
            if bool(champion_gate.get("promoted", False)):
                champion_model = best_serveable_model(result_rows, run_dir=run_dir)
                if not champion_model:
                    champion_alias_payload["reason"] = "no_serveable_models_in_promoted_run"
                    logger.info("Skipping champion alias update: promoted run has no serveable models.")
                else:
                    champion_model_name, champion_model_type = champion_model
                    alias_file = write_champion_alias(
                        output_dir=config.output_dir,
                        run_id=run_id,
                        run_dir=run_dir,
                        model_name=champion_model_name,
                        model_type=champion_model_type,
                    )
                    champion_alias_payload = {
                        "updated": True,
                        "alias_file": str(alias_file),
                        "run_id": run_id,
                        "run_dir": str(run_dir),
                        "model_name": champion_model_name,
                        "model_type": champion_model_type,
                        "reason": "promoted",
                    }
                    artifact_paths.append(alias_file)
                    logger.info(
                        "Champion alias updated: %s -> run_id=%s model=%s type=%s",
                        alias_file,
                        run_id,
                        champion_model_name,
                        champion_model_type,
                    )
            else:
                champion_alias_payload["reason"] = "gate_not_promoted"
            phase["promoted"] = bool(champion_gate.get("promoted", False))
            phase["champion_alias_updated"] = bool(champion_alias_payload.get("updated", False))
            phase["strict_gate_enabled"] = bool(strict_gate)

        artifact_cleanup_mode = os.getenv("SPOTIFY_ARTIFACT_CLEANUP", "light")
        artifact_cleanup_min_mb_raw = os.getenv("SPOTIFY_ARTIFACT_CLEANUP_MIN_MB", "100").strip()
        try:
            artifact_cleanup_min_mb = max(0.0, float(artifact_cleanup_min_mb_raw))
        except Exception:
            artifact_cleanup_min_mb = 100.0
        with phase_recorder.phase("artifact_cleanup_and_retention") as phase:
            cleanup_summary = prune_run_artifacts(
                run_dir=run_dir,
                result_rows=result_rows,
                selected_model=champion_model,
                logger=logger,
                cleanup_mode=artifact_cleanup_mode,
                min_size_mb=artifact_cleanup_min_mb,
            )
            _write_json_artifact(run_dir / "artifact_cleanup.json", cleanup_summary, artifact_paths)

            prune_old_prediction_bundles_raw = os.getenv("SPOTIFY_PRUNE_OLD_PREDICTION_BUNDLES", "1").strip().lower()
            prune_old_prediction_bundles = prune_old_prediction_bundles_raw in ("1", "true", "yes", "on")
            prune_old_run_dbs_raw = os.getenv("SPOTIFY_PRUNE_OLD_RUN_DATABASES", "1").strip().lower()
            prune_old_run_dbs = prune_old_run_dbs_raw in ("1", "true", "yes", "on")
            keep_full_runs_raw = os.getenv("SPOTIFY_KEEP_FULL_RUNS", "2").strip()
            try:
                keep_full_runs = max(0, int(keep_full_runs_raw))
            except Exception:
                keep_full_runs = 2
            retention_summary = prune_old_auxiliary_artifacts(
                output_dir=config.output_dir,
                current_run_dir=run_dir,
                logger=logger,
                keep_last_full_runs=keep_full_runs,
                prune_prediction_bundles=prune_old_prediction_bundles,
                prune_run_databases=prune_old_run_dbs,
            )
            _write_json_artifact(run_dir / "artifact_retention.json", retention_summary, artifact_paths)
            mlflow_artifact_cleanup_summary = prune_mlflow_artifacts(
                output_dir=config.output_dir,
                logger=logger,
            )
            _write_json_artifact(
                run_dir / "mlflow_artifact_cleanup.json",
                mlflow_artifact_cleanup_summary,
                artifact_paths,
            )
            phase["cleanup_mode"] = artifact_cleanup_mode
            phase["cleanup_deleted_files"] = int(cleanup_summary.get("deleted_file_count", 0) or 0)
            phase["retention_deleted_prediction_bundles"] = int(
                retention_summary.get("deleted_prediction_bundle_count", 0) or 0
            )
            phase["mlflow_deleted_files"] = int(mlflow_artifact_cleanup_summary.get("deleted_file_count", 0) or 0)

        with phase_recorder.phase("research_artifacts") as phase:
            artifact_paths.extend(
                write_benchmark_protocol(
                    output_dir=run_dir,
                    run_id=run_id,
                    profile=config.profile,
                    data=prepared,
                    cache_info=cache_info_payload,
                    config=config,
                )
            )
            artifact_paths.append(
                write_experiment_registry(
                    output_dir=run_dir,
                    run_id=run_id,
                    profile=config.profile,
                    results=result_rows,
                    backtest_rows=backtest_rows,
                    config=config,
                )
            )
            artifact_paths.extend(write_ablation_summary(output_dir=run_dir / "analysis", results=result_rows))
            artifact_paths.extend(
                write_significance_summary(
                    output_dir=run_dir / "analysis",
                    results=result_rows,
                    backtest_rows=backtest_rows,
                )
            )
            phase["result_count"] = int(len(result_rows))
            phase["backtest_row_count"] = int(len(backtest_rows))

        with phase_recorder.phase("history_rollups") as phase:
            if optuna_rows:
                optuna_history_csv = append_optuna_history(
                    history_csv=history_dir / "optuna_history.csv",
                    run_id=run_id,
                    profile=config.profile,
                    run_name=config.run_name,
                    results=optuna_rows,
                )
                optuna_history_plot = plot_optuna_best_runs(optuna_history_csv, history_dir)
                artifact_paths.append(optuna_history_csv)
                if optuna_history_plot is not None:
                    artifact_paths.append(optuna_history_plot)

            if backtest_rows:
                backtest_history_csv = append_backtest_history(
                    history_csv=history_dir / "backtest_history.csv",
                    run_id=run_id,
                    profile=config.profile,
                    run_name=config.run_name,
                    rows=backtest_rows,
                )
                backtest_history_plot = plot_backtest_history(backtest_history_csv, history_dir)
                artifact_paths.append(backtest_history_csv)
                if backtest_history_plot is not None:
                    artifact_paths.append(backtest_history_plot)
            phase["optuna_row_count"] = int(len(optuna_rows))
            phase["backtest_row_count"] = int(len(backtest_rows))

        manifest = {
            "run_id": run_id,
            "run_name": config.run_name,
            "profile": config.profile,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "data_records": len(prepared.df),
            "num_artists": prepared.num_artists,
            "num_context_features": prepared.num_ctx,
            "deep_models": list(config.model_names),
            "classical_models": list(config.classical_model_names) if run_classical_models else [],
            "enable_retrieval_stack": config.enable_retrieval_stack,
            "enable_self_supervised_pretraining": config.enable_self_supervised_pretraining,
            "enable_friction_analysis": config.enable_friction_analysis,
            "enable_moonshot_lab": config.enable_moonshot_lab,
            "retrieval_candidate_k": config.retrieval_candidate_k,
            "enable_mlflow": config.enable_mlflow,
            "enable_optuna": config.enable_optuna,
            "optuna_models": list(config.optuna_model_names),
            "optuna_trials": config.optuna_trials,
            "enable_temporal_backtest": config.enable_temporal_backtest,
            "temporal_backtest_models": list(config.temporal_backtest_model_names),
            "temporal_backtest_folds": config.temporal_backtest_folds,
            "temporal_backtest_adaptation_mode": os.getenv("SPOTIFY_BACKTEST_ADAPTATION_MODE", "cold"),
            "backtest_rows": len(backtest_rows),
            "optuna_rows": len(optuna_rows),
            "cache": cache_info_payload,
            "champion_gate": champion_gate,
            "champion_alias": champion_alias_payload,
            "artifact_cleanup": cleanup_summary,
            "artifact_retention": retention_summary,
            "mlflow_artifact_cleanup": mlflow_artifact_cleanup_summary,
        }
        manifest_payload = manifest
        _write_json_artifact(manifest_path, manifest, artifact_paths)
        _write_json_artifact(run_dir / "run_results.json", result_rows, artifact_paths)

        refresh_analytics_raw = os.getenv("SPOTIFY_REFRESH_ANALYTICS_DB", "1").strip().lower()
        refresh_analytics = refresh_analytics_raw not in ("0", "false", "no", "off")
        if refresh_analytics:
            with phase_recorder.phase("analytics_refresh") as phase:
                try:
                    analytics_db_path = refresh_analytics_database(
                        data_dir=config.data_dir,
                        output_dir=config.output_dir,
                        include_video=config.include_video,
                        logger=logger,
                        raw_df=raw_df,
                    )
                except Exception as exc:
                    logger.warning("Analytics database refresh failed but the run will continue: %s", exc)
                    phase["status"] = "warning"
                    phase["warning"] = str(exc)
                else:
                    if analytics_db_path is not None and analytics_db_path.exists():
                        artifact_paths.append(analytics_db_path)
                    phase["db_path"] = analytics_db_path
                    phase["refreshed"] = bool(analytics_db_path is not None and analytics_db_path.exists())
        else:
            phase_recorder.skip("analytics_refresh", reason="analytics_refresh_disabled")

        with phase_recorder.phase("run_report") as phase:
            report_path = write_run_report(
                run_dir=run_dir,
                history_dir=history_dir,
                manifest=manifest,
                results=result_rows,
                champion_gate=champion_gate,
                history_csv=history_csv,
            )
            artifact_paths.append(report_path)
            phase["report_path"] = report_path

        with phase_recorder.phase("control_room_report", top_n=5) as phase:
            try:
                control_room_json, control_room_md = write_control_room_report(config.output_dir, top_n=5)
            except Exception as exc:
                logger.warning("Control room report generation failed but the run will continue: %s", exc)
                phase["status"] = "warning"
                phase["warning"] = str(exc)
            else:
                artifact_paths.extend([control_room_json, control_room_md])
                phase["json_path"] = control_room_json
                phase["markdown_path"] = control_room_md

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
    "_analysis_prefix_for_model_type",
    "_append_existing_artifact_path",
    "_build_run_id",
    "_load_current_risk_metrics",
    "_release_deep_runtime_resources",
    "_track_file",
    "_write_json_artifact",
    "run_pipeline",
]
