from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
import os
import random
import re
import sys

import numpy as np

from .champion_alias import best_serveable_model, write_champion_alias
from .config import PipelineConfig, configure_logging
from .runtime import configure_process_env, load_tensorflow_runtime, select_distribution_strategy
from .tracking import MlflowTracker


def _slugify(raw: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_-]+", "-", raw.strip())
    value = re.sub(r"-{2,}", "-", value).strip("-")
    return value or "run"


def _build_run_id(config: PipelineConfig) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = _slugify(config.run_name) if config.run_name else config.profile
    return f"{stamp}_{suffix}"


def _track_file(tracker: MlflowTracker | None, path: Path) -> None:
    if tracker is not None and path.exists():
        tracker.log_artifact(path)


def run_pipeline(config: PipelineConfig) -> None:
    run_id = _build_run_id(config)
    run_dir = config.output_dir / "runs" / run_id
    history_dir = config.output_dir / "history"
    run_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)

    logger = configure_logging(run_dir / "train.log")
    logger.info("Starting Spotify training pipeline")
    logger.info("Run ID: %s", run_id)
    if config.run_name:
        logger.info("Run Name: %s", config.run_name)
    logger.info("Data directory: %s", config.data_dir)
    logger.info("Output root: %s", config.output_dir)
    logger.info("Run output directory: %s", run_dir)

    # Use shared matplotlib/font caches by default so font cache is built once
    # and reused across runs (large startup-time win on macOS).
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
    artifact_paths: list[Path] = [run_dir / "train.log"]
    tracker: MlflowTracker | None = None
    final_tracker_status = "FAILED"

    try:
        run_deep_models = (not config.classical_only) and bool(config.model_names)
        run_classical_models = bool(config.enable_classical_models)
        if config.classical_only:
            run_classical_models = True

        tf = None
        strategy = None
        if run_deep_models:
            # Initialize TensorFlow before heavy numpy/pandas/sklearn preprocessing.
            # On some macOS TensorFlow builds, late initialization can deadlock in the
            # internal prefetch pipeline at the first training batch.
            configure_process_env()
            tf = load_tensorflow_runtime(logger)
            tf.random.set_seed(config.random_seed)
            strategy = select_distribution_strategy(tf, logger=logger)
            logger.info("Number of devices: %s", getattr(strategy, "num_replicas_in_sync", 1))

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
                "enable_optuna": config.enable_optuna,
                "optuna_trials": config.optuna_trials,
                "optuna_models": config.optuna_model_names,
                "enable_temporal_backtest": config.enable_temporal_backtest,
                "temporal_backtest_folds": config.temporal_backtest_folds,
                "temporal_backtest_models": config.temporal_backtest_model_names,
            }
        )

        # Delay sklearn/pandas-heavy imports until after TensorFlow initialization
        # for deep runs. Importing sklearn before TensorFlow can trigger a deadlock
        # on some macOS TensorFlow builds at the first training batch.
        from .backtesting import run_temporal_backtest
        from .analytics_db import refresh_analytics_database
        from .benchmarks import build_classical_feature_bundle, run_classical_benchmarks
        from .data import (
            PreparedDataCacheInfo,
            SKEW_CONTEXT_FEATURES,
            load_streaming_history,
            load_or_prepare_training_data,
        )
        from .data_quality import run_data_quality_gate
        from .ensemble import build_probability_ensemble
        from .governance import evaluate_champion_gate
        from .explainability import run_shap_analysis
        from .evaluation import run_extended_evaluation
        from .modeling import build_model_builders
        from .monitoring import ResourceMonitor
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

        raw_df = load_streaming_history(
            config.data_dir,
            include_video=config.include_video,
            logger=logger,
        )
        data_quality_report_path = run_dir / "data_quality_report.json"
        run_data_quality_gate(raw_df, report_path=data_quality_report_path, logger=logger)
        artifact_paths.append(data_quality_report_path)

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
        with metadata_path.open("w", encoding="utf-8") as out:
            json.dump(
                {
                    "sequence_length": config.sequence_length,
                    "context_features": list(prepared.context_features),
                    "skew_context_features": list(SKEW_CONTEXT_FEATURES),
                    "artist_labels": artist_labels,
                },
                out,
                indent=2,
            )
        artifact_paths.append(metadata_path)

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
                bundle_raw = str(artifacts.prediction_bundle_paths.get(name, "")).strip()
                bundle_path = Path(bundle_raw) if bundle_raw else None
                if bundle_path is not None and bundle_path.exists():
                    artifact_paths.append(bundle_path)
        else:
            logger.info("Skipping deep models for this run.")

        classical_feature_bundle = build_classical_feature_bundle(prepared) if run_classical_models else None

        if run_classical_models:
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
                bundle_raw = str(row.prediction_bundle_path).strip()
                bundle_path = Path(bundle_raw) if bundle_raw else None
                if bundle_path is not None and bundle_path.exists():
                    artifact_paths.append(bundle_path)
                estimator_raw = str(row.estimator_artifact_path).strip()
                estimator_path = Path(estimator_raw) if estimator_raw else None
                if estimator_path is not None and estimator_path.exists():
                    artifact_paths.append(estimator_path)
        else:
            logger.info("Skipping classical model benchmarks for this run.")

        if run_classical_models and config.enable_optuna:
            optuna_dir = run_dir / "optuna"
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
                bundle_raw = str(row.prediction_bundle_path).strip()
                bundle_path = Path(bundle_raw) if bundle_raw else None
                if bundle_path is not None and bundle_path.exists():
                    artifact_paths.append(bundle_path)
                estimator_raw = str(row.estimator_artifact_path).strip()
                estimator_path = Path(estimator_raw) if estimator_raw else None
                if estimator_path is not None and estimator_path.exists():
                    artifact_paths.append(estimator_path)
            if optuna_dir.exists():
                artifact_paths.extend(sorted(p for p in optuna_dir.glob("*") if p.is_file()))
        elif config.enable_optuna:
            logger.info("Skipping Optuna tuning because classical models are disabled.")

        if run_classical_models and config.enable_temporal_backtest:
            backtest_dir = run_dir / "backtest"
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
            )
            for row in backtest_results:
                backtest_rows.append(
                    {
                        "model_name": row.model_name,
                        "model_family": row.model_family,
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
        elif config.enable_temporal_backtest:
            logger.info("Skipping temporal backtesting because classical models are disabled.")

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

        extended_eval_artifacts = run_extended_evaluation(
            data=prepared,
            results=result_rows,
            sequence_length=config.sequence_length,
            run_dir=run_dir,
            random_seed=config.random_seed,
            max_train_samples=config.classical_max_train_samples,
            logger=logger,
            feature_bundle=classical_feature_bundle,
        )
        artifact_paths.extend(extended_eval_artifacts)

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
        )
        champion_gate_path = run_dir / "champion_gate.json"
        champion_gate_path.write_text(json.dumps(champion_gate, indent=2), encoding="utf-8")
        artifact_paths.append(champion_gate_path)
        logger.info(
            "Champion gate: source=%s promoted=%s regression=%.6f threshold=%.6f",
            str(champion_gate.get("metric_source", champion_gate_metric)),
            bool(champion_gate.get("promoted", False)),
            float(champion_gate.get("regression", 0.0)),
            float(champion_gate.get("threshold", champion_gate_threshold)),
        )
        strict_gate_raw = os.getenv("SPOTIFY_CHAMPION_GATE_STRICT", "0").strip().lower()
        strict_gate = strict_gate_raw in ("1", "true", "yes", "on")
        if strict_gate and not bool(champion_gate.get("promoted", False)):
            raise RuntimeError(
                "Champion gate failed in strict mode: "
                f"regression={champion_gate.get('regression')} threshold={champion_gate.get('threshold')}"
            )

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
            "enable_mlflow": config.enable_mlflow,
            "enable_optuna": config.enable_optuna,
            "optuna_models": list(config.optuna_model_names),
            "optuna_trials": config.optuna_trials,
            "enable_temporal_backtest": config.enable_temporal_backtest,
            "temporal_backtest_models": list(config.temporal_backtest_model_names),
            "temporal_backtest_folds": config.temporal_backtest_folds,
            "backtest_rows": len(backtest_rows),
            "optuna_rows": len(optuna_rows),
            "cache": cache_info_payload,
            "champion_gate": champion_gate,
            "champion_alias": champion_alias_payload,
        }
        manifest_path = run_dir / "run_manifest.json"
        with manifest_path.open("w", encoding="utf-8") as out:
            json.dump(manifest, out, indent=2)
        artifact_paths.append(manifest_path)

        run_results_path = run_dir / "run_results.json"
        with run_results_path.open("w", encoding="utf-8") as out:
            json.dump(result_rows, out, indent=2)
        artifact_paths.append(run_results_path)

        refresh_analytics_raw = os.getenv("SPOTIFY_REFRESH_ANALYTICS_DB", "1").strip().lower()
        refresh_analytics = refresh_analytics_raw not in ("0", "false", "no", "off")
        if refresh_analytics:
            analytics_db_path = refresh_analytics_database(
                data_dir=config.data_dir,
                output_dir=config.output_dir,
                include_video=config.include_video,
                logger=logger,
                raw_df=raw_df,
            )
            if analytics_db_path is not None and analytics_db_path.exists():
                artifact_paths.append(analytics_db_path)

        report_path = write_run_report(
            run_dir=run_dir,
            history_dir=history_dir,
            manifest=manifest,
            results=result_rows,
            champion_gate=champion_gate,
            history_csv=history_csv,
        )
        artifact_paths.append(report_path)

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

        if tracker is not None:
            tracker.log_result_rows(result_rows)
            tracker.log_backtest_rows(backtest_rows)
        for path in artifact_paths:
            _track_file(tracker, path)

        logger.info("Pipeline completed successfully")
        final_tracker_status = "FINISHED"
    except KeyboardInterrupt:
        final_tracker_status = "KILLED"
        raise
    except Exception:
        final_tracker_status = "FAILED"
        raise
    finally:
        if tracker is not None:
            tracker.end(status=final_tracker_status)
