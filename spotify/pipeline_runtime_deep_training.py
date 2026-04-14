from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np

from .pipeline_helpers import _append_existing_artifact_path
from .pipeline_runtime_experiment_types import PipelineExperimentContext, PipelineExperimentDeps


def run_deep_model_training(
    *,
    context: PipelineExperimentContext,
    deps: PipelineExperimentDeps,
    deep_cache_plan: Any,
    deep_cache_stats: dict[str, object],
    strategy: Any,
) -> Any:
    if not context.run_deep_models:
        context.phase_recorder.skip("deep_model_training", reason="deep_models_disabled")
        context.logger.info("Skipping deep models for this run.")
        return None

    with context.phase_recorder.phase(
        "deep_model_training",
        model_names=list(context.config.model_names),
        batch_size=context.config.batch_size,
        epochs=context.config.epochs,
    ) as phase:
        deep_model_names_to_build = tuple(deep_cache_plan.miss_model_names) if deep_cache_plan is not None else context.config.model_names
        model_builders = None
        if deep_model_names_to_build:
            model_builders = deps.build_model_builders(
                sequence_length=context.config.sequence_length,
                num_artists=context.prepared.num_artists,
                num_ctx=context.prepared.num_ctx,
                selected_names=deep_model_names_to_build,
            )

        disable_monitor = os.getenv("SPOTIFY_DISABLE_MONITOR", "auto").strip().lower()
        monitor_enabled = bool(deep_model_names_to_build)
        if disable_monitor in ("1", "true", "yes", "on"):
            monitor_enabled = False
        elif disable_monitor == "auto" and sys.platform == "darwin":
            monitor_enabled = False

        monitor = deps.ResourceMonitor(context.logger) if monitor_enabled else None
        if monitor is not None:
            monitor.start()
        try:
            artifacts = deps.train_and_evaluate_models(
                data=context.prepared,
                model_builders=model_builders,
                batch_size=context.config.batch_size,
                epochs=context.config.epochs,
                output_dir=context.run_dir,
                strategy=strategy,
                logger=context.logger,
                random_seed=context.config.random_seed,
                cache_root=context.config.output_dir / "cache" / "deep_training",
                cache_fingerprint=context.cache_fingerprint,
                cache_stats_out=deep_cache_stats,
                cache_plan=deep_cache_plan,
            )
        finally:
            if monitor is not None:
                monitor.stop()

        cpu_usage = monitor.cpu_usage if monitor is not None else []
        gpu_usage = monitor.gpu_usage if monitor is not None else []
        sqlite_path = context.run_dir / "spotify_training.db"
        restored_reporting = deps.restore_deep_reporting_artifacts(
            histories=artifacts.histories,
            cpu_usage=cpu_usage,
            gpu_usage=gpu_usage,
            output_dir=context.run_dir,
            db_path=sqlite_path,
            cache_root=context.config.output_dir / "cache" / "deep_reporting",
            cache_fingerprint=context.cache_fingerprint,
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
            model_comparison_path = deps.plot_model_comparison(artifacts.histories, context.run_dir)
            learning_paths = deps.plot_learning_curves(artifacts.histories, context.run_dir)
            histories_path = deps.save_histories_json(artifacts.histories, context.run_dir)
            utilization_path = deps.save_utilization_plot(cpu_usage, gpu_usage, context.run_dir)
            sqlite_path = deps.persist_to_sqlite(
                df=context.prepared.df,
                histories=artifacts.histories,
                cpu_usage=cpu_usage,
                gpu_usage=gpu_usage,
                db_path=sqlite_path,
            )
            deps.save_deep_reporting_artifacts(
                histories=artifacts.histories,
                cpu_usage=cpu_usage,
                gpu_usage=gpu_usage,
                output_dir=context.run_dir,
                db_path=sqlite_path,
                cache_root=context.config.output_dir / "cache" / "deep_reporting",
                cache_fingerprint=context.cache_fingerprint,
            )
        context.artifact_paths.extend([model_comparison_path, histories_path, utilization_path, *learning_paths])

        if context.config.enable_shap:
            deps.run_shap_analysis(
                artifacts.histories,
                context.run_dir,
                context.prepared,
                context.logger,
                cache_root=context.config.output_dir / "cache" / "shap",
                cache_fingerprint=context.cache_fingerprint,
            )
        else:
            context.logger.info("Skipping SHAP analysis because --no-shap was set.")

        context.artifact_paths.append(sqlite_path)

        context.logger.info("Final Validation Artist Accuracy (Top-1 / Top-5):")
        for name, history in artifacts.histories.items():
            val_key = deps.VAL_KEY if deps.VAL_KEY in history.history else "val_sparse_categorical_accuracy"
            top1 = history.history[val_key][-1]
            top5_key = "val_artist_output_top_5" if "val_artist_output_top_5" in history.history else "val_top_5"
            top5 = history.history.get(top5_key, [np.nan])[-1]
            context.logger.info("%s: Top-1=%.4f | Top-5=%.4f", name, top1, top5)

            context.result_rows.append(
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
            _append_existing_artifact_path(context.artifact_paths, artifacts.prediction_bundle_paths.get(name, ""))
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
        phase["warm_start_model_names"] = list(deep_cache_stats.get("warm_start_model_names", []))
        phase["screening_selected_model_names"] = list(deep_cache_stats.get("screening_selected_model_names", []))
        phase["screening_screened_out_model_names"] = list(
            deep_cache_stats.get("screening_screened_out_model_names", [])
        )
        phase["reporting_cache_reused"] = bool(restored_reporting is not None)
        return model_builders


__all__ = ["run_deep_model_training"]
