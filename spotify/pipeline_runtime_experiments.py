from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import sys
from typing import Any

import numpy as np

from .config import PipelineConfig
from .pipeline_helpers import _append_existing_artifact_path, _release_deep_runtime_resources
from .pipeline_runtime_shortlists import (
    _resolve_shortlist_top_n,
    _shortlist_classical_model_names,
    _tuned_backtest_specs,
)
from .runtime import configure_process_env, load_tensorflow_runtime, select_distribution_strategy


@dataclass
class PipelineExperimentContext:
    artifact_paths: list[Path]
    backtest_rows: list[dict[str, object]]
    cache_fingerprint: str
    config: PipelineConfig
    logger: Any
    optuna_rows: list[dict[str, object]]
    phase_recorder: Any
    prepared: Any
    result_rows: list[dict[str, object]]
    run_classical_models: bool
    run_deep_backtest: bool
    run_deep_models: bool
    run_dir: Path


@dataclass
class PipelineExperimentDeps:
    ResourceMonitor: Any
    VAL_KEY: str
    build_classical_feature_bundle: Any
    build_model_builders: Any
    persist_to_sqlite: Any
    plot_learning_curves: Any
    plot_model_comparison: Any
    resolve_cached_deep_training_artifacts: Any
    restore_deep_reporting_artifacts: Any
    run_classical_benchmarks: Any
    run_optuna_tuning: Any
    run_shap_analysis: Any
    run_temporal_backtest: Any
    save_deep_reporting_artifacts: Any
    save_histories_json: Any
    save_utilization_plot: Any
    train_and_evaluate_models: Any
    train_retrieval_stack: Any


@dataclass
class PipelineExperimentOutputs:
    classical_feature_bundle: Any


def _prepare_deep_training_cache(*, context: PipelineExperimentContext, deps: PipelineExperimentDeps) -> tuple[Any, dict[str, object]]:
    if not context.run_deep_models:
        return None, {}
    deep_cache_plan = deps.resolve_cached_deep_training_artifacts(
        data=context.prepared,
        selected_model_names=context.config.model_names,
        batch_size=context.config.batch_size,
        epochs=context.config.epochs,
        output_dir=context.run_dir,
        logger=context.logger,
        random_seed=context.config.random_seed,
        cache_root=context.config.output_dir / "cache" / "deep_training",
        cache_fingerprint=context.cache_fingerprint,
    )
    deep_cache_stats = {
        "enabled": bool(deep_cache_plan.enabled),
        "fingerprint": str(deep_cache_plan.fingerprint),
        "hit_model_names": list(deep_cache_plan.hit_model_names),
        "miss_model_names": list(deep_cache_plan.miss_model_names),
    }
    return deep_cache_plan, deep_cache_stats


def _init_tensorflow_runtime(
    *,
    context: PipelineExperimentContext,
    deep_cache_stats: dict[str, object],
    needs_tf_for_deep_training: bool,
) -> tuple[Any, Any]:
    tf = None
    strategy = None
    if needs_tf_for_deep_training or context.run_deep_backtest:
        with context.phase_recorder.phase(
            "tensorflow_runtime_init",
            run_deep_models=context.run_deep_models,
            run_deep_backtest=context.run_deep_backtest,
            deep_cache_hit_models=list(deep_cache_stats.get("hit_model_names", [])),
            deep_cache_miss_models=list(deep_cache_stats.get("miss_model_names", [])),
        ) as phase:
            configure_process_env()
            tf = load_tensorflow_runtime(context.logger)
            tf.random.set_seed(context.config.random_seed)
            strategy = select_distribution_strategy(tf, logger=context.logger)
            device_count = int(getattr(strategy, "num_replicas_in_sync", 1))
            phase["device_count"] = device_count
            phase["initialized_for_deep_training"] = bool(needs_tf_for_deep_training)
            phase["initialized_for_deep_backtest"] = bool(context.run_deep_backtest)
            phase["cache_hit_count"] = int(len(deep_cache_stats.get("hit_model_names", [])))
            phase["cache_miss_count"] = int(len(deep_cache_stats.get("miss_model_names", [])))
            context.logger.info("Number of devices: %s", device_count)
        return tf, strategy

    context.phase_recorder.skip(
        "tensorflow_runtime_init",
        reason=(
            "deep_training_fully_cached_and_deep_backtest_disabled"
            if context.run_deep_models
            else "deep_models_and_deep_backtest_disabled"
        ),
    )
    return None, None


def _run_deep_model_training(
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
        if deep_model_names_to_build:
            model_builders = deps.build_model_builders(
                sequence_length=context.config.sequence_length,
                num_artists=context.prepared.num_artists,
                num_ctx=context.prepared.num_ctx,
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
        phase["reporting_cache_reused"] = bool(restored_reporting is not None)
        return model_builders


def _run_classical_benchmarks(
    *,
    context: PipelineExperimentContext,
    deps: PipelineExperimentDeps,
    classical_feature_bundle: Any,
    classical_cache_stats: dict[str, object],
) -> list[Any]:
    if not context.run_classical_models:
        context.phase_recorder.skip("classical_benchmarks", reason="classical_models_disabled")
        context.logger.info("Skipping classical model benchmarks for this run.")
        return []

    with context.phase_recorder.phase(
        "classical_benchmarks",
        model_names=list(context.config.classical_model_names),
        max_train_samples=context.config.classical_max_train_samples,
        max_eval_samples=context.config.classical_max_eval_samples,
    ) as phase:
        classical_results = deps.run_classical_benchmarks(
            data=context.prepared,
            output_dir=context.run_dir,
            selected_models=context.config.classical_model_names,
            random_seed=context.config.random_seed,
            max_train_samples=context.config.classical_max_train_samples,
            max_eval_samples=context.config.classical_max_eval_samples,
            logger=context.logger,
            feature_bundle=classical_feature_bundle,
            cache_root=context.config.output_dir / "cache" / "classical_benchmarks",
            cache_fingerprint=context.cache_fingerprint,
            cache_stats_out=classical_cache_stats,
        )
        context.artifact_paths.append(context.run_dir / "classical_results.json")
        for row in classical_results:
            context.result_rows.append(
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
            _append_existing_artifact_path(context.artifact_paths, row.prediction_bundle_path)
            _append_existing_artifact_path(context.artifact_paths, row.estimator_artifact_path)
        phase["model_count"] = int(len(classical_results))
        phase["cache_enabled"] = bool(classical_cache_stats.get("enabled", False))
        phase["cache_fingerprint"] = str(classical_cache_stats.get("fingerprint", ""))
        phase["cache_hit_models"] = list(classical_cache_stats.get("hit_model_names", []))
        phase["cache_hit_count"] = int(len(classical_cache_stats.get("hit_model_names", [])))
        phase["cache_miss_models"] = list(classical_cache_stats.get("miss_model_names", []))
        phase["cache_miss_count"] = int(len(classical_cache_stats.get("miss_model_names", [])))
        return classical_results


def _run_optuna_tuning(
    *,
    context: PipelineExperimentContext,
    deps: PipelineExperimentDeps,
    classical_feature_bundle: Any,
    classical_results: list[Any],
    optuna_cache_stats: dict[str, object],
) -> tuple[tuple[str, ...], dict[str, dict[str, object]]]:
    selected_optuna_model_names = context.config.optuna_model_names
    selected_backtest_model_names = context.config.temporal_backtest_model_names
    if classical_results:
        selected_optuna_model_names = _shortlist_classical_model_names(
            context.config.optuna_model_names,
            classical_results,
            top_n=_resolve_shortlist_top_n("SPOTIFY_OPTUNA_SHORTLIST_TOP_N"),
            logger=context.logger,
            stage_label="Optuna",
        )
        selected_backtest_model_names = _shortlist_classical_model_names(
            context.config.temporal_backtest_model_names,
            classical_results,
            top_n=_resolve_shortlist_top_n("SPOTIFY_BACKTEST_SHORTLIST_TOP_N"),
            logger=context.logger,
            stage_label="Temporal backtest",
        )
    tuned_backtest_specs: dict[str, dict[str, object]] = {}

    if context.run_classical_models and context.config.enable_optuna:
        optuna_dir = context.run_dir / "optuna"
        with context.phase_recorder.phase(
            "optuna_tuning",
            model_names=list(selected_optuna_model_names),
            candidate_model_names=list(context.config.optuna_model_names),
            trials=context.config.optuna_trials,
            timeout_seconds=context.config.optuna_timeout_seconds,
        ) as phase:
            tuned_results = deps.run_optuna_tuning(
                data=context.prepared,
                output_dir=optuna_dir,
                selected_models=selected_optuna_model_names,
                random_seed=context.config.random_seed,
                trials=context.config.optuna_trials,
                timeout_seconds=context.config.optuna_timeout_seconds,
                max_train_samples=context.config.classical_max_train_samples,
                max_eval_samples=context.config.classical_max_eval_samples,
                logger=context.logger,
                feature_bundle=classical_feature_bundle,
                cache_root=context.config.output_dir / "cache" / "optuna",
                cache_fingerprint=context.cache_fingerprint,
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
                context.result_rows.append(payload)
                context.optuna_rows.append(payload)
                _append_existing_artifact_path(context.artifact_paths, row.prediction_bundle_path)
                _append_existing_artifact_path(context.artifact_paths, row.estimator_artifact_path)
            if optuna_dir.exists():
                context.artifact_paths.extend(sorted(p for p in optuna_dir.glob("*") if p.is_file()))
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
                logger=context.logger,
            )
        return selected_backtest_model_names, tuned_backtest_specs

    if context.config.enable_optuna:
        context.phase_recorder.skip("optuna_tuning", reason="classical_models_disabled")
        context.logger.info("Skipping Optuna tuning because classical models are disabled.")
    else:
        context.phase_recorder.skip("optuna_tuning", reason="optuna_disabled")
    return selected_backtest_model_names, tuned_backtest_specs


def _run_retrieval_stack(*, context: PipelineExperimentContext, deps: PipelineExperimentDeps) -> None:
    if context.config.enable_retrieval_stack:
        with context.phase_recorder.phase(
            "retrieval_stack",
            candidate_k=context.config.retrieval_candidate_k,
            enable_self_supervised_pretraining=context.config.enable_self_supervised_pretraining,
        ) as phase:
            retrieval_result = deps.train_retrieval_stack(
                data=context.prepared,
                output_dir=context.run_dir,
                random_seed=context.config.random_seed,
                candidate_k=context.config.retrieval_candidate_k,
                enable_self_supervised_pretraining=context.config.enable_self_supervised_pretraining,
                logger=context.logger,
            )
            for row in retrieval_result.rows:
                context.result_rows.append(dict(row))
            context.artifact_paths.extend(retrieval_result.artifact_paths)
            phase["result_count"] = int(len(retrieval_result.rows))
        return

    context.phase_recorder.skip("retrieval_stack", reason="retrieval_disabled")
    context.logger.info("Skipping retrieval stack for this run.")


def _run_temporal_backtest(
    *,
    context: PipelineExperimentContext,
    deps: PipelineExperimentDeps,
    classical_feature_bundle: Any,
    model_builders: Any,
    selected_backtest_model_names: tuple[str, ...],
    strategy: Any,
    tuned_backtest_specs: dict[str, dict[str, object]],
) -> None:
    if not context.config.enable_temporal_backtest:
        context.phase_recorder.skip("temporal_backtest", reason="temporal_backtest_disabled")
        return

    backtest_cache_stats: dict[str, object] = {}
    backtest_dir = context.run_dir / "backtest"
    with context.phase_recorder.phase(
        "temporal_backtest",
        folds=context.config.temporal_backtest_folds,
        model_names=list(selected_backtest_model_names),
        candidate_model_names=list(context.config.temporal_backtest_model_names),
        adaptation_mode=os.getenv("SPOTIFY_BACKTEST_ADAPTATION_MODE", "cold"),
    ) as phase:
        deep_backtest_builders = model_builders
        backtest_results = deps.run_temporal_backtest(
            data=context.prepared,
            output_dir=backtest_dir,
            selected_models=selected_backtest_model_names,
            random_seed=context.config.random_seed,
            folds=context.config.temporal_backtest_folds,
            max_train_samples=context.config.classical_max_train_samples,
            max_eval_samples=context.config.classical_max_eval_samples,
            logger=context.logger,
            feature_bundle=classical_feature_bundle,
            deep_model_builders=deep_backtest_builders,
            strategy=strategy,
            adaptation_mode=os.getenv("SPOTIFY_BACKTEST_ADAPTATION_MODE", "cold"),
            tuned_model_specs=tuned_backtest_specs,
            cache_root=context.config.output_dir / "cache" / "temporal_backtest",
            cache_fingerprint=context.cache_fingerprint,
            cache_stats_out=backtest_cache_stats,
        )
        for row in backtest_results:
            context.backtest_rows.append(
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
            context.artifact_paths.extend(sorted(p for p in backtest_dir.glob("*") if p.is_file()))
        phase["row_count"] = int(len(context.backtest_rows))
        phase["deep_backtest_builders"] = bool(deep_backtest_builders)
        phase["cache_enabled"] = bool(backtest_cache_stats.get("enabled", False))
        phase["cache_fingerprint"] = str(backtest_cache_stats.get("fingerprint", ""))
        phase["cache_key"] = str(backtest_cache_stats.get("cache_key", ""))
        phase["cache_hit"] = bool(backtest_cache_stats.get("hit", False))


def run_experiment_stages(*, context: PipelineExperimentContext, deps: PipelineExperimentDeps) -> PipelineExperimentOutputs:
    deep_cache_plan, deep_cache_stats = _prepare_deep_training_cache(context=context, deps=deps)
    needs_tf_for_deep_training = bool(
        context.run_deep_models and deep_cache_plan is not None and deep_cache_plan.miss_model_names
    )
    tf, strategy = _init_tensorflow_runtime(
        context=context,
        deep_cache_stats=deep_cache_stats,
        needs_tf_for_deep_training=needs_tf_for_deep_training,
    )
    model_builders = _run_deep_model_training(
        context=context,
        deps=deps,
        deep_cache_plan=deep_cache_plan,
        deep_cache_stats=deep_cache_stats,
        strategy=strategy,
    )

    classical_feature_bundle = deps.build_classical_feature_bundle(context.prepared) if context.run_classical_models else None
    classical_cache_stats: dict[str, object] = {}
    classical_results = _run_classical_benchmarks(
        context=context,
        deps=deps,
        classical_feature_bundle=classical_feature_bundle,
        classical_cache_stats=classical_cache_stats,
    )

    optuna_cache_stats: dict[str, object] = {}
    selected_backtest_model_names, tuned_backtest_specs = _run_optuna_tuning(
        context=context,
        deps=deps,
        classical_feature_bundle=classical_feature_bundle,
        classical_results=classical_results,
        optuna_cache_stats=optuna_cache_stats,
    )
    _run_retrieval_stack(context=context, deps=deps)
    _run_temporal_backtest(
        context=context,
        deps=deps,
        classical_feature_bundle=classical_feature_bundle,
        model_builders=model_builders,
        selected_backtest_model_names=selected_backtest_model_names,
        strategy=strategy,
        tuned_backtest_specs=tuned_backtest_specs,
    )

    if tf is not None:
        with context.phase_recorder.phase("release_deep_runtime_resources") as phase:
            _release_deep_runtime_resources(tf, context.logger)
            phase["tensorflow_loaded"] = True
    else:
        context.phase_recorder.skip("release_deep_runtime_resources", reason="tensorflow_not_initialized")

    return PipelineExperimentOutputs(classical_feature_bundle=classical_feature_bundle)


__all__ = [
    "PipelineExperimentContext",
    "PipelineExperimentDeps",
    "PipelineExperimentOutputs",
    "run_experiment_stages",
]
