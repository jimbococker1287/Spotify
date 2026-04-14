from __future__ import annotations

import os

from .pipeline_runtime_deep_stage import (
    init_tensorflow_runtime,
    release_deep_runtime_resources,
    run_deep_model_training,
)
from .pipeline_runtime_experiment_types import (
    PipelineExperimentContext,
    PipelineExperimentDeps,
    PipelineExperimentOutputs,
)
from .pipeline_runtime_search_stage import (
    run_classical_benchmarks,
    run_optuna_tuning,
    run_retrieval_stack,
    run_temporal_backtest,
)


def run_experiment_stages(*, context: PipelineExperimentContext, deps: PipelineExperimentDeps) -> PipelineExperimentOutputs:
    if context.run_deep_models:
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
    else:
        deep_cache_plan = None
        deep_cache_stats = {}

    classical_feature_bundle = deps.build_classical_feature_bundle(context.prepared) if context.run_classical_models else None
    classical_cache_stats: dict[str, object] = {}
    classical_results = run_classical_benchmarks(
        context=context,
        deps=deps,
        classical_feature_bundle=classical_feature_bundle,
        classical_cache_stats=classical_cache_stats,
    )

    optuna_cache_stats: dict[str, object] = {}
    selected_backtest_model_names, tuned_backtest_specs = run_optuna_tuning(
        context=context,
        deps=deps,
        classical_feature_bundle=classical_feature_bundle,
        classical_results=classical_results,
        optuna_cache_stats=optuna_cache_stats,
    )
    run_retrieval_stack(context=context, deps=deps)
    if context.config.enable_temporal_backtest:
        deep_backtest_cache_inspection = deps.inspect_temporal_backtest_cache(
            data=context.prepared,
            selected_models=selected_backtest_model_names,
            random_seed=context.config.random_seed,
            folds=context.config.temporal_backtest_folds,
            max_train_samples=context.config.classical_max_train_samples,
            max_eval_samples=context.config.classical_max_eval_samples,
            adaptation_mode=os.getenv("SPOTIFY_BACKTEST_ADAPTATION_MODE", "cold"),
            tuned_model_specs=tuned_backtest_specs,
            cache_root=context.config.output_dir / "cache" / "temporal_backtest",
            cache_fingerprint=context.cache_fingerprint,
        )
    else:
        deep_backtest_cache_inspection = None
    needs_tf_for_deep_training = bool(
        context.run_deep_models and deep_cache_plan is not None and deep_cache_plan.miss_model_names
    )
    needs_tf_for_deep_backtest = bool(
        context.config.temporal_backtest_folds > 0
        and getattr(deep_backtest_cache_inspection, "deep_models", ())
        and not getattr(deep_backtest_cache_inspection, "hit", False)
    )
    tf, strategy = init_tensorflow_runtime(
        context=context,
        deep_cache_stats=deep_cache_stats,
        needs_tf_for_deep_training=needs_tf_for_deep_training,
        needs_tf_for_deep_backtest=needs_tf_for_deep_backtest,
        deep_backtest_cache_inspection=deep_backtest_cache_inspection,
    )
    run_deep_model_training(
        context=context,
        deps=deps,
        deep_cache_plan=deep_cache_plan,
        deep_cache_stats=deep_cache_stats,
        strategy=strategy,
    )
    run_temporal_backtest(
        context=context,
        deps=deps,
        classical_feature_bundle=classical_feature_bundle,
        selected_backtest_model_names=selected_backtest_model_names,
        strategy=strategy,
        tuned_backtest_specs=tuned_backtest_specs,
    )
    release_deep_runtime_resources(context=context, tf=tf)

    return PipelineExperimentOutputs(classical_feature_bundle=classical_feature_bundle)


__all__ = [
    "PipelineExperimentContext",
    "PipelineExperimentDeps",
    "PipelineExperimentOutputs",
    "run_experiment_stages",
]
