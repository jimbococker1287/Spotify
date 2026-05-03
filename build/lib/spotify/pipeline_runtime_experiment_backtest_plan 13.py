from __future__ import annotations

import os

from .pipeline_runtime_experiment_plan_types import PipelineExperimentPlan
from .pipeline_runtime_experiment_types import PipelineExperimentContext, PipelineExperimentDeps


def update_experiment_plan_for_backtest(
    *,
    context: PipelineExperimentContext,
    deps: PipelineExperimentDeps,
    plan: PipelineExperimentPlan,
    selected_backtest_model_names: tuple[str, ...],
    tuned_backtest_specs: dict[str, dict[str, object]],
) -> PipelineExperimentPlan:
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

    return PipelineExperimentPlan(
        classical_feature_bundle=plan.classical_feature_bundle,
        deep_backtest_cache_inspection=deep_backtest_cache_inspection,
        deep_cache_plan=plan.deep_cache_plan,
        deep_cache_stats=plan.deep_cache_stats,
        needs_tf_for_deep_backtest=bool(
            context.config.temporal_backtest_folds > 0
            and getattr(deep_backtest_cache_inspection, "deep_models", ())
            and not getattr(deep_backtest_cache_inspection, "hit", False)
        ),
        needs_tf_for_deep_training=plan.needs_tf_for_deep_training,
    )


__all__ = ["update_experiment_plan_for_backtest"]
