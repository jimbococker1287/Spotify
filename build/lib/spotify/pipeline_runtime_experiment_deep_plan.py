from __future__ import annotations

from .pipeline_runtime_experiment_plan_types import PipelineExperimentPlan
from .pipeline_runtime_experiment_types import PipelineExperimentContext, PipelineExperimentDeps


def build_experiment_plan(*, context: PipelineExperimentContext, deps: PipelineExperimentDeps) -> PipelineExperimentPlan:
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
    return PipelineExperimentPlan(
        classical_feature_bundle=classical_feature_bundle,
        deep_backtest_cache_inspection=None,
        deep_cache_plan=deep_cache_plan,
        deep_cache_stats=deep_cache_stats,
        needs_tf_for_deep_backtest=False,
        needs_tf_for_deep_training=bool(
            context.run_deep_models and deep_cache_plan is not None and deep_cache_plan.miss_model_names
        ),
    )


__all__ = ["build_experiment_plan"]
