from __future__ import annotations

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
from .pipeline_runtime_experiment_planning import build_experiment_plan, update_experiment_plan_for_backtest
from .pipeline_runtime_search_stage import (
    run_classical_benchmarks,
    run_optuna_tuning,
    run_retrieval_stack,
    run_temporal_backtest,
)


def run_experiment_stages(*, context: PipelineExperimentContext, deps: PipelineExperimentDeps) -> PipelineExperimentOutputs:
    plan = build_experiment_plan(context=context, deps=deps)
    classical_cache_stats: dict[str, object] = {}
    classical_results = run_classical_benchmarks(
        context=context,
        deps=deps,
        classical_feature_bundle=plan.classical_feature_bundle,
        classical_cache_stats=classical_cache_stats,
    )

    optuna_cache_stats: dict[str, object] = {}
    selected_backtest_model_names, tuned_backtest_specs = run_optuna_tuning(
        context=context,
        deps=deps,
        classical_feature_bundle=plan.classical_feature_bundle,
        classical_results=classical_results,
        optuna_cache_stats=optuna_cache_stats,
    )
    run_retrieval_stack(context=context, deps=deps)
    plan = update_experiment_plan_for_backtest(
        context=context,
        deps=deps,
        plan=plan,
        selected_backtest_model_names=selected_backtest_model_names,
        tuned_backtest_specs=tuned_backtest_specs,
    )
    tf, strategy = init_tensorflow_runtime(
        context=context,
        deep_cache_stats=plan.deep_cache_stats,
        needs_tf_for_deep_training=plan.needs_tf_for_deep_training,
        needs_tf_for_deep_backtest=plan.needs_tf_for_deep_backtest,
        deep_backtest_cache_inspection=plan.deep_backtest_cache_inspection,
    )
    run_deep_model_training(
        context=context,
        deps=deps,
        deep_cache_plan=plan.deep_cache_plan,
        deep_cache_stats=plan.deep_cache_stats,
        strategy=strategy,
    )
    run_temporal_backtest(
        context=context,
        deps=deps,
        classical_feature_bundle=plan.classical_feature_bundle,
        selected_backtest_model_names=selected_backtest_model_names,
        strategy=strategy,
        tuned_backtest_specs=tuned_backtest_specs,
    )
    release_deep_runtime_resources(context=context, tf=tf)

    return PipelineExperimentOutputs(classical_feature_bundle=plan.classical_feature_bundle)


__all__ = [
    "PipelineExperimentContext",
    "PipelineExperimentDeps",
    "PipelineExperimentOutputs",
    "run_experiment_stages",
]
