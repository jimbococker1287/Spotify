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
from .pipeline_runtime_experiment_planning import (
    build_experiment_plan,
    experiment_plan_metadata,
    update_experiment_plan_for_backtest,
)
from .pipeline_runtime_search_stage import (
    run_classical_benchmarks,
    run_optuna_tuning,
    run_retrieval_stack,
    run_temporal_backtest,
)


def run_experiment_stages(*, context: PipelineExperimentContext, deps: PipelineExperimentDeps) -> PipelineExperimentOutputs:
    with context.phase_recorder.phase(
        "experiment_stage_planning",
        cpu_stages_before_tensorflow=[
            "classical_benchmarks",
            "optuna_tuning",
            "retrieval_stack",
        ],
        tensorflow_initialization_deferred_until_after_backtest_cache_inspection=True,
    ) as phase:
        plan = build_experiment_plan(context=context, deps=deps)
        phase.update(experiment_plan_metadata(plan))

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
    with context.phase_recorder.phase(
        "backtest_stage_planning",
        selected_backtest_model_names=list(selected_backtest_model_names),
        tuned_backtest_model_names=sorted(tuned_backtest_specs),
    ) as phase:
        plan = update_experiment_plan_for_backtest(
            context=context,
            deps=deps,
            plan=plan,
            selected_backtest_model_names=selected_backtest_model_names,
            tuned_backtest_specs=tuned_backtest_specs,
        )
        phase.update(experiment_plan_metadata(plan))

    tf, strategy = init_tensorflow_runtime(
        context=context,
        deep_cache_stats=plan.deep_cache_stats,
        needs_tf_for_deep_training=plan.needs_tf_for_deep_training,
        needs_tf_for_deep_backtest=plan.needs_tf_for_deep_backtest,
        deep_backtest_cache_inspection=plan.deep_backtest_cache_inspection,
    )
    tensorflow_release_recorded = False
    try:
        run_deep_model_training(
            context=context,
            deps=deps,
            deep_cache_plan=plan.deep_cache_plan,
            deep_cache_stats=plan.deep_cache_stats,
            strategy=strategy,
        )
        if tf is not None and not plan.needs_tf_for_deep_backtest:
            strategy = None
            release_deep_runtime_resources(
                context=context,
                tf=tf,
                release_point="before_temporal_backtest",
                next_stage="temporal_backtest",
                deep_backtest_required=False,
            )
            tf = None
            tensorflow_release_recorded = True

        run_temporal_backtest(
            context=context,
            deps=deps,
            classical_feature_bundle=plan.classical_feature_bundle,
            selected_backtest_model_names=selected_backtest_model_names,
            strategy=strategy,
            tuned_backtest_specs=tuned_backtest_specs,
        )
    finally:
        if tf is not None:
            strategy = None
            release_deep_runtime_resources(
                context=context,
                tf=tf,
                release_point="after_temporal_backtest",
                next_stage=None,
                deep_backtest_required=plan.needs_tf_for_deep_backtest,
            )
        elif not tensorflow_release_recorded:
            release_deep_runtime_resources(
                context=context,
                tf=None,
                release_point="not_initialized",
                next_stage=None,
                deep_backtest_required=plan.needs_tf_for_deep_backtest,
            )

    return PipelineExperimentOutputs(classical_feature_bundle=plan.classical_feature_bundle)


__all__ = [
    "PipelineExperimentContext",
    "PipelineExperimentDeps",
    "PipelineExperimentOutputs",
    "run_experiment_stages",
]
