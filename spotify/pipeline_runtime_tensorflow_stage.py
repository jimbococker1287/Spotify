from __future__ import annotations

from typing import Any

from .pipeline_helpers import _release_deep_runtime_resources
from .pipeline_runtime_experiment_types import PipelineExperimentContext
from .runtime import configure_process_env, load_tensorflow_runtime, select_distribution_strategy


def init_tensorflow_runtime(
    *,
    context: PipelineExperimentContext,
    deep_cache_stats: dict[str, object],
    needs_tf_for_deep_training: bool,
    needs_tf_for_deep_backtest: bool,
    deep_backtest_cache_inspection,
) -> tuple[Any, Any]:
    tf = None
    strategy = None
    selected_deep_backtest_models = list(getattr(deep_backtest_cache_inspection, "deep_models", ()))
    if needs_tf_for_deep_training and needs_tf_for_deep_backtest:
        initialization_reason = "deep_training_and_deep_backtest"
        planned_release_point = "after_temporal_backtest"
    elif needs_tf_for_deep_training:
        initialization_reason = "deep_training"
        planned_release_point = "before_temporal_backtest"
    else:
        initialization_reason = "deep_backtest"
        planned_release_point = "after_temporal_backtest"
    if needs_tf_for_deep_training or needs_tf_for_deep_backtest:
        with context.phase_recorder.phase(
            "tensorflow_runtime_init",
            run_deep_models=context.run_deep_models,
            run_deep_backtest=context.run_deep_backtest,
            deep_cache_hit_models=list(deep_cache_stats.get("hit_model_names", [])),
            deep_cache_miss_models=list(deep_cache_stats.get("miss_model_names", [])),
            selected_deep_backtest_models=selected_deep_backtest_models,
            backtest_cache_enabled=bool(getattr(deep_backtest_cache_inspection, "enabled", False)),
            backtest_cache_hit=bool(getattr(deep_backtest_cache_inspection, "hit", False)),
            backtest_cache_key=str(getattr(deep_backtest_cache_inspection, "cache_key", "")),
            initialization_reason=initialization_reason,
            planned_release_point=planned_release_point,
            pre_tensorflow_stages_completed=[
                "classical_benchmarks",
                "optuna_tuning",
                "retrieval_stack",
                "backtest_stage_planning",
            ],
        ) as phase:
            configure_process_env()
            tf = load_tensorflow_runtime(context.logger)
            tf.random.set_seed(context.config.random_seed)
            strategy = select_distribution_strategy(tf, logger=context.logger)
            device_count = int(getattr(strategy, "num_replicas_in_sync", 1))
            phase["device_count"] = device_count
            phase["initialized_for_deep_training"] = bool(needs_tf_for_deep_training)
            phase["initialized_for_deep_backtest"] = bool(needs_tf_for_deep_backtest)
            phase["cache_hit_count"] = int(len(deep_cache_stats.get("hit_model_names", [])))
            phase["cache_miss_count"] = int(len(deep_cache_stats.get("miss_model_names", [])))
            context.logger.info("Number of devices: %s", device_count)
        return tf, strategy

    if selected_deep_backtest_models and context.run_deep_models:
        skip_reason = "deep_training_fully_cached_and_deep_backtest_cached"
    elif selected_deep_backtest_models:
        skip_reason = "deep_training_not_requested_and_deep_backtest_cached"
    elif context.run_deep_models:
        skip_reason = "deep_training_fully_cached_and_deep_backtest_not_requested"
    else:
        skip_reason = "deep_models_disabled_and_deep_backtest_not_requested"
    context.phase_recorder.skip(
        "tensorflow_runtime_init",
        reason=skip_reason,
        selected_deep_backtest_models=selected_deep_backtest_models,
        backtest_cache_enabled=bool(getattr(deep_backtest_cache_inspection, "enabled", False)),
        backtest_cache_hit=bool(getattr(deep_backtest_cache_inspection, "hit", False)),
        backtest_cache_key=str(getattr(deep_backtest_cache_inspection, "cache_key", "")),
        tensorflow_required=False,
        deep_training_requires_tensorflow=bool(needs_tf_for_deep_training),
        deep_backtest_requires_tensorflow=bool(needs_tf_for_deep_backtest),
        deep_cache_hit_models=list(deep_cache_stats.get("hit_model_names", [])),
        deep_cache_miss_models=list(deep_cache_stats.get("miss_model_names", [])),
    )
    return None, None


def release_deep_runtime_resources(
    *,
    context: PipelineExperimentContext,
    tf,
    release_point: str,
    next_stage: str | None,
    deep_backtest_required: bool,
) -> None:
    if tf is not None:
        with context.phase_recorder.phase(
            "release_deep_runtime_resources",
            release_point=release_point,
            next_stage=next_stage or "",
            deep_backtest_required=deep_backtest_required,
            cleanup_gc_env_var="SPOTIFY_TF_CLEANUP_GC",
        ) as phase:
            phase.update(_release_deep_runtime_resources(tf, context.logger))
    else:
        context.phase_recorder.skip(
            "release_deep_runtime_resources",
            reason="tensorflow_not_initialized",
            release_point=release_point,
            next_stage=next_stage or "",
            deep_backtest_required=deep_backtest_required,
        )


__all__ = ["init_tensorflow_runtime", "release_deep_runtime_resources"]
