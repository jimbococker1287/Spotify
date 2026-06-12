from __future__ import annotations

from .pipeline_runtime_experiment_backtest_plan import update_experiment_plan_for_backtest
from .pipeline_runtime_experiment_deep_plan import build_experiment_plan
from .pipeline_runtime_experiment_plan_types import PipelineExperimentPlan


def experiment_plan_metadata(plan: PipelineExperimentPlan) -> dict[str, object]:
    backtest_inspection = plan.deep_backtest_cache_inspection
    needs_tf_for_deep_training = bool(plan.needs_tf_for_deep_training)
    needs_tf_for_deep_backtest = bool(plan.needs_tf_for_deep_backtest)
    if needs_tf_for_deep_backtest:
        tensorflow_release_point = "after_temporal_backtest"
    elif needs_tf_for_deep_training:
        tensorflow_release_point = "before_temporal_backtest"
    else:
        tensorflow_release_point = "not_initialized"

    return {
        "tensorflow_required": needs_tf_for_deep_training or needs_tf_for_deep_backtest,
        "tensorflow_required_for_deep_training": needs_tf_for_deep_training,
        "tensorflow_required_for_deep_backtest": needs_tf_for_deep_backtest,
        "tensorflow_release_point": tensorflow_release_point,
        "deep_training_cache_enabled": bool(plan.deep_cache_stats.get("enabled", False)),
        "deep_training_cache_hit_models": list(plan.deep_cache_stats.get("hit_model_names", [])),
        "deep_training_cache_miss_models": list(plan.deep_cache_stats.get("miss_model_names", [])),
        "backtest_cache_inspected": backtest_inspection is not None,
        "backtest_cache_enabled": bool(getattr(backtest_inspection, "enabled", False)),
        "backtest_cache_hit": bool(getattr(backtest_inspection, "hit", False)),
        "backtest_cache_key": str(getattr(backtest_inspection, "cache_key", "")),
        "backtest_classical_models": list(getattr(backtest_inspection, "classical_models", ())),
        "backtest_deep_models": list(getattr(backtest_inspection, "deep_models", ())),
        "backtest_retrieval_models": list(getattr(backtest_inspection, "retrieval_models", ())),
    }


__all__ = [
    "PipelineExperimentPlan",
    "build_experiment_plan",
    "experiment_plan_metadata",
    "update_experiment_plan_for_backtest",
]
