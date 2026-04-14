from __future__ import annotations

from pathlib import Path
from typing import Any

from .pipeline_helpers import _append_existing_artifact_path
from .pipeline_runtime_experiment_types import PipelineExperimentContext


def _build_optuna_result_payload(*, row) -> dict[str, object]:
    return {
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


def append_optuna_results(
    *,
    context: PipelineExperimentContext,
    optuna_dir: Path,
    tuned_results: list[Any],
) -> None:
    for row in tuned_results:
        payload = _build_optuna_result_payload(row=row)
        context.result_rows.append(payload)
        context.optuna_rows.append(payload)
        _append_existing_artifact_path(context.artifact_paths, row.prediction_bundle_path)
        _append_existing_artifact_path(context.artifact_paths, row.estimator_artifact_path)
    if optuna_dir.exists():
        context.artifact_paths.extend(sorted(p for p in optuna_dir.glob("*") if p.is_file()))


def record_optuna_phase_summary(*, phase, optuna_cache_stats: dict[str, object], tuned_results: list[Any]) -> None:
    phase["result_count"] = int(len(tuned_results))
    phase["cache_enabled"] = bool(optuna_cache_stats.get("enabled", False))
    phase["cache_fingerprint"] = str(optuna_cache_stats.get("fingerprint", ""))
    phase["cache_hit_models"] = list(optuna_cache_stats.get("hit_model_names", []))
    phase["cache_hit_count"] = int(len(optuna_cache_stats.get("hit_model_names", [])))
    phase["cache_miss_models"] = list(optuna_cache_stats.get("miss_model_names", []))
    phase["cache_miss_count"] = int(len(optuna_cache_stats.get("miss_model_names", [])))
    phase["warm_start_model_names"] = list(optuna_cache_stats.get("warm_start_model_names", []))


__all__ = [
    "append_optuna_results",
    "record_optuna_phase_summary",
]
