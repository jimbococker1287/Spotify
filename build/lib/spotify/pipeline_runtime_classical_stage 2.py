from __future__ import annotations

from typing import Any

from .pipeline_helpers import _append_existing_artifact_path
from .pipeline_runtime_experiment_types import PipelineExperimentContext, PipelineExperimentDeps


def run_classical_benchmarks(
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


__all__ = ["run_classical_benchmarks"]
