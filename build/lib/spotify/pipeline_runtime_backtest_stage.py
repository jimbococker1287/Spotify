from __future__ import annotations

from typing import Any
import os

from .pipeline_runtime_experiment_types import PipelineExperimentContext, PipelineExperimentDeps


def run_temporal_backtest(
    *,
    context: PipelineExperimentContext,
    deps: PipelineExperimentDeps,
    classical_feature_bundle: Any,
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
            deep_model_builders=None,
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
        phase["deep_backtest_builders"] = False
        phase["cache_enabled"] = bool(backtest_cache_stats.get("enabled", False))
        phase["cache_fingerprint"] = str(backtest_cache_stats.get("fingerprint", ""))
        phase["cache_key"] = str(backtest_cache_stats.get("cache_key", ""))
        phase["cache_hit"] = bool(backtest_cache_stats.get("hit", False))


__all__ = ["run_temporal_backtest"]
