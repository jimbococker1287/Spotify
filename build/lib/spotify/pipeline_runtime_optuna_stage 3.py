from __future__ import annotations

from typing import Any

from .pipeline_runtime_experiment_types import PipelineExperimentContext, PipelineExperimentDeps
from .pipeline_runtime_optuna_results import append_optuna_results, record_optuna_phase_summary
from .pipeline_runtime_optuna_shortlist import resolve_optuna_stage_model_names, resolve_tuned_backtest_specs


def run_optuna_tuning(
    *,
    context: PipelineExperimentContext,
    deps: PipelineExperimentDeps,
    classical_feature_bundle: Any,
    classical_results: list[Any],
    optuna_cache_stats: dict[str, object],
) -> tuple[tuple[str, ...], dict[str, dict[str, object]]]:
    selected_optuna_model_names, selected_backtest_model_names = resolve_optuna_stage_model_names(
        context=context,
        classical_results=classical_results,
    )
    tuned_backtest_specs: dict[str, dict[str, object]] = {}

    if context.run_classical_models and context.config.enable_optuna:
        optuna_dir = context.run_dir / "optuna"
        with context.phase_recorder.phase(
            "optuna_tuning",
            model_names=list(selected_optuna_model_names),
            candidate_model_names=list(context.config.optuna_model_names),
            trials=context.config.optuna_trials,
            timeout_seconds=context.config.optuna_timeout_seconds,
        ) as phase:
            tuned_results = deps.run_optuna_tuning(
                data=context.prepared,
                output_dir=optuna_dir,
                selected_models=selected_optuna_model_names,
                random_seed=context.config.random_seed,
                trials=context.config.optuna_trials,
                timeout_seconds=context.config.optuna_timeout_seconds,
                max_train_samples=context.config.classical_max_train_samples,
                max_eval_samples=context.config.classical_max_eval_samples,
                logger=context.logger,
                feature_bundle=classical_feature_bundle,
                cache_root=context.config.output_dir / "cache" / "optuna",
                cache_fingerprint=context.cache_fingerprint,
                cache_stats_out=optuna_cache_stats,
            )
            append_optuna_results(
                context=context,
                optuna_dir=optuna_dir,
                tuned_results=tuned_results,
            )
            record_optuna_phase_summary(
                phase=phase,
                optuna_cache_stats=optuna_cache_stats,
                tuned_results=tuned_results,
            )
            selected_backtest_model_names, tuned_backtest_specs = resolve_tuned_backtest_specs(
                selected_backtest_model_names=selected_backtest_model_names,
                tuned_results=tuned_results,
                logger=context.logger,
            )
        return selected_backtest_model_names, tuned_backtest_specs

    if context.config.enable_optuna:
        context.phase_recorder.skip("optuna_tuning", reason="classical_models_disabled")
        context.logger.info("Skipping Optuna tuning because classical models are disabled.")
    else:
        context.phase_recorder.skip("optuna_tuning", reason="optuna_disabled")
    return selected_backtest_model_names, tuned_backtest_specs


__all__ = ["run_optuna_tuning"]
