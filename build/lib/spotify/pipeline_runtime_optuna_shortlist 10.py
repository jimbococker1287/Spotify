from __future__ import annotations

from typing import Any

from .pipeline_runtime_experiment_types import PipelineExperimentContext
from .pipeline_runtime_shortlists import (
    _resolve_shortlist_top_n,
    _shortlist_classical_model_names,
    _tuned_backtest_specs,
)


def resolve_optuna_stage_model_names(
    *,
    context: PipelineExperimentContext,
    classical_results: list[Any],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    selected_optuna_model_names = context.config.optuna_model_names
    selected_backtest_model_names = context.config.temporal_backtest_model_names
    if classical_results:
        selected_optuna_model_names = _shortlist_classical_model_names(
            context.config.optuna_model_names,
            classical_results,
            top_n=_resolve_shortlist_top_n("SPOTIFY_OPTUNA_SHORTLIST_TOP_N"),
            logger=context.logger,
            stage_label="Optuna",
        )
        selected_backtest_model_names = _shortlist_classical_model_names(
            context.config.temporal_backtest_model_names,
            classical_results,
            top_n=_resolve_shortlist_top_n("SPOTIFY_BACKTEST_SHORTLIST_TOP_N"),
            logger=context.logger,
            stage_label="Temporal backtest",
            passthrough_names=tuple(
                name
                for name in context.config.temporal_backtest_model_names
                if name not in context.config.classical_model_names
            ),
        )
    return selected_optuna_model_names, selected_backtest_model_names


def resolve_tuned_backtest_specs(
    *,
    selected_backtest_model_names: tuple[str, ...],
    tuned_results: list[Any],
    logger,
) -> tuple[tuple[str, ...], dict[str, dict[str, object]]]:
    return _tuned_backtest_specs(
        selected_backtest_model_names,
        tuned_results,
        logger=logger,
    )


__all__ = [
    "resolve_optuna_stage_model_names",
    "resolve_tuned_backtest_specs",
]
