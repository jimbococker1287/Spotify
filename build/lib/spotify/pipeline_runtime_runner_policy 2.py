from __future__ import annotations

from .config import DEFAULT_MODEL_NAMES, PipelineConfig
from .pipeline_runtime_runner_types import PipelineRunPolicy
from .runtime import (
    detect_acceleration_environment,
    should_disable_deep_models_for_cpu_only_full_pass,
    should_prefer_compatibility_python_for_deep_runtime,
)


def _resolve_deep_model_policy(*, config: PipelineConfig, logger) -> bool:
    run_deep_models = (not config.classical_only) and bool(config.model_names)
    acceleration_summary = detect_acceleration_environment()
    if run_deep_models:
        prefer_alt_python, prefer_alt_reason = should_prefer_compatibility_python_for_deep_runtime(
            acceleration_summary
        )
        if prefer_alt_python:
            logger.warning(
                "Current deep-runtime interpreter looks compatibility-limited (%s). "
                "Prefer the launcher auto-route or set PYTHON_BIN=.venv-metal/bin/python.",
                prefer_alt_reason,
            )
    if run_deep_models and config.profile == "full":
        disable_deep_models, deep_disable_reason = should_disable_deep_models_for_cpu_only_full_pass(
            acceleration_summary
        )
        if disable_deep_models:
            run_deep_models = False
            logger.info(
                "Auto-disabling deep models for this full pass (%s). Requested models: %s. "
                "Override with SPOTIFY_FULL_DEEP_MODE_POLICY=on.",
                deep_disable_reason,
                ",".join(config.model_names),
            )
    return run_deep_models


def resolve_pipeline_run_policy(*, config: PipelineConfig, logger) -> PipelineRunPolicy:
    run_deep_models = _resolve_deep_model_policy(config=config, logger=logger)
    run_classical_models = bool(config.enable_classical_models)
    if config.classical_only:
        run_classical_models = True
    run_deep_backtest = run_deep_models and bool(config.enable_temporal_backtest) and any(
        model_name in DEFAULT_MODEL_NAMES for model_name in config.temporal_backtest_model_names
    )
    return PipelineRunPolicy(
        run_classical_models=run_classical_models,
        run_deep_backtest=run_deep_backtest,
        run_deep_models=run_deep_models,
    )


__all__ = ["resolve_pipeline_run_policy"]
