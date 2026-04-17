from __future__ import annotations

import logging
from types import SimpleNamespace

import spotify.pipeline_runtime_runner_setup as runner_setup
import spotify.pipeline_runtime_runner_policy as runner_policy


def _logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    return logger


def test_resolve_pipeline_run_policy_disables_deep_models_for_full_cpu_only_pass(monkeypatch) -> None:
    config = SimpleNamespace(
        classical_only=False,
        enable_classical_models=True,
        enable_temporal_backtest=True,
        model_names=("dense",),
        profile="full",
        temporal_backtest_model_names=("dense",),
    )
    monkeypatch.setattr(
        runner_policy,
        "detect_acceleration_environment",
        lambda: {"platform": "darwin", "machine": "arm64", "python_version": "3.13"},
    )
    monkeypatch.setattr(
        runner_policy,
        "should_prefer_compatibility_python_for_deep_runtime",
        lambda summary: (False, None),
    )
    monkeypatch.setattr(
        runner_policy,
        "should_disable_deep_models_for_cpu_only_full_pass",
        lambda summary: (True, "cpu_only_full_pass"),
    )

    policy = runner_setup.resolve_pipeline_run_policy(config=config, logger=_logger("spotify.test.runner_setup"))

    assert policy.run_deep_models is False
    assert policy.run_deep_backtest is False
    assert policy.run_classical_models is True
