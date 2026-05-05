from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import PipelineConfig


@dataclass
class PipelineExperimentContext:
    artifact_paths: list[Path]
    backtest_rows: list[dict[str, object]]
    cache_fingerprint: str
    config: PipelineConfig
    logger: Any
    optuna_rows: list[dict[str, object]]
    phase_recorder: Any
    prepared: Any
    result_rows: list[dict[str, object]]
    run_classical_models: bool
    run_deep_backtest: bool
    run_deep_models: bool
    run_dir: Path


@dataclass
class PipelineExperimentDeps:
    ResourceMonitor: Any
    VAL_KEY: str
    build_classical_feature_bundle: Any
    build_model_builders: Any
    persist_to_sqlite: Any
    plot_learning_curves: Any
    plot_model_comparison: Any
    resolve_cached_deep_training_artifacts: Any
    restore_deep_reporting_artifacts: Any
    inspect_temporal_backtest_cache: Any
    run_classical_benchmarks: Any
    run_optuna_tuning: Any
    run_shap_analysis: Any
    run_temporal_backtest: Any
    save_deep_reporting_artifacts: Any
    save_histories_json: Any
    save_utilization_plot: Any
    train_and_evaluate_models: Any
    train_retrieval_stack: Any


@dataclass
class PipelineExperimentOutputs:
    classical_feature_bundle: Any


__all__ = [
    "PipelineExperimentContext",
    "PipelineExperimentDeps",
    "PipelineExperimentOutputs",
]
