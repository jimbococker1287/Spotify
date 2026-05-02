from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class PipelineAnalysisContext:
    artifact_paths: list[Path]
    artist_labels: list[str]
    backtest_rows: list[dict[str, object]]
    cache_info_payload: dict[str, object]
    classical_feature_bundle: Any
    config: Any
    history_dir: Path
    logger: Any
    manifest_path: Path
    optuna_rows: list[dict[str, object]]
    phase_recorder: Any
    prepared: Any
    raw_df: Any
    result_rows: list[dict[str, object]]
    run_classical_models: bool
    run_dir: Path
    run_id: str


@dataclass
class PipelineAnalysisDeps:
    append_backtest_history: Any
    append_experiment_history: Any
    append_optuna_history: Any
    build_probability_ensemble: Any
    evaluate_champion_gate: Any
    plot_backtest_history: Any
    plot_history_best_runs: Any
    plot_optuna_best_runs: Any
    plot_run_leaderboard: Any
    refresh_analytics_database: Any
    run_drift_diagnostics: Any
    run_extended_evaluation: Any
    run_friction_proxy_analysis: Any
    run_moonshot_lab: Any
    run_policy_simulation: Any
    run_robustness_slice_evaluation: Any
    write_ablation_summary: Any
    write_benchmark_protocol: Any
    write_control_room_report: Any
    write_experiment_registry: Any
    write_run_report: Any
    write_significance_summary: Any


@dataclass
class PipelineAnalysisOutputs:
    manifest_payload: dict[str, object]
    strict_gate_error: str | None


__all__ = [
    "PipelineAnalysisContext",
    "PipelineAnalysisDeps",
    "PipelineAnalysisOutputs",
]
