from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PipelineRuntimeDeps:
    PreparedDataCacheInfo: type
    SKEW_CONTEXT_FEATURES: Any
    ResourceMonitor: Any
    VAL_KEY: str
    append_backtest_history: Any
    append_experiment_history: Any
    append_optuna_history: Any
    build_classical_feature_bundle: Any
    build_model_builders: Any
    build_probability_ensemble: Any
    compute_baselines: Any
    evaluate_champion_gate: Any
    load_or_prepare_training_data: Any
    load_streaming_history: Any
    persist_to_sqlite: Any
    plot_backtest_history: Any
    plot_history_best_runs: Any
    plot_learning_curves: Any
    plot_model_comparison: Any
    plot_optuna_best_runs: Any
    plot_run_leaderboard: Any
    refresh_analytics_database: Any
    resolve_cached_deep_training_artifacts: Any
    restore_deep_reporting_artifacts: Any
    inspect_temporal_backtest_cache: Any
    run_classical_benchmarks: Any
    run_data_quality_gate: Any
    run_drift_diagnostics: Any
    run_extended_evaluation: Any
    run_friction_proxy_analysis: Any
    run_moonshot_lab: Any
    run_optuna_tuning: Any
    run_policy_simulation: Any
    run_robustness_slice_evaluation: Any
    run_shap_analysis: Any
    run_temporal_backtest: Any
    save_deep_reporting_artifacts: Any
    save_histories_json: Any
    save_utilization_plot: Any
    train_and_evaluate_models: Any
    train_retrieval_stack: Any
    write_ablation_summary: Any
    write_benchmark_protocol: Any
    write_control_room_report: Any
    write_experiment_registry: Any
    write_run_report: Any
    write_significance_summary: Any


__all__ = ["PipelineRuntimeDeps"]
