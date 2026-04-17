from __future__ import annotations

from .pipeline_runtime_analysis import PipelineAnalysisDeps
from .pipeline_runtime_dependency_types import PipelineRuntimeDeps


def build_analysis_deps(*, runtime_deps: PipelineRuntimeDeps) -> PipelineAnalysisDeps:
    return PipelineAnalysisDeps(
        append_backtest_history=runtime_deps.append_backtest_history,
        append_experiment_history=runtime_deps.append_experiment_history,
        append_optuna_history=runtime_deps.append_optuna_history,
        build_probability_ensemble=runtime_deps.build_probability_ensemble,
        evaluate_champion_gate=runtime_deps.evaluate_champion_gate,
        plot_backtest_history=runtime_deps.plot_backtest_history,
        plot_history_best_runs=runtime_deps.plot_history_best_runs,
        plot_optuna_best_runs=runtime_deps.plot_optuna_best_runs,
        plot_run_leaderboard=runtime_deps.plot_run_leaderboard,
        refresh_analytics_database=runtime_deps.refresh_analytics_database,
        run_drift_diagnostics=runtime_deps.run_drift_diagnostics,
        run_extended_evaluation=runtime_deps.run_extended_evaluation,
        run_friction_proxy_analysis=runtime_deps.run_friction_proxy_analysis,
        run_moonshot_lab=runtime_deps.run_moonshot_lab,
        run_policy_simulation=runtime_deps.run_policy_simulation,
        run_robustness_slice_evaluation=runtime_deps.run_robustness_slice_evaluation,
        write_ablation_summary=runtime_deps.write_ablation_summary,
        write_benchmark_protocol=runtime_deps.write_benchmark_protocol,
        write_control_room_report=runtime_deps.write_control_room_report,
        write_experiment_registry=runtime_deps.write_experiment_registry,
        write_run_report=runtime_deps.write_run_report,
        write_significance_summary=runtime_deps.write_significance_summary,
    )


__all__ = ["build_analysis_deps"]
