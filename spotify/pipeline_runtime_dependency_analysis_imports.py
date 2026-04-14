from __future__ import annotations


def load_pipeline_runtime_analysis_imports() -> dict[str, object]:
    from .drift import run_drift_diagnostics
    from .ensemble import build_probability_ensemble
    from .evaluation import run_extended_evaluation
    from .friction import run_friction_proxy_analysis
    from .moonshot_lab import run_moonshot_lab
    from .policy_eval import run_policy_simulation
    from .reporting import (
        append_backtest_history,
        append_experiment_history,
        append_optuna_history,
        plot_backtest_history,
        plot_history_best_runs,
        plot_optuna_best_runs,
        plot_run_leaderboard,
        write_run_report,
    )
    from .research_artifacts import (
        write_ablation_summary,
        write_benchmark_protocol,
        write_experiment_registry,
        write_significance_summary,
    )
    from .robustness import run_robustness_slice_evaluation

    return {
        "append_backtest_history": append_backtest_history,
        "append_experiment_history": append_experiment_history,
        "append_optuna_history": append_optuna_history,
        "build_probability_ensemble": build_probability_ensemble,
        "plot_backtest_history": plot_backtest_history,
        "plot_history_best_runs": plot_history_best_runs,
        "plot_optuna_best_runs": plot_optuna_best_runs,
        "plot_run_leaderboard": plot_run_leaderboard,
        "run_drift_diagnostics": run_drift_diagnostics,
        "run_extended_evaluation": run_extended_evaluation,
        "run_friction_proxy_analysis": run_friction_proxy_analysis,
        "run_moonshot_lab": run_moonshot_lab,
        "run_policy_simulation": run_policy_simulation,
        "run_robustness_slice_evaluation": run_robustness_slice_evaluation,
        "write_ablation_summary": write_ablation_summary,
        "write_benchmark_protocol": write_benchmark_protocol,
        "write_experiment_registry": write_experiment_registry,
        "write_run_report": write_run_report,
        "write_significance_summary": write_significance_summary,
    }


__all__ = ["load_pipeline_runtime_analysis_imports"]
