from __future__ import annotations

from collections.abc import Mapping

from .pipeline_runtime_analysis import PipelineAnalysisDeps
from .pipeline_runtime_dependency_types import PipelineRuntimeDeps
from .pipeline_runtime_experiment_types import PipelineExperimentDeps


def build_runtime_deps(*, imported_deps: Mapping[str, object]) -> PipelineRuntimeDeps:
    return PipelineRuntimeDeps(**dict(imported_deps))


def build_experiment_deps(*, runtime_deps: PipelineRuntimeDeps) -> PipelineExperimentDeps:
    return PipelineExperimentDeps(
        ResourceMonitor=runtime_deps.ResourceMonitor,
        VAL_KEY=runtime_deps.VAL_KEY,
        build_classical_feature_bundle=runtime_deps.build_classical_feature_bundle,
        build_model_builders=runtime_deps.build_model_builders,
        persist_to_sqlite=runtime_deps.persist_to_sqlite,
        plot_learning_curves=runtime_deps.plot_learning_curves,
        plot_model_comparison=runtime_deps.plot_model_comparison,
        resolve_cached_deep_training_artifacts=runtime_deps.resolve_cached_deep_training_artifacts,
        restore_deep_reporting_artifacts=runtime_deps.restore_deep_reporting_artifacts,
        inspect_temporal_backtest_cache=runtime_deps.inspect_temporal_backtest_cache,
        run_classical_benchmarks=runtime_deps.run_classical_benchmarks,
        run_optuna_tuning=runtime_deps.run_optuna_tuning,
        run_shap_analysis=runtime_deps.run_shap_analysis,
        run_temporal_backtest=runtime_deps.run_temporal_backtest,
        save_deep_reporting_artifacts=runtime_deps.save_deep_reporting_artifacts,
        save_histories_json=runtime_deps.save_histories_json,
        save_utilization_plot=runtime_deps.save_utilization_plot,
        train_and_evaluate_models=runtime_deps.train_and_evaluate_models,
        train_retrieval_stack=runtime_deps.train_retrieval_stack,
    )


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


__all__ = [
    "build_analysis_deps",
    "build_experiment_deps",
    "build_runtime_deps",
]
