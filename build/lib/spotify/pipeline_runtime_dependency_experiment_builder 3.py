from __future__ import annotations

from .pipeline_runtime_dependency_types import PipelineRuntimeDeps
from .pipeline_runtime_experiment_types import PipelineExperimentDeps


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


__all__ = ["build_experiment_deps"]
