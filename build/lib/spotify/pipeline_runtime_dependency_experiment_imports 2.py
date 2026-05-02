from __future__ import annotations


def _lazy_train_and_evaluate_models(*args, **kwargs):
    from .training import train_and_evaluate_models

    return train_and_evaluate_models(*args, **kwargs)


def load_pipeline_runtime_experiment_imports() -> dict[str, object]:
    from .benchmarks import build_classical_feature_bundle, run_classical_benchmarks
    from .explainability import run_shap_analysis
    from .modeling import build_model_builders
    from .reporting import (
        VAL_KEY,
        persist_to_sqlite,
        plot_learning_curves,
        plot_model_comparison,
        restore_deep_reporting_artifacts,
        save_deep_reporting_artifacts,
        save_histories_json,
        save_utilization_plot,
    )
    from .retrieval import train_retrieval_stack
    from .tuning import run_optuna_tuning

    return {
        "VAL_KEY": VAL_KEY,
        "build_classical_feature_bundle": build_classical_feature_bundle,
        "build_model_builders": build_model_builders,
        "persist_to_sqlite": persist_to_sqlite,
        "plot_learning_curves": plot_learning_curves,
        "plot_model_comparison": plot_model_comparison,
        "restore_deep_reporting_artifacts": restore_deep_reporting_artifacts,
        "run_classical_benchmarks": run_classical_benchmarks,
        "run_optuna_tuning": run_optuna_tuning,
        "run_shap_analysis": run_shap_analysis,
        "save_deep_reporting_artifacts": save_deep_reporting_artifacts,
        "save_histories_json": save_histories_json,
        "save_utilization_plot": save_utilization_plot,
        "train_and_evaluate_models": _lazy_train_and_evaluate_models,
        "train_retrieval_stack": train_retrieval_stack,
    }


__all__ = ["load_pipeline_runtime_experiment_imports"]
