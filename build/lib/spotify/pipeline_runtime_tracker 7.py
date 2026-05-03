from __future__ import annotations

import os

from .config import PipelineConfig
from .tracking import MlflowTracker


def init_mlflow_tracker(*, config: PipelineConfig, logger, phase_recorder, run_id: str) -> MlflowTracker:
    with phase_recorder.phase("mlflow_tracking_init", enabled=config.enable_mlflow) as phase:
        tracker = MlflowTracker(
            enabled=config.enable_mlflow,
            run_id=run_id,
            run_name=config.run_name,
            tracking_uri=config.mlflow_tracking_uri,
            experiment_name=config.mlflow_experiment,
            default_tracking_dir=config.output_dir / "mlruns",
            logger=logger,
        )
        tracker.log_params(
            {
                "profile": config.profile,
                "run_name": config.run_name or "",
                "sequence_length": config.sequence_length,
                "max_artists": config.max_artists,
                "batch_size": config.batch_size,
                "epochs": config.epochs,
                "random_seed": config.random_seed,
                "include_video": config.include_video,
                "enable_spotify_features": config.enable_spotify_features,
                "enable_shap": config.enable_shap,
                "classical_only": config.classical_only,
                "deep_models": config.model_names,
                "classical_models": config.classical_model_names,
                "enable_retrieval_stack": config.enable_retrieval_stack,
                "enable_self_supervised_pretraining": config.enable_self_supervised_pretraining,
                "enable_friction_analysis": config.enable_friction_analysis,
                "enable_moonshot_lab": config.enable_moonshot_lab,
                "retrieval_candidate_k": config.retrieval_candidate_k,
                "enable_optuna": config.enable_optuna,
                "optuna_trials": config.optuna_trials,
                "optuna_models": config.optuna_model_names,
                "enable_temporal_backtest": config.enable_temporal_backtest,
                "temporal_backtest_folds": config.temporal_backtest_folds,
                "temporal_backtest_models": config.temporal_backtest_model_names,
                "temporal_backtest_adaptation_mode": os.getenv("SPOTIFY_BACKTEST_ADAPTATION_MODE", "cold"),
            }
        )
        phase["tracking_enabled"] = bool(config.enable_mlflow)
        return tracker


__all__ = ["init_mlflow_tracker"]
