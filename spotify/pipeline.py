from __future__ import annotations

import os
import random

import numpy as np

from .config import PipelineConfig, configure_logging
from .data import append_audio_features, engineer_features, load_streaming_history, prepare_training_data
from .explainability import run_shap_analysis
from .modeling import build_model_builders
from .monitoring import ResourceMonitor
from .reporting import (
    VAL_KEY,
    persist_to_sqlite,
    plot_learning_curves,
    plot_model_comparison,
    save_histories_json,
    save_utilization_plot,
)
from .runtime import configure_process_env, load_tensorflow_runtime, select_distribution_strategy
from .training import compute_baselines, train_and_evaluate_models


def run_pipeline(config: PipelineConfig) -> None:
    logger = configure_logging(config.log_path)
    logger.info("Starting Spotify training pipeline")
    logger.info("Data directory: %s", config.data_dir)
    logger.info("Output directory: %s", config.output_dir)

    # Keep matplotlib/font caches inside project outputs to avoid macOS
    # permission issues in sandboxed environments.
    mpl_config_dir = config.output_dir / ".mplconfig"
    xdg_cache_dir = config.output_dir / ".cache"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    xdg_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache_dir))

    configure_process_env()
    tf = load_tensorflow_runtime(logger)

    np.random.seed(config.random_seed)
    random.seed(config.random_seed)
    tf.random.set_seed(config.random_seed)

    strategy = select_distribution_strategy(tf)
    logger.info("Number of devices: %s", getattr(strategy, "num_replicas_in_sync", 1))

    df = load_streaming_history(config.data_dir, config.include_video, logger)
    df = engineer_features(df, config.max_artists, logger)
    df = append_audio_features(df, config.enable_spotify_features, logger)

    prepared = prepare_training_data(
        df=df,
        sequence_length=config.sequence_length,
        scaler_path=config.scaler_path,
        logger=logger,
    )

    compute_baselines(prepared, logger)

    model_builders = build_model_builders(
        sequence_length=config.sequence_length,
        num_artists=prepared.num_artists,
        num_ctx=prepared.num_ctx,
        selected_names=config.model_names,
    )

    monitor = ResourceMonitor(logger)
    monitor.start()
    try:
        artifacts = train_and_evaluate_models(
            data=prepared,
            model_builders=model_builders,
            batch_size=config.batch_size,
            epochs=config.epochs,
            output_dir=config.output_dir,
            strategy=strategy,
            logger=logger,
        )
    finally:
        monitor.stop()

    plot_model_comparison(artifacts.histories, config.output_dir)
    plot_learning_curves(artifacts.histories, config.output_dir)
    save_histories_json(artifacts.histories, config.output_dir)
    save_utilization_plot(monitor.cpu_usage, monitor.gpu_usage, config.output_dir)

    if config.enable_shap:
        run_shap_analysis(artifacts.histories, config.output_dir, prepared, logger)
    else:
        logger.info("Skipping SHAP analysis because --no-shap was set.")

    persist_to_sqlite(
        df=prepared.df,
        histories=artifacts.histories,
        cpu_usage=monitor.cpu_usage,
        gpu_usage=monitor.gpu_usage,
        db_path=config.db_path,
    )

    logger.info("Final Validation Artist Accuracy (Top-1 / Top-5):")
    for name, history in artifacts.histories.items():
        val_key = VAL_KEY if VAL_KEY in history.history else "val_sparse_categorical_accuracy"
        top1 = history.history[val_key][-1]
        top5_key = "val_artist_output_top_5" if "val_artist_output_top_5" in history.history else "val_top_5"
        top5 = history.history.get(top5_key, [np.nan])[-1]
        logger.info("%s: Top-1=%.4f | Top-5=%.4f", name, top1, top5)

    logger.info("Pipeline completed successfully")
