from __future__ import annotations

from .config import PipelineConfig, configure_logging
from .pipeline_helpers import _build_run_id
from .pipeline_runtime_runner_types import PipelineRunSetup
from .run_timing import RunPhaseRecorder


def initialize_pipeline_run(*, config: PipelineConfig) -> PipelineRunSetup:
    run_id = _build_run_id(config)
    run_dir = config.output_dir / "runs" / run_id
    history_dir = config.output_dir / "history"
    manifest_path = run_dir / "run_manifest.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)

    logger = configure_logging(run_dir / "train.log")
    phase_recorder = RunPhaseRecorder(run_id=run_id)
    logger.info("Starting Spotify training pipeline")
    logger.info("Run ID: %s", run_id)
    if config.run_name:
        logger.info("Run Name: %s", config.run_name)
    logger.info("Data directory: %s", config.data_dir)
    logger.info("Output root: %s", config.output_dir)
    logger.info("Run output directory: %s", run_dir)

    return PipelineRunSetup(
        artifact_paths=[run_dir / "train.log"],
        history_dir=history_dir,
        logger=logger,
        manifest_path=manifest_path,
        phase_recorder=phase_recorder,
        run_dir=run_dir,
        run_id=run_id,
    )


__all__ = ["initialize_pipeline_run"]
