from __future__ import annotations

from pathlib import Path

from .config import PipelineConfig
from .pipeline_runtime_dependency_types import PipelineRuntimeDeps


def load_pipeline_runtime_raw_data(
    *,
    artifact_paths: list[Path],
    config: PipelineConfig,
    deps: PipelineRuntimeDeps,
    logger,
    phase_recorder,
    run_dir: Path,
):
    with phase_recorder.phase("data_loading", include_video=config.include_video) as phase:
        raw_df = deps.load_streaming_history(
            config.data_dir,
            include_video=config.include_video,
            logger=logger,
        )
        phase["raw_rows"] = int(len(raw_df))
        phase["raw_columns"] = int(len(getattr(raw_df, "columns", [])))

    data_quality_report_path = run_dir / "data_quality_report.json"
    with phase_recorder.phase("data_quality_gate") as phase:
        deps.run_data_quality_gate(raw_df, report_path=data_quality_report_path, logger=logger)
        phase["report_path"] = data_quality_report_path
    artifact_paths.append(data_quality_report_path)
    return raw_df


__all__ = ["load_pipeline_runtime_raw_data"]
