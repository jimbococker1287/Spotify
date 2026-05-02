from __future__ import annotations

from .pipeline_helpers import _write_json_artifact
from .pipeline_postrun_reporting import build_run_manifest, refresh_analytics_if_enabled, write_postrun_reports
from .pipeline_postrun_types import PipelinePostRunResult


def finalize_postrun(*, context, deps, stage_outputs) -> PipelinePostRunResult:
    artifact_paths = context.artifact_paths
    config = context.config
    logger = context.logger
    phase_recorder = context.phase_recorder
    prepared = context.prepared
    raw_df = context.raw_df
    result_rows = context.result_rows
    run_dir = context.run_dir
    run_id = context.run_id

    manifest = build_run_manifest(
        backtest_rows=context.backtest_rows,
        cache_info_payload=context.cache_info_payload,
        champion_alias_payload=stage_outputs.champion_outcome.champion_alias_payload,
        champion_gate=stage_outputs.champion_outcome.champion_gate,
        config=config,
        mlflow_artifact_cleanup_summary=stage_outputs.cleanup_outcome.mlflow_artifact_cleanup_summary,
        optuna_rows=context.optuna_rows,
        prepared=prepared,
        result_rows=result_rows,
        retention_summary=stage_outputs.cleanup_outcome.retention_summary,
        run_classical_models=context.run_classical_models,
        run_id=run_id,
        run_name=config.run_name,
        cleanup_summary=stage_outputs.cleanup_outcome.cleanup_summary,
    )
    _write_json_artifact(context.manifest_path, manifest, artifact_paths)
    _write_json_artifact(run_dir / "run_results.json", result_rows, artifact_paths)

    refresh_analytics_if_enabled(
        artifact_paths=artifact_paths,
        config=config,
        logger=logger,
        phase_recorder=phase_recorder,
        raw_df=raw_df,
        refresh_analytics_database=deps.refresh_analytics_database,
    )

    write_postrun_reports(
        artifact_paths=artifact_paths,
        champion_gate=stage_outputs.champion_outcome.champion_gate,
        config=config,
        history_csv=stage_outputs.history_csv,
        history_dir=context.history_dir,
        logger=logger,
        manifest=manifest,
        phase_recorder=phase_recorder,
        result_rows=result_rows,
        run_dir=run_dir,
        write_control_room_report=deps.write_control_room_report,
        write_run_report=deps.write_run_report,
    )

    return PipelinePostRunResult(
        manifest_payload=manifest,
        strict_gate_error=stage_outputs.champion_outcome.strict_gate_error,
    )


__all__ = ["finalize_postrun"]
