from __future__ import annotations

from .pipeline_helpers import _track_file
from .run_artifacts import write_json


def track_pipeline_outputs(*, artifact_paths, backtest_rows, result_rows, tracker) -> None:
    if tracker is not None:
        tracker.log_result_rows(result_rows)
        tracker.log_backtest_rows(backtest_rows)
    for path in artifact_paths:
        _track_file(tracker, path)


def finalize_pipeline_run(
    *,
    final_tracker_status: str,
    logger,
    manifest_path,
    manifest_payload,
    phase_recorder,
    run_dir,
    tracker,
) -> None:
    try:
        phase_json_path, phase_csv_path, phase_payload = phase_recorder.write_artifacts(
            run_dir=run_dir,
            final_status=final_tracker_status,
        )
        slowest_phase = phase_payload.get("slowest_phase", {})
        if manifest_payload is not None:
            manifest_payload["pipeline_status"] = final_tracker_status
            manifest_payload["phase_timings"] = {
                "json_path": str(phase_json_path),
                "csv_path": str(phase_csv_path),
                "total_seconds": float(phase_payload.get("total_seconds", 0.0)),
                "measured_seconds": float(phase_payload.get("measured_seconds", 0.0)),
                "unmeasured_overhead_seconds": float(phase_payload.get("unmeasured_overhead_seconds", 0.0)),
                "phase_count": int(phase_payload.get("phase_count", 0)),
                "completed_phase_count": int(phase_payload.get("completed_phase_count", 0)),
                "non_skipped_phase_count": int(phase_payload.get("non_skipped_phase_count", 0)),
                "slowest_phase": dict(slowest_phase) if isinstance(slowest_phase, dict) else {},
                "slowest_phases": list(phase_payload.get("slowest_phases", [])),
            }
            write_json(manifest_path, manifest_payload)
        logger.info(
            "Recorded pipeline phase timings: total=%.2fs slowest=%s (%.2fs)",
            float(phase_payload.get("total_seconds", 0.0)),
            str((slowest_phase or {}).get("phase_name", "n/a")),
            float((slowest_phase or {}).get("duration_seconds", 0.0)),
        )
    except Exception as exc:
        logger.warning("Unable to persist pipeline phase timings: %s", exc)
    if tracker is not None:
        tracker.end(status=final_tracker_status)


__all__ = [
    "finalize_pipeline_run",
    "track_pipeline_outputs",
]
