from __future__ import annotations

from datetime import datetime
from pathlib import Path
import os
from typing import Any

from .pipeline_artifact_cache import (
    phase_artifact_cache_enabled,
    resolve_phase_artifact_cache_paths,
    restore_phase_artifact_cache,
    save_phase_artifact_cache,
    source_digest_for_paths,
    stable_payload_digest,
)

_THIS_SOURCE = Path(__file__).resolve()


def _normalized_result_rows_for_cache(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    ignored_keys = {"fit_seconds", "epochs"}
    for row in rows:
        payload = {
            str(key): value
            for key, value in row.items()
            if not str(key).endswith("_path") and str(key) not in ignored_keys
        }
        normalized.append(payload)
    normalized.sort(
        key=lambda row: (
            str(row.get("model_name", "")),
            str(row.get("model_type", "")),
            str(row.get("base_model_name", "")),
        )
    )
    return normalized


def _normalized_backtest_rows_for_cache(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    normalized = [{str(key): value for key, value in row.items()} for row in rows]
    normalized.sort(
        key=lambda row: (
            str(row.get("model_name", "")),
            int(row.get("fold", 0) or 0),
        )
    )
    return normalized


def _postrun_phase_cache_root(config: Any) -> Path:
    return config.output_dir / "cache" / "analysis_phase_artifacts"


def _phase_source_digest(*paths: Path) -> str:
    return source_digest_for_paths((_THIS_SOURCE, *paths))


def _restore_or_build_postrun_phase_artifacts(
    *,
    cache_enabled: bool,
    cache_root: Path,
    phase_name: str,
    cache_payload: dict[str, object],
    run_dir: Path,
    logger,
    build_artifacts,
) -> tuple[list[Path], bool]:
    cache_key = stable_payload_digest(cache_payload) if cache_enabled else ""
    cache_paths = (
        resolve_phase_artifact_cache_paths(
            cache_root=cache_root,
            namespace="postrun",
            phase_name=phase_name,
            cache_key=cache_key,
        )
        if cache_enabled
        else None
    )
    if cache_enabled and cache_paths is not None:
        restored = restore_phase_artifact_cache(
            cache_paths=cache_paths,
            run_dir=run_dir,
            logger=logger,
        )
        if restored is not None:
            return restored, True

    artifacts = list(build_artifacts())
    if cache_enabled and cache_paths is not None:
        save_phase_artifact_cache(
            cache_paths=cache_paths,
            cache_payload=cache_payload,
            run_dir=run_dir,
            artifact_paths=artifacts,
            logger=logger,
        )
    return artifacts, False


def run_history_exports(
    *,
    append_experiment_history,
    artifact_paths: list[Path],
    data_records: int,
    history_dir: Path,
    phase_recorder,
    plot_history_best_runs,
    plot_run_leaderboard,
    profile: str,
    result_rows: list[dict[str, object]],
    run_dir: Path,
    run_id: str,
    run_name: str | None,
) -> Path:
    with phase_recorder.phase("history_exports") as phase:
        leaderboard_path = plot_run_leaderboard(result_rows, run_dir)
        if leaderboard_path is not None:
            artifact_paths.append(leaderboard_path)

        history_csv = append_experiment_history(
            history_csv=history_dir / "experiment_history.csv",
            run_id=run_id,
            profile=profile,
            run_name=run_name,
            results=result_rows,
            data_records=data_records,
        )
        history_plot = plot_history_best_runs(history_csv, history_dir)
        artifact_paths.append(history_csv)
        if history_plot is not None:
            artifact_paths.append(history_plot)
        phase["leaderboard_plot_written"] = bool(leaderboard_path is not None)
        phase["experiment_history_rows"] = int(len(result_rows))
    return history_csv


def write_research_artifacts(
    *,
    artifact_paths: list[Path],
    backtest_rows: list[dict[str, object]],
    cache_info_payload: dict[str, object],
    config: Any,
    data,
    logger,
    phase_recorder,
    result_rows: list[dict[str, object]],
    run_dir: Path,
    run_id: str,
    write_ablation_summary,
    write_benchmark_protocol,
    write_experiment_registry,
    write_significance_summary,
) -> None:
    with phase_recorder.phase("research_artifacts") as phase:
        prepared_fingerprint = str(cache_info_payload.get("fingerprint", "")).strip()
        cache_enabled = bool(prepared_fingerprint) and phase_artifact_cache_enabled(
            env_var="SPOTIFY_CACHE_POSTRUN_RESEARCH_ARTIFACTS",
            default="1",
        )
        result_rows_digest = stable_payload_digest(_normalized_result_rows_for_cache(result_rows))
        backtest_rows_digest = stable_payload_digest(_normalized_backtest_rows_for_cache(backtest_rows))

        artifact_paths.extend(
            write_benchmark_protocol(
                output_dir=run_dir,
                run_id=run_id,
                profile=config.profile,
                data=data,
                cache_info=cache_info_payload,
                config=config,
            )
        )
        artifact_paths.append(
            write_experiment_registry(
                output_dir=run_dir,
                run_id=run_id,
                profile=config.profile,
                results=result_rows,
                backtest_rows=backtest_rows,
                config=config,
            )
        )

        ablation_artifacts, ablation_cache_hit = _restore_or_build_postrun_phase_artifacts(
            cache_enabled=cache_enabled,
            cache_root=_postrun_phase_cache_root(config),
            phase_name="ablation_summary",
            cache_payload={
                "prepared_fingerprint": prepared_fingerprint,
                "profile": str(config.profile),
                "result_rows_digest": result_rows_digest,
                "source_digest": _phase_source_digest(Path(write_ablation_summary.__code__.co_filename)),
            },
            run_dir=run_dir,
            logger=logger,
            build_artifacts=lambda: write_ablation_summary(output_dir=run_dir / "analysis", results=result_rows),
        )
        artifact_paths.extend(ablation_artifacts)

        significance_artifacts, significance_cache_hit = _restore_or_build_postrun_phase_artifacts(
            cache_enabled=cache_enabled,
            cache_root=_postrun_phase_cache_root(config),
            phase_name="backtest_significance",
            cache_payload={
                "prepared_fingerprint": prepared_fingerprint,
                "profile": str(config.profile),
                "result_rows_digest": result_rows_digest,
                "backtest_rows_digest": backtest_rows_digest,
                "source_digest": _phase_source_digest(Path(write_significance_summary.__code__.co_filename)),
            },
            run_dir=run_dir,
            logger=logger,
            build_artifacts=lambda: write_significance_summary(
                output_dir=run_dir / "analysis",
                results=result_rows,
                backtest_rows=backtest_rows,
            ),
        )
        artifact_paths.extend(significance_artifacts)
        phase["result_count"] = int(len(result_rows))
        phase["backtest_row_count"] = int(len(backtest_rows))
        phase["cache_enabled"] = bool(cache_enabled)
        phase["ablation_cache_hit"] = bool(ablation_cache_hit)
        phase["significance_cache_hit"] = bool(significance_cache_hit)
        phase["ablation_artifact_count"] = int(len(ablation_artifacts))
        phase["significance_artifact_count"] = int(len(significance_artifacts))


def write_history_rollups(
    *,
    append_backtest_history,
    append_optuna_history,
    artifact_paths: list[Path],
    backtest_rows: list[dict[str, object]],
    history_dir: Path,
    optuna_rows: list[dict[str, object]],
    phase_recorder,
    plot_backtest_history,
    plot_optuna_best_runs,
    profile: str,
    run_id: str,
    run_name: str | None,
) -> None:
    with phase_recorder.phase("history_rollups") as phase:
        if optuna_rows:
            optuna_history_csv = append_optuna_history(
                history_csv=history_dir / "optuna_history.csv",
                run_id=run_id,
                profile=profile,
                run_name=run_name,
                results=optuna_rows,
            )
            optuna_history_plot = plot_optuna_best_runs(optuna_history_csv, history_dir)
            artifact_paths.append(optuna_history_csv)
            if optuna_history_plot is not None:
                artifact_paths.append(optuna_history_plot)

        if backtest_rows:
            backtest_history_csv = append_backtest_history(
                history_csv=history_dir / "backtest_history.csv",
                run_id=run_id,
                profile=profile,
                run_name=run_name,
                rows=backtest_rows,
            )
            backtest_history_plot = plot_backtest_history(backtest_history_csv, history_dir)
            artifact_paths.append(backtest_history_csv)
            if backtest_history_plot is not None:
                artifact_paths.append(backtest_history_plot)
        phase["optuna_row_count"] = int(len(optuna_rows))
        phase["backtest_row_count"] = int(len(backtest_rows))


def build_run_manifest(
    *,
    backtest_rows: list[dict[str, object]],
    cache_info_payload: dict[str, object],
    champion_alias_payload: dict[str, object],
    champion_gate: dict[str, object],
    config: Any,
    mlflow_artifact_cleanup_summary: dict[str, object],
    optuna_rows: list[dict[str, object]],
    prepared,
    result_rows: list[dict[str, object]],
    retention_summary: dict[str, object],
    run_classical_models: bool,
    run_id: str,
    run_name: str | None,
    cleanup_summary: dict[str, object],
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "run_name": run_name,
        "profile": config.profile,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "data_records": len(prepared.df),
        "num_artists": prepared.num_artists,
        "num_context_features": prepared.num_ctx,
        "deep_models": list(config.model_names),
        "classical_models": list(config.classical_model_names) if run_classical_models else [],
        "enable_retrieval_stack": config.enable_retrieval_stack,
        "enable_self_supervised_pretraining": config.enable_self_supervised_pretraining,
        "enable_friction_analysis": config.enable_friction_analysis,
        "enable_moonshot_lab": config.enable_moonshot_lab,
        "retrieval_candidate_k": config.retrieval_candidate_k,
        "enable_mlflow": config.enable_mlflow,
        "enable_optuna": config.enable_optuna,
        "optuna_models": list(config.optuna_model_names),
        "optuna_trials": config.optuna_trials,
        "enable_temporal_backtest": config.enable_temporal_backtest,
        "temporal_backtest_models": list(config.temporal_backtest_model_names),
        "temporal_backtest_folds": config.temporal_backtest_folds,
        "temporal_backtest_adaptation_mode": os.getenv("SPOTIFY_BACKTEST_ADAPTATION_MODE", "cold"),
        "backtest_rows": len(backtest_rows),
        "optuna_rows": len(optuna_rows),
        "cache": cache_info_payload,
        "champion_gate": champion_gate,
        "champion_alias": champion_alias_payload,
        "artifact_cleanup": cleanup_summary,
        "artifact_retention": retention_summary,
        "mlflow_artifact_cleanup": mlflow_artifact_cleanup_summary,
    }


def refresh_analytics_if_enabled(
    *,
    artifact_paths: list[Path],
    config: Any,
    logger,
    phase_recorder,
    raw_df,
    refresh_analytics_database,
) -> None:
    refresh_analytics_raw = os.getenv("SPOTIFY_REFRESH_ANALYTICS_DB", "1").strip().lower()
    refresh_analytics = refresh_analytics_raw not in ("0", "false", "no", "off")
    if refresh_analytics:
        with phase_recorder.phase("analytics_refresh") as phase:
            try:
                analytics_db_path = refresh_analytics_database(
                    data_dir=config.data_dir,
                    output_dir=config.output_dir,
                    include_video=config.include_video,
                    logger=logger,
                    raw_df=raw_df,
                )
            except Exception as exc:
                logger.warning("Analytics database refresh failed but the run will continue: %s", exc)
                phase["status"] = "warning"
                phase["warning"] = str(exc)
            else:
                if analytics_db_path is not None and analytics_db_path.exists():
                    artifact_paths.append(analytics_db_path)
                phase["db_path"] = analytics_db_path
                phase["refreshed"] = bool(analytics_db_path is not None and analytics_db_path.exists())
    else:
        phase_recorder.skip("analytics_refresh", reason="analytics_refresh_disabled")


def write_postrun_reports(
    *,
    artifact_paths: list[Path],
    champion_gate: dict[str, object],
    config: Any,
    history_csv: Path,
    history_dir: Path,
    logger,
    manifest: dict[str, object],
    phase_recorder,
    result_rows: list[dict[str, object]],
    run_dir: Path,
    write_control_room_report,
    write_run_report,
) -> None:
    with phase_recorder.phase("run_report") as phase:
        report_path = write_run_report(
            run_dir=run_dir,
            history_dir=history_dir,
            manifest=manifest,
            results=result_rows,
            champion_gate=champion_gate,
            history_csv=history_csv,
        )
        artifact_paths.append(report_path)
        phase["report_path"] = report_path

    with phase_recorder.phase("control_room_report", top_n=5) as phase:
        try:
            control_room_json, control_room_md = write_control_room_report(config.output_dir, top_n=5)
        except Exception as exc:
            logger.warning("Control room report generation failed but the run will continue: %s", exc)
            phase["status"] = "warning"
            phase["warning"] = str(exc)
        else:
            artifact_paths.extend([control_room_json, control_room_md])
            phase["json_path"] = control_room_json
            phase["markdown_path"] = control_room_md


__all__ = [
    "build_run_manifest",
    "refresh_analytics_if_enabled",
    "run_history_exports",
    "write_history_rollups",
    "write_postrun_reports",
    "write_research_artifacts",
]
