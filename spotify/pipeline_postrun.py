from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import os
from typing import Any

from .artifact_cleanup import prune_mlflow_artifacts, prune_old_auxiliary_artifacts, prune_run_artifacts
from .champion_alias import best_serveable_model, write_champion_alias
from .pipeline_helpers import _load_current_risk_metrics, _write_json_artifact


@dataclass
class PipelinePostRunContext:
    artifact_paths: list[Path]
    backtest_rows: list[dict[str, object]]
    cache_info_payload: dict[str, object]
    config: Any
    history_dir: Path
    logger: Any
    manifest_path: Path
    optuna_rows: list[dict[str, object]]
    phase_recorder: Any
    prepared: Any
    raw_df: Any
    result_rows: list[dict[str, object]]
    run_classical_models: bool
    run_dir: Path
    run_id: str


@dataclass
class PipelinePostRunDeps:
    append_backtest_history: Any
    append_experiment_history: Any
    append_optuna_history: Any
    evaluate_champion_gate: Any
    plot_backtest_history: Any
    plot_history_best_runs: Any
    plot_optuna_best_runs: Any
    plot_run_leaderboard: Any
    refresh_analytics_database: Any
    write_ablation_summary: Any
    write_benchmark_protocol: Any
    write_control_room_report: Any
    write_experiment_registry: Any
    write_run_report: Any
    write_significance_summary: Any


@dataclass
class PipelinePostRunResult:
    manifest_payload: dict[str, object]
    strict_gate_error: str | None


def run_pipeline_postrun(*, context: PipelinePostRunContext, deps: PipelinePostRunDeps) -> PipelinePostRunResult:
    artifact_paths = context.artifact_paths
    backtest_rows = context.backtest_rows
    cache_info_payload = context.cache_info_payload
    config = context.config
    history_dir = context.history_dir
    logger = context.logger
    manifest_path = context.manifest_path
    optuna_rows = context.optuna_rows
    phase_recorder = context.phase_recorder
    prepared = context.prepared
    raw_df = context.raw_df
    result_rows = context.result_rows
    run_classical_models = context.run_classical_models
    run_dir = context.run_dir
    run_id = context.run_id

    champion_alias_payload: dict[str, object] = {
        "updated": False,
        "alias_file": "",
        "run_id": "",
        "run_dir": "",
        "model_name": "",
        "reason": "gate_not_evaluated",
    }
    mlflow_artifact_cleanup_summary: dict[str, object] = {
        "enabled": False,
        "artifact_mode": "off",
        "max_artifact_mb": 0.0,
        "status": "not_run",
        "artifact_dir_count": 0,
        "artifact_dirs": [],
        "deleted_file_count": 0,
        "deleted_files": [],
        "freed_bytes": 0,
    }

    with phase_recorder.phase("history_exports") as phase:
        leaderboard_path = deps.plot_run_leaderboard(result_rows, run_dir)
        if leaderboard_path is not None:
            artifact_paths.append(leaderboard_path)

        history_csv = deps.append_experiment_history(
            history_csv=history_dir / "experiment_history.csv",
            run_id=run_id,
            profile=config.profile,
            run_name=config.run_name,
            results=result_rows,
            data_records=len(prepared.df),
        )
        history_plot = deps.plot_history_best_runs(history_csv, history_dir)
        artifact_paths.append(history_csv)
        if history_plot is not None:
            artifact_paths.append(history_plot)
        phase["leaderboard_plot_written"] = bool(leaderboard_path is not None)
        phase["experiment_history_rows"] = int(len(result_rows))

    champion_gate_threshold_raw = os.getenv("SPOTIFY_CHAMPION_GATE_MAX_REGRESSION", "0.005").strip()
    try:
        champion_gate_threshold = max(0.0, float(champion_gate_threshold_raw))
    except Exception:
        champion_gate_threshold = 0.005
    champion_gate_metric = os.getenv("SPOTIFY_CHAMPION_GATE_METRIC", "backtest_top1").strip().lower()
    champion_gate_match_profile_raw = os.getenv("SPOTIFY_CHAMPION_GATE_MATCH_PROFILE", "1").strip().lower()
    champion_gate_match_profile = champion_gate_match_profile_raw in ("1", "true", "yes", "on")
    champion_gate_significance_raw = os.getenv("SPOTIFY_CHAMPION_GATE_SIGNIFICANCE", "0").strip().lower()
    champion_gate_require_significance = champion_gate_significance_raw in ("1", "true", "yes", "on")
    champion_gate_significance_z_raw = os.getenv("SPOTIFY_CHAMPION_GATE_SIGNIFICANCE_Z", "1.96").strip()
    try:
        champion_gate_significance_z = max(0.0, float(champion_gate_significance_z_raw))
    except Exception:
        champion_gate_significance_z = 1.96
    current_risk_metrics = _load_current_risk_metrics(
        run_dir,
        result_rows,
        val_y=prepared.y_val,
        test_y=prepared.y_test,
        conformal_alpha=config.conformal_alpha,
    )
    gate_max_selective_risk_raw = os.getenv("SPOTIFY_CHAMPION_GATE_MAX_SELECTIVE_RISK", "").strip()
    gate_max_abstention_raw = os.getenv("SPOTIFY_CHAMPION_GATE_MAX_ABSTENTION_RATE", "").strip()
    try:
        gate_max_selective_risk = float(gate_max_selective_risk_raw) if gate_max_selective_risk_raw else None
    except Exception:
        gate_max_selective_risk = None
    try:
        gate_max_abstention_rate = float(gate_max_abstention_raw) if gate_max_abstention_raw else None
    except Exception:
        gate_max_abstention_rate = None

    with phase_recorder.phase(
        "champion_gate_and_alias",
        metric_source=champion_gate_metric,
        require_profile_match=champion_gate_match_profile,
        require_significant_lift=champion_gate_require_significance,
    ) as phase:
        champion_gate = deps.evaluate_champion_gate(
            history_csv=history_csv,
            current_run_id=run_id,
            current_results=result_rows,
            regression_threshold=champion_gate_threshold,
            backtest_history_csv=(history_dir / "backtest_history.csv"),
            current_backtest_rows=backtest_rows,
            metric_source=champion_gate_metric,
            current_profile=config.profile,
            require_profile_match=champion_gate_match_profile,
            require_significant_lift=champion_gate_require_significance,
            significance_z=champion_gate_significance_z,
            current_risk_metrics=current_risk_metrics,
            max_selective_risk=gate_max_selective_risk,
            max_abstention_rate=gate_max_abstention_rate,
        )
        _write_json_artifact(run_dir / "champion_gate.json", champion_gate, artifact_paths)
        logger.info(
            "Champion gate: source=%s promoted=%s regression=%.6f threshold=%.6f",
            str(champion_gate.get("metric_source", champion_gate_metric)),
            bool(champion_gate.get("promoted", False)),
            float(champion_gate.get("regression", 0.0)),
            float(champion_gate.get("threshold", champion_gate_threshold)),
        )
        strict_gate_raw = os.getenv("SPOTIFY_CHAMPION_GATE_STRICT", "0").strip().lower()
        strict_gate = strict_gate_raw in ("1", "true", "yes", "on")
        strict_gate_error: str | None = None
        if strict_gate and not bool(champion_gate.get("promoted", False)):
            strict_gate_error = (
                "Champion gate failed in strict mode: "
                f"regression={champion_gate.get('regression')} threshold={champion_gate.get('threshold')}"
            )

        champion_model: tuple[str, str] | None = None
        if bool(champion_gate.get("promoted", False)):
            champion_model = best_serveable_model(result_rows, run_dir=run_dir)
            if not champion_model:
                champion_alias_payload["reason"] = "no_serveable_models_in_promoted_run"
                logger.info("Skipping champion alias update: promoted run has no serveable models.")
            else:
                champion_model_name, champion_model_type = champion_model
                alias_file = write_champion_alias(
                    output_dir=config.output_dir,
                    run_id=run_id,
                    run_dir=run_dir,
                    model_name=champion_model_name,
                    model_type=champion_model_type,
                )
                champion_alias_payload = {
                    "updated": True,
                    "alias_file": str(alias_file),
                    "run_id": run_id,
                    "run_dir": str(run_dir),
                    "model_name": champion_model_name,
                    "model_type": champion_model_type,
                    "reason": "promoted",
                }
                artifact_paths.append(alias_file)
                logger.info(
                    "Champion alias updated: %s -> run_id=%s model=%s type=%s",
                    alias_file,
                    run_id,
                    champion_model_name,
                    champion_model_type,
                )
        else:
            champion_alias_payload["reason"] = "gate_not_promoted"
        phase["promoted"] = bool(champion_gate.get("promoted", False))
        phase["champion_alias_updated"] = bool(champion_alias_payload.get("updated", False))
        phase["strict_gate_enabled"] = bool(strict_gate)

    artifact_cleanup_mode = os.getenv("SPOTIFY_ARTIFACT_CLEANUP", "light")
    artifact_cleanup_min_mb_raw = os.getenv("SPOTIFY_ARTIFACT_CLEANUP_MIN_MB", "100").strip()
    try:
        artifact_cleanup_min_mb = max(0.0, float(artifact_cleanup_min_mb_raw))
    except Exception:
        artifact_cleanup_min_mb = 100.0
    with phase_recorder.phase("artifact_cleanup_and_retention") as phase:
        cleanup_summary = prune_run_artifacts(
            run_dir=run_dir,
            result_rows=result_rows,
            selected_model=champion_model,
            logger=logger,
            cleanup_mode=artifact_cleanup_mode,
            min_size_mb=artifact_cleanup_min_mb,
        )
        _write_json_artifact(run_dir / "artifact_cleanup.json", cleanup_summary, artifact_paths)

        prune_old_prediction_bundles_raw = os.getenv("SPOTIFY_PRUNE_OLD_PREDICTION_BUNDLES", "1").strip().lower()
        prune_old_prediction_bundles = prune_old_prediction_bundles_raw in ("1", "true", "yes", "on")
        prune_old_run_dbs_raw = os.getenv("SPOTIFY_PRUNE_OLD_RUN_DATABASES", "1").strip().lower()
        prune_old_run_dbs = prune_old_run_dbs_raw in ("1", "true", "yes", "on")
        keep_full_runs_raw = os.getenv("SPOTIFY_KEEP_FULL_RUNS", "2").strip()
        try:
            keep_full_runs = max(0, int(keep_full_runs_raw))
        except Exception:
            keep_full_runs = 2
        retention_summary = prune_old_auxiliary_artifacts(
            output_dir=config.output_dir,
            current_run_dir=run_dir,
            logger=logger,
            keep_last_full_runs=keep_full_runs,
            prune_prediction_bundles=prune_old_prediction_bundles,
            prune_run_databases=prune_old_run_dbs,
        )
        _write_json_artifact(run_dir / "artifact_retention.json", retention_summary, artifact_paths)
        mlflow_artifact_cleanup_summary = prune_mlflow_artifacts(
            output_dir=config.output_dir,
            logger=logger,
        )
        _write_json_artifact(
            run_dir / "mlflow_artifact_cleanup.json",
            mlflow_artifact_cleanup_summary,
            artifact_paths,
        )
        phase["cleanup_mode"] = artifact_cleanup_mode
        phase["cleanup_deleted_files"] = int(cleanup_summary.get("deleted_file_count", 0) or 0)
        phase["retention_deleted_prediction_bundles"] = int(
            retention_summary.get("deleted_prediction_bundle_count", 0) or 0
        )
        phase["mlflow_deleted_files"] = int(mlflow_artifact_cleanup_summary.get("deleted_file_count", 0) or 0)

    with phase_recorder.phase("research_artifacts") as phase:
        artifact_paths.extend(
            deps.write_benchmark_protocol(
                output_dir=run_dir,
                run_id=run_id,
                profile=config.profile,
                data=prepared,
                cache_info=cache_info_payload,
                config=config,
            )
        )
        artifact_paths.append(
            deps.write_experiment_registry(
                output_dir=run_dir,
                run_id=run_id,
                profile=config.profile,
                results=result_rows,
                backtest_rows=backtest_rows,
                config=config,
            )
        )
        artifact_paths.extend(deps.write_ablation_summary(output_dir=run_dir / "analysis", results=result_rows))
        artifact_paths.extend(
            deps.write_significance_summary(
                output_dir=run_dir / "analysis",
                results=result_rows,
                backtest_rows=backtest_rows,
            )
        )
        phase["result_count"] = int(len(result_rows))
        phase["backtest_row_count"] = int(len(backtest_rows))

    with phase_recorder.phase("history_rollups") as phase:
        if optuna_rows:
            optuna_history_csv = deps.append_optuna_history(
                history_csv=history_dir / "optuna_history.csv",
                run_id=run_id,
                profile=config.profile,
                run_name=config.run_name,
                results=optuna_rows,
            )
            optuna_history_plot = deps.plot_optuna_best_runs(optuna_history_csv, history_dir)
            artifact_paths.append(optuna_history_csv)
            if optuna_history_plot is not None:
                artifact_paths.append(optuna_history_plot)

        if backtest_rows:
            backtest_history_csv = deps.append_backtest_history(
                history_csv=history_dir / "backtest_history.csv",
                run_id=run_id,
                profile=config.profile,
                run_name=config.run_name,
                rows=backtest_rows,
            )
            backtest_history_plot = deps.plot_backtest_history(backtest_history_csv, history_dir)
            artifact_paths.append(backtest_history_csv)
            if backtest_history_plot is not None:
                artifact_paths.append(backtest_history_plot)
        phase["optuna_row_count"] = int(len(optuna_rows))
        phase["backtest_row_count"] = int(len(backtest_rows))

    manifest = {
        "run_id": run_id,
        "run_name": config.run_name,
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
    _write_json_artifact(manifest_path, manifest, artifact_paths)
    _write_json_artifact(run_dir / "run_results.json", result_rows, artifact_paths)

    refresh_analytics_raw = os.getenv("SPOTIFY_REFRESH_ANALYTICS_DB", "1").strip().lower()
    refresh_analytics = refresh_analytics_raw not in ("0", "false", "no", "off")
    if refresh_analytics:
        with phase_recorder.phase("analytics_refresh") as phase:
            try:
                analytics_db_path = deps.refresh_analytics_database(
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

    with phase_recorder.phase("run_report") as phase:
        report_path = deps.write_run_report(
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
            control_room_json, control_room_md = deps.write_control_room_report(config.output_dir, top_n=5)
        except Exception as exc:
            logger.warning("Control room report generation failed but the run will continue: %s", exc)
            phase["status"] = "warning"
            phase["warning"] = str(exc)
        else:
            artifact_paths.extend([control_room_json, control_room_md])
            phase["json_path"] = control_room_json
            phase["markdown_path"] = control_room_md

    return PipelinePostRunResult(
        manifest_payload=manifest,
        strict_gate_error=strict_gate_error,
    )
