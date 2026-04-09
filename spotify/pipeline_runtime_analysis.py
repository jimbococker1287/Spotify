from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .pipeline_postrun import PipelinePostRunContext, PipelinePostRunDeps, run_pipeline_postrun


@dataclass
class PipelineAnalysisContext:
    artifact_paths: list[Path]
    artist_labels: list[str]
    backtest_rows: list[dict[str, object]]
    cache_info_payload: dict[str, object]
    classical_feature_bundle: Any
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
class PipelineAnalysisDeps:
    append_backtest_history: Any
    append_experiment_history: Any
    append_optuna_history: Any
    build_probability_ensemble: Any
    evaluate_champion_gate: Any
    plot_backtest_history: Any
    plot_history_best_runs: Any
    plot_optuna_best_runs: Any
    plot_run_leaderboard: Any
    refresh_analytics_database: Any
    run_drift_diagnostics: Any
    run_extended_evaluation: Any
    run_friction_proxy_analysis: Any
    run_moonshot_lab: Any
    run_policy_simulation: Any
    run_robustness_slice_evaluation: Any
    write_ablation_summary: Any
    write_benchmark_protocol: Any
    write_control_room_report: Any
    write_experiment_registry: Any
    write_run_report: Any
    write_significance_summary: Any


@dataclass
class PipelineAnalysisOutputs:
    manifest_payload: dict[str, object]
    strict_gate_error: str | None


def _run_analysis_artifacts(*, context: PipelineAnalysisContext, deps: PipelineAnalysisDeps) -> None:
    ensemble_result = deps.build_probability_ensemble(
        data=context.prepared,
        results=context.result_rows,
        sequence_length=context.config.sequence_length,
        run_dir=context.run_dir,
        logger=context.logger,
    )
    if ensemble_result is not None:
        context.result_rows.append(dict(ensemble_result.row))
        context.artifact_paths.extend(ensemble_result.artifact_paths)

    with context.phase_recorder.phase(
        "extended_evaluation",
        enable_conformal=context.config.enable_conformal,
        conformal_alpha=context.config.conformal_alpha,
    ) as phase:
        extended_eval_artifacts = deps.run_extended_evaluation(
            data=context.prepared,
            results=context.result_rows,
            sequence_length=context.config.sequence_length,
            run_dir=context.run_dir,
            random_seed=context.config.random_seed,
            max_train_samples=context.config.classical_max_train_samples,
            enable_conformal=context.config.enable_conformal,
            conformal_alpha=context.config.conformal_alpha,
            logger=context.logger,
            feature_bundle=context.classical_feature_bundle,
        )
        context.artifact_paths.extend(extended_eval_artifacts)
        phase["artifact_count"] = int(len(extended_eval_artifacts))

    with context.phase_recorder.phase("drift_diagnostics") as phase:
        drift_artifacts = deps.run_drift_diagnostics(
            data=context.prepared,
            sequence_length=context.config.sequence_length,
            output_dir=context.run_dir / "analysis",
            logger=context.logger,
        )
        context.artifact_paths.extend(drift_artifacts)
        phase["artifact_count"] = int(len(drift_artifacts))

    with context.phase_recorder.phase("robustness_slice_evaluation") as phase:
        robustness_artifacts = deps.run_robustness_slice_evaluation(
            data=context.prepared,
            results=context.result_rows,
            sequence_length=context.config.sequence_length,
            run_dir=context.run_dir,
            logger=context.logger,
        )
        context.artifact_paths.extend(robustness_artifacts)
        phase["artifact_count"] = int(len(robustness_artifacts))

    with context.phase_recorder.phase("policy_simulation") as phase:
        policy_artifacts = deps.run_policy_simulation(
            data=context.prepared,
            results=context.result_rows,
            run_dir=context.run_dir,
            logger=context.logger,
        )
        context.artifact_paths.extend(policy_artifacts)
        phase["artifact_count"] = int(len(policy_artifacts))

    if context.config.enable_friction_analysis:
        with context.phase_recorder.phase("friction_proxy_analysis") as phase:
            friction_artifacts = deps.run_friction_proxy_analysis(
                data=context.prepared,
                output_dir=context.run_dir / "analysis",
                logger=context.logger,
            )
            context.artifact_paths.extend(friction_artifacts)
            phase["artifact_count"] = int(len(friction_artifacts))
    else:
        context.phase_recorder.skip("friction_proxy_analysis", reason="friction_analysis_disabled")
        context.logger.info("Skipping friction proxy analysis for this run.")

    if context.config.enable_moonshot_lab:
        with context.phase_recorder.phase("moonshot_lab") as phase:
            moonshot_artifacts = deps.run_moonshot_lab(
                data=context.prepared,
                results=context.result_rows,
                run_dir=context.run_dir,
                sequence_length=context.config.sequence_length,
                artist_labels=context.artist_labels,
                random_seed=context.config.random_seed,
                logger=context.logger,
            )
            context.artifact_paths.extend(moonshot_artifacts)
            phase["artifact_count"] = int(len(moonshot_artifacts))
    else:
        context.phase_recorder.skip("moonshot_lab", reason="moonshot_lab_disabled")
        context.logger.info("Skipping moonshot lab for this run.")


def _run_postrun(*, context: PipelineAnalysisContext, deps: PipelineAnalysisDeps) -> PipelineAnalysisOutputs:
    postrun_result = run_pipeline_postrun(
        context=PipelinePostRunContext(
            artifact_paths=context.artifact_paths,
            backtest_rows=context.backtest_rows,
            cache_info_payload=context.cache_info_payload,
            config=context.config,
            history_dir=context.history_dir,
            logger=context.logger,
            manifest_path=context.manifest_path,
            optuna_rows=context.optuna_rows,
            phase_recorder=context.phase_recorder,
            prepared=context.prepared,
            raw_df=context.raw_df,
            result_rows=context.result_rows,
            run_classical_models=context.run_classical_models,
            run_dir=context.run_dir,
            run_id=context.run_id,
        ),
        deps=PipelinePostRunDeps(
            append_backtest_history=deps.append_backtest_history,
            append_experiment_history=deps.append_experiment_history,
            append_optuna_history=deps.append_optuna_history,
            evaluate_champion_gate=deps.evaluate_champion_gate,
            plot_backtest_history=deps.plot_backtest_history,
            plot_history_best_runs=deps.plot_history_best_runs,
            plot_optuna_best_runs=deps.plot_optuna_best_runs,
            plot_run_leaderboard=deps.plot_run_leaderboard,
            refresh_analytics_database=deps.refresh_analytics_database,
            write_ablation_summary=deps.write_ablation_summary,
            write_benchmark_protocol=deps.write_benchmark_protocol,
            write_control_room_report=deps.write_control_room_report,
            write_experiment_registry=deps.write_experiment_registry,
            write_run_report=deps.write_run_report,
            write_significance_summary=deps.write_significance_summary,
        ),
    )
    return PipelineAnalysisOutputs(
        manifest_payload=postrun_result.manifest_payload,
        strict_gate_error=postrun_result.strict_gate_error,
    )


def _log_leaderboard(*, context: PipelineAnalysisContext) -> None:
    with context.phase_recorder.phase("leaderboard_logging") as phase:
        result_rows_sorted = sorted(
            context.result_rows,
            key=lambda row: float(row.get("val_top1", float("-inf"))),
            reverse=True,
        )
        context.logger.info("Run Leaderboard (sorted by val top-1):")
        for row in result_rows_sorted:
            fit_seconds = row.get("fit_seconds")
            fit_display = "n/a"
            if fit_seconds not in ("", None):
                fit_display = f"{float(fit_seconds):.2f}"
            context.logger.info(
                "%s [%s]: val_top1=%.4f test_top1=%.4f fit_s=%s",
                row.get("model_name"),
                row.get("model_type"),
                float(row.get("val_top1", np.nan)),
                float(row.get("test_top1", np.nan)),
                fit_display,
            )
        phase["result_count"] = int(len(result_rows_sorted))


def run_analysis_and_postrun(*, context: PipelineAnalysisContext, deps: PipelineAnalysisDeps) -> PipelineAnalysisOutputs:
    _run_analysis_artifacts(context=context, deps=deps)
    outputs = _run_postrun(context=context, deps=deps)
    _log_leaderboard(context=context)
    return outputs


__all__ = [
    "PipelineAnalysisContext",
    "PipelineAnalysisDeps",
    "PipelineAnalysisOutputs",
    "run_analysis_and_postrun",
]
