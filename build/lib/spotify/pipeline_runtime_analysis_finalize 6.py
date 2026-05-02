from __future__ import annotations

import numpy as np

from .pipeline_postrun import PipelinePostRunContext, PipelinePostRunDeps, run_pipeline_postrun
from .pipeline_runtime_analysis_types import (
    PipelineAnalysisContext,
    PipelineAnalysisDeps,
    PipelineAnalysisOutputs,
)


def run_analysis_postrun(*, context: PipelineAnalysisContext, deps: PipelineAnalysisDeps) -> PipelineAnalysisOutputs:
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


def log_analysis_leaderboard(*, context: PipelineAnalysisContext) -> None:
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


__all__ = ["log_analysis_leaderboard", "run_analysis_postrun"]
