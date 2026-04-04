from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .pipeline_postrun_governance import run_artifact_cleanup_and_retention, run_champion_gate_and_alias
from .pipeline_postrun_reporting import run_history_exports, write_history_rollups, write_research_artifacts


@dataclass
class PipelinePostRunStageOutputs:
    history_csv: Path
    champion_outcome: object
    cleanup_outcome: object


def run_postrun_stages(*, context, deps) -> PipelinePostRunStageOutputs:
    artifact_paths = context.artifact_paths
    config = context.config
    history_dir = context.history_dir
    logger = context.logger
    phase_recorder = context.phase_recorder
    prepared = context.prepared
    result_rows = context.result_rows
    run_dir = context.run_dir
    run_id = context.run_id

    history_csv = run_history_exports(
        append_experiment_history=deps.append_experiment_history,
        artifact_paths=artifact_paths,
        data_records=len(prepared.df),
        history_dir=history_dir,
        phase_recorder=phase_recorder,
        plot_history_best_runs=deps.plot_history_best_runs,
        plot_run_leaderboard=deps.plot_run_leaderboard,
        profile=config.profile,
        result_rows=result_rows,
        run_dir=run_dir,
        run_id=run_id,
        run_name=config.run_name,
    )

    champion_outcome = run_champion_gate_and_alias(
        artifact_paths=artifact_paths,
        backtest_rows=context.backtest_rows,
        config=config,
        evaluate_champion_gate=deps.evaluate_champion_gate,
        history_csv=history_csv,
        history_dir=history_dir,
        logger=logger,
        phase_recorder=phase_recorder,
        prepared=prepared,
        result_rows=result_rows,
        run_dir=run_dir,
        run_id=run_id,
    )

    cleanup_outcome = run_artifact_cleanup_and_retention(
        artifact_paths=artifact_paths,
        config=config,
        logger=logger,
        phase_recorder=phase_recorder,
        result_rows=result_rows,
        run_dir=run_dir,
        selected_model=champion_outcome.champion_model,
    )

    write_research_artifacts(
        artifact_paths=artifact_paths,
        backtest_rows=context.backtest_rows,
        cache_info_payload=context.cache_info_payload,
        config=config,
        data=prepared,
        phase_recorder=phase_recorder,
        result_rows=result_rows,
        run_dir=run_dir,
        run_id=run_id,
        write_ablation_summary=deps.write_ablation_summary,
        write_benchmark_protocol=deps.write_benchmark_protocol,
        write_experiment_registry=deps.write_experiment_registry,
        write_significance_summary=deps.write_significance_summary,
    )

    write_history_rollups(
        append_backtest_history=deps.append_backtest_history,
        append_optuna_history=deps.append_optuna_history,
        artifact_paths=artifact_paths,
        backtest_rows=context.backtest_rows,
        history_dir=history_dir,
        optuna_rows=context.optuna_rows,
        phase_recorder=phase_recorder,
        plot_backtest_history=deps.plot_backtest_history,
        plot_optuna_best_runs=deps.plot_optuna_best_runs,
        profile=config.profile,
        run_id=run_id,
        run_name=config.run_name,
    )

    return PipelinePostRunStageOutputs(
        history_csv=history_csv,
        champion_outcome=champion_outcome,
        cleanup_outcome=cleanup_outcome,
    )


__all__ = ["PipelinePostRunStageOutputs", "run_postrun_stages"]
