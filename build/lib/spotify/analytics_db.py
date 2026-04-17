from __future__ import annotations

from pathlib import Path

import pandas as pd
from .run_artifacts import collect_run_analysis_rows
from .run_artifacts import collect_run_manifests
from .run_artifacts import collect_run_results
from .run_artifacts import rows_to_frame
from .run_artifacts import safe_read_csv as _safe_read_csv


def _replace_table(con, table_name: str, df: pd.DataFrame) -> None:
    if list(df.columns):
        relation_name = f"{table_name}_df"
        con.register(relation_name, df)
        con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM {relation_name}")
    else:
        con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT CAST(NULL AS VARCHAR) AS _empty WHERE 1=0")


def _collect_run_manifests(output_dir: Path) -> pd.DataFrame:
    return rows_to_frame(collect_run_manifests(output_dir))


def _collect_run_results(output_dir: Path) -> pd.DataFrame:
    return rows_to_frame(collect_run_results(output_dir))


def _collect_run_analysis_summaries(output_dir: Path, filename: str) -> pd.DataFrame:
    return rows_to_frame(collect_run_analysis_rows(output_dir, filename))


def refresh_analytics_database(
    *,
    data_dir: Path,
    output_dir: Path,
    include_video: bool,
    logger,
    raw_df: pd.DataFrame | None = None,
) -> Path | None:
    try:
        import duckdb
    except Exception as exc:
        logger.warning("DuckDB is unavailable; skipping analytics database refresh: %s", exc)
        return None

    analytics_dir = output_dir / "analytics"
    analytics_dir.mkdir(parents=True, exist_ok=True)
    db_path = analytics_dir / "spotify_analytics.duckdb"

    if raw_df is None:
        from .data import load_streaming_history

        raw_df = load_streaming_history(data_dir, include_video=include_video, logger=logger)
    experiment_history = _safe_read_csv(output_dir / "history" / "experiment_history.csv")
    backtest_history = _safe_read_csv(output_dir / "history" / "backtest_history.csv")
    benchmark_history = _safe_read_csv(output_dir / "history" / "benchmark_history.csv")
    optuna_history = _safe_read_csv(output_dir / "history" / "optuna_history.csv")
    run_manifests = _collect_run_manifests(output_dir)
    run_results = _collect_run_results(output_dir)
    robustness_summary = _collect_run_analysis_summaries(output_dir, "robustness_summary.json")
    policy_summary = _collect_run_analysis_summaries(output_dir, "policy_simulation_summary.json")
    moonshot_summary = _collect_run_analysis_summaries(output_dir, "moonshot_summary.json")

    try:
        con = duckdb.connect(str(db_path))
    except Exception as exc:
        logger.warning("DuckDB analytics refresh skipped because the database is busy or unavailable: %s", exc)
        return None
    try:
        _replace_table(con, "raw_streaming_history", raw_df)
        _replace_table(con, "experiment_history", experiment_history)
        _replace_table(con, "backtest_history", backtest_history)
        _replace_table(con, "benchmark_history", benchmark_history)
        _replace_table(con, "optuna_history", optuna_history)
        _replace_table(con, "run_manifests", run_manifests)
        _replace_table(con, "run_results", run_results)
        _replace_table(con, "robustness_summary", robustness_summary)
        _replace_table(con, "policy_summary", policy_summary)
        _replace_table(con, "moonshot_summary", moonshot_summary)

        if {"run_id", "timestamp"}.issubset(set(run_manifests.columns)) and {"run_id"}.issubset(set(run_results.columns)):
            con.execute(
                """
                CREATE OR REPLACE VIEW latest_run_results AS
                SELECT rr.*
                FROM run_results rr
                JOIN (
                  SELECT run_id
                  FROM run_manifests
                  ORDER BY timestamp DESC NULLS LAST, run_id DESC
                  LIMIT 1
                ) latest
                  ON rr.run_id = latest.run_id
                """
            )
        if {"profile", "model_name", "model_type", "val_top1", "test_top1"}.issubset(set(experiment_history.columns)):
            con.execute(
                """
                CREATE OR REPLACE VIEW best_models_by_profile AS
                SELECT
                  profile,
                  model_name,
                  model_type,
                  AVG(CAST(val_top1 AS DOUBLE)) AS mean_val_top1,
                  AVG(CAST(test_top1 AS DOUBLE)) AS mean_test_top1,
                  COUNT(*) AS runs
                FROM experiment_history
                GROUP BY profile, model_name, model_type
                ORDER BY profile, mean_val_top1 DESC
                """
            )
        if {"profile", "model_name", "top1", "model_type"}.issubset(set(backtest_history.columns)):
            con.execute(
                """
                CREATE OR REPLACE VIEW backtest_model_summary AS
                SELECT
                  profile,
                  model_name,
                  model_type,
                  AVG(CAST(top1 AS DOUBLE)) AS mean_backtest_top1,
                  STDDEV_SAMP(CAST(top1 AS DOUBLE)) AS std_backtest_top1,
                  COUNT(*) AS folds
                FROM backtest_history
                GROUP BY profile, model_name, model_type
                ORDER BY profile, mean_backtest_top1 DESC
                """
            )
        elif {"profile", "model_name", "top1"}.issubset(set(backtest_history.columns)):
            con.execute(
                """
                CREATE OR REPLACE VIEW backtest_model_summary AS
                SELECT
                  profile,
                  model_name,
                  AVG(CAST(top1 AS DOUBLE)) AS mean_backtest_top1,
                  STDDEV_SAMP(CAST(top1 AS DOUBLE)) AS std_backtest_top1,
                  COUNT(*) AS folds
                FROM backtest_history
                GROUP BY profile, model_name
                ORDER BY profile, mean_backtest_top1 DESC
                """
            )
        if {
            "run_id",
            "profile",
            "timestamp",
            "champion_gate.promoted",
            "champion_gate.metric_source",
            "champion_alias.model_name",
            "champion_alias.model_type",
        }.issubset(set(run_manifests.columns)):
            con.execute(
                """
                CREATE OR REPLACE VIEW champion_runs AS
                SELECT
                  run_id,
                  profile,
                  "timestamp",
                  CAST("champion_gate.promoted" AS BOOLEAN) AS promoted,
                  "champion_gate.metric_source" AS metric_source,
                  "champion_alias.model_name" AS alias_model_name,
                  "champion_alias.model_type" AS alias_model_type
                FROM run_manifests
                WHERE COALESCE(CAST("champion_gate.promoted" AS BOOLEAN), FALSE)
                """
            )
        if {"model_name", "max_top1_gap", "worst_segment", "worst_bucket", "run_id"}.issubset(set(robustness_summary.columns)):
            con.execute(
                """
                CREATE OR REPLACE VIEW robustness_model_summary AS
                SELECT
                  model_name,
                  AVG(CAST(max_top1_gap AS DOUBLE)) AS mean_max_top1_gap,
                  MAX(CAST(max_top1_gap AS DOUBLE)) AS worst_observed_gap,
                  COUNT(*) AS runs
                FROM robustness_summary
                GROUP BY model_name
                ORDER BY mean_max_top1_gap DESC
                """
            )
        if {"model_name", "model_type", "test_discounted_reward", "test_hit_at_k", "run_id"}.issubset(set(policy_summary.columns)):
            con.execute(
                """
                CREATE OR REPLACE VIEW policy_model_summary AS
                SELECT
                  model_name,
                  model_type,
                  AVG(CAST(test_discounted_reward AS DOUBLE)) AS mean_test_discounted_reward,
                  AVG(CAST(test_hit_at_k AS DOUBLE)) AS mean_test_hit_at_k,
                  COUNT(*) AS runs
                FROM policy_summary
                GROUP BY model_name, model_type
                ORDER BY mean_test_discounted_reward DESC
                """
            )
        if {"benchmark_id", "model_name", "val_top1_mean", "test_top1_mean"}.issubset(set(benchmark_history.columns)):
            con.execute(
                """
                CREATE OR REPLACE VIEW benchmark_model_summary AS
                SELECT
                  benchmark_id,
                  model_name,
                  AVG(CAST(val_top1_mean AS DOUBLE)) AS mean_val_top1,
                  AVG(CAST(test_top1_mean AS DOUBLE)) AS mean_test_top1,
                  COUNT(*) AS rows
                FROM benchmark_history
                GROUP BY benchmark_id, model_name
                ORDER BY benchmark_id, mean_val_top1 DESC
                """
            )
        if {
            "run_id",
            "multimodal_embedding_dim",
            "digital_twin_test_auc",
            "causal_test_auc_total",
            "stress_worst_skip_scenario",
            "stress_worst_skip_risk",
        }.issubset(set(moonshot_summary.columns)):
            con.execute(
                """
                CREATE OR REPLACE VIEW moonshot_run_summary AS
                SELECT
                  ms.run_id,
                  rm.profile,
                  CAST(ms.multimodal_embedding_dim AS INTEGER) AS multimodal_embedding_dim,
                  CAST(ms.multimodal_feature_count AS INTEGER) AS multimodal_feature_count,
                  CAST(ms.multimodal_retrieval_fusion_enabled AS BOOLEAN) AS multimodal_retrieval_fusion_enabled,
                  CAST(ms.digital_twin_test_auc AS DOUBLE) AS digital_twin_test_auc,
                  CAST(ms.causal_test_auc_total AS DOUBLE) AS causal_test_auc_total,
                  CAST(ms.journey_mean_horizon AS DOUBLE) AS journey_mean_horizon,
                  CAST(ms.safe_policy_bucket_count AS INTEGER) AS safe_policy_bucket_count,
                  ms.stress_worst_skip_scenario AS stress_worst_skip_scenario,
                  CAST(ms.stress_worst_skip_risk AS DOUBLE) AS stress_worst_skip_risk
                FROM moonshot_summary ms
                LEFT JOIN run_manifests rm
                  ON ms.run_id = rm.run_id
                ORDER BY digital_twin_test_auc DESC NULLS LAST, causal_test_auc_total DESC NULLS LAST
                """
            )
    except Exception as exc:
        logger.warning("DuckDB analytics refresh did not complete: %s", exc)
        return None
    finally:
        con.close()

    logger.info("Analytics DuckDB refreshed: %s", db_path)
    return db_path
