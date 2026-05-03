from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
from .analytics_warehouse import build_analytics_warehouse_bundle
from .analytics_warehouse import write_analytics_warehouse


def _replace_table(con, table_name: str, df: pd.DataFrame) -> None:
    if list(df.columns):
        relation_name = f"{table_name}_df"
        con.register(relation_name, df)
        con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM {relation_name}")
    else:
        con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT CAST(NULL AS VARCHAR) AS _empty WHERE 1=0")


def _duckdb_retry_policy() -> tuple[int, float]:
    try:
        retries = int(str(os.getenv("SPOTIFY_ANALYTICS_DB_RETRIES", "3")).strip())
    except Exception:
        retries = 3
    retries = max(0, retries)
    try:
        sleep_seconds = float(str(os.getenv("SPOTIFY_ANALYTICS_DB_RETRY_SLEEP_S", "0.35")).strip())
    except Exception:
        sleep_seconds = 0.35
    return retries, max(0.0, sleep_seconds)


def _connect_duckdb_with_retries(*, duckdb, target_path: Path, retries: int, sleep_seconds: float):
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return duckdb.connect(str(target_path)), attempt
        except Exception as exc:
            last_error = exc
            if attempt >= retries:
                break
            time.sleep(sleep_seconds)
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Unable to connect DuckDB database at {target_path}")


def refresh_analytics_database(
    *,
    data_dir: Path,
    output_dir: Path,
    include_video: bool,
    logger,
    raw_df: pd.DataFrame | None = None,
) -> Path | None:
    if raw_df is None:
        from .data import load_streaming_history

        raw_df = load_streaming_history(data_dir, include_video=include_video, logger=logger)

    warehouse_bundle = build_analytics_warehouse_bundle(
        data_dir=data_dir,
        output_dir=output_dir,
        include_video=include_video,
        logger=logger,
        raw_df=raw_df,
    )
    write_analytics_warehouse(warehouse_bundle, logger=logger)

    try:
        import duckdb
    except Exception as exc:
        logger.warning("DuckDB is unavailable; skipping analytics database refresh: %s", exc)
        return None

    analytics_dir = output_dir / "analytics"
    analytics_dir.mkdir(parents=True, exist_ok=True)
    db_path = analytics_dir / "spotify_analytics.duckdb"
    fallback_db_path = analytics_dir / "spotify_analytics.refresh.duckdb"

    retries, sleep_seconds = _duckdb_retry_policy()
    connected_path = db_path
    fallback_used = False
    retry_attempts = 0
    try:
        con, retry_attempts = _connect_duckdb_with_retries(
            duckdb=duckdb,
            target_path=db_path,
            retries=retries,
            sleep_seconds=sleep_seconds,
        )
    except Exception as exc:
        logger.warning(
            "Primary DuckDB analytics database is busy or unavailable after %d attempt(s): %s",
            retries + 1,
            exc,
        )
        try:
            con, _ = _connect_duckdb_with_retries(
                duckdb=duckdb,
                target_path=fallback_db_path,
                retries=0,
                sleep_seconds=sleep_seconds,
            )
            connected_path = fallback_db_path
            fallback_used = True
        except Exception as fallback_exc:
            logger.warning("DuckDB analytics refresh skipped because fallback database creation failed: %s", fallback_exc)
            return None
    try:
        for table_name, table_df in warehouse_bundle.tables().items():
            _replace_table(con, table_name, table_df)

        run_manifests = warehouse_bundle.bronze["run_manifests"]
        run_results = warehouse_bundle.bronze["run_results"]
        experiment_history = warehouse_bundle.bronze["experiment_history"]
        backtest_history = warehouse_bundle.bronze["backtest_history"]
        benchmark_history = warehouse_bundle.bronze["benchmark_history"]
        robustness_summary = warehouse_bundle.bronze["robustness_summary"]
        policy_summary = warehouse_bundle.bronze["policy_summary"]
        moonshot_summary = warehouse_bundle.bronze["moonshot_summary"]

        if {"run_id", "run_timestamp"}.issubset(set(run_manifests.columns)) and {"run_id"}.issubset(set(run_results.columns)):
            con.execute(
                """
                CREATE OR REPLACE VIEW latest_run_results AS
                SELECT rr.*
                FROM run_results rr
                JOIN (
                  SELECT run_id
                  FROM run_manifests
                  ORDER BY run_timestamp DESC NULLS LAST, run_id DESC
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
            "run_timestamp",
            "champion_gate_promoted",
            "champion_gate_metric_source",
            "champion_alias_model_name",
            "champion_alias_model_type",
        }.issubset(set(run_manifests.columns)):
            con.execute(
                """
                CREATE OR REPLACE VIEW champion_runs AS
                SELECT
                  run_id,
                  profile,
                  run_timestamp,
                  CAST(champion_gate_promoted AS BOOLEAN) AS promoted,
                  champion_gate_metric_source AS metric_source,
                  champion_alias_model_name AS alias_model_name,
                  champion_alias_model_type AS alias_model_type
                FROM run_manifests
                WHERE COALESCE(CAST(champion_gate_promoted AS BOOLEAN), FALSE)
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
        if {"latest_run_id", "ops_health_status", "operating_rhythm_status"}.issubset(
            set(warehouse_bundle.gold["mart_ops_overview"].columns)
        ):
            con.execute(
                """
                CREATE OR REPLACE VIEW latest_ops_overview AS
                SELECT *
                FROM mart_ops_overview
                """
            )
        if {"artist_name", "priority_now_count", "max_opportunity_score"}.issubset(
            set(warehouse_bundle.gold["mart_creator_opportunities"].columns)
        ):
            con.execute(
                """
                CREATE OR REPLACE VIEW creator_priority_now AS
                SELECT *
                FROM mart_creator_opportunities
                WHERE COALESCE(priority_now_count, 0) > 0
                ORDER BY max_opportunity_score DESC NULLS LAST, priority_now_count DESC NULLS LAST
                """
            )
    except Exception as exc:
        logger.warning("DuckDB analytics refresh did not complete: %s", exc)
        return None
    finally:
        con.close()

    if fallback_used:
        logger.warning(
            "Analytics DuckDB refreshed to fallback path because the primary database is locked: %s",
            connected_path,
        )
    else:
        logger.info(
            "Analytics DuckDB refreshed: %s (attempt=%d/%d)",
            connected_path,
            retry_attempts + 1,
            retries + 1,
        )
    return connected_path
