from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd
from .analytics_warehouse import build_analytics_warehouse_bundle
from .analytics_warehouse import warehouse_manifest_frames
from .analytics_warehouse import write_analytics_warehouse


def _replace_table(con, table_name: str, df: pd.DataFrame) -> None:
    if list(df.columns):
        relation_name = f"{table_name}_df"
        con.register(relation_name, df)
        con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM {relation_name}")
    else:
        con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT CAST(NULL AS VARCHAR) AS _empty WHERE 1=0")


def _quote_identifier(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def _replace_table_from_parquet(con, table_name: str, parquet_path: Path) -> None:
    con.execute(
        f"CREATE OR REPLACE TABLE {_quote_identifier(table_name)} AS SELECT * FROM read_parquet(?)",
        [str(parquet_path)],
    )


def _asset_columns_by_object(asset_column_manifest: pd.DataFrame) -> dict[str, set[str]]:
    if asset_column_manifest.empty:
        return {}
    ordered = asset_column_manifest.sort_values(["object_name", "column_position"])
    lookup: dict[str, set[str]] = {}
    for object_name, rows in ordered.groupby("object_name", dropna=False):
        lookup[str(object_name)] = {str(value) for value in rows["column_name"].tolist()}
    return lookup


def _duckdb_logical_type(value: object) -> str:
    normalized = str(value or "").strip().upper()
    if any(token in normalized for token in ("BOOL",)):
        return "boolean"
    if any(
        token in normalized
        for token in ("TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT", "UTINYINT", "USMALLINT", "UINTEGER", "UBIGINT")
    ):
        return "integer"
    if any(token in normalized for token in ("DECIMAL", "DOUBLE", "FLOAT", "REAL")):
        return "float"
    if "TIMESTAMP" in normalized:
        return "timestamp"
    return "string"


WAREHOUSE_CONSISTENCY_TARGETS: tuple[dict[str, object], ...] = (
    {"expected_asset_name": "raw_streaming_history", "object_name": "raw_streaming_history", "object_kind": "table"},
    {"expected_asset_name": "run_manifests", "object_name": "run_manifests", "object_kind": "table"},
    {"expected_asset_name": "run_results", "object_name": "run_results", "object_kind": "table"},
    {"expected_asset_name": "control_room_snapshot", "object_name": "control_room_snapshot", "object_kind": "table"},
    {"expected_asset_name": "listener_daily_activity", "object_name": "listener_daily_activity", "object_kind": "table"},
    {"expected_asset_name": "model_run_summary", "object_name": "model_run_summary", "object_kind": "table"},
    {"expected_asset_name": "creator_market_scene_summary", "object_name": "creator_market_scene_summary", "object_kind": "table"},
    {"expected_asset_name": "research_platform_status_summary", "object_name": "research_platform_status_summary", "object_kind": "table"},
    {"expected_asset_name": "mart_run_quality", "object_name": "mart_run_quality", "object_kind": "table"},
    {"expected_asset_name": "mart_model_registry", "object_name": "mart_model_registry", "object_kind": "table"},
    {"expected_asset_name": "mart_ops_overview", "object_name": "mart_ops_overview", "object_kind": "table"},
    {"expected_asset_name": "mart_creator_market_watchlist", "object_name": "mart_creator_market_watchlist", "object_kind": "table"},
    {"expected_asset_name": "mart_research_platform_status", "object_name": "mart_research_platform_status", "object_kind": "table"},
    {"expected_asset_name": "mart_ops_overview", "object_name": "latest_ops_overview", "object_kind": "view"},
    {
        "expected_asset_name": "mart_research_platform_status",
        "object_name": "latest_research_platform_status",
        "object_kind": "view",
    },
)


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


def verify_analytics_duckdb_consistency(con, *, logger) -> pd.DataFrame:
    asset_manifest = con.execute("SELECT * FROM warehouse_asset_manifest").fetchdf()
    asset_column_manifest = con.execute(
        "SELECT * FROM warehouse_asset_columns ORDER BY object_name, column_position"
    ).fetchdf()
    asset_lookup = {
        str(row["object_name"]): row
        for row in asset_manifest.to_dict(orient="records")
    }
    expected_schema_lookup: dict[str, list[dict[str, object]]] = {}
    if not asset_column_manifest.empty:
        ordered = asset_column_manifest.sort_values(["object_name", "column_position"])
        for object_name, rows in ordered.groupby("object_name", dropna=False):
            expected_schema_lookup[str(object_name)] = [
                {
                    "column_name": str(value["column_name"]),
                    "logical_type": str(value["logical_type"]),
                }
                for value in rows.to_dict(orient="records")
            ]

    results: list[dict[str, object]] = []
    failures: list[str] = []
    for target in WAREHOUSE_CONSISTENCY_TARGETS:
        expected_asset_name = str(target["expected_asset_name"])
        object_name = str(target["object_name"])
        object_kind = str(target["object_kind"])
        asset_row = asset_lookup.get(expected_asset_name)

        expected_rows: int | None = None
        expected_columns: list[str] = []
        expected_logical_types: list[str] = []
        actual_rows: int | None = None
        actual_columns: list[str] = []
        actual_logical_types: list[str] = []
        error_message = ""
        row_count_match = False
        column_match = False
        logical_type_match = False
        layer = ""

        if asset_row is None:
            error_message = f"Missing warehouse metadata for asset `{expected_asset_name}`"
        else:
            layer = str(asset_row.get("layer", "") or "")
            expected_rows = int(asset_row.get("expected_rows", 0) or 0)
            expected_schema = expected_schema_lookup.get(expected_asset_name, [])
            expected_columns = [str(record["column_name"]) for record in expected_schema]
            expected_logical_types = [str(record["logical_type"]) for record in expected_schema]
            try:
                actual_rows = int(
                    con.execute(f"SELECT COUNT(*) FROM {_quote_identifier(object_name)}").fetchone()[0]
                )
                describe_df = con.execute(
                    f"DESCRIBE SELECT * FROM {_quote_identifier(object_name)}"
                ).fetchdf()
                actual_columns = [str(value) for value in describe_df["column_name"].tolist()]
                actual_logical_types = [
                    _duckdb_logical_type(value) for value in describe_df["column_type"].tolist()
                ]
                row_count_match = actual_rows == expected_rows
                column_match = actual_columns == expected_columns
                logical_type_match = actual_logical_types == expected_logical_types
            except Exception as exc:
                error_message = str(exc)

        status = "pass" if row_count_match and column_match and logical_type_match and not error_message else "fail"
        results.append(
            {
                "expected_asset_name": expected_asset_name,
                "object_name": object_name,
                "object_kind": object_kind,
                "layer": layer,
                "expected_rows": expected_rows,
                "actual_rows": actual_rows,
                "row_count_match": row_count_match,
                "expected_columns_json": json.dumps(expected_columns),
                "actual_columns_json": json.dumps(actual_columns),
                "column_match": column_match,
                "expected_logical_types_json": json.dumps(expected_logical_types),
                "actual_logical_types_json": json.dumps(actual_logical_types),
                "logical_type_match": logical_type_match,
                "status": status,
                "error": error_message,
            }
        )
        if status != "pass":
            failures.append(object_name)

    results_df = pd.DataFrame(results)
    _replace_table(con, "warehouse_consistency_checks", results_df)
    _replace_table(
        con,
        "warehouse_consistency_summary",
        pd.DataFrame(
            [
                {
                    "checked_objects": int(len(results_df.index)),
                    "passed_objects": int((results_df["status"] == "pass").sum()) if not results_df.empty else 0,
                    "failed_objects": int((results_df["status"] != "pass").sum()) if not results_df.empty else 0,
                    "status": "pass" if not failures else "fail",
                }
            ]
        ),
    )

    if failures:
        raise ValueError(
            "Analytics DuckDB consistency verification failed for objects: " + ", ".join(sorted(set(failures)))
        )

    logger.info("Analytics DuckDB consistency verified: %d object(s)", len(results_df.index))
    return results_df


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
    asset_manifest, asset_column_manifest = warehouse_manifest_frames(warehouse_bundle.root)
    asset_columns_lookup = _asset_columns_by_object(asset_column_manifest)

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
        for asset in asset_manifest.to_dict(orient="records"):
            table_name = str(asset.get("object_name", "") or "")
            parquet_path = Path(str(asset.get("parquet_path", "") or ""))
            expected_column_count = int(asset.get("expected_column_count", 0) or 0)
            if expected_column_count == 0:
                _replace_table(con, table_name, pd.DataFrame())
            else:
                _replace_table_from_parquet(con, table_name, parquet_path)

        _replace_table(con, "warehouse_asset_manifest", asset_manifest)
        _replace_table(con, "warehouse_asset_columns", asset_column_manifest)

        run_manifests_columns = asset_columns_lookup.get("run_manifests", set())
        run_results_columns = asset_columns_lookup.get("run_results", set())
        experiment_history_columns = asset_columns_lookup.get("experiment_history", set())
        backtest_history_columns = asset_columns_lookup.get("backtest_history", set())
        benchmark_history_columns = asset_columns_lookup.get("benchmark_history", set())
        robustness_summary_columns = asset_columns_lookup.get("robustness_summary", set())
        policy_summary_columns = asset_columns_lookup.get("policy_summary", set())
        moonshot_summary_columns = asset_columns_lookup.get("moonshot_summary", set())
        mart_ops_overview_columns = asset_columns_lookup.get("mart_ops_overview", set())
        mart_creator_opportunities_columns = asset_columns_lookup.get("mart_creator_opportunities", set())
        mart_creator_market_watchlist_columns = asset_columns_lookup.get("mart_creator_market_watchlist", set())
        mart_research_platform_status_columns = asset_columns_lookup.get("mart_research_platform_status", set())
        mart_research_claim_watchlist_columns = asset_columns_lookup.get("mart_research_claim_watchlist", set())

        if {"run_id", "run_timestamp"}.issubset(run_manifests_columns) and {"run_id"}.issubset(run_results_columns):
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
        if {"profile", "model_name", "model_type", "val_top1", "test_top1"}.issubset(experiment_history_columns):
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
        if {"profile", "model_name", "top1", "model_type"}.issubset(backtest_history_columns):
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
        elif {"profile", "model_name", "top1"}.issubset(backtest_history_columns):
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
        }.issubset(run_manifests_columns):
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
        if {"model_name", "max_top1_gap", "worst_segment", "worst_bucket", "run_id"}.issubset(robustness_summary_columns):
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
        if {"model_name", "model_type", "test_discounted_reward", "test_hit_at_k", "run_id"}.issubset(policy_summary_columns):
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
        if {"benchmark_id", "model_name", "val_top1_mean", "test_top1_mean"}.issubset(benchmark_history_columns):
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
        }.issubset(moonshot_summary_columns):
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
        if {"latest_run_id", "ops_health_status", "operating_rhythm_status"}.issubset(mart_ops_overview_columns):
            con.execute(
                """
                CREATE OR REPLACE VIEW latest_ops_overview AS
                SELECT *
                FROM mart_ops_overview
                """
            )
        if {"artist_name", "priority_now_count", "max_opportunity_score"}.issubset(mart_creator_opportunities_columns):
            con.execute(
                """
                CREATE OR REPLACE VIEW creator_priority_now AS
                SELECT *
                FROM mart_creator_opportunities
                WHERE COALESCE(CAST(priority_now_count AS BIGINT), 0) > 0
                ORDER BY
                  CAST(max_opportunity_score AS DOUBLE) DESC NULLS LAST,
                  CAST(priority_now_count AS BIGINT) DESC NULLS LAST
                """
            )
        if {
            "artist_name",
            "scene_name",
            "scene_priority_now_count",
            "market_priority_score",
            "avg_opportunity_score",
        }.issubset(mart_creator_market_watchlist_columns):
            con.execute(
                """
                CREATE OR REPLACE VIEW creator_market_priority_now AS
                SELECT *
                FROM mart_creator_market_watchlist
                WHERE
                  COALESCE(CAST(scene_priority_now_count AS BIGINT), 0) > 0
                  OR COALESCE(CAST(market_priority_score AS DOUBLE), 0) > 0
                ORDER BY
                  CAST(market_priority_score AS DOUBLE) DESC NULLS LAST,
                  CAST(avg_opportunity_score AS DOUBLE) DESC NULLS LAST,
                  CAST(scene_priority_now_count AS BIGINT) DESC NULLS LAST,
                  artist_name
                """
            )
        if {
            "anchor_run_id",
            "anchor_profile",
            "anchor_research_stage",
            "status_posture",
            "submission_status",
            "ready_for_external_review",
        }.issubset(mart_research_platform_status_columns):
            con.execute(
                """
                CREATE OR REPLACE VIEW latest_research_platform_status AS
                SELECT *
                FROM mart_research_platform_status
                """
            )
        if {
            "claim_key",
            "blocked",
            "watchlist_score",
            "missing_check_count",
        }.issubset(mart_research_claim_watchlist_columns):
            con.execute(
                """
                CREATE OR REPLACE VIEW research_platform_blocked_claims AS
                SELECT *
                FROM mart_research_claim_watchlist
                WHERE COALESCE(CAST(blocked AS BOOLEAN), FALSE)
                ORDER BY
                  CAST(watchlist_score AS DOUBLE) DESC NULLS LAST,
                  CAST(missing_check_count AS BIGINT) DESC NULLS LAST,
                  claim_key
                """
            )
        verify_analytics_duckdb_consistency(con, logger=logger)
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
