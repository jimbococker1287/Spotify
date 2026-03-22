from __future__ import annotations

from pathlib import Path
import json

import pandas as pd


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, on_bad_lines="skip")
    except Exception:
        return pd.read_csv(path, engine="python", on_bad_lines="skip")


def _replace_table(con, table_name: str, df: pd.DataFrame) -> None:
    if list(df.columns):
        relation_name = f"{table_name}_df"
        con.register(relation_name, df)
        con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM {relation_name}")
    else:
        con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT CAST(NULL AS VARCHAR) AS _empty WHERE 1=0")


def _collect_run_manifests(output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted((output_dir / "runs").glob("*/run_manifest.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        payload["run_dir"] = str(path.parent.resolve())
        rows.append(payload)
    return pd.json_normalize(rows) if rows else pd.DataFrame()


def _collect_run_results(output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted((output_dir / "runs").glob("*/run_results.json")):
        run_id = path.parent.name
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, list):
            continue
        for row in payload:
            if not isinstance(row, dict):
                continue
            flat = dict(row)
            flat["run_id"] = run_id
            flat["run_dir"] = str(path.parent.resolve())
            rows.append(flat)
    return pd.json_normalize(rows) if rows else pd.DataFrame()


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
    optuna_history = _safe_read_csv(output_dir / "history" / "optuna_history.csv")
    run_manifests = _collect_run_manifests(output_dir)
    run_results = _collect_run_results(output_dir)

    try:
        con = duckdb.connect(str(db_path))
    except Exception as exc:
        logger.warning("DuckDB analytics refresh skipped because the database is busy or unavailable: %s", exc)
        return None
    try:
        _replace_table(con, "raw_streaming_history", raw_df)
        _replace_table(con, "experiment_history", experiment_history)
        _replace_table(con, "backtest_history", backtest_history)
        _replace_table(con, "optuna_history", optuna_history)
        _replace_table(con, "run_manifests", run_manifests)
        _replace_table(con, "run_results", run_results)

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
    except Exception as exc:
        logger.warning("DuckDB analytics refresh did not complete: %s", exc)
        return None
    finally:
        con.close()

    logger.info("Analytics DuckDB refreshed: %s", db_path)
    return db_path
