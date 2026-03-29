from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import shutil

import pandas as pd

from .data import discover_streaming_files, discover_technical_log_files, load_streaming_history


@dataclass(frozen=True)
class AthenaTableExport:
    name: str
    local_path: str
    row_count: int
    partitioned: bool = False


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, on_bad_lines="skip")
    except Exception:
        return pd.read_csv(path, engine="python", on_bad_lines="skip")


def _json_string(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, sort_keys=True)
    except Exception:
        return str(value)


def _copy_files(paths: list[Path], *, destination_dir: Path) -> int:
    destination_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for source_path in paths:
        if not source_path.exists() or not source_path.is_file():
            continue
        shutil.copy2(source_path, destination_dir / source_path.name)
        copied += 1
    return copied


def _discover_account_data_files(data_dir: Path) -> list[Path]:
    grouped_files: dict[Path, list[Path]] = {}
    for path in sorted(candidate for candidate in data_dir.rglob("*.json") if candidate.is_file()):
        parent_name = path.parent.name.lower()
        if "account data" not in parent_name:
            continue
        grouped_files.setdefault(path.parent, []).append(path)
    if not grouped_files:
        return []
    preferred_dir = max(grouped_files, key=lambda parent: (len(grouped_files[parent]), str(parent)))
    return list(grouped_files[preferred_dir])


def _prepare_raw_streaming_history(raw_df: pd.DataFrame) -> pd.DataFrame:
    prepared = raw_df.copy()
    if "ip_addr" in prepared.columns:
        prepared = prepared.drop(columns=["ip_addr"])

    if "ts" in prepared.columns:
        prepared["played_at"] = pd.to_datetime(prepared["ts"], utc=True, errors="coerce").dt.tz_localize(None)
        prepared = prepared.drop(columns=["ts"])
    else:
        prepared["played_at"] = pd.NaT

    content_type = pd.Series("track", index=prepared.index, dtype="object")
    if "spotify_track_uri" in prepared.columns:
        has_track = prepared["spotify_track_uri"].fillna("").astype(str).str.strip().ne("")
        content_type = content_type.where(has_track, "unknown")
    if "spotify_episode_uri" in prepared.columns:
        has_episode = prepared["spotify_episode_uri"].fillna("").astype(str).str.strip().ne("")
        content_type = content_type.mask(has_episode, "episode")
    if "audiobook_uri" in prepared.columns:
        has_audiobook = prepared["audiobook_uri"].fillna("").astype(str).str.strip().ne("")
        content_type = content_type.mask(has_audiobook, "audiobook")
    prepared["content_type"] = content_type

    prepared = prepared.loc[prepared["played_at"].notna()].copy()
    prepared["year"] = prepared["played_at"].dt.year.astype("int64")
    prepared["month"] = prepared["played_at"].dt.month.astype("int64")

    expected_columns = [
        "played_at",
        "platform",
        "ms_played",
        "conn_country",
        "master_metadata_track_name",
        "master_metadata_album_artist_name",
        "master_metadata_album_album_name",
        "spotify_track_uri",
        "episode_name",
        "episode_show_name",
        "spotify_episode_uri",
        "audiobook_title",
        "audiobook_uri",
        "audiobook_chapter_uri",
        "audiobook_chapter_title",
        "reason_start",
        "reason_end",
        "shuffle",
        "skipped",
        "offline",
        "offline_timestamp",
        "incognito_mode",
        "content_type",
        "year",
        "month",
    ]
    for column in expected_columns:
        if column not in prepared.columns:
            prepared[column] = None
    return prepared[expected_columns]


def _prepare_experiment_history(path: Path) -> pd.DataFrame:
    df = _safe_read_csv(path)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "event_timestamp",
                "run_id",
                "run_name",
                "profile",
                "model_name",
                "model_type",
                "model_family",
                "val_top1",
                "val_top5",
                "test_top1",
                "test_top5",
                "fit_seconds",
                "epochs",
                "data_records",
            ]
        )
    if "timestamp" in df.columns:
        df["event_timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.drop(columns=["timestamp"])
    return df[
        [
            "event_timestamp",
            "run_id",
            "run_name",
            "profile",
            "model_name",
            "model_type",
            "model_family",
            "val_top1",
            "val_top5",
            "test_top1",
            "test_top5",
            "fit_seconds",
            "epochs",
            "data_records",
        ]
    ]


def _prepare_backtest_history(path: Path) -> pd.DataFrame:
    df = _safe_read_csv(path)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "event_timestamp",
                "run_id",
                "run_name",
                "profile",
                "model_name",
                "model_family",
                "fold",
                "train_rows",
                "test_rows",
                "fit_seconds",
                "top1",
                "top5",
            ]
        )
    if "timestamp" in df.columns:
        df["event_timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.drop(columns=["timestamp"])
    return df[
        [
            "event_timestamp",
            "run_id",
            "run_name",
            "profile",
            "model_name",
            "model_family",
            "fold",
            "train_rows",
            "test_rows",
            "fit_seconds",
            "top1",
            "top5",
        ]
    ]


def _prepare_optuna_history(path: Path) -> pd.DataFrame:
    df = _safe_read_csv(path)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "event_timestamp",
                "run_id",
                "run_name",
                "profile",
                "model_name",
                "base_model_name",
                "n_trials",
                "val_top1",
                "test_top1",
                "fit_seconds",
                "best_params_json",
            ]
        )
    if "timestamp" in df.columns:
        df["event_timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.drop(columns=["timestamp"])
    return df[
        [
            "event_timestamp",
            "run_id",
            "run_name",
            "profile",
            "model_name",
            "base_model_name",
            "n_trials",
            "val_top1",
            "test_top1",
            "fit_seconds",
            "best_params_json",
        ]
    ]


def _prepare_benchmark_history(path: Path) -> pd.DataFrame:
    df = _safe_read_csv(path)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "event_timestamp",
                "benchmark_id",
                "model_name",
                "model_type",
                "model_family",
                "runs",
                "val_top1_mean",
                "val_top1_std",
                "val_top1_ci95",
                "test_top1_mean",
                "test_top1_std",
                "test_top1_ci95",
            ]
        )
    if "timestamp" in df.columns:
        df["event_timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.drop(columns=["timestamp"])
    return df[
        [
            "event_timestamp",
            "benchmark_id",
            "model_name",
            "model_type",
            "model_family",
            "runs",
            "val_top1_mean",
            "val_top1_std",
            "val_top1_ci95",
            "test_top1_mean",
            "test_top1_std",
            "test_top1_ci95",
        ]
    ]


def _prepare_run_manifests(output_dir: Path) -> pd.DataFrame:
    columns = [
        "run_id",
        "run_name",
        "profile",
        "run_timestamp",
        "data_records",
        "num_artists",
        "num_context_features",
        "enable_retrieval_stack",
        "enable_self_supervised_pretraining",
        "enable_friction_analysis",
        "enable_moonshot_lab",
        "retrieval_candidate_k",
        "enable_mlflow",
        "enable_optuna",
        "optuna_trials",
        "enable_temporal_backtest",
        "temporal_backtest_folds",
        "temporal_backtest_adaptation_mode",
        "backtest_rows",
        "optuna_rows",
        "cache_enabled",
        "cache_hit",
        "cache_fingerprint",
        "cache_source_file_count",
        "champion_gate_status",
        "champion_gate_promoted",
        "champion_gate_metric_source",
        "champion_gate_threshold",
        "champion_gate_regression",
        "champion_gate_champion_run_id",
        "champion_gate_champion_model_name",
        "champion_gate_champion_score",
        "champion_gate_challenger_model_name",
        "champion_gate_challenger_score",
        "champion_alias_updated",
        "champion_alias_model_name",
        "champion_alias_model_type",
        "artifact_cleanup_mode",
        "artifact_cleanup_status",
        "artifact_cleanup_selected_model_name",
        "artifact_cleanup_freed_bytes",
        "artifact_retention_keep_last_full_runs",
        "run_dir",
        "deep_models_json",
        "classical_models_json",
        "optuna_models_json",
        "temporal_backtest_models_json",
        "cache_json",
        "champion_gate_json",
        "champion_alias_json",
        "artifact_cleanup_json",
        "artifact_retention_json",
        "mlflow_artifact_cleanup_json",
        "raw_json",
    ]
    rows: list[dict[str, object]] = []
    for path in sorted((output_dir / "runs").glob("*/run_manifest.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        cache = payload.get("cache", {}) if isinstance(payload.get("cache"), dict) else {}
        champion_gate = payload.get("champion_gate", {}) if isinstance(payload.get("champion_gate"), dict) else {}
        champion_alias = payload.get("champion_alias", {}) if isinstance(payload.get("champion_alias"), dict) else {}
        artifact_cleanup = payload.get("artifact_cleanup", {}) if isinstance(payload.get("artifact_cleanup"), dict) else {}
        artifact_retention = payload.get("artifact_retention", {}) if isinstance(payload.get("artifact_retention"), dict) else {}
        rows.append(
            {
                "run_id": payload.get("run_id"),
                "run_name": payload.get("run_name"),
                "profile": payload.get("profile"),
                "run_timestamp": pd.to_datetime(payload.get("timestamp"), errors="coerce"),
                "data_records": payload.get("data_records"),
                "num_artists": payload.get("num_artists"),
                "num_context_features": payload.get("num_context_features"),
                "enable_retrieval_stack": payload.get("enable_retrieval_stack"),
                "enable_self_supervised_pretraining": payload.get("enable_self_supervised_pretraining"),
                "enable_friction_analysis": payload.get("enable_friction_analysis"),
                "enable_moonshot_lab": payload.get("enable_moonshot_lab"),
                "retrieval_candidate_k": payload.get("retrieval_candidate_k"),
                "enable_mlflow": payload.get("enable_mlflow"),
                "enable_optuna": payload.get("enable_optuna"),
                "optuna_trials": payload.get("optuna_trials"),
                "enable_temporal_backtest": payload.get("enable_temporal_backtest"),
                "temporal_backtest_folds": payload.get("temporal_backtest_folds"),
                "temporal_backtest_adaptation_mode": payload.get("temporal_backtest_adaptation_mode"),
                "backtest_rows": payload.get("backtest_rows"),
                "optuna_rows": payload.get("optuna_rows"),
                "cache_enabled": cache.get("enabled"),
                "cache_hit": cache.get("hit"),
                "cache_fingerprint": cache.get("fingerprint"),
                "cache_source_file_count": cache.get("source_file_count"),
                "champion_gate_status": champion_gate.get("status"),
                "champion_gate_promoted": champion_gate.get("promoted"),
                "champion_gate_metric_source": champion_gate.get("metric_source"),
                "champion_gate_threshold": champion_gate.get("threshold"),
                "champion_gate_regression": champion_gate.get("regression"),
                "champion_gate_champion_run_id": champion_gate.get("champion_run_id"),
                "champion_gate_champion_model_name": champion_gate.get("champion_model_name"),
                "champion_gate_champion_score": champion_gate.get("champion_score"),
                "champion_gate_challenger_model_name": champion_gate.get("challenger_model_name"),
                "champion_gate_challenger_score": champion_gate.get("challenger_score"),
                "champion_alias_updated": champion_alias.get("updated"),
                "champion_alias_model_name": champion_alias.get("model_name"),
                "champion_alias_model_type": champion_alias.get("model_type"),
                "artifact_cleanup_mode": artifact_cleanup.get("mode"),
                "artifact_cleanup_status": artifact_cleanup.get("status"),
                "artifact_cleanup_selected_model_name": artifact_cleanup.get("selected_model_name"),
                "artifact_cleanup_freed_bytes": artifact_cleanup.get("freed_bytes"),
                "artifact_retention_keep_last_full_runs": artifact_retention.get("keep_last_full_runs"),
                "run_dir": str(path.parent.resolve()),
                "deep_models_json": _json_string(payload.get("deep_models")),
                "classical_models_json": _json_string(payload.get("classical_models")),
                "optuna_models_json": _json_string(payload.get("optuna_models")),
                "temporal_backtest_models_json": _json_string(payload.get("temporal_backtest_models")),
                "cache_json": _json_string(payload.get("cache")),
                "champion_gate_json": _json_string(payload.get("champion_gate")),
                "champion_alias_json": _json_string(payload.get("champion_alias")),
                "artifact_cleanup_json": _json_string(payload.get("artifact_cleanup")),
                "artifact_retention_json": _json_string(payload.get("artifact_retention")),
                "mlflow_artifact_cleanup_json": _json_string(payload.get("mlflow_artifact_cleanup")),
                "raw_json": _json_string(payload),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _prepare_run_results(output_dir: Path) -> pd.DataFrame:
    columns = [
        "run_id",
        "run_dir",
        "model_name",
        "model_type",
        "model_family",
        "base_model_name",
        "val_top1",
        "val_top5",
        "val_ndcg_at5",
        "val_mrr_at5",
        "val_coverage_at5",
        "val_diversity_at5",
        "test_top1",
        "test_top5",
        "test_ndcg_at5",
        "test_mrr_at5",
        "test_coverage_at5",
        "test_diversity_at5",
        "fit_seconds",
        "epochs",
        "prediction_bundle_path",
        "estimator_artifact_path",
        "n_trials",
        "ensemble_members_json",
        "best_params_json",
        "raw_json",
    ]
    rows: list[dict[str, object]] = []
    for path in sorted((output_dir / "runs").glob("*/run_results.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, list):
            continue
        run_id = path.parent.name
        run_dir = str(path.parent.resolve())
        for row in payload:
            if not isinstance(row, dict):
                continue
            rows.append(
                {
                    "run_id": run_id,
                    "run_dir": run_dir,
                    "model_name": row.get("model_name"),
                    "model_type": row.get("model_type"),
                    "model_family": row.get("model_family"),
                    "base_model_name": row.get("base_model_name"),
                    "val_top1": row.get("val_top1"),
                    "val_top5": row.get("val_top5"),
                    "val_ndcg_at5": row.get("val_ndcg_at5"),
                    "val_mrr_at5": row.get("val_mrr_at5"),
                    "val_coverage_at5": row.get("val_coverage_at5"),
                    "val_diversity_at5": row.get("val_diversity_at5"),
                    "test_top1": row.get("test_top1"),
                    "test_top5": row.get("test_top5"),
                    "test_ndcg_at5": row.get("test_ndcg_at5"),
                    "test_mrr_at5": row.get("test_mrr_at5"),
                    "test_coverage_at5": row.get("test_coverage_at5"),
                    "test_diversity_at5": row.get("test_diversity_at5"),
                    "fit_seconds": row.get("fit_seconds"),
                    "epochs": row.get("epochs"),
                    "prediction_bundle_path": row.get("prediction_bundle_path"),
                    "estimator_artifact_path": row.get("estimator_artifact_path"),
                    "n_trials": row.get("n_trials"),
                    "ensemble_members_json": _json_string(row.get("ensemble_members")),
                    "best_params_json": _json_string(row.get("best_params")),
                    "raw_json": _json_string(row),
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _write_parquet_file(con, df: pd.DataFrame, output_file: Path, *, relation_name: str) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    con.register(relation_name, df)
    try:
        con.execute(f"COPY (SELECT * FROM {relation_name}) TO '{output_file.as_posix()}' (FORMAT PARQUET)")
    finally:
        con.unregister(relation_name)


def _write_partitioned_parquet_table(
    con,
    df: pd.DataFrame,
    *,
    base_dir: Path,
    relation_name_prefix: str,
    partition_columns: tuple[str, ...],
) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    if df.empty:
        return
    for partition_index, (keys, part_df) in enumerate(
        df.groupby(list(partition_columns), dropna=False, sort=True), start=1
    ):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        output_dir = base_dir
        for column_name, column_value in zip(partition_columns, key_tuple):
            output_dir = output_dir / f"{column_name}={column_value}"
        _write_parquet_file(
            con,
            part_df.drop(columns=list(partition_columns)),
            output_dir / "data.parquet",
            relation_name=f"{relation_name_prefix}_{partition_index}",
        )


def _normalize_s3_prefix(s3_prefix: str) -> str:
    return str(s3_prefix).rstrip("/")


def build_athena_sql(*, database_name: str, s3_prefix: str) -> str:
    normalized_prefix = _normalize_s3_prefix(s3_prefix)
    return f"""CREATE DATABASE IF NOT EXISTS {database_name};
USE {database_name};

CREATE EXTERNAL TABLE IF NOT EXISTS raw_streaming_history (
  played_at timestamp,
  platform string,
  ms_played bigint,
  conn_country string,
  master_metadata_track_name string,
  master_metadata_album_artist_name string,
  master_metadata_album_album_name string,
  spotify_track_uri string,
  episode_name string,
  episode_show_name string,
  spotify_episode_uri string,
  audiobook_title string,
  audiobook_uri string,
  audiobook_chapter_uri string,
  audiobook_chapter_title string,
  reason_start string,
  reason_end string,
  shuffle boolean,
  skipped boolean,
  offline boolean,
  offline_timestamp bigint,
  incognito_mode boolean,
  content_type string
)
PARTITIONED BY (year int, month int)
STORED AS PARQUET
LOCATION '{normalized_prefix}/curated/raw_streaming_history/';

CREATE EXTERNAL TABLE IF NOT EXISTS experiment_history (
  event_timestamp timestamp,
  run_id string,
  run_name string,
  profile string,
  model_name string,
  model_type string,
  model_family string,
  val_top1 double,
  val_top5 double,
  test_top1 double,
  test_top5 double,
  fit_seconds double,
  epochs int,
  data_records bigint
)
STORED AS PARQUET
LOCATION '{normalized_prefix}/curated/experiment_history/';

CREATE EXTERNAL TABLE IF NOT EXISTS backtest_history (
  event_timestamp timestamp,
  run_id string,
  run_name string,
  profile string,
  model_name string,
  model_family string,
  fold int,
  train_rows bigint,
  test_rows bigint,
  fit_seconds double,
  top1 double,
  top5 double
)
STORED AS PARQUET
LOCATION '{normalized_prefix}/curated/backtest_history/';

CREATE EXTERNAL TABLE IF NOT EXISTS optuna_history (
  event_timestamp timestamp,
  run_id string,
  run_name string,
  profile string,
  model_name string,
  base_model_name string,
  n_trials int,
  val_top1 double,
  test_top1 double,
  fit_seconds double,
  best_params_json string
)
STORED AS PARQUET
LOCATION '{normalized_prefix}/curated/optuna_history/';

CREATE EXTERNAL TABLE IF NOT EXISTS benchmark_history (
  event_timestamp timestamp,
  benchmark_id string,
  model_name string,
  model_type string,
  model_family string,
  runs int,
  val_top1_mean double,
  val_top1_std double,
  val_top1_ci95 double,
  test_top1_mean double,
  test_top1_std double,
  test_top1_ci95 double
)
STORED AS PARQUET
LOCATION '{normalized_prefix}/curated/benchmark_history/';

CREATE EXTERNAL TABLE IF NOT EXISTS run_manifests (
  run_id string,
  run_name string,
  profile string,
  run_timestamp timestamp,
  data_records bigint,
  num_artists bigint,
  num_context_features bigint,
  enable_retrieval_stack boolean,
  enable_self_supervised_pretraining boolean,
  enable_friction_analysis boolean,
  enable_moonshot_lab boolean,
  retrieval_candidate_k bigint,
  enable_mlflow boolean,
  enable_optuna boolean,
  optuna_trials bigint,
  enable_temporal_backtest boolean,
  temporal_backtest_folds bigint,
  temporal_backtest_adaptation_mode string,
  backtest_rows bigint,
  optuna_rows bigint,
  cache_enabled boolean,
  cache_hit boolean,
  cache_fingerprint string,
  cache_source_file_count bigint,
  champion_gate_status string,
  champion_gate_promoted boolean,
  champion_gate_metric_source string,
  champion_gate_threshold double,
  champion_gate_regression double,
  champion_gate_champion_run_id string,
  champion_gate_champion_model_name string,
  champion_gate_champion_score double,
  champion_gate_challenger_model_name string,
  champion_gate_challenger_score double,
  champion_alias_updated boolean,
  champion_alias_model_name string,
  champion_alias_model_type string,
  artifact_cleanup_mode string,
  artifact_cleanup_status string,
  artifact_cleanup_selected_model_name string,
  artifact_cleanup_freed_bytes bigint,
  artifact_retention_keep_last_full_runs bigint,
  run_dir string,
  deep_models_json string,
  classical_models_json string,
  optuna_models_json string,
  temporal_backtest_models_json string,
  cache_json string,
  champion_gate_json string,
  champion_alias_json string,
  artifact_cleanup_json string,
  artifact_retention_json string,
  mlflow_artifact_cleanup_json string,
  raw_json string
)
STORED AS PARQUET
LOCATION '{normalized_prefix}/curated/run_manifests/';

CREATE EXTERNAL TABLE IF NOT EXISTS run_results (
  run_id string,
  run_dir string,
  model_name string,
  model_type string,
  model_family string,
  base_model_name string,
  val_top1 double,
  val_top5 double,
  val_ndcg_at5 double,
  val_mrr_at5 double,
  val_coverage_at5 double,
  val_diversity_at5 double,
  test_top1 double,
  test_top5 double,
  test_ndcg_at5 double,
  test_mrr_at5 double,
  test_coverage_at5 double,
  test_diversity_at5 double,
  fit_seconds double,
  epochs int,
  prediction_bundle_path string,
  estimator_artifact_path string,
  n_trials int,
  ensemble_members_json string,
  best_params_json string,
  raw_json string
)
STORED AS PARQUET
LOCATION '{normalized_prefix}/curated/run_results/';

MSCK REPAIR TABLE raw_streaming_history;

CREATE OR REPLACE VIEW latest_run_results AS
SELECT rr.*
FROM run_results rr
JOIN (
  SELECT run_id
  FROM run_manifests
  ORDER BY run_timestamp DESC, run_id DESC
  LIMIT 1
) latest
  ON rr.run_id = latest.run_id;

CREATE OR REPLACE VIEW best_models_by_profile AS
SELECT
  profile,
  model_name,
  model_type,
  AVG(val_top1) AS mean_val_top1,
  AVG(test_top1) AS mean_test_top1,
  COUNT(*) AS runs
FROM experiment_history
GROUP BY profile, model_name, model_type;

CREATE OR REPLACE VIEW backtest_model_summary AS
SELECT
  profile,
  model_name,
  model_family,
  AVG(top1) AS mean_backtest_top1,
  STDDEV_SAMP(top1) AS std_backtest_top1,
  COUNT(*) AS folds
FROM backtest_history
GROUP BY profile, model_name, model_family;

CREATE OR REPLACE VIEW champion_runs AS
SELECT
  run_id,
  run_name,
  profile,
  run_timestamp,
  champion_gate_metric_source,
  champion_alias_model_name,
  champion_alias_model_type,
  champion_gate_champion_run_id,
  champion_gate_champion_model_name,
  champion_gate_challenger_model_name,
  champion_gate_threshold,
  champion_gate_regression
FROM run_manifests
WHERE champion_gate_promoted = TRUE;
"""


def build_athena_queries(*, database_name: str) -> str:
    return f"""USE {database_name};

SELECT master_metadata_album_artist_name, COUNT(*) AS plays
FROM raw_streaming_history
GROUP BY 1
ORDER BY plays DESC
LIMIT 25;

SELECT profile, model_name, mean_backtest_top1, folds
FROM backtest_model_summary
ORDER BY profile, mean_backtest_top1 DESC;

SELECT run_id, model_name, model_type, val_top1, test_top1, fit_seconds
FROM latest_run_results
ORDER BY val_top1 DESC;

SELECT profile, model_name, model_type, mean_val_top1, mean_test_top1, runs
FROM best_models_by_profile
ORDER BY profile, mean_val_top1 DESC;
"""


def export_athena_bundle(
    *,
    data_dir: Path,
    output_dir: Path,
    export_dir: Path,
    include_video: bool,
    s3_prefix: str,
    database_name: str,
    logger,
) -> dict[str, object]:
    try:
        import duckdb
    except Exception as exc:
        raise RuntimeError(f"DuckDB is required for Athena export: {exc}") from exc

    normalized_prefix = _normalize_s3_prefix(s3_prefix)
    data_dir = data_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    export_dir = export_dir.expanduser().resolve()

    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    raw_export_dir = export_dir / "raw"
    curated_export_dir = export_dir / "curated"
    ddl_dir = export_dir / "ddl"
    notes_dir = export_dir / "notes"
    ddl_dir.mkdir(parents=True, exist_ok=True)
    notes_dir.mkdir(parents=True, exist_ok=True)

    streaming_files = discover_streaming_files(data_dir, include_video, logger)
    technical_log_files = discover_technical_log_files(data_dir, logger)
    account_data_files = _discover_account_data_files(data_dir)

    raw_copy_summary = {
        "streaming_history_files": _copy_files(
            [path for path in streaming_files if path.name.startswith("Streaming_History_Audio_")],
            destination_dir=raw_export_dir / "spotify_streaming_history",
        ),
        "technical_log_files": _copy_files(
            technical_log_files,
            destination_dir=raw_export_dir / "spotify_technical_logs",
        ),
        "account_data_files": _copy_files(
            account_data_files,
            destination_dir=raw_export_dir / "spotify_account_data",
        ),
    }

    raw_streaming_history = _prepare_raw_streaming_history(
        load_streaming_history(data_dir, include_video=include_video, logger=logger)
    )
    experiment_history = _prepare_experiment_history(output_dir / "history" / "experiment_history.csv")
    backtest_history = _prepare_backtest_history(output_dir / "history" / "backtest_history.csv")
    optuna_history = _prepare_optuna_history(output_dir / "history" / "optuna_history.csv")
    benchmark_history = _prepare_benchmark_history(output_dir / "history" / "benchmark_history.csv")
    run_manifests = _prepare_run_manifests(output_dir)
    run_results = _prepare_run_results(output_dir)

    exported_tables: list[AthenaTableExport] = []
    with duckdb.connect() as con:
        _write_partitioned_parquet_table(
            con,
            raw_streaming_history,
            base_dir=curated_export_dir / "raw_streaming_history",
            relation_name_prefix="raw_streaming_history",
            partition_columns=("year", "month"),
        )
        exported_tables.append(
            AthenaTableExport(
                name="raw_streaming_history",
                local_path=str((curated_export_dir / "raw_streaming_history").resolve()),
                row_count=int(len(raw_streaming_history)),
                partitioned=True,
            )
        )

        non_partitioned_tables = [
            ("experiment_history", experiment_history),
            ("backtest_history", backtest_history),
            ("optuna_history", optuna_history),
            ("benchmark_history", benchmark_history),
            ("run_manifests", run_manifests),
            ("run_results", run_results),
        ]
        for relation_name, frame in non_partitioned_tables:
            table_dir = curated_export_dir / relation_name
            _write_parquet_file(
                con,
                frame,
                table_dir / "data.parquet",
                relation_name=f"{relation_name}_df",
            )
            exported_tables.append(
                AthenaTableExport(
                    name=relation_name,
                    local_path=str(table_dir.resolve()),
                    row_count=int(len(frame)),
                    partitioned=False,
                )
            )

    ddl_path = ddl_dir / "athena.sql"
    ddl_path.write_text(
        build_athena_sql(database_name=database_name, s3_prefix=normalized_prefix),
        encoding="utf-8",
    )

    queries_path = ddl_dir / "sample_queries.sql"
    queries_path.write_text(build_athena_queries(database_name=database_name), encoding="utf-8")

    commands_path = notes_dir / "aws_cli_commands.txt"
    commands_path.write_text(
        "\n".join(
            [
                f"aws s3 sync {export_dir} {normalized_prefix}",
                f"aws s3 ls {normalized_prefix}/curated/",
                f"aws s3 ls {normalized_prefix}/raw/",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    notes_path = notes_dir / "README.md"
    notes_path.write_text(
        "\n".join(
            [
                "# AWS Athena Export Bundle",
                "",
                f"- Database name: `{database_name}`",
                f"- S3 prefix: `{normalized_prefix}`",
                f"- DDL file: `{ddl_path.name}`",
                f"- Sample queries: `{queries_path.name}`",
                "",
                "The curated Parquet tables are safe to register in Athena.",
                "The raw JSON files still contain the original Spotify export payloads and should stay private in S3.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = {
        "database_name": database_name,
        "s3_prefix": normalized_prefix,
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "export_dir": str(export_dir),
        "raw_copy_summary": raw_copy_summary,
        "tables": [table.__dict__ for table in exported_tables],
        "ddl_path": str(ddl_path),
        "sample_queries_path": str(queries_path),
        "aws_cli_commands_path": str(commands_path),
    }
    report_path = export_dir / "export_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
