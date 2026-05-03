from __future__ import annotations


def normalize_s3_prefix(s3_prefix: str) -> str:
    return str(s3_prefix).rstrip("/")


def build_athena_sql(*, database_name: str, s3_prefix: str) -> str:
    normalized_prefix = normalize_s3_prefix(s3_prefix)
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
