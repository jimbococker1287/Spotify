# AWS Athena Import Guide

This project can be imported into AWS as a very small analytics lakehouse:

- `Amazon S3` stores the raw Spotify export plus curated Parquet tables.
- `AWS Glue Data Catalog` stores Athena table metadata.
- `Amazon Athena` queries the curated tables with SQL.

This is the lowest-friction AWS-native replacement for the local DuckDB analytics database in `outputs/analytics/spotify_analytics.duckdb`.

## What This Repo Now Generates

Run the local export helper:

```bash
make athena-export EXTRA_ARGS="--s3-prefix s3://YOUR-BUCKET/spotify-athena --database-name spotify_taste_os"
```

That writes a bundle under:

```text
outputs/aws_athena_bundle/
```

The bundle contains:

- `raw/spotify_streaming_history/`: copied raw Spotify streaming-history JSON files
- `raw/spotify_technical_logs/`: copied raw technical-log JSON files
- `raw/spotify_account_data/`: copied account-data JSON files
- `curated/raw_streaming_history/`: partitioned Parquet table for Athena
- `curated/experiment_history/`: Parquet table built from `outputs/history/experiment_history.csv`
- `curated/backtest_history/`: Parquet table built from `outputs/history/backtest_history.csv`
- `curated/optuna_history/`: Parquet table built from `outputs/history/optuna_history.csv`
- `curated/benchmark_history/`: Parquet table built from `outputs/history/benchmark_history.csv`
- `curated/run_manifests/`: flattened Parquet table built from `outputs/runs/*/run_manifest.json`
- `curated/run_results/`: flattened Parquet table built from `outputs/runs/*/run_results.json`
- `ddl/athena.sql`: ready-to-run Athena DDL
- `ddl/sample_queries.sql`: validation/sample queries
- `notes/aws_cli_commands.txt`: copy-paste sync commands
- `export_report.json`: export summary

Notes:

- The curated raw-history table intentionally omits `ip_addr`.
- The `raw/` area still contains original Spotify JSON and should stay private in S3.

## Recommended S3 Layout

Use one bucket dedicated to this project. Example:

```text
s3://spotify-taste-os-data/spotify-athena/
```

After syncing the bundle, the S3 layout should look like:

```text
s3://spotify-taste-os-data/spotify-athena/
  raw/
    spotify_streaming_history/
    spotify_technical_logs/
    spotify_account_data/
  curated/
    raw_streaming_history/
      year=2014/month=3/data.parquet
      year=2024/month=7/data.parquet
    experiment_history/
      data.parquet
    backtest_history/
      data.parquet
    optuna_history/
      data.parquet
    benchmark_history/
      data.parquet
    run_manifests/
      data.parquet
    run_results/
      data.parquet
  athena-results/
```

## One-Time AWS Setup

### 1. Create The S3 Bucket

Create a bucket in the same AWS region you plan to use for Athena. Example region:

- `us-east-1`

Recommended bucket options:

- Block all public access: `On`
- Versioning: optional
- Default encryption: `SSE-S3` is enough for this project

Create an empty prefix for Athena query results:

```text
s3://YOUR-BUCKET/spotify-athena/athena-results/
```

### 2. Create Or Reuse An Athena Workgroup

In Athena:

1. Open `Workgroups`
2. Create a workgroup such as `spotify-taste-os`
3. Set query result location to:

```text
s3://YOUR-BUCKET/spotify-athena/athena-results/
```

Recommended settings:

- Enforce workgroup settings: `On`
- Publish metrics to CloudWatch: `On`
- Bytes scanned cutoff per query: set a small guardrail like `1 GB`

### 3. IAM Permissions

The identity running Athena needs access to:

- the S3 bucket
- Athena APIs
- Glue Data Catalog APIs

Minimal starter policy example:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:ListBucket"
      ],
      "Resource": [
        "arn:aws:s3:::YOUR-BUCKET"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject"
      ],
      "Resource": [
        "arn:aws:s3:::YOUR-BUCKET/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "athena:StartQueryExecution",
        "athena:GetQueryExecution",
        "athena:GetQueryResults",
        "athena:StopQueryExecution",
        "athena:GetWorkGroup",
        "athena:ListWorkGroups"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "glue:CreateDatabase",
        "glue:GetDatabase",
        "glue:GetDatabases",
        "glue:CreateTable",
        "glue:UpdateTable",
        "glue:GetTable",
        "glue:GetTables",
        "glue:GetPartitions",
        "glue:BatchCreatePartition"
      ],
      "Resource": "*"
    }
  ]
}
```

If you use the AWS CLI to upload files from your machine, your local AWS credentials also need S3 write access.

## Local Export And Upload

### 1. Generate The Bundle

Run:

```bash
make athena-export EXTRA_ARGS="--s3-prefix s3://YOUR-BUCKET/spotify-athena --database-name spotify_taste_os"
```

Or directly:

```bash
PYTHONPATH=. .venv/bin/python scripts/export_athena_bundle.py \
  --data-dir data/raw \
  --output-dir outputs \
  --export-dir outputs/aws_athena_bundle \
  --s3-prefix s3://YOUR-BUCKET/spotify-athena \
  --database-name spotify_taste_os
```

### 2. Sync The Bundle To S3

```bash
aws s3 sync outputs/aws_athena_bundle s3://YOUR-BUCKET/spotify-athena
```

Validate:

```bash
aws s3 ls s3://YOUR-BUCKET/spotify-athena/curated/
aws s3 ls s3://YOUR-BUCKET/spotify-athena/raw/
```

## Athena Import Steps

### 1. Open Athena Query Editor

Select:

- Workgroup: `spotify-taste-os` or your chosen workgroup
- Data source: `AwsDataCatalog`

### 2. Run The Generated DDL

Open and paste:

- `outputs/aws_athena_bundle/ddl/athena.sql`

That file will:

- create the Athena database
- create the curated external tables
- repair partitions for `raw_streaming_history`
- create helper views

### 3. Validate The Import

Run the generated sample SQL from:

- `outputs/aws_athena_bundle/ddl/sample_queries.sql`

Expected useful queries include:

```sql
USE spotify_taste_os;

SELECT master_metadata_album_artist_name, COUNT(*) AS plays
FROM raw_streaming_history
GROUP BY 1
ORDER BY plays DESC
LIMIT 25;
```

```sql
SELECT profile, model_name, mean_backtest_top1, folds
FROM backtest_model_summary
ORDER BY profile, mean_backtest_top1 DESC;
```

```sql
SELECT run_id, model_name, model_type, val_top1, test_top1, fit_seconds
FROM latest_run_results
ORDER BY val_top1 DESC;
```

## Tables Created

### `raw_streaming_history`

Curated from the Spotify audio history JSON files.

Columns:

- `played_at`
- `platform`
- `ms_played`
- `conn_country`
- `master_metadata_track_name`
- `master_metadata_album_artist_name`
- `master_metadata_album_album_name`
- `spotify_track_uri`
- `episode_name`
- `episode_show_name`
- `spotify_episode_uri`
- `audiobook_title`
- `audiobook_uri`
- `audiobook_chapter_uri`
- `audiobook_chapter_title`
- `reason_start`
- `reason_end`
- `shuffle`
- `skipped`
- `offline`
- `offline_timestamp`
- `incognito_mode`
- `content_type`
- partitions: `year`, `month`

### `experiment_history`

Derived from `outputs/history/experiment_history.csv`.

### `backtest_history`

Derived from `outputs/history/backtest_history.csv`.

### `optuna_history`

Derived from `outputs/history/optuna_history.csv`.

### `benchmark_history`

Derived from `outputs/history/benchmark_history.csv`.

### `run_manifests`

Flattened from `outputs/runs/*/run_manifest.json`.

Important extracted columns:

- `run_id`
- `run_name`
- `profile`
- `run_timestamp`
- `data_records`
- `num_artists`
- `num_context_features`
- `enable_mlflow`
- `enable_optuna`
- `enable_temporal_backtest`
- `cache_hit`
- `champion_gate_promoted`
- `champion_gate_metric_source`
- `champion_gate_threshold`
- `champion_gate_regression`
- `champion_alias_model_name`
- `champion_alias_model_type`
- `artifact_cleanup_freed_bytes`

It also keeps the original manifest as `raw_json`.

### `run_results`

Flattened from `outputs/runs/*/run_results.json`.

Important extracted columns:

- `run_id`
- `model_name`
- `model_type`
- `model_family`
- `base_model_name`
- `val_top1`
- `val_top5`
- `val_ndcg_at5`
- `val_mrr_at5`
- `test_top1`
- `test_top5`
- `test_ndcg_at5`
- `test_mrr_at5`
- `fit_seconds`
- `epochs`
- `prediction_bundle_path`
- `estimator_artifact_path`
- `n_trials`

It also keeps the original row as `raw_json`.

## Views Created

The generated SQL also creates these views:

- `latest_run_results`
- `best_models_by_profile`
- `backtest_model_summary`
- `champion_runs`

These mirror the main local analytics patterns in the DuckDB workflow.

## Ongoing Refresh Workflow

After each daily fast run or weekly full run:

1. Regenerate the Athena bundle locally
2. Sync the bundle to S3
3. Re-run `athena.sql`

In practice, the DDL is idempotent, so rerunning it is fine.

Suggested local command:

```bash
make athena-export EXTRA_ARGS="--s3-prefix s3://YOUR-BUCKET/spotify-athena --database-name spotify_taste_os"
aws s3 sync outputs/aws_athena_bundle s3://YOUR-BUCKET/spotify-athena
```

## Cost Notes

This pattern stays cheap because:

- S3 storage at this project size is pennies
- Athena queries bill by scanned data, so Parquet keeps costs low
- Glue Data Catalog usage should stay inside or near free-tier territory for this footprint

To keep the bill low:

- keep the curated tables in Parquet
- avoid Glue crawlers until you truly need them
- use an Athena workgroup bytes-scanned limit
- keep raw JSON private and query the curated Parquet tables instead

## If You Want More Later

Once the base import is working, the next worthwhile upgrades are:

- a scheduled export after the daily and weekly training jobs
- an Athena view layer for public-insights style reporting
- optional QuickSight dashboards on top of the curated tables
