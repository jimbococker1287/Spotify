# Analytics Warehouse

This project now has a local-first analytics stack with two aligned layers:

- DuckDB SQL file: `outputs/analytics/spotify_analytics.duckdb`
- Local warehouse: `outputs/analytics/warehouse`

The DuckDB file is the interactive query surface. The warehouse is the local data-engineering layer with bronze, silver, and gold Parquet assets plus lineage artifacts.

## Refresh Commands

Refresh both the DuckDB database and the warehouse:

```bash
make analytics-db
```

Build only the warehouse:

```bash
make analytics-warehouse
```

Direct script usage:

```bash
python scripts/build_analytics_db.py
python scripts/build_analytics_db.py --warehouse-only
```

## Warehouse Layout

The warehouse writes:

- `outputs/analytics/warehouse/bronze/*.parquet`
- `outputs/analytics/warehouse/silver/*.parquet`
- `outputs/analytics/warehouse/gold/*.parquet`
- `outputs/analytics/warehouse/warehouse_manifest.json`
- `outputs/analytics/warehouse/warehouse_manifest.md`

### Bronze

Raw-ish prepared assets for local analytics:

- `raw_streaming_history`
- `experiment_history`
- `backtest_history`
- `benchmark_history`
- `optuna_history`
- `run_manifests`
- `run_results`
- `robustness_summary`
- `policy_summary`
- `moonshot_summary`
- `control_room_snapshot`
- `control_room_review_actions`
- `control_room_history`
- `creator_report_families`
- `creator_ranking_opportunities`
- `creator_scene_summary`
- `creator_scene_seed_summary`

### Silver

Curated modeling and business-analysis tables:

- `listener_daily_activity`
- `model_run_summary`
- `ops_review_snapshot`
- `creator_report_family_summary`

### Gold

Semantic marts for decision work:

- `mart_run_quality`
- `mart_model_registry`
- `mart_ops_overview`
- `mart_creator_opportunities`
- `mart_creator_scene_pressure`

## DuckDB Surface

Useful base tables:

- `raw_streaming_history`
- `run_manifests`
- `run_results`
- `control_room_snapshot`
- `control_room_history`
- `creator_ranking_opportunities`
- `creator_scene_summary`
- `listener_daily_activity`
- `model_run_summary`
- `mart_run_quality`
- `mart_model_registry`
- `mart_ops_overview`
- `mart_creator_opportunities`
- `mart_creator_scene_pressure`

Useful views:

- `latest_run_results`
- `best_models_by_profile`
- `backtest_model_summary`
- `champion_runs`
- `robustness_model_summary`
- `policy_model_summary`
- `benchmark_model_summary`
- `moonshot_run_summary`
- `latest_ops_overview`
- `creator_priority_now`

## Example Queries

Current operating snapshot:

```sql
select *
from mart_ops_overview;
```

Best model per run:

```sql
select run_id, best_model_name, best_test_top1, serving_model_name, promoted
from mart_run_quality
order by run_timestamp desc;
```

Reusable model registry:

```sql
select model_name, model_type, runs, mean_test_top1, mean_backtest_top1
from mart_model_registry
order by mean_test_top1 desc nulls last;
```

Listener behavior by day:

```sql
select played_date, total_streams, unique_artists, skip_rate, primary_platform
from listener_daily_activity
order by played_date desc
limit 30;
```

Creator whitespace:

```sql
select artist_name, max_opportunity_score, family_count, top_scene_name
from mart_creator_opportunities
order by max_opportunity_score desc nulls last
limit 25;
```

Scene pressure:

```sql
select scene_name, mean_opportunity_score, total_priority_now_count, mean_scene_local_play_share
from mart_creator_scene_pressure
order by mean_opportunity_score desc nulls last;
```

## Recommended Workflow

1. Run `make analytics-db` after a major training or reporting refresh.
2. Open `outputs/analytics/spotify_analytics.duckdb` in DuckDB tooling or query it directly from Python.
3. Use the warehouse Parquet assets when you want a stable, file-based local data pipeline.
4. Treat the gold marts as the default starting point for notebooks, dashboards, and experimental branch work.
