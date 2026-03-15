# Analytics Database

This project now refreshes a DuckDB analytics file at:

`outputs/analytics/spotify_analytics.duckdb`

Refresh it manually with:

```bash
make analytics-db
```

Recommended VS Code setup:

1. Install the `DuckDB` extension.
2. Open `outputs/analytics/spotify_analytics.duckdb`.
3. Run SQL queries directly against the project data.

Useful tables:

- `raw_streaming_history`
- `experiment_history`
- `backtest_history`
- `optuna_history`
- `run_manifests`
- `run_results`

Useful views:

- `latest_run_results`
- `best_models_by_profile`
- `backtest_model_summary`
- `champion_runs`

Example queries:

```sql
select model_name, model_type, val_top1, test_top1
from latest_run_results
order by val_top1 desc;
```

```sql
select profile, model_name, mean_backtest_top1, folds
from backtest_model_summary
order by profile, mean_backtest_top1 desc;
```

```sql
select master_metadata_album_artist_name, count(*) as plays
from raw_streaming_history
group by 1
order by plays desc
limit 25;
```
