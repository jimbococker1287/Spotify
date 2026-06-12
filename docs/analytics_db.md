# Analytics Warehouse

This project now has a local-first analytics stack with two aligned layers:

- DuckDB SQL file: `outputs/analytics/spotify_analytics.duckdb`
- Local warehouse: `outputs/analytics/warehouse`

The DuckDB file is the interactive query surface. The warehouse is the local data-engineering layer with bronze, silver, and gold Parquet assets plus lineage and verification artifacts.

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

Both flows now run a post-build verification step. The warehouse verifies its written Parquet assets against the manifest, and the DuckDB refresh verifies key tables/views against warehouse metadata persisted inside DuckDB.

## Warehouse Layout

The warehouse writes:

- `outputs/analytics/warehouse/bronze/*.parquet`
- `outputs/analytics/warehouse/silver/*.parquet`
- `outputs/analytics/warehouse/gold/*.parquet`
- `outputs/analytics/warehouse/warehouse_manifest.json`
- `outputs/analytics/warehouse/warehouse_manifest.md`
- `outputs/analytics/warehouse/warehouse_lineage.json`
- `outputs/analytics/warehouse/warehouse_lineage.md`
- `outputs/analytics/warehouse/warehouse_verification.json`
- `outputs/analytics/warehouse/warehouse_verification.md`

The manifest now records per-asset schema metadata alongside row counts, lineage, refresh status, branch-backed source fingerprints, and quality metadata so the warehouse and DuckDB layers can compare the same artifact contract.

`warehouse_lineage.json` and `warehouse_lineage.md` are the local lineage + quality report. They map bronze inputs into silver tables and gold marts, then summarize:

- Empty assets that may need upstream data.
- Built, rebuilt, and reused assets from the content-hash refresh policy.
- Row-count anomalies such as assets that dropped to zero rows or downstream tables that are empty while upstream inputs are populated.
- Branch-backed artifact freshness based on local source availability and freshness/staleness fields in creator-market and research-platform assets.

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
- `creator_market_scene_pulse`
- `creator_market_opportunity_lane_atlas`
- `creator_market_migration_network`
- `creator_market_seed_bridge_atlas`
- `creator_market_release_whitespace_atlas`
- `creator_market_brief_snapshot`
- `creator_market_manifest_snapshot`
- `research_platform_run_registry`
- `research_platform_benchmark_lock_atlas`
- `research_platform_claim_registry`
- `research_platform_maturity_snapshot`
- `research_platform_manifest_snapshot`
- `scope_expansion_scorecard`
- `scope_expansion_implementation_queue`
- `scope_expansion_strategy_cards`
- `scope_expansion_manifest_snapshot`

### Silver

Curated modeling and business-analysis tables:

- `listener_daily_activity`
- `model_run_summary`
- `ops_review_snapshot`
- `creator_report_family_summary`
- `creator_market_scene_summary`
- `research_platform_status_summary`
- `scope_expansion_branch_health`

### Gold

Semantic marts for decision work:

- `mart_run_quality`
- `mart_model_registry`
- `mart_ops_overview`
- `mart_creator_opportunities`
- `mart_creator_scene_pressure`
- `mart_creator_market_watchlist`
- `mart_research_platform_status`
- `mart_research_claim_watchlist`
- `mart_scope_expansion_health`

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
- `creator_market_scene_summary`
- `research_platform_status_summary`
- `scope_expansion_branch_health`
- `mart_run_quality`
- `mart_model_registry`
- `mart_ops_overview`
- `mart_creator_opportunities`
- `mart_creator_scene_pressure`
- `mart_creator_market_watchlist`
- `mart_research_platform_status`
- `mart_research_claim_watchlist`
- `mart_scope_expansion_health`
- `warehouse_asset_manifest`
- `warehouse_asset_columns`
- `warehouse_consistency_checks`
- `warehouse_consistency_summary`

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
- `creator_market_priority_now`
- `latest_research_platform_status`
- `research_platform_blocked_claims`
- `scope_expansion_priority_queue`

## Example Queries

Current operating snapshot:

```sql
select *
from mart_ops_overview;
```

Latest consistency status:

```sql
select object_name, row_count_match, column_match, logical_type_match, status
from warehouse_consistency_checks
order by object_name;
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

Daily global-versus-U.S. public-listening similarity:

```sql
select
  listening_date,
  dimension,
  reference_alignment,
  closer_scope,
  global_similarity,
  united_states_similarity,
  global_duration_share_on_public_top,
  united_states_duration_share_on_public_top
from public_listening_daily_trend
order by listening_date desc, dimension;
```

The 2025 reference is date-aligned from January 1 through November 15, 2025. Other dates are labeled as historical or post-window projections against that fixed benchmark.

Flagged daily similarity anomalies:

```sql
select listening_date, dimension, closer_scope, similarity_anomaly_flag
from public_listening_similarity_anomalies
order by listening_date desc, dimension;
```

Daily explanations:

```sql
select listening_date, headline, concise_summary, caveats
from mart_public_listening_daily_narratives
order by listening_date desc;
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

Creator-market priority watch:

```sql
select artist_name, scene_name, market_priority_score, scene_strategy_posture
from creator_market_priority_now
order by market_priority_score desc nulls last
limit 25;
```

Research-platform status:

```sql
select anchor_run_id, anchor_research_stage, status_posture, submission_status, top_blocker
from latest_research_platform_status;
```

Blocked research claims:

```sql
select claim_key, title, next_gate, watchlist_score
from research_platform_blocked_claims
order by watchlist_score desc nulls last;
```

Scope-expansion branch queue:

```sql
select branch_key, branch_posture, development_mode, queue_rank, next_initiative, validation_command
from scope_expansion_priority_queue
order by queue_rank asc nulls last, risk_reduction_score desc nulls last;
```

## Recommended Workflow

1. Run `make analytics-db` after a major training or reporting refresh.
2. Open `outputs/analytics/spotify_analytics.duckdb` in DuckDB tooling or query it directly from Python.
3. Use the warehouse Parquet assets when you want a stable, file-based local data pipeline.
4. Treat the gold marts as the default starting point for notebooks, dashboards, and experimental branch work, including the creator-market, research-platform, and scope-expansion branch portfolio.
5. Run `make scope-expansion-lab` after the warehouse and branch artifacts are fresh to generate a four-branch scorecard and implementation queue under `outputs/analysis/scope_expansion/`.
6. Run `make analytics-db` again after `make scope-expansion-lab` when you want `mart_scope_expansion_health` and `scope_expansion_priority_queue` refreshed inside DuckDB.
