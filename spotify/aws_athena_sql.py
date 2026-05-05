from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AthenaTableSpec:
    name: str
    columns: tuple[tuple[str, str], ...]
    partition_columns: tuple[tuple[str, str], ...] = ()


ATHENA_TABLE_SPECS: tuple[AthenaTableSpec, ...] = (
    AthenaTableSpec(
        name="raw_streaming_history",
        columns=(
            ("played_at", "timestamp"),
            ("platform", "string"),
            ("ms_played", "bigint"),
            ("conn_country", "string"),
            ("master_metadata_track_name", "string"),
            ("master_metadata_album_artist_name", "string"),
            ("master_metadata_album_album_name", "string"),
            ("spotify_track_uri", "string"),
            ("episode_name", "string"),
            ("episode_show_name", "string"),
            ("spotify_episode_uri", "string"),
            ("audiobook_title", "string"),
            ("audiobook_uri", "string"),
            ("audiobook_chapter_uri", "string"),
            ("audiobook_chapter_title", "string"),
            ("reason_start", "string"),
            ("reason_end", "string"),
            ("shuffle", "boolean"),
            ("skipped", "boolean"),
            ("offline", "boolean"),
            ("offline_timestamp", "bigint"),
            ("incognito_mode", "boolean"),
            ("content_type", "string"),
        ),
        partition_columns=(("year", "int"), ("month", "int")),
    ),
    AthenaTableSpec(
        name="experiment_history",
        columns=(
            ("event_timestamp", "timestamp"),
            ("run_id", "string"),
            ("run_name", "string"),
            ("profile", "string"),
            ("model_name", "string"),
            ("model_type", "string"),
            ("model_family", "string"),
            ("val_top1", "double"),
            ("val_top5", "double"),
            ("test_top1", "double"),
            ("test_top5", "double"),
            ("fit_seconds", "double"),
            ("epochs", "int"),
            ("data_records", "bigint"),
        ),
    ),
    AthenaTableSpec(
        name="backtest_history",
        columns=(
            ("event_timestamp", "timestamp"),
            ("run_id", "string"),
            ("run_name", "string"),
            ("profile", "string"),
            ("model_name", "string"),
            ("model_family", "string"),
            ("fold", "int"),
            ("train_rows", "bigint"),
            ("test_rows", "bigint"),
            ("fit_seconds", "double"),
            ("top1", "double"),
            ("top5", "double"),
        ),
    ),
    AthenaTableSpec(
        name="optuna_history",
        columns=(
            ("event_timestamp", "timestamp"),
            ("run_id", "string"),
            ("run_name", "string"),
            ("profile", "string"),
            ("model_name", "string"),
            ("base_model_name", "string"),
            ("n_trials", "int"),
            ("val_top1", "double"),
            ("test_top1", "double"),
            ("fit_seconds", "double"),
            ("best_params_json", "string"),
        ),
    ),
    AthenaTableSpec(
        name="benchmark_history",
        columns=(
            ("event_timestamp", "timestamp"),
            ("benchmark_id", "string"),
            ("model_name", "string"),
            ("model_type", "string"),
            ("model_family", "string"),
            ("runs", "int"),
            ("val_top1_mean", "double"),
            ("val_top1_std", "double"),
            ("val_top1_ci95", "double"),
            ("test_top1_mean", "double"),
            ("test_top1_std", "double"),
            ("test_top1_ci95", "double"),
        ),
    ),
    AthenaTableSpec(
        name="run_manifests",
        columns=(
            ("run_id", "string"),
            ("run_name", "string"),
            ("profile", "string"),
            ("run_timestamp", "timestamp"),
            ("data_records", "bigint"),
            ("num_artists", "bigint"),
            ("num_context_features", "bigint"),
            ("enable_retrieval_stack", "boolean"),
            ("enable_self_supervised_pretraining", "boolean"),
            ("enable_friction_analysis", "boolean"),
            ("enable_moonshot_lab", "boolean"),
            ("retrieval_candidate_k", "bigint"),
            ("enable_mlflow", "boolean"),
            ("enable_optuna", "boolean"),
            ("optuna_trials", "bigint"),
            ("enable_temporal_backtest", "boolean"),
            ("temporal_backtest_folds", "bigint"),
            ("temporal_backtest_adaptation_mode", "string"),
            ("backtest_rows", "bigint"),
            ("optuna_rows", "bigint"),
            ("cache_enabled", "boolean"),
            ("cache_hit", "boolean"),
            ("cache_fingerprint", "string"),
            ("cache_source_file_count", "bigint"),
            ("champion_gate_status", "string"),
            ("champion_gate_promoted", "boolean"),
            ("champion_gate_metric_source", "string"),
            ("champion_gate_threshold", "double"),
            ("champion_gate_regression", "double"),
            ("champion_gate_champion_run_id", "string"),
            ("champion_gate_champion_model_name", "string"),
            ("champion_gate_champion_score", "double"),
            ("champion_gate_challenger_model_name", "string"),
            ("champion_gate_challenger_score", "double"),
            ("champion_alias_updated", "boolean"),
            ("champion_alias_model_name", "string"),
            ("champion_alias_model_type", "string"),
            ("artifact_cleanup_mode", "string"),
            ("artifact_cleanup_status", "string"),
            ("artifact_cleanup_selected_model_name", "string"),
            ("artifact_cleanup_freed_bytes", "bigint"),
            ("artifact_retention_keep_last_full_runs", "bigint"),
            ("run_dir", "string"),
            ("deep_models_json", "string"),
            ("classical_models_json", "string"),
            ("optuna_models_json", "string"),
            ("temporal_backtest_models_json", "string"),
            ("cache_json", "string"),
            ("champion_gate_json", "string"),
            ("champion_alias_json", "string"),
            ("artifact_cleanup_json", "string"),
            ("artifact_retention_json", "string"),
            ("mlflow_artifact_cleanup_json", "string"),
            ("raw_json", "string"),
        ),
    ),
    AthenaTableSpec(
        name="run_results",
        columns=(
            ("run_id", "string"),
            ("run_dir", "string"),
            ("model_name", "string"),
            ("model_type", "string"),
            ("model_family", "string"),
            ("base_model_name", "string"),
            ("val_top1", "double"),
            ("val_top5", "double"),
            ("val_ndcg_at5", "double"),
            ("val_mrr_at5", "double"),
            ("val_coverage_at5", "double"),
            ("val_diversity_at5", "double"),
            ("test_top1", "double"),
            ("test_top5", "double"),
            ("test_ndcg_at5", "double"),
            ("test_mrr_at5", "double"),
            ("test_coverage_at5", "double"),
            ("test_diversity_at5", "double"),
            ("fit_seconds", "double"),
            ("epochs", "int"),
            ("prediction_bundle_path", "string"),
            ("estimator_artifact_path", "string"),
            ("n_trials", "int"),
            ("ensemble_members_json", "string"),
            ("best_params_json", "string"),
            ("raw_json", "string"),
        ),
    ),
    AthenaTableSpec(
        name="listener_daily_activity",
        columns=(
            ("played_date", "string"),
            ("total_streams", "bigint"),
            ("total_ms_played", "bigint"),
            ("unique_artists", "bigint"),
            ("unique_tracks", "bigint"),
            ("skip_rate", "double"),
            ("shuffle_rate", "double"),
            ("offline_rate", "double"),
            ("primary_platform", "string"),
            ("track_stream_share", "double"),
        ),
    ),
    AthenaTableSpec(
        name="model_run_summary",
        columns=(
            ("run_id", "string"),
            ("run_name", "string"),
            ("profile", "string"),
            ("run_timestamp", "timestamp"),
            ("model_name", "string"),
            ("model_type", "string"),
            ("model_family", "string"),
            ("base_model_name", "string"),
            ("val_top1", "double"),
            ("test_top1", "double"),
            ("fit_seconds", "double"),
            ("epochs", "bigint"),
            ("mean_backtest_top1", "double"),
            ("backtest_folds", "bigint"),
            ("promoted", "boolean"),
            ("champion_gate_status", "string"),
            ("champion_gate_metric_source", "string"),
            ("champion_alias_model_name", "string"),
            ("champion_alias_model_type", "string"),
            ("is_serving_alias", "boolean"),
            ("val_rank_within_run", "double"),
            ("test_rank_within_run", "double"),
            ("data_records", "bigint"),
        ),
    ),
    AthenaTableSpec(
        name="ops_review_snapshot",
        columns=(
            ("generated_at", "timestamp"),
            ("selected_run_id", "string"),
            ("latest_run_id", "string"),
            ("latest_run_best_model_name", "string"),
            ("latest_run_serving_model_name", "string"),
            ("ops_health_status", "string"),
            ("operating_rhythm_status", "string"),
            ("recommended_run_command", "string"),
            ("review_action_count", "bigint"),
            ("high_priority_actions", "bigint"),
            ("medium_priority_actions", "bigint"),
            ("history_points", "bigint"),
            ("current_target_drift_jsd", "double"),
            ("current_stress_benchmark_skip_risk", "double"),
            ("current_selective_risk", "double"),
            ("latest_fast_cadence_status", "string"),
            ("latest_full_cadence_status", "string"),
        ),
    ),
    AthenaTableSpec(
        name="creator_report_family_summary",
        columns=(
            ("report_family_id", "string"),
            ("seed_group_slug", "string"),
            ("ranking_rows", "bigint"),
            ("scene_rows", "bigint"),
            ("scene_seed_rows", "bigint"),
            ("priority_now_count", "bigint"),
            ("avg_opportunity_score", "double"),
            ("top_opportunity_artist", "string"),
            ("top_scene_name", "string"),
            ("primary_report_path", "string"),
            ("artifact_index_path", "string"),
            ("backfilled_artifact_index_at", "timestamp"),
        ),
    ),
    AthenaTableSpec(
        name="mart_run_quality",
        columns=(
            ("run_id", "string"),
            ("run_name", "string"),
            ("profile", "string"),
            ("run_timestamp", "timestamp"),
            ("promoted", "boolean"),
            ("champion_gate_status", "string"),
            ("champion_gate_metric_source", "string"),
            ("best_model_name", "string"),
            ("best_model_type", "string"),
            ("best_val_top1", "double"),
            ("best_test_top1", "double"),
            ("best_mean_backtest_top1", "double"),
            ("serving_model_name", "string"),
            ("serving_model_type", "string"),
            ("models_evaluated", "bigint"),
            ("data_records", "bigint"),
        ),
    ),
    AthenaTableSpec(
        name="mart_model_registry",
        columns=(
            ("model_name", "string"),
            ("model_type", "string"),
            ("model_family", "string"),
            ("latest_run_id", "string"),
            ("latest_profile", "string"),
            ("latest_run_timestamp", "timestamp"),
            ("runs", "bigint"),
            ("promoted_runs", "bigint"),
            ("mean_val_top1", "double"),
            ("mean_test_top1", "double"),
            ("max_test_top1", "double"),
            ("mean_backtest_top1", "double"),
            ("mean_fit_seconds", "double"),
        ),
    ),
    AthenaTableSpec(
        name="mart_ops_overview",
        columns=(
            ("selected_run_id", "string"),
            ("latest_run_id", "string"),
            ("latest_run_best_model_name", "string"),
            ("latest_run_serving_model_name", "string"),
            ("ops_health_status", "string"),
            ("operating_rhythm_status", "string"),
            ("review_action_count", "bigint"),
            ("high_priority_actions", "bigint"),
            ("medium_priority_actions", "bigint"),
            ("history_points", "bigint"),
            ("history_window_start", "timestamp"),
            ("history_window_end", "timestamp"),
            ("mean_target_drift_jsd", "double"),
            ("mean_stress_benchmark_skip_risk", "double"),
            ("mean_selective_risk", "double"),
            ("recommended_run_command", "string"),
        ),
    ),
    AthenaTableSpec(
        name="mart_creator_opportunities",
        columns=(
            ("artist_name", "string"),
            ("appearance_count", "bigint"),
            ("family_count", "bigint"),
            ("priority_now_count", "bigint"),
            ("mean_opportunity_score", "double"),
            ("max_opportunity_score", "double"),
            ("best_opportunity_rank", "double"),
            ("top_scene_name", "string"),
            ("top_primary_driver", "string"),
            ("example_seed_bridges", "string"),
            ("example_why_now", "string"),
        ),
    ),
    AthenaTableSpec(
        name="mart_creator_scene_pressure",
        columns=(
            ("scene_name", "string"),
            ("family_count", "bigint"),
            ("total_priority_now_count", "bigint"),
            ("mean_opportunity_score", "double"),
            ("max_top_opportunity_score", "double"),
            ("mean_scene_local_play_share", "double"),
            ("mean_scene_label_concentration", "double"),
            ("mean_scene_release_pressure", "double"),
            ("top_opportunity_artist_example", "string"),
            ("top_seed_artists_example", "string"),
        ),
    ),
)

ATHENA_TABLE_SPEC_LOOKUP: dict[str, AthenaTableSpec] = {spec.name: spec for spec in ATHENA_TABLE_SPECS}


def normalize_s3_prefix(s3_prefix: str) -> str:
    return str(s3_prefix).rstrip("/")


def _render_column_block(columns: tuple[tuple[str, str], ...]) -> str:
    return ",\n".join(f"  {column_name} {column_type}" for column_name, column_type in columns)


def _render_create_table_sql(*, spec: AthenaTableSpec, normalized_prefix: str) -> str:
    statement_lines = [
        f"CREATE EXTERNAL TABLE IF NOT EXISTS {spec.name} (",
        _render_column_block(spec.columns),
        ")",
    ]
    if spec.partition_columns:
        statement_lines.append(
            "PARTITIONED BY (" + ", ".join(f"{column_name} {column_type}" for column_name, column_type in spec.partition_columns) + ")"
        )
    statement_lines.extend(
        [
            "STORED AS PARQUET",
            f"LOCATION '{normalized_prefix}/curated/{spec.name}/';",
        ]
    )
    return "\n".join(statement_lines)


def build_athena_sql(*, database_name: str, s3_prefix: str) -> str:
    normalized_prefix = normalize_s3_prefix(s3_prefix)
    statements: list[str] = [
        f"CREATE DATABASE IF NOT EXISTS {database_name};",
        f"USE {database_name};",
        "",
    ]
    for index, spec in enumerate(ATHENA_TABLE_SPECS):
        statements.append(_render_create_table_sql(spec=spec, normalized_prefix=normalized_prefix))
        if index < len(ATHENA_TABLE_SPECS) - 1:
            statements.append("")

    statements.extend(
        [
            "",
            "MSCK REPAIR TABLE raw_streaming_history;",
            "",
            "CREATE OR REPLACE VIEW latest_run_results AS",
            "SELECT rr.*",
            "FROM run_results rr",
            "JOIN (",
            "  SELECT run_id",
            "  FROM run_manifests",
            "  ORDER BY run_timestamp DESC, run_id DESC",
            "  LIMIT 1",
            ") latest",
            "  ON rr.run_id = latest.run_id;",
            "",
            "CREATE OR REPLACE VIEW best_models_by_profile AS",
            "SELECT",
            "  profile,",
            "  model_name,",
            "  model_type,",
            "  AVG(val_top1) AS mean_val_top1,",
            "  AVG(test_top1) AS mean_test_top1,",
            "  COUNT(*) AS runs",
            "FROM experiment_history",
            "GROUP BY profile, model_name, model_type;",
            "",
            "CREATE OR REPLACE VIEW backtest_model_summary AS",
            "SELECT",
            "  profile,",
            "  model_name,",
            "  model_family,",
            "  AVG(top1) AS mean_backtest_top1,",
            "  STDDEV_SAMP(top1) AS std_backtest_top1,",
            "  COUNT(*) AS folds",
            "FROM backtest_history",
            "GROUP BY profile, model_name, model_family;",
            "",
            "CREATE OR REPLACE VIEW champion_runs AS",
            "SELECT",
            "  run_id,",
            "  run_name,",
            "  profile,",
            "  run_timestamp,",
            "  champion_gate_metric_source,",
            "  champion_alias_model_name,",
            "  champion_alias_model_type,",
            "  champion_gate_champion_run_id,",
            "  champion_gate_champion_model_name,",
            "  champion_gate_challenger_model_name,",
            "  champion_gate_threshold,",
            "  champion_gate_regression",
            "FROM run_manifests",
            "WHERE champion_gate_promoted = TRUE;",
            "",
            "CREATE OR REPLACE VIEW latest_ops_overview AS",
            "SELECT *",
            "FROM mart_ops_overview;",
            "",
            "CREATE OR REPLACE VIEW creator_priority_now AS",
            "SELECT *",
            "FROM mart_creator_opportunities",
            "WHERE COALESCE(priority_now_count, 0) > 0;",
        ]
    )
    return "\n".join(statements)


def build_athena_queries(*, database_name: str) -> str:
    return f"""USE {database_name};

SELECT master_metadata_album_artist_name, COUNT(*) AS plays
FROM raw_streaming_history
GROUP BY 1
ORDER BY plays DESC
LIMIT 25;

SELECT played_date, total_streams, unique_artists, skip_rate
FROM listener_daily_activity
ORDER BY played_date DESC
LIMIT 30;

SELECT run_id, best_model_name, best_test_top1, serving_model_name, promoted
FROM mart_run_quality
ORDER BY run_timestamp DESC;

SELECT model_name, model_type, runs, mean_test_top1, mean_backtest_top1
FROM mart_model_registry
ORDER BY mean_test_top1 DESC;

SELECT selected_run_id, latest_run_id, ops_health_status, operating_rhythm_status, recommended_run_command
FROM latest_ops_overview;

SELECT artist_name, max_opportunity_score, priority_now_count, top_scene_name
FROM creator_priority_now
ORDER BY max_opportunity_score DESC
LIMIT 25;
"""
