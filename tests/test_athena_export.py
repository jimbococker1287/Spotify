from __future__ import annotations

import json
import logging
from pathlib import Path

import duckdb

from spotify.aws_athena import build_athena_sql, export_athena_bundle


def test_build_athena_sql_includes_expected_locations() -> None:
    sql = build_athena_sql(
        database_name="spotify_taste_os",
        s3_prefix="s3://demo-bucket/spotify-athena",
    )

    assert "CREATE DATABASE IF NOT EXISTS spotify_taste_os;" in sql
    assert "LOCATION 's3://demo-bucket/spotify-athena/curated/raw_streaming_history/'" in sql
    assert "LOCATION 's3://demo-bucket/spotify-athena/curated/mart_run_quality/'" in sql
    assert "CREATE OR REPLACE VIEW latest_run_results AS" in sql
    assert "CREATE OR REPLACE VIEW latest_ops_overview AS" in sql
    assert "MSCK REPAIR TABLE raw_streaming_history;" in sql


def _write_minimal_control_room(output_dir: Path, *, run_id: str) -> None:
    analytics_dir = output_dir / "analytics"
    analytics_dir.mkdir(parents=True, exist_ok=True)
    (analytics_dir / "control_room.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-03-25T01:00:00+00:00",
                "latest_run": {
                    "run_id": run_id,
                    "profile": "full",
                    "best_model_name": "mlp",
                    "best_model_type": "classical",
                    "best_model_test_top1": 0.2884,
                    "serving_model_name": "mlp",
                    "serving_model_type": "classical",
                    "promoted": False,
                },
                "run_selection": {
                    "selected_run": {
                        "run_id": run_id,
                        "profile": "full",
                        "best_model_name": "mlp",
                        "best_model_type": "classical",
                    },
                    "selection_reason": "latest full run",
                },
                "ops_health": {
                    "status": "attention",
                    "headline": "Review cadence and drift.",
                },
                "operating_rhythm": {
                    "overall_status": "stale",
                    "recommended_run_command": "make schedule-run MODE=fast",
                    "recommended_run_reason": "Restore cadence.",
                },
                "safety": {
                    "test_jsd_target_drift": 0.218,
                    "test_selective_risk": 0.372,
                    "test_abstention_rate": 0.241,
                    "test_accepted_rate": 0.759,
                    "repeat_from_prev_new_gap": 0.096,
                },
                "qoe": {
                    "stress_benchmark_skip_risk": 0.591,
                    "stress_benchmark_scenario": "evening_drift",
                    "stress_benchmark_policy_name": "safe_global",
                },
                "review_actions": [
                    {
                        "priority": "medium",
                        "area": "cadence",
                        "title": "Restore cadence",
                        "detail": "Run the fast lane.",
                        "inspect": ["outputs/analytics/control_room_history.csv"],
                    }
                ],
                "next_bets": [{"title": "Investigate drift"}],
            }
        ),
        encoding="utf-8",
    )
    (analytics_dir / "control_room_history.csv").write_text(
        "\n".join(
            [
                "generated_at,run_id,run_timestamp,profile,promoted,promotion_status,best_model_name,best_model_type,best_model_val_top1,best_model_test_top1,champion_gate_regression,target_drift_jsd,test_ece,test_selective_risk,test_abstention_rate,robustness_gap,stress_skip_risk,review_action_count,high_priority_review_actions,medium_priority_review_actions,review_action_areas,baseline_run_id,next_bet_count,ops_coverage_ratio,available_summary_count,expected_summary_count,operating_status,fast_cadence_status,full_cadence_status,async_handoff_status,recommended_run_command,ops_health_status,operational_high_priority_review_actions,strategic_high_priority_review_actions,test_accepted_rate,conformal_operating_threshold,repeat_from_prev_new_gap,stress_benchmark_skip_risk",
                f"2026-03-25T01:00:00+00:00,{run_id},2026-03-25T00:45:06,full,0,fail,mlp,classical,0.3741,0.2884,0.019,0.218,0.08,0.372,0.241,0.096,0.593,1,0,1,cadence,old_run,1,1.0,6,6,stale,stale,healthy,attention,make schedule-run MODE=fast,attention,0,0,0.759,0.24,0.096,0.591",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_minimal_creator_family(output_dir: Path) -> None:
    base_dir = output_dir / "analysis" / "public_spotify" / "creator_label_intelligence"
    base_dir.mkdir(parents=True, exist_ok=True)
    family_id = "creator_label_intelligence_seed-a-seed-b"
    (base_dir / f"{family_id}.md").write_text("# Primary report\n", encoding="utf-8")
    (base_dir / f"{family_id}_report_family.md").write_text("# Report family\n", encoding="utf-8")
    (base_dir / f"{family_id}_report_family.json").write_text(
        json.dumps(
            {
                "primary_report": str((base_dir / f"{family_id}.md").resolve()),
                "artifact_index_markdown": str((base_dir / f"{family_id}_report_family.md").resolve()),
                "backfilled_artifact_index_at": "2026-03-25T01:05:00+00:00",
                "comparison_view_markdown": {
                    "ranking_comparison": str((base_dir / f"{family_id}_ranking_comparison.md").resolve()),
                    "scene_comparison": str((base_dir / f"{family_id}_scene_comparison.md").resolve()),
                    "scene_seed_comparison": str((base_dir / f"{family_id}_scene_seed_comparison.md").resolve()),
                    "seed_comparison": str((base_dir / f"{family_id}_seed_comparison.md").resolve()),
                },
                "comparison_view_csv": {
                    "ranking_comparison": str((base_dir / f"{family_id}_ranking_comparison.csv").resolve()),
                    "scene_comparison": str((base_dir / f"{family_id}_scene_comparison.csv").resolve()),
                    "scene_seed_comparison": str((base_dir / f"{family_id}_scene_seed_comparison.csv").resolve()),
                    "seed_comparison": str((base_dir / f"{family_id}_seed_comparison.csv").resolve()),
                },
            }
        ),
        encoding="utf-8",
    )
    (base_dir / f"{family_id}_ranking_comparison.csv").write_text(
        "\n".join(
            [
                "artist_name,opportunity_score,opportunity_rank,opportunity_band,scene_name,primary_driver,seed_bridges,why_now",
                'Artist X,0.41,1,priority_now,scene-1,seed_adjacency,"[""Seed A""]",Strong adjacency.',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (base_dir / f"{family_id}_scene_comparison.csv").write_text(
        "\n".join(
            [
                "scene_name,avg_opportunity_score,priority_now_count,scene_local_play_share,scene_label_concentration,scene_release_pressure,top_opportunity_artist,top_opportunity_score,top_seed_artists",
                'scene-1,0.41,1,0.33,0.10,0.20,Artist X,0.41,"[""Seed A""]"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (base_dir / f"{family_id}_scene_seed_comparison.csv").write_text(
        "\n".join(
            [
                "scene_name,seed_artist,avg_opportunity_score,top_opportunity_artist,top_opportunity_score",
                "scene-1,Seed A,0.41,Artist X,0.41",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_export_athena_bundle_writes_expected_bundle(tmp_path: Path) -> None:
    data_dir = tmp_path / "data" / "raw"
    output_dir = tmp_path / "outputs"
    export_dir = tmp_path / "athena_bundle"

    streaming_dir = data_dir / "Spotify Extended Streaming History"
    account_dir = data_dir / "Spotify Account Data"
    streaming_dir.mkdir(parents=True)
    account_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    (output_dir / "history").mkdir(parents=True)

    streaming_payload = [
        {
            "ts": "2024-07-01T08:03:54Z",
            "platform": "ios",
            "ms_played": 198973,
            "conn_country": "US",
            "ip_addr": "10.0.0.1",
            "master_metadata_track_name": "God's Plan",
            "master_metadata_album_artist_name": "Drake",
            "master_metadata_album_album_name": "Scorpion",
            "spotify_track_uri": "spotify:track:6DCZcSspjsKoFjzjrWoCdn",
            "episode_name": None,
            "episode_show_name": None,
            "spotify_episode_uri": None,
            "audiobook_title": None,
            "audiobook_uri": None,
            "audiobook_chapter_uri": None,
            "audiobook_chapter_title": None,
            "reason_start": "trackdone",
            "reason_end": "trackdone",
            "shuffle": False,
            "skipped": False,
            "offline": False,
            "offline_timestamp": None,
            "incognito_mode": False,
        }
    ]
    (streaming_dir / "Streaming_History_Audio_2024_0.json").write_text(
        json.dumps(streaming_payload),
        encoding="utf-8",
    )
    (account_dir / "Userdata.json").write_text("{}", encoding="utf-8")

    (output_dir / "history" / "experiment_history.csv").write_text(
        "timestamp,run_id,run_name,profile,model_name,model_type,model_family,val_top1,val_top5,test_top1,test_top5,fit_seconds,epochs,data_records\n"
        "2026-03-05T21:46:16,run_1,fast-run,fast,mlp,classical,shallow_neural,0.31,0.66,0.28,0.59,3.5,4,1000\n",
        encoding="utf-8",
    )
    (output_dir / "history" / "backtest_history.csv").write_text(
        "timestamp,run_id,run_name,profile,model_name,model_family,fold,train_rows,test_rows,fit_seconds,top1,top5\n"
        "2026-03-05T22:15:27,run_1,fast-run,fast,mlp,shallow_neural,1,400,150,0.01,0.12,0.34\n",
        encoding="utf-8",
    )
    (output_dir / "history" / "optuna_history.csv").write_text(
        "timestamp,run_id,run_name,profile,model_name,base_model_name,n_trials,val_top1,test_top1,fit_seconds,best_params_json\n"
        "2026-03-05T22:17:05,run_1,fast-run,fast,mlp_optuna,mlp,8,0.33,0.29,6.3,\"{\"\"alpha\"\": 0.001}\"\n",
        encoding="utf-8",
    )
    (output_dir / "history" / "benchmark_history.csv").write_text(
        "timestamp,benchmark_id,model_name,model_type,model_family,runs,val_top1_mean,val_top1_std,val_top1_ci95,test_top1_mean,test_top1_std,test_top1_ci95\n"
        "2026-03-06T12:58:10,smokebench,mlp,classical,shallow_neural,1,0.22,0.0,0.0,0.14,0.0,0.0\n",
        encoding="utf-8",
    )

    run_dir = output_dir / "runs" / "20260324_222814_taste-os-full"
    run_dir.mkdir(parents=True)
    run_manifest = {
        "run_id": "20260324_222814_taste-os-full",
        "run_name": "taste-os-full",
        "profile": "full",
        "timestamp": "2026-03-25T00:45:06",
        "data_records": 133381,
        "num_artists": 200,
        "num_context_features": 60,
        "deep_models": ["dense", "lstm"],
        "classical_models": ["logreg", "mlp"],
        "enable_retrieval_stack": True,
        "enable_self_supervised_pretraining": True,
        "enable_friction_analysis": True,
        "enable_moonshot_lab": True,
        "retrieval_candidate_k": 30,
        "enable_mlflow": True,
        "enable_optuna": True,
        "optuna_models": ["logreg", "mlp"],
        "optuna_trials": 18,
        "enable_temporal_backtest": True,
        "temporal_backtest_models": ["logreg", "mlp"],
        "temporal_backtest_folds": 4,
        "temporal_backtest_adaptation_mode": "cold",
        "backtest_rows": 28,
        "optuna_rows": 6,
        "cache": {"enabled": True, "hit": True, "fingerprint": "abc123", "source_file_count": 17},
        "champion_gate": {
            "status": "fail",
            "promoted": False,
            "metric_source": "backtest_top1",
            "threshold": 0.005,
            "regression": 0.019,
            "champion_run_id": "old_run",
            "champion_model_name": "mlp",
            "champion_score": 0.28,
            "challenger_model_name": "extra_trees",
            "challenger_score": 0.26,
        },
        "champion_alias": {"updated": True, "model_name": "mlp", "model_type": "classical"},
        "artifact_cleanup": {
            "mode": "light",
            "status": "completed",
            "selected_model_name": "blended_ensemble",
            "freed_bytes": 12345,
        },
        "artifact_retention": {"keep_last_full_runs": 2},
        "mlflow_artifact_cleanup": {"status": "completed"},
    }
    run_results = [
        {
            "model_name": "mlp",
            "model_type": "classical",
            "model_family": "shallow_neural",
            "val_top1": 0.3741,
            "val_top5": 0.6781,
            "val_ndcg_at5": 0.5324,
            "val_mrr_at5": 0.4842,
            "val_coverage_at5": 0.89,
            "val_diversity_at5": 0.96,
            "test_top1": 0.2884,
            "test_top5": 0.5742,
            "test_ndcg_at5": 0.44,
            "test_mrr_at5": 0.3954,
            "test_coverage_at5": 0.87,
            "test_diversity_at5": 0.95,
            "fit_seconds": 15.69,
            "epochs": 4,
            "prediction_bundle_path": "/tmp/bundle.npz",
        }
    ]
    (run_dir / "run_manifest.json").write_text(json.dumps(run_manifest), encoding="utf-8")
    (run_dir / "run_results.json").write_text(json.dumps(run_results), encoding="utf-8")
    _write_minimal_control_room(output_dir, run_id=run_manifest["run_id"])
    _write_minimal_creator_family(output_dir)

    report = export_athena_bundle(
        data_dir=data_dir,
        output_dir=output_dir,
        export_dir=export_dir,
        include_video=False,
        s3_prefix="s3://demo-bucket/spotify-athena",
        database_name="spotify_taste_os",
        logger=logging.getLogger("test_athena_export"),
    )

    assert report["database_name"] == "spotify_taste_os"
    assert (export_dir / "ddl" / "athena.sql").exists()
    assert (export_dir / "ddl" / "sample_queries.sql").exists()
    assert (export_dir / "raw" / "spotify_streaming_history" / "Streaming_History_Audio_2024_0.json").exists()
    assert (export_dir / "curated" / "experiment_history" / "data.parquet").exists()
    assert (export_dir / "curated" / "run_manifests" / "data.parquet").exists()
    assert (export_dir / "curated" / "listener_daily_activity" / "data.parquet").exists()
    assert (export_dir / "curated" / "model_run_summary" / "data.parquet").exists()
    assert (export_dir / "curated" / "mart_run_quality" / "data.parquet").exists()
    assert (export_dir / "curated" / "mart_ops_overview" / "data.parquet").exists()
    assert (export_dir / "curated" / "mart_creator_opportunities" / "data.parquet").exists()

    raw_partition_path = export_dir / "curated" / "raw_streaming_history" / "year=2024" / "month=7" / "data.parquet"
    assert raw_partition_path.exists()

    exported_table_lookup = {table["name"]: table for table in report["tables"]}
    assert exported_table_lookup["listener_daily_activity"]["warehouse_layer"] == "silver"
    assert exported_table_lookup["mart_run_quality"]["warehouse_layer"] == "gold"
    assert exported_table_lookup["mart_creator_opportunities"]["warehouse_layer"] == "gold"

    with duckdb.connect() as con:
        raw_rows = con.execute(f"SELECT COUNT(*) FROM read_parquet('{raw_partition_path.as_posix()}')").fetchone()[0]
        manifest_rows = con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{(export_dir / 'curated' / 'run_manifests' / 'data.parquet').as_posix()}')"
        ).fetchone()[0]
        daily_row = con.execute(
            f"SELECT played_date, total_streams, unique_artists FROM read_parquet('{(export_dir / 'curated' / 'listener_daily_activity' / 'data.parquet').as_posix()}')"
        ).fetchone()
        model_summary_row = con.execute(
            f"SELECT run_id, model_name, promoted, is_serving_alias FROM read_parquet('{(export_dir / 'curated' / 'model_run_summary' / 'data.parquet').as_posix()}')"
        ).fetchone()
        run_quality_row = con.execute(
            f"SELECT run_id, best_model_name, serving_model_name FROM read_parquet('{(export_dir / 'curated' / 'mart_run_quality' / 'data.parquet').as_posix()}')"
        ).fetchone()
        ops_row = con.execute(
            f"SELECT latest_run_id, ops_health_status, operating_rhythm_status FROM read_parquet('{(export_dir / 'curated' / 'mart_ops_overview' / 'data.parquet').as_posix()}')"
        ).fetchone()
        creator_row = con.execute(
            f"SELECT artist_name, priority_now_count, top_scene_name FROM read_parquet('{(export_dir / 'curated' / 'mart_creator_opportunities' / 'data.parquet').as_posix()}')"
        ).fetchone()
        raw_columns = {
            row[0]
            for row in con.execute(
                f"DESCRIBE SELECT * FROM read_parquet('{raw_partition_path.as_posix()}')"
            ).fetchall()
        }

    assert raw_rows == 1
    assert manifest_rows == 1
    assert daily_row == ("2024-07-01", 1, 1)
    assert model_summary_row == ("20260324_222814_taste-os-full", "mlp", False, True)
    assert run_quality_row == ("20260324_222814_taste-os-full", "mlp", "mlp")
    assert ops_row == ("20260324_222814_taste-os-full", "attention", "stale")
    assert creator_row == ("Artist X", 1, "scene-1")
    assert "ip_addr" not in raw_columns
    assert "played_at" in raw_columns
