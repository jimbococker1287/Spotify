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
    assert "CREATE OR REPLACE VIEW latest_run_results AS" in sql
    assert "MSCK REPAIR TABLE raw_streaming_history;" in sql


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
        "champion_alias": {"updated": False, "model_name": "", "model_type": ""},
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

    raw_partition_path = export_dir / "curated" / "raw_streaming_history" / "year=2024" / "month=7" / "data.parquet"
    assert raw_partition_path.exists()

    with duckdb.connect() as con:
        raw_rows = con.execute(f"SELECT COUNT(*) FROM read_parquet('{raw_partition_path.as_posix()}')").fetchone()[0]
        manifest_rows = con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{(export_dir / 'curated' / 'run_manifests' / 'data.parquet').as_posix()}')"
        ).fetchone()[0]
        raw_columns = {
            row[0]
            for row in con.execute(
                f"DESCRIBE SELECT * FROM read_parquet('{raw_partition_path.as_posix()}')"
            ).fetchall()
        }

    assert raw_rows == 1
    assert manifest_rows == 1
    assert "ip_addr" not in raw_columns
    assert "played_at" in raw_columns
