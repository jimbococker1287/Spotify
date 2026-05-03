from __future__ import annotations

import json
import logging
from pathlib import Path

import duckdb
import pandas as pd

from spotify.analytics_db import refresh_analytics_database
from spotify.analytics_warehouse import build_analytics_warehouse


def _write_minimal_streaming_history(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "Streaming_History_Audio_2024_0.json").write_text(
        json.dumps(
            [
                {
                    "ts": "2026-01-01T00:00:00Z",
                    "platform": "ios",
                    "ms_played": 180000,
                    "master_metadata_track_name": "Track A",
                    "master_metadata_album_artist_name": "Artist A",
                    "reason_start": "trackdone",
                    "reason_end": "trackdone",
                    "shuffle": False,
                    "skipped": False,
                    "offline": False,
                    "spotify_track_uri": "spotify:track:a",
                },
                {
                    "ts": "2026-01-01T01:00:00Z",
                    "platform": "ios",
                    "ms_played": 120000,
                    "master_metadata_track_name": "Track B",
                    "master_metadata_album_artist_name": "Artist B",
                    "reason_start": "clickrow",
                    "reason_end": "fwdbtn",
                    "shuffle": True,
                    "skipped": True,
                    "offline": False,
                    "spotify_track_uri": "spotify:track:b",
                },
            ]
        ),
        encoding="utf-8",
    )


def _write_minimal_run_artifacts(output_dir: Path) -> None:
    (output_dir / "history").mkdir(parents=True, exist_ok=True)
    (output_dir / "runs" / "run_a" / "analysis").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "timestamp": "2026-01-01T00:05:00",
                "run_id": "run_a",
                "run_name": "run-a",
                "profile": "full",
                "model_name": "mlp",
                "model_type": "classical",
                "model_family": "shallow_neural",
                "val_top1": 0.35,
                "val_top5": 0.71,
                "test_top1": 0.28,
                "test_top5": 0.62,
                "fit_seconds": 2.5,
                "epochs": 4,
                "data_records": 2,
            }
        ]
    ).to_csv(output_dir / "history" / "experiment_history.csv", index=False)
    pd.DataFrame(
        [
            {
                "timestamp": "2026-01-01T00:06:00",
                "run_id": "run_a",
                "run_name": "run-a",
                "profile": "full",
                "model_name": "mlp",
                "model_family": "shallow_neural",
                "model_type": "classical",
                "fold": 1,
                "train_rows": 2,
                "test_rows": 1,
                "fit_seconds": 0.1,
                "top1": 0.25,
                "top5": 0.6,
            }
        ]
    ).to_csv(output_dir / "history" / "backtest_history.csv", index=False)
    pd.DataFrame(
        [
            {
                "timestamp": "2026-01-01T00:07:00",
                "benchmark_id": "smokebench",
                "model_name": "mlp",
                "model_type": "classical",
                "model_family": "shallow_neural",
                "runs": 1,
                "val_top1_mean": 0.35,
                "val_top1_std": 0.0,
                "val_top1_ci95": 0.0,
                "test_top1_mean": 0.28,
                "test_top1_std": 0.0,
                "test_top1_ci95": 0.0,
            }
        ]
    ).to_csv(output_dir / "history" / "benchmark_history.csv", index=False)
    run_dir = output_dir / "runs" / "run_a"
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_id": "run_a",
                "run_name": "run-a",
                "profile": "full",
                "timestamp": "2026-01-01T00:04:00",
                "data_records": 2,
                "champion_gate": {
                    "status": "pass",
                    "promoted": True,
                    "metric_source": "backtest_top1",
                },
                "champion_alias": {
                    "updated": True,
                    "model_name": "mlp",
                    "model_type": "classical",
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "run_results.json").write_text(
        json.dumps(
            [
                {
                    "model_name": "mlp",
                    "model_type": "classical",
                    "model_family": "shallow_neural",
                    "val_top1": 0.35,
                    "test_top1": 0.28,
                    "fit_seconds": 2.5,
                    "epochs": 4,
                }
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "analysis" / "moonshot_summary.json").write_text(
        json.dumps(
            {
                "multimodal_embedding_dim": 8,
                "multimodal_feature_count": 14,
                "multimodal_retrieval_fusion_enabled": True,
                "digital_twin_test_auc": 0.71,
                "causal_test_auc_total": 0.68,
                "journey_mean_horizon": 6.0,
                "safe_policy_bucket_count": 2,
                "stress_worst_skip_scenario": "high_friction_spike",
                "stress_worst_skip_risk": 0.42,
            }
        ),
        encoding="utf-8",
    )


def _write_minimal_control_room(output_dir: Path) -> None:
    analytics_dir = output_dir / "analytics"
    analytics_dir.mkdir(parents=True, exist_ok=True)
    (analytics_dir / "control_room.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-01-01T01:00:00+00:00",
                "latest_run": {
                    "run_id": "run_a",
                    "profile": "full",
                    "best_model_name": "mlp",
                    "best_model_type": "classical",
                    "best_model_test_top1": 0.28,
                    "serving_model_name": "mlp",
                    "serving_model_type": "classical",
                    "promoted": True,
                },
                "run_selection": {
                    "selected_run": "run_a",
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
    pd.DataFrame(
        [
            {
                "generated_at": "2026-01-01T01:00:00+00:00",
                "run_id": "run_a",
                "run_timestamp": "2026-01-01T00:04:00",
                "profile": "full",
                "promoted": 1,
                "promotion_status": "pass",
                "best_model_name": "mlp",
                "best_model_type": "classical",
                "best_model_val_top1": 0.35,
                "best_model_test_top1": 0.28,
                "champion_gate_regression": 0.0,
                "target_drift_jsd": 0.218,
                "test_ece": 0.08,
                "test_selective_risk": 0.372,
                "test_abstention_rate": 0.241,
                "robustness_gap": 0.096,
                "stress_skip_risk": 0.593,
                "review_action_count": 1,
                "high_priority_review_actions": 0,
                "medium_priority_review_actions": 1,
                "review_action_areas": "cadence",
                "baseline_run_id": "run_prev",
                "next_bet_count": 1,
                "ops_coverage_ratio": 1.0,
                "available_summary_count": 6,
                "expected_summary_count": 6,
                "operating_status": "stale",
                "fast_cadence_status": "stale",
                "full_cadence_status": "healthy",
                "async_handoff_status": "attention",
                "recommended_run_command": "make schedule-run MODE=fast",
                "ops_health_status": "attention",
                "operational_high_priority_review_actions": 0,
                "strategic_high_priority_review_actions": 0,
                "test_accepted_rate": 0.759,
                "conformal_operating_threshold": 0.24,
                "repeat_from_prev_new_gap": 0.096,
                "stress_benchmark_skip_risk": 0.591,
            }
        ]
    ).to_csv(analytics_dir / "control_room_history.csv", index=False)


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
                "backfilled_artifact_index_at": "2026-01-01T01:05:00+00:00",
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
    pd.DataFrame(
        [
            {
                "artist_name": "Artist X",
                "opportunity_score": 0.41,
                "opportunity_rank": 1,
                "opportunity_band": "priority_now",
                "scene_name": "scene-1",
                "primary_driver": "seed_adjacency",
                "seed_bridges": '["Seed A"]',
                "why_now": "Strong adjacency.",
            }
        ]
    ).to_csv(base_dir / f"{family_id}_ranking_comparison.csv", index=False)
    pd.DataFrame(
        [
            {
                "scene_name": "scene-1",
                "avg_opportunity_score": 0.41,
                "priority_now_count": 1,
                "scene_local_play_share": 0.33,
                "scene_label_concentration": 0.1,
                "scene_release_pressure": 0.2,
                "top_opportunity_artist": "Artist X",
                "top_opportunity_score": 0.41,
                "top_seed_artists": '["Seed A"]',
            }
        ]
    ).to_csv(base_dir / f"{family_id}_scene_comparison.csv", index=False)
    pd.DataFrame(
        [
            {
                "scene_name": "scene-1",
                "seed_artist": "Seed A",
                "avg_opportunity_score": 0.41,
                "top_opportunity_artist": "Artist X",
                "top_opportunity_score": 0.41,
            }
        ]
    ).to_csv(base_dir / f"{family_id}_scene_seed_comparison.csv", index=False)


def _build_minimal_workspace(tmp_path: Path) -> tuple[Path, Path]:
    data_dir = tmp_path / "data" / "raw"
    output_dir = tmp_path / "outputs"
    _write_minimal_streaming_history(data_dir)
    _write_minimal_run_artifacts(output_dir)
    _write_minimal_control_room(output_dir)
    _write_minimal_creator_family(output_dir)
    return data_dir, output_dir


def test_build_analytics_warehouse_writes_curated_layers(tmp_path: Path) -> None:
    logger = logging.getLogger("spotify.test.analytics_warehouse")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    data_dir, output_dir = _build_minimal_workspace(tmp_path)

    warehouse_root = build_analytics_warehouse(
        data_dir=data_dir,
        output_dir=output_dir,
        include_video=False,
        logger=logger,
    )

    assert warehouse_root.exists()
    assert (warehouse_root / "warehouse_manifest.json").exists()
    assert (warehouse_root / "bronze" / "control_room_snapshot.parquet").exists()
    assert (warehouse_root / "silver" / "model_run_summary.parquet").exists()
    assert (warehouse_root / "gold" / "mart_creator_opportunities.parquet").exists()

    model_run_summary = pd.read_parquet(warehouse_root / "silver" / "model_run_summary.parquet")
    assert model_run_summary.loc[0, "is_serving_alias"]

    ops_overview = pd.read_parquet(warehouse_root / "gold" / "mart_ops_overview.parquet")
    assert ops_overview.loc[0, "latest_run_id"] == "run_a"

    creator_opportunities = pd.read_parquet(warehouse_root / "gold" / "mart_creator_opportunities.parquet")
    assert creator_opportunities.loc[0, "artist_name"] == "Artist X"


def test_refresh_analytics_database_registers_warehouse_marts(tmp_path: Path) -> None:
    logger = logging.getLogger("spotify.test.analytics_warehouse.duckdb")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    data_dir, output_dir = _build_minimal_workspace(tmp_path)

    db_path = refresh_analytics_database(
        data_dir=data_dir,
        output_dir=output_dir,
        include_video=False,
        logger=logger,
    )

    assert db_path is not None
    assert (output_dir / "analytics" / "warehouse" / "warehouse_manifest.md").exists()

    with duckdb.connect(str(db_path), read_only=True) as con:
        ops_row = con.execute(
            "SELECT latest_run_id, ops_health_status, operating_rhythm_status FROM mart_ops_overview"
        ).fetchone()
        creator_row = con.execute(
            "SELECT artist_name, priority_now_count FROM creator_priority_now ORDER BY max_opportunity_score DESC"
        ).fetchone()

    assert ops_row == ("run_a", "attention", "stale")
    assert creator_row == ("Artist X", 1)
