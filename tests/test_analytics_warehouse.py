from __future__ import annotations

import json
import logging
from pathlib import Path
import time

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
                    "selected_run": {
                        "run_id": "run_a",
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


def _write_minimal_creator_market_branch(output_dir: Path) -> None:
    base_dir = output_dir / "analysis" / "creator_market_intelligence"
    base_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {
                "scene_name": "scene-1",
                "family_count": 1,
                "avg_scene_local_play_share": 0.44,
                "avg_opportunity_score": 0.56,
                "total_priority_now": 2,
                "total_watchlist": 1,
                "avg_release_pressure": 0.31,
                "avg_label_concentration": 0.12,
                "avg_inbound_target_share": 0.27,
                "avg_outbound_source_share": 0.19,
                "avg_seed_bridge_count": 1.5,
                "dominant_driver": "seed_adjacency",
                "top_opportunity_artist": "Artist X",
                "top_migration_route": "Seed Artist -> Artist X",
                "strategy_posture": "accelerate",
                "momentum_score": 0.88,
            }
        ]
    ).to_csv(base_dir / "scene_market_pulse.csv", index=False)
    pd.DataFrame(
        [
            {
                "scene_name": "scene-1",
                "primary_driver": "seed_adjacency",
                "family_count": 1,
                "artist_count": 1,
                "opportunity_count": 1,
                "priority_now_count": 1,
                "watchlist_count": 0,
                "avg_opportunity_score": 0.58,
                "avg_scene_local_play_share": 0.44,
                "avg_scene_release_pressure": 0.31,
                "avg_scene_label_concentration": 0.12,
                "avg_seed_bridge_count": 2.0,
                "avg_fan_migration_score": 0.35,
                "avg_release_whitespace_score": 0.61,
                "avg_local_gap_score": 0.4,
                "avg_scene_momentum_score": 0.88,
                "representative_artist": "Artist X",
                "lane_posture": "expand",
                "lane_attractiveness_score": 0.91,
            }
        ]
    ).to_csv(base_dir / "opportunity_lane_atlas.csv", index=False)
    pd.DataFrame(
        [
            {
                "source_artist": "Seed Artist",
                "target_artist": "Artist X",
                "family_count": 1,
                "route_mentions": 1,
                "total_transition_count": 14,
                "avg_source_out_share": 0.29,
                "avg_target_in_share": 0.34,
                "source_scene_name": "scene-seed",
                "target_scene_name": "scene-1",
                "route_strength_score": 0.79,
            }
        ]
    ).to_csv(base_dir / "market_migration_network.csv", index=False)
    pd.DataFrame(
        [
            {
                "scene_name": "scene-1",
                "seed_artist": "Seed A",
                "family_count": 1,
                "opportunity_count": 2,
                "avg_opportunity_score": 0.55,
                "avg_bridge_artist_count": 3.0,
                "avg_scene_local_play_share": 0.44,
                "avg_scene_release_pressure": 0.31,
                "avg_scene_label_concentration": 0.12,
                "top_opportunity_artist": "Artist X",
                "dominant_driver": "seed_adjacency",
                "bridge_score": 0.83,
            }
        ]
    ).to_csv(base_dir / "seed_scene_bridge_atlas.csv", index=False)
    pd.DataFrame(
        [
            {
                "artist_name": "Artist X",
                "scene_name": "scene-1",
                "family_count": 1,
                "avg_opportunity_score": 0.58,
                "avg_release_whitespace_score": 0.67,
                "max_days_since_latest_release": 42,
                "avg_seed_bridge_count": 2.0,
                "dominant_labels": "Label A|Label B",
                "primary_driver": "seed_adjacency",
                "whitespace_signal_score": 0.94,
            }
        ]
    ).to_csv(base_dir / "release_whitespace_atlas.csv", index=False)
    (base_dir / "creator_market_brief.json").write_text(
        json.dumps(
            {
                "report_family_count": 1,
                "top_scene": {"scene_name": "scene-1", "momentum_score": 0.88},
                "top_lane": {"scene_name": "scene-1", "primary_driver": "seed_adjacency"},
                "top_route": {"source_artist": "Seed Artist", "target_artist": "Artist X"},
                "top_bridge": {"scene_name": "scene-1", "seed_artist": "Seed A"},
                "top_whitespace": {"artist_name": "Artist X", "scene_name": "scene-1"},
                "summary": ["Creator market branch is ready for warehouse ingestion."],
                "actions": ["Prioritize the whitespace watchlist in the semantic layer."],
            }
        ),
        encoding="utf-8",
    )
    (base_dir / "creator_market_manifest.json").write_text(
        json.dumps(
            {
                "report_family_count": 1,
                "manifest_backed_report_family_count": 1,
                "asset_backed_report_family_count": 1,
                "complete_report_family_count": 1,
                "partial_report_family_count": 0,
                "partial_report_family_ids": [],
                "artifact_root": str(base_dir.resolve()),
                "tables": {
                    "scene_market_pulse": {"row_count": 1},
                    "opportunity_lane_atlas": {"row_count": 1},
                    "market_migration_network": {"row_count": 1},
                    "seed_scene_bridge_atlas": {"row_count": 1},
                    "release_whitespace_atlas": {"row_count": 1},
                },
            }
        ),
        encoding="utf-8",
    )


def _write_minimal_research_platform_branch(output_dir: Path) -> None:
    base_dir = output_dir / "analysis" / "research_platform_lab"
    base_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {
                "run_id": "run_a",
                "profile": "full",
                "timestamp": "2026-01-01T00:04:00+00:00",
                "promoted": True,
                "champion_gate_status": "pass",
                "benchmark_protocol_present": True,
                "safety_platform_contract_present": True,
                "conformal_summary_count": 1,
                "backtest_model_count": 1,
                "benchmark_contract_version": "v1",
                "benchmark_comparison_mode": "temporal",
                "safety_api_group_count": 2,
                "spotify_wrapper_count": 1,
                "portability_note_count": 1,
                "portability_signal_status": "ready",
                "research_artifact_ratio": 1.0,
                "research_stage": "review_ready",
                "claim_pack_attached": True,
                "claim_pack_path": str((output_dir / "analysis" / "research_claims" / "research_claims.json").resolve()),
                "claim_pack_freshness_status": "fresh",
                "claim_pack_stale_source_path": "",
                "claim_pack_stale_source_count": 0,
                "run_manifest_path": str((output_dir / "runs" / "run_a" / "run_manifest.json").resolve()),
                "run_manifest_timestamp": "2026-01-01T00:04:00+00:00",
                "run_manifest_age_hours": 1.0,
                "benchmark_protocol_path": str((output_dir / "runs" / "run_a" / "benchmark_protocol.json").resolve()),
                "safety_platform_contract_path": str((output_dir / "runs" / "run_a" / "safety_platform_contract.json").resolve()),
                "target_drift_jsd": 0.218,
                "test_selective_risk": 0.372,
                "test_abstention_rate": 0.241,
                "robustness_gap": 0.096,
                "stress_skip_risk": 0.591,
                "ops_coverage_ratio": 1.0,
            }
        ]
    ).to_csv(base_dir / "run_research_registry.csv", index=False)
    pd.DataFrame(
        [
            {
                "benchmark_id": "smokebench",
                "canonical_profile": "full",
                "comparison_mode": "temporal",
                "comparison_ready": False,
                "comparison_status": "incomplete",
                "run_count": 3,
                "model_count": 2,
                "present_artifact_count": 3,
                "required_artifact_count": 4,
                "required_artifact_ratio": 0.75,
                "significant_pair_count": 1,
                "comparison_blocker_count": 1,
                "top_comparison_blocker": "Need repeated benchmark",
                "comparison_blockers_json": json.dumps(["Need repeated benchmark"]),
                "comparator_guard_status": "attention",
                "deep_comparator_ready": False,
                "observed_model_classes_json": json.dumps(["classical"]),
                "best_model_name": "mlp",
                "best_model_type": "classical",
                "best_val_top1_mean": 0.35,
                "best_test_top1_mean": 0.28,
                "top_significant_pair": "mlp vs deep",
                "top_significant_margin": 0.02,
                "manifest_freshness_status": "fresh",
                "manifest_stale_source_path": "",
                "manifest_stale_source_count": 0,
                "manifest_age_hours": 2.0,
                "summary_path": str((output_dir / "history" / "benchmark_lock_smokebench_summary.csv").resolve()),
                "significance_path": str((output_dir / "history" / "benchmark_lock_smokebench_significance.csv").resolve()),
                "benchmark_strength_score": 0.73,
                "manifest_path": str((output_dir / "history" / "benchmark_lock_smokebench_manifest.json").resolve()),
            }
        ]
    ).to_csv(base_dir / "benchmark_lock_atlas.csv", index=False)
    pd.DataFrame(
        [
            {
                "claim_key": "claim_ready",
                "title": "Model is ready",
                "role": "primary",
                "status": "analysis_ready",
                "claim_readiness_status": "ready",
                "summary": "Primary claim is ready.",
                "live_signal_status": "ready",
                "benchmark_evidence_status": "ready",
                "repeated_evidence_status": "ready",
                "slice_evidence_status": "ready",
                "risk_evidence_status": "ready",
                "artifact_pack_status": "fresh",
                "supporting_artifact_count": 3,
                "existing_supporting_artifact_count": 3,
                "missing_supporting_artifact_count": 0,
                "stale_supporting_artifact_count": 0,
                "supporting_artifact_path_status": "ready",
                "supporting_artifact_freshness_status": "fresh",
                "missing_supporting_artifact_path": "",
                "stale_supporting_artifact_path": "",
                "missing_check_count": 0,
                "blocked": False,
                "next_gate": "ready_to_package",
                "target_drift_jsd": 0.12,
                "selective_risk": 0.18,
                "stress_skip_risk": 0.21,
                "live_test_top1_lift_vs_deep": 0.04,
                "benchmark_comparison_ready": True,
                "benchmark_significant_lift": True,
                "claims_path": str((output_dir / "analysis" / "research_claims" / "research_claims.json").resolve()),
                "metrics_json": json.dumps({"target_drift_jsd": 0.12}),
                "missing_checks_json": json.dumps([]),
            },
            {
                "claim_key": "claim_gap",
                "title": "Benchmark evidence is incomplete",
                "role": "backup",
                "status": "attention",
                "claim_readiness_status": "blocked",
                "summary": "Backup claim is blocked pending repeated benchmark evidence.",
                "live_signal_status": "ready",
                "benchmark_evidence_status": "missing",
                "repeated_evidence_status": "attention",
                "slice_evidence_status": "ready",
                "risk_evidence_status": "attention",
                "artifact_pack_status": "stale",
                "supporting_artifact_count": 2,
                "existing_supporting_artifact_count": 1,
                "missing_supporting_artifact_count": 1,
                "stale_supporting_artifact_count": 1,
                "supporting_artifact_path_status": "partial",
                "supporting_artifact_freshness_status": "stale",
                "missing_supporting_artifact_path": "/tmp/missing.json",
                "stale_supporting_artifact_path": "/tmp/stale.json",
                "missing_check_count": 2,
                "blocked": True,
                "next_gate": "rerun_benchmark",
                "target_drift_jsd": 0.22,
                "selective_risk": 0.37,
                "stress_skip_risk": 0.59,
                "live_test_top1_lift_vs_deep": 0.01,
                "benchmark_comparison_ready": False,
                "benchmark_significant_lift": False,
                "claims_path": str((output_dir / "analysis" / "research_claims" / "research_claims.json").resolve()),
                "metrics_json": json.dumps({"target_drift_jsd": 0.22}),
                "missing_checks_json": json.dumps(["rerun benchmark", "refresh pack"]),
            },
        ]
    ).to_csv(base_dir / "research_claim_registry.csv", index=False)
    (base_dir / "research_platform_maturity.json").write_text(
        json.dumps(
            {
                "anchor_run_id": "run_a",
                "anchor_run": {"run_id": "run_a", "research_stage": "review_ready"},
                "strongest_benchmark_lock": {"benchmark_id": "smokebench", "benchmark_strength_score": 0.73},
                "claim_ready_count": 1,
                "claim_blocked_count": 1,
                "claim_total_count": 2,
                "incomplete_benchmark_lock_count": 1,
                "stale_benchmark_manifest_count": 0,
                "stale_claim_artifact_count": 1,
                "submission_status": "internal_review",
                "ready_for_external_review": False,
                "blockers": ["Need repeated benchmark"],
                "top_next_gate": "rerun_benchmark",
                "summary": ["Research platform branch is ready for warehouse ingestion."],
                "actions": ["Resolve the repeated benchmark gap before external review."],
            }
        ),
        encoding="utf-8",
    )
    (base_dir / "research_platform_manifest.json").write_text(
        json.dumps(
            {
                "anchor_run_id": "run_a",
                "artifact_root": str(base_dir.resolve()),
                "tables": {
                    "run_research_registry": {"row_count": 1},
                    "benchmark_lock_atlas": {"row_count": 1},
                    "research_claim_registry": {"row_count": 2},
                },
            }
        ),
        encoding="utf-8",
    )


def _write_minimal_scope_expansion_branch(output_dir: Path) -> None:
    base_dir = output_dir / "analysis" / "scope_expansion"
    base_dir.mkdir(parents=True, exist_ok=True)
    generated_at = "2026-01-01T01:00:00+00:00"

    pd.DataFrame(
        [
            {
                "branch_key": "analytics_engineering",
                "branch_name": "Data Engineering + Analytics Engineering",
                "scope_lane": "local warehouse",
                "audience": "operator",
                "status": "ready",
                "readiness_score": 0.88,
                "evidence_score": 1.0,
                "freshness_score": 0.9,
                "risk_score": 0.12,
                "primary_metric_name": "warehouse_asset_count",
                "primary_metric_value": 42.0,
                "artifact_count": 4,
                "artifact_root": str((output_dir / "analytics" / "warehouse").resolve()),
                "top_signal": "Warehouse assets are queryable.",
                "top_gap": "Need dashboard-facing branch queue.",
                "recommended_next_step": "Publish branch-health mart.",
                "proof_artifacts": json.dumps(["warehouse_manifest.json"]),
            },
            {
                "branch_key": "data_science_quant",
                "branch_name": "Data Science + Quant",
                "scope_lane": "decision science",
                "audience": "researcher",
                "status": "attention",
                "readiness_score": 0.81,
                "evidence_score": 1.0,
                "freshness_score": 1.0,
                "risk_score": 0.62,
                "primary_metric_name": "top_scenario_utility",
                "primary_metric_value": 0.79,
                "artifact_count": 5,
                "artifact_root": str((output_dir / "analysis" / "quant_decision_lab").resolve()),
                "top_signal": "Policy frontier has enough signal for notebooks.",
                "top_gap": "Need repeated benchmark linkage.",
                "recommended_next_step": "Connect quant decisions to research locks.",
                "proof_artifacts": json.dumps(["scenario_policy_frontier.csv"]),
            },
            {
                "branch_key": "creator_market_intelligence",
                "branch_name": "Creator / Market Intelligence",
                "scope_lane": "market intelligence",
                "audience": "creator strategist",
                "status": "attention",
                "readiness_score": 0.86,
                "evidence_score": 0.84,
                "freshness_score": 0.8,
                "risk_score": 0.37,
                "primary_metric_name": "report_family_count",
                "primary_metric_value": 1.0,
                "artifact_count": 5,
                "artifact_root": str((output_dir / "analysis" / "creator_market_intelligence").resolve()),
                "top_signal": "Creator-market watchlist is populated.",
                "top_gap": "Need multi-family market breadth.",
                "recommended_next_step": "Backfill additional creator families.",
                "proof_artifacts": json.dumps(["creator_market_brief.json"]),
            },
            {
                "branch_key": "research_platform",
                "branch_name": "Research Platform",
                "scope_lane": "evidence platform",
                "audience": "reviewer",
                "status": "attention",
                "readiness_score": 0.54,
                "evidence_score": 0.43,
                "freshness_score": 1.0,
                "risk_score": 0.76,
                "primary_metric_name": "blocked_claim_count",
                "primary_metric_value": 1.0,
                "artifact_count": 5,
                "artifact_root": str((output_dir / "analysis" / "research_platform_lab").resolve()),
                "top_signal": "Research registry is present.",
                "top_gap": "Blocked claims still prevent external packaging.",
                "recommended_next_step": "Resolve research-platform blockers.",
                "proof_artifacts": json.dumps(["research_platform_maturity.json"]),
            },
        ]
    ).to_csv(base_dir / "branch_expansion_scorecard.csv", index=False)
    pd.DataFrame(
        [
            {
                "rank": 1,
                "branch_key": "research_platform",
                "branch_name": "Research Platform",
                "initiative": "Close blocked research claims",
                "why_now": "Evidence blockers hold back all downstream branch narratives.",
                "success_metric": "claim_blocked_count == 0",
                "required_artifacts": json.dumps(["research_claim_registry.csv"]),
                "command": "make research-platform-lab",
                "effort": "medium",
                "impact_score": 0.93,
                "risk_reduction_score": 0.88,
                "dependencies": json.dumps(["benchmark lock"]),
            },
            {
                "rank": 2,
                "branch_key": "data_science_quant",
                "branch_name": "Data Science + Quant",
                "initiative": "Tie scenario frontier to benchmark evidence",
                "why_now": "Quant work is useful but needs stronger proof loops.",
                "success_metric": "frontier rows linked to benchmark ids",
                "required_artifacts": json.dumps(["scenario_policy_frontier.csv"]),
                "command": "make quant-decision-lab",
                "effort": "medium",
                "impact_score": 0.87,
                "risk_reduction_score": 0.7,
                "dependencies": json.dumps(["research platform"]),
            },
            {
                "rank": 3,
                "branch_key": "analytics_engineering",
                "branch_name": "Data Engineering + Analytics Engineering",
                "initiative": "Expose branch health in DuckDB",
                "why_now": "The local warehouse should be the branch cockpit.",
                "success_metric": "scope_expansion_priority_queue view passes consistency checks",
                "required_artifacts": json.dumps(["mart_scope_expansion_health.parquet"]),
                "command": "make analytics-db",
                "effort": "small",
                "impact_score": 0.82,
                "risk_reduction_score": 0.55,
                "dependencies": json.dumps(["scope expansion lab"]),
            },
            {
                "rank": 4,
                "branch_key": "creator_market_intelligence",
                "branch_name": "Creator / Market Intelligence",
                "initiative": "Increase market family breadth",
                "why_now": "Creator intelligence needs repeated scenes, lanes, and whitespace signals.",
                "success_metric": "report_family_count >= 3",
                "required_artifacts": json.dumps(["creator_market_trend_deltas.csv"]),
                "command": "make creator-market-intelligence",
                "effort": "medium",
                "impact_score": 0.76,
                "risk_reduction_score": 0.49,
                "dependencies": json.dumps(["creator ranking families"]),
            },
        ]
    ).to_csv(base_dir / "branch_expansion_implementation_queue.csv", index=False)
    pd.DataFrame(
        [
            {
                "branch_key": "research_platform",
                "branch_name": "Research Platform",
                "development_mode": "stabilize",
                "status": "attention",
                "readiness_score": 0.54,
                "risk_score": 0.76,
                "sprint_objective": "Reduce the highest research risk.",
                "next_initiative": "Close blocked research claims",
                "why_now": "Evidence blockers hold back all downstream branch narratives.",
                "success_metric": "claim_blocked_count == 0",
                "primary_command": "make research-platform-lab",
                "validation_command": ".venv/bin/python -m pytest tests/test_research_platform_lab.py",
                "required_artifacts": json.dumps(["research_claim_registry.csv"]),
                "proof_artifacts": json.dumps(["research_platform_maturity.json"]),
                "decision_rule": "Do not expand until blocked evidence falls.",
                "handoff_summary": "Research Platform is in stabilize mode.",
            },
            {
                "branch_key": "data_science_quant",
                "branch_name": "Data Science + Quant",
                "development_mode": "extend",
                "status": "attention",
                "readiness_score": 0.81,
                "risk_score": 0.62,
                "sprint_objective": "Deepen quant decision evidence.",
                "next_initiative": "Tie scenario frontier to benchmark evidence",
                "why_now": "Quant work is useful but needs stronger proof loops.",
                "success_metric": "frontier rows linked to benchmark ids",
                "primary_command": "make quant-decision-lab",
                "validation_command": ".venv/bin/python -m pytest tests/test_quant_decision_lab.py",
                "required_artifacts": json.dumps(["scenario_policy_frontier.csv"]),
                "proof_artifacts": json.dumps(["scenario_policy_frontier.csv"]),
                "decision_rule": "Extend if the generated artifacts prove the frontier.",
                "handoff_summary": "Data Science + Quant is in extend mode.",
            },
            {
                "branch_key": "analytics_engineering",
                "branch_name": "Data Engineering + Analytics Engineering",
                "development_mode": "scale",
                "status": "ready",
                "readiness_score": 0.88,
                "risk_score": 0.12,
                "sprint_objective": "Turn branch health into a reusable cockpit.",
                "next_initiative": "Expose branch health in DuckDB",
                "why_now": "The local warehouse should be the branch cockpit.",
                "success_metric": "scope_expansion_priority_queue view passes consistency checks",
                "primary_command": "make analytics-db",
                "validation_command": ".venv/bin/python -m pytest tests/test_analytics_warehouse.py",
                "required_artifacts": json.dumps(["mart_scope_expansion_health.parquet"]),
                "proof_artifacts": json.dumps(["warehouse_manifest.json"]),
                "decision_rule": "Scale if the warehouse contract passes.",
                "handoff_summary": "Analytics Engineering is in scale mode.",
            },
            {
                "branch_key": "creator_market_intelligence",
                "branch_name": "Creator / Market Intelligence",
                "development_mode": "extend",
                "status": "attention",
                "readiness_score": 0.86,
                "risk_score": 0.37,
                "sprint_objective": "Convert market signals into strategy cards.",
                "next_initiative": "Increase market family breadth",
                "why_now": "Creator intelligence needs repeated scenes, lanes, and whitespace signals.",
                "success_metric": "report_family_count >= 3",
                "primary_command": "make creator-market-intelligence",
                "validation_command": ".venv/bin/python -m pytest tests/test_creator_market_intelligence.py",
                "required_artifacts": json.dumps(["creator_market_trend_deltas.csv"]),
                "proof_artifacts": json.dumps(["creator_market_brief.json"]),
                "decision_rule": "Extend if trend deltas become repeated strategy signals.",
                "handoff_summary": "Creator / Market Intelligence is in extend mode.",
            },
        ]
    ).to_csv(base_dir / "branch_strategy_cards.csv", index=False)
    (base_dir / "scope_expansion_manifest.json").write_text(
        json.dumps(
            {
                "generated_at": generated_at,
                "artifact_root": str(base_dir.resolve()),
                "branch_count": 4,
                "ready_branch_count": 1,
                "attention_branch_count": 3,
                "blocked_branch_count": 0,
                "missing_branch_count": 0,
                "queue_count": 4,
                "strategy_card_count": 4,
                "top_queue_item": {
                    "branch_key": "research_platform",
                    "initiative": "Close blocked research claims",
                    "command": "make research-platform-lab",
                },
            }
        ),
        encoding="utf-8",
    )


def _build_minimal_workspace(tmp_path: Path) -> tuple[Path, Path]:
    data_dir = tmp_path / "data" / "raw"
    output_dir = tmp_path / "outputs"
    _write_minimal_streaming_history(data_dir)
    _write_minimal_run_artifacts(output_dir)
    _write_minimal_control_room(output_dir)
    _write_minimal_creator_family(output_dir)
    _write_minimal_creator_market_branch(output_dir)
    _write_minimal_research_platform_branch(output_dir)
    _write_minimal_scope_expansion_branch(output_dir)
    return data_dir, output_dir


def _load_manifest(warehouse_root: Path) -> dict[str, object]:
    return json.loads((warehouse_root / "warehouse_manifest.json").read_text(encoding="utf-8"))


def _manifest_asset_lookup(manifest: dict[str, object]) -> dict[str, dict[str, object]]:
    lookup: dict[str, dict[str, object]] = {}
    layers = manifest.get("layers", {})
    if not isinstance(layers, dict):
        return lookup
    for assets in layers.values():
        if not isinstance(assets, list):
            continue
        for asset in assets:
            if not isinstance(asset, dict):
                continue
            name = str(asset.get("name", "") or "")
            if name:
                lookup[name] = asset
    return lookup


def _asset_refs_by_name(rows: list[dict[str, object]]) -> set[str]:
    return {
        f"{row.get('layer')}.{row.get('asset_name')}"
        for row in rows
        if isinstance(row, dict)
    }


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
    assert (warehouse_root / "warehouse_verification.json").exists()
    assert (warehouse_root / "warehouse_lineage.json").exists()
    assert (warehouse_root / "warehouse_lineage.md").exists()
    assert (warehouse_root / "bronze" / "control_room_snapshot.parquet").exists()
    assert (warehouse_root / "bronze" / "creator_market_scene_pulse.parquet").exists()
    assert (warehouse_root / "bronze" / "research_platform_run_registry.parquet").exists()
    assert (warehouse_root / "bronze" / "scope_expansion_scorecard.parquet").exists()
    assert (warehouse_root / "bronze" / "scope_expansion_strategy_cards.parquet").exists()
    assert (warehouse_root / "silver" / "model_run_summary.parquet").exists()
    assert (warehouse_root / "silver" / "creator_market_scene_summary.parquet").exists()
    assert (warehouse_root / "silver" / "research_platform_status_summary.parquet").exists()
    assert (warehouse_root / "silver" / "scope_expansion_branch_health.parquet").exists()
    assert (warehouse_root / "gold" / "mart_creator_opportunities.parquet").exists()
    assert (warehouse_root / "gold" / "mart_creator_market_watchlist.parquet").exists()
    assert (warehouse_root / "gold" / "mart_research_platform_status.parquet").exists()
    assert (warehouse_root / "gold" / "mart_scope_expansion_health.parquet").exists()

    control_room_snapshot = pd.read_parquet(warehouse_root / "bronze" / "control_room_snapshot.parquet")
    assert control_room_snapshot.loc[0, "selected_run_id"] == "run_a"

    model_run_summary = pd.read_parquet(warehouse_root / "silver" / "model_run_summary.parquet")
    assert model_run_summary.loc[0, "is_serving_alias"]

    ops_overview = pd.read_parquet(warehouse_root / "gold" / "mart_ops_overview.parquet")
    assert ops_overview.loc[0, "latest_run_id"] == "run_a"

    creator_opportunities = pd.read_parquet(warehouse_root / "gold" / "mart_creator_opportunities.parquet")
    assert creator_opportunities.loc[0, "artist_name"] == "Artist X"

    creator_market_scene_summary = pd.read_parquet(warehouse_root / "silver" / "creator_market_scene_summary.parquet")
    assert creator_market_scene_summary.loc[0, "scene_name"] == "scene-1"
    assert creator_market_scene_summary.loc[0, "report_family_count"] == 1

    research_platform_status = pd.read_parquet(warehouse_root / "silver" / "research_platform_status_summary.parquet")
    assert research_platform_status.loc[0, "anchor_run_id"] == "run_a"
    assert research_platform_status.loc[0, "claim_total_count"] == 2

    creator_market_watchlist = pd.read_parquet(warehouse_root / "gold" / "mart_creator_market_watchlist.parquet")
    assert creator_market_watchlist.loc[0, "artist_name"] == "Artist X"

    research_platform_mart = pd.read_parquet(warehouse_root / "gold" / "mart_research_platform_status.parquet")
    assert research_platform_mart.loc[0, "status_posture"] == "blocked"

    scope_branch_health = pd.read_parquet(warehouse_root / "silver" / "scope_expansion_branch_health.parquet")
    assert scope_branch_health.loc[0, "branch_key"] == "research_platform"
    assert scope_branch_health.loc[0, "next_command"] == "make research-platform-lab"
    assert scope_branch_health.loc[0, "development_mode"] == "stabilize"
    assert "test_research_platform_lab.py" in scope_branch_health.loc[0, "validation_command"]

    scope_health_mart = pd.read_parquet(warehouse_root / "gold" / "mart_scope_expansion_health.parquet")
    assert scope_health_mart.loc[0, "branch_key"] == "research_platform"
    assert scope_health_mart.loc[0, "branch_posture"] == "blocked"
    assert scope_health_mart.loc[0, "sprint_objective"] == "Reduce the highest research risk."

    manifest = _load_manifest(warehouse_root)
    manifest_assets = _manifest_asset_lookup(manifest)
    assert manifest["refresh"]["built_assets"] >= 1
    assert manifest["refresh"]["reused_assets"] == 0
    assert "lineage_graph" in manifest
    assert "quality" in manifest
    assert manifest_assets["creator_market_scene_summary"]["refresh_status"] == "built"
    assert manifest_assets["mart_research_platform_status"]["branch_backed"]
    assert manifest_assets["mart_research_platform_status"]["source_fingerprint"]
    assert manifest_assets["research_platform_claim_registry"]["branch_freshness"]["status"] == "attention"
    assert manifest_assets["mart_scope_expansion_health"]["branch_backed"]
    assert manifest_assets["mart_scope_expansion_health"]["source_fingerprint"]
    assert manifest_assets["scope_expansion_branch_health"]["rows"] == 4
    assert manifest_assets["scope_expansion_strategy_cards"]["rows"] == 4

    lineage_report = json.loads((warehouse_root / "warehouse_lineage.json").read_text(encoding="utf-8"))
    lineage_edges = {
        (
            edge["upstream_layer"],
            edge["upstream_asset"],
            edge["downstream_layer"],
            edge["downstream_asset"],
        )
        for edge in lineage_report["lineage"]["edges"]
    }
    assert ("bronze", "raw_streaming_history", "silver", "listener_daily_activity") in lineage_edges
    assert ("silver", "creator_market_scene_summary", "gold", "mart_creator_market_watchlist") in lineage_edges
    assert ("silver", "research_platform_status_summary", "gold", "mart_research_platform_status") in lineage_edges
    assert ("bronze", "scope_expansion_scorecard", "silver", "scope_expansion_branch_health") in lineage_edges
    assert ("bronze", "scope_expansion_strategy_cards", "silver", "scope_expansion_branch_health") in lineage_edges
    assert ("silver", "scope_expansion_branch_health", "gold", "mart_scope_expansion_health") in lineage_edges
    assert lineage_report["quality"]["summary"]["empty_asset_count"] >= 1
    assert "bronze.optuna_history" in _asset_refs_by_name(lineage_report["quality"]["empty_assets"])
    assert lineage_report["quality"]["summary"]["row_count_anomaly_count"] == 0
    freshness_refs = _asset_refs_by_name(lineage_report["quality"]["branch_backed_artifact_freshness"])
    assert "bronze.research_platform_claim_registry" in freshness_refs
    assert "gold.mart_scope_expansion_health" in freshness_refs

    verification = json.loads((warehouse_root / "warehouse_verification.json").read_text(encoding="utf-8"))
    assert verification["status"] == "pass"
    assert verification["summary"]["failed_assets"] == 0
    assert verification["refresh"]["built_assets"] >= 1
    assert verification["refresh"]["reused_assets"] == 0


def test_build_analytics_warehouse_reuses_unchanged_branch_backed_assets(tmp_path: Path) -> None:
    logger = logging.getLogger("spotify.test.analytics_warehouse.incremental")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    data_dir, output_dir = _build_minimal_workspace(tmp_path)

    warehouse_root = build_analytics_warehouse(
        data_dir=data_dir,
        output_dir=output_dir,
        include_video=False,
        logger=logger,
    )
    creator_parquet = warehouse_root / "bronze" / "creator_market_release_whitespace_atlas.parquet"
    research_parquet = warehouse_root / "gold" / "mart_research_platform_status.parquet"
    scope_parquet = warehouse_root / "gold" / "mart_scope_expansion_health.parquet"
    creator_mtime_first = creator_parquet.stat().st_mtime_ns
    research_mtime_first = research_parquet.stat().st_mtime_ns
    scope_mtime_first = scope_parquet.stat().st_mtime_ns

    time.sleep(0.05)
    warehouse_root = build_analytics_warehouse(
        data_dir=data_dir,
        output_dir=output_dir,
        include_video=False,
        logger=logger,
    )
    creator_mtime_second = creator_parquet.stat().st_mtime_ns
    research_mtime_second = research_parquet.stat().st_mtime_ns
    scope_mtime_second = scope_parquet.stat().st_mtime_ns
    manifest_second = _load_manifest(warehouse_root)
    assets_second = _manifest_asset_lookup(manifest_second)

    assert creator_mtime_second == creator_mtime_first
    assert research_mtime_second == research_mtime_first
    assert scope_mtime_second == scope_mtime_first
    assert manifest_second["refresh"]["reused_assets"] >= 1
    assert manifest_second["refresh"]["branch_backed_reused_assets"] >= 1
    assert assets_second["creator_market_release_whitespace_atlas"]["refresh_status"] == "reused"
    assert assets_second["mart_research_platform_status"]["refresh_status"] == "reused"
    assert assets_second["mart_scope_expansion_health"]["refresh_status"] == "reused"

    creator_market_dir = output_dir / "analysis" / "creator_market_intelligence"
    whitespace_path = creator_market_dir / "release_whitespace_atlas.csv"
    whitespace_df = pd.read_csv(whitespace_path)
    whitespace_df.loc[0, "avg_release_whitespace_score"] = 0.73
    whitespace_df.to_csv(whitespace_path, index=False)

    time.sleep(0.05)
    warehouse_root = build_analytics_warehouse(
        data_dir=data_dir,
        output_dir=output_dir,
        include_video=False,
        logger=logger,
    )
    creator_mtime_third = creator_parquet.stat().st_mtime_ns
    research_mtime_third = research_parquet.stat().st_mtime_ns
    scope_mtime_third = scope_parquet.stat().st_mtime_ns
    manifest_third = _load_manifest(warehouse_root)
    assets_third = _manifest_asset_lookup(manifest_third)

    assert creator_mtime_third > creator_mtime_second
    assert research_mtime_third == research_mtime_second
    assert scope_mtime_third == scope_mtime_second
    assert assets_third["creator_market_release_whitespace_atlas"]["refresh_status"] == "rebuilt"
    assert assets_third["creator_market_release_whitespace_atlas"]["refresh_reason"] == "content_hash_changed"
    assert assets_third["mart_creator_market_watchlist"]["refresh_status"] == "rebuilt"
    assert assets_third["mart_research_platform_status"]["refresh_status"] == "reused"
    assert assets_third["mart_scope_expansion_health"]["refresh_status"] == "reused"
    assert manifest_third["refresh"]["rebuilt_assets"] >= 1


def test_build_analytics_warehouse_reports_row_count_anomalies(tmp_path: Path) -> None:
    logger = logging.getLogger("spotify.test.analytics_warehouse.quality")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    data_dir, output_dir = _build_minimal_workspace(tmp_path)

    build_analytics_warehouse(
        data_dir=data_dir,
        output_dir=output_dir,
        include_video=False,
        logger=logger,
    )
    whitespace_path = output_dir / "analysis" / "creator_market_intelligence" / "release_whitespace_atlas.csv"
    pd.read_csv(whitespace_path).head(0).to_csv(whitespace_path, index=False)

    warehouse_root = build_analytics_warehouse(
        data_dir=data_dir,
        output_dir=output_dir,
        include_video=False,
        logger=logger,
    )

    lineage_report = json.loads((warehouse_root / "warehouse_lineage.json").read_text(encoding="utf-8"))
    anomalies = lineage_report["quality"]["row_count_anomalies"]
    anomaly_keys = {
        (row["layer"], row["asset_name"], row["type"])
        for row in anomalies
    }
    assert lineage_report["quality"]["summary"]["row_count_anomaly_count"] >= 1
    assert (
        "bronze",
        "creator_market_release_whitespace_atlas",
        "row_count_dropped_to_zero",
    ) in anomaly_keys
    assert (
        "gold",
        "mart_creator_market_watchlist",
        "row_count_dropped_to_zero",
    ) in anomaly_keys


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
        creator_market_row = con.execute(
            """
            SELECT artist_name, scene_name, scene_priority_now_count
            FROM creator_market_priority_now
            ORDER BY market_priority_score DESC
            """
        ).fetchone()
        research_status_row = con.execute(
            """
            SELECT anchor_run_id, status_posture, submission_status
            FROM latest_research_platform_status
            """
        ).fetchone()
        blocked_claim_row = con.execute(
            """
            SELECT claim_key, blocked
            FROM research_platform_blocked_claims
            ORDER BY watchlist_score DESC
            """
        ).fetchone()
        scope_queue_row = con.execute(
            """
            SELECT branch_key, branch_posture, next_command
            FROM scope_expansion_priority_queue
            ORDER BY queue_rank
            LIMIT 1
            """
        ).fetchone()
        consistency_rows = con.execute(
            """
            SELECT object_name, row_count_match, column_match, logical_type_match
            FROM warehouse_consistency_checks
            WHERE object_name IN (
              'control_room_snapshot',
              'creator_market_scene_summary',
              'mart_creator_market_watchlist',
              'mart_ops_overview',
              'mart_research_platform_status',
              'mart_scope_expansion_health',
              'latest_ops_overview',
              'latest_research_platform_status',
              'scope_expansion_branch_health',
              'scope_expansion_strategy_cards',
              'scope_expansion_priority_queue'
            )
            ORDER BY object_name
            """
        ).fetchall()
        metadata_row = con.execute(
            """
            SELECT expected_rows, expected_column_count, refresh_status, branch_backed, source_fingerprint
            FROM warehouse_asset_manifest
            WHERE asset_name = 'mart_research_platform_status'
            """
        ).fetchone()
        selected_run_row = con.execute(
            """
            SELECT selected_run_id
            FROM control_room_snapshot
            """
        ).fetchone()

    assert ops_row == ("run_a", "attention", "stale")
    assert creator_row == ("Artist X", 1)
    assert creator_market_row == ("Artist X", "scene-1", 2)
    assert research_status_row == ("run_a", "blocked", "internal_review")
    assert blocked_claim_row == ("claim_gap", True)
    assert scope_queue_row == ("research_platform", "blocked", "make research-platform-lab")
    assert consistency_rows == [
        ("control_room_snapshot", True, True, True),
        ("creator_market_scene_summary", True, True, True),
        ("latest_ops_overview", True, True, True),
        ("latest_research_platform_status", True, True, True),
        ("mart_creator_market_watchlist", True, True, True),
        ("mart_ops_overview", True, True, True),
        ("mart_research_platform_status", True, True, True),
        ("mart_scope_expansion_health", True, True, True),
        ("scope_expansion_branch_health", True, True, True),
        ("scope_expansion_priority_queue", True, True, True),
        ("scope_expansion_strategy_cards", True, True, True),
    ]
    assert metadata_row is not None
    assert metadata_row[0] == 1
    assert metadata_row[1] >= 1
    assert metadata_row[2] == "built"
    assert metadata_row[3] is True
    assert metadata_row[4]
    assert selected_run_row == ("run_a",)
