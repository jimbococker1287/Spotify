from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from spotify.control_room import build_control_room_report, write_control_room_report


def _write_run_fixture(
    output_dir: Path,
    *,
    run_id: str,
    timestamp: str,
    profile: str,
    promoted: bool,
    promotion_status: str,
    best_model_name: str,
    best_model_type: str,
    val_top1: float,
    test_top1: float,
    gate_regression: float,
    drift_jsd: float,
    ece: float,
    selective_risk: float,
    abstention_rate: float,
    robustness_gap: float,
    stress_skip_risk: float,
    stress_scenario: str,
) -> None:
    run_dir = output_dir / "runs" / run_id
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    champion_alias = (
        {"model_name": best_model_name, "model_type": best_model_type}
        if promoted
        else {"model_name": "", "model_type": ""}
    )
    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "run_name": run_id,
                "profile": profile,
                "timestamp": timestamp,
                "data_records": 1234,
                "num_artists": 87,
                "num_context_features": 12,
                "champion_gate": {
                    "status": promotion_status,
                    "promoted": promoted,
                    "metric_source": "backtest_top1",
                    "regression": gate_regression,
                },
                "champion_alias": champion_alias,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "run_results.json").write_text(
        json.dumps(
            [
                {
                    "model_name": best_model_name,
                    "model_type": best_model_type,
                    "val_top1": val_top1,
                    "test_top1": test_top1,
                }
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    (analysis_dir / f"{best_model_type}_{best_model_name}_confidence_summary.json").write_text(
        json.dumps(
            {
                "test_ece": ece,
                "test_selective_risk": selective_risk,
                "test_abstention_rate": abstention_rate,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (analysis_dir / "data_drift_summary.json").write_text(
        json.dumps(
            {
                "target_drift": {"train_vs_test_jsd": drift_jsd},
                "largest_context_shift": {"feature": "tech_stutter_events_24h", "max_abs_std_mean_diff": 1.2},
                "largest_segment_shift": {
                    "split": "test",
                    "segment": "repeat_from_prev",
                    "bucket": "new",
                    "abs_share_shift": 0.24,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (analysis_dir / "friction_proxy_summary.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "friction_feature_count": 3,
                "proxy_counterfactual": {"test_mean_delta": 0.01},
                "top_friction_features": [{"feature": "offline", "mean_risk_delta": 0.01}],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (analysis_dir / "robustness_summary.json").write_text(
        json.dumps(
            [
                {
                    "model_name": best_model_name,
                    "max_top1_gap": robustness_gap,
                    "worst_segment": "repeat_from_prev",
                    "worst_bucket": "new",
                }
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    (analysis_dir / "moonshot_summary.json").write_text(
        json.dumps(
            {
                "digital_twin_test_auc": 0.72,
                "causal_test_auc_total": 0.69,
                "stress_worst_skip_scenario": stress_scenario,
                "stress_worst_skip_risk": stress_skip_risk,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def test_build_control_room_report_summarizes_latest_run(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    history_dir = output_dir / "history"
    run_dir = output_dir / "runs" / "run_a"
    analysis_dir = run_dir / "analysis"
    history_dir.mkdir(parents=True)
    analysis_dir.mkdir(parents=True)

    pd.DataFrame(
        [
            {"run_id": "run_a", "model_name": "retrieval_reranker", "model_type": "retrieval_reranker", "val_top1": 0.61},
            {"run_id": "run_a", "model_name": "gru_artist", "model_type": "deep", "val_top1": 0.55},
        ]
    ).to_csv(history_dir / "experiment_history.csv", index=False)
    pd.DataFrame(
        [
            {"run_id": "run_a", "model_name": "retrieval_reranker", "model_type": "retrieval_reranker", "top1": 0.58},
            {"run_id": "run_a", "model_name": "gru_artist", "model_type": "deep", "top1": 0.51},
        ]
    ).to_csv(history_dir / "backtest_history.csv", index=False)
    pd.DataFrame(
        [{"run_id": "run_a", "model_name": "logreg_tuned", "base_model_name": "logreg", "val_top1": 0.53}]
    ).to_csv(history_dir / "optuna_history.csv", index=False)

    (run_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_id": "run_a",
                "run_name": "nightly",
                "profile": "full",
                "timestamp": "2026-03-22T20:00:00",
                "data_records": 1234,
                "num_artists": 87,
                "num_context_features": 12,
                "champion_gate": {
                    "status": "pass",
                    "promoted": True,
                    "metric_source": "backtest_top1",
                    "regression": -0.01,
                },
                "champion_alias": {
                    "model_name": "retrieval_reranker",
                    "model_type": "retrieval_reranker",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "run_results.json").write_text(
        json.dumps(
            [
                {
                    "model_name": "retrieval_reranker",
                    "model_type": "retrieval_reranker",
                    "val_top1": 0.61,
                    "test_top1": 0.57,
                },
                {
                    "model_name": "gru_artist",
                    "model_type": "deep",
                    "val_top1": 0.55,
                    "test_top1": 0.51,
                },
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    (analysis_dir / "retrieval_reranker_retrieval_reranker_confidence_summary.json").write_text(
        json.dumps(
            {
                "test_ece": 0.06,
                "test_selective_risk": 0.12,
                "test_abstention_rate": 0.08,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (analysis_dir / "data_drift_summary.json").write_text(
        json.dumps(
            {
                "target_drift": {"train_vs_test_jsd": 0.14},
                "largest_context_shift": {"feature": "tech_stutter_events_24h", "max_abs_std_mean_diff": 1.2},
                "largest_segment_shift": {
                    "split": "test",
                    "segment": "friction_regime",
                    "bucket": "high_friction",
                    "abs_share_shift": 0.24,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (analysis_dir / "friction_proxy_summary.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "friction_feature_count": 3,
                "proxy_counterfactual": {"test_mean_delta": 0.07},
                "top_friction_features": [{"feature": "tech_stutter_events_24h", "mean_risk_delta": 0.04}],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (analysis_dir / "robustness_summary.json").write_text(
        json.dumps(
            [
                {
                    "model_name": "retrieval_reranker",
                    "max_top1_gap": 0.19,
                    "worst_segment": "friction_regime",
                    "worst_bucket": "high_friction",
                }
            ],
            indent=2,
        ),
        encoding="utf-8",
    )
    (analysis_dir / "moonshot_summary.json").write_text(
        json.dumps(
            {
                "digital_twin_test_auc": 0.72,
                "causal_test_auc_total": 0.69,
                "stress_worst_skip_scenario": "high_friction_spike",
                "stress_worst_skip_risk": 0.41,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    report = build_control_room_report(output_dir, top_n=3)

    assert report["portfolio"]["total_runs"] == 1
    assert report["latest_run"]["run_id"] == "run_a"
    assert report["latest_run"]["promoted"] is True
    assert report["latest_run"]["best_model_name"] == "retrieval_reranker"
    assert report["safety"]["largest_context_shift_feature"] == "tech_stutter_events_24h"
    assert report["qoe"]["stress_worst_skip_scenario"] == "high_friction_spike"
    assert report["leaderboards"]["experiment_top_models"][0]["model_name"] == "retrieval_reranker"
    assert report["next_bets"]


def test_control_room_compares_latest_run_to_last_promoted_baseline(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    history_dir = output_dir / "history"
    history_dir.mkdir(parents=True)

    pd.DataFrame(
        [
            {"run_id": "run_a", "model_name": "retrieval_reranker", "model_type": "retrieval_reranker", "val_top1": 0.60},
            {"run_id": "run_b", "model_name": "blended_ensemble", "model_type": "ensemble", "val_top1": 0.54},
        ]
    ).to_csv(history_dir / "experiment_history.csv", index=False)
    pd.DataFrame(
        [
            {"run_id": "run_a", "model_name": "retrieval_reranker", "model_type": "retrieval_reranker", "top1": 0.58},
            {"run_id": "run_b", "model_name": "blended_ensemble", "model_type": "ensemble", "top1": 0.49},
        ]
    ).to_csv(history_dir / "backtest_history.csv", index=False)
    pd.DataFrame(
        [{"run_id": "run_b", "model_name": "logreg_tuned", "base_model_name": "logreg", "val_top1": 0.47}]
    ).to_csv(history_dir / "optuna_history.csv", index=False)

    _write_run_fixture(
        output_dir,
        run_id="run_a",
        timestamp="2026-03-20T20:00:00",
        profile="full",
        promoted=True,
        promotion_status="pass",
        best_model_name="retrieval_reranker",
        best_model_type="retrieval_reranker",
        val_top1=0.60,
        test_top1=0.58,
        gate_regression=-0.02,
        drift_jsd=0.11,
        ece=0.05,
        selective_risk=0.14,
        abstention_rate=0.08,
        robustness_gap=0.11,
        stress_skip_risk=0.22,
        stress_scenario="steady_evening",
    )
    _write_run_fixture(
        output_dir,
        run_id="run_b",
        timestamp="2026-03-22T20:00:00",
        profile="full",
        promoted=False,
        promotion_status="fail",
        best_model_name="blended_ensemble",
        best_model_type="ensemble",
        val_top1=0.54,
        test_top1=0.49,
        gate_regression=0.03,
        drift_jsd=0.19,
        ece=0.09,
        selective_risk=0.61,
        abstention_rate=0.0,
        robustness_gap=0.27,
        stress_skip_risk=0.44,
        stress_scenario="evening_drift",
    )

    report = build_control_room_report(output_dir, top_n=3)

    assert report["latest_run"]["run_id"] == "run_b"
    assert report["baseline_comparison"]["baseline_available"] is True
    assert report["baseline_comparison"]["baseline_run"]["run_id"] == "run_a"
    assert any(
        row["key"] == "best_model_test_top1" and row["status"] == "worse"
        for row in report["baseline_comparison"]["metric_deltas"]
    )
    assert any("worsened" in item for item in report["baseline_comparison"]["summary"])
    assert any(action["area"] == "promotion" for action in report["review_actions"])
    assert any(action["area"] == "robustness" for action in report["review_actions"])
    assert any(action["area"] == "stress_test" for action in report["review_actions"])


def test_write_control_room_report_creates_json_and_markdown(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    run_dir = output_dir / "runs" / "run_a"
    run_dir.mkdir(parents=True)
    (run_dir / "run_manifest.json").write_text(
        json.dumps({"run_id": "run_a", "profile": "core", "timestamp": "2026-03-22T20:00:00"}),
        encoding="utf-8",
    )
    (run_dir / "run_results.json").write_text(
        json.dumps([{"model_name": "mlp", "model_type": "classical", "val_top1": 0.33, "test_top1": 0.31}]),
        encoding="utf-8",
    )

    json_path, md_path = write_control_room_report(output_dir, top_n=2)

    assert json_path.exists()
    assert md_path.exists()
    assert (output_dir / "analytics" / "control_room_history.csv").exists()
    assert (output_dir / "analytics" / "control_room_weekly_summary.json").exists()
    assert (output_dir / "analytics" / "control_room_weekly_summary.md").exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["latest_run"]["run_id"] == "run_a"
    assert payload["ops_coverage"]["available_summary_count"] == 1
    assert payload["ops_coverage"]["expected_summary_count"] == 6
    assert any(action["area"] == "instrumentation" for action in payload["review_actions"])
    assert payload["ops_history"]["snapshot_count"] == 1
    assert payload["ops_trends"]["history_available"] is True
    assert payload["weekly_ops_summary"]["snapshots_considered"] == 1
    assert payload["operating_rhythm"]["overall_status"]
    assert payload["ops_health"]["status"] == "blocked"
    assert payload["async_handoff"]["status"]
    markdown = md_path.read_text(encoding="utf-8")
    assert "Control Room" in markdown
    assert "Operating Rhythm" in markdown
    assert "Ops Health" in markdown
    assert "Ops Coverage" in markdown
    assert "Since Last Strong Run" in markdown
    assert "Recent Trends" in markdown
    assert "Weekly Window" in markdown
    assert "Review Actions" in markdown
    assert "Async Handoff" in markdown


def test_control_room_history_and_weekly_summary_track_multiple_runs(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    history_dir = output_dir / "history"
    history_dir.mkdir(parents=True)

    pd.DataFrame(
        [
            {"run_id": "run_a", "model_name": "retrieval_reranker", "model_type": "retrieval_reranker", "val_top1": 0.60},
            {"run_id": "run_b", "model_name": "blended_ensemble", "model_type": "ensemble", "val_top1": 0.54},
        ]
    ).to_csv(history_dir / "experiment_history.csv", index=False)
    pd.DataFrame(
        [
            {"run_id": "run_a", "model_name": "retrieval_reranker", "model_type": "retrieval_reranker", "top1": 0.58},
            {"run_id": "run_b", "model_name": "blended_ensemble", "model_type": "ensemble", "top1": 0.49},
        ]
    ).to_csv(history_dir / "backtest_history.csv", index=False)

    _write_run_fixture(
        output_dir,
        run_id="run_a",
        timestamp="2026-03-20T20:00:00",
        profile="full",
        promoted=True,
        promotion_status="pass",
        best_model_name="retrieval_reranker",
        best_model_type="retrieval_reranker",
        val_top1=0.60,
        test_top1=0.58,
        gate_regression=-0.02,
        drift_jsd=0.11,
        ece=0.05,
        selective_risk=0.14,
        abstention_rate=0.08,
        robustness_gap=0.11,
        stress_skip_risk=0.22,
        stress_scenario="steady_evening",
    )

    write_control_room_report(output_dir, top_n=3)

    _write_run_fixture(
        output_dir,
        run_id="run_b",
        timestamp="2026-03-22T20:00:00",
        profile="full",
        promoted=False,
        promotion_status="fail",
        best_model_name="blended_ensemble",
        best_model_type="ensemble",
        val_top1=0.54,
        test_top1=0.49,
        gate_regression=0.03,
        drift_jsd=0.19,
        ece=0.09,
        selective_risk=0.61,
        abstention_rate=0.0,
        robustness_gap=0.27,
        stress_skip_risk=0.44,
        stress_scenario="evening_drift",
    )

    json_path, md_path = write_control_room_report(output_dir, top_n=3)

    history_path = output_dir / "analytics" / "control_room_history.csv"
    weekly_json_path = output_dir / "analytics" / "control_room_weekly_summary.json"
    weekly_md_path = output_dir / "analytics" / "control_room_weekly_summary.md"

    history_frame = pd.read_csv(history_path)
    assert set(history_frame["run_id"].astype(str)) == {"run_a", "run_b"}

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["ops_history"]["snapshot_count"] == 2
    assert payload["ops_trends"]["snapshot_count"] == 2
    assert payload["ops_trends"]["previous_snapshot"]["run_id"] == "run_a"
    assert any("Previous snapshot was" in item for item in payload["ops_trends"]["summary"])

    weekly_payload = json.loads(weekly_json_path.read_text(encoding="utf-8"))
    assert weekly_payload["snapshots_considered"] == 2
    assert weekly_payload["failed_promotions"] == 1
    assert weekly_md_path.exists()

    markdown = md_path.read_text(encoding="utf-8")
    assert "Recent Trends" in markdown
    assert "Weekly Window" in markdown
    assert "Previous snapshot was `run_a`" in markdown


def test_control_room_prefers_high_signal_ops_run_over_newer_smoke_run(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    history_dir = output_dir / "history"
    history_dir.mkdir(parents=True)

    pd.DataFrame(
        [
            {"run_id": "run_full", "model_name": "retrieval_reranker", "model_type": "retrieval_reranker", "val_top1": 0.60},
            {"run_id": "run_smoke_check", "model_name": "dense", "model_type": "deep", "val_top1": 0.57},
        ]
    ).to_csv(history_dir / "experiment_history.csv", index=False)
    pd.DataFrame(
        [
            {"run_id": "run_full", "model_name": "retrieval_reranker", "model_type": "retrieval_reranker", "top1": 0.58},
            {"run_id": "run_smoke_check", "model_name": "dense", "model_type": "deep", "top1": 0.54},
        ]
    ).to_csv(history_dir / "backtest_history.csv", index=False)

    _write_run_fixture(
        output_dir,
        run_id="run_full",
        timestamp="2026-03-20T20:00:00",
        profile="full",
        promoted=True,
        promotion_status="pass",
        best_model_name="retrieval_reranker",
        best_model_type="retrieval_reranker",
        val_top1=0.60,
        test_top1=0.58,
        gate_regression=-0.02,
        drift_jsd=0.11,
        ece=0.05,
        selective_risk=0.14,
        abstention_rate=0.08,
        robustness_gap=0.11,
        stress_skip_risk=0.22,
        stress_scenario="steady_evening",
    )
    _write_run_fixture(
        output_dir,
        run_id="run_smoke_check",
        timestamp="2026-03-22T20:00:00",
        profile="dev",
        promoted=True,
        promotion_status="pass",
        best_model_name="dense",
        best_model_type="deep",
        val_top1=0.57,
        test_top1=0.54,
        gate_regression=-0.01,
        drift_jsd=0.10,
        ece=0.05,
        selective_risk=0.15,
        abstention_rate=0.09,
        robustness_gap=0.09,
        stress_skip_risk=0.18,
        stress_scenario="quick_probe",
    )

    report = build_control_room_report(output_dir, top_n=3)

    assert report["latest_run"]["run_id"] == "run_full"
    assert report["run_selection"]["latest_observed_run"]["run_id"] == "run_smoke_check"
    assert report["run_selection"]["selected_run"]["run_id"] == "run_full"
    assert report["run_selection"]["observed_matches_selected"] is False
    assert "smoke/check run" in report["run_selection"]["selection_reason"]


def test_control_room_operating_rhythm_flags_stale_full_lane_and_async_handoff(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    history_dir = output_dir / "history"
    history_dir.mkdir(parents=True)

    pd.DataFrame(
        [
            {"run_id": "run_full_old", "model_name": "retrieval_reranker", "model_type": "retrieval_reranker", "val_top1": 0.60},
            {"run_id": "run_fast_recent", "model_name": "mlp", "model_type": "classical", "val_top1": 0.57},
        ]
    ).to_csv(history_dir / "experiment_history.csv", index=False)
    pd.DataFrame(
        [
            {"run_id": "run_full_old", "model_name": "retrieval_reranker", "model_type": "retrieval_reranker", "top1": 0.58},
            {"run_id": "run_fast_recent", "model_name": "mlp", "model_type": "classical", "top1": 0.55},
        ]
    ).to_csv(history_dir / "backtest_history.csv", index=False)

    _write_run_fixture(
        output_dir,
        run_id="run_full_old",
        timestamp="2026-03-01T20:00:00+00:00",
        profile="full",
        promoted=True,
        promotion_status="pass",
        best_model_name="retrieval_reranker",
        best_model_type="retrieval_reranker",
        val_top1=0.60,
        test_top1=0.58,
        gate_regression=-0.02,
        drift_jsd=0.11,
        ece=0.05,
        selective_risk=0.14,
        abstention_rate=0.08,
        robustness_gap=0.11,
        stress_skip_risk=0.22,
        stress_scenario="steady_evening",
    )
    _write_run_fixture(
        output_dir,
        run_id="run_fast_recent",
        timestamp="2026-03-28T18:00:00+00:00",
        profile="fast",
        promoted=True,
        promotion_status="pass",
        best_model_name="mlp",
        best_model_type="classical",
        val_top1=0.57,
        test_top1=0.55,
        gate_regression=-0.01,
        drift_jsd=0.10,
        ece=0.05,
        selective_risk=0.15,
        abstention_rate=0.09,
        robustness_gap=0.09,
        stress_skip_risk=0.18,
        stress_scenario="quick_probe",
    )

    report = build_control_room_report(
        output_dir,
        top_n=3,
        reference_time=datetime(2026, 3, 29, 12, 0, tzinfo=timezone.utc),
    )

    assert report["latest_run"]["run_id"] == "run_fast_recent"
    assert report["operating_rhythm"]["overall_status"] in {"attention", "stale"}
    assert report["operating_rhythm"]["lanes"]["fast"]["status"] == "healthy"
    assert report["operating_rhythm"]["lanes"]["full"]["status"] == "stale"
    assert report["operating_rhythm"]["recommended_run_command"] == "make schedule-run MODE=full"
    assert any(action["area"] == "cadence" for action in report["review_actions"])
    assert report["async_handoff"]["status"] == "attention"
    assert any("make schedule-run MODE=full" in item for item in report["async_handoff"]["summary"])


def test_control_room_distinguishes_ops_health_from_strategic_findings(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    history_dir = output_dir / "history"
    history_dir.mkdir(parents=True)

    pd.DataFrame(
        [
            {"run_id": "run_full_recent", "model_name": "blended_ensemble", "model_type": "ensemble", "val_top1": 0.63},
            {"run_id": "run_fast_recent", "model_name": "mlp", "model_type": "classical", "val_top1": 0.57},
        ]
    ).to_csv(history_dir / "experiment_history.csv", index=False)
    pd.DataFrame(
        [
            {"run_id": "run_full_recent", "model_name": "blended_ensemble", "model_type": "ensemble", "top1": 0.60},
            {"run_id": "run_fast_recent", "model_name": "mlp", "model_type": "classical", "top1": 0.55},
        ]
    ).to_csv(history_dir / "backtest_history.csv", index=False)

    _write_run_fixture(
        output_dir,
        run_id="run_full_recent",
        timestamp="2026-03-29T11:00:00+00:00",
        profile="full",
        promoted=False,
        promotion_status="fail",
        best_model_name="blended_ensemble",
        best_model_type="ensemble",
        val_top1=0.63,
        test_top1=0.58,
        gate_regression=0.02,
        drift_jsd=0.11,
        ece=0.05,
        selective_risk=0.18,
        abstention_rate=0.08,
        robustness_gap=0.22,
        stress_skip_risk=0.29,
        stress_scenario="evening_drift",
    )
    _write_run_fixture(
        output_dir,
        run_id="run_fast_recent",
        timestamp="2026-03-29T08:00:00+00:00",
        profile="fast",
        promoted=True,
        promotion_status="pass",
        best_model_name="mlp",
        best_model_type="classical",
        val_top1=0.57,
        test_top1=0.55,
        gate_regression=-0.01,
        drift_jsd=0.10,
        ece=0.05,
        selective_risk=0.15,
        abstention_rate=0.09,
        robustness_gap=0.09,
        stress_skip_risk=0.18,
        stress_scenario="quick_probe",
    )

    report = build_control_room_report(
        output_dir,
        top_n=3,
        reference_time=datetime(2026, 3, 29, 12, 0, tzinfo=timezone.utc),
    )

    assert report["operating_rhythm"]["overall_status"] == "healthy"
    assert report["ops_health"]["status"] == "healthy"
    assert report["ops_health"]["strategic_high_priority_count"] >= 1
    assert report["ops_health"]["operational_high_priority_count"] == 0
    assert report["async_handoff"]["status"] == "ready"


def test_control_room_fast_lane_ignores_dev_runs_for_cadence(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    history_dir = output_dir / "history"
    history_dir.mkdir(parents=True)

    pd.DataFrame(
        [
            {"run_id": "run_full_recent", "model_name": "retrieval_reranker", "model_type": "retrieval_reranker", "val_top1": 0.60},
            {"run_id": "run_dev_recent", "model_name": "dense", "model_type": "deep", "val_top1": 0.57},
        ]
    ).to_csv(history_dir / "experiment_history.csv", index=False)
    pd.DataFrame(
        [
            {"run_id": "run_full_recent", "model_name": "retrieval_reranker", "model_type": "retrieval_reranker", "top1": 0.58},
            {"run_id": "run_dev_recent", "model_name": "dense", "model_type": "deep", "top1": 0.54},
        ]
    ).to_csv(history_dir / "backtest_history.csv", index=False)

    _write_run_fixture(
        output_dir,
        run_id="run_full_recent",
        timestamp="2026-03-29T10:00:00+00:00",
        profile="full",
        promoted=True,
        promotion_status="pass",
        best_model_name="retrieval_reranker",
        best_model_type="retrieval_reranker",
        val_top1=0.60,
        test_top1=0.58,
        gate_regression=-0.01,
        drift_jsd=0.10,
        ece=0.05,
        selective_risk=0.15,
        abstention_rate=0.09,
        robustness_gap=0.09,
        stress_skip_risk=0.18,
        stress_scenario="steady_evening",
    )
    _write_run_fixture(
        output_dir,
        run_id="run_dev_recent",
        timestamp="2026-03-29T11:30:00+00:00",
        profile="dev",
        promoted=True,
        promotion_status="pass",
        best_model_name="dense",
        best_model_type="deep",
        val_top1=0.57,
        test_top1=0.54,
        gate_regression=-0.01,
        drift_jsd=0.10,
        ece=0.05,
        selective_risk=0.15,
        abstention_rate=0.09,
        robustness_gap=0.09,
        stress_skip_risk=0.18,
        stress_scenario="quick_probe",
    )

    report = build_control_room_report(
        output_dir,
        top_n=3,
        reference_time=datetime(2026, 3, 29, 12, 0, tzinfo=timezone.utc),
    )

    assert report["operating_rhythm"]["lanes"]["fast"]["status"] == "missing"
