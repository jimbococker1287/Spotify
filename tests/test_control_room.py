from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from spotify.control_room import build_control_room_report, write_control_room_report


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
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["latest_run"]["run_id"] == "run_a"
    assert "Control Room" in md_path.read_text(encoding="utf-8")
