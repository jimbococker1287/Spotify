from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from spotify.quant_decision_lab import build_quant_decision_lab


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_quant_decision_lab_creates_frontiers_and_brief(tmp_path: Path) -> None:
    logger = logging.getLogger("spotify.test.quant_decision_lab")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())

    output_dir = tmp_path / "outputs"
    run_dir = output_dir / "runs" / "run_a"
    analysis_dir = run_dir / "analysis"
    stress_dir = analysis_dir / "stress_test"
    stress_dir.mkdir(parents=True, exist_ok=True)

    _write_json(
        output_dir / "analytics" / "control_room.json",
        {
            "run_selection": {
                "selected_run": {"run_id": "run_a"},
            }
        },
    )
    _write_json(
        run_dir / "run_results.json",
        [
            {
                "model_name": "retrieval_reranker",
                "model_type": "retrieval_reranker",
                "model_family": "candidate_reranker",
                "test_top1": 0.45,
                "val_top1": 0.40,
                "fit_seconds": 100.0,
            },
            {
                "model_name": "mlp",
                "model_type": "classical",
                "model_family": "shallow_neural",
                "test_top1": 0.30,
                "val_top1": 0.33,
                "fit_seconds": 10.0,
            },
        ],
    )
    _write_json(
        analysis_dir / "policy_simulation_summary.json",
        [
            {
                "model_name": "retrieval_reranker",
                "test_hit_at_k": 0.66,
                "test_discounted_reward": 0.023,
                "test_expected_utility_mass": 0.037,
            },
            {
                "model_name": "mlp",
                "test_hit_at_k": 0.50,
                "test_discounted_reward": 0.018,
                "test_expected_utility_mass": 0.030,
            },
        ],
    )
    _write_json(
        analysis_dir / "retrieval_reranker_retrieval_reranker_conformal_summary.json",
        {
            "tag": "retrieval_reranker_retrieval_reranker",
            "operating_point": {"target_selective_risk": 0.4, "min_accepted_rate": 0.25, "abstention_threshold": 0.10},
            "val": {"selective_risk": 0.39, "abstention_rate": 0.21},
            "test": {"selective_risk": 0.38, "abstention_rate": 0.22, "accepted_rate": 0.78},
        },
    )
    _write_json(
        analysis_dir / "classical_mlp_conformal_summary.json",
        {
            "tag": "classical_mlp",
            "operating_point": {"target_selective_risk": 0.4, "min_accepted_rate": 0.25, "abstention_threshold": 0.12},
            "val": {"selective_risk": 0.50, "abstention_rate": 0.28},
            "test": {"selective_risk": 0.52, "abstention_rate": 0.30, "accepted_rate": 0.70},
        },
    )
    _write_json(
        stress_dir / "stress_test_summary.json",
        [
            {
                "scenario": "baseline",
                "policy_name": "baseline_exploit",
                "policy_family": "baseline",
                "mean_session_length": 1.2,
                "mean_skip_risk": 0.60,
                "mean_end_risk": 0.90,
            },
            {
                "scenario": "baseline",
                "policy_name": "safe_global",
                "policy_family": "safe",
                "mean_session_length": 1.1,
                "mean_skip_risk": 0.55,
                "mean_end_risk": 0.88,
            },
            {
                "scenario": "evening_drift",
                "policy_name": "baseline_exploit",
                "policy_family": "baseline",
                "mean_session_length": 1.0,
                "mean_skip_risk": 0.68,
                "mean_end_risk": 0.97,
            },
            {
                "scenario": "evening_drift",
                "policy_name": "safe_routed_evening",
                "policy_family": "safe",
                "mean_session_length": 1.08,
                "mean_skip_risk": 0.57,
                "mean_end_risk": 0.90,
            },
        ],
    )
    _write_json(
        stress_dir / "stress_test_benchmark.json",
        {
            "benchmark_policy_name": "safe_global",
            "benchmark_selected_policy_name": "safe_routed_evening",
            "benchmark_scenario": "evening_drift",
        },
    )
    _write_json(
        analysis_dir / "data_drift_summary.json",
        {"target_drift": {"train_vs_test_jsd": 0.218}},
    )

    paths = build_quant_decision_lab(output_dir=output_dir, run_dir=None, logger=logger)

    assert paths
    result_root = output_dir / "analysis" / "quant_decision_lab"
    model_frontier = pd.read_csv(result_root / "model_decision_frontier.csv")
    policy_frontier = pd.read_csv(result_root / "policy_decision_frontier.csv")
    scenario_sensitivity = pd.read_csv(result_root / "scenario_sensitivity.csv")
    brief_text = (result_root / "quant_decision_brief.md").read_text(encoding="utf-8")

    assert model_frontier.iloc[0]["model_name"] == "retrieval_reranker"
    assert bool(policy_frontier["is_pareto_efficient"].any())
    assert scenario_sensitivity.iloc[0]["scenario"] == "evening_drift"
    assert "Quant Decision Brief" in brief_text
