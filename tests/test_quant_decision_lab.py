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
    _write_json(
        output_dir / "analysis" / "listener_archetypes" / "taste_state_brief.json",
        {
            "status": "ok",
            "dominant_archetype": "steady_replay",
            "dominant_archetype_days": 14,
            "highest_skip_archetype": "skip_surfing",
            "highest_skip_rate": 0.61,
            "highest_exploration_archetype": "exploratory_shuffle",
            "highest_exploration_ratio": 0.72,
            "summary": [
                "Dominant listener archetype is `steady_replay` across `14` days.",
                "Highest-skip archetype is `skip_surfing` at skip rate `0.610`.",
                "Most exploratory archetype is `exploratory_shuffle` with exploration ratio `0.720`.",
            ],
            "actions": [
                "Use the dominant archetype when choosing the default Taste OS mode for a local demo.",
                "Use the highest-skip archetype as the first slice for skip-risk and recovery experiments.",
                "Use the most exploratory archetype when testing discovery or novelty-routing ideas.",
            ],
        },
    )
    _write_json(
        output_dir / "analysis" / "listener_archetypes" / "listener_archetype_summary.json",
        {
            "cluster_count": 3,
            "day_count": 28,
            "archetypes": [
                {
                    "cluster_id": 0,
                    "day_count": 14,
                    "skip_rate": 0.42,
                    "exploration_ratio": 0.31,
                    "archetype_label": "steady_replay",
                },
                {
                    "cluster_id": 1,
                    "day_count": 8,
                    "skip_rate": 0.61,
                    "exploration_ratio": 0.44,
                    "archetype_label": "skip_surfing",
                },
                {
                    "cluster_id": 2,
                    "day_count": 6,
                    "skip_rate": 0.22,
                    "exploration_ratio": 0.72,
                    "archetype_label": "exploratory_shuffle",
                },
            ],
        },
    )

    paths = build_quant_decision_lab(output_dir=output_dir, run_dir=None, logger=logger)

    assert paths
    result_root = output_dir / "analysis" / "quant_decision_lab"
    model_frontier = pd.read_csv(result_root / "model_decision_frontier.csv")
    policy_frontier = pd.read_csv(result_root / "policy_decision_frontier.csv")
    scenario_sensitivity = pd.read_csv(result_root / "scenario_sensitivity.csv")
    brief_text = (result_root / "quant_decision_brief.md").read_text(encoding="utf-8")
    bridge_payload = json.loads((result_root / "archetype_decision_bridge.json").read_text(encoding="utf-8"))
    bridge_text = (result_root / "archetype_decision_bridge.md").read_text(encoding="utf-8")

    assert model_frontier.iloc[0]["model_name"] == "retrieval_reranker"
    assert bool(policy_frontier["is_pareto_efficient"].any())
    assert scenario_sensitivity.iloc[0]["scenario"] == "evening_drift"
    assert "Quant Decision Brief" in brief_text
    assert bridge_payload["status"] == "ok"
    assert bridge_payload["listener_archetypes_available"] is True
    assert {row["role"] for row in bridge_payload["archetype_recommendations"]} == {
        "dominant",
        "high_skip",
        "exploratory",
    }
    dominant = next(row for row in bridge_payload["archetype_recommendations"] if row["role"] == "dominant")
    high_skip = next(row for row in bridge_payload["archetype_recommendations"] if row["role"] == "high_skip")
    exploratory = next(row for row in bridge_payload["archetype_recommendations"] if row["role"] == "exploratory")
    assert dominant["archetype_label"] == "steady_replay"
    assert dominant["recommended_model"]["model_name"] == "retrieval_reranker"
    assert dominant["scenario_focus"]["scenario"] == "baseline"
    assert high_skip["archetype_label"] == "skip_surfing"
    assert high_skip["scenario_focus"]["scenario"] == "evening_drift"
    assert exploratory["archetype_label"] == "exploratory_shuffle"
    assert exploratory["recommended_model"]["model_name"] == "retrieval_reranker"
    assert "Archetype Decision Bridge" in bridge_text
    assert "steady_replay" in bridge_text


def test_build_quant_decision_lab_marks_bridge_missing_without_listener_outputs(tmp_path: Path) -> None:
    logger = logging.getLogger("spotify.test.quant_decision_lab_missing_bridge")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())

    output_dir = tmp_path / "outputs"
    run_dir = output_dir / "runs" / "run_b"
    analysis_dir = run_dir / "analysis"
    stress_dir = analysis_dir / "stress_test"
    stress_dir.mkdir(parents=True, exist_ok=True)

    _write_json(
        output_dir / "analytics" / "control_room.json",
        {
            "run_selection": {
                "selected_run": {"run_id": "run_b"},
            }
        },
    )
    _write_json(
        run_dir / "run_results.json",
        [
            {
                "model_name": "mlp",
                "model_type": "classical",
                "model_family": "shallow_neural",
                "test_top1": 0.31,
                "val_top1": 0.30,
                "fit_seconds": 11.0,
            }
        ],
    )
    _write_json(
        analysis_dir / "policy_simulation_summary.json",
        [
            {
                "model_name": "mlp",
                "test_hit_at_k": 0.52,
                "test_discounted_reward": 0.017,
                "test_expected_utility_mass": 0.028,
            }
        ],
    )
    _write_json(
        analysis_dir / "classical_mlp_conformal_summary.json",
        {
            "tag": "classical_mlp",
            "operating_point": {"target_selective_risk": 0.4, "min_accepted_rate": 0.25, "abstention_threshold": 0.12},
            "val": {"selective_risk": 0.41, "abstention_rate": 0.20},
            "test": {"selective_risk": 0.39, "abstention_rate": 0.22, "accepted_rate": 0.78},
        },
    )
    _write_json(
        stress_dir / "stress_test_summary.json",
        [
            {
                "scenario": "baseline",
                "policy_name": "safe_global",
                "policy_family": "safe",
                "mean_session_length": 1.1,
                "mean_skip_risk": 0.55,
                "mean_end_risk": 0.88,
            }
        ],
    )
    _write_json(analysis_dir / "data_drift_summary.json", {"target_drift": {"train_vs_test_jsd": 0.101}})

    build_quant_decision_lab(output_dir=output_dir, run_dir=None, logger=logger)

    bridge_payload = json.loads(
        (output_dir / "analysis" / "quant_decision_lab" / "archetype_decision_bridge.json").read_text(encoding="utf-8")
    )

    assert bridge_payload["status"] == "listener_archetypes_missing"
    assert bridge_payload["listener_archetypes_available"] is False
    assert bridge_payload["archetype_recommendations"] == []
