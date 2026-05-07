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
    _write_json(
        output_dir / "analysis" / "listener_archetypes" / "taste_evolution_brief.json",
        {
            "status": "ok",
            "biggest_regime_shift": {
                "month": "2026-04",
                "previous_month": "2026-03",
                "dominant_archetype": "skip_surfing",
                "previous_dominant_archetype": "steady_replay",
                "regime_shift_score": 0.44,
                "top_share_gain_archetype": "skip_surfing",
                "top_share_loss_archetype": "steady_replay",
            },
            "most_seasonal_archetype": {
                "archetype_label": "exploratory_shuffle",
                "seasonality_gap": 0.33,
                "peak_season": "spring",
                "peak_season_share": 0.55,
            },
            "top_transition": {
                "from_archetype": "steady_replay",
                "to_archetype": "steady_replay",
                "transition_count": 8,
                "transition_share": 0.31,
            },
            "top_cross_state_transition": {
                "from_archetype": "steady_replay",
                "to_archetype": "exploratory_shuffle",
                "transition_count": 4,
                "transition_share": 0.18,
            },
            "summary": [
                "Largest month-over-month regime shift lands in `2026-04`, where dominance moved from `steady_replay` to `skip_surfing`.",
                "Most seasonal archetype is `exploratory_shuffle`, peaking in `spring`.",
                "Largest cross-state transition is `steady_replay -> exploratory_shuffle`.",
            ],
            "actions": [
                "Use the highest regime-shift month as the first retrospective slice when explaining taste evolution changes over time.",
                "Use the most seasonal archetype as the calendar-aware slice for playlist, context, or discovery experiments.",
                "Use the biggest cross-state transition as the handoff sequence when designing taste-state recovery or escalation paths.",
            ],
        },
    )
    _write_json(
        output_dir / "analysis" / "listener_archetypes" / "taste_evolution_regime_shifts.json",
        [
            {
                "month": "2026-01",
                "previous_month": "",
                "dominant_archetype": "steady_replay",
                "previous_dominant_archetype": "",
                "dominant_share": 0.72,
                "runner_up_archetype": "exploratory_shuffle",
                "runner_up_share": 0.18,
                "dominance_gap": 0.54,
                "active_archetype_count": 2,
                "dominant_changed_from_prev_month": False,
                "regime_shift_score": 0.0,
                "top_share_gain_archetype": "",
                "top_share_gain_delta": 0.0,
                "top_share_loss_archetype": "",
                "top_share_loss_delta": 0.0,
                "dominant_mean_skip_rate": 0.11,
                "dominant_mean_exploration_ratio": 0.28,
            },
            {
                "month": "2026-02",
                "previous_month": "2026-01",
                "dominant_archetype": "exploratory_shuffle",
                "previous_dominant_archetype": "steady_replay",
                "dominant_share": 0.51,
                "runner_up_archetype": "steady_replay",
                "runner_up_share": 0.34,
                "dominance_gap": 0.17,
                "active_archetype_count": 3,
                "dominant_changed_from_prev_month": True,
                "regime_shift_score": 0.31,
                "top_share_gain_archetype": "exploratory_shuffle",
                "top_share_gain_delta": 0.22,
                "top_share_loss_archetype": "steady_replay",
                "top_share_loss_delta": -0.18,
                "dominant_mean_skip_rate": 0.19,
                "dominant_mean_exploration_ratio": 0.71,
            },
            {
                "month": "2026-03",
                "previous_month": "2026-02",
                "dominant_archetype": "steady_replay",
                "previous_dominant_archetype": "exploratory_shuffle",
                "dominant_share": 0.49,
                "runner_up_archetype": "exploratory_shuffle",
                "runner_up_share": 0.27,
                "dominance_gap": 0.22,
                "active_archetype_count": 3,
                "dominant_changed_from_prev_month": True,
                "regime_shift_score": 0.27,
                "top_share_gain_archetype": "steady_replay",
                "top_share_gain_delta": 0.16,
                "top_share_loss_archetype": "exploratory_shuffle",
                "top_share_loss_delta": -0.14,
                "dominant_mean_skip_rate": 0.12,
                "dominant_mean_exploration_ratio": 0.30,
            },
            {
                "month": "2026-04",
                "previous_month": "2026-03",
                "dominant_archetype": "skip_surfing",
                "previous_dominant_archetype": "steady_replay",
                "dominant_share": 0.57,
                "runner_up_archetype": "exploratory_shuffle",
                "runner_up_share": 0.23,
                "dominance_gap": 0.34,
                "active_archetype_count": 3,
                "dominant_changed_from_prev_month": True,
                "regime_shift_score": 0.44,
                "top_share_gain_archetype": "skip_surfing",
                "top_share_gain_delta": 0.29,
                "top_share_loss_archetype": "steady_replay",
                "top_share_loss_delta": -0.26,
                "dominant_mean_skip_rate": 0.62,
                "dominant_mean_exploration_ratio": 0.42,
            },
        ],
    )
    _write_json(
        output_dir / "analysis" / "listener_archetypes" / "listener_archetype_seasonal.json",
        [
            {
                "season_label": "2026-winter",
                "season_year": 2026,
                "season": "winter",
                "archetype_label": "steady_replay",
                "day_count": 12,
                "archetype_share": 0.60,
                "mean_streams": 42.0,
                "mean_skip_rate": 0.12,
                "mean_exploration_ratio": 0.30,
                "start_date": "2026-01-01",
                "end_date": "2026-02-28",
            },
            {
                "season_label": "2026-winter",
                "season_year": 2026,
                "season": "winter",
                "archetype_label": "exploratory_shuffle",
                "day_count": 5,
                "archetype_share": 0.25,
                "mean_streams": 13.0,
                "mean_skip_rate": 0.18,
                "mean_exploration_ratio": 0.71,
                "start_date": "2026-01-01",
                "end_date": "2026-02-28",
            },
            {
                "season_label": "2026-spring",
                "season_year": 2026,
                "season": "spring",
                "archetype_label": "exploratory_shuffle",
                "day_count": 9,
                "archetype_share": 0.55,
                "mean_streams": 14.0,
                "mean_skip_rate": 0.22,
                "mean_exploration_ratio": 0.74,
                "start_date": "2026-03-01",
                "end_date": "2026-05-31",
            },
            {
                "season_label": "2026-spring",
                "season_year": 2026,
                "season": "spring",
                "archetype_label": "skip_surfing",
                "day_count": 8,
                "archetype_share": 0.35,
                "mean_streams": 18.0,
                "mean_skip_rate": 0.60,
                "mean_exploration_ratio": 0.44,
                "start_date": "2026-03-01",
                "end_date": "2026-05-31",
            },
        ],
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
    assert bridge_payload["listener_archetype_source"]["lifecycle_available"] is True
    assert bridge_payload["listener_archetype_source"]["biggest_regime_shift"]["month"] == "2026-04"
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
    assert "regime_shift_active" in dominant["recommended_policy"]["lifecycle_signals"]
    assert "steady_replay -> exploratory_shuffle" in dominant["scenario_focus"]["lifecycle_annotation"]
    assert high_skip["archetype_label"] == "skip_surfing"
    assert high_skip["scenario_focus"]["scenario"] == "evening_drift"
    assert "biggest_regime_shift" in high_skip["recommended_policy"]["lifecycle_signals"]
    assert "2026-04" in high_skip["scenario_focus"]["lifecycle_annotation"]
    assert exploratory["archetype_label"] == "exploratory_shuffle"
    assert exploratory["recommended_model"]["model_name"] == "retrieval_reranker"
    assert "most_seasonal_archetype" in exploratory["recommended_policy"]["lifecycle_signals"]
    assert "spring" in exploratory["scenario_focus"]["lifecycle_annotation"]
    assert "Archetype Decision Bridge" in bridge_text
    assert "steady_replay" in bridge_text
    assert "Policy lifecycle:" in bridge_text
    assert "Scenario lifecycle:" in bridge_text


def test_build_quant_decision_lab_writes_scenario_utility_simulation(tmp_path: Path) -> None:
    logger = logging.getLogger("spotify.test.quant_decision_lab_utility_sim")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())

    output_dir = tmp_path / "outputs"
    run_dir = output_dir / "runs" / "sim_run"
    analysis_dir = run_dir / "analysis"
    stress_dir = analysis_dir / "stress_test"
    stress_dir.mkdir(parents=True, exist_ok=True)

    _write_json(
        output_dir / "analytics" / "control_room.json",
        {"run_selection": {"selected_run": {"run_id": "sim_run"}}},
    )
    _write_json(
        run_dir / "run_results.json",
        [
            {
                "model_name": "safe_model",
                "model_type": "classical",
                "model_family": "calibrated",
                "test_top1": 0.52,
                "val_top1": 0.49,
                "fit_seconds": 18.0,
            },
            {
                "model_name": "risky_model",
                "model_type": "classical",
                "model_family": "fast",
                "test_top1": 0.48,
                "val_top1": 0.46,
                "fit_seconds": 8.0,
            },
        ],
    )
    _write_json(
        analysis_dir / "policy_simulation_summary.json",
        [
            {
                "model_name": "safe_model",
                "test_hit_at_k": 0.70,
                "test_discounted_reward": 0.030,
                "test_expected_utility_mass": 0.041,
            },
            {
                "model_name": "risky_model",
                "test_hit_at_k": 0.62,
                "test_discounted_reward": 0.025,
                "test_expected_utility_mass": 0.036,
            },
        ],
    )
    _write_json(
        analysis_dir / "classical_safe_model_conformal_summary.json",
        {
            "tag": "classical_safe_model",
            "operating_point": {"target_selective_risk": 0.4, "min_accepted_rate": 0.25, "abstention_threshold": 0.10},
            "val": {"selective_risk": 0.25, "abstention_rate": 0.18},
            "test": {"selective_risk": 0.24, "abstention_rate": 0.19, "accepted_rate": 0.81},
        },
    )
    _write_json(
        analysis_dir / "classical_risky_model_conformal_summary.json",
        {
            "tag": "classical_risky_model",
            "operating_point": {"target_selective_risk": 0.4, "min_accepted_rate": 0.25, "abstention_threshold": 0.10},
            "val": {"selective_risk": 0.50, "abstention_rate": 0.25},
            "test": {"selective_risk": 0.53, "abstention_rate": 0.28, "accepted_rate": 0.72},
        },
    )
    _write_json(
        stress_dir / "stress_test_summary.json",
        [
            {
                "scenario": "baseline",
                "policy_name": "safe_policy",
                "policy_family": "safe",
                "mean_session_length": 1.12,
                "mean_skip_risk": 0.42,
                "mean_end_risk": 0.79,
            },
            {
                "scenario": "baseline",
                "policy_name": "baseline_exploit",
                "policy_family": "baseline",
                "mean_session_length": 1.20,
                "mean_skip_risk": 0.58,
                "mean_end_risk": 0.88,
            },
            {
                "scenario": "drift_spike",
                "policy_name": "safe_policy",
                "policy_family": "safe",
                "mean_session_length": 1.06,
                "mean_skip_risk": 0.50,
                "mean_end_risk": 0.84,
            },
            {
                "scenario": "drift_spike",
                "policy_name": "baseline_exploit",
                "policy_family": "baseline",
                "mean_session_length": 0.92,
                "mean_skip_risk": 0.72,
                "mean_end_risk": 0.96,
            },
        ],
    )
    _write_json(
        analysis_dir / "data_drift_summary.json",
        {"target_drift": {"train_vs_test_jsd": 0.241}},
    )
    _write_json(
        output_dir / "analysis" / "listener_archetypes" / "taste_state_brief.json",
        {
            "status": "ok",
            "dominant_archetype": "steady_replay",
            "dominant_archetype_days": 9,
            "highest_skip_archetype": "skip_surfing",
            "highest_skip_rate": 0.66,
            "highest_exploration_archetype": "exploratory_shuffle",
            "highest_exploration_ratio": 0.68,
            "summary": [],
            "actions": [],
        },
    )
    _write_json(
        output_dir / "analysis" / "listener_archetypes" / "listener_archetype_summary.json",
        {
            "cluster_count": 3,
            "day_count": 20,
            "archetypes": [
                {"cluster_id": 0, "archetype_label": "steady_replay", "skip_rate": 0.30, "exploration_ratio": 0.24},
                {"cluster_id": 1, "archetype_label": "skip_surfing", "skip_rate": 0.66, "exploration_ratio": 0.39},
                {"cluster_id": 2, "archetype_label": "exploratory_shuffle", "skip_rate": 0.29, "exploration_ratio": 0.68},
            ],
        },
    )
    _write_json(
        output_dir / "analysis" / "listener_archetypes" / "taste_evolution_brief.json",
        {
            "status": "ok",
            "biggest_regime_shift": {
                "month": "2026-04",
                "previous_month": "2026-03",
                "dominant_archetype": "skip_surfing",
                "previous_dominant_archetype": "steady_replay",
                "regime_shift_score": 0.40,
                "top_share_gain_archetype": "skip_surfing",
                "top_share_loss_archetype": "steady_replay",
            },
            "summary": [],
            "actions": [],
        },
    )

    paths = build_quant_decision_lab(output_dir=output_dir, run_dir=None, logger=logger)

    result_root = output_dir / "analysis" / "quant_decision_lab"
    simulation_csv = pd.read_csv(result_root / "scenario_utility_simulation.csv")
    simulation_payload = json.loads((result_root / "scenario_utility_simulation.json").read_text(encoding="utf-8"))
    simulation_text = (result_root / "scenario_utility_simulation.md").read_text(encoding="utf-8")

    assert result_root / "scenario_utility_simulation.csv" in paths
    assert result_root / "scenario_utility_simulation.json" in paths
    assert result_root / "scenario_utility_simulation.md" in paths
    assert simulation_payload["status"] == "ok"
    assert simulation_payload["row_count"] == len(simulation_csv.index)
    assert simulation_payload["score_formula"]["weights"]["archetype_bridge_alignment"] == 0.10
    assert simulation_csv["utility_score"].tolist() == sorted(simulation_csv["utility_score"].tolist(), reverse=True)
    assert simulation_csv.iloc[0]["model_name"] == "safe_model"
    assert simulation_csv.iloc[0]["policy_name"] == "safe_policy"
    assert simulation_csv.iloc[0]["scenario"] == "baseline"

    drift_row = simulation_csv.loc[simulation_csv["scenario"] == "drift_spike"].sort_values(
        "utility_score",
        ascending=False,
    ).iloc[0]
    assert bool(drift_row["high_skip_context"])
    assert bool(drift_row["high_drift_context"])
    assert "High-skip bridge context" in drift_row["notes"]
    assert "High-drift context" in drift_row["notes"]
    assert "biggest_regime_shift" in drift_row["lifecycle_signals"]
    assert "Scenario Utility Simulation" in simulation_text
    assert "safe skip improvement" in simulation_text


def test_build_quant_decision_lab_bridge_stays_ok_without_lifecycle_outputs(tmp_path: Path) -> None:
    logger = logging.getLogger("spotify.test.quant_decision_lab_no_lifecycle")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())

    output_dir = tmp_path / "outputs"
    run_dir = output_dir / "runs" / "run_c"
    analysis_dir = run_dir / "analysis"
    stress_dir = analysis_dir / "stress_test"
    stress_dir.mkdir(parents=True, exist_ok=True)

    _write_json(
        output_dir / "analytics" / "control_room.json",
        {
            "run_selection": {
                "selected_run": {"run_id": "run_c"},
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
                "test_top1": 0.42,
                "val_top1": 0.39,
                "fit_seconds": 21.0,
            }
        ],
    )
    _write_json(
        analysis_dir / "policy_simulation_summary.json",
        [
            {
                "model_name": "retrieval_reranker",
                "test_hit_at_k": 0.62,
                "test_discounted_reward": 0.021,
                "test_expected_utility_mass": 0.034,
            }
        ],
    )
    _write_json(
        analysis_dir / "retrieval_reranker_retrieval_reranker_conformal_summary.json",
        {
            "tag": "retrieval_reranker_retrieval_reranker",
            "operating_point": {"target_selective_risk": 0.4, "min_accepted_rate": 0.25, "abstention_threshold": 0.10},
            "val": {"selective_risk": 0.39, "abstention_rate": 0.20},
            "test": {"selective_risk": 0.36, "abstention_rate": 0.21, "accepted_rate": 0.79},
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
                "mean_skip_risk": 0.54,
                "mean_end_risk": 0.87,
            }
        ],
    )
    _write_json(analysis_dir / "data_drift_summary.json", {"target_drift": {"train_vs_test_jsd": 0.144}})
    _write_json(
        output_dir / "analysis" / "listener_archetypes" / "taste_state_brief.json",
        {
            "status": "ok",
            "dominant_archetype": "steady_replay",
            "dominant_archetype_days": 10,
            "highest_skip_archetype": "skip_surfing",
            "highest_skip_rate": 0.59,
            "highest_exploration_archetype": "exploratory_shuffle",
            "highest_exploration_ratio": 0.70,
            "summary": [],
            "actions": [],
        },
    )
    _write_json(
        output_dir / "analysis" / "listener_archetypes" / "listener_archetype_summary.json",
        {
            "cluster_count": 3,
            "day_count": 24,
            "archetypes": [
                {"cluster_id": 0, "archetype_label": "steady_replay", "skip_rate": 0.41, "exploration_ratio": 0.30},
                {"cluster_id": 1, "archetype_label": "skip_surfing", "skip_rate": 0.59, "exploration_ratio": 0.43},
                {"cluster_id": 2, "archetype_label": "exploratory_shuffle", "skip_rate": 0.20, "exploration_ratio": 0.70},
            ],
        },
    )

    build_quant_decision_lab(output_dir=output_dir, run_dir=None, logger=logger)

    bridge_payload = json.loads(
        (output_dir / "analysis" / "quant_decision_lab" / "archetype_decision_bridge.json").read_text(encoding="utf-8")
    )

    assert bridge_payload["status"] == "ok"
    assert bridge_payload["listener_archetype_source"]["lifecycle_available"] is False
    assert any("steady-state archetype summaries only" in item for item in bridge_payload["summary"])
    for row in bridge_payload["archetype_recommendations"]:
        assert row["recommended_policy"]["lifecycle_signals"] == []
        assert row["scenario_focus"]["lifecycle_annotation"] == ""


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
