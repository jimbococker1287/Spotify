from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import spotify.safe_policy as safe_policy


def _logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    return logger


def _policy_name_for(weights: dict[str, float]) -> str:
    for policy_name, template in safe_policy.POLICY_TEMPLATES.items():
        if template == weights:
            return policy_name
    if weights.get("continuity", 0.0) >= 0.6 and weights.get("novelty", 1.0) <= 0.05:
        return "evening_safe"
    if weights.get("novelty", 1.0) <= 0.05 and weights.get("repeat", 0.0) >= 0.8:
        return "comfort_policy"
    raise AssertionError(f"Unknown policy weights: {weights}")


def test_learn_safe_bandit_policy_selects_global_policy_that_beats_benchmark_reference(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data = SimpleNamespace(
        X_seq_val=np.array([[0, 1], [1, 2], [2, 0]], dtype="int32"),
        X_ctx_val=np.array([[6.0, 0.0, 0.0], [7.0, 1.0, 0.0], [8.0, 2.0, 1.0]], dtype="float32"),
        y_val=np.zeros(3, dtype="int32"),
        context_features=["hour", "offline", "tech_playback_errors_24h"],
        num_ctx=3,
    )
    multimodal_space = SimpleNamespace(popularity=np.array([0.2, 0.4, 0.6], dtype="float32"))

    def _fake_rollout_batch_summary(
        *,
        twin,
        multimodal_space,
        causal_artifact,
        start_sequences,
        start_contexts,
        horizon,
        policy_weights,
        scenario,
        rng,
    ) -> dict[str, np.ndarray]:
        _ = (twin, multimodal_space, causal_artifact, start_contexts, scenario, rng)
        policy_name = _policy_name_for(policy_weights)
        batch_size = len(start_sequences)
        if horizon == 1:
            metrics = {
                "exploit_preference": (0.10, 0.10),
                "novelty_boosted": (0.12, 0.11),
                "comfort_policy": (0.14, 0.12),
                "safe_balance": (0.13, 0.11),
                "evening_safe": (0.15, 0.11),
            }
            skip_risk, end_risk = metrics[policy_name]
            first_choice = np.zeros(batch_size, dtype="int32")
            session_length = np.ones(batch_size, dtype="int32")
        else:
            metrics = {
                "exploit_preference": (0.62, 0.14),
                "novelty_boosted": (0.61, 0.13),
                "comfort_policy": (0.57, 0.11),
                "safe_balance": (0.59, 0.12),
                "evening_safe": (0.54, 0.10),
            }
            skip_risk, end_risk = metrics[policy_name]
            first_choice = np.zeros(batch_size, dtype="int32")
            session_length = np.full(batch_size, 6, dtype="int32")
        return {
            "first_choice": first_choice,
            "mean_skip_risk": np.full(batch_size, skip_risk, dtype="float32"),
            "mean_end_risk": np.full(batch_size, end_risk, dtype="float32"),
            "session_length": session_length,
        }

    monkeypatch.setattr(safe_policy, "simulate_rollout_batch_summary", _fake_rollout_batch_summary)
    monkeypatch.setenv("SPOTIFY_STRESS_BENCHMARK_SCENARIO", "evening_drift")

    artifact, paths = safe_policy.learn_safe_bandit_policy(
        data=data,
        digital_twin=object(),
        multimodal_space=multimodal_space,
        causal_artifact=None,
        output_dir=tmp_path,
        logger=_logger("spotify.test.safe_policy"),
        random_seed=7,
    )

    summary = json.loads((tmp_path / "safe_bandit_policy_summary.json").read_text(encoding="utf-8"))
    benchmark_rows = (tmp_path / "safe_bandit_policy_benchmark.csv").read_text(encoding="utf-8")
    scenario_rows = (tmp_path / "safe_bandit_policy_scenarios.csv").read_text(encoding="utf-8")

    assert artifact.global_policy_name == "comfort_policy"
    assert artifact.global_policy == safe_policy.POLICY_TEMPLATES["comfort_policy"]
    assert artifact.scenario_policy_names["evening_drift"].startswith("safe_routed_evening")
    assert artifact.scenario_policy_map["evening_drift"]["novelty"] <= 0.0
    assert artifact.scenario_policy_map["evening_drift"]["repeat"] >= 0.8
    assert artifact.benchmark_scenario == "evening_drift"
    assert artifact.benchmark_reference_policy_name == "exploit_preference"
    assert summary["global_policy_name"] == "comfort_policy"
    assert str(summary["scenario_policy_names"]["evening_drift"]).startswith("safe_routed_evening")
    assert summary["benchmark_scenario"] == "evening_drift"
    assert summary["benchmark_reference_policy_name"] == "exploit_preference"
    assert summary["global_selection_strategy"] == "beats_reference_on_stress_benchmark"
    assert summary["benchmark_beats_reference"] is True
    assert abs(float(summary["selected_skip_risk"]) - 0.57) < 1e-6
    assert abs(float(summary["reference_skip_risk"]) - 0.62) < 1e-6
    assert "comfort_policy" in benchmark_rows
    assert "safe_routed_evening" in scenario_rows
    assert (tmp_path / "safe_bandit_policy_benchmark.csv") in paths
    assert (tmp_path / "safe_bandit_policy_scenarios.csv") in paths


def test_learn_safe_bandit_policy_explores_evening_variants_from_global_safe_policy(
    tmp_path: Path,
    monkeypatch,
) -> None:
    data = SimpleNamespace(
        X_seq_val=np.array([[0, 1], [1, 2], [2, 0]], dtype="int32"),
        X_ctx_val=np.array([[6.0, 0.0, 0.0], [7.0, 1.0, 0.0], [8.0, 2.0, 1.0]], dtype="float32"),
        y_val=np.zeros(3, dtype="int32"),
        context_features=["hour", "offline", "tech_playback_errors_24h"],
        num_ctx=3,
    )
    multimodal_space = SimpleNamespace(popularity=np.array([0.2, 0.4, 0.6], dtype="float32"))

    def _fake_rollout_batch_summary(
        *,
        twin,
        multimodal_space,
        causal_artifact,
        start_sequences,
        start_contexts,
        horizon,
        policy_weights,
        scenario,
        rng,
    ) -> dict[str, np.ndarray]:
        _ = (twin, multimodal_space, causal_artifact, start_contexts, scenario, rng)
        batch_size = len(start_sequences)
        if horizon == 1:
            policy_name = _policy_name_for(policy_weights)
            metrics = {
                "exploit_preference": (0.10, 0.10),
                "novelty_boosted": (0.11, 0.10),
                "comfort_policy": (0.14, 0.11),
                "safe_balance": (0.12, 0.11),
                "evening_safe": (0.13, 0.11),
            }
            skip_risk, end_risk = metrics[policy_name]
            return {
                "first_choice": np.zeros(batch_size, dtype="int32"),
                "mean_skip_risk": np.full(batch_size, skip_risk, dtype="float32"),
                "mean_end_risk": np.full(batch_size, end_risk, dtype="float32"),
                "session_length": np.ones(batch_size, dtype="int32"),
            }

        try:
            policy_name = _policy_name_for(policy_weights)
        except AssertionError:
            policy_name = ""
        if (
            scenario == {"hour_shift": 8.0}
            and policy_weights.get("continuity", 0.0) >= 0.8
            and policy_weights.get("repeat", 0.0) >= 1.3
            and policy_weights.get("transition", 1.0) <= 0.7
            and policy_weights.get("novelty", 1.0) <= 0.05
        ):
            skip_risk, end_risk = 0.49, 0.10
        elif scenario == {"hour_shift": 8.0} and policy_name == "comfort_policy":
            skip_risk, end_risk = 0.53, 0.10
        elif scenario == {"hour_shift": 8.0} and policy_name == "safe_balance":
            skip_risk, end_risk = 0.56, 0.11
        elif scenario == {"hour_shift": 8.0} and policy_name == "exploit_preference":
            skip_risk, end_risk = 0.59, 0.12
        elif scenario == {"hour_shift": 8.0} and policy_name == "novelty_boosted":
            skip_risk, end_risk = 0.61, 0.12
        elif scenario == {"hour_shift": 8.0} and policy_weights.get("continuity", 0.0) >= 0.6:
            skip_risk, end_risk = 0.55, 0.11
        else:
            skip_risk, end_risk = 0.60, 0.12
        return {
            "first_choice": np.zeros(batch_size, dtype="int32"),
            "mean_skip_risk": np.full(batch_size, skip_risk, dtype="float32"),
            "mean_end_risk": np.full(batch_size, end_risk, dtype="float32"),
            "session_length": np.full(batch_size, 6, dtype="int32"),
        }

    monkeypatch.setattr(safe_policy, "simulate_rollout_batch_summary", _fake_rollout_batch_summary)

    artifact, _paths = safe_policy.learn_safe_bandit_policy(
        data=data,
        digital_twin=object(),
        multimodal_space=multimodal_space,
        causal_artifact=None,
        output_dir=tmp_path,
        logger=_logger("spotify.test.safe_policy.global_evening"),
        random_seed=11,
    )

    selected_name = artifact.scenario_policy_names["evening_drift"]
    selected_weights = artifact.scenario_policy_map["evening_drift"]
    scenario_rows = (tmp_path / "safe_bandit_policy_scenarios.csv").read_text(encoding="utf-8")

    assert artifact.global_policy_name == "comfort_policy"
    assert selected_name.startswith("safe_routed_evening__global_variant_")
    assert selected_weights["continuity"] >= 0.8
    assert selected_weights["repeat"] >= 1.3
    assert selected_weights["transition"] <= 0.7
    assert selected_weights["novelty"] <= 0.05
    assert "global_variant:evening_transition_min" in scenario_rows
