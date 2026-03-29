from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import spotify.stress_test as stress_test


def _logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    return logger


def test_run_stress_test_lab_samples_sessions_and_logs_progress(
    tmp_path: Path,
    monkeypatch,
    caplog,
) -> None:
    data = SimpleNamespace(
        X_seq_test=np.array(
            [
                [0, 1],
                [1, 2],
                [2, 0],
                [0, 2],
                [2, 1],
                [1, 0],
            ],
            dtype="int32",
        ),
        X_ctx_test=np.array(
            [
                [8.0, 0.0, 0.0],
                [9.0, 1.0, 0.0],
                [10.0, 0.0, 1.0],
                [11.0, 2.0, 0.0],
                [12.0, 1.0, 1.0],
                [13.0, 0.0, 0.0],
            ],
            dtype="float32",
        ),
    )
    safe_policy = SimpleNamespace(global_policy={"transition": 0.8, "continuity": 0.3, "novelty": 0.2, "repeat": 0.7})
    calls: list[tuple[tuple[int, ...], tuple[float, ...]]] = []

    def _fake_rollout(
        *,
        twin,
        multimodal_space,
        causal_artifact,
        start_sequence,
        start_context,
        horizon,
        policy_weights,
        scenario,
        rng,
    ) -> dict[str, float]:
        _ = (twin, multimodal_space, causal_artifact, horizon, policy_weights, scenario, rng)
        calls.append(
            (
                tuple(int(item) for item in np.asarray(start_sequence).tolist()),
                tuple(float(item) for item in np.asarray(start_context).tolist()),
            )
        )
        return {
            "session_length": 4.0,
            "mean_skip_risk": 0.25,
            "mean_end_risk": 0.15,
        }

    monkeypatch.setattr(stress_test, "simulate_rollout", _fake_rollout)
    monkeypatch.setenv("SPOTIFY_STRESS_TEST_MAX_SESSIONS", "3")
    monkeypatch.setenv("SPOTIFY_STRESS_TEST_PROGRESS_EVERY", "2")

    with caplog.at_level(logging.INFO):
        paths = stress_test.run_stress_test_lab(
            data=data,
            digital_twin=object(),
            multimodal_space=object(),
            safe_policy=safe_policy,
            causal_artifact=None,
            output_dir=tmp_path,
            logger=_logger("spotify.test.stress_test"),
            random_seed=7,
        )

    assert len(calls) == len(stress_test.SCENARIOS) * 2 * 3
    assert tmp_path / "stress_test_summary.csv" in paths
    assert tmp_path / "stress_test_summary.json" in paths

    payload = json.loads((tmp_path / "stress_test_summary.json").read_text(encoding="utf-8"))
    assert len(payload) == len(stress_test.SCENARIOS) * 2
    assert all(row["evaluated_sessions"] == 3 for row in payload)
    assert all(row["total_test_sessions"] == 6 for row in payload)
    assert all(row["sample_fraction"] == 0.5 for row in payload)
    assert all(row["elapsed_seconds"] >= 0.0 for row in payload)

    assert "Stress-test lab evaluating 3/6 held-out sessions" in caplog.text
    assert "Stress-test progress scenario=baseline policy=baseline_exploit processed=2/3" in caplog.text
    assert "Ran stress-test lab across 5 scenarios using 3/6 sessions." in caplog.text
