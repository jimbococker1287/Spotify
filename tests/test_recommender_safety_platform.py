from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from spotify.recommender_safety import (
    SequenceSplitSnapshot,
    build_conformal_abstention_summary,
    compute_context_feature_drift_rows,
    compute_segment_share_shift_rows,
    compute_target_distribution_drift,
    evaluate_promotion_gate,
    run_temporal_backtest_benchmark,
)


def _logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    return logger


def _write_history(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_run_temporal_backtest_benchmark_supports_generic_sequence_metrics(tmp_path: Path) -> None:
    rows = run_temporal_backtest_benchmark(
        n_rows=12,
        folds=2,
        metric_name="sequence_reward",
        output_dir=tmp_path,
        logger=_logger("spotify.test.recommender_safety.backtest"),
        min_train_rows=4,
        evaluators={
            "retrieval": lambda window: {"sequence_reward": 0.40 + (0.05 * window.fold), "latency_ms": 20 + window.fold},
            "reranker": lambda window: {"sequence_reward": 0.55 + (0.03 * window.fold), "latency_ms": 35 + window.fold},
        },
    )

    assert len(rows) == 4
    assert (tmp_path / "temporal_backtest.csv").exists()
    assert (tmp_path / "temporal_backtest.json").exists()
    assert (tmp_path / "temporal_backtest_summary.csv").exists()
    assert (tmp_path / "temporal_backtest_summary.json").exists()
    assert (tmp_path / "temporal_backtest_sequence_reward.png").exists()

    summary_payload = (tmp_path / "temporal_backtest_summary.json").read_text(encoding="utf-8")
    assert "reranker" in summary_payload
    assert "sequence_reward" in summary_payload


def test_evaluate_promotion_gate_supports_custom_metric_name_and_risk_caps(tmp_path: Path) -> None:
    history_csv = tmp_path / "watch_time_history.csv"
    _write_history(
        history_csv,
        [
            {"run_id": "run_a", "profile": "prod", "model_name": "baseline", "expected_watch_time": 42.0},
            {"run_id": "run_b", "profile": "prod", "model_name": "champion", "expected_watch_time": 48.0},
        ],
        fieldnames=["run_id", "profile", "model_name", "expected_watch_time"],
    )

    result = evaluate_promotion_gate(
        history_csv=history_csv,
        current_run_id="run_c",
        current_rows=[{"model_name": "candidate", "expected_watch_time": 49.5}],
        metric_name="expected_watch_time",
        regression_threshold=0.5,
        current_profile="prod",
        current_risk_metrics={
            "candidate": {
                "val_selective_risk": 0.18,
                "val_abstention_rate": 0.04,
            }
        },
        max_selective_risk=0.10,
    )

    assert result["metric_name"] == "expected_watch_time"
    assert result["champion_model_name"] == "champion"
    assert result["challenger_model_name"] == "candidate"
    assert result["promoted"] is False
    assert result["status"] == "fail_selective_risk"


def test_generic_drift_helpers_support_custom_segments_and_targets() -> None:
    reference = SequenceSplitSnapshot(
        name="baseline",
        context=np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]], dtype="float32"),
        targets=np.array([0, 1, 0], dtype="int32"),
        frame=pd.DataFrame({"device": ["mobile", "mobile", "desktop"], "phase": ["early", "mid", "late"]}),
    )
    canary = SequenceSplitSnapshot(
        name="canary",
        context=np.array([[4.0, 4.0], [5.0, 6.0]], dtype="float32"),
        targets=np.array([1, 1], dtype="int32"),
        frame=pd.DataFrame({"device": ["desktop", "desktop"], "phase": ["late", "late"]}),
    )

    context_rows = compute_context_feature_drift_rows(
        feature_names=["recency_score", "session_depth"],
        reference_split=reference,
        comparison_splits=[canary],
    )
    segment_rows = compute_segment_share_shift_rows(
        reference_split=reference,
        comparison_splits=[canary],
        segment_extractors={
            "device_type": lambda frame: frame["device"].to_numpy(),
            "session_phase": lambda frame: frame["phase"].to_numpy(),
        },
    )
    target_drift = compute_target_distribution_drift(reference_split=reference, comparison_splits=[canary])

    assert context_rows
    assert context_rows[0]["compare_split"] == "canary"
    assert segment_rows
    assert any(row["segment"] == "device_type" for row in segment_rows)
    assert "baseline_vs_canary_jsd" in target_drift


def test_build_conformal_abstention_summary_returns_generic_payload() -> None:
    payload = build_conformal_abstention_summary(
        tag="video_home_feed",
        val_proba=np.array(
            [
                [0.82, 0.10, 0.08],
                [0.12, 0.70, 0.18],
                [0.15, 0.22, 0.63],
                [0.61, 0.25, 0.14],
            ],
            dtype="float32",
        ),
        val_y=np.array([0, 1, 2, 0], dtype="int32"),
        test_proba=np.array(
            [
                [0.55, 0.25, 0.20],
                [0.34, 0.33, 0.33],
            ],
            dtype="float32",
        ),
        test_y=np.array([0, 1], dtype="int32"),
        alpha=0.10,
    )

    assert payload is not None
    assert payload["tag"] == "video_home_feed"
    assert float(payload["calibration"]["threshold"]) > 0.0
    assert "abstention_rate" in payload["val"]
    assert "abstention_rate" in payload["test"]
