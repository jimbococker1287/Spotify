from __future__ import annotations

import csv

from spotify.governance import evaluate_champion_gate


def _write_history(path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=["run_id", "model_name", "val_top1"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_backtest_history(path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=["run_id", "profile", "model_name", "top1"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_champion_gate_passes_within_threshold(tmp_path) -> None:
    history_csv = tmp_path / "history.csv"
    _write_history(
        history_csv,
        [
            {"run_id": "run_a", "model_name": "dense", "val_top1": 0.25},
            {"run_id": "run_b", "model_name": "gru", "val_top1": 0.30},
        ],
    )
    result = evaluate_champion_gate(
        history_csv=history_csv,
        current_run_id="run_c",
        current_results=[{"model_name": "new_model", "val_top1": 0.297}],
        regression_threshold=0.005,
        metric_source="val_top1",
    )

    assert result["status"] == "pass"
    assert result["promoted"] is True


def test_champion_gate_fails_when_regression_exceeds_threshold(tmp_path) -> None:
    history_csv = tmp_path / "history.csv"
    _write_history(
        history_csv,
        [
            {"run_id": "run_a", "model_name": "dense", "val_top1": 0.25},
            {"run_id": "run_b", "model_name": "gru", "val_top1": 0.30},
        ],
    )
    result = evaluate_champion_gate(
        history_csv=history_csv,
        current_run_id="run_c",
        current_results=[{"model_name": "new_model", "val_top1": 0.28}],
        regression_threshold=0.005,
        metric_source="val_top1",
    )

    assert result["status"] == "fail"
    assert result["promoted"] is False


def test_champion_gate_uses_backtest_mean_top1_by_default(tmp_path) -> None:
    history_csv = tmp_path / "history.csv"
    _write_history(history_csv, [])
    backtest_history_csv = tmp_path / "backtest_history.csv"
    _write_backtest_history(
        backtest_history_csv,
        [
            {"run_id": "run_a", "profile": "full", "model_name": "logreg", "top1": 0.21},
            {"run_id": "run_a", "profile": "full", "model_name": "logreg", "top1": 0.19},
            {"run_id": "run_b", "profile": "full", "model_name": "mlp", "top1": 0.29},
            {"run_id": "run_b", "profile": "full", "model_name": "mlp", "top1": 0.31},
        ],
    )

    result = evaluate_champion_gate(
        history_csv=history_csv,
        backtest_history_csv=backtest_history_csv,
        current_run_id="run_c",
        current_results=[{"model_name": "new_model", "val_top1": 0.01}],
        current_backtest_rows=[
            {"model_name": "new_model", "top1": 0.287},
            {"model_name": "new_model", "top1": 0.283},
        ],
        regression_threshold=0.02,
        current_profile="full",
    )

    assert result["metric_source"] == "backtest_top1"
    assert result["status"] == "pass"
    assert result["promoted"] is True


def test_champion_gate_profile_matching_prevents_cross_profile_blocking(tmp_path) -> None:
    history_csv = tmp_path / "history.csv"
    _write_history(history_csv, [])
    backtest_history_csv = tmp_path / "backtest_history.csv"
    _write_backtest_history(
        backtest_history_csv,
        [
            {"run_id": "run_full_1", "profile": "full", "model_name": "mlp", "top1": 0.35},
            {"run_id": "run_full_1", "profile": "full", "model_name": "mlp", "top1": 0.36},
            {"run_id": "run_fast_1", "profile": "fast", "model_name": "extra_trees", "top1": 0.17},
            {"run_id": "run_fast_1", "profile": "fast", "model_name": "extra_trees", "top1": 0.18},
        ],
    )

    result = evaluate_champion_gate(
        history_csv=history_csv,
        backtest_history_csv=backtest_history_csv,
        current_run_id="run_fast_2",
        current_results=[{"model_name": "candidate", "val_top1": 0.1}],
        current_backtest_rows=[
            {"model_name": "candidate", "top1": 0.19},
            {"model_name": "candidate", "top1": 0.20},
        ],
        regression_threshold=0.01,
        current_profile="fast",
        require_profile_match=True,
    )

    assert result["promoted"] is True
    assert result["champion_run_id"] == "run_fast_1"
    assert result["metric_source"] == "backtest_top1"
