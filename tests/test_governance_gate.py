from __future__ import annotations

import csv

from spotify.governance import evaluate_champion_gate


def _write_history(path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=["run_id", "model_name", "val_top1"])
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
    )

    assert result["status"] == "fail"
    assert result["promoted"] is False
