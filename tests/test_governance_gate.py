from __future__ import annotations

import csv
import json

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


def _write_champion_alias(outputs_dir, *, run_id: str, model_name: str, profile: str = "full") -> None:
    alias_dir = outputs_dir / "models" / "champion"
    alias_dir.mkdir(parents=True, exist_ok=True)
    run_dir = outputs_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_manifest.json").write_text(
        json.dumps({"run_id": run_id, "profile": profile}, indent=2),
        encoding="utf-8",
    )
    (alias_dir / "alias.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "run_dir": str(run_dir),
                "model_name": model_name,
                "model_type": "retrieval_reranker",
                "promoted_at": "2026-04-09T00:39:54+00:00",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


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


def test_champion_gate_prefers_current_alias_baseline_when_available(tmp_path) -> None:
    outputs_dir = tmp_path / "outputs"
    history_dir = outputs_dir / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    history_csv = history_dir / "history.csv"
    _write_history(history_csv, [])
    backtest_history_csv = history_dir / "backtest_history.csv"
    _write_backtest_history(
        backtest_history_csv,
        [
            {"run_id": "run_old_best", "profile": "full", "model_name": "mlp", "top1": 0.31},
            {"run_id": "run_old_best", "profile": "full", "model_name": "mlp", "top1": 0.29},
            {"run_id": "run_alias", "profile": "full", "model_name": "retrieval_reranker", "top1": 0.45},
            {"run_id": "run_alias", "profile": "full", "model_name": "retrieval_reranker", "top1": 0.43},
        ],
    )
    _write_champion_alias(outputs_dir, run_id="run_alias", model_name="retrieval_reranker", profile="full")

    result = evaluate_champion_gate(
        history_csv=history_csv,
        backtest_history_csv=backtest_history_csv,
        current_run_id="run_current",
        current_results=[{"model_name": "retrieval_reranker", "val_top1": 0.01}],
        current_backtest_rows=[
            {"model_name": "retrieval_reranker", "top1": 0.44},
            {"model_name": "retrieval_reranker", "top1": 0.42},
        ],
        regression_threshold=0.05,
        current_profile="full",
    )

    assert result["champion_run_id"] == "run_alias"
    assert result["champion_model_name"] == "retrieval_reranker"
    assert result["metric_source"] == "backtest_top1"
